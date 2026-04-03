"""Kontextualisiert die ROI-Token-Sequenz.
Input
token embeddings (B, T, D_tok)
mask (B, T)
from torch import nn


Output
context states (B, T, D_ctx)
mask (B, T)
Empfehlung

Für den ersten sauberen C-Stand:

BiGRU behalten
from torch import nn


später optional Transformer testen
"""

# model/kuronet/context/roi_context.py

from __future__ import annotations

from typing import Literal, Tuple

import torch
import torch.nn as nn


class ROIContextEncoder(nn.Module):
    """
    Contextualize ordered ROI token embeddings before decoding.

    Expected role in Option C:
        refined ROI features
        -> token projection
        -> ROIContextEncoder
        -> decoder

    Supported modes:
        - "bigru": stable default for moderate datasets
        - "gru":   lighter unidirectional variant

    Input:
        seq:  (B, T, D_in)
        mask: (B, T) bool, True = valid token

    Output:
        out:  (B, T, D_out)
        mask: (B, T)
    """

    def __init__(
        self,
        in_dim: int = 256,
        hidden_dim: int = 256,
        out_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.1,
        mode: Literal["bigru", "gru"] = "bigru",
        use_layernorm: bool = True,
        use_residual: bool = True,
    ):
        super().__init__()

        if mode not in {"bigru", "gru"}:
            raise ValueError(f"Unsupported mode='{mode}'. Use 'bigru' or 'gru'.")

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.mode = mode
        self.use_layernorm = bool(use_layernorm)
        self.use_residual = bool(use_residual)

        bidirectional = (mode == "bigru")
        rnn_out_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.input_proj = None

        self.rnn = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.out_proj = nn.Sequential(
            nn.Linear(rnn_out_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.layernorm = nn.LayerNorm(out_dim) if use_layernorm else None

        self.residual_proj = None
        if use_residual and in_dim != out_dim:
            self.residual_proj = nn.Linear(in_dim, out_dim)

    def _safe_lengths_from_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Convert boolean mask to CPU lengths for pack_padded_sequence.
        Ensures all lengths are at least 1 to avoid runtime errors.
        """
        lengths = mask.sum(dim=1)
        lengths = lengths.clamp(min=1)
        return lengths.detach().cpu()

    def forward(
        self,
        seq: torch.Tensor,   # (B, T, D_in)
        mask: torch.Tensor,  # (B, T) bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq.dim() != 3:
            raise ValueError(f"seq must have shape (B, T, D), got {tuple(seq.shape)}")
        if mask.dim() != 2:
            raise ValueError(f"mask must have shape (B, T), got {tuple(mask.shape)}")
        if seq.size(0) != mask.size(0) or seq.size(1) != mask.size(1):
            raise ValueError(
                f"seq shape {tuple(seq.shape)} and mask shape {tuple(mask.shape)} are incompatible"
            )

        if self.input_proj is not None:
            seq_in = self.input_proj(seq)
        else:
            seq_in = seq

        lengths = self._safe_lengths_from_mask(mask)

        packed = nn.utils.rnn.pack_padded_sequence(
            seq_in,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )

        packed_out, _ = self.rnn(packed)

        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=seq.size(1),
        )  # (B, T, rnn_out_dim)

        out = self.out_proj(out)  # (B, T, out_dim)

        if self.use_residual:
            if self.residual_proj is not None:
                residual = self.residual_proj(seq)
            else:
                residual = seq
            out = out + residual

        if self.layernorm is not None:
            out = self.layernorm(out)

        # Zero out padded positions explicitly
        out = out * mask.unsqueeze(-1).to(out.dtype)

        return out, mask