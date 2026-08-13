# model/suminanet/context/roi_context.py

from __future__ import annotations

from typing import Literal, Optional, Tuple

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

        self.pos_proj = nn.Linear(2, in_dim)

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
        seq: torch.Tensor,                             # (B, T, D_in)
        mask: torch.Tensor,                            # (B, T) bool
        spatial_coords: Optional[torch.Tensor] = None, # (B, T, 2) normalized (cx, cy)
        refine_scores: Optional[torch.Tensor] = None,  # (B, T) logits from ROIRefinementHead
        block_ids: Optional[torch.Tensor] = None,      # (B, T) long — per-block GRU reset; -1=invalid
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq.dim() != 3:
            raise ValueError(f"seq must have shape (B, T, D), got {tuple(seq.shape)}")
        if mask.dim() != 2:
            raise ValueError(f"mask must have shape (B, T), got {tuple(mask.shape)}")
        if seq.size(0) != mask.size(0) or seq.size(1) != mask.size(1):
            raise ValueError(
                f"seq shape {tuple(seq.shape)} and mask shape {tuple(mask.shape)} are incompatible"
            )

        else:
            seq_in = seq

        if spatial_coords is not None:
            seq_in = seq_in + self.pos_proj(spatial_coords)

        # Soft-gate token embeddings by detection confidence before the GRU.
        # FP proposals (low refine_score) get near-zero weight so they don't corrupt the
        # recurrent state.  Scores are detached so BiGRU gradients don't flow back into
        # the refinement head — that head is already supervised by its own BCE loss and
        # mixing in BiGRU signal created conflicting objectives (IoU quality vs. contextual
        # usefulness).
        if refine_scores is not None:
            gate = torch.sigmoid(refine_scores.detach()).unsqueeze(-1)  # (B, T, 1)
            seq_in = seq_in * gate

        if block_ids is not None:
            # Per-block GRU: reset hidden state at text-block boundaries so that
            # characters from unrelated blocks (main text, marginal annotations,
            # header cartouches) do not pollute each other's context vectors.
            out = torch.zeros(
                seq_in.size(0), seq_in.size(1),
                self.hidden_dim * (2 if self.mode == "bigru" else 1),
                device=seq_in.device, dtype=seq_in.dtype,
            )
            B = seq_in.size(0)
            for b in range(B):
                valid_b = mask[b]        # (T,) bool
                bids_b  = block_ids[b]   # (T,) long; -1 = padded
                max_bid = int(bids_b[valid_b].max().item()) if valid_b.any() else -1
                for bid in range(max_bid + 1):
                    pos = (bids_b == bid)   # (T,) bool — valid positions in this block
                    n_pos = int(pos.sum().item())
                    if n_pos == 0:
                        continue
                    block_seq = seq_in[b, pos].unsqueeze(0)   # (1, n_pos, D)
                    block_len = torch.tensor([n_pos], dtype=torch.long)
                    packed_b  = nn.utils.rnn.pack_padded_sequence(
                        block_seq, block_len, batch_first=True, enforce_sorted=False
                    )
                    packed_out_b, _ = self.rnn(packed_b)     # fresh h_0=None per block
                    block_out, _ = nn.utils.rnn.pad_packed_sequence(
                        packed_out_b, batch_first=True, total_length=n_pos
                    )                                         # (1, n_pos, rnn_out_dim)
                    out[b, pos] = block_out[0]
        else:
            lengths = self._safe_lengths_from_mask(mask)
            packed = nn.utils.rnn.pack_padded_sequence(
                seq_in, lengths, batch_first=True, enforce_sorted=False,
            )
            packed_out, _ = self.rnn(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=seq.size(1),
            )   # (B, T, rnn_out_dim)

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