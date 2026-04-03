import random
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LuongAttention(nn.Module):
    """
    Standard Luong-style dot attention.

    Args:
        enc_dim: Dimension of encoder outputs
        dec_dim: Dimension of decoder hidden state
    """

    def __init__(self, enc_dim: int, dec_dim: int):
        super().__init__()
        self.project_dec = nn.Linear(dec_dim, enc_dim, bias=False)

    def forward(
        self,
        dec_hidden: torch.Tensor,   # (B, dec_dim)
        enc_outputs: torch.Tensor,  # (B, T_enc, enc_dim)
        mask: Optional[torch.Tensor] = None,  # (B, T_enc), True = valid
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            context: (B, enc_dim)
            attn:    (B, T_enc)
        """
        dec_proj = self.project_dec(dec_hidden).unsqueeze(2)   # (B, enc_dim, 1)
        scores = torch.bmm(enc_outputs, dec_proj).squeeze(2)   # (B, T_enc)

        if mask is not None:
            neg_fill = -1e4 if scores.dtype == torch.float16 else -1e9
            scores = scores.masked_fill(~mask.bool(), neg_fill)

        attn = F.softmax(scores, dim=1)                        # (B, T_enc)
        context = torch.bmm(attn.unsqueeze(1), enc_outputs).squeeze(1)  # (B, enc_dim)
        return context, attn


class SeqDecoderAttention(nn.Module):
    """
    Autoregressive attention decoder for ROI/context sequence features.

    Expected use in Option C:
        refined ROI features
        -> context encoder
        -> this decoder

    This decoder is intentionally limited to token decoding only.
    It does NOT predict boxes or refinement signals.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        vocab_size: int,
        enc_dim: int,
        num_layers: int = 1,
        init_from_encoder: bool = True,
        sampling_method: str = "argmax",
        dropout: float = 0.1,
    ):
        super().__init__()

        if sampling_method not in {"argmax", "multinomial", "gumbel"}:
            raise ValueError(
                f"Unsupported sampling_method='{sampling_method}'. "
                "Use one of: {'argmax', 'multinomial', 'gumbel'}."
            )

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.attn = LuongAttention(enc_dim=enc_dim, dec_dim=hidden_dim)

        self.rnn = nn.GRU(
            input_size=embed_dim + enc_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.out = nn.Linear(hidden_dim, vocab_size)

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.init_from_encoder = init_from_encoder
        self.sampling_method = sampling_method

        if init_from_encoder:
            self.enc2hidden = nn.Sequential(
                nn.Linear(enc_dim, hidden_dim),
                nn.Tanh(),
            )

    def _sample_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Sample next token IDs from logits.

        Args:
            logits: (B, V)

        Returns:
            next_token_ids: (B,)
        """
        if self.sampling_method == "argmax":
            return logits.argmax(dim=-1)

        probs = F.softmax(logits, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

        probs_sum = probs.sum(dim=-1, keepdim=True)
        fallback = probs.new_full(probs.shape, 1.0 / probs.size(-1))
        probs = torch.where(probs_sum > 0, probs / probs_sum.clamp_min(1e-12), fallback)

        if self.sampling_method == "multinomial":
            return torch.multinomial(probs, num_samples=1).squeeze(1)

        if self.sampling_method == "gumbel":
            g = -torch.log(-torch.log(torch.rand_like(probs).clamp_min(1e-12)))
            return (torch.log(probs.clamp_min(1e-12)) + g).argmax(dim=-1)

        # Should never happen because of validation in __init__
        return logits.argmax(dim=-1)

    def _init_hidden(
        self,
        enc_outputs: torch.Tensor,              # (B, T_enc, enc_dim)
        enc_mask: Optional[torch.Tensor] = None # (B, T_enc)
    ) -> torch.Tensor:
        """
        Initialize decoder hidden state from encoder outputs.

        Returns:
            hidden: (num_layers, B, hidden_dim)
        """
        bsz = enc_outputs.size(0)
        device = enc_outputs.device

        if not self.init_from_encoder:
            return torch.zeros(self.num_layers, bsz, self.hidden_dim, device=device)

        if enc_mask is not None:
            lengths = enc_mask.sum(dim=1).clamp(min=1).unsqueeze(1).to(enc_outputs.dtype)
            pooled = enc_outputs.sum(dim=1) / lengths
        else:
            pooled = enc_outputs.mean(dim=1)

        h0 = self.enc2hidden(pooled)  # (B, hidden_dim)
        hidden = h0.unsqueeze(0).repeat(self.num_layers, 1, 1)
        return hidden

    def forward(
        self,
        enc_outputs: torch.Tensor,                   # (B, T_enc, enc_dim)
        enc_mask: Optional[torch.Tensor] = None,     # (B, T_enc)
        input_seq: Optional[torch.Tensor] = None,    # (B, T_in), optional
        targets: Optional[torch.Tensor] = None,      # (B, T_tgt), optional
        hidden: Optional[torch.Tensor] = None,       # (num_layers, B, hidden_dim)
        teacher_forcing_ratio: float = 0.0,
        sos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        max_len: Optional[int] = None,
        token_logit_bias: Optional[torch.Tensor] = None,  # optional, (B, T_dec, V)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decoding modes:
        1. Training with targets:
            - targets determines T_dec
            - teacher forcing can be used
        2. Decoding with input_seq:
            - first token taken from input_seq[:, 0]
        3. Free decoding:
            - requires sos_id and max_len

        Returns:
            logits_cat: (B, T_dec, vocab_size)
            hidden:     (num_layers, B, hidden_dim)
            attn_cat:   (B, T_dec, T_enc)
        """
        if enc_outputs.dim() != 3:
            raise ValueError(f"enc_outputs must have shape (B, T_enc, C), got {tuple(enc_outputs.shape)}")

        device = enc_outputs.device
        bsz = enc_outputs.size(0)

        if targets is not None:
            t_dec = targets.size(1)
        elif input_seq is not None:
            t_dec = input_seq.size(1)
        elif max_len is not None:
            t_dec = max_len
        else:
            raise ValueError("Provide targets, input_seq, or max_len to determine decoding length.")

        if hidden is None:
            hidden = self._init_hidden(enc_outputs=enc_outputs, enc_mask=enc_mask)

        if input_seq is not None:
            input_tok = input_seq[:, 0]
        else:
            if sos_id is None:
                raise ValueError("sos_id must be provided when input_seq is None.")
            input_tok = torch.full((bsz,), int(sos_id), dtype=torch.long, device=device)

        outputs = []
        attn_weights = []

        finished = torch.zeros(bsz, dtype=torch.bool, device=device)
        eos_tensor = None
        if eos_id is not None:
            eos_tensor = torch.full((bsz,), int(eos_id), dtype=torch.long, device=device)

        for t in range(t_dec):
            emb = self.embed(input_tok)                    # (B, embed_dim)
            dec_hidden_top = hidden[-1]                    # (B, hidden_dim)

            context, attn = self.attn(
                dec_hidden=dec_hidden_top,
                enc_outputs=enc_outputs,
                mask=enc_mask,
            )

            rnn_in = torch.cat([emb, context], dim=1).unsqueeze(1)  # (B, 1, embed+enc)
            out_rnn, hidden = self.rnn(rnn_in, hidden)              # out_rnn: (B, 1, H)

            logits = self.out(out_rnn.squeeze(1))                   # (B, V)

            if token_logit_bias is not None:
                if token_logit_bias.dim() != 3:
                    raise ValueError(
                        f"token_logit_bias must have shape (B, T_dec, V), got {tuple(token_logit_bias.shape)}"
                    )
                if t < token_logit_bias.size(1):
                    logits = logits + token_logit_bias[:, t, :]

            outputs.append(logits.unsqueeze(1))
            attn_weights.append(attn.unsqueeze(1))

            use_teacher = (input_seq is not None) and ((t + 1) < input_seq.size(1)) and (random.random() < teacher_forcing_ratio)

            if use_teacher:
                next_tok = input_seq[:, t + 1]
            else:
                next_tok = self._sample_from_logits(logits)

            if eos_tensor is not None:
                input_tok = torch.where(finished, eos_tensor, next_tok)
            else:
                input_tok = next_tok

            # Only stop early during free decoding, not during supervised training.
            if targets is None and eos_id is not None:
                finished = finished | (input_tok == eos_id)
                if finished.all():
                    break

        logits_cat = torch.cat(outputs, dim=1)        # (B, T_dec, V)
        attn_cat = torch.cat(attn_weights, dim=1)     # (B, T_dec, T_enc)

        return logits_cat, hidden, attn_cat