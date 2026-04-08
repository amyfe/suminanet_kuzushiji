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
        #scale for attention weighted encoder bias
        self.bias_scale = nn.Parameter(torch.tensor(0.4))
        #stop head gets decoder state + context + 2 scalar features (e.g. max confidence, entropy)
        self.stop_head = nn.Sequential(
            nn.Linear(hidden_dim + enc_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
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
    
    def _topk_mask_logits(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        """
        Keep only top-k entries per row, heavily suppress the rest.
        """
        if k <= 0 or k >= logits.size(-1):
            return logits

        neg_fill = -1e4 if logits.dtype == torch.float16 else -1e9
        topk_vals, topk_idx = logits.topk(k=k, dim=-1)
        masked = torch.full_like(logits, neg_fill)
        masked.scatter_(1, topk_idx, topk_vals)
        return masked
    
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

    def _apply_decode_constraints(
        self,
        logits: torch.Tensor,
        prev_tokens: torch.Tensor,
        token_counts: Optional[torch.Tensor],
        step_idx: int,
        eos_id: Optional[int],
        stop_logit: Optional[torch.Tensor],
        progress_ratio: Optional[torch.Tensor],
        coverage_mean: Optional[torch.Tensor],
        decode_constraints: Optional[dict],
    ) -> torch.Tensor:
        """
        Apply lightweight constraints for free decoding to reduce repetition collapse.
        """
        if not decode_constraints:
            return logits

        # Normalize optional gating inputs to shape (B,) to avoid accidental
        # broadcasting (e.g. (B,) & (B,1) -> (B,B)).
        if stop_logit is not None:
            stop_logit = stop_logit.reshape(-1)
        if progress_ratio is not None:
            progress_ratio = progress_ratio.reshape(-1)
        if coverage_mean is not None:
            coverage_mean = coverage_mean.reshape(-1)

        adjusted = logits
        neg_fill = -1e4 if adjusted.dtype == torch.float16 else -1e9

        min_steps = int(decode_constraints.get("min_steps", 0))
        if eos_id is not None and step_idx < min_steps:
            adjusted = adjusted.clone()
            adjusted[:, int(eos_id)] = neg_fill

        require_stop_for_eos = bool(decode_constraints.get("require_stop_for_eos", False))
        stop_threshold = float(decode_constraints.get("stop_threshold", 0.0))
        min_progress_for_eos = float(decode_constraints.get("min_progress_for_eos", 0.0))
        min_coverage_for_eos = float(decode_constraints.get("min_coverage_for_eos", 0.0))
        eos_bonus_scale = float(decode_constraints.get("eos_bonus_scale", 0.0))

        allow_eos = None
        if eos_id is not None:
            allow_eos = torch.ones(
                (logits.size(0),),
                device=logits.device,
                dtype=torch.bool,
            )

            if require_stop_for_eos and stop_logit is not None:
                allow_eos = allow_eos & (stop_logit > stop_threshold)

            if progress_ratio is not None:
                allow_eos = allow_eos & (progress_ratio > min_progress_for_eos)

            if coverage_mean is not None:
                allow_eos = allow_eos & (coverage_mean > min_coverage_for_eos)

            if adjusted is logits:
                adjusted = adjusted.clone()

            eos_vals = adjusted[:, int(eos_id)]
            blocked = torch.full_like(eos_vals, neg_fill)
            adjusted[:, int(eos_id)] = torch.where(allow_eos, eos_vals, blocked)

            if eos_bonus_scale > 0.0 and stop_logit is not None:
                eos_bonus = eos_bonus_scale * stop_logit
                adjusted[:, int(eos_id)] = torch.where(
                    allow_eos,
                    adjusted[:, int(eos_id)] + eos_bonus,
                    adjusted[:, int(eos_id)],
                )

        if bool(decode_constraints.get("block_immediate_repeat", False)):
            if adjusted is logits:
                adjusted = adjusted.clone()
            adjusted.scatter_(1, prev_tokens.unsqueeze(1), neg_fill)

        repetition_penalty = float(decode_constraints.get("repetition_penalty", 0.0))
        if repetition_penalty > 0.0 and token_counts is not None:
            if adjusted is logits:
                adjusted = adjusted.clone()
            adjusted = adjusted - repetition_penalty * token_counts

        return adjusted

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
        encoder_token_bias: Optional[torch.Tensor] = None,  # optional, (B, T_enc, V)
        decode_constraints: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
            stop_cat:   (B, T_dec)
        """
        if enc_outputs.dim() != 3:
            raise ValueError(f"enc_outputs must have shape (B, T_enc, C), got {tuple(enc_outputs.shape)}")

        device = enc_outputs.device
        bsz, t_enc, _ = enc_outputs.shape

        if encoder_token_bias is not None:
            if encoder_token_bias.dim() != 3:
                raise ValueError(
                    f"encoder_token_bias must have shape (B, T_enc, V), got {tuple(encoder_token_bias.shape)}"
                )
            if encoder_token_bias.size(0) != bsz or encoder_token_bias.size(1) != t_enc:
                raise ValueError(
                    f"encoder_token_bias shape {tuple(encoder_token_bias.shape)} must match (B, T_enc, V)=({bsz}, {t_enc}, V)"
                )
            if encoder_token_bias.size(2) != self.vocab_size:
                raise ValueError(
                    f"encoder_token_bias vocab dim must be {self.vocab_size}, got {encoder_token_bias.size(2)}"
                )

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
        if hidden is None:
            raise RuntimeError("Decoder hidden state initialization failed.")
        hidden_state = hidden

        if input_seq is not None:
            input_tok = input_seq[:, 0]
        else:
            if sos_id is None:
                raise ValueError("sos_id must be provided when input_seq is None.")
            input_tok = torch.full((bsz,), int(sos_id), dtype=torch.long, device=device)

        outputs = []
        attn_weights = []
        stop_logits_all = []

        finished = torch.zeros(bsz, dtype=torch.bool, device=device)
        eos_tensor = None
        if eos_id is not None:
            eos_tensor = torch.full((bsz,), int(eos_id), dtype=torch.long, device=device)

        token_counts = None
        if decode_constraints and targets is None:
            token_counts = torch.zeros(
                (bsz, self.vocab_size),
                device=device,
                dtype=enc_outputs.dtype,
            )
        coverage = torch.zeros((bsz, t_enc), device=device, dtype=enc_outputs.dtype)

        for t in range(t_dec):
            emb = self.embed(input_tok)                    # (B, embed_dim)
            dec_hidden_top = hidden_state[-1]              # (B, hidden_dim)

            context, attn = self.attn(
                dec_hidden=dec_hidden_top,
                enc_outputs=enc_outputs,
                mask=enc_mask,
            )

            coverage = coverage + attn
            if enc_mask is not None:
                coverage = coverage * enc_mask.to(coverage.dtype)
                valid_counts = enc_mask.sum(dim=1, keepdim=True).clamp(min=1).to(enc_outputs.dtype)
                coverage_mean = coverage.sum(dim=1, keepdim=True) / valid_counts
            else:
                coverage_mean = coverage.mean(dim=1, keepdim=True)

            progress_ratio = torch.full((bsz, ), float(t) / max(t_dec, 1), device=device, dtype=enc_outputs.dtype)
            
            rnn_in = torch.cat([emb, context], dim=1).unsqueeze(1)  # (B, 1, embed+enc)
            out_rnn, hidden_state = self.rnn(rnn_in, hidden_state)   # out_rnn: (B, 1, H)
            dec_out = out_rnn.squeeze(1)  
            
            logits = self.out(out_rnn.squeeze(1))                   # (B, V)

            step_bias = None
            if encoder_token_bias is not None:
                step_bias = torch.bmm(
                    attn.unsqueeze(1),          # (B,1,T_enc)
                    encoder_token_bias          # (B,T_enc,V)
                ).squeeze(1)                    # (B,V)

                aux_bias_topk = 0
                if decode_constraints is not None:
                    aux_bias_topk = int(decode_constraints.get("aux_bias_topk", 0))

                if aux_bias_topk > 0:
                    step_bias = self._topk_mask_logits(step_bias, aux_bias_topk)

                logits = logits + self.bias_scale * step_bias

            if token_logit_bias is not None:
                if token_logit_bias.dim() != 3:
                    raise ValueError(
                        f"token_logit_bias must have shape (B, T_dec, V), got {tuple(token_logit_bias.shape)}"
                    )
                if t < token_logit_bias.size(1):
                    logits = logits + token_logit_bias[:, t, :]
            
            stop_features = torch.cat([
                dec_out,
                context,
                coverage_mean,
                progress_ratio.unsqueeze(1),
            ], dim=1)
            stop_logit = self.stop_head(stop_features).squeeze(1)   # 
            if targets is None:
                logits = self._apply_decode_constraints(
                    logits=logits,
                    prev_tokens=input_tok,
                    token_counts=token_counts,
                    step_idx=t,
                    eos_id=eos_id,
                    stop_logit=stop_logit,
                    progress_ratio=progress_ratio,
                    coverage_mean=coverage_mean,
                    decode_constraints=decode_constraints,
                )

            outputs.append(logits.unsqueeze(1))
            attn_weights.append(attn.unsqueeze(1))
            stop_logits_all.append(stop_logit.unsqueeze(1))

            sampled_next_tok = self._sample_from_logits(logits)

            if (input_seq is not None) and ((t + 1) < input_seq.size(1)):
                teacher_next_tok = input_seq[:, t + 1]
                if teacher_forcing_ratio >= 1.0:
                    next_tok = teacher_next_tok
                elif teacher_forcing_ratio <= 0.0:
                    next_tok = sampled_next_tok
                else:
                    teacher_mask = (
                        torch.rand((bsz,), device=device) < float(teacher_forcing_ratio)
                    )
                    next_tok = torch.where(teacher_mask, teacher_next_tok, sampled_next_tok)
            else:
                next_tok = sampled_next_tok

            if eos_tensor is not None:
                input_tok = torch.where(finished, eos_tensor, next_tok)
            else:
                input_tok = next_tok

            if token_counts is not None:
                ones = torch.ones((bsz, 1), device=device, dtype=token_counts.dtype)
                token_counts.scatter_add_(1, next_tok.unsqueeze(1), ones)

            # Only stop early during free decoding, not during supervised training.
            if targets is None and eos_id is not None:
                finished = finished | (input_tok == eos_id)
                if finished.all():
                    break

        logits_cat = torch.cat(outputs, dim=1)        # (B, T_dec, V)
        attn_cat = torch.cat(attn_weights, dim=1)     # (B, T_dec, T_enc)
        stop_cat = torch.cat(stop_logits_all, dim=1).squeeze(-1)  # (B, T_dec)

        return logits_cat, hidden_state, attn_cat, stop_cat