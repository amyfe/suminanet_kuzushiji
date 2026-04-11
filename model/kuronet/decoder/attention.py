#Action-Targets aus Alignment / Pointer-Supervision
#oder monotone/coverage-basierte Targets
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
        pointer_window_radius: int=2,
    ):
        super().__init__()

        if sampling_method not in {"argmax", "multinomial", "gumbel"}:
            raise ValueError(
                f"Unsupported sampling_method='{sampling_method}'. "
                "Use one of: {'argmax', 'multinomial', 'gumbel'}."
            )
        self.pointer_window_radius = int(pointer_window_radius)
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
            nn.Linear(hidden_dim + enc_dim + enc_dim + 6, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # local pointer context + global attention context -> fused context
        self.pointer_context_proj = nn.Sequential(
            nn.Linear(enc_dim * 3, enc_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # action head: stay / advance / stop
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim + enc_dim + enc_dim + enc_dim + 6, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
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
    
    def _gather_local_pointer_window(
        self,
        enc_outputs: torch.Tensor,             # (B, T_enc, C)
        pointer_idx: torch.Tensor,             # (B,)
        enc_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Gather a local window around the current pointer and return:

        - local_window_mean: (B, C)
        - local_window_mask: (B, W) bool

        Window is centered around pointer_idx with radius self.pointer_window_radius.
        """
        bsz, t_enc, enc_dim = enc_outputs.shape
        radius = max(0, int(self.pointer_window_radius))
        window_size = 2 * radius + 1
        device = enc_outputs.device

        offsets = torch.arange(-radius, radius + 1, device=device, dtype=torch.long)  # (W,)
        raw_idx = pointer_idx.unsqueeze(1) + offsets.unsqueeze(0)  # (B, W)
        clamped_idx = raw_idx.clamp(min=0, max=max(t_enc - 1, 0))  # (B, W)

        gather_idx = clamped_idx.unsqueeze(-1).expand(-1, -1, enc_dim)  # (B, W, C)
        local_window = enc_outputs.gather(dim=1, index=gather_idx)      # (B, W, C)

        if enc_mask is not None:
            local_valid = (
                (raw_idx >= 0) & (raw_idx < t_enc)
            )
            gathered_mask = enc_mask.gather(dim=1, index=clamped_idx)
            local_window_mask = local_valid & gathered_mask.bool()       # (B, W)
        else:
            local_window_mask = (raw_idx >= 0) & (raw_idx < t_enc)

        local_window = local_window * local_window_mask.unsqueeze(-1).to(local_window.dtype)

        denom = local_window_mask.sum(dim=1, keepdim=True).clamp(min=1).to(local_window.dtype)
        local_window_mean = local_window.sum(dim=1) / denom

        return local_window_mean, local_window_mask
    
    def _gather_local_coverage_stats(
        self,
        coverage: torch.Tensor,                 # (B, T_enc)
        pointer_idx: torch.Tensor,              # (B,)
        enc_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Gather coverage statistics around the current pointer window.

        Returns:
            coverage_at_pointer:   (B, 1)
            coverage_window_mean:  (B, 1)
            coverage_window_max:   (B, 1)
        """
        bsz, t_enc = coverage.shape
        radius = max(0, int(self.pointer_window_radius))
        device = coverage.device

        ptr = pointer_idx.clamp(min=0, max=max(t_enc - 1, 0))
        coverage_at_pointer = coverage.gather(dim=1, index=ptr.unsqueeze(1))  # (B,1)

        offsets = torch.arange(-radius, radius + 1, device=device, dtype=torch.long)  # (W,)
        raw_idx = pointer_idx.unsqueeze(1) + offsets.unsqueeze(0)  # (B,W)
        clamped_idx = raw_idx.clamp(min=0, max=max(t_enc - 1, 0))  # (B,W)

        cov_window = coverage.gather(dim=1, index=clamped_idx)  # (B,W)

        if enc_mask is not None:
            local_valid = (raw_idx >= 0) & (raw_idx < t_enc)
            gathered_mask = enc_mask.gather(dim=1, index=clamped_idx).bool()
            window_mask = local_valid & gathered_mask
        else:
            window_mask = (raw_idx >= 0) & (raw_idx < t_enc)

        cov_window = cov_window * window_mask.to(cov_window.dtype)
        denom = window_mask.sum(dim=1, keepdim=True).clamp(min=1).to(cov_window.dtype)

        coverage_window_mean = cov_window.sum(dim=1, keepdim=True) / denom

        neg_fill = -1e4 if cov_window.dtype == torch.float16 else -1e9
        cov_window_for_max = cov_window.masked_fill(~window_mask, neg_fill)
        coverage_window_max = cov_window_for_max.max(dim=1, keepdim=True).values
        coverage_window_max = torch.where(
            window_mask.any(dim=1, keepdim=True),
            coverage_window_max,
            torch.zeros_like(coverage_window_mean),
        )

        return coverage_at_pointer, coverage_window_mean, coverage_window_max
    def _gather_pointer_context(
        self,
        enc_outputs: torch.Tensor,   # (B, T_enc, C)
        pointer_idx: torch.Tensor,   # (B,)
    ) -> torch.Tensor:
        """
        Gather encoder feature at current pointer position.
        """
        bsz, t_enc, enc_dim = enc_outputs.shape
        ptr = pointer_idx.clamp(min=0, max=max(t_enc - 1, 0))
        gather_idx = ptr.view(bsz, 1, 1).expand(-1, 1, enc_dim)
        return enc_outputs.gather(dim=1, index=gather_idx).squeeze(1)  # (B, C)
    
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
        Apply lightweight constraints for free decoding.

        Important:
        - EOS/stop is NOT controlled here anymore.
        - Primary stop decision comes from action_head (action=stop).
        - This function only handles generic repetition/length constraints.
        """
        if not decode_constraints:
            return logits

        adjusted = logits
        neg_fill = -1e4 if adjusted.dtype == torch.float16 else -1e9

        min_steps = int(decode_constraints.get("min_steps", 0))
        if eos_id is not None and step_idx < min_steps:
            adjusted = adjusted.clone()
            adjusted[:, int(eos_id)] = neg_fill

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

    def _apply_stop_constraints(
        self,
        *,
        stop_mask: torch.Tensor,
        step_idx: int,
        stop_logit: Optional[torch.Tensor],
        progress_ratio: Optional[torch.Tensor],
        coverage_mean: Optional[torch.Tensor],
        decode_constraints: Optional[dict],
    ) -> torch.Tensor:
        """
        Gate action-driven stop decisions with the same stabilization constraints
        used for free decoding.
        """
        if not decode_constraints:
            return stop_mask

        constrained = stop_mask

        min_steps = int(decode_constraints.get("min_steps", 0))
        if step_idx < min_steps:
            constrained = constrained & False

        stop_threshold = float(decode_constraints.get("stop_threshold", 0.0))
        if stop_threshold > 0.0 and stop_logit is not None:
            stop_prob = torch.sigmoid(stop_logit)
            constrained = constrained & stop_prob.ge(stop_threshold)

        min_progress = float(decode_constraints.get("min_progress_for_eos", 0.0))
        if min_progress > 0.0 and progress_ratio is not None:
            constrained = constrained & progress_ratio.ge(min_progress)

        min_coverage = float(decode_constraints.get("min_coverage_for_eos", 0.0))
        if min_coverage > 0.0 and coverage_mean is not None:
            constrained = constrained & coverage_mean.squeeze(1).ge(min_coverage)

        return constrained

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
        oracle_pointer_positions: Optional[torch.Tensor] = None,  # optional, (B, T_dec)
        decode_constraints: Optional[dict] = None,
    )   -> Tuple[
        torch.Tensor,  # logits_cat
        torch.Tensor,  # hidden_state
        torch.Tensor,  # attn_cat
        torch.Tensor,  # stop_cat
        torch.Tensor,  # action_cat
        torch.Tensor,  # pointer_cat
        torch.Tensor,  # emitted_cat
    ]:
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

        # Some samples can yield empty decoder targets (T_dec == 0). Return
        # empty tensors with consistent shapes instead of concatenating lists.
        if t_dec <= 0:
            logits_cat = enc_outputs.new_zeros((bsz, 0, self.vocab_size))
            attn_cat = enc_outputs.new_zeros((bsz, 0, t_enc))
            stop_cat = enc_outputs.new_zeros((bsz, 0))
            action_cat = enc_outputs.new_zeros((bsz, 0, 3))
            pointer_cat = torch.zeros((bsz, 0), dtype=torch.long, device=device)
            emitted_cat = torch.zeros((bsz, 0), dtype=torch.long, device=device)
            return logits_cat, hidden_state, attn_cat, stop_cat, action_cat, pointer_cat, emitted_cat

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

        if oracle_pointer_positions is not None:
            if oracle_pointer_positions.shape != (bsz, t_dec):
                raise ValueError(
                    "oracle_pointer_positions must have shape "
                    f"({bsz}, {t_dec}), got {tuple(oracle_pointer_positions.shape)}"
                )
            oracle_pointer_positions = oracle_pointer_positions.to(device=device, dtype=torch.long)

        if enc_mask is not None:
            enc_mask_long = enc_mask.long()
            max_valid_idx_global = enc_mask_long.sum(dim=1).clamp(min=1) - 1
            valid_counts = enc_mask_long.sum(dim=1, keepdim=True).clamp(min=1).to(enc_outputs.dtype)
        else:
            max_valid_idx_global = torch.full(
                (bsz,),
                t_enc - 1,
                dtype=torch.long,
                device=device,
            )
            valid_counts = None

        coverage = torch.zeros((bsz, t_enc), device=device, dtype=enc_outputs.dtype)
        pointer_idx = torch.zeros((bsz,), dtype=torch.long, device=device)
        pointer_history = []
        action_logits_all = []
        emitted_tokens = []

        for t in range(t_dec):
            if oracle_pointer_positions is not None and targets is not None:
                pointer_idx = torch.minimum(oracle_pointer_positions[:, t], max_valid_idx_global)

            emb = self.embed(input_tok)                    # (B, embed_dim)
            dec_hidden_top = hidden_state[-1]              # (B, hidden_dim)

            context, attn = self.attn(
                dec_hidden=dec_hidden_top,
                enc_outputs=enc_outputs,
                mask=enc_mask,
            )
            
            pointer_context = self._gather_pointer_context(
                enc_outputs=enc_outputs,
                pointer_idx=pointer_idx,
            )

            local_window_context, local_window_mask = self._gather_local_pointer_window(
                enc_outputs=enc_outputs,
                pointer_idx=pointer_idx,
                enc_mask=enc_mask,
            )

            fused_context = self.pointer_context_proj(
                torch.cat([context, pointer_context, local_window_context], dim=1)
            )

            coverage = coverage + attn
            if enc_mask is not None:
                coverage = coverage * enc_mask.to(coverage.dtype)
                coverage_mean = coverage.sum(dim=1, keepdim=True) / valid_counts
                max_valid_idx = max_valid_idx_global
            else:
                coverage_mean = coverage.mean(dim=1, keepdim=True)
                max_valid_idx = max_valid_idx_global

            coverage_at_pointer, coverage_window_mean, coverage_window_max = self._gather_local_coverage_stats(
                coverage=coverage,
                pointer_idx=pointer_idx,
                enc_mask=enc_mask,
            )

            pointer_norm_pos = pointer_idx.to(enc_outputs.dtype) / max_valid_idx.clamp(min=1).to(enc_outputs.dtype)
            pointer_norm_pos = pointer_norm_pos.unsqueeze(1)  # (B,1)

            progress_ratio = torch.full(
                (bsz,),
                float(t) / max(t_dec, 1),
                device=device,
                dtype=enc_outputs.dtype,
            )
            
            rnn_in = torch.cat([emb, fused_context], dim=1).unsqueeze(1)            
            out_rnn, hidden_state = self.rnn(rnn_in, hidden_state)   # out_rnn: (B, 1, H)
            dec_out = out_rnn.squeeze(1)  
            
            logits = self.out(out_rnn.squeeze(1))                   # (B, V)

            step_bias = None
            if encoder_token_bias is not None:
                bias_for_step = encoder_token_bias
                if bias_for_step.dtype != attn.dtype:
                    bias_for_step = bias_for_step.to(dtype=attn.dtype)
                step_bias = torch.bmm(
                    attn.unsqueeze(1),          # (B,1,T_enc)
                    bias_for_step               # (B,T_enc,V)
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
                fused_context,
                local_window_context,
                coverage_mean,
                coverage_at_pointer,
                coverage_window_mean,
                coverage_window_max,
                pointer_norm_pos,
                progress_ratio.unsqueeze(1),
            ], dim=1)

            action_features = torch.cat([
                dec_out,
                context,
                pointer_context,
                local_window_context,
                coverage_mean,
                coverage_at_pointer,
                coverage_window_mean,
                coverage_window_max,
                pointer_norm_pos,
                progress_ratio.unsqueeze(1),
            ], dim=1)
            
            action_logits = self.action_head(action_features)   # (B, 3)
            stop_logit = self.stop_head(stop_features).squeeze(-1)  # (B,)
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
            action_logits_all.append(action_logits.unsqueeze(1))
            pointer_history.append(pointer_idx.unsqueeze(1))
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
            
            if targets is None:
                action_pred = action_logits.argmax(dim=-1)  # 0=stay, 1=advance, 2=stop

                advance_mask = action_pred.eq(1)
                stop_mask = action_pred.eq(2)
                stop_mask = self._apply_stop_constraints(
                    stop_mask=stop_mask,
                    step_idx=t,
                    stop_logit=stop_logit,
                    progress_ratio=progress_ratio,
                    coverage_mean=coverage_mean,
                    decode_constraints=decode_constraints,
                )

                new_pointer = pointer_idx + advance_mask.long()
                pointer_idx = torch.minimum(new_pointer, max_valid_idx_global)

                if eos_id is not None:
                    eos_tensor_local = torch.full_like(next_tok, int(eos_id))

                    # Build a deterministic non-EOS fallback from current logits.
                    # If sampling produced EOS while action!=stop, use this fallback.
                    logits_no_eos = logits.clone()
                    neg_fill = -1e4 if logits_no_eos.dtype == torch.float16 else -1e9
                    logits_no_eos[:, int(eos_id)] = neg_fill
                    best_non_eos_tok = logits_no_eos.argmax(dim=-1)

                    suppressed_eos_tok = torch.where(
                        next_tok.eq(int(eos_id)),
                        best_non_eos_tok,
                        next_tok,
                    )

                    # Emit EOS only when action predicts stop.
                    next_tok = torch.where(
                        stop_mask,
                        eos_tensor_local,
                        suppressed_eos_tok,
                    )
            if eos_tensor is not None:
                emitted_tok = torch.where(finished, eos_tensor, next_tok)
            else:
                emitted_tok = next_tok

            emitted_tokens.append(emitted_tok.unsqueeze(1))
            input_tok = emitted_tok

            if token_counts is not None:
                ones = torch.ones((bsz, 1), device=device, dtype=token_counts.dtype)
                token_counts.scatter_add_(1, emitted_tok.unsqueeze(1), ones)

            # Only stop early during free decoding, not during supervised training.
            if targets is None and eos_id is not None:
                finished = finished | (input_tok == eos_id)
                if finished.all():
                    break

        logits_cat = torch.cat(outputs, dim=1)                  # (B, T_dec, V)
        attn_cat = torch.cat(attn_weights, dim=1)               # (B, T_dec, T_enc)
        stop_cat = torch.cat(stop_logits_all, dim=1)  # (B, T_dec)
        action_cat = torch.cat(action_logits_all, dim=1)        # (B, T_dec, 3)
        pointer_cat = torch.cat(pointer_history, dim=1)         # (B, T_dec)
        emitted_cat = torch.cat(emitted_tokens, dim=1)          # (B, T_dec)

        return logits_cat, hidden_state, attn_cat, stop_cat, action_cat, pointer_cat, emitted_cat
