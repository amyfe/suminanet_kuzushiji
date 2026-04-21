"""Funktionen
compute_refinement_loss(...)
compute_refine_score_loss(...)
compute_aux_classification_loss(...)
compute_decoder_ce_loss(...)
compute_stage2_total_loss(...)
Aufgabe

Saubere Loss-Zerlegung.

Empfohlene Losses für Option C v1
L_refine_box
L_refine_score
L_aux_cls optional
L_decoder_ce

Später optional:

coverage loss
ambiguity loss


ToDos:
1. Box-Loss und Delta-Loss gleichzeitig

Ich habe beide drin gelassen. Das ist als Start okay, aber nicht zwingend dauerhaft optimal.

Warum?

loss_delta passt direkt zur Parametrisierung des RefinementHeads
loss_box kontrolliert die tatsächlichen Endboxen

Das kann am Anfang stabilisieren.
Aber wenn du merkst, dass einer der beiden dominiert, dann würde ich mittelfristig eher delta loss priorisieren und den direkten box loss schwächer machen.

2. refine_score_bce_loss ist nur Positiv/Negativ, keine feinere Qualitätsregression

Das ist bewusst simpel.
Später könntest du stattdessen:

IoU als kontinuierliches Target
oder Quality Focal Loss

verwenden. Aber zuerst sollte das System überhaupt stabil lernen.

3. Aux-Klassifikation nur auf positiven ROIs

Das ist methodisch sauberer als auf allen ROIs.
Negative oder unklare Kandidaten dort zu klassifizieren wäre eher Rauschen.

4. Decoder-CE ist aktuell “plain”

Kein EOS-Bias, kein curriculum, kein scheduled weirdness.
Das ist Absicht. Altcode-Fallen vermeiden.
"""


# utils/stage2_losses.py

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def smooth_l1_box_loss(
    pred_boxes: torch.Tensor,   # (B, T, 4)
    target_boxes: torch.Tensor, # (B, T, 4)
    pos_mask: torch.Tensor,     # (B, T)
) -> torch.Tensor:
    """
    Box regression loss on positive ROIs only.
    """
    if pos_mask.sum() == 0:
        return pred_boxes.new_tensor(0.0)

    pred_pos = pred_boxes[pos_mask]
    target_pos = target_boxes[pos_mask]
    return F.smooth_l1_loss(pred_pos, target_pos, reduction="mean")


def delta_regression_loss(
    pred_deltas: torch.Tensor,    # (B, T, 4)
    target_deltas: torch.Tensor,  # (B, T, 4)
    pos_mask: torch.Tensor,       # (B, T)
) -> torch.Tensor:
    """
    Delta regression loss on positive ROIs only.

    This is typically more aligned with ROIRefinementHead than direct box loss.
    """
    if pos_mask.sum() == 0:
        return pred_deltas.new_tensor(0.0)

    pred_pos = pred_deltas[pos_mask]
    target_pos = target_deltas[pos_mask]
    return F.smooth_l1_loss(pred_pos, target_pos, reduction="mean")


def refine_score_bce_loss(
    refine_scores: torch.Tensor,      # (B, T) logits
    pos_mask: torch.Tensor,           # (B, T)
    neg_mask: torch.Tensor,           # (B, T)
    ignore_mask: Optional[torch.Tensor] = None,
    pos_weight: float = 1.0,
) -> torch.Tensor:
    """
    BCEWithLogits loss for ROI validity/quality prediction.

    Positives = matched ROIs with sufficient IoU
    Negatives = unmatched or poor ROIs
    Ignored = neither positive nor negative
    """
    valid_mask = pos_mask | neg_mask
    if ignore_mask is not None:
        valid_mask = valid_mask & (~ignore_mask)

    if valid_mask.sum() == 0:
        return refine_scores.new_tensor(0.0)

    targets = torch.zeros_like(refine_scores)
    targets[pos_mask] = 1.0

    logits_valid = refine_scores[valid_mask]
    targets_valid = targets[valid_mask]

    pos_weight_tensor = torch.tensor(
        float(pos_weight),
        device=refine_scores.device,
        dtype=refine_scores.dtype,
    )

    loss = F.binary_cross_entropy_with_logits(
        logits_valid,
        targets_valid,
        pos_weight=pos_weight_tensor,
        reduction="mean",
    )
    return loss


def aux_classification_loss(
    aux_logits: Optional[torch.Tensor],  # (B, T, V) or None
    target_labels: torch.Tensor,         # (B, T), -1 for invalid/ignore
    pos_mask: torch.Tensor,              # (B, T)
    ignore_index: int = -1,
) -> torch.Tensor:
    """
    Auxiliary ROI-level token classification loss on positive matched ROIs only.
    """
    if aux_logits is None:
        # Return zero on same device/dtype if possible
        device = target_labels.device
        return torch.tensor(0.0, device=device)

    valid_mask = pos_mask & target_labels.ne(ignore_index)
    if valid_mask.sum() == 0:
        return aux_logits.new_tensor(0.0)

    logits_valid = aux_logits[valid_mask]     # (N_pos, V)
    labels_valid = target_labels[valid_mask]  # (N_pos,)

    return F.cross_entropy(logits_valid, labels_valid, reduction="mean")


def decoder_cross_entropy_loss(
    decoder_logits: torch.Tensor,   # (B, T_dec, V)
    target_tokens: torch.Tensor,    # (B, T_dec)
    target_mask: torch.Tensor,      # (B, T_dec)
    label_smoothing: float = 0.0,
    eos_id: Optional[int] = None,
    eos_weight: float = 1.0,
) -> torch.Tensor:
    """
    Standard sequence cross-entropy on valid target positions.
    """
    if decoder_logits.dim() != 3:
        raise ValueError(
            f"decoder_logits must have shape (B, T, V), got {tuple(decoder_logits.shape)}"
        )
    if target_tokens.dim() != 2:
        raise ValueError(
            f"target_tokens must have shape (B, T), got {tuple(target_tokens.shape)}"
        )
    if target_mask.dim() != 2:
        raise ValueError(
            f"target_mask must have shape (B, T), got {tuple(target_mask.shape)}"
        )

    bsz, t_dec, vocab_size = decoder_logits.shape
    if target_tokens.shape != (bsz, t_dec):
        raise ValueError(
            f"target_tokens shape {tuple(target_tokens.shape)} must match (B, T_dec)=({bsz}, {t_dec})"
        )

    valid_mask = target_mask
    if valid_mask.sum() == 0:
        return decoder_logits.new_tensor(0.0)

    logits_valid = decoder_logits[valid_mask]  # (N, V)
    targets_valid = target_tokens[valid_mask]  # (N,)

    per_token_loss = F.cross_entropy(
        logits_valid,
        targets_valid,
        reduction="none",
        label_smoothing=float(label_smoothing),
    )

    if eos_id is not None and float(eos_weight) != 1.0:
        weights = torch.ones_like(per_token_loss)
        weights = weights.masked_fill(targets_valid.eq(int(eos_id)), float(eos_weight))
        return (per_token_loss * weights).sum() / weights.sum().clamp_min(1.0)

    return per_token_loss.mean()


def decoder_cross_entropy_loss_weighted(
    decoder_logits: torch.Tensor,   # (B, T_dec, V)
    target_tokens: torch.Tensor,    # (B, T_dec)
    target_weights: torch.Tensor,   # (B, T_dec)
    label_smoothing: float = 0.0,
    eos_id: Optional[int] = None,
    eos_weight: float = 1.0,
) -> torch.Tensor:
    """
    Weighted sequence cross-entropy on positions with positive target weight.
    """
    if decoder_logits.dim() != 3:
        raise ValueError(
            f"decoder_logits must have shape (B, T, V), got {tuple(decoder_logits.shape)}"
        )
    if target_tokens.dim() != 2:
        raise ValueError(
            f"target_tokens must have shape (B, T), got {tuple(target_tokens.shape)}"
        )
    if target_weights.dim() != 2:
        raise ValueError(
            f"target_weights must have shape (B, T), got {tuple(target_weights.shape)}"
        )

    bsz, t_dec, _ = decoder_logits.shape
    if target_tokens.shape != (bsz, t_dec):
        raise ValueError(
            f"target_tokens shape {tuple(target_tokens.shape)} must match (B, T_dec)=({bsz}, {t_dec})"
        )
    if target_weights.shape != (bsz, t_dec):
        raise ValueError(
            f"target_weights shape {tuple(target_weights.shape)} must match (B, T_dec)=({bsz}, {t_dec})"
        )

    valid_mask = target_weights.gt(0)
    if valid_mask.sum() == 0:
        return decoder_logits.new_tensor(0.0)

    logits_valid = decoder_logits[valid_mask]
    targets_valid = target_tokens[valid_mask]
    weights_valid = target_weights[valid_mask].to(dtype=decoder_logits.dtype)

    per_token_loss = F.cross_entropy(
        logits_valid,
        targets_valid,
        reduction="none",
        label_smoothing=float(label_smoothing),
    )

    if eos_id is not None and float(eos_weight) != 1.0:
        weights_valid = torch.where(
            targets_valid.eq(int(eos_id)),
            weights_valid * float(eos_weight),
            weights_valid,
        )

    return (per_token_loss * weights_valid).sum() / weights_valid.sum().clamp_min(1e-8)

def decoder_stop_bce_loss(
    stop_logits: torch.Tensor,    # (B, T_dec)
    target_tokens: torch.Tensor,  # (B, T_dec)
    target_mask: torch.Tensor,    # (B, T_dec)
    eos_id: int,
    pos_weight: float = 1.0,
    loss_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    BCE stop loss:
    target = 1 exactly on EOS token positions among valid decoder targets.
    """
    if stop_logits.dim() != 2:
        raise ValueError(f"stop_logits must have shape (B, T), got {tuple(stop_logits.shape)}")
    if target_tokens.dim() != 2:
        raise ValueError(f"target_tokens must have shape (B, T), got {tuple(target_tokens.shape)}")
    if target_mask.dim() != 2:
        raise ValueError(f"target_mask must have shape (B, T), got {tuple(target_mask.shape)}")

    if stop_logits.shape != target_tokens.shape:
        raise ValueError(
            f"stop_logits shape {tuple(stop_logits.shape)} must match target_tokens shape {tuple(target_tokens.shape)}"
        )

    valid_mask = target_mask.bool()
    if loss_mask is not None:
        if loss_mask.shape != target_mask.shape:
            raise ValueError(
                f"loss_mask shape {tuple(loss_mask.shape)} must match target_mask shape {tuple(target_mask.shape)}"
            )
        valid_mask = valid_mask & loss_mask.bool()
    if valid_mask.sum() == 0:
        return stop_logits.new_tensor(0.0)

    targets = target_tokens.eq(int(eos_id)).to(stop_logits.dtype)

    logits_valid = stop_logits[valid_mask]
    targets_valid = targets[valid_mask]

    pos_weight_tensor = torch.tensor(
        float(pos_weight),
        device=stop_logits.device,
        dtype=stop_logits.dtype,
    )

    return F.binary_cross_entropy_with_logits(
        logits_valid,
        targets_valid,
        pos_weight=pos_weight_tensor,
        reduction="mean",
    )


def build_free_prefix_masks(
    free_pred_ids: torch.Tensor,   # (B, T_use)
    target_tokens: torch.Tensor,   # (B, T_use)
    target_mask: torch.Tensor,     # (B, T_use)
    eos_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build masks for free-decoding supervision.

    Returns:
        decoder_mask:
            valid, matched non-EOS prefix only
        control_mask:
            matched prefix, plus one EOS-related divergence step if present
    """
    if free_pred_ids.shape != target_tokens.shape or target_tokens.shape != target_mask.shape:
        raise ValueError(
            "free_pred_ids, target_tokens, and target_mask must share the same shape. "
            f"Got {tuple(free_pred_ids.shape)}, {tuple(target_tokens.shape)}, {tuple(target_mask.shape)}."
        )

    decoder_mask = torch.zeros_like(target_mask, dtype=torch.bool)
    control_mask = torch.zeros_like(target_mask, dtype=torch.bool)

    bsz, t_use = target_tokens.shape
    eos_id = int(eos_id)

    for b in range(bsz):
        for t in range(t_use):
            if not bool(target_mask[b, t].item()):
                break

            pred_id = int(free_pred_ids[b, t].item())
            tgt_id = int(target_tokens[b, t].item())

            pred_is_eos = pred_id == eos_id
            tgt_is_eos = tgt_id == eos_id

            if pred_id == tgt_id:
                control_mask[b, t] = True
                if tgt_is_eos:
                    break
                decoder_mask[b, t] = True
                continue

            if pred_is_eos or tgt_is_eos:
                control_mask[b, t] = True
            break

    return decoder_mask, control_mask


def build_free_prefix_weights(
    *,
    free_pred_ids: torch.Tensor,   # (B, T_use)
    target_tokens: torch.Tensor,   # (B, T_use)
    target_mask: torch.Tensor,     # (B, T_use)
    eos_id: int,
    min_prefix_len: int = 0,
    recovery_window: int = 0,
    recovery_weight: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        decoder_weights:
            matched non-EOS prefix gets weight 1.0, a short post-mismatch
            recovery window gets a smaller positive weight.
        control_mask:
            matched prefix plus divergence step and recovery window.
        prefix_mask:
            exact matched non-EOS prefix only.
    """
    if free_pred_ids.shape != target_tokens.shape or target_tokens.shape != target_mask.shape:
        raise ValueError(
            "free_pred_ids, target_tokens, and target_mask must share the same shape. "
            f"Got {tuple(free_pred_ids.shape)}, {tuple(target_tokens.shape)}, {tuple(target_mask.shape)}."
        )

    decoder_weights = torch.zeros_like(target_tokens, dtype=torch.float32)
    control_mask = torch.zeros_like(target_mask, dtype=torch.bool)
    prefix_mask = torch.zeros_like(target_mask, dtype=torch.bool)

    bsz, t_use = target_tokens.shape
    eos_id = int(eos_id)
    min_prefix_len = max(0, int(min_prefix_len))
    recovery_window = max(0, int(recovery_window))
    recovery_weight = max(0.0, float(recovery_weight))

    for b in range(bsz):
        mismatch_idx = None
        for t in range(t_use):
            if not bool(target_mask[b, t].item()):
                break

            pred_id = int(free_pred_ids[b, t].item())
            tgt_id = int(target_tokens[b, t].item())
            tgt_is_eos = tgt_id == eos_id

            if pred_id == tgt_id:
                control_mask[b, t] = True
                if tgt_is_eos:
                    break
                prefix_mask[b, t] = True
                decoder_weights[b, t] = 1.0
                continue

            mismatch_idx = t
            control_mask[b, t] = True
            break

        if mismatch_idx is None or recovery_window <= 0 or recovery_weight <= 0.0:
            # Enforce a minimum supervised span even when mismatches happen early.
            if min_prefix_len > 0:
                valid_positions = []
                for t in range(t_use):
                    if not bool(target_mask[b, t].item()):
                        break
                    if int(target_tokens[b, t].item()) == eos_id:
                        break
                    valid_positions.append(t)
                for t in valid_positions[:min_prefix_len]:
                    decoder_weights[b, t] = max(float(decoder_weights[b, t].item()), 1.0)
            continue

        end_idx = min(t_use, mismatch_idx + recovery_window)
        for t in range(mismatch_idx, end_idx):
            if not bool(target_mask[b, t].item()):
                break
            tgt_id = int(target_tokens[b, t].item())
            control_mask[b, t] = True
            if tgt_id != eos_id:
                decoder_weights[b, t] = max(float(decoder_weights[b, t].item()), recovery_weight)

        if min_prefix_len > 0:
            valid_positions = []
            for t in range(t_use):
                if not bool(target_mask[b, t].item()):
                    break
                if int(target_tokens[b, t].item()) == eos_id:
                    break
                valid_positions.append(t)
            for t in valid_positions[:min_prefix_len]:
                decoder_weights[b, t] = max(float(decoder_weights[b, t].item()), 1.0)

    # Never apply token loss on EOS targets.
    eos_mask = target_tokens.eq(int(eos_id)) & target_mask.bool()
    decoder_weights = decoder_weights.masked_fill(eos_mask, 0.0)

    return decoder_weights, control_mask, prefix_mask


def _masked_argmax(scores: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """
    scores: (B, T_dec, T_enc)
    mask:   (B, T_enc) bool
    returns:
        argmax indices: (B, T_dec)
    """
    if mask is None:
        return scores.argmax(dim=-1)

    if mask.dim() != 2:
        raise ValueError(f"mask must have shape (B, T_enc), got {tuple(mask.shape)}")

    neg_fill = -1e4 if scores.dtype == torch.float16 else -1e9
    masked_scores = scores.masked_fill(~mask[:, None, :].bool(), neg_fill)
    return masked_scores.argmax(dim=-1)

def build_decoder_roi_targets_monotonic(
    target_tokens: torch.Tensor,      # (B, T_dec)
    target_mask: torch.Tensor,        # (B, T_dec)
    enc_mask: Optional[torch.Tensor], # (B, T_enc)
    eos_id: int,
) -> torch.Tensor:
    """
    Build monotonic pseudo ROI assignments by distributing valid non-EOS decoder steps
    across valid encoder positions.

    Returns:
        roi_targets: (B, T_dec) long in [0, T_enc-1]
    """
    if target_tokens.dim() != 2:
        raise ValueError(f"target_tokens must have shape (B, T), got {tuple(target_tokens.shape)}")
    if target_mask.dim() != 2:
        raise ValueError(f"target_mask must have shape (B, T), got {tuple(target_mask.shape)}")

    bsz, t_dec = target_tokens.shape
    device = target_tokens.device

    roi_targets = torch.zeros((bsz, t_dec), dtype=torch.long, device=device)

    if enc_mask is not None:
        if enc_mask.dim() != 2:
            raise ValueError(f"enc_mask must have shape (B, T_enc), got {tuple(enc_mask.shape)}")
        valid_enc_counts = enc_mask.sum(dim=1).clamp(min=1).long()  # (B,)
    else:
        valid_enc_counts = torch.full((bsz,), 1, dtype=torch.long, device=device)

    valid_non_eos = target_mask.bool() & target_tokens.ne(int(eos_id))  # (B, T_dec)

    for b in range(bsz):
        valid_steps = torch.nonzero(valid_non_eos[b], as_tuple=False).squeeze(1)
        if valid_steps.numel() == 0:
            continue

        num_steps = int(valid_steps.numel())
        num_rois = int(valid_enc_counts[b].item())

        if num_rois <= 1:
            roi_targets[b, valid_steps] = 0
            continue

        # map valid decoder steps monotonically onto ROI range [0, num_rois-1]
        if num_steps == 1:
            roi_targets[b, valid_steps[0]] = 0
        else:
            positions = torch.linspace(
                0,
                num_rois - 1,
                steps=num_steps,
                device=device,
                dtype=torch.float32,
            ).round().long()
            roi_targets[b, valid_steps] = positions.clamp(min=0, max=num_rois - 1)

        # propagate EOS and padded positions with previous valid ROI if possible
        # clone to avoid writing a tensor view back into overlapping storage
        last_roi = roi_targets[b, valid_steps[-1]].clone()
        eos_steps = torch.nonzero(target_mask[b].bool() & target_tokens[b].eq(int(eos_id)), as_tuple=False).squeeze(1)
        if eos_steps.numel() > 0:
            roi_targets[b, eos_steps] = last_roi

    return roi_targets

def build_decoder_roi_targets_from_alignment(
    *,
    target_tokens: torch.Tensor,          # (B, T_dec)
    target_mask: torch.Tensor,            # (B, T_dec)
    attn_weights: torch.Tensor,           # (B, T_dec, T_enc)
    enc_mask: Optional[torch.Tensor],     # (B, T_enc)
    eos_id: int,
    encoder_token_bias: Optional[torch.Tensor] = None,   # (B, T_enc, V) or None
    aux_weight: float = 0.35,
) -> torch.Tensor:
    """
    Build pseudo ROI assignments from attention and optional encoder token bias.

    Returns:
        monotonic_idx: (B, T_dec) long
    """
    if attn_weights.dim() != 3:
        raise ValueError(f"attn_weights must have shape (B, T_dec, T_enc), got {tuple(attn_weights.shape)}")

    bsz, t_dec, t_enc = attn_weights.shape
    if target_tokens.shape != (bsz, t_dec):
        raise ValueError(
            f"target_tokens shape {tuple(target_tokens.shape)} must match (B, T_dec)=({bsz}, {t_dec})"
        )
    if target_mask.shape != (bsz, t_dec):
        raise ValueError(
            f"target_mask shape {tuple(target_mask.shape)} must match (B, T_dec)=({bsz}, {t_dec})"
        )

    attn_score = torch.log(attn_weights.clamp_min(1e-8))
    combined_score = attn_score

    if encoder_token_bias is not None:
        if encoder_token_bias.dim() != 3:
            raise ValueError(
                f"encoder_token_bias must have shape (B, T_enc, V), got {tuple(encoder_token_bias.shape)}"
            )
        if encoder_token_bias.size(0) != bsz or encoder_token_bias.size(1) != t_enc:
            raise ValueError(
                f"encoder_token_bias shape {tuple(encoder_token_bias.shape)} must match (B, T_enc, V)=({bsz}, {t_enc}, V)"
            )

        vocab_size = encoder_token_bias.size(-1)
        tgt = target_tokens.clamp(min=0, max=vocab_size - 1)

        bias_exp = encoder_token_bias.unsqueeze(1).expand(-1, t_dec, -1, -1)
        tgt_idx = tgt.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, t_enc, 1)
        aux_score = bias_exp.gather(dim=-1, index=tgt_idx).squeeze(-1)

        combined_score = combined_score + float(aux_weight) * aux_score

    pseudo_roi_idx = _masked_argmax(combined_score, enc_mask)

    # monotonic projection
    monotonic_idx = pseudo_roi_idx.clone()
    for t in range(1, t_dec):
        monotonic_idx[:, t] = torch.maximum(monotonic_idx[:, t], monotonic_idx[:, t - 1])

    return monotonic_idx


def build_decoder_roi_targets_from_bias(
    *,
    target_tokens: torch.Tensor,          # (B, T_dec)
    target_mask: torch.Tensor,            # (B, T_dec)
    enc_mask: Optional[torch.Tensor],     # (B, T_enc)
    eos_id: int,
    encoder_token_bias: torch.Tensor,     # (B, T_enc, V)
    aux_weight: float = 0.35,
) -> torch.Tensor:
    """
    Build pseudo ROI assignments using encoder token bias only.
    This provides a softer oracle pointer heuristic than strict monotonic spacing.
    """
    if encoder_token_bias.dim() != 3:
        raise ValueError(
            f"encoder_token_bias must have shape (B, T_enc, V), got {tuple(encoder_token_bias.shape)}"
        )

    bsz, t_dec = target_tokens.shape
    t_enc = encoder_token_bias.size(1)
    if target_mask.shape != (bsz, t_dec):
        raise ValueError(
            f"target_mask shape {tuple(target_mask.shape)} must match (B, T_dec)=({bsz}, {t_dec})"
        )

    vocab_size = encoder_token_bias.size(-1)
    tgt = target_tokens.clamp(min=0, max=vocab_size - 1)

    bias_exp = encoder_token_bias.unsqueeze(1).expand(-1, t_dec, -1, -1)
    tgt_idx = tgt.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, t_enc, 1)
    score = bias_exp.gather(dim=-1, index=tgt_idx).squeeze(-1)
    score = score * float(aux_weight)

    pseudo_roi_idx = _masked_argmax(score, enc_mask)

    monotonic_idx = pseudo_roi_idx.clone()
    for t in range(1, t_dec):
        monotonic_idx[:, t] = torch.maximum(monotonic_idx[:, t], monotonic_idx[:, t - 1])

    # Propagate EOS positions with last valid ROI.
    valid = target_mask.bool()
    for b in range(bsz):
        valid_steps = torch.nonzero(valid[b], as_tuple=False).squeeze(1)
        if valid_steps.numel() == 0:
            continue
        last_roi = monotonic_idx[b, valid_steps[-1]].clone()
        eos_steps = torch.nonzero(
            valid[b] & target_tokens[b].eq(int(eos_id)),
            as_tuple=False,
        ).squeeze(1)
        if eos_steps.numel() > 0:
            monotonic_idx[b, eos_steps] = last_roi

    return monotonic_idx

def build_decoder_action_targets_from_alignment(
    *,
    target_tokens: torch.Tensor,          # (B, T_dec)
    target_mask: torch.Tensor,            # (B, T_dec)
    attn_weights: torch.Tensor,           # (B, T_dec, T_enc)
    enc_mask: Optional[torch.Tensor],     # (B, T_enc)
    eos_id: int,
    encoder_token_bias: Optional[torch.Tensor] = None,   # (B, T_enc, V) or None
    aux_weight: float = 0.35,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build action targets from pseudo ROI alignment.

    Returns:
        action_targets: (B, T_dec) long
            0 = stay
            1 = advance
            2 = stop
        pseudo_roi_idx: (B, T_dec) long
    """
    if attn_weights.dim() != 3:
        raise ValueError(f"attn_weights must have shape (B, T_dec, T_enc), got {tuple(attn_weights.shape)}")

    bsz, t_dec, t_enc = attn_weights.shape
    if target_tokens.shape != (bsz, t_dec):
        raise ValueError(
            f"target_tokens shape {tuple(target_tokens.shape)} must match (B, T_dec)=({bsz}, {t_dec})"
        )
    if target_mask.shape != (bsz, t_dec):
        raise ValueError(
            f"target_mask shape {tuple(target_mask.shape)} must match (B, T_dec)=({bsz}, {t_dec})"
        )

    # Base score = log attention.
    attn_score = torch.log(attn_weights.clamp_min(1e-8))
    combined_score = attn_score

    # Optional aux/token-bias compatibility term.
    if encoder_token_bias is not None:
        if encoder_token_bias.dim() != 3:
            raise ValueError(
                f"encoder_token_bias must have shape (B, T_enc, V), got {tuple(encoder_token_bias.shape)}"
            )
        if encoder_token_bias.size(0) != bsz or encoder_token_bias.size(1) != t_enc:
            raise ValueError(
                f"encoder_token_bias shape {tuple(encoder_token_bias.shape)} must match (B, T_enc, V)=({bsz}, {t_enc}, V)"
            )

        vocab_size = encoder_token_bias.size(-1)
        tgt = target_tokens.clamp(min=0, max=vocab_size - 1)

        bias_exp = encoder_token_bias.unsqueeze(1).expand(-1, t_dec, -1, -1)
        tgt_idx = tgt.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, t_enc, 1)
        aux_score = bias_exp.gather(dim=-1, index=tgt_idx).squeeze(-1)

        combined_score = combined_score + float(aux_weight) * aux_score

    pseudo_roi_idx = _masked_argmax(combined_score, enc_mask)

    # Enforce monotonic non-decreasing assignments.
    monotonic_idx = pseudo_roi_idx.clone()
    for t in range(1, t_dec):
        monotonic_idx[:, t] = torch.maximum(monotonic_idx[:, t], monotonic_idx[:, t - 1])

    action_targets = torch.zeros_like(target_tokens, dtype=torch.long)
    valid = target_mask.bool()

    # EOS -> stop.
    eos_mask = valid & target_tokens.eq(int(eos_id))
    action_targets[eos_mask] = 2

    # Non-EOS valid positions.
    for t in range(1, t_dec):
        curr_valid = valid[:, t]
        curr_non_eos = curr_valid & target_tokens[:, t].ne(int(eos_id))
        advanced = monotonic_idx[:, t].gt(monotonic_idx[:, t - 1])
        action_targets[curr_non_eos & advanced, t] = 1
        action_targets[curr_non_eos & (~advanced), t] = 0

    # First valid non-EOS step: stay by default.
    first_valid_non_eos = valid[:, 0] & target_tokens[:, 0].ne(int(eos_id))
    action_targets[first_valid_non_eos, 0] = 0

    return action_targets, monotonic_idx

def compute_free_decoder_prefix_losses(
    *,
    free_decoder_logits: torch.Tensor,          # (B, T_free, V)
    free_pred_ids: Optional[torch.Tensor],      # (B, T_free) or None
    free_stop_logits: Optional[torch.Tensor],   # (B, T_free) or None
    free_action_logits: Optional[torch.Tensor], # (B, T_free, 3) or None
    free_attn_weights: Optional[torch.Tensor],  # (B, T_free, T_enc) or None
    target_tokens: torch.Tensor,                # (B, T_tgt)
    target_mask: torch.Tensor,                  # (B, T_tgt)
    enc_mask: Optional[torch.Tensor],           # (B, T_enc)
    encoder_token_bias: Optional[torch.Tensor], # (B, T_enc, V)
    eos_id: int,
    label_smoothing: float = 0.0,
    eos_weight: float = 1.0,
    stop_pos_weight: float = 1.0,
    min_prefix_len: int = 0,
    lambda_stop: float = 0.0,
    lambda_action: float = 0.0,
    action_aux_weight: float = 0.35,
    action_class_weights: Optional[torch.Tensor] = None,
    current_epoch: int = 0,
    total_epochs: int = 1,
    action_alignment_start_epoch: int = 2,
    action_alignment_full_epoch: int = 6,
    action_degenerate_same_fraction_threshold: float = 0.90,
    action_degenerate_zero_fraction_threshold: float = 0.90,
    recovery_window: int = 0,
    recovery_weight: float = 0.0,
) -> dict:
    """
    Compute prefix-aligned losses for a free-decoding rollout.
    We align by truncating targets to the produced free length and
    only supervising a matched prefix (optionally enforcing a minimum
    supervised span).
    """
    bsz, t_free, _ = free_decoder_logits.shape
    t_tgt = target_tokens.size(1)
    t_use = min(t_free, t_tgt)

    if t_use <= 0:
        zero = free_decoder_logits.new_tensor(0.0)
        zero_long = free_decoder_logits.new_tensor(0, dtype=torch.long)
        return {
            "loss_decoder_free": zero,
            "loss_stop_free": zero,
            "loss_action_free": zero,
            "loss_total_free": zero,
            "prefix_decoder_tokens": zero_long,
            "prefix_control_tokens": zero_long,
            "prefix_target_tokens": zero_long,
            "prefix_sequence_count": zero_long,
            "recovery_decoder_tokens": zero_long,
        }

    tgt_tokens_cut = target_tokens[:, :t_use]
    tgt_mask_cut = target_mask[:, :t_use]
    dec_logits_cut = free_decoder_logits[:, :t_use, :]
    if free_pred_ids is None:
        pred_ids_cut = dec_logits_cut.argmax(dim=-1)
    else:
        if free_pred_ids.dim() != 2:
            raise ValueError(
                f"free_pred_ids must have shape (B, T_free), got {tuple(free_pred_ids.shape)}"
            )
        pred_ids_cut = free_pred_ids[:, :t_use]
    decoder_weights_free, control_mask_free, prefix_mask_free = build_free_prefix_weights(
        free_pred_ids=pred_ids_cut,
        target_tokens=tgt_tokens_cut,
        target_mask=tgt_mask_cut,
        eos_id=int(eos_id),
        min_prefix_len=int(min_prefix_len),
        recovery_window=int(recovery_window),
        recovery_weight=float(recovery_weight),
    )
    prefix_decoder_tokens = prefix_mask_free.sum()
    prefix_control_tokens = control_mask_free.sum()
    prefix_target_tokens = tgt_mask_cut.bool().sum()
    prefix_sequence_count = tgt_mask_cut.any(dim=1).sum()
    recovery_decoder_tokens = decoder_weights_free.gt(0).sum()

    loss_decoder_free = decoder_cross_entropy_loss_weighted(
        decoder_logits=dec_logits_cut,
        target_tokens=tgt_tokens_cut,
        target_weights=decoder_weights_free.to(dtype=dec_logits_cut.dtype),
        label_smoothing=label_smoothing,
        eos_id=None,
        eos_weight=1.0,
    )

    loss_stop_free = dec_logits_cut.new_tensor(0.0)
    if free_stop_logits is not None and float(lambda_stop) > 0.0:
        if free_stop_logits.dim() == 1:
            free_stop_logits = free_stop_logits.unsqueeze(1)
        stop_cut = free_stop_logits[:, :t_use]
        loss_stop_free = decoder_stop_bce_loss(
            stop_logits=stop_cut,
            target_tokens=tgt_tokens_cut,
            target_mask=tgt_mask_cut,     # ← use full valid mask so EOS position is included
            eos_id=int(eos_id),
            pos_weight=float(stop_pos_weight),
            loss_mask=None,               # no additional mask
        )

    loss_action_free = dec_logits_cut.new_tensor(0.0)
    if (
        free_action_logits is not None
        and free_attn_weights is not None
        and float(lambda_action) > 0.0
    ):
        action_cut = free_action_logits[:, :t_use, :]
        attn_cut = free_attn_weights[:, :t_use, :]

        if total_epochs <= 1:
            action_alignment_mix = 1.0 if current_epoch >= action_alignment_start_epoch else 0.0
        elif current_epoch < action_alignment_start_epoch:
            action_alignment_mix = 0.0
        elif current_epoch >= action_alignment_full_epoch:
            action_alignment_mix = 1.0
        else:
            denom = max(1, action_alignment_full_epoch - action_alignment_start_epoch)
            action_alignment_mix = float(current_epoch - action_alignment_start_epoch) / float(denom)

        loss_action_free, _, _ = decoder_action_ce_loss_from_alignment(
            action_logits=action_cut,
            target_tokens=tgt_tokens_cut,
            target_mask=tgt_mask_cut,
            attn_weights=attn_cut,
            enc_mask=enc_mask,
            eos_id=int(eos_id),
            encoder_token_bias=encoder_token_bias,
            aux_weight=float(action_aux_weight),
            class_weights=action_class_weights,
            alignment_mix=float(action_alignment_mix),
            degenerate_same_fraction_threshold=float(action_degenerate_same_fraction_threshold),
            degenerate_zero_fraction_threshold=float(action_degenerate_zero_fraction_threshold),
            loss_mask=tgt_mask_cut,
        )

    # Continuation supervision: when free decoding diverges early, still train
    # non-prefix target positions to provide a direct signal against early stop.
    post_prefix_mask = (
        tgt_mask_cut.bool()
        & ~prefix_mask_free
        & ~tgt_tokens_cut.eq(int(eos_id))
    )
    if post_prefix_mask.any():
        continuation_weight = 0.3
        continuation_alpha = 0.5
        continuation_weights = torch.where(
            post_prefix_mask,
            torch.full_like(decoder_weights_free, float(continuation_weight)),
            torch.zeros_like(decoder_weights_free),
        )
        loss_continuation = decoder_cross_entropy_loss_weighted(
            decoder_logits=dec_logits_cut,
            target_tokens=tgt_tokens_cut,
            target_weights=continuation_weights.to(dtype=dec_logits_cut.dtype),
            label_smoothing=label_smoothing,
            eos_id=None,
            eos_weight=1.0,
        )
        loss_decoder_free = loss_decoder_free + float(continuation_alpha) * loss_continuation
    total_free = (
        loss_decoder_free
        + float(lambda_stop) * loss_stop_free
        + float(lambda_action) * loss_action_free
    )
    return {
        "loss_decoder_free": loss_decoder_free,
        "loss_stop_free": loss_stop_free,
        "loss_action_free": loss_action_free,
        "loss_total_free": total_free,
        "prefix_decoder_tokens": prefix_decoder_tokens,
        "prefix_control_tokens": prefix_control_tokens,
        "prefix_target_tokens": prefix_target_tokens,
        "prefix_sequence_count": prefix_sequence_count,
        "recovery_decoder_tokens": recovery_decoder_tokens,
    }

def decoder_action_ce_loss_from_alignment(
    *,
    action_logits: torch.Tensor,          # (B, T_dec, 3)
    target_tokens: torch.Tensor,          # (B, T_dec)
    target_mask: torch.Tensor,            # (B, T_dec)
    attn_weights: torch.Tensor,           # (B, T_dec, T_enc)
    enc_mask: Optional[torch.Tensor],     # (B, T_enc)
    eos_id: int,
    encoder_token_bias: Optional[torch.Tensor] = None,  # (B, T_enc, V)
    aux_weight: float = 0.35,
    class_weights: Optional[torch.Tensor] = None,       # (3,)
    alignment_mix: float = 0.0,
    degenerate_same_fraction_threshold: float = 0.90,
    degenerate_zero_fraction_threshold: float = 0.90,
    loss_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        loss_action
        action_targets
        roi_targets
    """
    if action_logits.dim() != 3 or action_logits.size(-1) != 3:
        raise ValueError(f"action_logits must have shape (B, T, 3), got {tuple(action_logits.shape)}")

    valid_mask = target_mask.bool()
    if loss_mask is not None:
        if loss_mask.shape != target_mask.shape:
            raise ValueError(
                f"loss_mask shape {tuple(loss_mask.shape)} must match target_mask shape {tuple(target_mask.shape)}"
            )
        valid_mask = valid_mask & loss_mask.bool()
    if valid_mask.sum() == 0:
        zero = action_logits.new_tensor(0.0)
        dummy_targets = torch.zeros_like(target_tokens, dtype=torch.long)
        dummy_roi = torch.zeros_like(target_tokens, dtype=torch.long)
        return zero, dummy_targets, dummy_roi

    roi_targets = build_decoder_roi_targets_hybrid(
        target_tokens=target_tokens,
        target_mask=target_mask,
        attn_weights=attn_weights,
        enc_mask=enc_mask,
        eos_id=int(eos_id),
        encoder_token_bias=encoder_token_bias,
        aux_weight=float(aux_weight),
        alignment_mix=float(alignment_mix),
        degenerate_same_fraction_threshold=float(degenerate_same_fraction_threshold),
        degenerate_zero_fraction_threshold=float(degenerate_zero_fraction_threshold),
    )

    action_targets = build_action_targets_from_roi_targets(
        roi_targets=roi_targets,
        target_tokens=target_tokens,
        target_mask=target_mask,
        eos_id=int(eos_id),
    )

    logits_valid = action_logits[valid_mask]
    targets_valid = action_targets[valid_mask]

    loss = F.cross_entropy(
        logits_valid,
        targets_valid,
        weight=class_weights,
        reduction="mean",
    )
    return loss, action_targets, roi_targets

def build_decoder_action_targets(
    target_tokens: torch.Tensor,   # (B, T_dec)
    target_mask: torch.Tensor,     # (B, T_dec)
    eos_id: int,
) -> torch.Tensor:
    """
    Heuristic action targets:
      0 = stay
      1 = advance
      2 = stop

    Rule:
    - EOS token position -> stop
    - valid non-EOS token:
        if next valid target token exists and differs -> advance
        else stay
    """
    bsz, t_dec = target_tokens.shape
    action_targets = torch.zeros_like(target_tokens, dtype=torch.long)

    eos_mask = target_tokens.eq(int(eos_id)) & target_mask.bool()
    action_targets[eos_mask] = 2

    for t in range(t_dec - 1):
        curr_valid = target_mask[:, t].bool()
        next_valid = target_mask[:, t + 1].bool()

        curr_tok = target_tokens[:, t]
        next_tok = target_tokens[:, t + 1]

        advance_mask = curr_valid & next_valid & curr_tok.ne(int(eos_id)) & next_tok.ne(curr_tok)
        stay_mask = curr_valid & (~advance_mask) & curr_tok.ne(int(eos_id))

        action_targets[advance_mask, t] = 1
        action_targets[stay_mask, t] = 0

    return action_targets

def build_decoder_roi_targets_hybrid(
    *,
    target_tokens: torch.Tensor,          # (B, T_dec)
    target_mask: torch.Tensor,            # (B, T_dec)
    attn_weights: torch.Tensor,           # (B, T_dec, T_enc)
    enc_mask: Optional[torch.Tensor],     # (B, T_enc)
    eos_id: int,
    encoder_token_bias: Optional[torch.Tensor] = None,
    aux_weight: float = 0.35,
    alignment_mix: float = 0.0,
    degenerate_same_fraction_threshold: float = 0.90,
    degenerate_zero_fraction_threshold: float = 0.90,
) -> torch.Tensor:
    """
    Hybrid ROI targets:
    - monotonic baseline
    - optional alignment refinement
    - fallback to monotonic if alignment degenerates

    alignment_mix:
        0.0 -> pure monotonic
        1.0 -> pure alignment (subject to fallback)
    """
    roi_mono = build_decoder_roi_targets_monotonic(
        target_tokens=target_tokens,
        target_mask=target_mask,
        enc_mask=enc_mask,
        eos_id=int(eos_id),
    )

    if alignment_mix <= 0.0:
        return roi_mono

    roi_align = build_decoder_roi_targets_from_alignment(
        target_tokens=target_tokens,
        target_mask=target_mask,
        attn_weights=attn_weights,
        enc_mask=enc_mask,
        eos_id=int(eos_id),
        encoder_token_bias=encoder_token_bias,
        aux_weight=float(aux_weight),
    )

    bsz, t_dec = roi_mono.shape
    valid_mask = target_mask.bool()

    roi_final = roi_mono.clone()

    for b in range(bsz):
        valid_idx = torch.nonzero(valid_mask[b], as_tuple=False).squeeze(1)
        if valid_idx.numel() == 0:
            continue

        align_seq = roi_align[b, valid_idx]
        mono_seq = roi_mono[b, valid_idx]

        # degeneracy checks
        same_fraction = 1.0
        if align_seq.numel() > 1:
            same_fraction = float((align_seq[1:] == align_seq[:-1]).float().mean().item())

        zero_fraction = float((align_seq == 0).float().mean().item())

        degenerate = (
            same_fraction >= float(degenerate_same_fraction_threshold)
            or zero_fraction >= float(degenerate_zero_fraction_threshold)
        )

        if degenerate:
            roi_final[b, valid_idx] = mono_seq
            continue

        # soft curriculum mixing:
        # choose alignment positions on a subset of steps, monotonic elsewhere
        if alignment_mix >= 1.0:
            roi_final[b, valid_idx] = align_seq
        else:
            n = int(valid_idx.numel())
            take_align = max(0, min(n, int(round(alignment_mix * n))))
            if take_align > 0:
                # use alignment on later steps, where decoder attention is often more stable
                selected = valid_idx[-take_align:]
                roi_final[b, selected] = roi_align[b, selected]

    # enforce final monotonicity again
    for t in range(1, t_dec):
        roi_final[:, t] = torch.maximum(roi_final[:, t], roi_final[:, t - 1])

    return roi_final

def decoder_action_ce_loss(
    action_logits: torch.Tensor,   # (B, T_dec, 3)
    target_tokens: torch.Tensor,   # (B, T_dec)
    target_mask: torch.Tensor,     # (B, T_dec)
    eos_id: int,
) -> torch.Tensor:
    if action_logits.dim() != 3 or action_logits.size(-1) != 3:
        raise ValueError(f"action_logits must have shape (B, T, 3), got {tuple(action_logits.shape)}")

    valid_mask = target_mask.bool()
    if valid_mask.sum() == 0:
        return action_logits.new_tensor(0.0)

    action_targets = build_decoder_action_targets(
        target_tokens=target_tokens,
        target_mask=target_mask,
        eos_id=int(eos_id),
    )

    logits_valid = action_logits[valid_mask]        # (N, 3)
    targets_valid = action_targets[valid_mask]      # (N,)

    return F.cross_entropy(logits_valid, targets_valid, reduction="mean")

def build_action_targets_from_roi_targets(
    *,
    roi_targets: torch.Tensor,     # (B, T_dec)
    target_tokens: torch.Tensor,   # (B, T_dec)
    target_mask: torch.Tensor,     # (B, T_dec)
    eos_id: int,
) -> torch.Tensor:
    """
    0 = stay
    1 = advance
    2 = stop
    """
    bsz, t_dec = roi_targets.shape
    action_targets = torch.zeros_like(target_tokens, dtype=torch.long)
    valid = target_mask.bool()

    eos_mask = valid & target_tokens.eq(int(eos_id))
    action_targets[eos_mask] = 2

    for t in range(1, t_dec):
        curr_valid = valid[:, t]
        curr_non_eos = curr_valid & target_tokens[:, t].ne(int(eos_id))
        advanced = roi_targets[:, t].gt(roi_targets[:, t - 1])

        action_targets[curr_non_eos & advanced, t] = 1
        action_targets[curr_non_eos & (~advanced), t] = 0

    first_valid_non_eos = valid[:, 0] & target_tokens[:, 0].ne(int(eos_id))
    action_targets[first_valid_non_eos, 0] = 0

    return action_targets

def compute_stage2_total_loss(
    *,
    refined_boxes: torch.Tensor,        # (B, T, 4)
    box_deltas: torch.Tensor,           # (B, T, 4)
    refine_scores: torch.Tensor,        # (B, T)
    aux_logits: Optional[torch.Tensor], # (B, T, V) or None
    decoder_logits: torch.Tensor,       # (B, T_dec, V)
    stop_logits: Optional[torch.Tensor], # (B, T_dec) or None
    action_logits: Optional[torch.Tensor], # (B, T_dec, 3) or None
    attn_weights: Optional[torch.Tensor], # (B, T_dec, T_enc) or None
    enc_mask: Optional[torch.Tensor], # (B, T_enc) or None
    encoder_token_bias: Optional[torch.Tensor], # (B, T_enc, V) or None

    matched_gt_boxes: torch.Tensor,     # (B, T, 4)
    target_deltas: torch.Tensor,        # (B, T, 4)
    matched_gt_labels: torch.Tensor,    # (B, T)
    refine_pos_mask: torch.Tensor,      # (B, T)
    refine_neg_mask: torch.Tensor,      # (B, T)
    refine_ignore_mask: torch.Tensor,   # (B, T)

    aux_target_labels: Optional[torch.Tensor] = None,
    aux_pos_mask: Optional[torch.Tensor] = None,

    target_tokens: torch.Tensor,        # (B, T_dec)
    target_mask: torch.Tensor,          # (B, T_dec)

    lambda_box: float = 1.0,
    lambda_delta: float = 1.0,
    lambda_score: float = 0.5,
    lambda_aux: float = 0.2,
    lambda_decoder: float = 1.0,
    lambda_stop: float = 0.0,
    lambda_action: float = 0.0,
    action_aux_weight: float = 0.35,
    action_class_weights: Optional[torch.Tensor] = None,

    refine_pos_weight: float = 1.0,
    decoder_label_smoothing: float = 0.0,
    decoder_eos_id: Optional[int] = None,
    decoder_eos_weight: float = 1.0,
    decoder_stop_pos_weight: float = 1.0,

    current_epoch: int = 0,
    total_epochs: int = 1,
    action_alignment_start_epoch: int = 2,
    action_alignment_full_epoch: int = 6,
    action_degenerate_same_fraction_threshold: float = 0.90,
    action_degenerate_zero_fraction_threshold: float = 0.90,
) -> dict:
    """
    Compute full Option-C Stage-2 loss.

    Notes:
    - You usually do NOT want strong direct box loss and strong delta loss forever.
      For the start, using both is okay.
    - If one dominates, you can later reduce/remove lambda_box or lambda_delta.
    """

    if total_epochs <= 1:
        action_alignment_mix = 1.0 if current_epoch >= action_alignment_start_epoch else 0.0
    elif current_epoch < action_alignment_start_epoch:
        action_alignment_mix = 0.0
    elif current_epoch >= action_alignment_full_epoch:
        action_alignment_mix = 1.0
    else:
        denom = max(1, action_alignment_full_epoch - action_alignment_start_epoch)
        action_alignment_mix = float(current_epoch - action_alignment_start_epoch) / float(denom)

    loss_box = smooth_l1_box_loss(
        pred_boxes=refined_boxes,
        target_boxes=matched_gt_boxes,
        pos_mask=refine_pos_mask,
    )

    loss_delta = delta_regression_loss(
        pred_deltas=box_deltas,
        target_deltas=target_deltas,
        pos_mask=refine_pos_mask,
    )

    loss_score = refine_score_bce_loss(
        refine_scores=refine_scores,
        pos_mask=refine_pos_mask,
        neg_mask=refine_neg_mask,
        ignore_mask=refine_ignore_mask,
        pos_weight=refine_pos_weight,
    )

    loss_aux = aux_classification_loss(
        aux_logits=aux_logits,
        target_labels=matched_gt_labels if aux_target_labels is None else aux_target_labels,
        pos_mask=refine_pos_mask if aux_pos_mask is None else aux_pos_mask,
        ignore_index=-1,
    )

    loss_decoder = decoder_cross_entropy_loss(
        decoder_logits=decoder_logits,
        target_tokens=target_tokens,
        target_mask=target_mask,
        label_smoothing=decoder_label_smoothing,
        eos_id=decoder_eos_id,
        eos_weight=decoder_eos_weight,
    )

    loss_stop = decoder_logits.new_tensor(0.0)
    loss_action = decoder_logits.new_tensor(0.0)
    if stop_logits is not None and decoder_eos_id is not None and float(lambda_stop) > 0.0:
        loss_stop = decoder_stop_bce_loss(
            stop_logits=stop_logits,
            target_tokens=target_tokens,
            target_mask=target_mask,
            eos_id=int(decoder_eos_id),
            pos_weight=float(decoder_stop_pos_weight),
        )
    action_targets = None
    action_roi_targets = None
    if (
        action_logits is not None
        and attn_weights is not None
        and decoder_eos_id is not None
        and float(lambda_action) > 0.0
    ):
        loss_action, action_targets, action_roi_targets = decoder_action_ce_loss_from_alignment(
            action_logits=action_logits,
            target_tokens=target_tokens,
            target_mask=target_mask,
            attn_weights=attn_weights,
            enc_mask=enc_mask,
            eos_id=int(decoder_eos_id),
            encoder_token_bias=encoder_token_bias,
            aux_weight=float(action_aux_weight),
            class_weights=action_class_weights,
            alignment_mix=float(action_alignment_mix),
            degenerate_same_fraction_threshold=float(action_degenerate_same_fraction_threshold),
            degenerate_zero_fraction_threshold=float(action_degenerate_zero_fraction_threshold),
        )
    total_loss = (
        float(lambda_box) * loss_box +
        float(lambda_delta) * loss_delta +
        float(lambda_score) * loss_score +
        float(lambda_aux) * loss_aux +
        float(lambda_decoder) * loss_decoder +
        float(lambda_stop) * loss_stop +
        float(lambda_action) * loss_action
    )

    return {
        "loss_total": total_loss,
        "loss_box": loss_box,
        "loss_delta": loss_delta,
        "loss_score": loss_score,
        "loss_aux": loss_aux,
        "loss_decoder": loss_decoder,
        "loss_stop": loss_stop,
        "loss_action": loss_action,
        "action_targets": action_targets,
        "action_roi_targets": action_roi_targets,
        "action_alignment_mix": action_alignment_mix,
    }
