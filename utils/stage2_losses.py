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

def decoder_stop_bce_loss(
    stop_logits: torch.Tensor,    # (B, T_dec)
    target_tokens: torch.Tensor,  # (B, T_dec)
    target_mask: torch.Tensor,    # (B, T_dec)
    eos_id: int,
    pos_weight: float = 1.0,
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
def compute_stage2_total_loss(
    *,
    refined_boxes: torch.Tensor,        # (B, T, 4)
    box_deltas: torch.Tensor,           # (B, T, 4)
    refine_scores: torch.Tensor,        # (B, T)
    aux_logits: Optional[torch.Tensor], # (B, T, V) or None
    decoder_logits: torch.Tensor,       # (B, T_dec, V)
    stop_logits: Optional[torch.Tensor], # (B, T_dec) or None

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

    refine_pos_weight: float = 1.0,
    decoder_label_smoothing: float = 0.0,
    decoder_eos_id: Optional[int] = None,
    decoder_eos_weight: float = 1.0,
    decoder_stop_pos_weight: float = 1.0,
) -> dict:
    """
    Compute full Option-C Stage-2 loss.

    Notes:
    - You usually do NOT want strong direct box loss and strong delta loss forever.
      For the start, using both is okay.
    - If one dominates, you can later reduce/remove lambda_box or lambda_delta.
    """
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
    if stop_logits is not None and decoder_eos_id is not None and float(lambda_stop) > 0.0:
        loss_stop = decoder_stop_bce_loss(
            stop_logits=stop_logits,
            target_tokens=target_tokens,
            target_mask=target_mask,
            eos_id=int(decoder_eos_id),
            pos_weight=float(decoder_stop_pos_weight),
        )
    total_loss = (
        float(lambda_box) * loss_box +
        float(lambda_delta) * loss_delta +
        float(lambda_score) * loss_score +
        float(lambda_aux) * loss_aux +
        float(lambda_decoder) * loss_decoder +
        float(lambda_stop) * loss_stop
    )

    return {
        "loss_total": total_loss,
        "loss_box": loss_box,
        "loss_delta": loss_delta,
        "loss_score": loss_score,
        "loss_aux": loss_aux,
        "loss_decoder": loss_decoder,
        "loss_stop": loss_stop,
    }