# utils/stage2_losses.py

from __future__ import annotations

from typing import Optional

import torch
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


def focal_classification_loss(
    aux_logits: Optional[torch.Tensor],  # (B, T, V) or None
    target_labels: torch.Tensor,         # (B, T), -1 for invalid/ignore
    pos_mask: torch.Tensor,              # (B, T)
    gamma: float = 2.0,
    ignore_index: int = -1,
    vocab_weights: Optional[torch.Tensor] = None,  # (V,) per-class multiplier, e.g. kanji boost
) -> torch.Tensor:
    """
    Focal-loss variant of aux_classification_loss.

    Down-weights easy/common examples so rare Kuzushiji characters receive
    proportionally larger gradient signal.  The focal weight (1-p_t)^gamma is
    detached so gradients flow only through the cross-entropy term.

    gamma=0 recovers plain cross-entropy; gamma=2 is the standard choice.

    vocab_weights: optional per-class weight tensor of shape (V,).  When provided,
        each token's loss is multiplied by vocab_weights[label].  Use this to
        up-weight script types that are harder to classify — e.g. kanji (1.5×)
        relative to hiragana (1.0×), following Clanuwat's observation that kanji
        need proportionally more gradient due to high visual ambiguity and sparsity.
    """
    if aux_logits is None:
        return torch.tensor(0.0, device=target_labels.device)

    valid_mask = pos_mask & target_labels.ne(ignore_index)
    if valid_mask.sum() == 0:
        return aux_logits.new_tensor(0.0)

    logits_valid = aux_logits[valid_mask]     # (N_pos, V)
    labels_valid = target_labels[valid_mask]  # (N_pos,)

    ce = F.cross_entropy(logits_valid, labels_valid, reduction="none")  # (N_pos,)
    p_t = torch.exp(-ce.detach())                                        # (N_pos,)
    focal_weight = (1.0 - p_t) ** gamma

    if vocab_weights is not None:
        # Per-token class weight: look up each token's label in the weight tensor.
        # Clamp label indices to valid range in case of rare -1 leakage.
        safe_labels = labels_valid.clamp(min=0, max=vocab_weights.size(0) - 1)
        token_w = vocab_weights[safe_labels].to(ce.dtype)
        return (focal_weight * ce * token_w).mean()

    return (focal_weight * ce).mean()


def background_classification_loss(
    char_logits: torch.Tensor,  # (B, T, V)  — V includes BG as last class
    neg_mask: torch.Tensor,     # (B, T) bool — negative (FP) ROIs
    bg_id: int,
) -> torch.Tensor:
    """Train FP proposals to predict the background class.

    Negative ROIs have no GT character match.  Training the classifier to output
    the <BG> class for them (rather than a spurious high-frequency character) reduces
    insertion errors in the assembled transcription and gives the BiGRU context
    encoder a meaningful supervised signal on every ROI, not just positives.
    """
    if not neg_mask.any():
        return char_logits.new_tensor(0.0)
    logits_neg = char_logits[neg_mask]           # (N_neg, V)
    targets = torch.full(
        (logits_neg.size(0),), bg_id,
        dtype=torch.long, device=char_logits.device,
    )
    return F.cross_entropy(logits_neg, targets, reduction="mean")


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
