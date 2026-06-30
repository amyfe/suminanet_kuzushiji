"""Erzeugt die Supervision für:
Hinweis:
Wenn später Lesereihenfolge oder andere Umsortierungen vor dem Lossaufbau machst, musst du sehr genau aufpassen, dass:

Boxen
Deltas
Labels
Positivmasken

noch korrekt zueinander passen.

Das heißt:
Target-Building sollte auf derselben ROI-Reihenfolge passieren, in der die Refinement-Outputs interpretiert werden.
Für die erste Version ist das noch handhabbar, aber später musst du bei ordered vs. unordered sauber bleiben.




ROI refinement
ROI quality/validity
optional aux classification
Wichtig

Hier wird definiert:

wie coarse ROIs an GT gematcht werden
welche proposals positiv/negativ sind
welche GT-Box zu welcher ROI gehört

Funktionen
match_coarse_to_gt_boxes(...)
build_refinement_targets(...)
build_aux_token_targets(...)


ToDos:
1. Statt greedy IoU evtl HUngarian Matching für stabilere Zuordnung verwenden
Das ist okay für eine erste Version, aber nicht optimal, wenn:
viele Proposals dicht beieinander liegen
ein GT-Objekt mehrere gute Kandidaten hat
NMS nicht sauber genug arbeitet

Dann kann Hungarian Matching stabiler sein. Aber zum Start ist greedy völlig vertretbar.

2. Positive/negative/ignore-Schwellen sind zentral

pos_iou_thresh=0.5, neg_iou_thresh=0.2 ist ein sinnvoller Start, aber keine Wahrheit.
Gerade weil deine coarse proposals aus einem noch unkalibrierten Detector kommen, musst du diese Schwellen später empirisch prüfen.

3. Decoder-Targets sind hier bewusst simpel

Ich mache hier noch keine exotische EOS-Sonderbehandlung oder curriculum-artige Tricks.
Das wäre wieder Altlasten-Denken. Erstmal soll die Option-C-Basis stabil werden.
"""


# utils/stage2_targets.py

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Pairwise IoU between two box sets.

    Args:
        boxes1: (N, 4) xyxy
        boxes2: (M, 4) xyxy

    Returns:
        iou: (N, M)
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.size(0), boxes2.size(0)))

    x1 = torch.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = torch.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = torch.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = torch.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter = inter_w * inter_h

    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) *
             (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0))
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) *
             (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0))

    union = (area1[:, None] + area2[None, :] - inter).clamp(min=1e-6)
    return inter / union


def greedy_match_iou(
    pred_boxes: torch.Tensor,   # (N, 4)
    gt_boxes: torch.Tensor,     # (M, 4)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Greedy one-to-one IoU matching.

    Returns:
        pred_idx: (K,)
        gt_idx:   (K,)
        ious:     (K,)
    """
    if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
        device = pred_boxes.device
        return (
            torch.zeros((0,), dtype=torch.long, device=device),
            torch.zeros((0,), dtype=torch.long, device=device),
            torch.zeros((0,), dtype=pred_boxes.dtype, device=device),
        )

    iou_mat = box_iou(pred_boxes, gt_boxes)  # (N, M)
    n, m = iou_mat.shape

    used_pred = set()
    used_gt = set()

    pred_idx = []
    gt_idx = []
    matched_ious = []

    flat = iou_mat.reshape(-1)
    order = torch.argsort(flat, descending=True)

    for flat_idx in order.tolist():
        pi = flat_idx // m
        gi = flat_idx % m

        if pi in used_pred or gi in used_gt:
            continue

        used_pred.add(pi)
        used_gt.add(gi)
        pred_idx.append(pi)
        gt_idx.append(gi)
        matched_ious.append(float(iou_mat[pi, gi].item()))

        if len(pred_idx) >= min(n, m):
            break

    device = pred_boxes.device
    dtype = pred_boxes.dtype

    return (
        torch.tensor(pred_idx, dtype=torch.long, device=device),
        torch.tensor(gt_idx, dtype=torch.long, device=device),
        torch.tensor(matched_ious, dtype=dtype, device=device),
    )


def hungarian_match_iou(
    iou_mat: torch.Tensor,   # (N, M)
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Optimal one-to-one assignment via Hungarian algorithm.

    Each GT box is matched to at most one proposal. Surplus proposals (N > M)
    fall back to their argmax GT for threshold classification — they will
    typically score below pos_iou_thresh and become negatives.

    Returns:
        best_gt:   (N,) long — GT index each proposal is matched to
        max_iou:   (N,) float — IoU of the match
    """
    from scipy.optimize import linear_sum_assignment

    # Start with argmax fallback for every proposal
    max_iou_fallback, best_gt_fallback = iou_mat.max(dim=1)
    best_gt = best_gt_fallback.clone()
    max_iou = max_iou_fallback.clone()

    # Optimal assignment on CPU numpy
    cost = -iou_mat.cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost)

    row_t = torch.tensor(row_ind, dtype=torch.long, device=iou_mat.device)
    col_t = torch.tensor(col_ind, dtype=torch.long, device=iou_mat.device)

    best_gt[row_t] = col_t
    max_iou[row_t] = iou_mat[row_t, col_t]

    return best_gt, max_iou


def boxes_to_deltas(
    src_boxes: torch.Tensor,   # (N, 4) coarse boxes
    tgt_boxes: torch.Tensor,   # (N, 4) GT boxes
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute relative box deltas from src_boxes to tgt_boxes.

    Parameterization matches ROIRefinementHead._apply_box_deltas approximately:
        dx = (tgt_cx - src_cx) / src_w
        dy = (tgt_cy - src_cy) / src_h
        dw = log(tgt_w / src_w)
        dh = log(tgt_h / src_h)

    Returns:
        deltas: (N, 4)
    """
    src_x1, src_y1, src_x2, src_y2 = src_boxes.unbind(dim=1)
    tgt_x1, tgt_y1, tgt_x2, tgt_y2 = tgt_boxes.unbind(dim=1)

    src_w = (src_x2 - src_x1).clamp(min=eps)
    src_h = (src_y2 - src_y1).clamp(min=eps)
    src_cx = (src_x1 + src_x2) * 0.5
    src_cy = (src_y1 + src_y2) * 0.5

    tgt_w = (tgt_x2 - tgt_x1).clamp(min=eps)
    tgt_h = (tgt_y2 - tgt_y1).clamp(min=eps)
    tgt_cx = (tgt_x1 + tgt_x2) * 0.5
    tgt_cy = (tgt_y1 + tgt_y2) * 0.5

    dx = (tgt_cx - src_cx) / src_w
    dy = (tgt_cy - src_cy) / src_h
    dw = torch.log(tgt_w / src_w)
    dh = torch.log(tgt_h / src_h)

    return torch.stack([dx, dy, dw, dh], dim=1)


def build_refinement_targets(
    coarse_boxes: torch.Tensor,   # (B, T, 4)
    roi_mask: torch.Tensor,       # (B, T)
    gt_boxes_list: List[torch.Tensor],
    gt_labels_list: Optional[List[torch.Tensor]] = None,
    pos_iou_thresh: float = 0.5,
    neg_iou_thresh: float = 0.2,
    ignore_label_ids: Optional[List[int]] = None,
    use_hungarian: bool = True,
) -> dict:
    """
    Build supervision targets for ROI refinement stage.

    For each ROI:
        - match to GT box greedily by IoU
        - assign:
            positive  if matched IoU >= pos_iou_thresh
            negative  if matched IoU <  neg_iou_thresh
            ignore    otherwise

    Args:
        coarse_boxes: (B, T, 4)
        roi_mask: (B, T)
        gt_boxes_list: list of length B, each (N_i, 4)
        gt_labels_list: optional list of length B, each (N_i,)
        pos_iou_thresh: positive threshold
        neg_iou_thresh: negative threshold

    Returns dict:
        matched_gt_boxes:   (B, T, 4)
        matched_gt_labels:  (B, T) long, -1 where unavailable/ignored
        target_deltas:      (B, T, 4)
        refine_pos_mask:    (B, T) bool
        refine_neg_mask:    (B, T) bool
        refine_ignore_mask: (B, T) bool
        matched_iou:        (B, T)
        matched_gt_index:   (B, T) long, -1 if none
    """
    bsz, t_max, _ = coarse_boxes.shape
    device = coarse_boxes.device
    dtype = coarse_boxes.dtype

    if len(gt_boxes_list) != bsz:
        raise ValueError(f"gt_boxes_list length {len(gt_boxes_list)} != batch size {bsz}")
    if gt_labels_list is not None and len(gt_labels_list) != bsz:
        raise ValueError(f"gt_labels_list length {len(gt_labels_list)} != batch size {bsz}")

    matched_gt_boxes = torch.zeros((bsz, t_max, 4), device=device, dtype=dtype)
    matched_gt_labels = torch.full((bsz, t_max), -1, device=device, dtype=torch.long)
    target_deltas = torch.zeros((bsz, t_max, 4), device=device, dtype=dtype)

    refine_pos_mask = torch.zeros((bsz, t_max), device=device, dtype=torch.bool)
    refine_neg_mask = torch.zeros((bsz, t_max), device=device, dtype=torch.bool)
    refine_ignore_mask = torch.zeros((bsz, t_max), device=device, dtype=torch.bool)

    matched_iou = torch.zeros((bsz, t_max), device=device, dtype=dtype)
    matched_gt_index = torch.full((bsz, t_max), -1, device=device, dtype=torch.long)

    # Precompute ignore-label tensor once (avoids repeated Python set / .item() loop per sample)
    _ignore_ids_t: Optional[torch.Tensor] = None
    if ignore_label_ids is not None:
        _ignore_ids_t = torch.tensor(ignore_label_ids, dtype=torch.long, device=device)

    for b in range(bsz):
        valid_t = int(roi_mask[b].sum().item())
        if valid_t == 0:
            continue

        boxes_b    = coarse_boxes[b, :valid_t]
        gt_boxes_b = gt_boxes_list[b].to(device=device, dtype=dtype)

        if gt_boxes_b.numel() == 0:
            refine_neg_mask[b, :valid_t] = True
            continue

        iou_mat = box_iou(boxes_b, gt_boxes_b)   # (valid_t, M)
        if use_hungarian:
            # One-to-one: each GT matched to at most one proposal.
            # Surplus proposals fall back to argmax (typically become negatives).
            best_gt, max_iou = hungarian_match_iou(iou_mat)
        else:
            # Many-to-one: each proposal independently picks its best GT.
            max_iou, best_gt = iou_mat.max(dim=1)

        best_gt_boxes = gt_boxes_b[best_gt]                # (valid_t, 4)
        matched_gt_boxes[b, :valid_t]  = best_gt_boxes
        target_deltas[b, :valid_t]     = boxes_to_deltas(boxes_b, best_gt_boxes)
        matched_iou[b, :valid_t]       = max_iou
        matched_gt_index[b, :valid_t]  = best_gt

        if gt_labels_list is not None:
            gt_labels_b = gt_labels_list[b].to(device=device, dtype=torch.long)
            matched_gt_labels[b, :valid_t] = gt_labels_b[best_gt]

        pos = max_iou >= pos_iou_thresh
        neg = max_iou <  neg_iou_thresh
        ign = (~pos) & (~neg)

        # Proposals matched to ignored label IDs (e.g. UNK) are excluded from all losses.
        if _ignore_ids_t is not None and gt_labels_list is not None:
            _label_ign = torch.isin(matched_gt_labels[b, :valid_t], _ignore_ids_t)
            if _label_ign.any():
                ign = ign | _label_ign
                pos = pos & ~_label_ign
                neg = neg & ~_label_ign

        refine_pos_mask[b, :valid_t]    = pos
        refine_neg_mask[b, :valid_t]    = neg
        refine_ignore_mask[b, :valid_t] = ign

    # padded positions should be neither pos nor neg nor ignore
    refine_pos_mask = refine_pos_mask & roi_mask
    refine_neg_mask = refine_neg_mask & roi_mask
    refine_ignore_mask = refine_ignore_mask & roi_mask

    return {
        "matched_gt_boxes": matched_gt_boxes,
        "matched_gt_labels": matched_gt_labels,
        "target_deltas": target_deltas,
        "refine_pos_mask": refine_pos_mask,
        "refine_neg_mask": refine_neg_mask,
        "refine_ignore_mask": refine_ignore_mask,
        "matched_iou": matched_iou,
        "matched_gt_index": matched_gt_index,
    }


def build_decoder_targets(
    text_ids: torch.Tensor,      # (B, T_text)
    pad_id: int,
) -> dict:
    """
    Build decoder supervision masks.

    This helper is intentionally simple for now.

    Returns:
        decoder_inputs: (B, T_text - 1) tokens fed to decoder (teacher forcing)
        target_tokens:  (B, T_text - 1) next-token supervision
        target_mask:    (B, T_text - 1) bool
    """
    if text_ids is None:
        raise ValueError("text_ids must not be None for decoder training.")
    if text_ids.dim() != 2:
        raise ValueError(f"text_ids must have shape (B, T), got {tuple(text_ids.shape)}")
    if text_ids.size(1) < 2:
        raise ValueError("text_ids must contain at least SOS and EOS for decoder training.")

    decoder_inputs = text_ids[:, :-1]
    target_tokens = text_ids[:, 1:]
    target_mask = target_tokens.ne(int(pad_id))
    return {
        "decoder_inputs": decoder_inputs,
        "target_tokens": target_tokens,
        "target_mask": target_mask,
    }