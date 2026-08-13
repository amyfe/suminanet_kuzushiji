"""
ToDo:
A:
torch.sigmoid(heat_logits) + Thresholding ist okay, aber nicht automatisch gut kalibriert.
Die Proposal-Qualität hängt stark davon ab, wie gut dein Detector kalibriert ist.

B:
NMS auf historischen Zeichen kann problematisch sein, wenn Zeichen dicht liegen.
Das musst du später auf echten Seiten prüfen. Eventuell brauchst du:

kleinere IoU-Schwelle
Soft-NMS
oder line-aware filtering


Diese Datei ist die Brücke von Stage 1 zu Stage 2:
Funktionen: 
extract_coarse_proposals(...)
filter_proposals(...)
clip_boxes_to_image(...)

Aufgaben: Brücke von Stage 1 zu Stage 2:
Heatmap + bbox outputs
in coarse candidate boxes umwandeln
confidence thresholding
top-k
NMS
clipping

Erwartete Outputs

Pro Bild eine Liste/Tensor von Proposals:

(N_i, 5) oder dict
[x1, y1, x2, y2, score]

Optional später:

source
center
detector_idx

"""

# model/suminanet/detection/proposal_utils.py

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F
from torchvision.ops import nms

from utils.detection_utils import spatial_density_filter_torch


def clip_boxes_to_image(
    boxes: torch.Tensor,  # (N, 4)
    image_size: Tuple[int, int],
    min_size: float = 1.0,
) -> torch.Tensor:
    """
    Clip boxes to image boundaries and enforce valid ordering.

    Args:
        boxes: (N, 4) in xyxy format
        image_size: (H, W)
        min_size: minimum box width/height after clipping

    Returns:
        clipped boxes: (N, 4)
    """
    if boxes.numel() == 0:
        return boxes

    h_img, w_img = image_size

    x1, y1, x2, y2 = boxes.unbind(dim=1)

    x1 = x1.clamp(0, float(w_img))
    y1 = y1.clamp(0, float(h_img))
    x2 = x2.clamp(0, float(w_img))
    y2 = y2.clamp(0, float(h_img))

    x1_new = torch.minimum(x1, x2)
    y1_new = torch.minimum(y1, y2)
    x2_new = torch.maximum(x1, x2)
    y2_new = torch.maximum(y1, y2)

    # enforce positive box size
    x2_new = torch.maximum(x2_new, x1_new + min_size)
    y2_new = torch.maximum(y2_new, y1_new + min_size)

    x2_new = x2_new.clamp(0, float(w_img))
    y2_new = y2_new.clamp(0, float(h_img))

    return torch.stack([x1_new, y1_new, x2_new, y2_new], dim=1)


def decode_detector_boxes(
    heat_logits: torch.Tensor,   # (B, 1, Hf, Wf)
    bbox_reg: torch.Tensor,      # (B, 4, Hf, Wf), predicts dx, dy, bw, bh
    image_size: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Decode detector outputs into dense per-cell scores and boxes.

    Detector convention:
        bbox_reg[:, 0] = dx
        bbox_reg[:, 1] = dy
        bbox_reg[:, 2] = bw
        bbox_reg[:, 3] = bh

    where:
        cx = (grid_x + dx) * stride_w
        cy = (grid_y + dy) * stride_h
        bw = bw * stride_w
        bh = bh * stride_h

    Returns:
        scores: (B, Hf, Wf)
        boxes:  (B, Hf, Wf, 4) in image xyxy
    """
    if heat_logits.dim() != 4 or heat_logits.size(1) != 1:
        raise ValueError(f"heat_logits must have shape (B, 1, Hf, Wf), got {tuple(heat_logits.shape)}")
    if bbox_reg.dim() != 4 or bbox_reg.size(1) != 4:
        raise ValueError(f"bbox_reg must have shape (B, 4, Hf, Wf), got {tuple(bbox_reg.shape)}")

    device = heat_logits.device
    bsz, _, hf, wf = heat_logits.shape
    h_img, w_img = image_size

    stride_h = float(h_img) / float(hf)
    stride_w = float(w_img) / float(wf)

    scores = torch.sigmoid(heat_logits).squeeze(1)  # (B, Hf, Wf)

    grid_y, grid_x = torch.meshgrid(
        torch.arange(hf, device=device, dtype=bbox_reg.dtype),
        torch.arange(wf, device=device, dtype=bbox_reg.dtype),
        indexing="ij",
    )

    dx = bbox_reg[:, 0]
    dy = bbox_reg[:, 1]
    bw = bbox_reg[:, 2].clamp_min(1e-3)
    bh = bbox_reg[:, 3].clamp_min(1e-3)

    cx = (grid_x.unsqueeze(0) + dx) * stride_w
    cy = (grid_y.unsqueeze(0) + dy) * stride_h
    bw_img = bw * stride_w
    bh_img = bh * stride_h

    x1 = cx - 0.5 * bw_img
    y1 = cy - 0.5 * bh_img
    x2 = cx + 0.5 * bw_img
    y2 = cy + 0.5 * bh_img

    boxes = torch.stack([x1, y1, x2, y2], dim=-1)  # (B, Hf, Wf, 4)
    return scores, boxes


def _topk_candidates_per_image(
    scores_i: torch.Tensor,  # (Hf, Wf)
    boxes_i: torch.Tensor,   # (Hf, Wf, 4)
    score_thresh: float,
    top_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Select raw top-k detector candidates for one image.

    Only local maxima (3×3 neighbourhood) are considered so that each character
    centre contributes at most one proposal before NMS.  Without this filter,
    every cell above the threshold fires → many duplicates per character →
    aggressive NMS suppresses neighbouring characters → lower recall.

    Returns:
        cand_boxes:  (N, 4)
        cand_scores: (N,)
    """
    # local-maxima mask: keep only cells that are the max in their 3×3 patch
    pooled = F.max_pool2d(
        scores_i[None, None], kernel_size=3, stride=1, padding=1
    ).squeeze(0).squeeze(0)  # (Hf, Wf)
    peak_mask = (scores_i == pooled) & (scores_i >= float(score_thresh))

    flat_scores = scores_i.reshape(-1)
    flat_boxes  = boxes_i.reshape(-1, 4)
    flat_peaks  = peak_mask.reshape(-1)

    keep = flat_peaks
    if keep.sum() == 0:
        return (
            flat_boxes.new_zeros((0, 4)),
            flat_scores.new_zeros((0,)),
        )

    sel_scores = flat_scores[keep]
    sel_boxes  = flat_boxes[keep]

    if sel_scores.numel() > top_k:
        topk_scores, topk_idx = torch.topk(sel_scores, k=top_k, largest=True, sorted=True)
        sel_boxes  = sel_boxes[topk_idx]
        sel_scores = topk_scores

    return sel_boxes, sel_scores


def extract_coarse_proposals(
    heat_logits: torch.Tensor,      # (B, 1, Hf, Wf)
    bbox_reg: torch.Tensor,         # (B, 4, Hf, Wf)
    image_size: Tuple[int, int],
    score_thresh: float = 0.40,
    top_k: int = 256,
    nms_iou: float = 0.5,
    min_size: float = 1.0,
    density_grid: int = 8,
    density_factor: float = 5.0,
    avg_gt_per_image: int = 236,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Convert detector outputs into coarse ROI proposals.

    Returns:
        proposal_boxes_list:
            list of length B, each tensor (N_i, 4)
        proposal_scores_list:
            list of length B, each tensor (N_i,)
    """
    scores, boxes = decode_detector_boxes(
        heat_logits=heat_logits,
        bbox_reg=bbox_reg,
        image_size=image_size,
    )

    bsz = scores.size(0)
    proposal_boxes_list: List[torch.Tensor] = []
    proposal_scores_list: List[torch.Tensor] = []

    for i in range(bsz):
        cand_boxes, cand_scores = _topk_candidates_per_image(
            scores_i=scores[i],
            boxes_i=boxes[i],
            score_thresh=score_thresh,
            top_k=top_k,
        )

        if cand_boxes.numel() == 0:
            proposal_boxes_list.append(cand_boxes)
            proposal_scores_list.append(cand_scores)
            continue

        cand_boxes = clip_boxes_to_image(cand_boxes, image_size=image_size, min_size=min_size)

        keep = nms(cand_boxes, cand_scores, iou_threshold=nms_iou)
        cand_boxes = cand_boxes[keep]
        cand_scores = cand_scores[keep]

        # Suppress illustration FPs: cap proposals per spatial grid cell
        cand_boxes, cand_scores = spatial_density_filter_torch(
            cand_boxes, cand_scores, image_size,
            grid_size=density_grid,
            density_factor=density_factor,
            avg_gt_per_image=avg_gt_per_image,
        )

        proposal_boxes_list.append(cand_boxes)
        proposal_scores_list.append(cand_scores)

    return proposal_boxes_list, proposal_scores_list