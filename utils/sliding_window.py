# utils/sliding_window.py
"""Sliding-window inference for full-size Edo page images.

Real Edo manuscript scans are 2000-4000px.  The model was trained on 512px
tiles, so directly resizing loses character detail.  This module divides the
full-resolution image into overlapping 512×512 tiles, runs detection on each
tile, maps boxes back to full-image coordinates, and merges duplicates with NMS.

Usage (inference only, no training changes):
    from utils.sliding_window import sliding_window_transcribe
    text = sliding_window_transcribe(model, image_tensor, vocab, device="cuda")
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torchvision.ops import nms


def _make_tiles(
    image: torch.Tensor,       # (3, H, W)
    tile_size: int = 512,
    overlap: int = 64,
) -> List[Tuple[torch.Tensor, int, int]]:
    """
    Divide a single image into overlapping square tiles.

    Returns:
        list of (tile_tensor, y0, x0) where (y0, x0) is the top-left
        corner of each tile in full-image coordinates.
    """
    _, H, W = image.shape
    step = tile_size - overlap

    tiles = []
    y0 = 0
    while y0 < H:
        y1 = min(y0 + tile_size, H)
        x0 = 0
        while x0 < W:
            x1 = min(x0 + tile_size, W)

            # Crop and pad to tile_size × tile_size if near borders
            tile = image[:, y0:y1, x0:x1]
            ph = tile_size - (y1 - y0)
            pw = tile_size - (x1 - x0)
            if ph > 0 or pw > 0:
                tile = F.pad(tile, (0, pw, 0, ph))

            tiles.append((tile, y0, x0))
            x0 += step
        y0 += step

    return tiles


def _rescale_boxes(
    boxes: torch.Tensor,   # (N, 4) xyxy in tile coords
    y0: int,
    x0: int,
    tile_size: int,
    orig_tile_h: int,      # actual (un-padded) tile height
    orig_tile_w: int,
) -> torch.Tensor:
    """Map tile-coordinate boxes back to full-image coordinates."""
    if boxes.numel() == 0:
        return boxes

    # Scale from [0, tile_size] back to [0, orig_tile_*] is identity when
    # orig_tile == tile_size; handles padded border tiles correctly.
    sx = float(orig_tile_w) / float(tile_size)
    sy = float(orig_tile_h) / float(tile_size)

    out = boxes.clone().float()
    out[:, 0] = boxes[:, 0] * sx + x0
    out[:, 1] = boxes[:, 1] * sy + y0
    out[:, 2] = boxes[:, 2] * sx + x0
    out[:, 3] = boxes[:, 3] * sy + y0
    return out


@torch.no_grad()
def sliding_window_detect(
    model,
    image: torch.Tensor,          # (3, H, W)  full-resolution, values in [0, 255]
    device: str = "cuda",
    tile_size: int = 512,
    overlap: int = 64,
    nms_iou: float = 0.35,
    score_thresh: float = 0.06,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run detection on a full-resolution image using a sliding window.

    Returns:
        all_boxes:  (N, 4) xyxy merged boxes in full-image coordinates
        all_scores: (N,)   detection scores
    """
    from model.kuronet.detection.proposal_utils import extract_coarse_proposals

    _, H, W = image.shape
    tiles = _make_tiles(image, tile_size=tile_size, overlap=overlap)

    all_boxes: List[torch.Tensor] = []
    all_scores: List[torch.Tensor] = []

    model.eval()

    for tile_img, y0, x0 in tiles:
        orig_h = min(tile_size, H - y0)
        orig_w = min(tile_size, W - x0)

        # Normalize to ImageNet stats (same as training transforms)
        _mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        _std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        tile_tensor = (tile_img.float().unsqueeze(0).to(device) / 255.0 - _mean) / _std

        shared_feats = model.backbone(tile_tensor)
        det_out = model.detector(shared_feats)

        boxes_list, scores_list = extract_coarse_proposals(
            heat_logits=det_out["heatmap"],
            bbox_reg=det_out["bbox"],
            image_size=(tile_size, tile_size),
            score_thresh=score_thresh,
            top_k=model.det_top_k,
            nms_iou=model.det_nms_iou,
            min_size=model.det_min_box_size,
        )

        if boxes_list[0].numel() > 0:
            full_boxes = _rescale_boxes(boxes_list[0], y0, x0, tile_size, orig_h, orig_w)
            all_boxes.append(full_boxes)
            all_scores.append(scores_list[0])

    if not all_boxes:
        return torch.zeros((0, 4)), torch.zeros((0,))

    merged_boxes = torch.cat(all_boxes, dim=0)
    merged_scores = torch.cat(all_scores, dim=0)

    # Merge overlapping boxes from adjacent tiles
    keep = nms(merged_boxes.to(device), merged_scores.to(device), nms_iou)
    return merged_boxes[keep], merged_scores[keep]


@torch.no_grad()
def sliding_window_transcribe(
    model,
    image: torch.Tensor,          # (3, H, W)  full-resolution, values in [0, 255]
    vocab,
    orientation: str = "vertical",
    device: str = "cuda",
    tile_size: int = 512,
    overlap: int = 64,
    nms_iou: float = 0.35,
    score_thresh: float = 0.06,
    cer_score_thresh: float = 0.0,
) -> str:
    """
    Full pipeline for a single full-resolution Edo page image.

    1. Sliding-window detection → merged full-image proposal boxes
    2. Single-pass recognition on the full image (resized to tile_size)
       with the merged boxes as pre-computed proposals
    3. Returns the assembled transcription string

    Note: step 2 resizes the image to tile_size for the recognition backbone,
    scaling the merged boxes accordingly.  This is the correct approach because
    the recognition model was trained at tile_size resolution.
    """
    model.eval()
    _, H, W = image.shape

    # Step 1: detect on full resolution
    merged_boxes, merged_scores = sliding_window_detect(
        model, image, device=device,
        tile_size=tile_size, overlap=overlap,
        nms_iou=nms_iou, score_thresh=score_thresh,
    )

    if merged_boxes.numel() == 0:
        return ""

    # Step 2: letterbox-resize image to tile_size for recognition backbone.
    # Scale the long edge to tile_size, pad the short edge with ImageNet-mean pixels.
    # This preserves aspect ratio so box proposals are not distorted for non-square pages.
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
    std  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

    scale = float(tile_size) / float(max(H, W))
    new_h = int(round(H * scale))
    new_w = int(round(W * scale))
    pad_top  = (tile_size - new_h) // 2
    pad_left = (tile_size - new_w) // 2

    img_scaled = F.interpolate(
        image.float().unsqueeze(0) / 255.0,
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
    )  # (1, 3, new_h, new_w)

    # Fill canvas with ImageNet mean value (normalises to 0.0 — neutral background)
    mean_t = mean.view(1, 3, 1, 1).to(image.device)
    img_canvas = mean_t.expand(1, 3, tile_size, tile_size).clone()
    img_canvas[:, :, pad_top:pad_top + new_h, pad_left:pad_left + new_w] = img_scaled

    img_resized = (img_canvas - mean_t) / std.view(1, 3, 1, 1).to(image.device)

    # Scale boxes uniformly (same scale in x and y) then shift for padding offset
    scaled_boxes = merged_boxes.clone() * scale
    scaled_boxes[:, 0] += pad_left   # x1
    scaled_boxes[:, 1] += pad_top    # y1
    scaled_boxes[:, 2] += pad_left   # x2
    scaled_boxes[:, 3] += pad_top    # y2

    coarse_boxes_list = [scaled_boxes.to(device)]
    coarse_scores_list = [merged_scores.to(device)]

    # Step 3: recognition forward pass with pre-computed proposals
    img_tensor = img_resized.to(device)
    outputs = model.forward(
        images=img_tensor,
        orientations=[orientation],
        coarse_boxes_list=coarse_boxes_list,
        coarse_scores_list=coarse_scores_list,
    )

    # Decode
    char_logits = outputs["char_logits"]    # (1, T, V)
    ordered_mask = outputs["ordered_mask"]  # (1, T)
    refine_scores = outputs["refine_scores"]
    sort_indices = outputs["sort_indices"]

    mask_b = ordered_mask[0]

    if cer_score_thresh > 0.0 and sort_indices is not None:
        si = sort_indices[0]
        valid_si = si[mask_b]
        scores_ord = refine_scores[0].index_select(0, valid_si)
        score_mask = torch.sigmoid(scores_ord) >= cer_score_thresh
        valid_positions = mask_b.nonzero(as_tuple=True)[0][score_mask]
    else:
        valid_positions = mask_b.nonzero(as_tuple=True)[0]

    if valid_positions.numel() == 0:
        return ""

    logits_b = char_logits[0, valid_positions]
    pred_ids = logits_b.argmax(dim=-1).tolist()

    if model.bg_id is not None:
        pred_ids = [p for p in pred_ids if p != model.bg_id]

    chars = vocab.decode(pred_ids, remove_special=True)
    return "".join(chars)
