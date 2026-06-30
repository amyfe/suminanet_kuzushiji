"""Helper functions for detection training."""
import math
from typing import List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import roi_align
from .focal_loss import focal_loss_heatmap


def build_detection_targets(
    boxes,
    labels,
    output_size,
    image_size,
    device,
    sigma=1.0,
    sigma_floor=1.5,    # raised from 0.5 — ensures small chars get learnable targets
    sigma_ceil=3.0,     # raised from 2.0 — gives large chars proportional coverage
    sigma_scale=0.20,   # multiplier: sigma_eff = sigma * sqrt(bw*bh) * sigma_scale
    bbox_radius=0,      # 0 = only center cell (most stable)
    heatmap_min=1e-6):
    """
    Build detection targets (heatmap, bbox regression, class) from ground truth boxes.
    
    Args:
        boxes: List of (N_i, 4) tensors, one per image in batch
        labels: List of (N_i,) tensors with class IDs
        output_size: (H_out, W_out) detection head output size
        image_size: (H_in, W_in) input image size
        device: torch device
        sigma: Gaussian sigma for heatmap
        
    Returns:
        gt_heatmap: (B, 1, H_out, W_out)
        gt_bbox: (B, 4, H_out, W_out)
        gt_cls: (B, H_out, W_out) with class IDs or -1 for background
    """
    B = len(boxes)
    H_out, W_out = output_size
    H_in, W_in = image_size
    
    stride_h = float(H_in) / float(H_out)
    stride_w = float(W_in) / float(W_out)
    
    gt_heatmap = torch.zeros((B, 1, H_out, W_out), device=device)
    gt_bbox = torch.zeros((B, 4, H_out, W_out), device=device)
    gt_bbox_mask = torch.zeros((B, H_out, W_out), dtype=torch.bool, device=device)
    gt_cls = torch.full((B, H_out, W_out), -1, dtype=torch.long, device=device)
    
    for i in range(B):
        boxes_i = boxes[i].to(device) if boxes[i].numel() > 0 else torch.empty((0, 4), device=device)
        labels_i = labels[i].to(device) if labels[i].numel() > 0 else torch.empty((0,), dtype=torch.long, device=device)
        
        if boxes_i.numel() == 0:
            continue
            
        for box, label in zip(boxes_i, labels_i):
            x1, y1, x2, y2 = box.tolist()
            
            # Box center in output grid coordinates
            cx = (x1 + x2) *0.5 / stride_w
            cy = (y1 + y2) *0.5 / stride_h
            
            if cx < 0 or cy < 0 or cx >= W_out or cy >= H_out:
                continue
                
            ix = int(cx)
            iy = int(cy)
            ix = max(0, min(ix, W_out - 1))
            iy = max(0, min(iy, H_out - 1))
            
            # Box size in output grid units
            bw = max((x2 - x1) / stride_w, 1.0)
            bh = max((y2 - y1) / stride_h, 1.0)
            
            # Gaussian heatmap
            dx = cx - float(ix)
            dy = cy - float(iy)

            gaussian_sigma = min(sigma_ceil, max(sigma_floor, sigma * math.sqrt(bw * bh) * sigma_scale))
            yy = torch.arange(0, H_out, device=device).view(H_out, 1).float()
            xx = torch.arange(0, W_out, device=device).view(1, W_out).float()
            g = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * gaussian_sigma ** 2))
            gt_heatmap[i, 0] = torch.maximum(gt_heatmap[i, 0], g)
            
            # Bbox targets — supervise a (2*radius+1)² neighbourhood around center.
            # Each cell gets the offset from *its own* position to the object center,
            # so dx/dy are different per cell while bw/bh stay the same.
            x0 = max(0, ix - bbox_radius)
            x1i = min(W_out - 1, ix + bbox_radius)
            y0 = max(0, iy - bbox_radius)
            y1i = min(H_out - 1, iy + bbox_radius)

            for ny in range(y0, y1i + 1):
                for nx in range(x0, x1i + 1):
                    gt_bbox[i, 0, ny, nx] = cx - float(nx)
                    gt_bbox[i, 1, ny, nx] = cy - float(ny)
                    gt_bbox[i, 2, ny, nx] = bw
                    gt_bbox[i, 3, ny, nx] = bh
                    gt_bbox_mask[i, ny, nx] = True

            # class label only at center cell
            gt_cls[i, iy, ix] = int(label.item())
    gt_heatmap = gt_heatmap.clamp(min=heatmap_min, max=1.0)
    return gt_heatmap, gt_bbox, gt_bbox_mask, gt_cls


def spatial_density_filter(boxes, scores, image_size,
                           grid_size=8, density_factor=3.0,
                           avg_gt_per_image=236):
    """Suppress proposals from over-dense grid cells to remove illustration FPs.

    Divides the image into grid_size×grid_size cells and caps each cell at
    floor(density_factor × expected_chars_per_cell) proposals (keeping the
    highest-scoring ones).  Cells with brushstroke / illustration activations
    routinely exceed this cap; genuine text columns do not.

    Works on numpy arrays or plain Python lists; returns the same types.
    """
    if len(boxes) == 0:
        return boxes, scores

    boxes_arr  = np.asarray(boxes,  dtype=np.float32)
    scores_arr = np.asarray(scores, dtype=np.float32)

    H, W = image_size
    cell_h = H / grid_size
    cell_w = W / grid_size

    expected_per_cell = avg_gt_per_image / (grid_size * grid_size)
    max_per_cell = max(5, int(expected_per_cell * density_factor))

    cx = 0.5 * (boxes_arr[:, 0] + boxes_arr[:, 2])
    cy = 0.5 * (boxes_arr[:, 1] + boxes_arr[:, 3])
    ci = np.clip((cy / cell_h).astype(int), 0, grid_size - 1)
    cj = np.clip((cx / cell_w).astype(int), 0, grid_size - 1)
    cell_keys = ci * grid_size + cj

    # Process in score-descending order so we keep the most confident predictions
    order = np.argsort(scores_arr)[::-1]
    cell_counts: dict[int, int] = {}
    keep = []
    for i in order:
        c = int(cell_keys[i])
        cnt = cell_counts.get(c, 0)
        if cnt < max_per_cell:
            keep.append(int(i))
            cell_counts[c] = cnt + 1

    keep = np.array(keep, dtype=np.int64)
    filtered_boxes  = boxes_arr[keep].tolist()
    filtered_scores = scores_arr[keep].tolist()
    return filtered_boxes, filtered_scores


def spatial_density_filter_torch(boxes, scores, image_size,
                                  grid_size=8, density_factor=3.0,
                                  avg_gt_per_image=236):
    """Torch-tensor version of spatial_density_filter for use inside model forward passes."""
    if boxes.numel() == 0:
        return boxes, scores

    H, W = image_size
    cell_h = H / grid_size
    cell_w = W / grid_size

    expected_per_cell = avg_gt_per_image / (grid_size * grid_size)
    max_per_cell = max(5, int(expected_per_cell * density_factor))

    cx = 0.5 * (boxes[:, 0] + boxes[:, 2])
    cy = 0.5 * (boxes[:, 1] + boxes[:, 3])
    ci = (cy / cell_h).long().clamp(0, grid_size - 1)
    cj = (cx / cell_w).long().clamp(0, grid_size - 1)
    cell_keys = ci * grid_size + cj

    order = torch.argsort(scores, descending=True)
    cell_counts: dict[int, int] = {}
    keep = []
    for idx in order.tolist():
        c = int(cell_keys[idx].item())
        cnt = cell_counts.get(c, 0)
        if cnt < max_per_cell:
            keep.append(idx)
            cell_counts[c] = cnt + 1

    if not keep:
        return boxes[:0], scores[:0]
    keep_t = torch.tensor(keep, device=boxes.device, dtype=torch.long)
    return boxes[keep_t], scores[keep_t]


def compute_detection_losses(pred, gt_heatmap, gt_bbox, gt_bbox_mask,gt_cls, weights=(1.0, 1.0, 1.0), use_focal_loss=True):
    """
    Compute detection losses.
    
    Args:
        pred: dict with 'heatmap', 'bbox', 'cls' from DetectorHead
        gt_heatmap: (B, 1, H, W)
        gt_bbox: (B, 4, H, W)
        gt_cls: (B, H, W)
        weights: (w_heat, w_bbox, w_cls) loss weights
        use_focal_loss: whether to use Focal Loss for heatmap (better for small objects)
        
    Returns:
        total_loss, (loss_heat, loss_bbox, loss_cls)
    """
    w_heat, w_bbox, w_cls = weights
    
    # Heatmap loss - use Focal Loss for better small object detection (furigana)
    if 'heatmap' in pred:
        if use_focal_loss:
            loss_heat = focal_loss_heatmap(pred['heatmap'], gt_heatmap, alpha=0.25, gamma=2.0)
        else:
            loss_heat = F.mse_loss(pred['heatmap'], gt_heatmap)
    else:
        loss_heat = torch.tensor(0.0, device=gt_heatmap.device)
    
    if 'bbox' in pred:
        if gt_bbox_mask.sum() > 0:
            pred_bbox = pred['bbox'].permute(0, 2, 3, 1)[gt_bbox_mask]  # (N_pos, 4)
            gt_bbox_pos = gt_bbox.permute(0, 2, 3, 1)[gt_bbox_mask]
            loss_bbox = F.smooth_l1_loss(pred_bbox, gt_bbox_pos)
        else:
            loss_bbox = torch.tensor(0.0, device=gt_heatmap.device)
    else:
        loss_bbox = torch.tensor(0.0, device=gt_heatmap.device)

    
    # Classification loss (CE at valid positions)
    if 'cls' in pred:
        logits = pred['cls'].permute(0, 2, 3, 1).reshape(-1, pred['cls'].shape[1])
        labels_flat = gt_cls.reshape(-1)
        valid = labels_flat >= 0
    
        if valid.sum() > 0:
            loss_cls = F.cross_entropy(logits[valid], labels_flat[valid])
        else:
            loss_cls = torch.tensor(0.0, device=gt_cls.device)
    else:
        loss_cls = torch.tensor(0.0, device=gt_cls.device)
    
    total_loss = w_heat * loss_heat + w_bbox * loss_bbox + w_cls * loss_cls
    
    return total_loss, (loss_heat, loss_bbox, loss_cls)


def _bbox_iou(boxes1, boxes2):
    """Compute IoU between matched boxes (B, 4)."""
    x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    y2 = torch.min(boxes1[:, 3], boxes2[:, 3])

    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter = inter_w * inter_h

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    union = (area1 + area2 - inter).clamp(min=1e-6)
    return inter / union


def _interval_iou_1d(pred_x, gt_x):
    """Compute pairwise 1D IoU for x-intervals, pred_x:(N,2), gt_x:(M,2)."""
    n = pred_x.size(0)
    m = gt_x.size(0)
    if n == 0 or m == 0:
        return pred_x.new_zeros((n, m))

    px1 = pred_x[:, 0].unsqueeze(1)
    px2 = pred_x[:, 1].unsqueeze(1)
    gx1 = gt_x[:, 0].unsqueeze(0)
    gx2 = gt_x[:, 1].unsqueeze(0)

    inter = (torch.minimum(px2, gx2) - torch.maximum(px1, gx1)).clamp(min=0)
    plen = (px2 - px1).clamp(min=1e-6)
    glen = (gx2 - gx1).clamp(min=1e-6)
    union = (plen + glen - inter).clamp(min=1e-6)
    return inter / union


def _greedy_iou_match(pred_boxes, gt_boxes, use_x_only=False):
    """Greedy one-to-one matching between predictions and GT by highest IoU."""
    n = pred_boxes.size(0)
    m = gt_boxes.size(0)
    if n == 0 or m == 0:
        return pred_boxes.new_zeros((0,), dtype=torch.long), gt_boxes.new_zeros((0,), dtype=torch.long)

    if use_x_only:
        pred_x = pred_boxes[:, [0, 2]]
        gt_x = gt_boxes[:, [0, 2]]
        iou_mat = _interval_iou_1d(pred_x, gt_x)
    else:
        iou_mat = _bbox_iou(
            pred_boxes.unsqueeze(1).expand(-1, m, -1).reshape(-1, 4),
            gt_boxes.unsqueeze(0).expand(n, -1, -1).reshape(-1, 4),
        ).reshape(n, m)

    used_pred = set()
    used_gt = set()
    pred_idx = []
    gt_idx = []

    flat_scores = iou_mat.reshape(-1)
    order = torch.argsort(flat_scores, descending=True)
    for flat_i in order.tolist():
        pi = flat_i // m
        gi = flat_i % m
        if pi in used_pred or gi in used_gt:
            continue
        used_pred.add(pi)
        used_gt.add(gi)
        pred_idx.append(pi)
        gt_idx.append(gi)
        if len(pred_idx) >= min(n, m):
            break

    if len(pred_idx) == 0:
        return pred_boxes.new_zeros((0,), dtype=torch.long), gt_boxes.new_zeros((0,), dtype=torch.long)

    return (
        torch.tensor(pred_idx, device=pred_boxes.device, dtype=torch.long),
        torch.tensor(gt_idx, device=gt_boxes.device, dtype=torch.long),
    )


def _extract_all_peaks(
    heatmap_probs: torch.Tensor,   # (1, 1, H, W) already sigmoid
    bbox_reg: torch.Tensor,        # (1, 4, H, W) = (dx, dy, bw, bh)
    output_size: tuple,            # (Hf, Wf)
    image_size: tuple,             # (H_img, W_img)
    top_k: int = 2000,
    min_box_size: float = 2.0,
) -> Tuple[List[List[float]], List[float]]:
    """Extract ALL decoded peaks without confidence threshold or NMS.

    Returns (boxes, scores) as Python lists.  Used to cache raw proposal candidates
    for post-hoc threshold sweeps and gap-filling.
    """
    hm = heatmap_probs[0, 0]
    bbox = bbox_reg[0]

    pooled = F.max_pool2d(hm[None, None], kernel_size=3, stride=1, padding=1).squeeze()
    peak_mask = hm == pooled
    peak_idx = peak_mask.nonzero(as_tuple=False)

    if peak_idx.numel() == 0:
        return [], []

    scores = hm[peak_mask]
    sorted_scores, order = torch.sort(scores, descending=True)
    if top_k is not None and len(order) > top_k:
        order = order[:top_k]

    ys = peak_idx[order, 0].detach().cpu().numpy()
    xs = peak_idx[order, 1].detach().cpu().numpy()
    scores_np = sorted_scores[:len(order)].detach().cpu().numpy().tolist()

    H_img, W_img = image_size
    H_out, W_out = output_size
    stride_h = H_img / float(H_out)
    stride_w = W_img / float(W_out)
    bbox_np = bbox.detach().cpu().numpy()

    boxes: List[List[float]] = []
    scs: List[float] = []
    for y, x, sc in zip(ys, xs, scores_np):
        dx, dy, bw, bh = bbox_np[:, y, x]
        cx = (x + dx) * stride_w
        cy = (y + dy) * stride_h
        w = max(bw * stride_w, min_box_size)
        h = max(bh * stride_h, min_box_size)
        x1 = float(np.clip(cx - 0.5 * w, 0, W_img))
        y1 = float(np.clip(cy - 0.5 * h, 0, H_img))
        x2 = float(np.clip(cx + 0.5 * w, 0, W_img))
        y2 = float(np.clip(cy + 0.5 * h, 0, H_img))
        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2, y2])
            scs.append(float(sc))

    return boxes, scs


def fill_detection_gaps(
    pred_boxes: List[List[float]],
    pred_scores: List[float],
    all_boxes: List[List[float]],
    all_scores: List[float],
    lower_thresh: float = 0.35,
    gap_factor: float = 1.8,
    col_merge_factor: float = 1.5,
    max_fill: int = 5,
    nms_iou: float = 0.4,
) -> Tuple[List[List[float]], List[float]]:
    """Fill gaps in column-structured detections using subthreshold heatmap peaks.

    After NMS+density filtering, Stage 1 sometimes misses every other character in a
    dense column (NMS suppression or merged Gaussian peaks).  This function:
      1. Clusters detections into vertical text columns by x-centre proximity.
      2. Within each column, finds y-gaps > gap_factor × median_height.
      3. For each gap, searches *all_boxes* (pre-extracted subthreshold peaks) for a
         candidate with score >= lower_thresh whose centre falls in the gap region.
      4. Accepts top-scoring candidates and merges them in; runs NMS to avoid duplicates.

    Returns augmented (boxes, scores).  If no real peak candidates are found for a gap,
    the gap is left unfilled (no synthesis).
    """
    if len(pred_boxes) < 4 or not all_boxes:
        return pred_boxes, pred_scores

    pred_arr   = np.asarray(pred_boxes,  dtype=np.float32)   # (N, 4)
    pred_sc    = np.asarray(pred_scores, dtype=np.float32)
    all_arr    = np.asarray(all_boxes,   dtype=np.float32)   # (M, 4)
    all_sc     = np.asarray(all_scores,  dtype=np.float32)

    pred_cx = 0.5 * (pred_arr[:, 0] + pred_arr[:, 2])
    pred_cy = 0.5 * (pred_arr[:, 1] + pred_arr[:, 3])
    pred_w  =        pred_arr[:, 2] - pred_arr[:, 0]
    pred_h  =        pred_arr[:, 3] - pred_arr[:, 1]

    all_cx  = 0.5 * (all_arr[:, 0] + all_arr[:, 2])
    all_cy  = 0.5 * (all_arr[:, 1] + all_arr[:, 3])

    # Pre-filter all_boxes to candidates above lower_thresh
    cand_mask = all_sc >= lower_thresh
    if not cand_mask.any():
        return pred_boxes, pred_scores
    cand_arr = all_arr[cand_mask]
    cand_sc  = all_sc[cand_mask]
    cand_cx  = all_cx[cand_mask]
    cand_cy  = all_cy[cand_mask]

    # Column clustering: sort by x-centre, break where gap > col_merge_factor × median_width
    w_med  = float(np.median(pred_w))
    col_thresh = max(col_merge_factor * w_med, 5.0)
    order_x    = np.argsort(pred_cx)
    sorted_cx  = pred_cx[order_x]

    cols: List[List[int]] = []
    cur: List[int] = [int(order_x[0])]
    for k in range(1, len(order_x)):
        if sorted_cx[k] - sorted_cx[k - 1] > col_thresh:
            cols.append(cur)
            cur = []
        cur.append(int(order_x[k]))
    cols.append(cur)

    extra_boxes:  List[List[float]] = []
    extra_scores: List[float]       = []

    for col_idx in cols:
        if len(col_idx) < 2:
            continue
        col_cy  = pred_cy[col_idx]
        col_h   = pred_h[col_idx]
        col_cx  = pred_cx[col_idx]
        h_med   = float(np.median(col_h))
        if h_med < 2.0:
            continue

        # Column x bounds (± half median width for candidate search)
        cx_lo = float(col_cx.min()) - w_med * 0.5
        cx_hi = float(col_cx.max()) + w_med * 0.5

        # Candidates whose x-centre is within this column
        col_cand_mask = (cand_cx >= cx_lo) & (cand_cx <= cx_hi)
        if not col_cand_mask.any():
            continue
        cc_boxes = cand_arr[col_cand_mask]
        cc_sc    = cand_sc[col_cand_mask]
        cc_cy    = cand_cy[col_cand_mask]

        sort_y = np.argsort(col_cy)
        sorted_y = col_cy[sort_y]

        for k in range(len(sort_y) - 1):
            gap = sorted_y[k + 1] - sorted_y[k]
            if gap <= gap_factor * h_med:
                continue
            # Gap region
            y_lo = sorted_y[k]     + h_med * 0.3
            y_hi = sorted_y[k + 1] - h_med * 0.3
            n_miss = min(max(1, math.floor(gap / h_med) - 1), max_fill)
            if n_miss <= 0:
                continue
            # Find candidates in gap y-range, take top-scoring per slot
            in_gap = (cc_cy >= y_lo) & (cc_cy <= y_hi)
            if not in_gap.any():
                continue
            gap_boxes = cc_boxes[in_gap]
            gap_sc    = cc_sc[in_gap]
            gap_cy_   = cc_cy[in_gap]

            # Sort by score, keep up to n_miss
            top_order = np.argsort(gap_sc)[::-1][:n_miss]
            for ti in top_order:
                b = gap_boxes[ti].tolist()
                # Skip if already very close to an existing prediction
                existing_cx = np.abs(pred_cx - (0.5 * (b[0] + b[2])))
                existing_cy = np.abs(pred_cy - (0.5 * (b[1] + b[3])))
                if np.any((existing_cx < w_med * 0.5) & (existing_cy < h_med * 0.5)):
                    continue
                extra_boxes.append(b)
                extra_scores.append(float(gap_sc[ti]))

    if not extra_boxes:
        return pred_boxes, pred_scores

    # Merge and NMS
    all_merged_boxes  = pred_boxes + extra_boxes
    all_merged_scores = list(pred_scores) + extra_scores

    merged_arr = np.asarray(all_merged_boxes, dtype=np.float32)
    merged_sc  = np.asarray(all_merged_scores, dtype=np.float32)

    # Simple greedy NMS
    order_sc  = np.argsort(merged_sc)[::-1]
    keep_mask = np.ones(len(order_sc), dtype=bool)
    for i in range(len(order_sc)):
        if not keep_mask[order_sc[i]]:
            continue
        bi = merged_arr[order_sc[i]]
        for j in range(i + 1, len(order_sc)):
            if not keep_mask[order_sc[j]]:
                continue
            bj = merged_arr[order_sc[j]]
            ix1 = max(bi[0], bj[0]); iy1 = max(bi[1], bj[1])
            ix2 = min(bi[2], bj[2]); iy2 = min(bi[3], bj[3])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            ai = (bi[2] - bi[0]) * (bi[3] - bi[1])
            aj = (bj[2] - bj[0]) * (bj[3] - bj[1])
            union = ai + aj - inter
            if union > 0 and inter / union >= nms_iou:
                keep_mask[order_sc[j]] = False

    kept = [order_sc[i] for i in range(len(order_sc)) if keep_mask[order_sc[i]]]
    out_boxes  = [all_merged_boxes[i]  for i in kept]
    out_scores = [all_merged_scores[i] for i in kept]
    return out_boxes, out_scores


def compute_roi_box_loss(predicted_boxes, gt_boxes, gt_lengths=None, reduction='mean', iou_weight=0.0, use_x_only=False, coord_scale=None):
    """
    SmoothL1 + optional IoU loss for ROI-based attention.
    - Supervise only up to the number of available GT boxes per sample
    - Ignore padded / empty boxes (zero-area)
    - If use_x_only is True, compare only x1/x2 (width-wise) since encoder flattens height
    """
    B, T_dec, _ = predicted_boxes.shape

    total_l1 = torch.tensor(0.0, device=predicted_boxes.device)
    total_iou = torch.tensor(0.0, device=predicted_boxes.device)
    count = 0

    for i in range(B):
        pred_i = predicted_boxes[i]  # (T_dec, 4)
        gt_i = gt_boxes[i].to(predicted_boxes.device)  # (N, 4)

        # Filter out padded/zero boxes
        valid_mask = (gt_i[:, 2] - gt_i[:, 0] > 1e-3) & (gt_i[:, 3] - gt_i[:, 1] > 1e-3)
        gt_i = gt_i[valid_mask]

        if gt_i.numel() == 0:
            continue

        max_t = min(T_dec, gt_i.shape[0]) if gt_lengths is None else min(T_dec, int(gt_lengths[i].item()))
        pred_slice = pred_i[:max_t]
        gt_slice = gt_i[:max_t]

        match_pred_idx, match_gt_idx = _greedy_iou_match(pred_slice, gt_slice, use_x_only=use_x_only)
        if match_pred_idx.numel() == 0:
            continue
        pred_slice = pred_slice.index_select(0, match_pred_idx)
        gt_slice = gt_slice.index_select(0, match_gt_idx)
        matched_t = int(match_pred_idx.numel())

        if use_x_only:
            pred_slice = pred_slice[:, [0, 2]]
            gt_slice = gt_slice[:, [0, 2]]

        # Normalize coordinates to avoid oversized pixel-space losses dominating CE.
        if coord_scale is None:
            scale = torch.clamp(gt_slice.abs().max(), min=1.0)
        else:
            scale = torch.tensor(float(coord_scale), device=pred_slice.device, dtype=pred_slice.dtype)
        pred_norm = pred_slice / scale
        gt_norm = gt_slice / scale

        # Per-sample mean over coords, then weighted by supervised timesteps.
        l1 = F.smooth_l1_loss(pred_norm, gt_norm, reduction='mean')
        total_l1 = total_l1 + l1 * matched_t

        if (not use_x_only) and iou_weight > 0.0:
            ious = _bbox_iou(pred_slice, gt_slice)
            # IoU loss: 1 - IoU
            total_iou = total_iou + (1.0 - ious).sum()

        count += matched_t

    if count == 0:
        return torch.tensor(0.0, device=predicted_boxes.device)

    if reduction == 'mean':
        total = total_l1 / count
        if iou_weight > 0.0:
            total = total + iou_weight * (total_iou / count)
        return total

    total = total_l1
    if iou_weight > 0.0:
        total = total + iou_weight * total_iou
    return total


