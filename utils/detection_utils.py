"""Helper functions for detection training."""
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
    bbox_radius=0,          # 0 = only center cell (most stable)
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

            gaussian_sigma = min(2.0, max(0.5, sigma * min(bw, bh) * 0.25))
            yy = torch.arange(0, H_out, device=device).view(H_out, 1).float()
            xx = torch.arange(0, W_out, device=device).view(1, W_out).float()
            g = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * gaussian_sigma ** 2))
            gt_heatmap[i, 0] = torch.maximum(gt_heatmap[i, 0], g)
            
            # Bbox targets (offset from grid cell + size)
            x0 = max(0, ix - bbox_radius)
            x1i = min(W_out - 1, ix + bbox_radius)
            y0 = max(0, iy - bbox_radius)
            y1i = min(H_out - 1, iy + bbox_radius)

            # bbox targets ONLY at center
            gt_bbox[i,0,iy,ix] = dx
            gt_bbox[i,1,iy,ix] = dy
            gt_bbox[i,2,iy,ix] = bw
            gt_bbox[i,3,iy,ix] = bh

            # class label only at center cell (keeps it sparse; even if class head disabled)
            gt_cls[i, iy, ix] = int(label.item())
            gt_bbox_mask[i,iy,ix] = True
    gt_heatmap = gt_heatmap.clamp(min=heatmap_min, max=1.0)
    return gt_heatmap, gt_bbox, gt_bbox_mask, gt_cls


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


