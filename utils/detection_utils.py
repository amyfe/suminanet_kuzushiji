"""Helper functions for detection training."""
import torch
import torch.nn.functional as F


def build_detection_targets(boxes, labels, output_size, image_size, device, sigma=2.0):
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
    gt_cls = torch.full((B, H_out, W_out), -1, dtype=torch.long, device=device)
    
    for i in range(B):
        boxes_i = boxes[i].to(device) if boxes[i].numel() > 0 else torch.empty((0, 4), device=device)
        labels_i = labels[i].to(device) if labels[i].numel() > 0 else torch.empty((0,), dtype=torch.long, device=device)
        
        if boxes_i.numel() == 0:
            continue
            
        for box, label in zip(boxes_i, labels_i):
            x1, y1, x2, y2 = box.tolist()
            
            # Box center in output grid coordinates
            cx = (x1 + x2) / 2.0 / stride_w
            cy = (y1 + y2) / 2.0 / stride_h
            
            if cx < 0 or cy < 0 or cx >= W_out or cy >= H_out:
                continue
                
            ix = int(cx)
            iy = int(cy)
            
            # Box size in output grid units
            bw = max((x2 - x1) / stride_w, 1.0)
            bh = max((y2 - y1) / stride_h, 1.0)
            
            # Gaussian heatmap
            gaussian_sigma = max(1.0, sigma * (bw + bh) / 2.0)
            yy = torch.arange(0, H_out, device=device).view(H_out, 1).float()
            xx = torch.arange(0, W_out, device=device).view(1, W_out).float()
            g = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * gaussian_sigma ** 2))
            
            gt_heatmap[i, 0] = torch.max(gt_heatmap[i, 0], g)
            
            # Bbox targets (offset from grid cell + size)
            dx = cx - ix
            dy = cy - iy
            gt_bbox[i, :, iy, ix] = torch.tensor([dx, dy, bw, bh], device=device)
            
            # Class label
            gt_cls[i, iy, ix] = label.item()
    
    return gt_heatmap, gt_bbox, gt_cls


def compute_detection_losses(pred, gt_heatmap, gt_bbox, gt_cls, weights=(1.0, 1.0, 1.0)):
    """
    Compute detection losses.
    
    Args:
        pred: dict with 'heatmap', 'bbox', 'cls' from DetectorHead
        gt_heatmap: (B, 1, H, W)
        gt_bbox: (B, 4, H, W)
        gt_cls: (B, H, W)
        weights: (w_heat, w_bbox, w_cls) loss weights
        
    Returns:
        total_loss, (loss_heat, loss_bbox, loss_cls)
    """
    w_heat, w_bbox, w_cls = weights
    
    # Heatmap loss (MSE)
    loss_heat = F.mse_loss(pred['heatmap'], gt_heatmap) if 'heatmap' in pred else torch.tensor(0.0, device=gt_heatmap.device)
    
    # Bbox loss (L1, only at valid positions)
    if 'bbox' in pred:
        loss_bbox = F.l1_loss(pred['bbox'], gt_bbox)
    else:
        loss_bbox = torch.tensor(0.0, device=gt_bbox.device)
    
    # Classification loss (CE at valid positions)
    logits = pred['cls'].permute(0, 2, 3, 1).reshape(-1, pred['cls'].shape[1])
    labels_flat = gt_cls.reshape(-1)
    valid = labels_flat >= 0
    
    if valid.sum() > 0:
        loss_cls = F.cross_entropy(logits[valid], labels_flat[valid])
    else:
        loss_cls = torch.tensor(0.0, device=gt_cls.device)
    
    total_loss = w_heat * loss_heat + w_bbox * loss_bbox + w_cls * loss_cls
    
    return total_loss, (loss_heat, loss_bbox, loss_cls)


def compute_roi_box_loss(predicted_boxes, gt_boxes, reduction='mean'):
    """
    Compute box regression loss for ROI-based attention.
    
    Args:
        predicted_boxes: (B, T_dec, 4) predicted from attention
        gt_boxes: (B, N, 4) ground truth boxes
        reduction: 'mean' or 'sum'
        
    Returns:
        box_loss: L1 loss between predicted and GT boxes
    """
    # Match predicted boxes to GT boxes (simple nearest matching)
    # For each predicted box, find closest GT box
    B, T_dec, _ = predicted_boxes.shape
    
    total_loss = torch.tensor(0.0, device=predicted_boxes.device)
    count = 0
    
    for i in range(B):
        pred_i = predicted_boxes[i]  # (T_dec, 4)
        gt_i = gt_boxes[i]  # (N, 4)
        
        if gt_i.numel() == 0:
            continue
        
        # For each prediction, find closest GT
        for j in range(T_dec):
            if j >= len(gt_i):
                # No more GT boxes, use last one
                target = gt_i[-1]
            else:
                target = gt_i[j]
            
            loss_j = F.l1_loss(pred_i[j], target)
            total_loss = total_loss + loss_j
            count += 1
    
    if count == 0:
        return torch.tensor(0.0, device=predicted_boxes.device)
    
    if reduction == 'mean':
        return total_loss / count
    return total_loss
