"""Extract bounding boxes from DetectorHead predictions."""
import torch
import torch.nn.functional as F
from torchvision.ops import nms


def extract_boxes_from_heatmap(heatmap, bbox_reg, cls_logits, 
                               conf_thresh=0.3, nms_thresh=0.5, top_k=500,
                               image_size=(512, 512), output_size=None):
    """Extract boxes from DetectorHead predictions.
    
    Args:
        heatmap: (B, 1, H_out, W_out) heatmap predictions [0, 1]
        bbox_reg: (B, 4, H_out, W_out) bbox regression offsets
        cls_logits: (B, num_classes, H_out, W_out) class logits
        conf_thresh: Minimum confidence threshold
        nms_thresh: NMS IoU threshold
        top_k: Maximum boxes to keep per image
        image_size: (H_in, W_in) input image size
        output_size: (H_out, W_out) detection output size. If None, inferred from heatmap
        
    Returns:
        List of dicts with keys:
            - boxes: (N, 4) in [x1, y1, x2, y2] format
            - scores: (N,) confidence scores
            - classes: (N,) class indices
    """
    if output_size is None:
        output_size = (heatmap.size(2), heatmap.size(3))
    
    B = heatmap.size(0)
    H_out, W_out = output_size
    H_in, W_in = image_size
    stride_h = H_in / H_out
    stride_w = W_in / W_out
    
    results = []
    
    for b in range(B):
        # Find peaks (local maxima)
        peaks = _find_peaks(heatmap[b:b+1], threshold=conf_thresh)
        
        if len(peaks) == 0:
            results.append({
                'boxes': torch.empty((0, 4), dtype=torch.float32, device=heatmap.device),
                'scores': torch.empty((0,), dtype=torch.float32, device=heatmap.device),
                'classes': torch.empty((0,), dtype=torch.long, device=heatmap.device),
            })
            continue
        
        # Decode boxes from peaks + regression offsets
        boxes, scores, class_ids = _decode_boxes(
            peaks, heatmap[b], bbox_reg[b], cls_logits[b],
            stride_h, stride_w, H_in, W_in
        )
        
        if len(boxes) == 0:
            results.append({
                'boxes': torch.empty((0, 4), dtype=torch.float32, device=heatmap.device),
                'scores': torch.empty((0,), dtype=torch.float32, device=heatmap.device),
                'classes': torch.empty((0,), dtype=torch.long, device=heatmap.device),
            })
            continue
        
        # Per-class NMS
        keep_idxs = _nms_per_class(boxes, scores, class_ids, nms_thresh)
        
        if len(keep_idxs) == 0:
            results.append({
                'boxes': torch.empty((0, 4), dtype=torch.float32, device=heatmap.device),
                'scores': torch.empty((0,), dtype=torch.float32, device=heatmap.device),
                'classes': torch.empty((0,), dtype=torch.long, device=heatmap.device),
            })
            continue
        
        boxes = boxes[keep_idxs]
        scores = scores[keep_idxs]
        class_ids = class_ids[keep_idxs]
        
        # Keep top-k by score
        if len(boxes) > top_k:
            _, top_idxs = torch.topk(scores, top_k)
            boxes = boxes[top_idxs]
            scores = scores[top_idxs]
            class_ids = class_ids[top_idxs]
        
        results.append({
            'boxes': boxes,
            'scores': scores,
            'classes': class_ids,
        })
    
    return results


def _find_peaks(heatmap, threshold=0.3, pool_size=3):
    """Find local maxima (peaks) in heatmap.
    
    Args:
        heatmap: (1, 1, H, W) heatmap
        threshold: Minimum value threshold
        pool_size: Max pooling size
        
    Returns:
        peaks: List of (y, x, value) tuples
    """
    # Apply sigmoid to get probabilities if needed
    if heatmap.max() > 1.0:
        prob = torch.sigmoid(heatmap)
    else:
        prob = heatmap
    
    # Max pool to find peaks
    pooled = F.max_pool2d(prob, pool_size, stride=1, padding=pool_size//2)
    is_peak = (prob == pooled) & (prob >= threshold)
    
    # Get peak locations
    y_coords, x_coords = torch.where(is_peak[0, 0])
    values = prob[0, 0, y_coords, x_coords]
    
    peaks = []
    for y, x, v in zip(y_coords, x_coords, values):
        peaks.append((int(y), int(x), float(v)))
    
    return peaks


def _decode_boxes(peaks, heatmap, bbox_reg, cls_logits, 
                  stride_h, stride_w, H_in, W_in):
    """Decode boxes from peaks + regression offsets.
    
    Args:
        peaks: List of (y, x, value) tuples
        heatmap: (1, H_out, W_out)
        bbox_reg: (4, H_out, W_out) [dy1, dx1, dy2, dx2]
        cls_logits: (num_classes, H_out, W_out)
        stride_h, stride_w: Stride values
        H_in, W_in: Input image size
        
    Returns:
        boxes: (N, 4) in [x1, y1, x2, y2]
        scores: (N,)
        class_ids: (N,)
    """
    if len(peaks) == 0:
        return torch.empty((0, 4)), torch.empty((0,)), torch.empty((0,), dtype=torch.long)
    
    boxes = []
    scores = []
    class_ids = []
    
    for y, x, conf in peaks:
        # Get bbox regression for this peak
        dy1, dx1, dy2, dx2 = bbox_reg[:, y, x]
        
        # Convert to image coordinates
        cx = x * stride_w
        cy = y * stride_h
        
        # Apply regression
        x1 = cx + dx1 * stride_w
        y1 = cy + dy1 * stride_h
        x2 = cx + dx2 * stride_w
        y2 = cy + dy2 * stride_h
        
        # Clip to image bounds
        x1 = torch.clamp(x1, 0, W_in)
        y1 = torch.clamp(y1, 0, H_in)
        x2 = torch.clamp(x2, 0, W_in)
        y2 = torch.clamp(y2, 0, H_in)
        
        # Get predicted class
        cls_pred = cls_logits[:, y, x]
        class_id = cls_pred.argmax()
        
        boxes.append([x1, y1, x2, y2])
        scores.append(conf)
        class_ids.append(class_id.item())
    
    boxes = torch.stack(boxes) if boxes else torch.empty((0, 4))
    scores = torch.tensor(scores, dtype=torch.float32, device=heatmap.device)
    class_ids = torch.tensor(class_ids, dtype=torch.long, device=heatmap.device)
    
    return boxes.to(heatmap.device), scores, class_ids


def _nms_per_class(boxes, scores, class_ids, nms_thresh):
    """Apply NMS per class to remove duplicate detections.
    
    Args:
        boxes: (N, 4)
        scores: (N,)
        class_ids: (N,)
        nms_thresh: IoU threshold
        
    Returns:
        keep_idxs: Indices to keep
    """
    if len(boxes) == 0:
        return torch.empty((0,), dtype=torch.long)
    
    keep = []
    unique_classes = class_ids.unique()
    
    for class_id in unique_classes:
        mask = class_ids == class_id
        class_boxes = boxes[mask]
        class_scores = scores[mask]
        class_idxs = torch.where(mask)[0]
        
        # NMS for this class
        nms_keep = nms(class_boxes, class_scores, nms_thresh)
        keep.extend(class_idxs[nms_keep].tolist())
    
    return torch.tensor(sorted(keep), dtype=torch.long)
