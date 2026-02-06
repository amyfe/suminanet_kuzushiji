"""
Validate Stage 1 (DetectorHead) - visualize box predictions and compute metrics.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

from config import (
    DATA_DIR, DEVICE, IMAGE_SIZE, NUM_WORKERS, CHECKPOINT_DIR
)
from model.kuronet import UNet, DetectorHead
from utils import KuzushijiDataset
from utils.vocab import VocabManager

def non_max_suppression(boxes, scores, iou_threshold=0.5):
    """Simple NMS to filter overlapping boxes."""
    if len(boxes) == 0:
        return []
    
    # Convert to numpy for easier manipulation
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    # Sort by scores
    indices = np.argsort(scores)[::-1]
    
    keep = []
    while len(indices) > 0:
        i = indices[0]
        keep.append(i)
        
        if len(indices) == 1:
            break
        
        # Compute IoU with remaining boxes
        ious = compute_iou_batch(boxes[i], boxes[indices[1:]])
        
        # Keep only boxes with IoU below threshold
        indices = indices[1:][ious < iou_threshold]
    
    return keep

def compute_iou_batch(box, boxes):
    """Compute IoU between one box and multiple boxes."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    union = box_area + boxes_area - intersection
    
    return intersection / (union + 1e-6)

def extract_boxes_from_heatmap(heatmap, bbox_reg, cls_logits, confidence_thresh=0.5,
                                output_size=(64, 64), image_size=(512, 512), top_k=200):
    """Extract boxes using local-maximum peaks to avoid covering the whole image."""
    import torch.nn.functional as F

    # heatmap: (1,1,H,W) sigmoid'd; bbox_reg: (1,4,H,W); cls_logits: (1,C,H,W) or None
    hm = heatmap[0, 0]  # (H, W)
    bbox = bbox_reg[0]  # (4, H, W)
    cls = cls_logits[0] if cls_logits is not None else None  # (C, H, W) or None

    # Find local maxima with 3x3 max pool
    pooled = F.max_pool2d(hm.unsqueeze(0).unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze()
    peak_mask = (hm == pooled) & (hm > confidence_thresh)
    peak_indices = peak_mask.nonzero(as_tuple=False)

    # Sort by score and keep top_k
    if peak_indices.numel() == 0:
        return [], [], []
    scores = hm[peak_mask]
    sorted_scores, order = torch.sort(scores, descending=True)
    if top_k is not None and len(order) > top_k:
        order = order[:top_k]
    ys = peak_indices[order, 0].cpu().numpy()
    xs = peak_indices[order, 1].cpu().numpy()
    scores_np = sorted_scores[: len(order)].cpu().numpy().tolist()

    H, W = hm.shape
    scale_h = image_size[0] / output_size[0]
    scale_w = image_size[1] / output_size[1]

    boxes = []
    classes = []
    scores_out = []
    cls_np = cls.cpu().numpy() if cls is not None else None
    bbox_np = bbox.cpu().numpy()

    for y, x, sc in zip(ys, xs, scores_np):
        dx1, dy1, dx2, dy2 = bbox_np[:, y, x]
        cx = (x + 0.5) * scale_w
        cy = (y + 0.5) * scale_h

        x1 = cx + dx1 * scale_w
        y1 = cy + dy1 * scale_h
        x2 = cx + dx2 * scale_w
        y2 = cy + dy2 * scale_h

        x1 = float(np.clip(x1, 0, image_size[1]))
        y1 = float(np.clip(y1, 0, image_size[0]))
        x2 = float(np.clip(x2, 0, image_size[1]))
        y2 = float(np.clip(y2, 0, image_size[0]))

        pred_cls = int(np.argmax(cls_np[:, y, x])) if cls_np is not None else 0

        boxes.append([x1, y1, x2, y2])
        scores_out.append(float(sc))
        classes.append(pred_cls)

    # Apply NMS to reduce duplicates
    if len(boxes) > 0:
        keep_indices = non_max_suppression(boxes, scores_out, iou_threshold=0.5)
        boxes = [boxes[i] for i in keep_indices]
        scores_out = [scores_out[i] for i in keep_indices]
        classes = [classes[i] for i in keep_indices]

    return boxes, scores_out, classes

def denormalize_image(image_tensor):
    """Denormalize image tensor to uint8 BGR format."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = image_tensor.cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
    img = (img * std + mean) * 255
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def visualize_boxes_only(image_tensor, gt_boxes, pred_boxes, before_path, after_path):
    """
    Create before/after visualizations with boxes only (no text labels).
    
    Args:
        image_tensor: (3, H, W) normalized image tensor
        gt_boxes: List of ground truth boxes [x1, y1, x2, y2]
        pred_boxes: List of predicted boxes [x1, y1, x2, y2]
        before_path: Path to save original image
        after_path: Path to save image with detection boxes
    """
    img = denormalize_image(image_tensor)
    
    # Save original (before)
    cv2.imwrite(str(before_path), img.copy())
    
    # Draw boxes on after image
    img_after = img.copy()
    
    # Draw ground truth boxes (green) - thinner
    for box in gt_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_after, (x1, y1), (x2, y2), (0, 255, 0), 1)
    
    # Draw predicted boxes (red) - slightly thicker
    for box in pred_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_after, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    cv2.imwrite(str(after_path), img_after)

def visualize_predictions(image_tensor, gt_boxes, pred_boxes, pred_scores, 
                          output_path, gt_labels=None, pred_classes=None, vocab=None):
    """
    Visualize ground truth and predicted boxes with labels.
    
    Args:
        image_tensor: (3, H, W) normalized image tensor
        gt_boxes: List of ground truth boxes [x1, y1, x2, y2]
        pred_boxes: List of predicted boxes [x1, y1, x2, y2]
        pred_scores: List of confidence scores
        output_path: Where to save visualization
        gt_labels: Optional ground truth class labels
        pred_classes: Optional predicted class IDs
        vocab: Optional VocabManager for label decoding
    """
    img = denormalize_image(image_tensor)
    
    # Draw ground truth boxes (green)
    for i, box in enumerate(gt_boxes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if gt_labels is not None and vocab is not None:
            label_id = gt_labels[i].item() if torch.is_tensor(gt_labels[i]) else gt_labels[i]
            label_text = vocab.id2char.get(label_id, '?')
            cv2.putText(img, f"GT:{label_text}", (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    # Draw predicted boxes (red)
    for i, box in enumerate(pred_boxes):
        x1, y1, x2, y2 = map(int, box)
        score = pred_scores[i]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        label_text = f"{score:.2f}"
        if pred_classes is not None and vocab is not None:
            pred_cls = pred_classes[i]
            char = vocab.id2char.get(pred_cls, '?')
            label_text = f"{char}:{score:.2f}"
        
        cv2.putText(img, label_text, (x1, y2+15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    cv2.imwrite(str(output_path), img)

def compute_detection_metrics(gt_boxes_list, pred_boxes_list, iou_threshold=0.5):
    """
    Compute precision, recall, F1 for detection.
    
    Args:
        gt_boxes_list: List of lists of ground truth boxes per image
        pred_boxes_list: List of lists of predicted boxes per image
        iou_threshold: IoU threshold for considering a match
    
    Returns:
        dict with precision, recall, f1
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for gt_boxes, pred_boxes in zip(gt_boxes_list, pred_boxes_list):
        if len(gt_boxes) == 0 and len(pred_boxes) == 0:
            continue
        
        if len(pred_boxes) == 0:
            total_fn += len(gt_boxes)
            continue
        
        if len(gt_boxes) == 0:
            total_fp += len(pred_boxes)
            continue
        
        # Compute IoU matrix
        ious = np.zeros((len(pred_boxes), len(gt_boxes)))
        for i, pred_box in enumerate(pred_boxes):
            ious[i] = compute_iou_batch(np.array(pred_box), np.array(gt_boxes))
        
        # Match predictions to ground truth (greedy)
        matched_gt = set()
        tp = 0
        fp = 0
        
        for i in range(len(pred_boxes)):
            best_iou = 0
            best_gt = -1
            for j in range(len(gt_boxes)):
                if j not in matched_gt and ious[i, j] > best_iou:
                    best_iou = ious[i, j]
                    best_gt = j
            
            if best_iou >= iou_threshold:
                tp += 1
                matched_gt.add(best_gt)
            else:
                fp += 1
        
        fn = len(gt_boxes) - len(matched_gt)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn
    }

def validate_stage1(checkpoint_path, num_samples=50, confidence_thresh=0.3, 
                    output_dir=None, split='val'):
    """
    Validate Stage 1 detector and visualize predictions.
    
    Args:
        checkpoint_path: Path to detector checkpoint
        num_samples: Number of samples to visualize
        confidence_thresh: Detection confidence threshold
        output_dir: Where to save visualizations
        split: 'train' or 'val'
    """
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = CHECKPOINT_DIR / "stage1_validation" / timestamp
    else:
        output_dir = Path(output_dir) / timestamp
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"🔍 Validating Stage 1 Detector")
    print(f"   Checkpoint: {checkpoint_path}")
    print(f"   Output: {output_dir}")
    
    # Load vocabulary
    ann_files = sorted(list((Path(DATA_DIR) / "annotations").glob("*.json")))
    vocab = VocabManager.from_annotations(ann_files)
    
    # Load dataset
    dataset = KuzushijiDataset(
        Path(DATA_DIR), 
        vocab=vocab, 
        use_sequences=False,  # Only need boxes
        resize=IMAGE_SIZE,
        split=split
    )
    
    # Single sample loader (no batching for visualization)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # Load model
    unet = UNet(in_channels=3, base_features=32).to(DEVICE)
    detector = DetectorHead(in_ch=32, num_classes=vocab.vocab_size).to(DEVICE)
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    missing, unexpected = unet.load_state_dict(checkpoint['unet_state_dict'], strict=False)
    if missing or unexpected:
        print(f"⚠️ UNet state_dict mismatches. Missing: {missing}, Unexpected: {unexpected}")
    missing_d, unexpected_d = detector.load_state_dict(checkpoint['detector_state_dict'], strict=False)
    if missing_d or unexpected_d:
        print(f"⚠️ Detector state_dict mismatches. Missing: {missing_d}, Unexpected: {unexpected_d}")
    
    unet.eval()
    detector.eval()
    
    print(f"✅ Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
    
    # Collect predictions for metrics
    all_gt_boxes = []
    all_pred_boxes = []
    
    # Visualize samples
    vis_count = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Validating")):
            if vis_count >= num_samples:
                break
            
            images = batch['image'].to(DEVICE)
            gt_boxes = batch['boxes'][0]  # First (and only) item in batch
            gt_labels = batch.get('labels', [None])[0]
            
            # Skip if no boxes
            if len(gt_boxes) == 0:
                continue
            
            # Forward pass
            features = unet(images)
            outputs = detector(features)
            heatmap = outputs.get('heatmap')
            bbox_reg = outputs.get('bbox')
            cls_logits = outputs.get('cls')
            
            # Extract predicted boxes
            B, _, H_feat, W_feat = features.shape
            heatmap = torch.sigmoid(heatmap)
            # Handle None cls_logits (when class head is disabled)
            if cls_logits is not None:
                cls_logits = torch.softmax(cls_logits, dim=1)
    
            pred_boxes, pred_scores, pred_classes = extract_boxes_from_heatmap(
                heatmap, bbox_reg, cls_logits,
                confidence_thresh=0.75,
                output_size=(H_feat, W_feat),
                image_size=IMAGE_SIZE,
                top_k=100,
            )
            
            # Store for metrics
            all_gt_boxes.append(gt_boxes.cpu().numpy().tolist())
            all_pred_boxes.append(pred_boxes)
            
            # Create before/after visualizations (boxes only)
            before_path = output_dir / f"sample_{vis_count:04d}_before.png"
            after_path = output_dir / f"sample_{vis_count:04d}_after.png"
            visualize_boxes_only(
                images[0],
                gt_boxes.cpu().numpy(),
                pred_boxes,
                before_path,
                after_path
            )
            
            # Also create detailed visualization with labels
            detail_path = output_dir / f"sample_{vis_count:04d}_detail.png"
            visualize_predictions(
                images[0],
                gt_boxes.cpu().numpy(),
                pred_boxes,
                pred_scores,
                detail_path,
                gt_labels=gt_labels,
                pred_classes=pred_classes,
                vocab=vocab
            )
            
            vis_count += 1
    
    # Compute metrics
    print(f"\n📊 Computing detection metrics...")
    metrics = compute_detection_metrics(all_gt_boxes, all_pred_boxes, iou_threshold=0.5)
    
    print(f"\n{'='*60}")
    print(f"STAGE 1 VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"  Samples evaluated: {len(all_gt_boxes)}")
    print(f"  Confidence threshold: {confidence_thresh}")
    print(f"  IoU threshold: 0.5")
    print(f"\n  Precision: {metrics['precision']:.3f}")
    print(f"  Recall:    {metrics['recall']:.3f}")
    print(f"  F1 Score:  {metrics['f1']:.3f}")
    print(f"\n  True Positives:  {metrics['tp']}")
    print(f"  False Positives: {metrics['fp']}")
    print(f"  False Negatives: {metrics['fn']}")
    print(f"Rest: {metrics}")
    print(f"{'='*60}")
    
    # Save metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✅ Metrics saved to: {metrics_path}")
    print(f"✅ Visualizations saved to: {output_dir}")
    
    return metrics

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate Stage 1 Detector")
    parser.add_argument("--checkpoint", type=str, required=True, 
                       help="Path to detector checkpoint")
    parser.add_argument("--num_samples", type=int, default=50,
                       help="Number of samples to visualize")
    parser.add_argument("--confidence", type=float, default=0.3,
                       help="Detection confidence threshold")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for visualizations")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"],
                       help="Dataset split to use")
    
    args = parser.parse_args()
    
    validate_stage1(
        checkpoint_path=args.checkpoint,
        num_samples=args.num_samples,
        confidence_thresh=args.confidence,
        output_dir=args.output_dir,
        split=args.split
    )
