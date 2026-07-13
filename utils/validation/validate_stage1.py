"""
Validate Stage 1 (DetectorHead) - visualize box predictions and compute metrics.
Fixes:
- bbox decode matches training targets (dx,dy,bw,bh)
- no double-sigmoid on heatmap
- confidence thresh wired correctly
- full-val metrics + extra logs

New:
- FN-highlighted visualization (green=TP GT, red=FN GT, blue=pred)
- Worst-N images by FN count saved instead of first-N
- PR curve sweep over confidence thresholds
- FN vs TP box-size analysis
"""
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import json
import os
from datetime import datetime
import torch.nn.functional as F
import matplotlib.pyplot as plt

from config import (DATA_DIR, DEVICE, IMAGE_SIZE, CHECKPOINT_DIR,
                    DET_SCORE_THRESH, DET_TOP_K, DET_NMS_IOU, DET_MIN_BOX_SIZE,
                    STAGE1_DENSITY_GRID, STAGE1_DENSITY_FACTOR, STAGE1_AVG_GT_PER_IMAGE,
                    BACKBONE_BASE_FEATURES, BACKBONE_TYPE, STAGE1_FOCAL_POS_THRESHOLD)
from model.kuronet import DetectorHead, build_backbone
from utils import KuzushijiDataset
from utils.detection_utils import spatial_density_filter, _extract_all_peaks, fill_detection_gaps
from utils.vocab import VocabManager


# -------------------------
# IoU + NMS
# -------------------------
def compute_iou_batch(box, boxes):
    box = np.asarray(box, dtype=np.float32)              # (4,)
    boxes = np.asarray(boxes, dtype=np.float32)          # (N,4)

    if boxes.ndim == 1:
        boxes = boxes.reshape(1, 4)
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - inter
    return inter / (union + 1e-6)


def non_max_suppression(boxes, scores, iou_threshold=0.5):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    scores = np.array(scores)
    idx = np.argsort(scores)[::-1]

    keep = []
    while len(idx) > 0:
        i = idx[0]
        keep.append(i)
        if len(idx) == 1:
            break
        ious = compute_iou_batch(boxes[i], boxes[idx[1:]])
        idx = idx[1:][ious < iou_threshold]
    return keep


# -------------------------
# Decode: (dx,dy,bw,bh) -> (x1,y1,x2,y2)
# -------------------------
def extract_boxes_from_heatmap(
    heatmap_probs,      # (1,1,H,W) already sigmoid
    bbox_reg,           # (1,4,H,W) = (dx,dy,bw,bh)
    confidence_thresh=STAGE1_FOCAL_POS_THRESHOLD,
    output_size=(64, 64),
    image_size=IMAGE_SIZE,
    top_k=200,
    nms_iou=DET_NMS_IOU,
    min_box_size=DET_MIN_BOX_SIZE,
    debug=False,
    avg_gt_per_image=STAGE1_AVG_GT_PER_IMAGE,
):
    hm = heatmap_probs[0, 0]
    bbox = bbox_reg[0]      # (4,H,W)

    # local maxima (3x3)
    pooled = F.max_pool2d(hm[None, None], kernel_size=3, stride=1, padding=1).squeeze()
    peak_mask = (hm == pooled) & (hm > confidence_thresh)
    peak_idx = peak_mask.nonzero(as_tuple=False)

    if peak_idx.numel() == 0:
        return [], [], []

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

    boxes = []
    scores_out = []
    skipped_invalid = 0
    for y, x, sc in zip(ys, xs, scores_np):
        dx, dy, bw, bh = bbox_np[:, y, x]

        cx = (x + dx) * stride_w
        cy = (y + dy) * stride_h
        w  = max(bw * stride_w, min_box_size)
        h  = max(bh * stride_h, min_box_size)

        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        x1 = cx - 0.5 * w

        x1 = float(np.clip(x1, 0, W_img))
        y1 = float(np.clip(y1, 0, H_img))
        x2 = float(np.clip(x2, 0, W_img))
        y2 = float(np.clip(y2, 0, H_img))

        if x2 <= x1 or y2 <= y1:
            skipped_invalid += 1
            continue

        boxes.append([x1, y1, x2, y2])
        scores_out.append(float(sc))

    if debug:
        bw_vals = bbox_np[2]
        bh_vals = bbox_np[3]
        print(f"  → peaks found: {len(ys)} | kept after valid decode: {len(scores_out)} | skipped: {skipped_invalid}")
        print(f"    heatmap stats min/max/mean: {hm.min().item():.4f} / {hm.max().item():.4f} / {hm.mean().item():.4f}")
        print(f"    bbox bw stats min/max/mean: {bw_vals.min():.4f} / {bw_vals.max():.4f} / {bw_vals.mean():.4f}")
        print(f"    bbox bh stats min/max/mean: {bh_vals.min():.4f} / {bh_vals.max():.4f} / {bh_vals.mean():.4f}")
        print(f"    stride_h={stride_h:.2f}, stride_w={stride_w:.2f}")

    if len(boxes) == 0:
        return [], [], []

    keep = non_max_suppression(boxes, scores_out, iou_threshold=nms_iou)
    boxes = [boxes[i] for i in keep]
    scores_out = [scores_out[i] for i in keep]

    boxes, scores_out = spatial_density_filter(
        boxes, scores_out, image_size,
        grid_size=STAGE1_DENSITY_GRID,
        density_factor=STAGE1_DENSITY_FACTOR,
        avg_gt_per_image=avg_gt_per_image,
    )

    classes = [0] * len(boxes)
    return boxes, scores_out, classes



def _filter_peaks(all_boxes, all_scores, confidence_thresh, nms_iou,
                  avg_gt_per_image=STAGE1_AVG_GT_PER_IMAGE):
    """Apply confidence threshold + NMS + density filter to pre-extracted peak candidates."""
    paired = [(b, s) for b, s in zip(all_boxes, all_scores) if s >= confidence_thresh]
    if not paired:
        return []
    boxes, scores = zip(*paired)
    keep = non_max_suppression(list(boxes), list(scores), iou_threshold=nms_iou)
    if not keep:
        return []
    kept_boxes = [boxes[i] for i in keep]
    kept_scores = [scores[i] for i in keep]
    kept_boxes, _ = spatial_density_filter(
        kept_boxes, kept_scores, IMAGE_SIZE,
        grid_size=STAGE1_DENSITY_GRID,
        density_factor=STAGE1_DENSITY_FACTOR,
        avg_gt_per_image=avg_gt_per_image,
    )
    return kept_boxes


# --------------------------------------------------
# Matching / metrics
# --------------------------------------------------
def match_predictions_to_gt(gt_boxes, pred_boxes, iou_threshold=0.5):
    gt_boxes = np.asarray(gt_boxes, dtype=np.float32)
    pred_boxes = np.asarray(pred_boxes, dtype=np.float32)

    if gt_boxes.size == 0:
        gt_boxes = gt_boxes.reshape(0, 4)
    elif gt_boxes.ndim == 1:
        gt_boxes = gt_boxes.reshape(1, 4)

    if pred_boxes.size == 0:
        pred_boxes = pred_boxes.reshape(0, 4)
    elif pred_boxes.ndim == 1:
        pred_boxes = pred_boxes.reshape(1, 4)

    matched_gt = set()
    tp = 0
    fp = 0
    matched_ious = []

    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes), []

    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0, []

    ious = np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.float32)
    for i, pred_box in enumerate(pred_boxes):
        ious[i] = compute_iou_batch(pred_box, gt_boxes)

    for i in range(len(pred_boxes)):
        best_iou = 0.0
        best_gt = -1
        for j in range(len(gt_boxes)):
            if j not in matched_gt and ious[i, j] > best_iou:
                best_iou = float(ious[i, j])
                best_gt = j

        if best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_gt)
            matched_ious.append(best_iou)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn, matched_ious


def _get_fn_indices(gt_boxes, pred_boxes, iou_threshold=0.5):
    """Return the set of GT box indices that have no matching prediction (false negatives).

    Uses the same greedy matching as match_predictions_to_gt so the highlighted
    FN boxes are consistent with the reported TP/FP/FN counts.
    """
    if not gt_boxes:
        return set()
    if not pred_boxes:
        return set(range(len(gt_boxes)))

    gt_arr = np.asarray(gt_boxes, dtype=np.float32)
    pred_arr = np.asarray(pred_boxes, dtype=np.float32)
    if gt_arr.ndim == 1:
        gt_arr = gt_arr.reshape(1, 4)
    if pred_arr.ndim == 1:
        pred_arr = pred_arr.reshape(1, 4)

    # Pre-compute full IoU matrix
    ious = np.zeros((len(pred_arr), len(gt_arr)), dtype=np.float32)
    for i, pred_box in enumerate(pred_arr):
        ious[i] = compute_iou_batch(pred_box, gt_arr)

    # Greedy: for each prediction, match to best *unmatched* GT — identical to
    # match_predictions_to_gt so visualized FNs always agree with the metrics
    matched_gt = set()
    for i in range(len(pred_arr)):
        best_iou, best_j = 0.0, -1
        for j in range(len(gt_arr)):
            if j not in matched_gt and ious[i, j] > best_iou:
                best_iou = ious[i, j]
                best_j = j
        if best_iou >= iou_threshold:
            matched_gt.add(best_j)

    return set(range(len(gt_boxes))) - matched_gt


def compute_detection_metrics(gt_boxes_list, pred_boxes_list, iou_threshold=0.5):
    total_tp = total_fp = total_fn = 0
    matched_ious = []

    for gt_boxes, pred_boxes in zip(gt_boxes_list, pred_boxes_list):
        tp, fp, fn, ious = match_predictions_to_gt(gt_boxes, pred_boxes, iou_threshold=iou_threshold)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        matched_ious.extend(ious)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "tp": total_tp, "fp": total_fp, "fn": total_fn, "num_matches": len(matched_ious), "mean_iou_tp": mean_iou}


def denormalize_image(image_tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = (img * std + mean) * 255
    img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


# -------------------------
# Visualisation functions
# -------------------------

def visualize_boxes_only(image_bgr, gt_boxes, pred_boxes, out_path):
    img = image_bgr.copy()
    for box in gt_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
    for box in pred_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imwrite(str(out_path), img)


def visualize_centers_only(image_bgr, gt_boxes, pred_boxes, out_path):
    img = image_bgr.copy()
    for box in gt_boxes:
        x1, y1, x2, y2 = box
        cx = int(0.5 * (x1 + x2))
        cy = int(0.5 * (y1 + y2))
        cv2.circle(img, (cx, cy), 2, (0, 255, 0), -1)
    for box in pred_boxes:
        x1, y1, x2, y2 = box
        cx = int(0.5 * (x1 + x2))
        cy = int(0.5 * (y1 + y2))
        cv2.circle(img, (cx, cy), 2, (0, 0, 255), -1)
    cv2.imwrite(str(out_path), img)


def visualize_fn_highlighted(image_bgr, gt_boxes, pred_boxes, out_path, iou_threshold=0.5,
                             fn_count=None, fn_rate=None):
    """Color-coded: green=TP GT, red=FN GT (thick), blue=all predictions."""
    img = image_bgr.copy()
    fn_idx = _get_fn_indices(gt_boxes, pred_boxes, iou_threshold)
    for i, box in enumerate(gt_boxes):
        x1, y1, x2, y2 = map(int, box)
        if i in fn_idx:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 220), 2)   # red   = FN GT
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 1)   # green = TP GT
    for box in pred_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (220, 100, 0), 1)     # blue  = pred
    if fn_count is not None and fn_rate is not None:
        label = f"FN={fn_count} ({fn_rate*100:.1f}%)  green=TP  RED=FN  blue=pred"
        cv2.putText(img, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
        cv2.putText(img, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.imwrite(str(out_path), img)


def visualize_fn_only(image_bgr, gt_boxes, pred_boxes, out_path, iou_threshold=0.5,
                      fn_count=None, fn_rate=None):
    """Show ONLY the missed GT boxes (FNs) — no TP or prediction clutter."""
    img = image_bgr.copy()
    fn_idx = _get_fn_indices(gt_boxes, pred_boxes, iou_threshold)
    for i in fn_idx:
        x1, y1, x2, y2 = map(int, gt_boxes[i])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)       # red = missed
        # small dot at centre so tiny boxes are still visible
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(img, (cx, cy), 3, (0, 0, 255), -1)
    if fn_count is not None and fn_rate is not None:
        label = f"FN={fn_count} ({fn_rate*100:.1f}%)  red=MISSED GT only"
        cv2.putText(img, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
        cv2.putText(img, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.imwrite(str(out_path), img)


# -------------------------
# Analysis plots
# -------------------------

def plot_pr_curve(pr_results, out_path):
    thresholds = [r[0] for r in pr_results]
    precisions = [r[1] for r in pr_results]
    recalls    = [r[2] for r in pr_results]
    f1s        = [r[3] for r in pr_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(recalls, precisions, "b-o", markersize=4)
    for i, (t, p, r, _) in enumerate(pr_results):
        if i % 3 == 0:
            ax1.annotate(f"{t:.2f}", (r, p), textcoords="offset points",
                         xytext=(4, 4), fontsize=7)
    ax1.set_xlabel("Recall")
    ax1.set_ylabel("Precision")
    ax1.set_title("Precision-Recall Curve")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    ax2.plot(thresholds, recalls,    "g-o", markersize=4, label="Recall")
    ax2.plot(thresholds, precisions, "b-o", markersize=4, label="Precision")
    ax2.plot(thresholds, f1s,        "r-o", markersize=4, label="F1")
    best = int(np.argmax(f1s))
    ax2.axvline(thresholds[best], color="r", linestyle="--", alpha=0.5,
                label=f"Best F1={f1s[best]:.3f} @ thr={thresholds[best]:.2f}")
    ax2.set_xlabel("Confidence Threshold")
    ax2.set_ylabel("Score")
    ax2.set_title("Metrics vs Confidence Threshold")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"PR curve saved: {out_path}")


def plot_fn_size_analysis(fn_sizes, tp_sizes, out_path):
    fn_arr = np.array(fn_sizes) if fn_sizes else np.zeros(0)
    tp_arr = np.array(tp_sizes) if tp_sizes else np.zeros(0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    if len(fn_arr):
        ax1.hist(fn_arr, bins=40, color="red",   alpha=0.6, label=f"FN (n={len(fn_arr)})")
    if len(tp_arr):
        ax1.hist(tp_arr, bins=40, color="green", alpha=0.6, label=f"TP (n={len(tp_arr)})")
    ax1.set_xlabel("Box area (px²)")
    ax1.set_ylabel("Count")
    ax1.set_title("Box Area: FN vs TP GT boxes")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    fn_side = np.sqrt(fn_arr) if len(fn_arr) else np.zeros(0)
    tp_side = np.sqrt(tp_arr) if len(tp_arr) else np.zeros(0)
    if len(fn_side):
        ax2.hist(fn_side, bins=40, color="red",   alpha=0.6, label="FN")
    if len(tp_side):
        ax2.hist(tp_side, bins=40, color="green", alpha=0.6, label="TP")
    ax2.set_xlabel("Box side ≈ √area (px)")
    ax2.set_ylabel("Count")
    ax2.set_title("Box Side Length: FN vs TP GT boxes")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    title_parts = []
    if len(fn_arr):
        title_parts.append(f"FN median area={np.median(fn_arr):.1f}px²")
    if len(tp_arr):
        title_parts.append(f"TP median area={np.median(tp_arr):.1f}px²")
    if title_parts:
        fig.suptitle("  |  ".join(title_parts), fontsize=11)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"FN size analysis saved: {out_path}")


# -------------------------
# Main validation function
# -------------------------

def validate_stage1(
    checkpoint_path,
    confidence_thresh=DET_SCORE_THRESH,
    split="val",
    num_samples=None,
    top_k=DET_TOP_K,
    nms_iou=DET_NMS_IOU,
    iou_threshold=0.5,
    min_box_size=DET_MIN_BOX_SIZE,
    job_id=None,
    worst_n=30,
    gap_fill=True,
    return_detailed=False,
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    resolved_job_id = job_id or os.environ.get("SLURM_JOB_ID") or os.environ.get("JOB_ID")
    run_tag = f"{timestamp}_job{resolved_job_id}" if resolved_job_id else timestamp
    out_dir = Path(CHECKPOINT_DIR) / "stage1_validation" / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🔍 Stage1 validation")
    print(f"  ckpt: {checkpoint_path}")
    print(f"  split: {split}")
    print(f"  job_id: {resolved_job_id if resolved_job_id else 'n/a'}")
    print(f"  conf: {confidence_thresh} | top_k: {top_k} | nms_iou: {nms_iou} | iou_thr: {iou_threshold}")
    print(f"  min_box_size: {min_box_size} | gap_fill: {gap_fill}")
    print(f"  out: {out_dir}")

    ann_files = sorted(list((Path(DATA_DIR) / "annotations").glob("*.json")))
    vocab = VocabManager.from_annotations(ann_files)

    dataset = KuzushijiDataset(
        Path(DATA_DIR),
        vocab=vocab,
        use_sequences=False,
        resize=IMAGE_SIZE,
        split=split,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    backbone_type = ckpt.get("backbone_type", BACKBONE_TYPE)
    backbone = build_backbone(backbone_type, BACKBONE_BASE_FEATURES, pretrained=False).to(DEVICE)
    detector = DetectorHead(in_ch=BACKBONE_BASE_FEATURES, num_classes=vocab.vocab_size, predict_classes=False).to(DEVICE)

    state_key = "backbone_state_dict" if "backbone_state_dict" in ckpt else "unet_state_dict"
    backbone.load_state_dict(ckpt[state_key], strict=True)
    detector.load_state_dict(ckpt["detector_state_dict"], strict=True)

    backbone.eval()
    detector.eval()

    all_gt_boxes = []
    all_pred_boxes = []
    all_raw_peaks = []   # list of (boxes, scores) per image — for PR curve sweep
    all_avg_gt    = []   # per-image GT count for density filter in PR sweep
    all_image_stems: list[str] = []

    # Buffer for worst-N FN visualization: (fn_count, img_idx, bgr_uint8, gt_boxes, pred_boxes)
    image_buffer = []

    # FN / TP size accumulation
    fn_areas = []
    tp_areas = []

    total_gt = 0
    total_pred = 0
    total_gap_filled = 0

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(loader, desc="Validating")):
            if num_samples is not None and idx >= num_samples:
                break

            images = batch["image"].to(DEVICE)
            gt_boxes = batch["boxes"][0].cpu().numpy().tolist()

            features = backbone(images)
            outputs = detector(features)

            heatmap_probs = torch.sigmoid(outputs["heatmap"])
            bbox_reg = outputs["bbox"]

            _, _, Hf, Wf = features.shape
            debug_this = (idx == 0)

            img_avg_gt = batch.get("avg_gt_per_image", [STAGE1_AVG_GT_PER_IMAGE])[0]
            pred_boxes, pred_scores, _ = extract_boxes_from_heatmap(
                heatmap_probs=heatmap_probs,
                bbox_reg=bbox_reg,
                confidence_thresh=confidence_thresh,
                output_size=(Hf, Wf),
                image_size=IMAGE_SIZE,
                top_k=top_k,
                nms_iou=nms_iou,
                min_box_size=min_box_size,
                debug=debug_this,
                avg_gt_per_image=img_avg_gt,
            )

            # Raw peaks (no threshold) for PR-curve sweep and gap-filling
            raw_boxes, raw_scores = _extract_all_peaks(
                heatmap_probs=heatmap_probs,
                bbox_reg=bbox_reg,
                output_size=(Hf, Wf),
                image_size=IMAGE_SIZE,
                top_k=top_k,
                min_box_size=min_box_size,
            )

            # Neighbor gap-fill: probe subthreshold peaks to recover alternating misses
            if gap_fill:
                n_before = len(pred_boxes)
                pred_boxes, pred_scores = fill_detection_gaps(
                    pred_boxes, pred_scores, raw_boxes, raw_scores
                )
                total_gap_filled += len(pred_boxes) - n_before

            all_gt_boxes.append(gt_boxes)
            all_pred_boxes.append(pred_boxes)
            all_raw_peaks.append((raw_boxes, raw_scores))
            all_avg_gt.append(img_avg_gt)
            stem = batch.get("image_stem")
            all_image_stems.append(stem[0] if isinstance(stem, (list, tuple)) else (stem or ""))

            total_gt += len(gt_boxes)
            total_pred += len(pred_boxes)

            if debug_this:
                print(f"    num gt boxes: {len(gt_boxes)}")
                if len(pred_boxes) > 0:
                    pred_arr = np.asarray(pred_boxes, dtype=np.float32)
                    widths = pred_arr[:, 2] - pred_arr[:, 0]
                    heights = pred_arr[:, 3] - pred_arr[:, 1]
                    print(f"    pred box width  stats min/max/mean: {widths.min():.2f} / {widths.max():.2f} / {widths.mean():.2f}")
                    print(f"    pred box height stats min/max/mean: {heights.min():.2f} / {heights.max():.2f} / {heights.mean():.2f}")
                else:
                    print("    no predicted boxes after decode")

            # Per-image FN count + size analysis
            fn_idx = _get_fn_indices(gt_boxes, pred_boxes, iou_threshold)
            fn_count = len(fn_idx)
            fn_rate = fn_count / max(1, len(gt_boxes))

            for i, box in enumerate(gt_boxes):
                x1, y1, x2, y2 = box
                area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                if i in fn_idx:
                    fn_areas.append(area)
                else:
                    tp_areas.append(area)

            # Store for worst-N selection (denormalize once, keep as uint8 to save memory)
            bgr = denormalize_image(images[0])
            image_buffer.append((fn_count, fn_rate, idx, bgr, gt_boxes, pred_boxes))

    # --------------------------------------------------
    # Global metrics
    # --------------------------------------------------
    metrics = compute_detection_metrics(all_gt_boxes, all_pred_boxes, iou_threshold=iou_threshold)

    avg_gt   = total_gt   / max(1, len(all_gt_boxes))
    avg_pred = total_pred / max(1, len(all_gt_boxes))

    print("\n" + "=" * 70)
    print("STAGE1 DETECTION METRICS (geometry-correct)")
    print("=" * 70)
    print(f"Images evaluated: {len(all_gt_boxes)}")
    print(f"Avg GT boxes/img:   {avg_gt:.2f}")
    print(f"Avg Pred boxes/img: {avg_pred:.2f}")
    if gap_fill:
        print(f"Gap-filled boxes:   {total_gap_filled} ({total_gap_filled / max(1, len(all_gt_boxes)):.1f}/img)")
    print(f"TP: {metrics['tp']} | FP: {metrics['fp']} | FN: {metrics['fn']}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print(f"Mean IoU(TP): {metrics['mean_iou_tp']:.4f} (matches={metrics['num_matches']})")
    print("=" * 70)

    # --------------------------------------------------
    # FN size stats (console)
    # --------------------------------------------------
    if fn_areas and tp_areas:
        fn_arr = np.array(fn_areas)
        tp_arr = np.array(tp_areas)
        print(f"\nFN box area  — median: {np.median(fn_arr):.1f}px²  mean: {np.mean(fn_arr):.1f}px²  p25: {np.percentile(fn_arr, 25):.1f}  p75: {np.percentile(fn_arr, 75):.1f}")
        print(f"TP box area  — median: {np.median(tp_arr):.1f}px²  mean: {np.mean(tp_arr):.1f}px²  p25: {np.percentile(tp_arr, 25):.1f}  p75: {np.percentile(tp_arr, 75):.1f}")
        fn_small = (fn_arr < np.median(tp_arr)).sum()
        print(f"FN boxes smaller than TP median: {fn_small}/{len(fn_arr)} ({100*fn_small/len(fn_arr):.1f}%)")

    # --------------------------------------------------
    # Save worst-N images — two orderings
    #   by_rate : highest FN proportion (FN / GT)  — best for spotting patterns
    #   by_count: highest raw FN count             — worst absolute misses
    # Each folder gets three views per image:
    #   _fn_only : ONLY the missed boxes (no clutter)
    #   _fn      : TP=green, FN=red, pred=blue
    #   _centers : GT=green dot, pred=red dot
    # --------------------------------------------------
    per_image_fn_stats = []

    def _save_worst(sorted_buf, folder_name):
        vis_dir = out_dir / folder_name
        vis_dir.mkdir(exist_ok=True)
        for rank, (fn_count, fn_rate, img_idx, bgr, gt_boxes, pred_boxes) in enumerate(sorted_buf[:worst_n]):
            stem = f"rank{rank:02d}_fn{fn_count}_rate{fn_rate:.2f}_img{img_idx:04d}"
            visualize_fn_only(bgr, gt_boxes, pred_boxes,
                              vis_dir / f"{stem}_fn_only.png", iou_threshold,
                              fn_count=fn_count, fn_rate=fn_rate)
            visualize_fn_highlighted(bgr, gt_boxes, pred_boxes,
                                     vis_dir / f"{stem}_fn.png", iou_threshold,
                                     fn_count=fn_count, fn_rate=fn_rate)
            visualize_centers_only(bgr, gt_boxes, pred_boxes,
                                   vis_dir / f"{stem}_centers.png")
            if rank == 0:  # full box overlay only for rank-0 to save disk
                visualize_boxes_only(bgr, gt_boxes, pred_boxes,
                                     vis_dir / f"{stem}_all.png")
        return vis_dir

    # Collect per-image stats
    for fn_count, fn_rate, img_idx, _, gt_boxes, pred_boxes in image_buffer:
        per_image_fn_stats.append({
            "img_idx": img_idx,
            "fn_count": fn_count,
            "fn_rate": round(fn_rate, 4),
            "gt_count": len(gt_boxes),
            "pred_count": len(pred_boxes),
        })

    by_rate  = sorted(image_buffer, key=lambda x: x[1], reverse=True)
    by_count = sorted(image_buffer, key=lambda x: x[0], reverse=True)

    dir_rate  = _save_worst(by_rate,  "worst_fn_rate")
    dir_count = _save_worst(by_count, "worst_fn_count")

    per_image_fn_stats.sort(key=lambda r: r["fn_rate"], reverse=True)
    with open(out_dir / "per_image_fn_stats.json", "w") as f:
        json.dump(per_image_fn_stats, f, indent=2)
    print(f"\nWorst-{worst_n} by FN rate:  {dir_rate}")
    print(f"Worst-{worst_n} by FN count: {dir_count}")

    # --------------------------------------------------
    # PR curve sweep
    # --------------------------------------------------
    print("\nRunning PR-curve threshold sweep...")
    pr_thresholds = np.linspace(0.05, 0.90, 18).tolist()
    pr_results = []
    for thresh in pr_thresholds:
        filtered = []
        for (raw_boxes, raw_scores), avg_gt in zip(all_raw_peaks, all_avg_gt):
            boxes_t = _filter_peaks(raw_boxes, raw_scores, thresh, nms_iou, avg_gt_per_image=avg_gt)
            if gap_fill:
                boxes_t, _ = fill_detection_gaps(boxes_t, [thresh] * len(boxes_t), raw_boxes, raw_scores)
            filtered.append(boxes_t)
        m = compute_detection_metrics(all_gt_boxes, filtered, iou_threshold=iou_threshold)
        pr_results.append((round(thresh, 3), m["precision"], m["recall"], m["f1"]))

    best_f1_row = max(pr_results, key=lambda r: r[3])
    print(f"  Optimal threshold: {best_f1_row[0]:.2f}  "
          f"→ P={best_f1_row[1]:.4f}  R={best_f1_row[2]:.4f}  F1={best_f1_row[3]:.4f}")
    print(f"  Current threshold: {confidence_thresh:.2f}  "
          f"→ P={metrics['precision']:.4f}  R={metrics['recall']:.4f}  F1={metrics['f1']:.4f}")

    plot_pr_curve(pr_results, out_dir / "pr_curve.png")

    with open(out_dir / "pr_curve.json", "w") as f:
        json.dump([{"threshold": r[0], "precision": r[1], "recall": r[2], "f1": r[3]}
                   for r in pr_results], f, indent=2)

    # --------------------------------------------------
    # FN size analysis plot
    # --------------------------------------------------
    plot_fn_size_analysis(fn_areas, tp_areas, out_dir / "fn_size_analysis.png")

    # --------------------------------------------------
    # Save metrics JSON
    # --------------------------------------------------
    metrics_to_save = {
        **metrics,
        "confidence_thresh": confidence_thresh,
        "top_k": top_k,
        "nms_iou": nms_iou,
        "iou_threshold": iou_threshold,
        "min_box_size": min_box_size,
        "images_evaluated": len(all_gt_boxes),
        "avg_gt_per_img": avg_gt,
        "avg_pred_per_img": avg_pred,
        "optimal_threshold": best_f1_row[0],
        "optimal_f1": best_f1_row[3],
        "fn_area_median": float(np.median(fn_areas)) if fn_areas else None,
        "tp_area_median": float(np.median(tp_areas)) if tp_areas else None,
    }

    def _to_jsonable(v):
        return v.item() if torch.is_tensor(v) else v

    with open(out_dir / "metrics.json", "w") as f:
        json.dump({k: _to_jsonable(v) for k, v in metrics_to_save.items()}, f, indent=2)

    print(f"\n✅ saved: {out_dir / 'metrics.json'}")
    print(f"✅ worst-{worst_n} by FN rate:  {dir_rate}")
    print(f"✅ worst-{worst_n} by FN count: {dir_count}")
    print(f"✅ PR curve: {out_dir / 'pr_curve.png'}")
    print(f"✅ FN size:  {out_dir / 'fn_size_analysis.png'}")

    if return_detailed:
        metrics["all_gt_boxes"]    = all_gt_boxes
        metrics["all_pred_boxes"]  = all_pred_boxes
        metrics["pr_results"]      = pr_results
        metrics["fn_areas"]        = fn_areas
        metrics["tp_areas"]        = tp_areas
        metrics["image_stems"]     = all_image_stems

    return metrics


# -------------------------
# Tiled inference
# -------------------------

def _run_tiled_inference(
    backbone,
    detector,
    image_tensor: torch.Tensor,     # (1, 3, H, W) already on device, normalized
    confidence_thresh: float,
    image_size: tuple,              # (H, W) of image_tensor (e.g. (512, 512))
    tile_size: int,                 # side of each tile in image_tensor pixels
    tile_overlap: int,              # overlap between adjacent tiles in pixels
    top_k: int,
    nms_iou: float,
    min_box_size: float,
    avg_gt_per_image: float = STAGE1_AVG_GT_PER_IMAGE,
) -> tuple:
    """
    Run the detector on overlapping tiles of the image, then merge results.

    Each tile is upscaled to `image_size` before inference so the model sees
    characters at a higher effective resolution — the primary purpose is to
    test whether small/furigana characters that are missed at full resolution
    become detectable when their tile is scaled up 2×.

    Returns (boxes, scores) in full-image pixel coordinates.
    """
    H, W = image_size
    stride = max(1, tile_size - tile_overlap)

    # Generate tile top-left corners, clamped so tiles never exceed image bounds
    def _tile_starts(length: int) -> list:
        starts = list(range(0, length, stride))
        # Ensure the last tile reaches the image edge
        if not starts or starts[-1] + tile_size < length:
            starts.append(max(0, length - tile_size))
        return sorted(set(starts))

    ys = _tile_starts(H)
    xs = _tile_starts(W)

    all_boxes: list = []
    all_scores: list = []

    for y0 in ys:
        y1 = min(y0 + tile_size, H)
        for x0 in xs:
            x1 = min(x0 + tile_size, W)

            tile = image_tensor[:, :, y0:y1, x0:x1]                           # (1,3,th,tw)
            tile_up = F.interpolate(tile, size=image_size, mode="bilinear",
                                    align_corners=False)                        # (1,3,H,W)

            with torch.no_grad():
                feats = backbone(tile_up)
                out = detector(feats)
                hm = torch.sigmoid(out["heatmap"])
                bbox = out["bbox"]

            _, _, Hf, Wf = feats.shape

            # Decode peaks in tile-upscaled (image_size) space — no density filter here
            tile_boxes, tile_scores = _extract_all_peaks(
                heatmap_probs=hm,
                bbox_reg=bbox,
                output_size=(Hf, Wf),
                image_size=image_size,
                top_k=top_k,
                min_box_size=min_box_size,
            )

            if not tile_boxes:
                continue

            # Filter by confidence threshold
            paired = [(b, s) for b, s in zip(tile_boxes, tile_scores)
                      if s >= confidence_thresh]
            if not paired:
                continue
            fboxes, fscores = zip(*paired)

            # Map from image_size coords → tile coords → global image coords
            scale_x = (x1 - x0) / float(W)
            scale_y = (y1 - y0) / float(H)
            for (bx1, by1, bx2, by2), sc in zip(fboxes, fscores):
                gx1 = float(np.clip(bx1 * scale_x + x0, 0, W))
                gy1 = float(np.clip(by1 * scale_y + y0, 0, H))
                gx2 = float(np.clip(bx2 * scale_x + x0, 0, W))
                gy2 = float(np.clip(by2 * scale_y + y0, 0, H))
                if gx2 > gx1 and gy2 > gy1:
                    all_boxes.append([gx1, gy1, gx2, gy2])
                    all_scores.append(float(sc))

    if not all_boxes:
        return [], []

    # Global NMS across all tiles
    keep = non_max_suppression(all_boxes, all_scores, iou_threshold=nms_iou)
    all_boxes  = [all_boxes[i] for i in keep]
    all_scores = [all_scores[i] for i in keep]

    # Global density filter (same parameters as full-image run)
    all_boxes, all_scores = spatial_density_filter(
        all_boxes, all_scores, image_size,
        grid_size=STAGE1_DENSITY_GRID,
        density_factor=STAGE1_DENSITY_FACTOR,
        avg_gt_per_image=avg_gt_per_image,
    )
    return all_boxes, all_scores


def validate_stage1_tiled(
    checkpoint_path,
    confidence_thresh=DET_SCORE_THRESH,
    split="val",
    num_samples=None,
    top_k=DET_TOP_K,
    nms_iou=DET_NMS_IOU,
    iou_threshold=0.2,
    min_box_size=DET_MIN_BOX_SIZE,
    tile_size=256,
    tile_overlap=64,
    job_id=None,
):
    """
    Run tile-based Stage 1 validation and compare against full-image results.

    Prints a side-by-side recall/precision/F1 table and a per-size-bucket
    breakdown showing which box sizes benefit from tiling.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    resolved_job_id = job_id or os.environ.get("SLURM_JOB_ID") or os.environ.get("JOB_ID")
    run_tag = f"{timestamp}_tiled_job{resolved_job_id}" if resolved_job_id else f"{timestamp}_tiled"
    out_dir = Path(CHECKPOINT_DIR) / "stage1_validation" / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🔬 Stage1 TILED validation")
    print(f"  ckpt: {checkpoint_path}")
    print(f"  split: {split}  |  tile_size: {tile_size}  |  tile_overlap: {tile_overlap}")
    print(f"  conf: {confidence_thresh} | top_k: {top_k} | nms_iou: {nms_iou} | iou_thr: {iou_threshold}")
    print(f"  out: {out_dir}")

    ann_files = sorted(list((Path(DATA_DIR) / "annotations").glob("*.json")))
    vocab = VocabManager.from_annotations(ann_files)

    dataset = KuzushijiDataset(
        Path(DATA_DIR), vocab=vocab, use_sequences=False, resize=IMAGE_SIZE, split=split,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    backbone_type = ckpt.get("backbone_type", BACKBONE_TYPE)
    backbone = build_backbone(backbone_type, BACKBONE_BASE_FEATURES, pretrained=False).to(DEVICE)
    detector = DetectorHead(in_ch=BACKBONE_BASE_FEATURES, num_classes=vocab.vocab_size,
                            predict_classes=False).to(DEVICE)
    state_key = "backbone_state_dict" if "backbone_state_dict" in ckpt else "unet_state_dict"
    backbone.load_state_dict(ckpt[state_key], strict=True)
    detector.load_state_dict(ckpt["detector_state_dict"], strict=True)
    backbone.eval()
    detector.eval()

    all_gt         = []
    all_full_pred  = []
    all_tiled_pred = []

    # Per-size-bucket accumulators: small (<12px side), medium (12-20), large (>20)
    SIZE_BUCKETS = [("small_<12", 0, 144), ("medium_12-20", 144, 400), ("large_>20", 400, 1e9)]
    full_fn_by_bucket  = {name: 0 for name, *_ in SIZE_BUCKETS}
    tiled_fn_by_bucket = {name: 0 for name, *_ in SIZE_BUCKETS}
    gt_by_bucket       = {name: 0 for name, *_ in SIZE_BUCKETS}

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(loader, desc="Tiled validation")):
            if num_samples is not None and idx >= num_samples:
                break

            images    = batch["image"].to(DEVICE)
            gt_boxes  = batch["boxes"][0].cpu().numpy().tolist()
            all_gt.append(gt_boxes)

            # --- Full-image prediction ---
            feats = backbone(images)
            out   = detector(feats)
            hm    = torch.sigmoid(out["heatmap"])
            bbox  = out["bbox"]
            _, _, Hf, Wf = feats.shape

            full_boxes, full_scores = _extract_all_peaks(
                heatmap_probs=hm, bbox_reg=bbox,
                output_size=(Hf, Wf), image_size=IMAGE_SIZE,
                top_k=top_k, min_box_size=min_box_size,
            )
            tiled_avg_gt = batch.get("avg_gt_per_image", [STAGE1_AVG_GT_PER_IMAGE])[0]
            full_pred = _filter_peaks(full_boxes, full_scores, confidence_thresh, nms_iou,
                                      avg_gt_per_image=tiled_avg_gt)
            all_full_pred.append(full_pred)

            # --- Tiled prediction ---
            tiled_boxes, _ = _run_tiled_inference(
                backbone=backbone, detector=detector,
                image_tensor=images,
                confidence_thresh=confidence_thresh,
                image_size=IMAGE_SIZE,
                tile_size=tile_size, tile_overlap=tile_overlap,
                top_k=top_k, nms_iou=nms_iou, min_box_size=min_box_size,
                avg_gt_per_image=tiled_avg_gt,
            )
            all_tiled_pred.append(tiled_boxes)

            # Per-size FN breakdown
            fn_full  = _get_fn_indices(gt_boxes, full_pred,  iou_threshold)
            fn_tiled = _get_fn_indices(gt_boxes, tiled_boxes, iou_threshold)
            for i, (x1, y1, x2, y2) in enumerate(gt_boxes):
                area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                for name, lo, hi in SIZE_BUCKETS:
                    if lo <= area < hi:
                        gt_by_bucket[name] += 1
                        if i in fn_full:
                            full_fn_by_bucket[name] += 1
                        if i in fn_tiled:
                            tiled_fn_by_bucket[name] += 1
                        break

    m_full  = compute_detection_metrics(all_gt, all_full_pred,  iou_threshold=iou_threshold)
    m_tiled = compute_detection_metrics(all_gt, all_tiled_pred, iou_threshold=iou_threshold)

    avg_full_pred  = sum(len(b) for b in all_full_pred)  / max(1, len(all_gt))
    avg_tiled_pred = sum(len(b) for b in all_tiled_pred) / max(1, len(all_gt))

    print("\n" + "=" * 70)
    print("TILED vs FULL-IMAGE DETECTION COMPARISON")
    print(f"  tile_size={tile_size}  overlap={tile_overlap}  "
          f"n_tiles≈{len(_tile_starts_static(IMAGE_SIZE[0], tile_size, tile_size - tile_overlap)) ** 2}")
    print("=" * 70)
    print(f"{'Mode':<12} {'Recall':>8} {'Precision':>10} {'F1':>8} {'Pred/img':>10}")
    print(f"{'Full':12} {m_full['recall']:8.4f} {m_full['precision']:10.4f} {m_full['f1']:8.4f} {avg_full_pred:10.1f}")
    print(f"{'Tiled':12} {m_tiled['recall']:8.4f} {m_tiled['precision']:10.4f} {m_tiled['f1']:8.4f} {avg_tiled_pred:10.1f}")
    recall_delta = m_tiled["recall"] - m_full["recall"]
    print(f"\n  Δ Recall  = {recall_delta:+.4f}  ({recall_delta*100:+.2f} pp)")
    print(f"  Δ Pred/img = {avg_tiled_pred - avg_full_pred:+.1f}")

    print("\n  Per-size-bucket recall (area thresholds: small<144px², medium 144-400, large>400):")
    print(f"  {'Bucket':<18} {'GT':>6} {'Full FN':>8} {'Full R':>8} {'Tiled FN':>9} {'Tiled R':>8} {'Δ R':>8}")
    for name, lo, hi in SIZE_BUCKETS:
        n = gt_by_bucket[name]
        ffn, tfn = full_fn_by_bucket[name], tiled_fn_by_bucket[name]
        fr  = 1 - ffn / max(1, n)
        tr  = 1 - tfn / max(1, n)
        print(f"  {name:<18} {n:>6} {ffn:>8} {fr:>8.3f} {tfn:>9} {tr:>8.3f} {tr-fr:>+8.3f}")

    results = {
        "tile_size": tile_size, "tile_overlap": tile_overlap,
        "full_image": m_full, "tiled": m_tiled,
        "avg_pred_per_img_full": avg_full_pred, "avg_pred_per_img_tiled": avg_tiled_pred,
        "per_bucket": {
            name: {
                "gt": gt_by_bucket[name],
                "full_fn": full_fn_by_bucket[name],
                "tiled_fn": tiled_fn_by_bucket[name],
                "full_recall": 1 - full_fn_by_bucket[name] / max(1, gt_by_bucket[name]),
                "tiled_recall": 1 - tiled_fn_by_bucket[name] / max(1, gt_by_bucket[name]),
            }
            for name, *_ in SIZE_BUCKETS
        },
    }
    with open(out_dir / "tiled_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ saved: {out_dir / 'tiled_metrics.json'}")
    return results


def _tile_starts_static(length: int, tile_size: int, stride: int) -> list:
    """Helper for counting tiles (mirrors _run_tiled_inference logic)."""
    starts = list(range(0, length, stride))
    if not starts or starts[-1] + tile_size < length:
        starts.append(max(0, length - tile_size))
    return sorted(set(starts))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument("--confidence", type=float, default=DET_SCORE_THRESH)
    p.add_argument("--top_k", type=int, default=DET_TOP_K)
    p.add_argument("--nms_iou", type=float, default=DET_NMS_IOU)
    p.add_argument("--iou_thr", type=float, default=0.5)
    p.add_argument("--min_box_size", type=float, default=DET_MIN_BOX_SIZE)
    p.add_argument("--job_id", type=str, default=None,
                   help="Optional job id suffix for output folder (default: SLURM_JOB_ID env)")
    p.add_argument("--num_samples", type=int, default=0, help="0 => full split")
    p.add_argument("--worst_n", type=int, default=30,
                   help="Save this many worst-FN images (sorted by FN count)")
    p.add_argument("--tiled", action="store_true",
                   help="Also run tiled inference and compare recall per size bucket")
    p.add_argument("--tile_size", type=int, default=256,
                   help="Tile side length in image pixels (default 256 = half of 512)")
    p.add_argument("--tile_overlap", type=int, default=64,
                   help="Overlap between adjacent tiles in pixels (default 64)")
    p.add_argument("--no_gap_fill", action="store_true",
                   help="Disable neighbor gap-filler (on by default). Use this to get the "
                        "raw Stage 1 baseline without gap recovery.")
    args = p.parse_args()

    validate_stage1(
        checkpoint_path=args.checkpoint,
        confidence_thresh=args.confidence,
        split=args.split,
        num_samples=None if args.num_samples == 0 else args.num_samples,
        top_k=args.top_k,
        nms_iou=args.nms_iou,
        iou_threshold=args.iou_thr,
        min_box_size=args.min_box_size,
        job_id=args.job_id,
        worst_n=args.worst_n,
        gap_fill=not args.no_gap_fill,
    )

    if args.tiled:
        validate_stage1_tiled(
            checkpoint_path=args.checkpoint,
            confidence_thresh=args.confidence,
            split=args.split,
            num_samples=None if args.num_samples == 0 else args.num_samples,
            top_k=args.top_k,
            nms_iou=args.nms_iou,
            iou_threshold=args.iou_thr,
            min_box_size=args.min_box_size,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            job_id=args.job_id,
        )
