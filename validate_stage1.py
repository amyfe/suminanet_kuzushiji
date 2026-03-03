"""
Validate Stage 1 (DetectorHead) - visualize box predictions and compute metrics.
Fixes:
- bbox decode matches training targets (dx,dy,bw,bh)
- no double-sigmoid on heatmap
- confidence thresh wired correctly
- full-val metrics + extra logs
"""
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

from config import DATA_DIR, DEVICE, IMAGE_SIZE, CHECKPOINT_DIR
from model.kuronet import UNet, DetectorHead
from utils import KuzushijiDataset
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
    heatmap,            # (1,1,H,W) already sigmoid
    bbox_reg,           # (1,4,H,W) = (dx,dy,bw,bh)
    confidence_thresh=0.5,     
    output_size=(64, 64),
    image_size=IMAGE_SIZE,
    top_k=200,
    nms_iou=0.5,
):
    import torch.nn.functional as F

    hm = heatmap[0, 0]      # (H,W)
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

    ys = peak_idx[order, 0].cpu().numpy()
    xs = peak_idx[order, 1].cpu().numpy()
    scores_np = sorted_scores[:len(order)].detach().cpu().numpy().tolist()

    H, W = hm.shape
    H_img, W_img = image_size
    H_out, W_out = output_size

    stride_h = H_img / float(H_out)
    stride_w = W_img / float(W_out)

    bbox_np = bbox.detach().cpu().numpy()

    boxes = []
    skipped_count = 0
    for y, x, sc in zip(ys, xs, scores_np):
        dx, dy, bw, bh = bbox_np[:, y, x]

        # Ensure width/height are positive (use softplus-like behavior)
        bw = max(bw, 0.1)  # Minimum 0.1 grid units
        bh = max(bh, 0.1)

        cx = (x + dx) * stride_w
        cy = (y + dy) * stride_h
        w  = bw * stride_w
        h  = bh * stride_h
        x1 = cx - 0.5*w
        y1 = cy - 0.5*h
        x2 = cx + 0.5*w
        y2 = cy + 0.5*h

        # enforce ordering
        if x2 <= x1 or y2 <= y1:
            skipped_count += 1
            continue

        boxes.append([x1, y1, x2, y2])
    
    # Debug: print stats on first batch
    if len(ys) > 0:
        print(f"  → Peak detection: {len(ys)} peaks found, {skipped_count} skipped (bad ordering)")
        print(f"    Sample predictions: bw={bbox_np[2, ys[0], xs[0]]:.2f}, bh={bbox_np[3, ys[0], xs[0]]:.2f}")
        print(f"    Stride: h={stride_h:.2f}, w={stride_w:.2f}")

    if len(boxes) == 0:
        return [], [], []

    # NMS
    keep = non_max_suppression(boxes, scores_np[:len(boxes)], iou_threshold=nms_iou)
    boxes = [boxes[i] for i in keep]
    scores_out = [scores_np[i] for i in keep]

    # you currently don’t predict classes in stage1 => dummy class=0
    classes = [0] * len(boxes)
    return boxes, scores_out, classes


# -------------------------
# Metrics: add mean IoU of matched TPs
# -------------------------
def compute_detection_metrics(gt_boxes_list, pred_boxes_list, iou_threshold=0.5):
    total_tp = total_fp = total_fn = 0
    matched_ious = []

    for gt_boxes, pred_boxes in zip(gt_boxes_list, pred_boxes_list):
        gt_boxes = np.asarray(gt_boxes, dtype=np.float32)
        pred_boxes = np.asarray(pred_boxes, dtype=np.float32)

        if gt_boxes.ndim == 1 and gt_boxes.size > 0:
            gt_boxes = gt_boxes.reshape(1, 4)
        if pred_boxes.ndim == 1 and pred_boxes.size > 0:
            pred_boxes = pred_boxes.reshape(1, 4)

        if len(gt_boxes) == 0 and len(pred_boxes) == 0:
            continue
        if len(pred_boxes) == 0:
            total_fn += len(gt_boxes)
            continue
        if len(gt_boxes) == 0:
            total_fp += len(pred_boxes)
            continue

        ious = np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.float32)
        for i, pred_box in enumerate(pred_boxes):
            ious[i] = compute_iou_batch(pred_box, gt_boxes)

        matched_gt = set()
        for i in range(len(pred_boxes)):
            best_iou = 0.0
            best_gt = -1
            for j in range(len(gt_boxes)):
                if j not in matched_gt and ious[i, j] > best_iou:
                    best_iou = float(ious[i, j])
                    best_gt = j
            if best_iou >= iou_threshold:
                total_tp += 1
                matched_gt.add(best_gt)
                matched_ious.append(best_iou)
            else:
                total_fp += 1

        total_fn += len(gt_boxes) - len(matched_gt)

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


def visualize_boxes_only(image_tensor, gt_boxes, pred_boxes, out_path):
    img = denormalize_image(image_tensor)
    for box in gt_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
    for box in pred_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imwrite(str(out_path), img)


def validate_stage1(
    checkpoint_path,
    confidence_thresh=0.5,      
    split="val",
    num_samples=None,          # None => full split
    top_k=200,
    nms_iou=0.5,
    iou_threshold=0.5,
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(CHECKPOINT_DIR) / "stage1_validation" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🔍 Stage1 validation")
    print(f"  ckpt: {checkpoint_path}")
    print(f"  split: {split}")
    print(f"  conf: {confidence_thresh} | top_k: {top_k} | nms_iou: {nms_iou} | iou_thr: {iou_threshold}")
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

    unet = UNet(in_channels=3, base_features=32).to(DEVICE)
    detector = DetectorHead(in_ch=32, num_classes=vocab.vocab_size, predict_classes=False).to(DEVICE)

    unet.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1], device=DEVICE)
        _ = unet(dummy)  # triggers FusionBlock conv/gn creation

    ckpt = torch.load(checkpoint_path, map_location=DEVICE)

    unet.load_state_dict(ckpt["unet_state_dict"], strict=True)
    detector.load_state_dict(ckpt["detector_state_dict"], strict=True)

    unet.eval()
    detector.eval()

    all_gt_boxes = []
    all_pred_boxes = []

    # logs
    total_gt = 0
    total_pred = 0

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(loader, desc="Validating")):
            if num_samples is not None and idx >= num_samples:
                break

            images = batch["image"].to(DEVICE)
            gt_boxes = batch["boxes"][0].cpu().numpy().tolist()

            features = unet(images)
            outputs = detector(features)

            heatmap = outputs["heatmap"]  
            bbox_reg = outputs["bbox"]

            _, _, Hf, Wf = features.shape

            pred_boxes, pred_scores, _ = extract_boxes_from_heatmap(
                heatmap=heatmap,
                bbox_reg=bbox_reg,
                confidence_thresh=confidence_thresh,
                output_size=(Hf, Wf),
                image_size=IMAGE_SIZE,
                top_k=top_k,
                nms_iou=nms_iou,
            )

            all_gt_boxes.append(gt_boxes)
            all_pred_boxes.append(pred_boxes)

            total_gt += len(gt_boxes)
            total_pred += len(pred_boxes)

            # check sth for first batch
            if idx == 0:
                hm = heatmap[0,0].detach().cpu()
                bb = bbox_reg[0].detach().cpu()
                print("heatmap stats min/max/mean:", float(hm.min()), float(hm.max()), float(hm.mean()))
                print("bbox dx/dy/bw/bh mean:", [float(bb[c].mean()) for c in range(4)])
                print("bbox bw/bh min/max:", float(bb[2].min()), float(bb[2].max()), float(bb[3].min()), float(bb[3].max()))
                print("num gt boxes", len(gt_boxes))
            if idx < 30:
                vis_path = out_dir / f"sample_{idx:04d}.png"
                visualize_boxes_only(images[0], gt_boxes, pred_boxes, vis_path)

    metrics = compute_detection_metrics(all_gt_boxes, all_pred_boxes, iou_threshold=iou_threshold)

    # extra logs
    avg_gt = total_gt / max(1, len(all_gt_boxes))
    avg_pred = total_pred / max(1, len(all_gt_boxes))

    print("\n" + "=" * 70)
    print("STAGE1 DETECTION METRICS (geometry-correct)")
    print("=" * 70)
    print(f"Images evaluated: {len(all_gt_boxes)}")
    print(f"Avg GT boxes/img:   {avg_gt:.2f}")
    print(f"Avg Pred boxes/img: {avg_pred:.2f}")
    print(f"TP: {metrics['tp']} | FP: {metrics['fp']} | FN: {metrics['fn']}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print(f"Mean IoU(TP): {metrics['mean_iou_tp']:.4f} (matches={metrics['num_matches']})")
    print("=" * 70)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(
            {
                **metrics,
                "confidence_thresh": confidence_thresh,
                "top_k": top_k,
                "nms_iou": nms_iou,
                "iou_threshold": iou_threshold,
                "images_evaluated": len(all_gt_boxes),
                "avg_gt_per_img": avg_gt,
                "avg_pred_per_img": avg_pred,
            },
            f,
            indent=2,
        )
    print(f"✅ saved: {out_dir/'metrics.json'}")
    print(f"✅ visuals: {out_dir} (first ~30 imgs)")

    return metrics


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--split", type=str, default="val", choices=["train", "val"])
    p.add_argument("--confidence", type=float, default=0.3)
    p.add_argument("--top_k", type=int, default=200)
    p.add_argument("--nms_iou", type=float, default=0.5)
    p.add_argument("--iou_thr", type=float, default=0.5)
    p.add_argument("--num_samples", type=int, default=0, help="0 => full split")
    args = p.parse_args()

    validate_stage1(
        checkpoint_path=args.checkpoint,
        confidence_thresh=args.confidence,
        split=args.split,
        num_samples=None if args.num_samples == 0 else args.num_samples,
        top_k=args.top_k,
        nms_iou=args.nms_iou,
        iou_threshold=args.iou_thr,
    )
