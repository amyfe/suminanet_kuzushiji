"""
Sweep DET_SCORE_THRESH × DET_NMS_IOU and plot recall / precision / F1 /
proposals-per-image.

Runs the frozen Stage 1 detector once per image (backbone + DetectorHead),
caches the raw (heatmap, bbox_reg) tensors, then applies different thresholds
post-hoc via extract_coarse_proposals — matching SuminaNet's exact proposal
pipeline.  GT matching uses IOU_MATCH (default 0.45 = STAGE2_REFINE_POS_IOU)
so numbers are directly comparable to SuminaNet coverage.

Usage:
    python scripts/sweep_det_thresholds.py
    python scripts/sweep_det_thresholds.py --iou_match 0.50   # Stage 1-style
    python scripts/sweep_det_thresholds.py --out_dir /tmp/sweep_results
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    BACKBONE_BASE_FEATURES, BACKBONE_TYPE, CHECKPOINT_DIR, DATA_DIR, DEVICE,
    IMAGE_SIZE, DENSITY_FACTOR, DENSITY_GRID, AVG_GT_PER_IMAGE,
    DET_MIN_BOX_SIZE,
)
from model.suminanet import DetectorHead, build_backbone
from model.suminanet.detection.proposal_utils import extract_coarse_proposals
from utils import KuzushijiDataset
from utils.vocab import VocabManager


# ── Sweep grid ──────────────────────────────────────────────────────────────
SCORE_THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
                    0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

NMS_IOU_VALUES   = [0.40, 0.50, 0.60, 0.70]   # one curve per NMS IoU value

TOP_K            = 500    # same as DET_TOP_K in config during SuminaNet forward


# ── GT matching ─────────────────────────────────────────────────────────────
def _iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute (M, N) IoU matrix between two sets of xyxy boxes."""
    if boxes_a.shape[0] == 0 or boxes_b.shape[0] == 0:
        return np.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float32)
    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union  = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-6)


def match(gt_boxes: np.ndarray, pred_boxes: np.ndarray, iou_thresh: float):
    """Return (tp, fp, fn) counts."""
    if pred_boxes.shape[0] == 0:
        return 0, 0, gt_boxes.shape[0]
    if gt_boxes.shape[0] == 0:
        return 0, pred_boxes.shape[0], 0

    iou = _iou_matrix(gt_boxes, pred_boxes)          # (N_gt, N_pred)
    gt_matched  = np.zeros(gt_boxes.shape[0],  dtype=bool)
    pred_matched = np.zeros(pred_boxes.shape[0], dtype=bool)

    flat_order = np.dstack(np.unravel_index(np.argsort(-iou.ravel()), iou.shape))[0]
    for gi, pi in flat_order:
        if iou[gi, pi] < iou_thresh:
            break
        if not gt_matched[gi] and not pred_matched[pi]:
            gt_matched[gi]   = True
            pred_matched[pi] = True

    tp = int(gt_matched.sum())
    fp = int((~pred_matched).sum())
    fn = int((~gt_matched).sum())
    return tp, fp, fn


# ── Model loading ────────────────────────────────────────────────────────────
def load_stage1(device):
    ckpt_path = CHECKPOINT_DIR / "stage1_detection" / "detector_best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    backbone_type = ckpt.get("backbone_type", BACKBONE_TYPE)
    backbone = build_backbone(backbone_type, BACKBONE_BASE_FEATURES, pretrained=False).to(device)
    detector = DetectorHead(in_ch=BACKBONE_BASE_FEATURES, num_classes=1,
                            predict_classes=False).to(device)
    state_key = "backbone_state_dict" if "backbone_state_dict" in ckpt else "unet_state_dict"
    backbone.load_state_dict(ckpt[state_key])
    detector.load_state_dict(ckpt["detector_state_dict"])
    backbone.eval()
    detector.eval()
    print(f"Loaded Stage 1 ({backbone_type}, epoch={ckpt.get('epoch', '?')}) from {ckpt_path}")
    return backbone, detector


# ── Main sweep ───────────────────────────────────────────────────────────────
def run_sweep(iou_match: float, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(DEVICE)

    vocab = VocabManager()
    val_dataset = KuzushijiDataset(DATA_DIR, split="val", resize=IMAGE_SIZE, vocab=vocab)
    val_loader  = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=2,
        collate_fn=getattr(val_dataset, "collate_fn", None),
    )

    backbone, detector = load_stage1(device)

    # ── Pass 1: cache raw detector outputs per image ─────────────────────────
    print(f"\nCaching detector outputs for {len(val_dataset)} images …")
    cached_heatmaps: list[torch.Tensor] = []  # list of (1, 1, Hf, Wf)
    cached_bboxregs: list[torch.Tensor] = []  # list of (1, 4, Hf, Wf)
    all_gt_boxes:    list[np.ndarray]   = []  # list of (N_gt, 4)

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Detector pass"):
            images = batch["image"].to(device)          # (1, 3, H, W)
            boxes  = batch["boxes"][0].numpy()          # (N_gt, 4)

            feats   = backbone(images)
            det_out = detector(feats)

            cached_heatmaps.append(det_out["heatmap"].cpu())
            cached_bboxregs.append(det_out["bbox"].cpu())
            all_gt_boxes.append(boxes.astype(np.float32))

    h_img, w_img = IMAGE_SIZE
    image_size = (h_img, w_img)

    # ── Pass 2: threshold sweep ───────────────────────────────────────────────
    print(f"\nSweeping {len(SCORE_THRESHOLDS)} thresholds × "
          f"{len(NMS_IOU_VALUES)} NMS IoU values …")

    # results[nms_iou] -> list of dicts (one per score_thresh)
    results: dict[float, list[dict]] = {nms: [] for nms in NMS_IOU_VALUES}

    for nms_iou in NMS_IOU_VALUES:
        for score_thresh in tqdm(SCORE_THRESHOLDS,
                                 desc=f"NMS IoU={nms_iou:.2f}", leave=False):
            tp_total = fp_total = fn_total = 0
            props_total = 0

            for heat, bbox_reg, gt_boxes in zip(cached_heatmaps, cached_bboxregs, all_gt_boxes):
                boxes_list, _ = extract_coarse_proposals(
                    heat_logits=heat,
                    bbox_reg=bbox_reg,
                    image_size=image_size,
                    score_thresh=score_thresh,
                    top_k=TOP_K,
                    nms_iou=nms_iou,
                    min_size=DET_MIN_BOX_SIZE,
                    density_grid=DENSITY_GRID,
                    density_factor=DENSITY_FACTOR,
                    avg_gt_per_image=AVG_GT_PER_IMAGE,
                )
                pred_boxes = boxes_list[0].numpy() if boxes_list[0].numel() > 0 else np.zeros((0, 4), dtype=np.float32)
                tp, fp, fn = match(gt_boxes, pred_boxes, iou_thresh=iou_match)
                tp_total  += tp
                fp_total  += fp
                fn_total  += fn
                props_total += pred_boxes.shape[0]

            n_img = len(all_gt_boxes)
            precision = tp_total / max(1, tp_total + fp_total)
            recall    = tp_total / max(1, tp_total + fn_total)
            f1        = 2 * precision * recall / max(1e-9, precision + recall)
            results[nms_iou].append({
                "score_thresh":  score_thresh,
                "recall":        recall,
                "precision":     precision,
                "f1":            f1,
                "proposals_img": props_total / n_img,
            })

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"Stage 1 detector sweep  (IoU match={iou_match:.2f}, "
        f"epoch={cached_heatmaps[0].shape}, val={len(val_dataset)} imgs)",
        fontsize=12,
    )

    metrics = [
        ("recall",        "Recall (coverage)",         "tab:blue"),
        ("precision",     "Precision",                  "tab:orange"),
        ("f1",            "F1",                         "tab:green"),
        ("proposals_img", "Proposals / image",          "tab:red"),
    ]
    colors = plt.cm.tab10.colors

    for ax, (key, label, _) in zip(axes.flat, metrics):
        for idx, nms_iou in enumerate(NMS_IOU_VALUES):
            xs = [r["score_thresh"] for r in results[nms_iou]]
            ys = [r[key]            for r in results[nms_iou]]
            ax.plot(xs, ys, marker="o", markersize=4,
                    color=colors[idx], label=f"NMS IoU={nms_iou:.2f}")

        ax.set_xlabel("Score threshold")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

        # Mark the two thresholds we already know
        for thresh, lbl in [(0.061, "prev"), (0.70, "current")]:
            if min(SCORE_THRESHOLDS) <= thresh <= max(SCORE_THRESHOLDS):
                ax.axvline(thresh, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
                ax.text(thresh + 0.005, ax.get_ylim()[0], lbl,
                        fontsize=7, color="grey", rotation=90, va="bottom")

    plt.tight_layout()
    out_path = out_dir / "det_threshold_sweep.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved: {out_path}")

    # Also print a summary table for the NMS IoU=0.5 curve (closest to Stage 1 val)
    ref_nms = 0.50
    print(f"\nSummary (NMS IoU={ref_nms:.2f}, IoU match={iou_match:.2f}):")
    print(f"  {'thresh':>7}  {'recall':>8}  {'prec':>8}  {'F1':>8}  {'props/img':>10}")
    for r in results[ref_nms]:
        print(f"  {r['score_thresh']:>7.2f}  {r['recall']:>8.4f}  "
              f"{r['precision']:>8.4f}  {r['f1']:>8.4f}  {r['proposals_img']:>10.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iou_match", type=float, default=0.45,
                        help="IoU threshold for GT matching (0.45=SuminaNet, 0.50=Stage1 val)")
    parser.add_argument("--out_dir",   type=str,
                        default=str(ROOT / "checkpoints" / "det_sweep"))
    args = parser.parse_args()
    run_sweep(iou_match=args.iou_match, out_dir=Path(args.out_dir))
