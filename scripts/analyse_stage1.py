"""Stage 1 (detector) result analysis for thesis.

Produces:
  - pr_f2_curve.png            PR + F2 curve (2-panel)
  - threshold_sweep.png        P/R/F1/F2 vs threshold (1-panel)
  - column_density_fn.png      FN rate by column density bucket (2-panel)
  - sam2_contribution.png      Grouped bar: Stage1-only / +gap-fill / +SAM2
  - learning_curves.png        Train/val loss curves from log file

Usage:
    python scripts/analyse_stage1.py --ckpt checkpoints/stage1_detection/detector_best.pt \\
        [--out-dir results/stage1_analysis] \\
        [--split val] \\
        [--log logs/train_stage1.log] \\
        [--sam2-dir assets/sam2_proposals] \\
        [--optuna-db checkpoints/optuna_stage1/optuna_stage1.db]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from utils.validation.validate_stage1 import validate_stage1
from visualization.stage1 import (
    plot_column_density_fn,
    plot_learning_curves,
    plot_pr_f2_curve,
    plot_sam2_contribution,
    plot_threshold_sweep,
)


# ---------------------------------------------------------------------------
# SAM2 recall helper
# ---------------------------------------------------------------------------

def _iou_matrix_np(gt_boxes: list, pred_boxes: list) -> np.ndarray:
    if not gt_boxes or not pred_boxes:
        return np.zeros((len(gt_boxes), len(pred_boxes)), dtype=np.float32)
    g = np.array(gt_boxes, dtype=np.float32)
    p = np.array(pred_boxes, dtype=np.float32)
    ix1 = np.maximum(g[:, None, 0], p[None, :, 0])
    iy1 = np.maximum(g[:, None, 1], p[None, :, 1])
    ix2 = np.minimum(g[:, None, 2], p[None, :, 2])
    iy2 = np.minimum(g[:, None, 3], p[None, :, 3])
    inter = (ix2 - ix1).clip(0) * (iy2 - iy1).clip(0)
    ag = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
    ap = (p[:, 2] - p[:, 0]) * (p[:, 3] - p[:, 1])
    return inter / (ag[:, None] + ap[None, :] - inter + 1e-6)


def _nms_cpu(boxes: list, scores: list, iou_thresh: float = 0.5) -> list:
    """Simple greedy NMS. Returns kept boxes."""
    if not boxes:
        return []
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    kept_boxes = []
    while order:
        i = order.pop(0)
        kept_boxes.append(boxes[i])
        b = boxes[i]
        remaining = []
        for j in order:
            bj = boxes[j]
            ix1, iy1 = max(b[0], bj[0]), max(b[1], bj[1])
            ix2, iy2 = min(b[2], bj[2]), min(b[3], bj[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            ag = (b[2] - b[0]) * (b[3] - b[1])
            abj = (bj[2] - bj[0]) * (bj[3] - bj[1])
            iou = inter / (ag + abj - inter + 1e-6)
            if iou < iou_thresh:
                remaining.append(j)
        order = remaining
    return kept_boxes


def _compute_sam2_metrics(
    all_gt_boxes: list,
    all_pred_boxes: list,
    image_stems: list[str],
    sam2_dir: Path,
    iou_threshold: float = 0.5,
    nms_iou: float = 0.6,
) -> "dict | None":
    """
    For each val image, merge SAM2 proposals with Stage 1 pred boxes,
    apply NMS, then recompute recall. Returns None if no proposals found.
    """
    found_any = False
    total_gt = 0
    total_tp = 0
    total_fp = 0
    total_pred = 0

    for stem, gt_boxes, pred_boxes in zip(image_stems, all_gt_boxes, all_pred_boxes):
        pt_path = sam2_dir / f"{stem}.pt"
        if pt_path.exists():
            sam2_data = torch.load(str(pt_path), map_location="cpu")
            sam2_boxes = sam2_data["boxes"].cpu().numpy().tolist()
            sam2_scores = sam2_data["scores"].cpu().numpy().tolist()
            found_any = True
        else:
            sam2_boxes = []
            sam2_scores = []

        # Merge: give Stage1 preds a score of 1.0 so they are always preferred
        merged_boxes  = pred_boxes + sam2_boxes
        merged_scores = [1.0] * len(pred_boxes) + sam2_scores

        merged_boxes = _nms_cpu(merged_boxes, merged_scores, iou_thresh=nms_iou)

        total_gt   += len(gt_boxes)
        total_pred += len(merged_boxes)

        if gt_boxes and merged_boxes:
            iou_mat = _iou_matrix_np(gt_boxes, merged_boxes)
            matched_pred = set()
            for gi in range(len(gt_boxes)):
                best_pi = int(np.argmax(iou_mat[gi]))
                if iou_mat[gi, best_pi] >= iou_threshold and best_pi not in matched_pred:
                    total_tp += 1
                    matched_pred.add(best_pi)

    if not found_any:
        return None

    precision = total_tp / max(1, total_pred)
    recall    = total_tp / max(1, total_gt)
    f1        = 2 * precision * recall / max(precision + recall, 1e-8)
    return {"precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# Optuna plots
# ---------------------------------------------------------------------------

def _build_optuna_plots(db_path: Path, out_dir: Path) -> None:
    try:
        import optuna
        from utils.optuna.train_stage1_optuna import generate_optuna_png_plots
        storage = f"sqlite:///{db_path.as_posix()}"
        studies = optuna.study.get_all_study_names(storage)
        study_name = studies[0] if studies else None
        study = optuna.load_study(study_name=study_name, storage=storage)
        plot_paths = generate_optuna_png_plots(study, out_dir / "optuna")
        if plot_paths:
            print(f"  Optuna plots ({len(plot_paths)}) → {out_dir / 'optuna'}")
    except Exception as exc:
        print(f"  WARNING: Optuna plot generation failed (non-fatal): {exc}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 1 detector result analysis for thesis")
    p.add_argument("--ckpt", required=True, default="checkpoints/stage1_detection/detector_best.pt", help="Stage 1 detector checkpoint (.pt)")
    p.add_argument("--out-dir", default="results/stage1_analysis")
    p.add_argument("--split", default="val", choices=["train", "val"])
    p.add_argument("--log", default="logs/train_stage1.log",
                   help="Training log file for learning curves (skipped if not found)")
    p.add_argument("--sam2-dir", default="assets/sam2_proposals",
                   help="Directory of SAM2 proposal .pt files")
    p.add_argument("--iou-threshold", type=float, default=0.5,
                   help="IoU threshold for TP/FN classification")
    p.add_argument("--optuna-db", default=None,
                   help="SQLite DB from train_stage1_optuna.py")
    p.add_argument("--top_k", type=int, default=700,
                   help="Number of top predictions to consider")
    p.add_argument("--nms_iou", type=float, default=0.6454054005824295,
                   help="IoU threshold for NMS")
    p.add_argument("--min_box_size", type=float, default=1.0539268301125129,
                   help="Minimum box size to keep")
    p.add_argument("--confidence", type=float, default=0.05833181975183368,
                   help="Confidence threshold for predictions")
    return p.parse_args()
    # Optuna {'confidence': 0.05833181975183368, 'top_k': 700, 'nms_iou': 0.6454054005824295, 'min_box_size': 1.0539268301125129}


def main() -> None:
    args = parse_args()
    ckpt    = Path(args.ckpt)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Stage 1 analysis]  ckpt={ckpt}  split={args.split}  out={out_dir}")

    # ------------------------------------------------------------------
    # 1. Run Stage 1 validation (with gap-fill) — collect detailed data
    # ------------------------------------------------------------------
    print("  validate_stage1(gap_fill=True, return_detailed=True) …")
    metrics = validate_stage1(
        checkpoint_path=str(ckpt),
        confidence_thresh=args.confidence,
        top_k=args.top_k,
        nms_iou=args.nms_iou,
        min_box_size=args.min_box_size,
        split=args.split,
        gap_fill=True,
        iou_threshold=args.iou_threshold,
        return_detailed=True,
    )

    all_gt_boxes   = metrics["all_gt_boxes"]
    all_pred_boxes = metrics["all_pred_boxes"]
    pr_results     = metrics["pr_results"]
    image_stems    = metrics.get("image_stems", [])

    from config import DET_SCORE_THRESH
    current_thresh = DET_SCORE_THRESH

    print(f"  recall={metrics.get('recall', float('nan')):.4f}  "
          f"precision={metrics.get('precision', float('nan')):.4f}  "
          f"f1={metrics.get('f1', float('nan')):.4f}")

    # ------------------------------------------------------------------
    # 2. Run without gap-fill to measure its contribution
    # ------------------------------------------------------------------
    print("  validate_stage1(gap_fill=False) …")
    metrics_no_gapfill = validate_stage1(
        checkpoint_path=str(ckpt),
        confidence_thresh=args.confidence,
        top_k=args.top_k,
        nms_iou=args.nms_iou,
        min_box_size=args.min_box_size,
        split=args.split,
        gap_fill=False,
        iou_threshold=args.iou_threshold,
        return_detailed=False,
    )

    # ------------------------------------------------------------------
    # 3. SAM2 contribution (merge proposals per image)
    # ------------------------------------------------------------------
    sam2_dir = Path(args.sam2_dir)
    sam2_metrics = None
    if sam2_dir.exists() and any(sam2_dir.glob("*.pt")):
        if image_stems:
            print(f"  Computing SAM2 recall from {sam2_dir} …")
            sam2_metrics = _compute_sam2_metrics(
                all_gt_boxes, all_pred_boxes, image_stems,
                sam2_dir, iou_threshold=args.iou_threshold,
            )
            if sam2_metrics is None:
                print("  No SAM2 proposals matched val image stems — skipping SAM2 bar.")
            else:
                print(f"  SAM2+Stage1 recall={sam2_metrics['recall']:.4f}  "
                      f"precision={sam2_metrics['precision']:.4f}")
        else:
            print("  WARNING: image_stems empty — cannot match SAM2 proposals.")
    else:
        print(f"  SAM2 proposals not found at {sam2_dir} — skipping.")

    # ------------------------------------------------------------------
    # 4. Generate plots
    # ------------------------------------------------------------------
    plots = {}

    print("  Plotting PR + F2 curve …")
    plots["pr_f2_curve"] = plot_pr_f2_curve(
        pr_results, current_thresh=current_thresh,
        out_path=out_dir / "pr_f2_curve.png",
    )

    print("  Plotting threshold sweep …")
    plots["threshold_sweep"] = plot_threshold_sweep(
        pr_results, current_thresh=current_thresh,
        out_path=out_dir / "threshold_sweep.png",
    )

    print("  Plotting column-density FN analysis …")
    plots["column_density_fn"] = plot_column_density_fn(
        all_gt_boxes, all_pred_boxes,
        iou_threshold=args.iou_threshold,
        out_path=out_dir / "column_density_fn.png",
    )

    print("  Plotting SAM2 contribution …")
    plots["sam2_contribution"] = plot_sam2_contribution(
        metrics_no_gapfill, metrics,
        sam2_metrics=sam2_metrics,
        out_path=out_dir / "sam2_contribution.png",
    )

    print("  Plotting learning curves …")
    lc_path = plot_learning_curves(
        log_file=args.log,
        out_path=out_dir / "learning_curves.png",
    )
    if lc_path:
        plots["learning_curves"] = lc_path

    if args.optuna_db:
        db = Path(args.optuna_db)
        if db.exists():
            print(f"  Building Optuna plots from {db} …")
            _build_optuna_plots(db, out_dir)
        else:
            print(f"  WARNING: --optuna-db not found: {db}")

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Stage 1 analysis complete")
    print(f"  recall     (gap-fill) : {metrics.get('recall', float('nan')):.4f}")
    print(f"  precision  (gap-fill) : {metrics.get('precision', float('nan')):.4f}")
    print(f"  f1         (gap-fill) : {metrics.get('f1', float('nan')):.4f}")
    print(f"  recall    (no gap-fill): {metrics_no_gapfill.get('recall', float('nan')):.4f}")
    if sam2_metrics:
        print(f"  recall  (+SAM2)       : {sam2_metrics['recall']:.4f}")
    if pr_results:
        import numpy as np
        f2s = [(1 + 4) * r[1] * r[2] / (4 * r[1] + r[2] + 1e-8) for r in pr_results]
        best_f2_idx = int(np.argmax(f2s))
        print(f"  best F2={f2s[best_f2_idx]:.4f} @ thresh={pr_results[best_f2_idx][0]:.2f}")
    print("Files written:")
    for k, p in plots.items():
        if p:
            print(f"  [{k}]  {p}")
    print("=" * 60)


if __name__ == "__main__":
    main()
