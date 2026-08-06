"""Unified Stage 2 result analysis for the master's thesis.

Loads a trained KuroNet checkpoint, runs detailed validation, and produces:
  1. confusion_matrix.png    — top-N character confusion heatmap + top-20 pairs
  2. cer_by_script.png       — CER breakdown: hiragana / katakana / kanji / rare kanji
  3. topk_errors.png         — fraction of wrong predictions where GT was in top-K
  4. per_page_cer.png        — distribution of CER across val images
  5. learning_curves.png     — train/val loss + val CER/top-1 vs epoch (skipped if --log not found)

Optionally, with --image:
  6. confidence_vis.png      — boxes colored by confidence (<10% red, <50% orange, >=50% green)

Usage:
    python scripts/analyse_stage2.py \\
        --ckpt checkpoints/kuronet_recognizer/kuronet_best.pt \\
        [--out-dir results/stage2_analysis] \\
        [--split val] \\
        [--top-n 40] \\
        [--log logs/train_stage2_kuronet.log] \\
        [--image path/to/page.jpg]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

import train_stage2_kuronet as train_module
from config import CHECKPOINT_DIR, DEVICE, KURONET_RARE_CHAR_THRESH
from utils.training_helpers.helper_stage2 import _load_compatible_state_dict
from visualization.stage2 import (
    draw_confidence_boxes,
    plot_cer_by_script,
    plot_confusion_matrix,
    plot_learning_curves,
    plot_per_page_cer,
    plot_topk_errors,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 2 thesis analysis — all plots in one command")
    p.add_argument("--ckpt", default=str(CHECKPOINT_DIR / "kuronet_recognizer" / "kuronet_best.pt"),
                   help="KuroNet checkpoint (.pt)")
    p.add_argument("--out-dir", default="results/stage2_analysis",
                   help="Directory to save output PNGs")
    p.add_argument("--split", default="val", choices=["val", "test"],
                   help="Dataset split to evaluate on")
    p.add_argument("--top-n", type=int, default=40,
                   help="Number of top classes for the confusion matrix heatmap")
    p.add_argument("--log", default="logs/train_stage2_kuronet.log",
                   help="Training log file for learning curves (skipped if not found)")
    p.add_argument("--image", default=None,
                   help="Optional: path to a page image for confidence visualization")
    return p.parse_args()


def _load_model(ckpt_path: str | Path, vocab) -> "train_module.KuroNetRecognizer":
    model = train_module.build_kuronet_model(vocab, load_stage1_weights=False)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    state = ckpt.get("model_state_dict", ckpt)
    _load_compatible_state_dict(model, state)
    epoch = ckpt.get("epoch", "?")
    print(f"Loaded checkpoint {ckpt_path}  (epoch={epoch})")
    model.eval()
    return model


def _print_summary(metrics: dict) -> None:
    print("\n" + "=" * 60)
    print("Stage 2 Validation Summary")
    print("=" * 60)
    rows = [
        ("top1_acc",               "Top-1 accuracy"),
        ("top5_acc",               "Top-5 accuracy"),
        ("assembled_cer",          "Assembled CER"),
        ("coverage",               "Coverage (IoU>0.45)"),
        ("stage1_equiv_recall",    "Stage-1 equiv recall (IoU>0.2)"),
        ("det_precision",          "Detection precision"),
        ("det_f1",                 "Detection F1"),
        ("avg_iou_on_positives",   "Avg IoU on positives"),
        ("avg_proposals_per_image","Proposals / image"),
        ("avg_gt_per_image",       "GT chars / image"),
    ]
    for key, label in rows:
        val = metrics.get(key)
        if val is not None:
            print(f"  {label:<32} {val:.4f}")

    print("\nPer-script accuracy:")
    for s in ["hiragana", "katakana", "kanji", "other"]:
        acc = metrics.get(f"acc_{s}", 0.0)
        n   = metrics.get(f"n_{s}", 0)
        print(f"  {s:<12} {acc:.4f}  (n={n:,})")

    cer_list = metrics.get("per_image_cer", [])
    if cer_list:
        import numpy as np
        arr = np.array(cer_list)
        print(f"\nPer-page CER: mean={arr.mean():.4f}  median={float(np.median(arr)):.4f}"
              f"  P90={float(np.percentile(arr, 90)):.4f}  n={len(arr)}")

    print("=" * 60 + "\n")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # Build vocab, model, data
    # ----------------------------------------------------------------
    print("Loading vocab...")
    vocab = train_module.load_vocab()

    print("Building model...")
    model = _load_model(args.ckpt, vocab)

    print("Building dataloader...")
    _, val_loader = train_module.build_dataloaders(vocab)

    # ----------------------------------------------------------------
    # Run detailed validation
    # ----------------------------------------------------------------
    print("Running validation (return_detailed=True) ...")
    metrics = train_module.validate_kuronet(
        model=model,
        val_loader=val_loader,
        vocab=vocab,
        return_detailed=True,
    )

    _print_summary(metrics)

    error_counter    = metrics["error_counter"]
    gt_total_counter = metrics["gt_total_counter"]
    per_image_cer    = metrics["per_image_cer"]
    topk_in_gt       = metrics["topk_in_gt"]

    # ----------------------------------------------------------------
    # Generate plots
    # ----------------------------------------------------------------
    written: list[Path] = []

    print("Generating confusion matrix...")
    p = plot_confusion_matrix(
        error_counter, gt_total_counter, vocab,
        top_n=args.top_n,
        out_path=out_dir / "confusion_matrix.png",
    )
    written.append(p)

    print("Generating CER by script...")
    p = plot_cer_by_script(
        error_counter, gt_total_counter, vocab,
        rare_thresh=KURONET_RARE_CHAR_THRESH,
        out_path=out_dir / "cer_by_script.png",
    )
    written.append(p)

    print("Generating top-K error analysis...")
    p = plot_topk_errors(
        topk_in_gt,
        out_path=out_dir / "topk_errors.png",
    )
    written.append(p)

    print("Generating per-page CER histogram...")
    p = plot_per_page_cer(
        per_image_cer,
        out_path=out_dir / "per_page_cer.png",
    )
    written.append(p)

    print("Generating learning curves...")
    lc_path = plot_learning_curves(
        log_file=args.log,
        out_path=out_dir / "learning_curves.png",
    )
    if lc_path:
        written.append(lc_path)

    # ----------------------------------------------------------------
    # Optional: confidence visualization on a single image
    # ----------------------------------------------------------------
    if args.image:
        from model.translation.infer import load_image, run_inference
        print(f"Running inference on {args.image} for confidence visualization...")
        img_tensor, orig_size, scale, pad = load_image(args.image)
        result = run_inference(model, img_tensor, vocab)
        p = draw_confidence_boxes(
            image_path=args.image,
            chars=result["chars"],
            out_path=out_dir / "confidence_vis.png",
        )
        written.append(p)
        print(f"  → {p}  ({result['n_chars']} characters detected)")

    print("\nOutput files:")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
