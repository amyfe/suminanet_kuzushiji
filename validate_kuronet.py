"""Standalone validation script for KuroNetRecognizer.

Runs the full validation pipeline on a saved checkpoint and prints:
  - Loss breakdown (char, box, delta, score)
  - Proposal quality (IoU+, coverage, proposals/img, GT/img)
  - Per-ROI classification accuracy (top-1, top-5)
  - Assembled transcription CER (argmax predictions in reading order)
  - Top predicted characters and most common confusion pairs
  - Prediction examples (pred vs GT)

Usage:
    python validate_kuronet.py
    python validate_kuronet.py --ckpt checkpoints/kuronet_recognizer/kuronet_best.pt
    python validate_kuronet.py --split val --batches 50
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch

from config import (
    CHECKPOINT_DIR,
    DATA_DIR,
    DEVICE,
    KURONET_CER_SCORE_THRESH,
    KURONET_CHECKPOINT_DIR,
    KURONET_PREDICTION_SAMPLES,
    STAGE2_REFINE_NEG_IOU,
    STAGE2_REFINE_POS_IOU,
)
from train_kuronet import (
    _compute_assembled_cer,
    _edit_distance,
    _ids_to_text,
    _top_k_accuracy,
    build_dataloaders,
    build_kuronet_model,
    compute_kuronet_loss,
    load_vocab,
)
from utils.stage2_targets import build_refinement_targets
from utils.training_helpers.helper_stage2 import (
    _normalize_orientation_label,
    reorder_by_sort_indices,
)


def _load_kuronet_weights(model: torch.nn.Module, ckpt_path: Path) -> int:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
        epoch = int(ckpt.get("epoch", -1))
    else:
        state = ckpt
        epoch = -1
    model.load_state_dict(state, strict=True)
    return epoch


def run_validation(
    ckpt_path: Path,
    max_batches: int | None,
    split: str = "val",
) -> None:
    vocab = load_vocab()
    _, val_loader = build_dataloaders(vocab)

    model = build_kuronet_model(vocab=vocab)
    epoch = _load_kuronet_weights(model, ckpt_path)
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path}  (epoch={epoch})")
    print("=" * 70)

    total_loss   = 0.0
    loss_parts   = {"loss_char": 0., "loss_box": 0., "loss_delta": 0., "loss_score": 0.}
    top1_sum     = 0.0
    top5_sum     = 0.0
    cer_sum      = 0.0
    iou_sum      = 0.0
    cov_sum      = 0.0
    props_sum    = 0.0
    gt_sum       = 0.0
    pos_sum      = 0.0
    n_batches    = 0
    n_images_tot = 0

    error_counter: Counter = Counter()
    pred_counter:  Counter = Counter()
    examples: list[dict]   = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images         = batch["image"].to(DEVICE, non_blocking=True)
            boxes_list     = [b.to(DEVICE, dtype=torch.float32) for b in batch["boxes"]]
            gt_labels_list = [l.to(DEVICE, dtype=torch.long) for l in batch["labels"]]
            orientations   = [
                _normalize_orientation_label(o) for o in batch["orientations"]
            ]

            outputs = model(images, orientations)

            refine_targets = build_refinement_targets(
                coarse_boxes=outputs["roi_boxes"],
                roi_mask=outputs["roi_mask"],
                gt_boxes_list=boxes_list,
                gt_labels_list=gt_labels_list,
                pos_iou_thresh=STAGE2_REFINE_POS_IOU,
                neg_iou_thresh=STAGE2_REFINE_NEG_IOU,
            )

            loss, parts = compute_kuronet_loss(outputs, refine_targets)
            total_loss += float(loss.item())
            for k in loss_parts:
                loss_parts[k] += parts[k]

            sort_indices = outputs.get("sort_indices", None)
            gt_labels    = refine_targets["matched_gt_labels"]
            pos_mask     = refine_targets["refine_pos_mask"]

            if sort_indices is not None:
                gt_labels_s = reorder_by_sort_indices(gt_labels, sort_indices)
                pos_mask_s  = reorder_by_sort_indices(pos_mask.long(), sort_indices).bool()
            else:
                gt_labels_s = gt_labels
                pos_mask_s  = pos_mask

            top1_sum += _top_k_accuracy(outputs["char_logits"], gt_labels_s, pos_mask_s, k=1)
            top5_sum += _top_k_accuracy(outputs["char_logits"], gt_labels_s, pos_mask_s, k=5)
            cer_sum  += _compute_assembled_cer(outputs, gt_labels_s, pos_mask_s, vocab,
                                              score_thresh=KURONET_CER_SCORE_THRESH)

            bsz = images.size(0)
            n_images_tot += bsz
            for b in range(bsz):
                n_gt   = boxes_list[b].size(0)
                n_pos  = int(refine_targets["refine_pos_mask"][b].sum().item())
                n_prop = int(outputs["roi_mask"][b].sum().item())
                cov_sum   += float(n_pos) / max(1, n_gt)
                pos_sum   += n_pos
                props_sum += n_prop
                gt_sum    += n_gt

            iou_b = refine_targets["matched_iou"]
            pm_b  = refine_targets["refine_pos_mask"]
            if pm_b.any():
                iou_sum += float(iou_b[pm_b].mean().item())

            # Confusion analysis on first sample
            valid_b = pos_mask_s[0] & gt_labels_s[0].ne(-1)
            if valid_b.any():
                pred_ids = outputs["char_logits"][0, valid_b].argmax(dim=-1).tolist()
                true_ids = gt_labels_s[0][valid_b].tolist()
                for pid, tid in zip(pred_ids, true_ids):
                    pred_counter[pid] += 1
                    if pid != tid:
                        error_counter[(tid, pid)] += 1

            # Prediction examples
            if len(examples) < KURONET_PREDICTION_SAMPLES:
                valid_pred = outputs["ordered_mask"][0]
                pred_ids_ex = outputs["char_logits"][0, valid_pred].argmax(dim=-1).tolist()
                pred_text = _ids_to_text(pred_ids_ex, vocab)

                valid_gt = pos_mask_s[0] & gt_labels_s[0].ne(-1)
                gt_ids_ex = gt_labels_s[0][valid_gt].tolist()
                gt_text = _ids_to_text(gt_ids_ex, vocab)
                examples.append({"pred": pred_text, "gt": gt_text})

            n_batches += 1

    if n_batches == 0:
        print("No batches evaluated.")
        return

    avg = lambda x: x / n_batches

    # -----------------------------------------------------------------------
    # Print results
    # -----------------------------------------------------------------------

    print("LOSS SUMMARY")
    print(
        f"  total={avg(total_loss):.4f}"
        f"  char={avg(loss_parts['loss_char']):.4f}"
        f"  box={avg(loss_parts['loss_box']):.4f}"
        f"  delta={avg(loss_parts['loss_delta']):.4f}"
        f"  score={avg(loss_parts['loss_score']):.4f}"
    )

    avg_props = props_sum / max(1, n_images_tot)
    avg_pos   = pos_sum   / max(1, n_images_tot)
    avg_gt    = gt_sum    / max(1, n_images_tot)
    det_recall    = avg_pos / max(1e-6, avg_gt)
    det_precision = avg_pos / max(1e-6, avg_props)
    det_f1        = 2 * det_precision * det_recall / max(1e-6, det_precision + det_recall)

    print("\nPROPOSAL SUMMARY")
    print(
        f"  avg_proposals/img={avg_props:.1f}"
        f"  avg_pos/img={avg_pos:.1f}"
        f"  avg_gt/img={avg_gt:.1f}"
        f"  coverage(recall)={det_recall:.4f}"
        f"  precision={det_precision:.4f}"
        f"  F1={det_f1:.4f}"
        f"  mean_IoU+={avg(iou_sum):.4f}"
    )

    print("\nCLASSIFICATION SUMMARY")
    print(
        f"  top1={avg(top1_sum):.4f}"
        f"  top5={avg(top5_sum):.4f}"
        f"  assembled_CER={avg(cer_sum):.4f}"
        f"  (lower CER = better; 0 = perfect)"
    )

    top_preds = [
        {"token": _ids_to_text([tid], vocab), "count": cnt}
        for tid, cnt in pred_counter.most_common(8)
    ]
    top_errors = [
        {
            "gt":   _ids_to_text([gt], vocab),
            "pred": _ids_to_text([pr], vocab),
            "count": cnt,
        }
        for (gt, pr), cnt in error_counter.most_common(8)
    ]

    print(f"\nTop predicted tokens:  {top_preds}")
    print(f"Top confusion pairs:   {top_errors}")

    print("\nPREDICTION EXAMPLES")
    for i, ex in enumerate(examples):
        print(f"  [sample={i}] PRED: {ex['pred'][:150]}")
        print(f"           GT:   {ex['gt'][:150]}")


def main():
    parser = argparse.ArgumentParser(description="Validate KuroNetRecognizer")
    parser.add_argument(
        "--ckpt",
        type=str,
        default=str(KURONET_CHECKPOINT_DIR / "kuronet_best.pt"),
        help="Path to KuroNet checkpoint",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="Dataset split to evaluate on",
    )
    parser.add_argument(
        "--batches",
        type=int,
        default=None,
        help="Max number of batches (None = full split)",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    run_validation(
        ckpt_path=ckpt_path,
        max_batches=args.batches,
        split=args.split,
    )


if __name__ == "__main__":
    main()
