"""Standalone validation script for KuroNetRecognizer.

Runs the full validation pipeline on a saved checkpoint and prints:
  - Loss breakdown (char, box, delta, score)
  - Proposal quality (IoU+, coverage, proposals/img, GT/img)
  - Per-ROI classification accuracy (top-1, top-5)
  - Assembled transcription CER (argmax predictions in reading order)
  - Confusion matrix (top confused classes, row-normalized, saved as PNG)
  - Per-class error rate table (worst-recalled symbols)
  - Prediction examples (pred vs GT)

Usage:
    python validate_kuronet.py
    python validate_kuronet.py --ckpt checkpoints/kuronet_recognizer/kuronet_best.pt
    python validate_kuronet.py --split val --batches 50
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from datetime import datetime
import os

import numpy as np
import matplotlib.pyplot as plt
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
from train_stage2_kuronet import (
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
    _load_compatible_state_dict,
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
    _load_compatible_state_dict(model, state)
    return epoch


# -------------------------
# Confusion matrix plot
# -------------------------

_CJK_FONT_PATH = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"


def _char_script_type(ch: str) -> str:
    """Categorize a character by script: hiragana, katakana, kanji, digit, latin, or other."""
    if not ch:
        return "other"
    cp = ord(ch[0])
    if 0x3041 <= cp <= 0x3096:
        return "hiragana"
    if 0x30A0 <= cp <= 0x30FF:
        return "katakana"
    if (0x4E00 <= cp <= 0x9FFF) or (0x3400 <= cp <= 0x4DBF) or (0x20000 <= cp <= 0x2A6DF):
        return "kanji"
    if (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A):
        return "latin"
    if 0x0030 <= cp <= 0x0039:
        return "digit"
    return "other"


def _is_kana_script_mixup(gt_ch: str, pred_ch: str) -> bool:
    """True when gt and pred are the same phoneme written in different kana scripts.

    Hiragana (U+3041–U+3096) and katakana (U+30A1–U+30F6) share a consistent
    offset of 0x60 for corresponding characters.  Iteration marks are a special case.
    """
    if not gt_ch or not pred_ch:
        return False
    g, p = ord(gt_ch[0]), ord(pred_ch[0])
    if 0x3041 <= g <= 0x3096 and 0x30A1 <= p <= 0x30F6 and g + 0x60 == p:
        return True
    if 0x30A1 <= g <= 0x30F6 and 0x3041 <= p <= 0x3096 and g - 0x60 == p:
        return True
    # Iteration marks: ゝ(309D)↔ヽ(30FD), ゞ(309E)↔ヾ(30FE)
    _iter_pairs = {0x309D: 0x30FD, 0x309E: 0x30FE, 0x30FD: 0x309D, 0x30FE: 0x309E}
    return _iter_pairs.get(g) == p


def _cjk_font_prop(size: float):
    """Return a FontProperties for CJK tick labels, or None if font unavailable."""
    try:
        import matplotlib.font_manager as fm
        from pathlib import Path as _P
        if _P(_CJK_FONT_PATH).exists():
            return fm.FontProperties(fname=_CJK_FONT_PATH, size=size)
    except Exception:
        pass
    return None


def plot_confusion_matrix(
    error_counter: Counter,
    gt_total_counter: Counter,
    vocab,
    out_path: Path,
    top_n: int = 25,
) -> None:
    """Save a row-normalised confusion matrix for the top-N most error-prone classes."""
    if not error_counter:
        print("No confusion data to plot.")
        return

    # Select the top_n GT classes with the most cumulative errors
    gt_error_totals: Counter = Counter()
    for (gt, pr), cnt in error_counter.items():
        gt_error_totals[gt] += cnt

    top_gt = [c for c, _ in gt_error_totals.most_common(top_n)]

    # Include their most-confused predicted classes too
    confused_classes: set = set(top_gt)
    for gt in top_gt:
        for (g, p), _ in error_counter.most_common():
            if g == gt and p not in confused_classes:
                confused_classes.add(p)
                break

    class_list = sorted(confused_classes)
    n = len(class_list)
    if n == 0:
        return

    idx_map = {c: i for i, c in enumerate(class_list)}

    mat = np.zeros((n, n), dtype=np.int64)
    for (gt, pr), cnt in error_counter.items():
        if gt in idx_map and pr in idx_map:
            mat[idx_map[gt], idx_map[pr]] += cnt
    for c in class_list:
        i = idx_map[c]
        total_gt = gt_total_counter.get(c, 0)
        total_errors = gt_error_totals.get(c, 0)
        mat[i, i] = max(0, total_gt - total_errors)

    row_sums = mat.sum(axis=1, keepdims=True).clip(min=1)
    mat_norm = mat / row_sums

    labels = [_ids_to_text([c], vocab) for c in class_list]

    # Scale cell size down for large matrices so the figure stays manageable.
    cell_size = max(0.4, min(0.65, 18.0 / max(n, 1)))
    fontsize  = max(8,   min(16,  int(cell_size * 72 * 0.55)))
    fig_w = n * cell_size + 2.5   # extra space for colorbar + y-labels
    fig_h = n * cell_size + 1.5

    cjk = _cjk_font_prop(fontsize)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(mat_norm, cmap="Reds", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))

    if cjk is not None:
        # Set tick labels with CJK font via individual Text objects
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(labels)
        for txt in ax.get_xticklabels():
            txt.set_fontproperties(cjk)
        for txt in ax.get_yticklabels():
            txt.set_fontproperties(cjk)
    else:
        # Fallback: show Unicode codepoints (readable without CJK font)
        cp_labels = []
        for lbl in labels:
            if lbl:
                cp_labels.append("U+{:04X}".format(ord(lbl[0])))
            else:
                cp_labels.append("?")
        ax.set_xticklabels(cp_labels, rotation=90, fontsize=max(7, fontsize - 2))
        ax.set_yticklabels(cp_labels, fontsize=max(7, fontsize - 2))

    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Ground Truth", fontsize=11)
    ax.set_title(f"Confusion Matrix — top {n} error-prone classes (row-normalised)", fontsize=11)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved: {out_path}")


# -------------------------
# Per-class error rate table
# -------------------------

def print_per_class_error_rates(
    error_counter: Counter,
    gt_total_counter: Counter,
    vocab,
    top_k: int = 20,
) -> list[dict]:
    """Print and return worst-recalled classes sorted by error rate."""
    gt_error_totals: Counter = Counter()
    for (gt, _), cnt in error_counter.items():
        gt_error_totals[gt] += cnt

    rows = []
    for cls_id, err_cnt in gt_error_totals.most_common():
        total = gt_total_counter.get(cls_id, 0)
        if total == 0:
            continue
        err_rate = err_cnt / total
        # Top confusees for this class
        confusees = [
            (_ids_to_text([pr], vocab), cnt)
            for (gt, pr), cnt in error_counter.most_common()
            if gt == cls_id
        ][:3]
        rows.append({
            "char": _ids_to_text([cls_id], vocab),
            "id": cls_id,
            "errors": err_cnt,
            "total_gt": total,
            "error_rate": round(err_rate, 4),
            "top_confusees": confusees,
        })

    rows.sort(key=lambda r: r["error_rate"], reverse=True)

    print(f"\n{'='*70}")
    print(f"WORST-RECALLED CLASSES (top {top_k})")
    print(f"{'='*70}")
    print(f"  {'Char':<8} {'Errors':>7} {'Total':>7} {'ErrRate':>8}  Top confusees")
    print(f"  {'-'*8} {'-'*7} {'-'*7} {'-'*8}  {'-'*30}")
    for row in rows[:top_k]:
        confusee_str = ", ".join(f"{ch}×{cnt}" for ch, cnt in row["top_confusees"])
        print(f"  {row['char']:<8} {row['errors']:>7} {row['total_gt']:>7} {row['error_rate']:>8.3f}  {confusee_str}")

    return rows


def save_per_class_errors_csv(
    rows: list[dict],
    out_path: Path,
    vocab,
) -> dict:
    """Save ALL per-class error statistics to CSV and return a per-script-type breakdown."""
    type_stats: dict[str, dict] = {}

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "char", "unicode_hex", "script_type",
            "total_gt", "errors", "error_rate",
            "confusee_1", "confusee_1_count",
            "confusee_2", "confusee_2_count",
            "confusee_3", "confusee_3_count",
        ])
        for row in rows:
            ch = row["char"]
            stype = _char_script_type(ch)
            uni_hex = f"U+{ord(ch[0]):04X}" if ch else "?"
            confusees = row["top_confusees"]
            c1, n1 = confusees[0] if len(confusees) > 0 else ("", 0)
            c2, n2 = confusees[1] if len(confusees) > 1 else ("", 0)
            c3, n3 = confusees[2] if len(confusees) > 2 else ("", 0)
            writer.writerow([
                ch, uni_hex, stype,
                row["total_gt"], row["errors"], f"{row['error_rate']:.4f}",
                c1, n1, c2, n2, c3, n3,
            ])
            if stype not in type_stats:
                type_stats[stype] = {"classes_with_errors": 0, "total_gt": 0, "errors": 0}
            type_stats[stype]["classes_with_errors"] += 1
            type_stats[stype]["total_gt"] += row["total_gt"]
            type_stats[stype]["errors"] += row["errors"]

    print(f"\nPER-CLASS ERRORS CSV: {out_path}  ({len(rows)} classes with ≥1 error)")
    print("\nERROR BREAKDOWN BY SCRIPT TYPE")
    print(f"  {'Type':<12} {'Classes':>8} {'GT tokens':>10} {'Errors':>8} {'Error %':>8}")
    print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*8} {'-'*8}")
    for stype in ["hiragana", "katakana", "kanji", "latin", "digit", "other"]:
        if stype in type_stats:
            st = type_stats[stype]
            er = st["errors"] / max(1, st["total_gt"]) * 100
            print(
                f"  {stype:<12} {st['classes_with_errors']:>8} "
                f"{st['total_gt']:>10} {st['errors']:>8} {er:>7.1f}%"
            )

    return type_stats


# -------------------------
# Main validation
# -------------------------

def run_validation(
    ckpt_path: Path,
    max_batches: int | None,
    split: str = "val",
) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    resolved_job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("JOB_ID")
    run_tag = f"{timestamp}_job{resolved_job_id}" if resolved_job_id else timestamp
    out_dir = KURONET_CHECKPOINT_DIR / "validation" / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    vocab = load_vocab()
    _, val_loader = build_dataloaders(vocab)

    model = build_kuronet_model(vocab=vocab)
    epoch = _load_kuronet_weights(model, ckpt_path)
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path}  (epoch={epoch})")
    print("=" * 70)

    bg_id = vocab.bg_id if hasattr(vocab, "bg_id") and vocab.BG_TOKEN in vocab.char2id else None

    total_loss        = 0.0
    loss_parts        = {"loss_char": 0., "loss_box": 0., "loss_delta": 0., "loss_score": 0., "loss_bg": 0.}
    top1_sum          = 0.0
    top5_sum          = 0.0
    cer_sum           = 0.0
    iou_sum           = 0.0
    cov_sum           = 0.0
    props_sum         = 0.0
    gt_sum            = 0.0
    pos_sum           = 0.0
    ordering_viol_sum = 0.0
    ordering_viol_n   = 0
    n_batches         = 0
    n_images_tot      = 0

    # Confusion tracking — accumulated over ALL images in every batch
    error_counter:       Counter = Counter()  # (gt_id, pred_id) -> count, all mismatches
    kana_mixup_counter:  Counter = Counter()  # (gt_id, pred_id) -> count, same phoneme wrong script
    pred_counter:        Counter = Counter()  # pred_id -> count
    gt_total_counter:    Counter = Counter()  # gt_id -> total occurrences in positives

    examples: list[dict] = []

    # Debug: IoU distribution for positive proposals where GT=し but model predicts BG.
    # High IoU → BG head over-fires; near-threshold IoU → borderline matching.
    _shi_bg_ious: list[float] = []
    _shi_id = vocab.char2id.get("し") if bg_id is not None else None

    # Debug: BG-vs-column breakdown.
    # For every positive proposal, classify it as (isolated / in-column) × (pred=BG / pred=char).
    # Expected healthy state: BG predictions cluster on isolated proposals.
    # Problem state: BG predictions appear inside text columns (real chars misclassified).
    _bg_col = {"bg_isolated": 0, "bg_in_column": 0,
               "char_isolated": 0, "char_in_column": 0}

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

            loss, parts = compute_kuronet_loss(outputs, refine_targets, bg_id=bg_id)
            total_loss += float(loss.item())
            for k in loss_parts:
                loss_parts[k] += parts.get(k, 0.0)

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

            ordering_diag = outputs.get("ordering_diagnostics")
            if ordering_diag is not None:
                viol_b   = ordering_diag["primary_violation_fraction"]  # (B,)
                valid_cb = ordering_diag["valid_counts"]                 # (B,)
                for b in range(bsz):
                    if int(valid_cb[b].item()) > 1:
                        ordering_viol_sum += float(viol_b[b].item())
                        ordering_viol_n   += 1

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

            # Confusion accumulation — loop over ALL images in the batch (was only [0])
            for b in range(bsz):
                valid_b = pos_mask_s[b] & gt_labels_s[b].ne(-1)
                if valid_b.any():
                    pred_ids = outputs["char_logits"][b, valid_b].argmax(dim=-1).tolist()
                    true_ids = gt_labels_s[b][valid_b].tolist()
                    for pid, tid in zip(pred_ids, true_ids):
                        pred_counter[pid] += 1
                        gt_total_counter[tid] += 1
                        if pid != tid:
                            error_counter[(tid, pid)] += 1
                            if _is_kana_script_mixup(
                                _ids_to_text([tid], vocab),
                                _ids_to_text([pid], vocab),
                            ):
                                kana_mixup_counter[(tid, pid)] += 1

            # Debug: し→BG IoU audit (unsorted space, matches refine_targets layout)
            if _shi_id is not None:
                pm_raw  = refine_targets["refine_pos_mask"]   # (B, T) bool
                lbl_raw = refine_targets["matched_gt_labels"]  # (B, T)
                iou_raw = refine_targets["matched_iou"]        # (B, T)
                pred_raw = outputs["char_logits"].argmax(dim=-1)  # (B, T)
                for b in range(bsz):
                    hit = pm_raw[b] & (lbl_raw[b] == _shi_id) & (pred_raw[b] == bg_id)
                    _shi_bg_ious.extend(iou_raw[b][hit].tolist())

            # Debug: BG-vs-column breakdown (sorted space, matches isolation_mask layout).
            # isolation_mask[b][t]=True means proposal t is NOT part of a text column.
            iso_mask = outputs.get("isolation_mask")  # (B, T) bool or None
            if bg_id is not None and iso_mask is not None:
                for b in range(bsz):
                    valid = pos_mask_s[b]              # (T,) positive proposals, sorted order
                    if not valid.any():
                        continue
                    preds_b = outputs["char_logits"][b, valid].argmax(dim=-1)  # (N_pos,)
                    iso_b   = iso_mask[b][valid]                               # (N_pos,)
                    is_bg   = preds_b == bg_id
                    _bg_col["bg_isolated"]    += int((is_bg  &  iso_b).sum())
                    _bg_col["bg_in_column"]   += int((is_bg  & ~iso_b).sum())
                    _bg_col["char_isolated"]  += int((~is_bg &  iso_b).sum())
                    _bg_col["char_in_column"] += int((~is_bg & ~iso_b).sum())

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
        f"  bg={avg(loss_parts['loss_bg']):.4f}"
        f"  delta={avg(loss_parts['loss_delta']):.4f}"
        f"  score={avg(loss_parts['loss_score']):.4f}"
    )

    avg_props = props_sum / max(1, n_images_tot)
    avg_pos   = pos_sum   / max(1, n_images_tot)
    avg_gt    = gt_sum    / max(1, n_images_tot)
    det_recall    = avg_pos / max(1e-6, avg_gt)
    det_precision = avg_pos / max(1e-6, avg_props)
    det_f1        = 2 * det_precision * det_recall / max(1e-6, det_precision + det_recall)

    mean_ordering_viol = ordering_viol_sum / max(1, ordering_viol_n)

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
    print(
        f"  ordering_violation_rate={mean_ordering_viol:.4f}"
        f"  (fraction of adjacent ROI pairs out of reading order; 0=perfect)"
    )

    print("\nCLASSIFICATION SUMMARY")
    print(
        f"  top1={avg(top1_sum):.4f}"
        f"  top5={avg(top5_sum):.4f}"
        f"  assembled_CER={avg(cer_sum):.4f}"
        f"  (lower CER = better; 0 = perfect)"
    )
    print(f"  images_evaluated={n_images_tot}  batches={n_batches}")

    top_preds = [
        {"token": _ids_to_text([tid], vocab), "count": cnt}
        for tid, cnt in pred_counter.most_common(10)
    ]
    top_errors = [
        {
            "gt":   _ids_to_text([gt], vocab),
            "pred": _ids_to_text([pr], vocab),
            "count": cnt,
        }
        for (gt, pr), cnt in error_counter.most_common(10)
    ]

    total_errors     = sum(error_counter.values())
    total_kana_mixup = sum(kana_mixup_counter.values())
    total_hard       = total_errors - total_kana_mixup

    hard_error_counter = Counter({
        k: v for k, v in error_counter.items()
        if not _is_kana_script_mixup(_ids_to_text([k[0]], vocab), _ids_to_text([k[1]], vocab))
    })
    top_hard_errors = [
        {"gt": _ids_to_text([gt], vocab), "pred": _ids_to_text([pr], vocab), "count": cnt}
        for (gt, pr), cnt in hard_error_counter.most_common(10)
    ]

    print(f"\nTop predicted tokens:  {top_preds}")
    print(f"Top confusion pairs:   {top_errors}")
    print(f"Top hard errors (excl. kana-script mixup):  {top_hard_errors}")

    if total_errors > 0:
        top_kana_mixup = [
            {"gt": _ids_to_text([gt], vocab), "pred": _ids_to_text([pr], vocab), "count": cnt}
            for (gt, pr), cnt in kana_mixup_counter.most_common(10)
        ]
        print(f"\nERROR SPLIT  (classification errors on detected proposals)")
        print(
            f"  total={total_errors}"
            f"  kana_script_mixup={total_kana_mixup} ({100*total_kana_mixup/total_errors:.1f}%)"
            f"  hard={total_hard} ({100*total_hard/total_errors:.1f}%)"
        )
        print(f"  (kana_script_mixup = same phoneme, wrong hiragana/katakana script — acceptable for translation)")
        print(f"Top kana-mixup pairs:  {top_kana_mixup}")

    if _shi_bg_ious:
        arr = sorted(_shi_bg_ious)
        n = len(arr)
        buckets = [(0.45, 0.55), (0.55, 0.65), (0.65, 0.75), (0.75, 0.85), (0.85, 1.01)]
        hist = {f"{lo:.2f}-{hi:.2f}": sum(lo <= v < hi for v in arr) for lo, hi in buckets}
        print(f"\nDEBUG し→BG IoU audit  (n={n} positive proposals where GT=し, pred=<BG>)")
        print(f"  IoU histogram: {hist}")
        print(f"  median={arr[n // 2]:.3f}  mean={sum(arr)/n:.3f}  "
              f"p25={arr[n // 4]:.3f}  p75={arr[3 * n // 4]:.3f}")

    total_bg   = _bg_col["bg_isolated"]   + _bg_col["bg_in_column"]
    total_char = _bg_col["char_isolated"] + _bg_col["char_in_column"]
    total_pos  = total_bg + total_char
    if total_pos > 0 and any(_bg_col.values()):
        def _pct(x): return f"{100*x/total_pos:.1f}%"
        print(f"\nDEBUG BG vs column breakdown  (positive proposals, sorted space)")
        print(f"  {'':20s}  {'isolated':>10}  {'in-column':>10}")
        print(f"  {'pred=<BG>':20s}  {_bg_col['bg_isolated']:>10}  {_bg_col['bg_in_column']:>10}"
              f"  ({_pct(_bg_col['bg_isolated'])} / {_pct(_bg_col['bg_in_column'])})")
        print(f"  {'pred=char':20s}  {_bg_col['char_isolated']:>10}  {_bg_col['char_in_column']:>10}"
              f"  ({_pct(_bg_col['char_isolated'])} / {_pct(_bg_col['char_in_column'])})")
        if total_bg > 0:
            bg_iso_rate = _bg_col["bg_isolated"] / total_bg
            print(f"  BG-in-isolated rate: {bg_iso_rate:.3f}  "
                  f"(1.0 = BG only fires on isolated proposals — ideal)")
        print(f"  isolation_mask: None" if iso_mask is None else
              f"  (isolation_mask available — {total_pos} positive ROIs evaluated)")

    print("\nPREDICTION EXAMPLES")
    for i, ex in enumerate(examples):
        print(f"  [sample={i}] PRED: {ex['pred'][:150]}")
        print(f"           GT:   {ex['gt'][:150]}")

    # -----------------------------------------------------------------------
    # Confusion matrix + per-class error rate
    # -----------------------------------------------------------------------
    per_class_rows = print_per_class_error_rates(error_counter, gt_total_counter, vocab)

    type_breakdown = save_per_class_errors_csv(
        rows=per_class_rows,
        out_path=out_dir / "per_class_errors.csv",
        vocab=vocab,
    )

    plot_confusion_matrix(
        error_counter=error_counter,
        gt_total_counter=gt_total_counter,
        vocab=vocab,
        out_path=out_dir / "confusion_matrix.png",
        top_n=40,
    )

    # Hard-errors confusion matrix: excludes kana script mixups (ハ↔は etc.).
    # Focuses on mistakes that actually matter for translation.
    if hard_error_counter:
        plot_confusion_matrix(
            error_counter=hard_error_counter,
            gt_total_counter=gt_total_counter,
            vocab=vocab,
            out_path=out_dir / "confusion_matrix_hard.png",
            top_n=40,
        )

    # Kana-focused confusion matrix: only rows where GT is hiragana or katakana.
    # Zooms in on script-type confusion (ハ↔は etc.) and iteration mark clusters.
    _kana_scripts = {"hiragana", "katakana"}
    kana_error_counter = Counter({
        (gt, pr): cnt for (gt, pr), cnt in error_counter.items()
        if _char_script_type(_ids_to_text([gt], vocab)) in _kana_scripts
    })
    if kana_error_counter:
        plot_confusion_matrix(
            error_counter=kana_error_counter,
            gt_total_counter=gt_total_counter,
            vocab=vocab,
            out_path=out_dir / "confusion_matrix_kana.png",
            top_n=30,
        )

    # -----------------------------------------------------------------------
    # Save summary JSON
    # -----------------------------------------------------------------------
    summary = {
        "checkpoint": str(ckpt_path),
        "epoch": epoch,
        "images_evaluated": n_images_tot,
        "batches": n_batches,
        "loss": {k: avg(v) for k, v in {**{"total": total_loss}, **loss_parts}.items()},
        "top1": avg(top1_sum),
        "top5": avg(top5_sum),
        "assembled_cer": avg(cer_sum),
        "coverage": det_recall,
        "det_precision": det_precision,
        "det_f1": det_f1,
        "ordering_violation_rate": mean_ordering_viol,
        "avg_proposals_per_image": avg_props,
        "avg_gt_per_image": avg_gt,
        "mean_iou_positives": avg(iou_sum),
        "top_confusion_pairs": top_errors,
        "top_hard_errors": top_hard_errors,
        "error_split": {
            "total": total_errors,
            "kana_script_mixup": total_kana_mixup,
            "hard_errors": total_hard,
            "kana_mixup_pct": round(100 * total_kana_mixup / max(1, total_errors), 2),
        },
        "worst_recalled_classes": per_class_rows[:20],
        "error_breakdown_by_script": type_breakdown,
        "total_classes_with_errors": len(per_class_rows),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n✅ metrics saved: {out_dir / 'metrics.json'}")
    print(f"✅ per-class CSV: {out_dir / 'per_class_errors.csv'}")
    print(f"✅ confusion matrix: {out_dir / 'confusion_matrix.png'}")


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
