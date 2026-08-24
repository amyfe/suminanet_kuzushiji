"""Standalone validation script for SuminaNetRecognizer.

Runs the full validation pipeline on a saved checkpoint and prints:
  - Loss breakdown (char, box, delta, score)
  - Proposal quality (IoU+, coverage, proposals/img, GT/img)
  - Per-ROI classification accuracy (top-1, top-5)
  - Assembled transcription CER (argmax predictions in reading order)
  - Confusion matrix (top confused classes, row-normalized, saved as PNG)
  - Per-class error rate table (worst-recalled symbols)
  - Prediction examples (pred vs GT)

Usage:
    python validate_suminanet.py
    python validate_suminanet.py --ckpt checkpoints/suminanet_recognizer/suminanet_best.pt
    python validate_suminanet.py --batches 50
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from functools import partial
from pathlib import Path
from datetime import datetime
import os

# Allow running as `python utils/validation/validate_suminanet.py` (not just
# `python -m utils.validation.validate_suminanet`) — Python only puts this
# script's own directory on sys.path, not the repo root, so the `config`
# import below fails without this.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from config import (
    WEBSITE_CHECKPOINT_DIR,
    DATA_DIR,
    DEVICE,
    EXCLUDE_BOOKS,
    EXCLUDE_PAGES,
    IMAGE_SIZE,
    NUM_WORKERS,
    STAGE2_BATCH_SIZE,
    SUMINANET_CER_SCORE_THRESH,
    SUMINANET_CHECKPOINT_DIR,
    SUMINANET_PREDICTION_SAMPLES,
    STAGE2_REFINE_NEG_IOU,
    STAGE2_REFINE_POS_IOU,
    STAGE2_USE_HUNGARIAN,
)
from train_stage2_suminanet import (
    _compute_assembled_cer,
    _ids_to_text,
    _top_k_accuracy,
    build_suminanet_model,
    compute_suminanet_loss,
    load_vocab,
)
from utils import KuzushijiDataset
from utils.stage2_targets import build_refinement_targets
from utils.training_helpers.helper_stage1 import collate_fn
from utils.training_helpers.helper_stage2 import (
    _load_compatible_state_dict,
    _normalize_orientation_label,
    reorder_by_sort_indices,
)
from visualization.common import cjk_font_prop, savefig
from visualization.stage2 import plot_loss_cer_curve, plot_top1_top5_curve


def _load_suminanet_weights(model: torch.nn.Module, ckpt_path: Path, vocab) -> int:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
        epoch = int(ckpt.get("epoch", -1))
    else:
        state = ckpt
        epoch = -1
    _load_compatible_state_dict(
        model, state,
        ckpt_context_mode=ckpt.get("context_mode"),
        ckpt_vocab_hash=ckpt.get("vocab_hash"),
        current_vocab_hash=vocab.content_hash(),
        ckpt_backbone_type=ckpt.get("backbone_type"),
    )
    return epoch


def _book_id_for_stem(stem: str) -> str:
    """Book/document ID for a validation image, derived from its annotation's
    image_path (first path component) — NOT from splitting the filename
    stem. Some books (e.g. umgy00000) have per-image filenames that don't
    share the directory's prefix, so a naive stem.split("_")[0] heuristic
    would fragment one book into many (see
    scripts/onetime_scripts/prepare_codh_annotations.py for the same
    gotcha during dataset prep)."""
    ann_path = DATA_DIR / "annotations" / f"{stem}.json"
    with open(ann_path, encoding="utf-8") as f:
        data = json.load(f)
    image_path = data.get("image_path", "")
    return Path(image_path).parts[0] if image_path else "unknown"


_VOLUME_SUFFIX_RE = re.compile(r"^([a-zA-Z]+)(\d+)$")


def _canonical_book_title(book_id: str) -> str:
    """Group volume-level scan directories into their parent book title,
    e.g. brsk001..brsk005 (5 volumes of one book) -> "brsk", hnsd001..
    hnsd012 (12 volumes of one book) -> "hnsd". Purely-numeric CODH IDs
    (e.g. 200021925) never match the letter-prefix pattern and pass
    through unchanged, as does any already-singular book (e.g. umgy00000
    -> "umgy", a no-op merge since it's the only volume).

    Verified against the actual assets/data/ layout: 59 raw scan
    directories canonicalize to exactly 44 books. NOTE: the `val`/`test`
    splits are missing one raw book_id entirely (train-only), so
    --split val alone surfaces 43 canonical books; the 44th only appears
    once --split includes train.
    """
    m = _VOLUME_SUFFIX_RE.fullmatch(book_id)
    return m.group(1) if m else book_id


def _build_eval_loader(vocab, split: str) -> DataLoader:
    """Deterministic, non-augmented single-pass loader for a given split —
    unlike train_stage2_suminanet.build_dataloaders()'s train_loader (a
    WeightedRandomSampler with replacement=True, not a clean single pass),
    and unlike passing split="train" with transform=None to KuzushijiDataset
    (which auto-applies ColorJitter for split=="train"). An explicit
    eval-only transform sidesteps both, so this is usable for either split.
    """
    eval_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = KuzushijiDataset(
        DATA_DIR, vocab=vocab, use_sequences=True, resize=IMAGE_SIZE,
        split=split, rare_chars=None, transform=eval_transform,
    )
    return DataLoader(
        dataset, batch_size=STAGE2_BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=partial(collate_fn, pad_id=vocab.pad_id),
        pin_memory=True, prefetch_factor=4, persistent_workers=NUM_WORKERS > 0,
    )


# -------------------------
# Confusion matrix plot
# -------------------------

_MAX_GALLERY_CROPS = 3


def _denormalize_image(image_tensor: torch.Tensor) -> np.ndarray:
    """RGB (H, W, 3) uint8 array from a (C, H, W) tensor normalized with
    ImageNet mean/std (see utils/__init__.py KuzushijiDataset.__getitem__)."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = (img * std + mean) * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


def _crop_glyph(img_np: np.ndarray, box) -> np.ndarray | None:
    """Crop a glyph out of a denormalized (H, W, 3) image using an xyxy
    pixel-space box, clamped to image bounds (same convention as
    utils/char_augmentation.py's copy-paste crop)."""
    h, w = img_np.shape[:2]
    x1, y1, x2, y2 = (int(round(float(v))) for v in box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return img_np[y1:y2, x1:x2]


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


def _build_confusion_matrix(
    error_counter: Counter,
    gt_total_counter: Counter,
    top_n: int = 25,
) -> tuple[list[int], np.ndarray]:
    """Select the top-N most error-prone GT classes (+ their confusees) and
    build a row-normalised confusion matrix over them.

    Returns (class_list, mat_norm) — class_list gives the vocab-id ordering
    of mat_norm's rows/columns (frequency order, not yet clustered).
    """
    if not error_counter:
        return [], np.zeros((0, 0))

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
        return [], np.zeros((0, 0))

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
    return class_list, mat_norm


def _plot_confusion_heatmap(
    class_list: list[int],
    mat_norm: np.ndarray,
    vocab,
    out_path: Path,
    title: str,
) -> None:
    """Shared heatmap rendering for plot_confusion_matrix and the clustered variant."""
    n = len(class_list)
    if n == 0:
        print("No confusion data to plot.")
        return

    labels = [_ids_to_text([c], vocab) for c in class_list]

    # Scale cell size down for large matrices so the figure stays manageable.
    cell_size = max(0.4, min(0.65, 18.0 / max(n, 1)))
    fontsize  = max(8,   min(16,  int(cell_size * 72 * 0.55)))
    fig_w = n * cell_size + 2.5   # extra space for colorbar + y-labels
    fig_h = n * cell_size + 1.5

    cjk = cjk_font_prop(fontsize)

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
    ax.set_title(title, fontsize=11)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix saved: {out_path}")


def plot_confusion_matrix(
    error_counter: Counter,
    gt_total_counter: Counter,
    vocab,
    out_path: Path,
    top_n: int = 25,
) -> None:
    """Save a row-normalised confusion matrix for the top-N most error-prone classes."""
    class_list, mat_norm = _build_confusion_matrix(error_counter, gt_total_counter, top_n)
    _plot_confusion_heatmap(
        class_list, mat_norm, vocab, out_path,
        title=f"Confusion Matrix — top {len(class_list)} error-prone classes (row-normalised)",
    )


def plot_confusion_matrix_clustered(
    error_counter: Counter,
    gt_total_counter: Counter,
    vocab,
    out_path: Path,
    top_n: int = 25,
) -> None:
    """Like plot_confusion_matrix, but rows/columns are hierarchically
    clustered (instead of left in frequency order) so mutually-confused
    character families group together visually — a dense top-N kanji
    heatmap in frequency order makes those clusters hard to spot.
    """
    class_list, mat_norm = _build_confusion_matrix(error_counter, gt_total_counter, top_n)
    n = len(class_list)
    if n < 3:
        # Clustering needs at least 3 points; fall back silently for tiny matrices.
        _plot_confusion_heatmap(
            class_list, mat_norm, vocab, out_path,
            title=f"Confusion Matrix (clustered) — top {n} error-prone classes",
        )
        return

    from scipy.cluster.hierarchy import dendrogram, linkage

    # Symmetrized confusion (rows and columns represent the same class set),
    # so mutually-confused pairs cluster regardless of direction.
    dist = 1.0 - np.maximum(mat_norm, mat_norm.T)
    np.fill_diagonal(dist, 0.0)
    condensed = dist[np.triu_indices(n, k=1)]
    link = linkage(condensed, method="average")
    order = dendrogram(link, no_plot=True)["leaves"]

    clustered_class_list = [class_list[i] for i in order]
    clustered_mat = mat_norm[np.ix_(order, order)]
    _plot_confusion_heatmap(
        clustered_class_list, clustered_mat, vocab, out_path,
        title=f"Confusion Matrix (clustered) — top {n} error-prone classes",
    )


def plot_confusion_gallery(
    pair_crop_examples: dict[tuple[int, int], list[np.ndarray]],
    class_correct_crops: dict[int, list[np.ndarray]],
    error_counter: Counter,
    vocab,
    out_path: Path,
    top_n: int = 12,
) -> None:
    """Grid of actual glyph crops for the top confused (gt, pred) pairs:
    real misclassified GT crops (left columns) next to canonical correctly-
    classified crops of the predicted class (right columns) — shows whether
    a confusion is genuine visual similarity or something else.
    """
    top_pairs = [
        (gt, pr) for (gt, pr), _ in error_counter.most_common()
        if pair_crop_examples.get((gt, pr))
    ][:top_n]
    if not top_pairs:
        print("No crop examples available for confusion gallery.")
        return

    n_rows = len(top_pairs)
    n_cols = 2 * _MAX_GALLERY_CROPS
    # CJK font path only covers CJK codepoints (no Latin), so English label
    # text and the character itself must be rendered as separate Text
    # objects — mirrors the tick-label convention in plot_confusion_matrix.
    cjk = cjk_font_prop(11)

    def _char_or_codepoint(ch: str) -> str:
        return ch if cjk is not None else ("U+{:04X}".format(ord(ch[0])) if ch else "?")

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.3 * n_cols, 1.5 * n_rows),
                              gridspec_kw={"hspace": 0.7})
    axes = np.atleast_2d(axes)

    for row, (gt, pr) in enumerate(top_pairs):
        gt_char = _ids_to_text([gt], vocab)
        pr_char = _ids_to_text([pr], vocab)
        misclassified = pair_crop_examples.get((gt, pr), [])
        canonical = class_correct_crops.get(pr, [])

        for col in range(_MAX_GALLERY_CROPS):
            ax = axes[row, col]
            if col < len(misclassified):
                ax.imshow(misclassified[col])
            ax.axis("off")
            if col == 0:
                # Two-line label: small English caption above, char/arrow/char
                # below — kept on separate Text objects since the CJK font
                # covers no Latin glyphs (see _char_or_codepoint above).
                ax.text(0.0, 1.28, "GT/pred:", transform=ax.transAxes, fontsize=7, ha="left", va="bottom")
                ax.text(0.0, 1.0, _char_or_codepoint(gt_char),
                        transform=ax.transAxes, fontsize=10, ha="left", va="bottom", fontproperties=cjk)
                ax.text(0.32, 1.0, "->", transform=ax.transAxes, fontsize=10, ha="left", va="bottom")
                ax.text(0.55, 1.0, _char_or_codepoint(pr_char),
                        transform=ax.transAxes, fontsize=10, ha="left", va="bottom", fontproperties=cjk)

        for col in range(_MAX_GALLERY_CROPS):
            ax = axes[row, _MAX_GALLERY_CROPS + col]
            if col < len(canonical):
                ax.imshow(canonical[col])
            ax.axis("off")
            if col == 0:
                ax.text(0.0, 1.28, "canonical:", transform=ax.transAxes, fontsize=7, ha="left", va="bottom")
                ax.text(0.0, 1.0, _char_or_codepoint(pr_char),
                        transform=ax.transAxes, fontsize=10, ha="left", va="bottom",
                        fontproperties=cjk)

    fig.suptitle("Confusion gallery — misclassified crops (left) vs. canonical predicted-class crops (right)", y=0.995)
    # Reserve a fixed ~0.5in of header space regardless of row count, rather
    # than a fixed fraction — a fraction blows up into a huge blank margin
    # once the figure gets tall (many rows).
    fig.subplots_adjust(hspace=0.7, top=max(0.85, 1.0 - 0.5 / fig.get_figheight()))
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion gallery saved: {out_path}")


def plot_stage_error_attribution(
    stage1_miss_rate: float,
    stage2_misclass_rate: float,
    ordering_violation_rate_positives: float,
    out_path: Path,
) -> None:
    """Bar chart of the three stage-wise error rates. NOT a partition —
    stage2_misclass and ordering_violation overlap (reading order is
    computed before classification and feeds the model's context/neighbor
    features, so an ordering mistake can directly cause a misclassification)
    — the caption makes this explicit so the chart isn't misread as a pie.
    """
    labels = ["Stage 1 miss\n(no matched proposal)",
              "Stage 2 misclass\n(among localized)",
              "Reading-order violation\n(among localized)"]
    values = [stage1_miss_rate, stage2_misclass_rate, ordering_violation_rate_positives]
    colors = ["#C44E52", "#DD8452", "#8172B2"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values, color=colors, alpha=0.75)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{v*100:.1f}%", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Rate")
    ax.set_ylim(0, max(0.05, max(values) * 1.25))
    ax.set_title("Stage-wise error attribution")
    ax.grid(axis="y", alpha=0.3)
    fig.text(
        0.5, -0.02,
        "NOTE: not a partition — reading-order violations can directly cause\n"
        "Stage 2 misclassifications (ordering feeds the classifier's context), "
        "so these overlap.",
        ha="center", va="top", fontsize=8, style="italic", color="#555555",
    )
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Stage error attribution chart saved: {out_path}")


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
# Per-book precision/recall/F1 table
# -------------------------

def _init_book_image_counts(
    book_stats: dict[str, dict[str, dict[str, float]]],
    image_stats: dict[str, dict],
    book_id: str,
    image_stem: str,
    split_name: str,
    n_gt: int,
    n_pos: int,
    n_prop: int,
) -> None:
    """First-touch per-image book/image stats bookkeeping (images/gt/pos/props
    counts) — shared between the main val pass and the optional lightweight
    train-only pass (_accumulate_book_image_stats_pass). Keyed by canonical
    book title (see _canonical_book_title) at the top level and split_name
    ("train"/"val") one level in, so a book present in both splits keeps
    them distinguishable.
    """
    canon = _canonical_book_title(book_id)
    bstat = (
        book_stats.setdefault(canon, {})
        .setdefault(split_name, {"images": 0, "gt": 0.0, "pos": 0.0, "props": 0.0,
                                  "correct": 0.0, "ordering_viol": 0.0, "ordering_pairs": 0.0})
    )
    bstat["images"] += 1
    bstat["gt"]     += n_gt
    bstat["pos"]    += n_pos
    bstat["props"]  += n_prop

    image_stats[image_stem] = {
        "book_id": book_id, "book_title": canon, "split": split_name,
        "gt": float(n_gt), "pos": float(n_pos), "props": float(n_prop), "correct": 0.0,
        "ordering_viol": 0.0, "ordering_pairs": 0.0,
    }


def _record_book_image_correct_and_ordering(
    book_stats: dict[str, dict[str, dict[str, float]]],
    image_stats: dict[str, dict],
    book_id: str,
    image_stem: str,
    split_name: str,
    n_correct: int,
    ordering_viol: int,
    ordering_pairs: int,
) -> None:
    """Second-touch bookkeeping (classification-correct count + reading-order
    violations among GT-matched positives) for an image already touched by
    _init_book_image_counts in the same forward pass — split out because the
    val pass computes these inside a separate loop that also does
    confusion-matrix tracking, sharing the same predicted/true id tensors.
    """
    canon = _canonical_book_title(book_id)
    bstat = book_stats[canon][split_name]
    bstat["correct"] += n_correct
    image_stats[image_stem]["correct"] += n_correct
    if ordering_pairs > 0:
        bstat["ordering_viol"]   += ordering_viol
        bstat["ordering_pairs"]  += ordering_pairs
        image_stats[image_stem]["ordering_viol"]  += ordering_viol
        image_stats[image_stem]["ordering_pairs"] += ordering_pairs


def _accumulate_book_image_stats_pass(
    model: torch.nn.Module,
    loader: "DataLoader",
    split_name: str,
    book_stats: dict[str, dict[str, dict[str, float]]],
    image_stats: dict[str, dict],
    max_batches: int | None,
) -> int:
    """Lightweight no_grad forward pass over `loader`, populating only
    book_stats/image_stats — not the headline loss/top1/CER/confusion-matrix
    tracking that run_validation's main pass computes. Used for the optional
    train-split pass requested via --split train/both: headline metrics stay
    val-only (for comparability with every historical validation run), but
    book/image-level performance is still worth seeing across the full
    corpus. Returns the number of images evaluated.
    """
    n_images = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images         = batch["image"].to(DEVICE, non_blocking=True)
            boxes_list     = [b.to(DEVICE, dtype=torch.float32) for b in batch["boxes"]]
            gt_labels_list = [l.to(DEVICE, dtype=torch.long) for l in batch["labels"]]
            orientations   = [_normalize_orientation_label(o) for o in batch["orientations"]]
            image_stems = batch["image_stems"]
            book_ids    = [_book_id_for_stem(s) for s in image_stems]

            outputs = model(images, orientations)
            refine_targets = build_refinement_targets(
                coarse_boxes=outputs["roi_boxes"],
                roi_mask=outputs["roi_mask"],
                gt_boxes_list=boxes_list,
                gt_labels_list=gt_labels_list,
                pos_iou_thresh=STAGE2_REFINE_POS_IOU,
                neg_iou_thresh=STAGE2_REFINE_NEG_IOU,
                use_hungarian=STAGE2_USE_HUNGARIAN,
            )

            sort_indices = outputs.get("sort_indices", None)
            gt_labels    = refine_targets["matched_gt_labels"]
            pos_mask     = refine_targets["refine_pos_mask"]
            if sort_indices is not None:
                gt_labels_s = reorder_by_sort_indices(gt_labels, sort_indices)
                pos_mask_s  = reorder_by_sort_indices(pos_mask.long(), sort_indices).bool()
            else:
                gt_labels_s = gt_labels
                pos_mask_s  = pos_mask

            bsz = images.size(0)
            n_images += bsz

            for b in range(bsz):
                n_gt   = boxes_list[b].size(0)
                n_pos  = int(refine_targets["refine_pos_mask"][b].sum().item())
                n_prop = int(outputs["roi_mask"][b].sum().item())
                _init_book_image_counts(book_stats, image_stats, book_ids[b], image_stems[b],
                                         split_name, n_gt, n_pos, n_prop)

                valid_b = pos_mask_s[b] & gt_labels_s[b].ne(-1)
                if not valid_b.any():
                    continue
                pred_ids_t = outputs["char_logits"][b, valid_b].argmax(dim=-1)
                true_ids_t = gt_labels_s[b][valid_b]
                n_correct_b = int((pred_ids_t == true_ids_t).sum().item())
                boxes_t = outputs["ordered_boxes"][b, valid_b]

                n_pos_b = boxes_t.size(0)
                ordering_viol_b, ordering_pairs_b = 0, 0
                if n_pos_b > 1:
                    _, viol_frac_pos = model.roi_order._primary_axis_monotonic_fraction(
                        boxes_t, orientations[b]
                    )
                    ordering_viol_b  = round(viol_frac_pos * (n_pos_b - 1))
                    ordering_pairs_b = n_pos_b - 1
                _record_book_image_correct_and_ordering(
                    book_stats, image_stats, book_ids[b], image_stems[b], split_name,
                    n_correct_b, ordering_viol_b, ordering_pairs_b,
                )
    return n_images


def _one_row_per_book_preferring_val(per_book_rows: list[dict]) -> list[dict]:
    """Collapse print_and_save_per_book_metrics' output (which can hold up
    to 3 rows per book — train/val/both — once --split includes train) down
    to one representative row per book, sorted worst-first. Prefers val
    (headline metrics.json stays val-only for cross-run comparability),
    then "both", then "train" for the one book that's train-only under
    val's split file. Under the default --split val this is a no-op
    (already exactly one row per book).
    """
    _preferred = {"val": 0, "both": 1, "train": 2}
    best: dict[str, dict] = {}
    for r in per_book_rows:
        cur = best.get(r["book_id"])
        if cur is None or _preferred.get(r["split"], 9) < _preferred.get(cur["split"], 9):
            best[r["book_id"]] = r
    return sorted(best.values(), key=lambda r: r["pipeline_f1"])


def print_and_save_per_book_metrics(
    book_stats: dict[str, dict[str, dict[str, float]]],
    out_path: Path,
) -> list[dict]:
    """Print and save a per-book precision/recall/F1 table (KuroNet-paper
    style — Clanuwat et al. report metrics broken down by book rather than
    only pooled across the whole validation set).

    Reports both the localization-only det_* metrics (IoU-match only, same
    definition as the global coverage/det_precision/det_f1) and the joint
    pipeline_* metrics (IoU-match AND correct class).

    book_stats is nested by split ("val", and "train" too under --split
    train/both). Emits one row per split actually present for a book, plus
    a "both" row (summed raw counts, not averaged ratios) only when a book
    has both train and val data — under the default --split val, every book
    has exactly one row (split="val"), same as before this column existed.
    Rows are grouped by book (worst-book-first, using the val row's
    pipeline_f1 as the representative score, falling back to train's for
    the one book that's train-only under val's split file), then ordered
    train/val/both within a book.
    """
    def _row_from_counts(book_id: str, split_name: str, s: dict[str, float]) -> dict:
        gt, pos, props, correct, images = s["gt"], s["pos"], s["props"], s["correct"], s["images"]
        ordering_viol  = s.get("ordering_viol", 0.0)
        ordering_pairs = s.get("ordering_pairs", 0.0)
        det_recall    = pos / max(1e-6, gt)
        det_precision = pos / max(1e-6, props)
        det_f1        = 2 * det_precision * det_recall / max(1e-6, det_precision + det_recall)
        pipeline_recall    = correct / max(1e-6, gt)
        pipeline_precision = correct / max(1e-6, props)
        pipeline_f1        = (2 * pipeline_precision * pipeline_recall
                               / max(1e-6, pipeline_precision + pipeline_recall))
        stage1_miss_rate      = (gt - pos) / max(1e-6, gt)
        stage2_misclass_rate  = (pos - correct) / max(1e-6, pos)
        ordering_violation_rate_positives = ordering_viol / max(1e-6, ordering_pairs)
        return {
            "book_id": book_id,
            "split": split_name,
            "images": int(images),
            "gt": int(gt),
            "mean_gt_per_image": round(gt / max(1e-6, images), 2),
            "det_precision": round(det_precision, 4),
            "det_recall": round(det_recall, 4),
            "det_f1": round(det_f1, 4),
            "pipeline_precision": round(pipeline_precision, 4),
            "pipeline_recall": round(pipeline_recall, 4),
            "pipeline_f1": round(pipeline_f1, 4),
            "stage1_miss_rate": round(stage1_miss_rate, 4),
            "stage2_misclass_rate": round(stage2_misclass_rate, 4),
            "ordering_violation_rate_positives": round(ordering_violation_rate_positives, 4),
        }

    rows: list[dict] = []
    for book_id, by_split in book_stats.items():
        for split_name, s in by_split.items():
            rows.append(_row_from_counts(book_id, split_name, s))
        if "train" in by_split and "val" in by_split:
            summed = {
                k: by_split["train"].get(k, 0.0) + by_split["val"].get(k, 0.0)
                for k in ("images", "gt", "pos", "props", "correct", "ordering_viol", "ordering_pairs")
            }
            rows.append(_row_from_counts(book_id, "both", summed))

    val_f1_by_book   = {r["book_id"]: r["pipeline_f1"] for r in rows if r["split"] == "val"}
    train_f1_by_book = {r["book_id"]: r["pipeline_f1"] for r in rows if r["split"] == "train"}
    _split_order = {"train": 0, "val": 1, "both": 2}

    def _rep_f1(book_id: str) -> float:
        return val_f1_by_book.get(book_id, train_f1_by_book.get(book_id, 0.0))

    rows.sort(key=lambda r: (_rep_f1(r["book_id"]), r["book_id"], _split_order.get(r["split"], 9)))

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "book_id", "split", "images", "gt", "mean_gt_per_image",
            "det_precision", "det_recall", "det_f1",
            "pipeline_precision", "pipeline_recall", "pipeline_f1",
            "stage1_miss_rate", "stage2_misclass_rate", "ordering_violation_rate_positives",
        ])
        for row in rows:
            writer.writerow([
                row["book_id"], row["split"], row["images"], row["gt"], row["mean_gt_per_image"],
                row["det_precision"], row["det_recall"], row["det_f1"],
                row["pipeline_precision"], row["pipeline_recall"], row["pipeline_f1"],
                row["stage1_miss_rate"], row["stage2_misclass_rate"],
                row["ordering_violation_rate_positives"],
            ])

    n_books = len({r["book_id"] for r in rows})
    print(f"\nPER-BOOK METRICS CSV: {out_path}  ({n_books} books, {len(rows)} rows)")
    print(
        f"{'Book':<10} {'Split':<6} {'Images':>7} {'GT':>7}  "
        f"{'DetP':>6} {'DetR':>6} {'DetF1':>6}   "
        f"{'PipeP':>6} {'PipeR':>6} {'PipeF1':>6}"
    )
    print(f"{'-'*10} {'-'*6} {'-'*7} {'-'*7}  {'-'*6} {'-'*6} {'-'*6}   {'-'*6} {'-'*6} {'-'*6}")
    for row in rows:
        print(
            f"{row['book_id']:<10} {row['split']:<6} {row['images']:>7} {row['gt']:>7}  "
            f"{row['det_precision']:>6.3f} {row['det_recall']:>6.3f} {row['det_f1']:>6.3f}   "
            f"{row['pipeline_precision']:>6.3f} {row['pipeline_recall']:>6.3f} {row['pipeline_f1']:>6.3f}"
        )

    return rows


def plot_per_book_metric_distribution(
    per_book_rows: list[dict],
    out_path: Path,
) -> Path:
    """Violin + jittered strip plot of det_f1/pipeline_f1 distributions
    across the canonical books, split train vs. val ("both" rows are
    excluded so the two splits stay a clean comparison — train performance
    is inflated since the model has already seen those images, and
    blending it into one distribution would hide that).

    With only ~44 books per split, a pure KDE violin can mislead, so the
    actual per-book points are overlaid as a jittered strip. Degrades
    gracefully to a single (val) violin under the default --split val,
    where no train rows exist.
    """
    _colors = {"train": "#4C72B0", "val": "#C44E52"}
    rows = [r for r in per_book_rows if r["split"] in ("train", "val")]
    present_splits = [s for s in ("train", "val") if any(r["split"] == s for r in rows)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharey=True)
    rng = np.random.default_rng(0)
    for ax, metric, title in zip(axes, ("det_f1", "pipeline_f1"), ("Detection F1", "Pipeline F1")):
        data = [[r[metric] for r in rows if r["split"] == s] for s in present_splits]
        positions = list(range(1, len(present_splits) + 1))
        if any(data):
            parts = ax.violinplot(data, positions=positions, showmedians=True, widths=0.7)
            for pc, s in zip(parts["bodies"], present_splits):
                pc.set_facecolor(_colors[s])
                pc.set_alpha(0.4)
            for key in ("cmedians", "cbars", "cmins", "cmaxes"):
                if key in parts:
                    parts[key].set_color("#333333")
            for pos, s, vals in zip(positions, present_splits, data):
                jitter = rng.uniform(-0.08, 0.08, size=len(vals))
                ax.scatter(np.full(len(vals), pos) + jitter, vals, s=14, alpha=0.7,
                           color=_colors[s], edgecolors="none", zorder=3)
        ax.set_xticks(positions)
        ax.set_xticklabels([s.capitalize() for s in present_splits])
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)

    n_books = len({r["book_id"] for r in rows})
    fig.suptitle(f"Per-book metric distribution (n={n_books} canonical books)")
    fig.tight_layout()
    return savefig(fig, out_path)


def plot_book_difficulty_vs_f1(
    per_book_rows: list[dict],
    out_path: Path,
) -> Path:
    """Scatter of mean_gt_per_image (x — a page-density/difficulty proxy,
    the existing "characters per image" concept aggregated per book) vs.
    pipeline precision/recall/F1 (one panel each), one point per (book,
    split). "both" rows are excluded for the same train/val-inflation
    reason as plot_per_book_metric_distribution. Annotates a per-split
    Pearson correlation coefficient in each panel.
    """
    _colors = {"train": "#4C72B0", "val": "#C44E52"}
    rows = [r for r in per_book_rows if r["split"] in ("train", "val")]
    metrics = [
        ("pipeline_precision", "Pipeline precision"),
        ("pipeline_recall", "Pipeline recall"),
        ("pipeline_f1", "Pipeline F1"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharex=True)
    for ax, (metric_key, title) in zip(axes, metrics):
        y_offset = 0.02
        for s in ("train", "val"):
            xs = [r["mean_gt_per_image"] for r in rows if r["split"] == s]
            ys = [r[metric_key] for r in rows if r["split"] == s]
            if not xs:
                continue
            ax.scatter(xs, ys, s=24, alpha=0.75, color=_colors[s], label=s.capitalize())
            if len(xs) >= 2:
                r_val = float(np.corrcoef(xs, ys)[0, 1])
                ax.annotate(f"{s}: r={r_val:.2f}", xy=(0.02, y_offset),
                            xycoords="axes fraction", color=_colors[s], fontsize=9)
                y_offset += 0.05

        ax.set_xlabel("Mean GT characters per image (difficulty/density proxy)")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle("Per-book performance vs. page density")
    fig.tight_layout()
    return savefig(fig, out_path)


def print_and_save_per_image_metrics(
    image_stats: dict[str, dict],
    out_path: Path,
) -> list[dict]:
    """Save a per-image precision/recall/F1 table, grouped by book — the
    same det_*/pipeline_* breakdown as print_and_save_per_book_metrics, but
    at image granularity so individual pages dragging a book's score down
    can be identified. Sorted by (book_id, image_stem) so it reads as "for
    each book, all of its images". Not printed to stdout in full (one row
    per validation image would be hundreds of lines) — only a summary line
    and the worst few images by pipeline F1.
    """
    rows = []
    for stem, s in image_stats.items():
        gt, pos, props, correct = s["gt"], s["pos"], s["props"], s["correct"]
        ordering_viol  = s.get("ordering_viol", 0.0)
        ordering_pairs = s.get("ordering_pairs", 0.0)
        det_recall    = pos / max(1e-6, gt)
        det_precision = pos / max(1e-6, props)
        det_f1        = 2 * det_precision * det_recall / max(1e-6, det_precision + det_recall)
        pipeline_recall    = correct / max(1e-6, gt)
        pipeline_precision = correct / max(1e-6, props)
        pipeline_f1        = (2 * pipeline_precision * pipeline_recall
                               / max(1e-6, pipeline_precision + pipeline_recall))
        stage1_miss_rate      = (gt - pos) / max(1e-6, gt)
        stage2_misclass_rate  = (pos - correct) / max(1e-6, pos)
        ordering_violation_rate_positives = ordering_viol / max(1e-6, ordering_pairs)
        rows.append({
            "book_id": s["book_id"],
            "book_title": s.get("book_title", s["book_id"]),
            "split": s.get("split", "val"),
            "image_stem": stem,
            "gt": int(gt),
            "det_precision": round(det_precision, 4),
            "det_recall": round(det_recall, 4),
            "det_f1": round(det_f1, 4),
            "pipeline_precision": round(pipeline_precision, 4),
            "pipeline_recall": round(pipeline_recall, 4),
            "pipeline_f1": round(pipeline_f1, 4),
            "stage1_miss_rate": round(stage1_miss_rate, 4),
            "stage2_misclass_rate": round(stage2_misclass_rate, 4),
            "ordering_violation_rate_positives": round(ordering_violation_rate_positives, 4),
        })

    rows.sort(key=lambda r: (r["book_id"], r["image_stem"]))

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "book_id", "book_title", "split", "image_stem", "gt",
            "det_precision", "det_recall", "det_f1",
            "pipeline_precision", "pipeline_recall", "pipeline_f1",
            "stage1_miss_rate", "stage2_misclass_rate", "ordering_violation_rate_positives",
        ])
        for row in rows:
            writer.writerow([
                row["book_id"], row["book_title"], row["split"], row["image_stem"], row["gt"],
                row["det_precision"], row["det_recall"], row["det_f1"],
                row["pipeline_precision"], row["pipeline_recall"], row["pipeline_f1"],
                row["stage1_miss_rate"], row["stage2_misclass_rate"],
                row["ordering_violation_rate_positives"],
            ])

    n_books = len({r["book_id"] for r in rows})
    print(f"\nPER-IMAGE METRICS CSV: {out_path}  ({len(rows)} images across {n_books} books, grouped by book)")

    worst = sorted(rows, key=lambda r: r["pipeline_f1"])[:10]
    print("  Worst 10 images by pipeline F1:")
    print(f"  {'Book':<14} {'Image':<24} {'GT':>5}  {'PipeF1':>6}")
    for row in worst:
        print(f"  {row['book_id']:<14} {row['image_stem']:<24} {row['gt']:>5}  {row['pipeline_f1']:>6.3f}")

    return rows


# -------------------------
# Excluded pages/books table
# -------------------------

def save_excluded_pages_table(out_path: Path) -> list[dict]:
    """Save a table of books/pages deliberately excluded from all splits
    (config.EXCLUDE_BOOKS / config.EXCLUDE_PAGES) — documents held-out
    material for reproducibility, mirroring how papers report which
    books/pages were excluded from training/evaluation. Independent of the
    checkpoint being validated (pure dataset-config data), but written
    alongside the other per-run tables for convenience.
    """
    rows = []
    for book_id in sorted(EXCLUDE_BOOKS):
        rows.append({"book_id": book_id, "page": "", "reason": "entire book excluded"})
    for book_id, pages in EXCLUDE_PAGES.items():
        for page in sorted(pages):
            rows.append({"book_id": book_id, "page": page, "reason": "held-out page"})
    rows.sort(key=lambda r: (r["book_id"], r["page"]))

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["book_id", "page", "reason"])
        for row in rows:
            writer.writerow([row["book_id"], row["page"], row["reason"]])

    n_pages = sum(len(v) for v in EXCLUDE_PAGES.values())
    print(f"\nEXCLUDED PAGES CSV: {out_path}  ({len(EXCLUDE_BOOKS)} excluded books, {n_pages} held-out pages)")
    print(f"  {'Book':<14} {'Page':<28} Reason")
    print(f"  {'-'*14} {'-'*28} {'-'*20}")
    for row in rows:
        print(f"  {row['book_id']:<14} {row['page']:<28} {row['reason']}")

    return rows


# -------------------------
# Main validation
# -------------------------

def run_validation(
    ckpt_path: Path,
    max_batches: int | None,
    out_dir: Path | None = None,
    backbone: str | None = None,
    context_mode: str | None = None,
    split: str = "val",
    log_files: "list[Path] | None" = None,
) -> None:
    if out_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        resolved_job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("JOB_ID")
        run_tag = f"{timestamp}_job{resolved_job_id}" if resolved_job_id else timestamp
        out_dir = SUMINANET_CHECKPOINT_DIR / "validation" / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    vocab = load_vocab()
    val_loader = _build_eval_loader(vocab, "val")

    model = build_suminanet_model(
        vocab=vocab, load_stage1_weights=False,
        backbone_type_override=backbone, context_mode_override=context_mode,
    )
    epoch = _load_suminanet_weights(model, ckpt_path, vocab)
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
    correct_sum       = 0.0
    ordering_viol_sum = 0.0
    ordering_viol_n   = 0
    # Reading-order violations scoped to GT-matched (positive) characters
    # only, unlike ordering_viol_sum/n above which macro-average over ALL
    # proposals including background — see stage-wise error attribution.
    ordering_viol_pos_sum  = 0.0
    ordering_pairs_pos_sum = 0.0
    n_batches         = 0
    n_images_tot      = 0

    # Confusion tracking — accumulated over ALL images in every batch
    error_counter:       Counter = Counter()  # (gt_id, pred_id) -> count, all mismatches
    kana_mixup_counter:  Counter = Counter()  # (gt_id, pred_id) -> count, same phoneme wrong script
    pred_counter:        Counter = Counter()  # pred_id -> count
    gt_total_counter:    Counter = Counter()  # gt_id -> total occurrences in positives

    # Glyph crops for the confusion gallery (plot_confusion_gallery), capped
    # per pair / per class so this stays small regardless of dataset size.
    pair_crop_examples:  dict[tuple[int, int], list[np.ndarray]] = {}
    class_correct_crops: dict[int, list[np.ndarray]] = {}

    # Per-book det/pipeline stats (print_and_save_per_book_metrics), keyed
    # by canonical book title (see _canonical_book_title) then by split
    # ("val", and "train" too when --split requests it) — a book that
    # appears in both splits keeps them distinguishable rather than blended,
    # since train-set performance is inflated (the model has seen those
    # images) and that signal needs to stay visible.
    book_stats: dict[str, dict[str, dict[str, float]]] = {}

    # Per-image det/pipeline stats, keyed by image_stem (every validation
    # image appears exactly once, so no collisions) — same quantities as
    # book_stats but kept at image granularity for print_and_save_per_image_metrics.
    image_stats: dict[str, dict[str, "str | float"]] = {}

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

    iso_mask = None  # set inside the loop below; kept here so it's defined even if n_batches == 0

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
            image_stems = batch["image_stems"]
            book_ids    = [_book_id_for_stem(s) for s in image_stems]

            outputs = model(images, orientations)

            refine_targets = build_refinement_targets(
                coarse_boxes=outputs["roi_boxes"],
                roi_mask=outputs["roi_mask"],
                gt_boxes_list=boxes_list,
                gt_labels_list=gt_labels_list,
                pos_iou_thresh=STAGE2_REFINE_POS_IOU,
                neg_iou_thresh=STAGE2_REFINE_NEG_IOU,
                use_hungarian=STAGE2_USE_HUNGARIAN,
            )

            loss, parts = compute_suminanet_loss(outputs, refine_targets, bg_id=bg_id)
            total_loss += float(loss.item())
            for k in loss_parts:
                loss_parts[k] += float(parts.get(k, 0.0))

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
                                              score_thresh=SUMINANET_CER_SCORE_THRESH)

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

                _init_book_image_counts(book_stats, image_stats, book_ids[b], image_stems[b],
                                         "val", n_gt, n_pos, n_prop)

            iou_b = refine_targets["matched_iou"]
            pm_b  = refine_targets["refine_pos_mask"]
            if pm_b.any():
                iou_sum += float(iou_b[pm_b].mean().item())

            # Confusion accumulation — loop over ALL images in the batch (was only [0])
            for b in range(bsz):
                valid_b = pos_mask_s[b] & gt_labels_s[b].ne(-1)
                if valid_b.any():
                    pred_ids_t = outputs["char_logits"][b, valid_b].argmax(dim=-1)
                    true_ids_t = gt_labels_s[b][valid_b]
                    n_correct_b = int((pred_ids_t == true_ids_t).sum().item())
                    correct_sum += n_correct_b
                    boxes_t = outputs["ordered_boxes"][b, valid_b]

                    n_pos_b = boxes_t.size(0)
                    ordering_viol_b, ordering_pairs_b = 0, 0
                    if n_pos_b > 1:
                        _, viol_frac_pos = model.roi_order._primary_axis_monotonic_fraction(
                            boxes_t, orientations[b]
                        )
                        ordering_viol_b  = round(viol_frac_pos * (n_pos_b - 1))
                        ordering_pairs_b = n_pos_b - 1
                        ordering_viol_pos_sum  += ordering_viol_b
                        ordering_pairs_pos_sum += ordering_pairs_b
                    _record_book_image_correct_and_ordering(
                        book_stats, image_stats, book_ids[b], image_stems[b], "val",
                        n_correct_b, ordering_viol_b, ordering_pairs_b,
                    )

                    pred_ids = pred_ids_t.tolist()
                    true_ids = true_ids_t.tolist()
                    img_np = None  # lazily denormalized only if a crop is actually stashed
                    for pid, tid, box in zip(pred_ids, true_ids, boxes_t):
                        pred_counter[pid] += 1
                        gt_total_counter[tid] += 1
                        if pid != tid:
                            error_counter[(tid, pid)] += 1
                            if _is_kana_script_mixup(
                                _ids_to_text([tid], vocab),
                                _ids_to_text([pid], vocab),
                            ):
                                kana_mixup_counter[(tid, pid)] += 1
                            pair_examples = pair_crop_examples.setdefault((tid, pid), [])
                            if len(pair_examples) < _MAX_GALLERY_CROPS:
                                if img_np is None:
                                    img_np = _denormalize_image(images[b])
                                crop = _crop_glyph(img_np, box.tolist())
                                if crop is not None and crop.size:
                                    pair_examples.append(crop)
                        else:
                            class_examples = class_correct_crops.setdefault(tid, [])
                            if len(class_examples) < _MAX_GALLERY_CROPS:
                                if img_np is None:
                                    img_np = _denormalize_image(images[b])
                                crop = _crop_glyph(img_np, box.tolist())
                                if crop is not None and crop.size:
                                    class_examples.append(crop)

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

            if len(examples) < SUMINANET_PREDICTION_SAMPLES:
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

    avg_props   = props_sum   / max(1, n_images_tot)
    avg_pos     = pos_sum     / max(1, n_images_tot)
    avg_gt      = gt_sum      / max(1, n_images_tot)
    avg_correct = correct_sum / max(1, n_images_tot)
    det_recall    = avg_pos / max(1e-6, avg_gt)
    det_precision = avg_pos / max(1e-6, avg_props)
    det_f1        = 2 * det_precision * det_recall / max(1e-6, det_precision + det_recall)

    # Pipeline F1: a proposal counts as correct only if it's both IoU-matched
    # AND classified correctly — unlike det_f1/coverage above, which only
    # check localization. This is the number comparable to Clanuwat et al.'s
    # page-level detection+recognition F1 (see visualization/baselines.py).
    pipeline_recall    = avg_correct / max(1e-6, avg_gt)
    pipeline_precision = avg_correct / max(1e-6, avg_props)
    pipeline_f1        = (2 * pipeline_precision * pipeline_recall
                           / max(1e-6, pipeline_precision + pipeline_recall))

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
        f"  pipeline_recall={pipeline_recall:.4f}"
        f"  pipeline_precision={pipeline_precision:.4f}"
        f"  pipeline_F1={pipeline_f1:.4f}"
        f"  (IoU-match AND correct class — comparable to Clanuwat et al. page-level F1)"
    )
    print(
        f"  ordering_violation_rate={mean_ordering_viol:.4f}"
        f"  (fraction of adjacent ROI pairs out of reading order; 0=perfect)"
    )

    # -----------------------------------------------------------------------
    # Stage-wise error attribution
    # -----------------------------------------------------------------------
    stage1_miss_count  = gt_sum - pos_sum
    stage1_miss_rate   = stage1_miss_count / max(1e-6, gt_sum)
    stage2_misclass_count = pos_sum - correct_sum
    stage2_misclass_rate  = stage2_misclass_count / max(1e-6, pos_sum)
    ordering_violation_rate_positives = (
        ordering_viol_pos_sum / max(1e-6, ordering_pairs_pos_sum)
    )

    print("\nSTAGE-WISE ERROR ATTRIBUTION")
    print(
        f"  stage1_miss={stage1_miss_rate:.4f} ({int(stage1_miss_count)}/{int(gt_sum)} GT chars "
        f"never got a matched proposal)"
    )
    print(
        f"  stage2_misclass={stage2_misclass_rate:.4f} ({int(stage2_misclass_count)}/{int(pos_sum)} "
        f"localized proposals classified wrong)"
    )
    print(
        f"  ordering_violation_rate_positives={ordering_violation_rate_positives:.4f} "
        f"({int(ordering_viol_pos_sum)}/{int(ordering_pairs_pos_sum)} adjacent GT-matched pairs "
        f"out of reading order)"
    )
    print(
        "  NOTE: stage2_misclass and ordering_violation are NOT a clean partition — "
        "reading order is computed BEFORE classification and feeds the context/neighbor "
        "features (SuminaNetRecognizer.encode()), so an ordering mistake can directly "
        "cause a misclassification. Treat these as overlapping contributions, not "
        "additive slices of one pie."
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
    )

    if split in ("train", "both"):
        print(f"\nRunning additional train-split pass for per-book/per-image stats "
              f"(headline metrics above stay val-only)...")
        train_loader = _build_eval_loader(vocab, "train")
        n_train_images = _accumulate_book_image_stats_pass(
            model, train_loader, "train", book_stats, image_stats, max_batches,
        )
        print(f"  train images evaluated: {n_train_images}")

    per_book_rows = print_and_save_per_book_metrics(
        book_stats=book_stats,
        out_path=out_dir / "per_book_metrics.csv",
    )

    plot_per_book_metric_distribution(
        per_book_rows, out_path=out_dir / "per_book_metric_distribution.png",
    )
    plot_book_difficulty_vs_f1(
        per_book_rows, out_path=out_dir / "book_difficulty_vs_f1.png",
    )

    per_image_rows = print_and_save_per_image_metrics(
        image_stats=image_stats,
        out_path=out_dir / "per_image_metrics.csv",
    )

    save_excluded_pages_table(out_path=out_dir / "excluded_pages.csv")

    if log_files:
        plot_loss_cer_curve(log_files, out_path=out_dir / "training_curve_loss_cer.png")
        plot_top1_top5_curve(log_files, out_path=out_dir / "training_curve_top1_top5.png")
    else:
        print("\nNo --log provided; skipping training-curve plots.")

    plot_confusion_matrix(
        error_counter=error_counter,
        gt_total_counter=gt_total_counter,
        vocab=vocab,
        out_path=out_dir / "confusion_matrix.png",
        top_n=40,
    )
    plot_confusion_matrix_clustered(
        error_counter=error_counter,
        gt_total_counter=gt_total_counter,
        vocab=vocab,
        out_path=out_dir / "confusion_matrix_clustered.png",
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
        plot_confusion_matrix_clustered(
            error_counter=hard_error_counter,
            gt_total_counter=gt_total_counter,
            vocab=vocab,
            out_path=out_dir / "confusion_matrix_hard_clustered.png",
            top_n=40,
        )

    # Confusion gallery: actual glyph crops for the top hard (non-kana-mixup)
    # confused pairs — favors genuine kanji-visual-similarity errors over
    # the already-understood hiragana/katakana script mixups.
    plot_confusion_gallery(
        pair_crop_examples=pair_crop_examples,
        class_correct_crops=class_correct_crops,
        error_counter=hard_error_counter if hard_error_counter else error_counter,
        vocab=vocab,
        out_path=out_dir / "confusion_gallery.png",
        top_n=12,
    )

    plot_stage_error_attribution(
        stage1_miss_rate=stage1_miss_rate,
        stage2_misclass_rate=stage2_misclass_rate,
        ordering_violation_rate_positives=ordering_violation_rate_positives,
        out_path=out_dir / "stage_error_attribution.png",
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

    # Kanji-focused confusion matrix: only rows where GT is kanji. This is
    # where genuine visual-similarity confusion lives (as opposed to the
    # kana-script mixups above), so both the frequency-ordered and
    # clustered views are useful here.
    kanji_error_counter = Counter({
        (gt, pr): cnt for (gt, pr), cnt in error_counter.items()
        if _char_script_type(_ids_to_text([gt], vocab)) == "kanji"
    })
    if kanji_error_counter:
        plot_confusion_matrix(
            error_counter=kanji_error_counter,
            gt_total_counter=gt_total_counter,
            vocab=vocab,
            out_path=out_dir / "confusion_matrix_kanji.png",
            top_n=40,
        )
        plot_confusion_matrix_clustered(
            error_counter=kanji_error_counter,
            gt_total_counter=gt_total_counter,
            vocab=vocab,
            out_path=out_dir / "confusion_matrix_kanji_clustered.png",
            top_n=40,
        )

    # -----------------------------------------------------------------------
    # Save summary JSON
    # -----------------------------------------------------------------------
    one_row_per_book = _one_row_per_book_preferring_val(per_book_rows)
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
        "pipeline_recall": pipeline_recall,
        "pipeline_precision": pipeline_precision,
        "pipeline_f1": pipeline_f1,
        "ordering_violation_rate": mean_ordering_viol,
        "stage1_miss_rate": stage1_miss_rate,
        "stage1_miss_count": int(stage1_miss_count),
        "stage2_misclass_rate": stage2_misclass_rate,
        "stage2_misclass_count": int(stage2_misclass_count),
        "ordering_violation_rate_positives": ordering_violation_rate_positives,
        "ordering_violation_count_positives": int(ordering_viol_pos_sum),
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
        # per_book_rows/per_image_rows can hold multiple rows per book
        # (train/val/both) once --split includes train — filter to one
        # row per book/image (preferring val, since headline metrics.json
        # stays val-only for comparability across runs) before summarizing,
        # so worst_books/total_books aren't inflated by duplicate rows.
        "worst_books": one_row_per_book[:10],
        "total_books": len(one_row_per_book),
        "worst_images": sorted(
            [r for r in per_image_rows if r["split"] == "val"] or per_image_rows,
            key=lambda r: r["pipeline_f1"],
        )[:10],
        "total_images_with_metrics": sum(1 for r in per_image_rows if r["split"] == "val") or len(per_image_rows),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n✅ metrics saved: {out_dir / 'metrics.json'}")
    print(f"✅ per-class CSV: {out_dir / 'per_class_errors.csv'}")
    print(f"✅ confusion matrix: {out_dir / 'confusion_matrix.png'}")


def main():
    parser = argparse.ArgumentParser(description="Validate SuminaNetRecognizer")
    parser.add_argument(
        "--ckpt",
        type=str,
        default=str(WEBSITE_CHECKPOINT_DIR),
        help="Path to SuminaNet checkpoint",
    )
    parser.add_argument(
        "--batches",
        type=int,
        default=None,
        help="Max number of batches (None = full split)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for metrics.json/per_class_errors.csv/confusion "
             "matrices. Default: checkpoints/suminanet_recognizer/validation/<timestamp>. "
             "Set this explicitly when validating a non-default checkpoint (e.g. "
             "an archived B/C/D variant) so results don't collide with the default "
             "checkpoint's validation history.",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        choices=["unet", "efficientnet_b2"],
        help="Backbone architecture to build before loading --ckpt onto it. "
             "Default: use the checkpoint's own saved backbone_type if present, "
             "else fall back to config.BACKBONE_TYPE. Required for checkpoints "
             "saved before backbone_type was recorded (e.g. an archived 'unet' "
             "checkpoint validated while config.py is set to 'efficientnet_b2') "
             "-- without it, the model is silently built with the wrong "
             "architecture and the checkpoint's backbone weights fail to load.",
    )
    parser.add_argument(
        "--context-mode",
        type=str,
        default=None,
        choices=["gru", "bigru"],
        help="Context-encoder mode to build before loading --ckpt onto it. "
             "Default: use config.STAGE2_CONTEXT_MODE. Needed when validating "
             "a checkpoint trained with a different context_mode than whatever "
             "is currently configured (e.g. an archived 'bigru' checkpoint "
             "while config.py is set to 'gru') -- without it, "
             "_load_compatible_state_dict raises rather than silently "
             "corrupting the mismatched RNN layer, so the run fails outright.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "both"],
        help="Which split(s) to cover in per_book_metrics.csv/per_image_metrics.csv "
             "and the new per-book plots. Headline metrics (metrics.json top-level "
             "fields, confusion matrices, stdout summary) always stay val-only "
             "regardless of this flag, for comparability across runs/checkpoints. "
             "Default: val (matches pre-existing behavior). 'both' also runs "
             "inference over the ~4,790 train images (~9x the compute of val "
             "alone) to show train-vs-val per-book performance side by side.",
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="Comma-separated stage-2 training log path(s) for this checkpoint, "
             "in chronological order (multiple for a resumed run), e.g. "
             "logs/training_suminanet/training-13979.out,logs/training_suminanet/training-14029.out. "
             "When given, writes training_curve_loss_cer.png and "
             "training_curve_top1_top5.png to --out-dir. Skipped if omitted.",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    log_files = [Path(p) for p in args.log.split(",")] if args.log else None

    run_validation(
        ckpt_path=ckpt_path,
        max_batches=args.batches,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        backbone=args.backbone,
        context_mode=args.context_mode,
        split=args.split,
        log_files=log_files,
    )


if __name__ == "__main__":
    main()
