"""Stage 2 (KuroNet) visualization helpers for thesis result analysis."""

from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from visualization.common import BAR_ALPHA, GRID_ALPHA, MARKER_SIZE, savefig

if TYPE_CHECKING:
    from utils.vocab import VocabManager


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _unicode_token_to_char(token: str) -> str:
    """Convert "U+XXXX" token to the actual Unicode character, or return token as-is."""
    if token.startswith("U+"):
        try:
            return chr(int(token[2:], 16))
        except ValueError:
            pass
    return token


_HIRAGANA = 0
_KATAKANA = 1
_KANJI    = 2
_OTHER    = 3
_SCRIPT_NAMES = ["Hiragana", "Katakana", "Kanji", "Other"]


def _char_to_script(ch: str) -> int:
    if not ch:
        return _OTHER
    ch = _unicode_token_to_char(ch)
    ch_nfc = unicodedata.normalize("NFC", ch)
    cp = ord(ch_nfc[0])
    if 0x3041 <= cp <= 0x3096:
        return _HIRAGANA
    if 0x30A1 <= cp <= 0x30F6:
        return _KATAKANA
    if (0x4E00 <= cp <= 0x9FFF
            or 0x3400 <= cp <= 0x4DBF
            or 0xF900 <= cp <= 0xFAFF):
        return _KANJI
    return _OTHER


def _display_char(token: str) -> str:
    """Convert token to a printable character for axis labels."""
    ch = _unicode_token_to_char(token)
    return ch if ch else token[:6]


# ---------------------------------------------------------------------------
# Plot 1 — Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    error_counter: dict,
    gt_total_counter: dict,
    vocab: "VocabManager",
    top_n: int = 40,
    out_path: "str | Path" = "confusion_matrix.png",
) -> Path:
    """
    Two-panel figure:
      Left:  top_n × top_n heatmap of predicted vs ground-truth character pairs.
      Right: top-20 most confused pairs as a horizontal bar chart.
    """
    special_ids = {vocab.pad_id, vocab.sos_id, vocab.eos_id, vocab.unk_id, vocab.bg_id}

    # Select top_n most frequent GT classes (excluding special tokens)
    sorted_gts = sorted(
        ((gid, cnt) for gid, cnt in gt_total_counter.items() if gid not in special_ids),
        key=lambda x: -x[1],
    )
    top_ids = [gid for gid, _ in sorted_gts[:top_n]]
    id_to_idx = {gid: i for i, gid in enumerate(top_ids)}
    labels = [_display_char(vocab.id2char.get(gid, "?")) for gid in top_ids]

    # Build dense confusion matrix
    mat = np.zeros((top_n, top_n), dtype=np.float32)
    for (gt_id, pred_id), cnt in error_counter.items():
        if gt_id in id_to_idx and pred_id in id_to_idx:
            mat[id_to_idx[gt_id], id_to_idx[pred_id]] += cnt

    # Top-20 confused pairs as text labels
    top_pairs = sorted(
        (
            (f"{_display_char(vocab.id2char.get(gt,'?'))}→{_display_char(vocab.id2char.get(pr,'?'))}", cnt)
            for (gt, pr), cnt in error_counter.items()
            if gt not in special_ids and pr not in special_ids
        ),
        key=lambda x: -x[1],
    )[:20]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Heatmap
    im = ax1.imshow(mat, cmap="YlOrRd", aspect="auto")
    ax1.set_xticks(range(top_n))
    ax1.set_yticks(range(top_n))
    ax1.set_xticklabels(labels, rotation=90, fontsize=6)
    ax1.set_yticklabels(labels, fontsize=6)
    ax1.set_xlabel("Predicted character")
    ax1.set_ylabel("Ground-truth character")
    ax1.set_title(f"Top-{top_n} character confusion matrix")
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    # Bar chart
    pair_labels = [p for p, _ in top_pairs]
    pair_counts = [c for _, c in top_pairs]
    y_pos = range(len(pair_labels))
    ax2.barh(y_pos, pair_counts, align="center", color="steelblue", alpha=BAR_ALPHA)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(pair_labels, fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel("Error count")
    ax2.set_title("Top-20 most confused pairs (GT→Pred)")
    ax2.grid(axis="x", alpha=GRID_ALPHA)
    for i, v in enumerate(pair_counts):
        ax2.text(v + 0.3, i, str(v), va="center", fontsize=8)

    fig.tight_layout()
    return savefig(fig, out_path)


# ---------------------------------------------------------------------------
# Plot 2 — CER breakdown by script type
# ---------------------------------------------------------------------------

def plot_cer_by_script(
    error_counter: dict,
    gt_total_counter: dict,
    vocab: "VocabManager",
    rare_thresh: int = 50,
    out_path: "str | Path" = "cer_by_script.png",
) -> Path:
    """
    Bar chart: character error rate broken down by script type and rarity.
    Groups: Hiragana | Katakana | Kanji (common) | Kanji (rare).
    """
    special_ids = {vocab.pad_id, vocab.sos_id, vocab.eos_id, vocab.unk_id, vocab.bg_id}

    # Accumulate per-group: (errors, total GT)
    groups = {
        "Hiragana":     [0, 0],
        "Katakana":     [0, 0],
        "Kanji\n(common)": [0, 0],
        "Kanji\n(rare)":   [0, 0],
        "Other":        [0, 0],
    }

    for gt_id, total in gt_total_counter.items():
        if gt_id in special_ids:
            continue
        ch = vocab.id2char.get(gt_id, "")
        stype = _char_to_script(ch)
        errors = sum(cnt for (gid, _), cnt in error_counter.items() if gid == gt_id)

        if stype == _HIRAGANA:
            groups["Hiragana"][0] += errors
            groups["Hiragana"][1] += total
        elif stype == _KATAKANA:
            groups["Katakana"][0] += errors
            groups["Katakana"][1] += total
        elif stype == _KANJI:
            key = "Kanji\n(rare)" if total < rare_thresh else "Kanji\n(common)"
            groups[key][0] += errors
            groups[key][1] += total
        else:
            groups["Other"][0] += errors
            groups["Other"][1] += total

    group_names = list(groups.keys())
    error_rates = [
        (errs / max(1, tot)) * 100
        for errs, tot in groups.values()
    ]
    totals = [tot for _, tot in groups.values()]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#DD8452", "#8172B2"]
    bars = ax.bar(group_names, error_rates, color=colors, alpha=BAR_ALPHA)

    for bar, rate, tot in zip(bars, error_rates, totals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            f"{rate:.1f}%\n(n={tot:,})",
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_ylabel("Character error rate (%)")
    ax.set_title(f"CER by script type  (rare = < {rare_thresh} GT occurrences)")
    ax.grid(axis="y", alpha=GRID_ALPHA)
    ax.set_ylim(0, max(error_rates) * 1.25 + 1)
    fig.tight_layout()
    return savefig(fig, out_path)


# ---------------------------------------------------------------------------
# Plot 3 — Top-K error analysis
# ---------------------------------------------------------------------------

def plot_topk_errors(
    topk_in_gt: list,
    out_path: "str | Path" = "topk_errors.png",
) -> Path:
    """
    For wrong predictions: what fraction had the correct answer in top-K?
    topk_in_gt: list of 5-element bool lists — one per wrong prediction.
    """
    if not topk_in_gt:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "No wrong predictions", ha="center", va="center", transform=ax.transAxes)
        return savefig(fig, out_path)

    arr = np.array(topk_in_gt, dtype=bool)   # (N, 5)
    n_wrong = len(arr)
    k_values = list(range(1, 6))
    fractions = [float(arr[:, k - 1].sum()) / n_wrong for k in k_values]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(k_values, [f * 100 for f in fractions], "b-o", markersize=MARKER_SIZE + 2)
    for k, f in zip(k_values, fractions):
        ax.annotate(f"{f*100:.1f}%", (k, f * 100), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)

    ax.set_xticks(k_values)
    ax.set_xlabel("K")
    ax.set_ylabel("% of wrong predictions where GT ∈ top-K")
    ax.set_title(
        f"Top-K recall for incorrect predictions  (n={n_wrong:,} errors)\n"
        "K=1 is 0% by definition (argmax is the wrong prediction)"
    )
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=GRID_ALPHA)
    fig.tight_layout()
    return savefig(fig, out_path)


# ---------------------------------------------------------------------------
# Plot 4 — Per-page CER histogram
# ---------------------------------------------------------------------------

def plot_per_page_cer(
    per_image_cer: list,
    out_path: "str | Path" = "per_page_cer.png",
) -> Path:
    """Histogram of CER values across all validation images."""
    if not per_image_cer:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No CER data", ha="center", va="center", transform=ax.transAxes)
        return savefig(fig, out_path)

    arr = np.array(per_image_cer, dtype=np.float32)
    median_cer = float(np.median(arr))
    p90_cer    = float(np.percentile(arr, 90))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(arr, bins=40, color="steelblue", alpha=BAR_ALPHA, edgecolor="white")
    ax.axvline(median_cer, color="royalblue", linestyle="--", linewidth=1.5,
               label=f"Median {median_cer:.3f}")
    ax.axvline(p90_cer, color="crimson", linestyle="--", linewidth=1.5,
               label=f"P90 {p90_cer:.3f}")
    ax.set_xlabel("CER per image")
    ax.set_ylabel("Number of images")
    ax.set_title(
        f"Per-page CER distribution  "
        f"(n={len(arr)}, mean={arr.mean():.3f}, median={median_cer:.3f}, P90={p90_cer:.3f})"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=GRID_ALPHA)
    fig.tight_layout()
    return savefig(fig, out_path)


# ---------------------------------------------------------------------------
# Plot 5 — Confidence-color bounding box visualization
# ---------------------------------------------------------------------------

def draw_confidence_boxes(
    image_path: "str | Path",
    chars: list,
    out_path: "str | Path" = "confidence_vis.png",
    low_thresh: float = 0.10,
    mid_thresh: float = 0.50,
) -> Path:
    """
    Draw bounding boxes on the image, colored by confidence score.
      green  : score >= mid_thresh
      orange : low_thresh <= score < mid_thresh
      red    : score < low_thresh

    chars: list of {char, box: [x1,y1,x2,y2], score} as returned by infer.run_inference().
    """
    import cv2

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # Colors in BGR
    COLOR_GREEN  = (34, 139, 34)
    COLOR_ORANGE = (0, 165, 255)
    COLOR_RED    = (0, 0, 220)

    tier_counts = {"green": 0, "orange": 0, "red": 0}

    for item in chars:
        box   = item["box"]        # [x1, y1, x2, y2]
        score = float(item["score"])
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

        if score >= mid_thresh:
            color = COLOR_GREEN
            tier_counts["green"] += 1
        elif score >= low_thresh:
            color = COLOR_ORANGE
            tier_counts["orange"] += 1
        else:
            color = COLOR_RED
            tier_counts["red"] += 1

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=2)

    # Legend in top-left corner
    legend_lines = [
        (f">=50%: {tier_counts['green']} chars",   COLOR_GREEN),
        (f"10-50%: {tier_counts['orange']} chars", COLOR_ORANGE),
        (f"<10%: {tier_counts['red']} chars",       COLOR_RED),
    ]
    lx, ly = 10, 20
    for i, (text, color) in enumerate(legend_lines):
        y = ly + i * 22
        cv2.rectangle(img, (lx, y - 12), (lx + 160, y + 5), (255, 255, 255), -1)
        cv2.putText(img, text, (lx + 2, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 2, cv2.LINE_AA)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return out_path
