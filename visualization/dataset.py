"""Dataset introduction visualizations for thesis (section 7.5.5).

All functions take a pre-computed `label_counts` Counter (label -> int)
so the annotation scan runs once in the entry-point script.
"""

from __future__ import annotations

import collections
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from visualization.common import BAR_ALPHA, GRID_ALPHA, savefig

# ---------------------------------------------------------------------------
# Font helper — Droid Sans Fallback covers all hiragana/katakana/kanji
# ---------------------------------------------------------------------------

_DROID_PATH = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"


def _jp_font(size: int = 10) -> fm.FontProperties | None:
    p = Path(_DROID_PATH)
    if p.exists():
        return fm.FontProperties(fname=str(p), size=size)
    return None


# ---------------------------------------------------------------------------
# Script classification
# ---------------------------------------------------------------------------

def _codepoint(label: str) -> int:
    """Convert 'U+306E' → 0x306E."""
    try:
        return int(label.replace("U+", ""), 16)
    except ValueError:
        return -1


def _script(label: str) -> str:
    cp = _codepoint(label)
    if 0x3041 <= cp <= 0x309F:
        return "Hiragana"
    if 0x30A0 <= cp <= 0x30FF:
        return "Katakana"
    if (0x4E00 <= cp <= 0x9FFF) or (0x3400 <= cp <= 0x4DBF) or \
       (0x20000 <= cp <= 0x2A6DF) or (0x2A700 <= cp <= 0x2CEAF) or \
       (0xF900 <= cp <= 0xFAFF):
        return "Kanji"
    return "Other"


def _label_to_char(label: str) -> str:
    """Convert 'U+306E' → actual Unicode character."""
    try:
        return chr(_codepoint(label))
    except (ValueError, OverflowError):
        return "?"


# ---------------------------------------------------------------------------
# 1. Zipf / frequency-rank curve
# ---------------------------------------------------------------------------

def plot_zipf_curve(
    label_counts: collections.Counter,
    out_path: "str | Path" = "zipf_curve.png",
) -> Path:
    """
    Log-log rank vs frequency plot showing the Zipf distribution.
    Annotates: top-10 / top-100 / top-500 cutoffs and their cumulative coverage.
    """
    freqs = sorted(label_counts.values(), reverse=True)
    ranks = np.arange(1, len(freqs) + 1)
    total = sum(freqs)
    cum_frac = np.cumsum(freqs) / total

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: log-log frequency vs rank ---
    ax1.loglog(ranks, freqs, "b-", linewidth=1.5, alpha=0.85)
    ax1.set_xlabel("Character rank (log)")
    ax1.set_ylabel("Frequency (log)")
    ax1.set_title("Character frequency distribution (Zipf law)")
    ax1.grid(True, which="both", alpha=GRID_ALPHA)

    # Annotate key cutoffs
    for k, color, label_txt in [
        (10,  "red",    "Top 10"),
        (100, "orange", "Top 100"),
        (500, "green",  "Top 500"),
    ]:
        if k <= len(freqs):
            ax1.axvline(k, color=color, linestyle="--", linewidth=1.1)
            ax1.text(k * 1.05, freqs[0] * 0.6 ** ([10, 100, 500].index(k) + 1),
                     f"{label_txt}\n{cum_frac[k-1]*100:.1f}% coverage",
                     color=color, fontsize=8)

    ax1.axhline(5, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)
    ax1.text(1.5, 6, "freq = 5 (rare threshold)", fontsize=7, color="grey")

    # --- Right: cumulative coverage curve ---
    ax2.plot(ranks, cum_frac * 100, "b-", linewidth=1.5)
    ax2.set_xlabel("Vocabulary size (top-K characters)")
    ax2.set_ylabel("Cumulative character coverage (%)")
    ax2.set_title("Cumulative coverage vs. vocabulary size")
    ax2.set_xscale("log")
    ax2.grid(True, which="both", alpha=GRID_ALPHA)
    ax2.set_ylim(0, 101)

    for k, color in [(100, "orange"), (500, "green"), (1000, "purple")]:
        if k <= len(cum_frac):
            cov = cum_frac[k - 1] * 100
            ax2.axvline(k, color=color, linestyle="--", linewidth=1.0)
            ax2.axhline(cov, color=color, linestyle=":", linewidth=0.8, alpha=0.5)
            ax2.scatter([k], [cov], color=color, s=60, zorder=5)
            ax2.text(k * 1.08, cov - 3, f"K={k}: {cov:.1f}%",
                     color=color, fontsize=8)

    fig.suptitle(
        f"Dataset: {len(label_counts):,} unique classes  |  "
        f"{total:,} total instances  |  "
        f"{sum(1 for v in label_counts.values() if v == 1):,} hapax legomena",
        fontsize=10,
    )
    fig.tight_layout()
    return savefig(fig, out_path)


# ---------------------------------------------------------------------------
# 2. Script-type distribution
# ---------------------------------------------------------------------------

def plot_script_distribution(
    label_counts: collections.Counter,
    split_label_counts: "dict[str, collections.Counter] | None" = None,
    out_path: "str | Path" = "script_distribution.png",
) -> Path:
    """
    2-panel: (left) pie chart of character instances by script;
             (right) grouped bar for unique classes per script per split.
    """
    # Aggregate instance counts and class counts per script
    instance_by_script: dict[str, int] = collections.defaultdict(int)
    classes_by_script:  dict[str, int] = collections.defaultdict(int)
    for lbl, cnt in label_counts.items():
        s = _script(lbl)
        instance_by_script[s] += cnt
        classes_by_script[s] += 1

    script_order = ["Hiragana", "Katakana", "Kanji", "Other"]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#DD8452"]
    instances = [instance_by_script.get(s, 0) for s in script_order]
    classes   = [classes_by_script.get(s, 0) for s in script_order]
    total_inst = sum(instances)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # --- Left: pie of instances ---
    wedge_labels = [
        f"{s}\n{instances[i]:,}\n({instances[i]/total_inst*100:.1f}%)"
        for i, s in enumerate(script_order)
        if instances[i] > 0
    ]
    non_zero_inst  = [v for v in instances if v > 0]
    non_zero_color = [colors[i] for i, v in enumerate(instances) if v > 0]
    ax1.pie(
        non_zero_inst,
        labels=wedge_labels,
        colors=non_zero_color,
        autopct=None,
        startangle=140,
        wedgeprops={"linewidth": 0.8, "edgecolor": "white"},
    )
    ax1.set_title("Character instances by script type")

    # --- Right: grouped bar of unique classes ---
    x = np.arange(len(script_order))
    width = 0.35

    if split_label_counts:
        # Show train vs val classes side-by-side
        train_c = split_label_counts.get("train", collections.Counter())
        val_c   = split_label_counts.get("val",   collections.Counter())
        train_classes = [
            len([l for l in train_c if _script(l) == s]) for s in script_order
        ]
        val_classes = [
            len([l for l in val_c   if _script(l) == s]) for s in script_order
        ]
        b1 = ax2.bar(x - width / 2, train_classes, width, label="Train",
                     color=colors, alpha=BAR_ALPHA)
        b2 = ax2.bar(x + width / 2, val_classes,   width, label="Val",
                     color=colors, alpha=0.45, edgecolor="k", linewidth=0.7)
        for bar in list(b1) + list(b2):
            h = bar.get_height()
            if h > 0:
                ax2.text(bar.get_x() + bar.get_width() / 2, h + 5,
                         f"{h:,}", ha="center", va="bottom", fontsize=7)
        ax2.legend(fontsize=9)
        ax2.set_title("Unique character classes per script (train vs val)")
    else:
        bars = ax2.bar(x, classes, color=colors, alpha=BAR_ALPHA)
        for bar, c in zip(bars, classes):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                     f"{c:,}", ha="center", va="bottom", fontsize=9)
        ax2.set_title("Unique character classes per script type")

    ax2.set_xticks(x)
    ax2.set_xticklabels(script_order)
    ax2.set_ylabel("Number of unique character classes")
    ax2.grid(axis="y", alpha=GRID_ALPHA)

    fig.suptitle("Dataset script-type breakdown", fontsize=11)
    fig.tight_layout()
    return savefig(fig, out_path)


# ---------------------------------------------------------------------------
# 3. Class-imbalance long-tail histogram
# ---------------------------------------------------------------------------

def plot_class_imbalance(
    label_counts: collections.Counter,
    rare_thresh: int = 50,
    out_path: "str | Path" = "class_imbalance.png",
) -> Path:
    """
    2-panel: (left) histogram of classes per frequency bucket (stacked by script);
             (right) Lorenz-style bar showing what % of instances come from top/mid/long-tail classes.
    """
    BUCKETS = [
        (1,   1,   "1\n(hapax)"),
        (2,   4,   "2–4"),
        (5,   19,  "5–19"),
        (20,  49,  "20–49"),
        (50,  99,  "50–99"),
        (100, 499, "100–499"),
        (500, 9999999, "≥500"),
    ]
    scripts = ["Hiragana", "Katakana", "Kanji", "Other"]
    colors  = ["#4C72B0", "#55A868", "#C44E52", "#DD8452"]

    # Count classes per (bucket × script)
    bucket_script_classes = [[0] * len(scripts) for _ in BUCKETS]
    bucket_instances      = [0] * len(BUCKETS)

    for lbl, cnt in label_counts.items():
        s_idx = scripts.index(_script(lbl))
        for b_idx, (lo, hi, _) in enumerate(BUCKETS):
            if lo <= cnt <= hi:
                bucket_script_classes[b_idx][s_idx] += 1
                bucket_instances[b_idx] += cnt
                break

    x      = np.arange(len(BUCKETS))
    labels = [b[2] for b in BUCKETS]
    total_inst = sum(bucket_instances)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: stacked bar — classes per bucket, coloured by script ---
    bottom = np.zeros(len(BUCKETS))
    for s_idx, (s, c) in enumerate(zip(scripts, colors)):
        vals = [bucket_script_classes[b][s_idx] for b in range(len(BUCKETS))]
        ax1.bar(x, vals, bottom=bottom, color=c, alpha=BAR_ALPHA, label=s)
        bottom += np.array(vals, dtype=float)

    for xi, tot in enumerate(bottom):
        if tot > 0:
            ax1.text(xi, tot + 5, f"{int(tot):,}", ha="center", va="bottom", fontsize=8)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_xlabel("Frequency bucket (occurrences in dataset)")
    ax1.set_ylabel("Number of unique character classes")
    ax1.set_title("Class distribution by frequency bucket")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=GRID_ALPHA)

    # --- Right: instance share per bucket ---
    inst_pcts = [v / total_inst * 100 for v in bucket_instances]
    bar_colors = ["#C44E52" if b[1] <= 19 else "#DD8452" if b[1] <= 99 else "#4C72B0"
                  for b in BUCKETS]
    bars = ax2.bar(x, inst_pcts, color=bar_colors, alpha=BAR_ALPHA)
    for bar, pct in zip(bars, inst_pcts):
        if pct > 0.1:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                     f"{pct:.1f}%", ha="center", va="bottom", fontsize=8)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_xlabel("Frequency bucket")
    ax2.set_ylabel("Share of total character instances (%)")
    ax2.set_title("Instance share per frequency bucket")
    ax2.grid(axis="y", alpha=GRID_ALPHA)

    # Add rare-threshold line on left panel
    rare_bucket_idx = next(
        (b_idx for b_idx, (lo, hi, _) in enumerate(BUCKETS) if lo <= rare_thresh <= hi), None
    )
    if rare_bucket_idx is not None:
        ax1.axvline(rare_bucket_idx + 0.5, color="grey", linestyle="--",
                    linewidth=1.2, label=f"rare threshold ({rare_thresh})")
        ax1.legend(fontsize=8)

    n_rare   = sum(1 for v in label_counts.values() if v < rare_thresh)
    n_common = len(label_counts) - n_rare
    fig.suptitle(
        f"Class imbalance: {n_rare:,} rare classes (< {rare_thresh} occurrences)  |  "
        f"{n_common:,} common classes  |  "
        f"top-10 chars = {sum(v for _, v in label_counts.most_common(10))/total_inst*100:.1f}% of instances",
        fontsize=9,
    )
    fig.tight_layout()
    return savefig(fig, out_path)


# ---------------------------------------------------------------------------
# 4. Top-N character frequency bar chart (with actual glyphs)
# ---------------------------------------------------------------------------

def plot_top_characters(
    label_counts: collections.Counter,
    top_n: int = 40,
    rare_thresh: int = 50,
    out_path: "str | Path" = "top_characters.png",
) -> Path:
    """
    Horizontal bar chart of the top-N most frequent characters.
    Uses DroidSansFallbackFull for rendering the actual glyphs.
    Bars are colour-coded by script type.
    """
    top = label_counts.most_common(top_n)
    labels  = [t[0] for t in top]
    counts  = [t[1] for t in top]
    chars   = [_label_to_char(l) for l in labels]
    scripts = [_script(l) for l in labels]

    script_color = {
        "Hiragana": "#4C72B0",
        "Katakana": "#55A868",
        "Kanji":    "#C44E52",
        "Other":    "#DD8452",
    }
    bar_colors = [script_color.get(s, "#999999") for s in scripts]

    jp_fp = _jp_font(size=9)

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.28)))
    y_pos = np.arange(len(top))[::-1]   # highest freq at top

    bars = ax.barh(y_pos, counts, color=bar_colors, alpha=BAR_ALPHA, height=0.7)

    # Y-tick labels: glyph + codepoint
    tick_labels = [f"  {c}  {l}" for c, l in zip(chars, labels)]
    ax.set_yticks(y_pos)

    if jp_fp:
        for yi, (char, label_code) in zip(y_pos, zip(chars, labels)):
            # Draw character in Japanese font
            ax.text(-max(counts) * 0.002, yi, char, fontproperties=jp_fp,
                    ha="right", va="center", fontsize=11)
            ax.text(-max(counts) * 0.12, yi, label_code,
                    ha="right", va="center", fontsize=7, color="grey")
        ax.set_yticklabels([""] * len(top))
        ax.set_xlim(-max(counts) * 0.15, max(counts) * 1.12)
    else:
        ax.set_yticklabels(tick_labels, fontsize=8)

    # Frequency labels on bars
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_width() + max(counts) * 0.005, bar.get_y() + bar.get_height() / 2,
                f"{cnt:,}", va="center", fontsize=7)

    # Legend for script colours
    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=c, alpha=BAR_ALPHA, label=s)
                  for s, c in script_color.items()]
    ax.legend(handles=legend_els, fontsize=8, loc="lower right")

    ax.set_xlabel("Frequency (total occurrences in dataset)")
    ax.set_title(f"Top-{top_n} most frequent characters in the dataset")
    ax.grid(axis="x", alpha=GRID_ALPHA)
    ax.axvline(rare_thresh, color="grey", linestyle=":", linewidth=0.8)

    fig.tight_layout()
    return savefig(fig, out_path)


# ---------------------------------------------------------------------------
# 5. Characters-per-image distribution
# ---------------------------------------------------------------------------

def plot_chars_per_image(
    chars_per_image: list[int],
    split_chars_per_image: "dict[str, list[int]] | None" = None,
    out_path: "str | Path" = "chars_per_image.png",
) -> Path:
    """
    Histogram of character count per page, optionally split by train/val.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    if split_chars_per_image:
        colors_map = {"train": "#4C72B0", "val": "#C44E52"}
        for s, vals in split_chars_per_image.items():
            ax.hist(vals, bins=40, alpha=0.55, label=f"{s} (n={len(vals):,})",
                    color=colors_map.get(s, "grey"))
    else:
        ax.hist(chars_per_image, bins=40, color="#4C72B0", alpha=BAR_ALPHA)

    median_v = float(np.median(chars_per_image))
    mean_v   = float(np.mean(chars_per_image))
    ax.axvline(median_v, color="crimson", linestyle="--", linewidth=1.5,
               label=f"Median = {median_v:.0f}")
    ax.axvline(mean_v, color="orange", linestyle="--", linewidth=1.5,
               label=f"Mean = {mean_v:.0f}")

    ax.set_xlabel("Characters per page image")
    ax.set_ylabel("Number of images")
    ax.set_title("Distribution of character count per page")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=GRID_ALPHA)

    p10 = int(np.percentile(chars_per_image, 10))
    p90 = int(np.percentile(chars_per_image, 90))
    ax.text(0.98, 0.95,
            f"P10={p10}  median={median_v:.0f}  P90={p90}\nmax={max(chars_per_image)}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    fig.tight_layout()
    return savefig(fig, out_path)


# ---------------------------------------------------------------------------
# 6. Dataset split overview (summary bar)
# ---------------------------------------------------------------------------

def plot_split_overview(
    split_stats: "dict[str, dict]",
    out_path: "str | Path" = "split_overview.png",
) -> Path:
    """
    3-metric grouped bar: images / characters / unique classes per split.
    split_stats = {'train': {'images':..., 'chars':..., 'unique':...}, ...}
    """
    splits  = list(split_stats.keys())
    images  = [split_stats[s]["images"]  for s in splits]
    chars   = [split_stats[s]["chars"]   for s in splits]
    unique  = [split_stats[s]["unique"]  for s in splits]

    x = np.arange(len(splits))
    width = 0.25
    colors = ["#4C72B0", "#55A868", "#C44E52"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, vals, title, color in zip(
        axes,
        [images, chars, unique],
        ["Page images", "Total character instances", "Unique character classes"],
        colors,
    ):
        bars = ax.bar(x, vals, color=color, alpha=BAR_ALPHA, width=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    f"{v:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([s.capitalize() for s in splits], fontsize=11)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=GRID_ALPHA)
        ax.set_ylim(0, max(vals) * 1.18)

    fig.suptitle("Dataset split summary", fontsize=12)
    fig.tight_layout()
    return savefig(fig, out_path)


# ---------------------------------------------------------------------------
# 7. Augmentation comparison — classic vs. rare-character strength
# ---------------------------------------------------------------------------

def plot_augmentation_comparison(
    image_path: "str | Path",
    crop_box: "tuple[int, int, int, int] | None" = None,
    seed: int = 0,
    out_path: "str | Path" = "augmentation_comparison.png",
) -> Path:
    """
    Side-by-side comparison of an original crop, the standard ("classic")
    train-time augmentation applied to every training image, and the
    stronger rare-character augmentation from
    utils.char_augmentation.build_rare_char_transform (Option A), which is
    applied only to images that contain at least one rare character.

    Both pixel-level transforms are drawn from the exact same code paths
    used at training time (utils/__init__.py and utils/char_augmentation.py),
    with the ToTensor/Normalize steps stripped so the result stays a
    displayable image.

    Args:
        image_path: path to a page image (e.g. assets/data/<book>/images/<id>.jpg)
        crop_box:   optional (x1, y1, x2, y2) region to crop before augmenting,
                    so individual glyphs are legible in the figure
        seed:       torch RNG seed, applied before each transform so both
                    panels sample from a comparable random draw
        out_path:   output image path
    """
    import torch
    import torchvision.transforms as T

    from utils.char_augmentation import build_rare_char_transform

    img = Image.open(image_path).convert("RGB")
    if crop_box is not None:
        img = img.crop(crop_box)

    classic_transform = T.Compose([
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    ])
    rare_transform = T.Compose([
        t for t in build_rare_char_transform().transforms
        if not isinstance(t, (T.ToTensor, T.Normalize))
    ])

    torch.manual_seed(seed)
    classic_img = classic_transform(img)
    torch.manual_seed(seed)
    rare_img = rare_transform(img)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    panels = [
        (img,         "Original"),
        (classic_img, "Classic augmentation\n(ColorJitter ±0.2, applied to all training images)"),
        (rare_img,    "Rare-character augmentation\n(ColorJitter ±0.4 + GaussianBlur + RandomGrayscale)"),
    ]
    for ax, (im, title) in zip(axes, panels):
        ax.imshow(im)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig.suptitle(f"Augmentation strength comparison — {Path(image_path).name}", fontsize=12)
    fig.tight_layout()
    return savefig(fig, out_path)


# ---------------------------------------------------------------------------
# 8. Copy-paste rare-kanji augmentation (Option C)
# ---------------------------------------------------------------------------

def plot_copy_paste_augmentation(
    image_path: "str | Path",
    boxes: list,
    labels: list,
    rare_chars: "set[str] | frozenset[str]",
    crop_db: "dict[str, list]",
    n_paste: int = 3,
    seed: int = 0,
    out_path: "str | Path" = "copy_paste_augmentation.png",
) -> Path:
    """
    Side-by-side comparison of an original page and the same page after
    utils.char_augmentation.copy_paste_rare_chars (Option C) pastes a few
    rare-kanji crops onto it. Pasted characters are highlighted with a red
    box and their glyph + codepoint so the injected instances stand out.

    Args:
        image_path: path to the page image to paste onto
        boxes:      existing GT boxes for this page, [[x1,y1,x2,y2], ...]
        labels:     existing GT labels for this page
        rare_chars: frozenset of rare character labels (paste candidates)
        crop_db:    {char_label: [crop_array, ...]}, e.g. from a small scan
                    of annotations for rare kanji (see build_rare_char_crop_db
                    for the general-purpose version)
        n_paste:    number of rare-kanji crops to paste
        seed:       Python `random` seed so the pasted positions/chars are
                    reproducible across runs
        out_path:   output image path
    """
    import random

    import cv2
    from matplotlib.patches import Rectangle

    from utils.char_augmentation import copy_paste_rare_chars

    img_pil = Image.open(image_path).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    random.seed(seed)
    pasted_bgr, new_boxes, new_labels = copy_paste_rare_chars(
        image=img_bgr,
        boxes=boxes,
        labels=labels,
        crop_db=crop_db,
        rare_chars=frozenset(rare_chars),
        n_paste=n_paste,
    )
    pasted_rgb = cv2.cvtColor(pasted_bgr, cv2.COLOR_BGR2RGB)

    n_before = len(labels)
    pasted = list(zip(new_boxes[n_before:], new_labels[n_before:]))

    fig, axes = plt.subplots(1, 2, figsize=(13, 7.5))
    axes[0].imshow(img_pil)
    axes[0].set_title("Original", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(pasted_rgb)
    axes[1].set_title(f"Copy-paste augmentation\n(+{len(pasted)} rare kanji pasted)", fontsize=11)
    axes[1].axis("off")

    jp_fp = _jp_font(size=13)
    for box, lbl in pasted:
        x1, y1, x2, y2 = box
        axes[1].add_patch(Rectangle(
            (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linewidth=1.6,
        ))
        char = _label_to_char(lbl)
        # Glyph and codepoint are drawn separately: DroidSansFallback renders
        # Latin/ASCII as tofu boxes, so the "U+XXXX" part needs the default font.
        axes[1].text(x1, y1 - 8, char, color="red", fontsize=13,
                     fontproperties=jp_fp, ha="left", va="bottom")
        axes[1].text(x1 + 18, y1 - 8, lbl, color="red", fontsize=8,
                     ha="left", va="bottom")

    fig.suptitle(f"Rare-kanji copy-paste augmentation — {Path(image_path).name}", fontsize=12)
    fig.tight_layout()
    return savefig(fig, out_path)
