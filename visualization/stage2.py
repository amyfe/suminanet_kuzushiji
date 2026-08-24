"""Stage 2 (SuminaNet) visualization helpers for thesis result analysis."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from visualization.common import BAR_ALPHA, GRID_ALPHA, MARKER_SIZE, cjk_font_prop, pastel_color, savefig

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
    cjk6 = cjk_font_prop(6)
    cjk9 = cjk_font_prop(9)

    # Heatmap
    im = ax1.imshow(mat, cmap="YlOrRd", aspect="auto")
    ax1.set_xticks(range(top_n))
    ax1.set_yticks(range(top_n))
    if cjk6 is not None:
        ax1.set_xticklabels(labels, rotation=90)
        ax1.set_yticklabels(labels)
        for txt in ax1.get_xticklabels():
            txt.set_fontproperties(cjk6)
        for txt in ax1.get_yticklabels():
            txt.set_fontproperties(cjk6)
    else:
        # Fallback: show Unicode codepoints (readable without a CJK font)
        cp_labels = [f"U+{ord(c):04X}" if c else "?" for c in labels]
        ax1.set_xticklabels(cp_labels, rotation=90, fontsize=6)
        ax1.set_yticklabels(cp_labels, fontsize=6)
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
    if cjk9 is not None:
        ax2.set_yticklabels(pair_labels)
        for txt in ax2.get_yticklabels():
            txt.set_fontproperties(cjk9)
    else:
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

    # Matches the canonical script-type palette used across dataset_analysis
    # and checkpoint_compare.py (hiragana/katakana/kanji/other), so the same
    # script always reads as the same color across every plot suite. Kanji
    # (rare) is a pastel tint of Kanji (common)'s red rather than its own
    # hue, so it reads as "still kanji" instead of colliding with another
    # script's canonical color.
    group_colors = {
        "Hiragana":        "#4C72B0",
        "Katakana":        "#55A868",
        "Kanji\n(common)": "#C44E52",
        "Kanji\n(rare)":   pastel_color("#C44E52"),
        "Other":           "#DD8452",
    }
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [group_colors[name] for name in group_names]
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


# ---------------------------------------------------------------------------
# Learning curves (train/val over epochs) — justifies the chosen epoch budget
# ---------------------------------------------------------------------------

_NUM = r"(?:[\d.]+|nan)"  # training instability occasionally logs literal "nan"
_EPOCH_RE = re.compile(
    rf"Epoch\s+(\d+)/\d+\s+\|\s+Train loss=({_NUM})\s+\(char={_NUM},\s*bg={_NUM},"
    rf"\s*delta={_NUM},\s*score={_NUM}\)\s+\|\s+train_top1=({_NUM})"
)
_VAL_LOSS_RE = re.compile(rf"Val \| loss=({_NUM})")
_VAL_METRICS_RE = re.compile(
    rf"Val \| top1=({_NUM})\s+top5=({_NUM})\s+CER=({_NUM})\s+coverage=({_NUM})"
)
_BEST_RE = re.compile(r"saved best: suminanet_best\.pt \(score=([\d.]+)\)")
_EARLY_STOP_RE = re.compile(r"Early stopping: (\d+) epochs without improvement\. best=([\d.]+)")


def _parse_suminanet_log(log_file: "str | Path") -> tuple[list[dict], int | None]:
    """
    Parse train_stage2_suminanet.log epoch/val summary lines.

    Each epoch is logged as three consecutive lines: an "Epoch N/M | Train
    loss=..." line, a "Val | loss=..." line, and a "Val | top1=... CER=..."
    line. Returns (records, early_stop_epoch).
    """
    records: list[dict] = []
    pending: dict = {}
    early_stop_epoch: int | None = None
    with open(log_file, encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = _EPOCH_RE.search(line)
            if m:
                if pending:
                    records.append(pending)
                pending = {
                    "epoch":      int(m.group(1)),
                    "train_loss": float(m.group(2)),
                    "train_top1": float(m.group(3)),
                }
                continue

            m = _VAL_LOSS_RE.search(line)
            if m and pending and "val_loss" not in pending:
                pending["val_loss"] = float(m.group(1))
                continue

            m = _VAL_METRICS_RE.search(line)
            if m and pending and "val_cer" not in pending:
                pending["val_top1"] = float(m.group(1))
                pending["val_top5"] = float(m.group(2))
                pending["val_cer"]  = float(m.group(3))
                records.append(pending)
                pending = {}
                continue

            m = _BEST_RE.search(line)
            if m and records:
                records[-1]["is_best"] = True
                continue

            m = _EARLY_STOP_RE.search(line)
            if m:
                early_stop_epoch = records[-1]["epoch"] if records else None

    if pending:
        records.append(pending)
    return records, early_stop_epoch


def _split_runs(records: list[dict]) -> list[list[dict]]:
    """Split into contiguous training runs whenever epoch resets (resume)."""
    if not records:
        return []
    runs, current = [], [records[0]]
    for rec in records[1:]:
        if rec["epoch"] <= current[-1]["epoch"]:
            runs.append(current)
            current = [rec]
        else:
            current.append(rec)
    runs.append(current)
    return runs


def merge_log_records(log_files: list["str | Path"]) -> tuple[list[dict], int | None]:
    """Concatenate + dedupe per-epoch records across one or more log files
    covering a single checkpoint's training history (e.g. a fresh run plus
    one or more resumes).

    The training script preserves true/global epoch numbers across resumes
    (a resumed job's first logged epoch is checkpoint_epoch + 1, not 1), so
    passing `log_files` in chronological order and letting a later file's
    record for a given epoch overwrite an earlier one's is correct: it's
    always the most recent/complete rerun of that epoch. A single-file
    input degenerates to _parse_suminanet_log's own output.

    Returns an epoch-ascending, duplicate-free record list plus an
    early-stop epoch (if any). Missing files are skipped with a printed
    warning rather than raising. An early-stop signal from an earlier file
    is dropped if a later file's records extend past it (e.g. training was
    resumed from a pre-early-stop checkpoint and continued further) — a
    stale "early stop" marker sitting in the middle of a continuing curve
    would be actively misleading rather than informative.
    """
    merged: dict[int, dict] = {}
    last_early_stop: int | None = None
    for log_file in log_files:
        log_file = Path(log_file)
        if not log_file.exists():
            print(f"merge_log_records: log file not found ({log_file}), skipping.")
            continue
        records, early_stop_epoch = _parse_suminanet_log(log_file)
        for rec in records:
            merged[rec["epoch"]] = rec
        if early_stop_epoch is not None:
            last_early_stop = early_stop_epoch
    max_epoch = max(merged) if merged else None
    if last_early_stop is not None and max_epoch is not None and last_early_stop < max_epoch:
        last_early_stop = None
    return [merged[ep] for ep in sorted(merged)], last_early_stop


def _normalize_log_files(log_files: "str | Path | list[str | Path]") -> list[Path]:
    if isinstance(log_files, (str, Path)):
        return [Path(log_files)]
    return [Path(p) for p in log_files]


def plot_learning_curves(
    log_file: "str | Path",
    out_path: "str | Path" = "learning_curves.png",
) -> "Path | None":
    """
    Parse train_stage2_suminanet.log and plot train/val loss + val CER/top1
    curves, marking the checkpoint that was actually kept (best composite
    score) and the early-stopping point, if any.

    Returns None if the log file is missing or has no matching lines.
    """
    log_file = Path(log_file)
    if not log_file.exists():
        print(f"Learning curve: log file not found ({log_file}), skipping.")
        return None

    records, early_stop_epoch = _parse_suminanet_log(log_file)
    if not records:
        print(f"Learning curve: no epoch summary lines found in {log_file}, skipping.")
        return None

    runs = _split_runs(records)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax2b = ax2.twinx()
    ax2b.set_ylabel("Val top-1 accuracy")
    colors = plt.cm.tab10.colors
    global_epoch = 0
    best_ep = None

    for run_idx, run in enumerate(runs):
        epochs     = [global_epoch + r["epoch"] for r in run]
        train_loss = [r["train_loss"] for r in run]
        val_loss   = [r.get("val_loss") for r in run]
        val_cer    = [r.get("val_cer")  for r in run]
        val_top1   = [r.get("val_top1") for r in run]

        c_t = colors[run_idx % len(colors)]
        c_v = colors[(run_idx + 5) % len(colors)]

        label_sfx = f" (run {run_idx + 1})" if len(runs) > 1 else ""
        ax1.plot(epochs, train_loss, "-",  color=c_t, linewidth=1.5, label=f"Train{label_sfx}")
        ax1.plot(epochs, val_loss,   "--", color=c_v, linewidth=1.5, label=f"Val{label_sfx}")

        ax2.plot(epochs, val_cer, "--", color=c_v, linewidth=1.5, label=f"Val CER{label_sfx}")
        ax2b.plot(epochs, val_top1, "-", color=c_t, linewidth=1.2, alpha=0.7,
                  label=f"Val top1{label_sfx}")

        for r, ep in zip(run, epochs):
            if r.get("is_best"):
                best_ep = ep

        if run_idx > 0:
            for ax in [ax1, ax2]:
                ax.axvline(epochs[0], color="grey", linestyle=":", linewidth=0.8, alpha=0.6)

        global_epoch = epochs[-1]

    if best_ep is not None:
        for ax in [ax1, ax2]:
            ax.axvline(best_ep, color="crimson", linestyle="--", linewidth=1.0,
                       alpha=0.7, label=f"Best checkpoint (epoch {best_ep})")

    if early_stop_epoch is not None:
        for ax in [ax1, ax2]:
            ax.axvline(early_stop_epoch, color="black", linestyle=":", linewidth=1.2,
                       alpha=0.7, label=f"Early stop (epoch {early_stop_epoch})")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Total loss")
    ax1.grid(True, alpha=GRID_ALPHA)
    ax1.legend(fontsize=7, ncol=2)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Val CER")
    ax2.set_title("Val CER vs top-1 accuracy")
    ax2.grid(True, alpha=GRID_ALPHA)
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines2b, labels2b = ax2b.get_legend_handles_labels()
    ax2.legend(lines2 + lines2b, labels2 + labels2b, fontsize=7, ncol=2)

    fig.suptitle("Stage 2 (SuminaNet) training curves", fontsize=11)
    fig.tight_layout()
    return savefig(fig, out_path)


def _latest_best_epoch(records: list[dict]) -> int | None:
    best_ep = None
    for r in records:
        if r.get("is_best"):
            best_ep = r["epoch"]
    return best_ep


def plot_loss_cer_curve(
    log_files: "str | Path | list[str | Path]",
    out_path: "str | Path" = "training_curve_loss_cer.png",
) -> "Path | None":
    """Train/val loss (left axis) + val CER (right axis) vs. true epoch
    number, merged/deduped across `log_files` via merge_log_records.

    Unlike plot_learning_curves, resumes are NOT stitched nose-to-tail on
    the x-axis — a resumed epoch keeps its real epoch number, and the
    latest file's value for it wins. Returns None if no file exists or
    nothing parses.
    """
    records, early_stop_epoch = merge_log_records(_normalize_log_files(log_files))
    if not records:
        print(f"Training curve (loss/CER): no epoch summary lines found in {log_files}, skipping.")
        return None

    epochs     = [r["epoch"] for r in records]
    train_loss = [r.get("train_loss") for r in records]
    val_loss   = [r.get("val_loss") for r in records]
    val_cer    = [r.get("val_cer") for r in records]
    best_ep    = _latest_best_epoch(records)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    ax1.plot(epochs, train_loss, "-", color="#4C72B0", linewidth=1.5, label="Train loss")
    ax1.plot(epochs, val_loss, "--", color="#C44E52", linewidth=1.5, label="Val loss")
    ax2.plot(epochs, val_cer, "-", color="#55A868", linewidth=1.5,
              marker="o", markersize=MARKER_SIZE, label="Val CER")

    if best_ep is not None:
        ax1.axvline(best_ep, color="crimson", linestyle="--", linewidth=1.0,
                    alpha=0.7, label=f"Best checkpoint (epoch {best_ep})")
    if early_stop_epoch is not None:
        ax1.axvline(early_stop_epoch, color="black", linestyle=":", linewidth=1.2,
                    alpha=0.7, label=f"Early stop (epoch {early_stop_epoch})")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax2.set_ylabel("Val CER")
    ax1.set_title("Stage 2 (SuminaNet) training curve: loss + CER")
    ax1.grid(True, alpha=GRID_ALPHA)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    fig.tight_layout()
    return savefig(fig, out_path)


def plot_top1_top5_curve(
    log_files: "str | Path | list[str | Path]",
    out_path: "str | Path" = "training_curve_top1_top5.png",
) -> "Path | None":
    """Val top-1 / top-5 accuracy vs. true epoch number, merged/deduped
    across `log_files` via merge_log_records. See plot_loss_cer_curve for
    the epoch-numbering convention. Returns None if nothing parses.
    """
    records, early_stop_epoch = merge_log_records(_normalize_log_files(log_files))
    if not records:
        print(f"Training curve (top1/top5): no epoch summary lines found in {log_files}, skipping.")
        return None

    epochs   = [r["epoch"] for r in records]
    val_top1 = [r.get("val_top1") for r in records]
    val_top5 = [r.get("val_top5") for r in records]
    best_ep  = _latest_best_epoch(records)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, val_top1, "-", color="#4C72B0", linewidth=1.5,
            marker="o", markersize=MARKER_SIZE, label="Val top-1")
    ax.plot(epochs, val_top5, "-", color="#DD8452", linewidth=1.5,
            marker="o", markersize=MARKER_SIZE, label="Val top-5")

    if best_ep is not None:
        ax.axvline(best_ep, color="crimson", linestyle="--", linewidth=1.0,
                   alpha=0.7, label=f"Best checkpoint (epoch {best_ep})")
    if early_stop_epoch is not None:
        ax.axvline(early_stop_epoch, color="black", linestyle=":", linewidth=1.2,
                   alpha=0.7, label=f"Early stop (epoch {early_stop_epoch})")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.7, 1.02)
    ax.set_title("Stage 2 (SuminaNet) training curve: top-1 / top-5 accuracy")
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return savefig(fig, out_path)
