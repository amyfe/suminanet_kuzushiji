"""Cross-checkpoint comparison utilities for Stage 2 SuminaNet variants
(e.g. checkpoints/B_gru_efficientnet_no_sam2 vs. C_gru_efficientnet_sam2
vs. D_bigru_efficientnet).

Consumes the metrics.json + per_class_errors.csv produced by
utils/validation/validate_suminanet.py for each checkpoint's own
validation run. Generic over N named checkpoints, not hardcoded to any
specific letters.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.patches as mpatches

from visualization.common import BAR_ALPHA, GRID_ALPHA, MARKER_SIZE, cjk_font_prop, pastel_color, savefig
from visualization.stage2 import merge_log_records

_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]

# (metrics.json key, display label, higher_is_better)
DEFAULT_METRIC_KEYS: list[tuple[str, str, bool]] = [
    ("top1", "Top-1 acc", True),
    ("top5", "Top-5 acc", True),
    ("assembled_cer", "Assembled CER", False),
    ("coverage", "Coverage", True),
    ("det_precision", "Det. precision", True),
    ("det_f1", "Det. F1", True),
    ("pipeline_precision", "Pipeline precision", True),
    ("pipeline_f1", "Pipeline F1", True),
    ("ordering_violation_rate", "Ordering violation rate", False),
]

_SCRIPT_TYPES = ["hiragana", "katakana", "kanji", "latin", "digit", "other"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_checkpoint_metrics(dirs: dict[str, "str | Path"]) -> dict[str, dict]:
    """Load metrics.json (+ per_class_errors.csv rows) for each named run dir.

    dirs: {"B": "results/checkpoint_compare/B", ...} — each value is the
    --out-dir passed to `python utils/validation/validate_suminanet.py
    --ckpt <checkpoint> --out-dir <dir>` for that checkpoint variant.

    Raises with the exact file path if a metrics.json is missing or fails
    to parse, so a stale/truncated copy can never silently feed a
    "comparison" again (see the tensor-serialization bug fixed in
    validate_suminanet.py — truncated files are exactly 190 bytes and cut
    off right after "loss_char":).
    """
    out: dict[str, dict] = {}
    for name, d in dirs.items():
        d = Path(d)
        metrics_path = d / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(
                f"[{name}] {metrics_path} does not exist. Run:\n"
                f"  python utils/validation/validate_suminanet.py --ckpt <checkpoint.pt> --out-dir {d}"
            )
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(
                f"[{name}] {metrics_path} is not valid JSON ({e}). If this file "
                "is truncated at ~190 bytes right after \"loss_char\":, it was "
                "written before the tensor-serialization fix in "
                "validate_suminanet.py — rerun the validation to regenerate it."
            ) from e

        csv_path = d / "per_class_errors.csv"
        per_class_rows: list[dict] = []
        if csv_path.exists():
            with open(csv_path, encoding="utf-8", newline="") as f:
                per_class_rows = list(csv.DictReader(f))
        metrics["_per_class_rows"] = per_class_rows
        metrics["_source_dir"] = str(d)
        out[name] = metrics
    return out


# ---------------------------------------------------------------------------
# Aggregate comparison: table + grouped bar chart
# ---------------------------------------------------------------------------

def _script_error_rate(metrics: dict, script: str) -> float | None:
    breakdown = metrics.get("error_breakdown_by_script", {}).get(script)
    if not breakdown or breakdown.get("total_gt", 0) == 0:
        return None
    return breakdown["errors"] / breakdown["total_gt"]


def compare_checkpoints_table(
    metrics_by_name: dict[str, dict],
    keys: list[tuple[str, str, bool]] | None = None,
    out_path: "str | Path" = "checkpoint_comparison.md",
) -> Path:
    """Write a Markdown + CSV table: metric rows x checkpoint columns."""
    keys = keys or DEFAULT_METRIC_KEYS
    names = list(metrics_by_name.keys())
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def fmt(v):
        return f"{v:.4f}" if isinstance(v, (int, float)) else "—"

    rows: list[tuple[str, list]] = []
    for key, label, _ in keys:
        rows.append((label, [metrics_by_name[n].get(key) for n in names]))
    for script in _SCRIPT_TYPES:
        vals = [_script_error_rate(metrics_by_name[n], script) for n in names]
        if any(v is not None for v in vals):
            rows.append((f"Error rate — {script}", vals))

    lines = ["| Metric | " + " | ".join(names) + " |",
             "|---" + "|---" * len(names) + "|"]
    for label, vals in rows:
        lines.append("| " + label + " | " + " | ".join(fmt(v) for v in vals) + " |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    csv_path = out_path.with_suffix(".csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric"] + names)
        for label, vals in rows:
            writer.writerow([label] + [v if v is not None else "" for v in vals])

    return out_path


def plot_checkpoint_comparison_bars(
    metrics_by_name: dict[str, dict],
    keys: list[tuple[str, str, bool]] | None = None,
    out_path: "str | Path" = "checkpoint_comparison.png",
) -> Path:
    """Grouped bar chart: one group per metric, one bar per checkpoint.

    Works for any number of named checkpoints (2 for a pairwise comparison,
    3+ for B/C/D together). Within each metric group, the checkpoint(s) with
    the best value (per that metric's higher_is_better direction) are drawn
    in their saturated base color; the rest are drawn in a pastel tint of
    that same color, so "which one won this metric" reads directly off the
    bar shade without needing to cross-reference the legend.
    """
    keys = keys or DEFAULT_METRIC_KEYS
    names = list(metrics_by_name.keys())
    n_names = len(names)
    n_metrics = len(keys)
    base_colors = [_COLORS[i % len(_COLORS)] for i in range(n_names)]
    pastel_colors = [pastel_color(c) for c in base_colors]

    values_by_name = {
        name: [float(metrics_by_name[name].get(k) or 0.0) for k, _, _ in keys]
        for name in names
    }

    fig, ax = plt.subplots(figsize=(max(8, 1.8 * n_metrics), 5.5))
    x = np.arange(n_metrics)
    width = 0.8 / n_names

    for j, (_, _, higher_better) in enumerate(keys):
        col_values = [values_by_name[name][j] for name in names]
        best = max(col_values) if higher_better else min(col_values)
        for i, name in enumerate(names):
            is_best = col_values[i] == best
            offset = (i - (n_names - 1) / 2) * width
            bar = ax.bar(x[j] + offset, col_values[i], width,
                         color=base_colors[i] if is_best else pastel_colors[i],
                         alpha=BAR_ALPHA)[0]
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{col_values[i]:.3f}", ha="center", va="bottom",
                     fontsize=7, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{label}\n{'(lower better)' if not higher_better else ''}"
         for _, label, higher_better in keys],
        rotation=15, ha="right",
    )
    ax.set_ylabel("Score")
    ax.set_title(
        "Checkpoint comparison: " + " vs. ".join(names)
        + "  (darker = best per metric)"
    )
    legend_handles = [mpatches.Patch(color=base_colors[i], label=name)
                       for i, name in enumerate(names)]
    ax.legend(handles=legend_handles)
    ax.grid(axis="y", alpha=GRID_ALPHA)
    fig.tight_layout()
    return savefig(fig, out_path)


# ---------------------------------------------------------------------------
# Error rate vs. character frequency: do errors concentrate on rare classes?
# ---------------------------------------------------------------------------

# Matches visualization/dataset.py's script->color scheme for the 4 buckets
# shared with dataset_analysis (hiragana/katakana/kanji/other), so the same
# script always reads as the same color across both plot suites; latin/digit
# have no dataset_analysis equivalent (folded into "Other" there) so they
# keep distinct colors of their own.
_SCRIPT_COLORS = {
    "hiragana": "#4C72B0",
    "katakana": "#55A868",
    "kanji":    "#C44E52",
    "other":    "#DD8452",
    "latin":    "#8172B2",
    "digit":    "#937860",
}


def plot_error_rate_vs_frequency(
    metrics_by_name: dict[str, dict],
    out_path: "str | Path" = "error_rate_vs_frequency.png",
) -> Path:
    """Scatter of GT frequency (log-x) vs. error rate, one panel per checkpoint,
    colored by script type — shows whether errors concentrate on rare/tail
    classes or are spread evenly regardless of frequency.

    Only classes with >=1 error appear (per_class_errors.csv only lists those,
    see save_per_class_errors_csv in validate_suminanet.py).
    """
    names = list(metrics_by_name.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4.5), sharey=True)
    if len(names) == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        rows = metrics_by_name[name]["_per_class_rows"]
        for script in _SCRIPT_TYPES:
            xs = [int(r["total_gt"]) for r in rows if r["script_type"] == script]
            ys = [float(r["error_rate"]) for r in rows if r["script_type"] == script]
            if xs:
                ax.scatter(xs, ys, s=14, alpha=BAR_ALPHA, label=script,
                           color=_SCRIPT_COLORS.get(script, "#333333"))
        ax.set_xscale("log")
        ax.set_xlabel("GT frequency (log)")
        ax.set_title(name)
        ax.grid(alpha=GRID_ALPHA)

    axes[0].set_ylabel("Error rate")
    axes[-1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Error rate vs. character frequency (classes with ≥1 error)")
    fig.tight_layout()
    return savefig(fig, out_path)


# ---------------------------------------------------------------------------
# Per-class error delta: which characters actually moved
# ---------------------------------------------------------------------------

def plot_per_class_error_delta(
    metrics_by_name: dict[str, dict],
    baseline: str,
    other: str,
    top_n: int = 20,
    min_support: int = 5,
    out_path: "str | Path" = "per_class_error_delta.png",
) -> Path:
    """For `other` vs. `baseline`: per-character error-rate delta.

    per_class_errors.csv only lists classes with >=1 error, so a class
    absent from one side's rows is treated as error_rate=0 on that side
    (matching how validate_suminanet.py builds the CSV — a class not in
    error_counter had zero errors).

    Characters are filtered by `min_support` (GT occurrence count, i.e.
    total_gt) before ranking. Without this, a rare character (total_gt=1)
    that flips from right to wrong produces a full 100pp swing regardless
    of how little evidence backs it — and a vocab this size has enough such
    singletons to fill the entire top-N-per-direction quota with that
    noise, making every pairwise plot look the same (saturated at +-100pp)
    instead of showing real signal. total_gt is a reliable cross-checkpoint
    proxy for occurrence count here since B/C/D share the same frozen
    Stage-1 detector (coverage/det_precision/det_f1 are identical across
    them), so `support = max(total_gt on either side)` isn't biased by
    which checkpoint happens to be plotted.

    Positive delta = `other` is worse than `baseline` for that character.
    """
    def _rate_and_support(name: str) -> tuple[dict[str, float], dict[str, int]]:
        rates: dict[str, float] = {}
        support: dict[str, int] = {}
        for row in metrics_by_name[name]["_per_class_rows"]:
            rates[row["char"]] = float(row["error_rate"])
            support[row["char"]] = int(row["total_gt"])
        return rates, support

    base_rates, base_support = _rate_and_support(baseline)
    other_rates, other_support = _rate_and_support(other)
    all_chars = set(base_rates) | set(other_rates)

    deltas = []
    for ch in all_chars:
        support = max(base_support.get(ch, 0), other_support.get(ch, 0))
        if support < min_support:
            continue
        delta = other_rates.get(ch, 0.0) - base_rates.get(ch, 0.0)
        deltas.append((ch, delta, support))
    deltas.sort(key=lambda t: t[1])

    regressed = [d for d in deltas if d[1] > 0][-top_n:]
    improved = [d for d in deltas if d[1] < 0][:top_n]
    combined = improved + regressed
    if not combined:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, f"No per-class differences found with >={min_support} GT occurrences",
                ha="center", va="center", transform=ax.transAxes)
        return savefig(fig, out_path)

    chars = [c for c, _, _ in combined]
    vals = [v * 100 for _, v, _ in combined]
    supports = [s for _, _, s in combined]
    colors = ["#55A868" if v < 0 else "#C44E52" for v in vals]

    fig, ax = plt.subplots(figsize=(9, max(4, 0.28 * len(combined))))
    bars = ax.barh(range(len(combined)), vals, color=colors, alpha=BAR_ALPHA)
    ax.set_yticks(range(len(combined)))
    cjk = cjk_font_prop(9)
    if cjk is not None:
        ax.set_yticklabels(chars)
        for txt in ax.get_yticklabels():
            txt.set_fontproperties(cjk)
    else:
        # Fallback: show Unicode codepoints (readable without a CJK font)
        ax.set_yticklabels([f"U+{ord(c):04X}" if c else "?" for c in chars], fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    for bar, v, s in zip(bars, vals, supports):
        ax.text(v + (1 if v >= 0 else -1), bar.get_y() + bar.get_height() / 2, f"n={s}",
                va="center", ha="left" if v >= 0 else "right", fontsize=7, color="#555555")
    ax.set_xlabel(
        f"Error rate delta (pp): {other} minus {baseline}  "
        f"(green = {other} improved, red = {other} regressed; "
        f"characters with >={min_support} GT occurrences only)"
    )
    ax.set_title(
        f"Per-character error rate: {other} vs. {baseline}  "
        f"(top {top_n} each direction, min_support={min_support})"
    )
    ax.grid(axis="x", alpha=GRID_ALPHA)
    fig.tight_layout()
    return savefig(fig, out_path)


# ---------------------------------------------------------------------------
# Training-curve comparison: overlay per-epoch history across checkpoints
# ---------------------------------------------------------------------------

def plot_training_curves_comparison(
    logs_by_name: dict[str, "list[str | Path]"],
    out_dir: "str | Path",
) -> tuple["Path | None", "Path | None"]:
    """Overlay each named checkpoint's training curve (merged/deduped via
    visualization.stage2.merge_log_records, so multi-file resumes are
    handled the same way as the single-checkpoint plots) on two figures:
    train/val loss + val CER, and val top-1/top-5.

    Returns (None, None) if `logs_by_name` is empty.
    """
    if not logs_by_name:
        return None, None
    out_dir = Path(out_dir)
    names = list(logs_by_name.keys())

    fig1, ax1 = plt.subplots(figsize=(max(8, 1.5 * len(names)), 5.5))
    ax1b = ax1.twinx()
    ax1b.set_ylabel("Val CER")
    fig2, ax2 = plt.subplots(figsize=(max(8, 1.5 * len(names)), 5.5))

    any_records = False
    for i, name in enumerate(names):
        records, _ = merge_log_records(logs_by_name[name])
        if not records:
            print(f"plot_training_curves_comparison: no epoch data for {name!r}, skipping it.")
            continue
        any_records = True
        color = _COLORS[i % len(_COLORS)]
        epochs     = [r["epoch"] for r in records]
        train_loss = [r.get("train_loss") for r in records]
        val_loss   = [r.get("val_loss") for r in records]
        val_cer    = [r.get("val_cer") for r in records]
        val_top1   = [r.get("val_top1") for r in records]
        val_top5   = [r.get("val_top5") for r in records]

        ax1.plot(epochs, train_loss, "-", color=color, linewidth=1.5, label=f"{name} train")
        ax1.plot(epochs, val_loss, "--", color=color, linewidth=1.5, label=f"{name} val")
        ax1b.plot(epochs, val_cer, ":", color=color, linewidth=1.2, alpha=0.8,
                  marker="o", markersize=MARKER_SIZE, label=f"{name} CER")

        ax2.plot(epochs, val_top1, "-", color=color, linewidth=1.5,
                 marker="o", markersize=MARKER_SIZE, label=f"{name} top-1")
        ax2.plot(epochs, val_top5, "--", color=color, linewidth=1.5,
                 marker="o", markersize=MARKER_SIZE, label=f"{name} top-5")

    if not any_records:
        plt.close(fig1)
        plt.close(fig2)
        return None, None

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training curve comparison: loss + CER")
    ax1.grid(True, alpha=GRID_ALPHA)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines1b, labels1b = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines1b, labels1 + labels1b, fontsize=7, ncol=2)
    fig1.tight_layout()
    path1 = savefig(fig1, out_dir / "training_curves_loss_cer.png")

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0.7, 1.02)
    ax2.set_title("Training curve comparison: top-1 / top-5 accuracy")
    ax2.grid(True, alpha=GRID_ALPHA)
    ax2.legend(fontsize=7, ncol=2)
    fig2.tight_layout()
    path2 = savefig(fig2, out_dir / "training_curves_top1_top5.png")

    return path1, path2
