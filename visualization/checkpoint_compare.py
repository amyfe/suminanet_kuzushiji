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

from visualization.common import BAR_ALPHA, GRID_ALPHA, savefig

_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]

# (metrics.json key, display label, higher_is_better)
DEFAULT_METRIC_KEYS: list[tuple[str, str, bool]] = [
    ("top1", "Top-1 acc", True),
    ("top5", "Top-5 acc", True),
    ("assembled_cer", "Assembled CER", False),
    ("coverage", "Coverage", True),
    ("det_precision", "Det. precision", True),
    ("det_f1", "Det. F1", True),
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
    3+ for B/C/D together).
    """
    keys = keys or DEFAULT_METRIC_KEYS
    names = list(metrics_by_name.keys())
    n_names = len(names)
    n_metrics = len(keys)

    fig, ax = plt.subplots(figsize=(max(8, 1.8 * n_metrics), 5.5))
    x = np.arange(n_metrics)
    width = 0.8 / n_names

    for i, name in enumerate(names):
        values = [float(metrics_by_name[name].get(k) or 0.0) for k, _, _ in keys]
        offset = (i - (n_names - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=name,
                       color=_COLORS[i % len(_COLORS)], alpha=BAR_ALPHA)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{v:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{label}\n{'(lower better)' if not higher_better else ''}"
         for _, label, higher_better in keys],
        rotation=15, ha="right",
    )
    ax.set_ylabel("Score")
    ax.set_title("Checkpoint comparison: " + " vs. ".join(names))
    ax.legend()
    ax.grid(axis="y", alpha=GRID_ALPHA)
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
    out_path: "str | Path" = "per_class_error_delta.png",
) -> Path:
    """For `other` vs. `baseline`: per-character error-rate delta.

    per_class_errors.csv only lists classes with >=1 error, so a class
    absent from one side's rows is treated as error_rate=0 on that side
    (matching how validate_suminanet.py builds the CSV — a class not in
    error_counter had zero errors).
    Positive delta = `other` is worse than `baseline` for that character.
    """
    def _rate_map(name: str) -> dict[str, float]:
        return {
            row["char"]: float(row["error_rate"])
            for row in metrics_by_name[name]["_per_class_rows"]
        }

    base_rates = _rate_map(baseline)
    other_rates = _rate_map(other)
    all_chars = set(base_rates) | set(other_rates)

    deltas = [
        (ch, other_rates.get(ch, 0.0) - base_rates.get(ch, 0.0))
        for ch in all_chars
    ]
    deltas.sort(key=lambda t: t[1])

    regressed = [d for d in deltas if d[1] > 0][-top_n:]
    improved = [d for d in deltas if d[1] < 0][:top_n]
    combined = improved + regressed
    if not combined:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "No per-class differences found", ha="center", va="center",
                transform=ax.transAxes)
        return savefig(fig, out_path)

    chars = [c for c, _ in combined]
    vals = [v * 100 for _, v in combined]
    colors = ["#55A868" if v < 0 else "#C44E52" for v in vals]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * len(combined))))
    ax.barh(range(len(combined)), vals, color=colors, alpha=BAR_ALPHA)
    ax.set_yticks(range(len(combined)))
    ax.set_yticklabels(chars, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(
        f"Error rate delta (pp): {other} minus {baseline}  "
        f"(green = {other} improved, red = {other} regressed)"
    )
    ax.set_title(f"Per-character error rate: {other} vs. {baseline}  (top {top_n} each direction)")
    ax.grid(axis="x", alpha=GRID_ALPHA)
    fig.tight_layout()
    return savefig(fig, out_path)
