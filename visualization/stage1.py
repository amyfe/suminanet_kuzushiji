"""Stage 1 (detector) visualization helpers for thesis result analysis."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from visualization.common import BAR_ALPHA, GRID_ALPHA, MARKER_SIZE, savefig


# ---------------------------------------------------------------------------
# Plot 1 — PR + F2 curve
# ---------------------------------------------------------------------------

def _compute_f2(precision: float, recall: float) -> float:
    beta2 = 4.0  # β=2 → β²=4
    denom = beta2 * precision + recall
    return (1 + beta2) * precision * recall / denom if denom > 0 else 0.0


def plot_pr_f2_curve(
    pr_results: list,
    current_thresh: float | None = None,
    out_path: "str | Path" = "pr_f2_curve.png",
) -> Path:
    """
    2-panel figure:
      Left:  Precision-Recall space with points colored by threshold.
      Right: P / R / F1 / F2(β=2) vs threshold; axvline at current_thresh.
    """
    thresholds  = [r[0] for r in pr_results]
    precisions  = [r[1] for r in pr_results]
    recalls     = [r[2] for r in pr_results]
    f1s         = [r[3] for r in pr_results]
    f2s         = [_compute_f2(p, r) for p, r in zip(precisions, recalls)]

    best_f1_idx = int(np.argmax(f1s))
    best_f2_idx = int(np.argmax(f2s))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: P-R space ---
    scatter = ax1.scatter(recalls, precisions, c=thresholds, cmap="viridis",
                          s=40, zorder=3)
    ax1.plot(recalls, precisions, "k-", alpha=0.3, linewidth=1)
    ax1.scatter([recalls[best_f1_idx]], [precisions[best_f1_idx]],
                marker="*", s=200, color="gold", zorder=4, label=f"Best F1 ({thresholds[best_f1_idx]:.2f})")
    ax1.scatter([recalls[best_f2_idx]], [precisions[best_f2_idx]],
                marker="D", s=100, color="crimson", zorder=4, label=f"Best F2 ({thresholds[best_f2_idx]:.2f})")
    # annotate every 3rd threshold
    for i, (r, p, t) in enumerate(zip(recalls, precisions, thresholds)):
        if i % 3 == 0:
            ax1.annotate(f"{t:.2f}", (r, p), textcoords="offset points",
                         xytext=(4, 4), fontsize=7)
    fig.colorbar(scatter, ax=ax1, label="Score threshold")
    ax1.set_xlabel("Recall")
    ax1.set_ylabel("Precision")
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=GRID_ALPHA)
    ax1.legend(fontsize=8)
    ax1.set_title("Precision-Recall curve")

    # --- Right: metrics vs threshold ---
    ax2.plot(thresholds, recalls,    "b-o",  markersize=MARKER_SIZE, label="Recall")
    ax2.plot(thresholds, precisions, "g-o",  markersize=MARKER_SIZE, label="Precision")
    ax2.plot(thresholds, f1s,        "r-o",  markersize=MARKER_SIZE, label="F1")
    ax2.plot(thresholds, f2s,        "m--s", markersize=MARKER_SIZE, label="F2 (β=2)")
    if current_thresh is not None:
        ax2.axvline(current_thresh, color="grey", linestyle="--", linewidth=1.2,
                    label=f"Config thresh ({current_thresh:.2f})")
    ax2.axvline(thresholds[best_f2_idx], color="crimson", linestyle=":",
                linewidth=1.2, label=f"Best F2 thresh ({thresholds[best_f2_idx]:.2f})")
    ax2.set_xlabel("Score threshold")
    ax2.set_ylabel("Score")
    ax2.set_xlim(min(thresholds) - 0.02, max(thresholds) + 0.02)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=GRID_ALPHA)
    ax2.legend(fontsize=8)
    ax2.set_title("Metrics vs. score threshold (F2 emphasises recall)")

    fig.suptitle(
        f"Stage 1 detector — PR/F2 analysis  "
        f"(best F1={f1s[best_f1_idx]:.3f} @ {thresholds[best_f1_idx]:.2f}, "
        f"best F2={f2s[best_f2_idx]:.3f} @ {thresholds[best_f2_idx]:.2f})",
        fontsize=10,
    )
    fig.tight_layout()
    return savefig(fig, out_path)


# ---------------------------------------------------------------------------
# Plot 2 — Threshold sweep (P / R / F1 / F2 in one panel)
# ---------------------------------------------------------------------------

def plot_threshold_sweep(
    pr_results: list,
    current_thresh: float | None = None,
    out_path: "str | Path" = "threshold_sweep.png",
) -> Path:
    """
    Single panel: P, R, F1, F2 as 4 lines vs confidence threshold.
    Uses pr_results already collected by validate_stage1() — no model rerun.
    """
    thresholds = [r[0] for r in pr_results]
    precisions = [r[1] for r in pr_results]
    recalls    = [r[2] for r in pr_results]
    f1s        = [r[3] for r in pr_results]
    f2s        = [_compute_f2(p, r) for p, r in zip(precisions, recalls)]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, recalls,    "b-o",  markersize=MARKER_SIZE, label="Recall")
    ax.plot(thresholds, precisions, "g-o",  markersize=MARKER_SIZE, label="Precision")
    ax.plot(thresholds, f1s,        "r-o",  markersize=MARKER_SIZE, label="F1")
    ax.plot(thresholds, f2s,        "m--s", markersize=MARKER_SIZE, label="F2 (β=2, recall-weighted)")

    if current_thresh is not None:
        ax.axvline(current_thresh, color="grey", linestyle="--", linewidth=1.5,
                   label=f"Config threshold ({current_thresh:.2f})")
        # Annotate current values
        def _interp(vals):
            return float(np.interp(current_thresh, thresholds, vals))
        yc = max(_interp(f2s), _interp(f1s), _interp(recalls), _interp(precisions))
        ax.text(current_thresh + 0.01, yc + 0.02,
                f"P={_interp(precisions):.3f}\nR={_interp(recalls):.3f}\n"
                f"F1={_interp(f1s):.3f}\nF2={_interp(f2s):.3f}",
                fontsize=8, color="grey")

    ax.set_xlabel("Score threshold")
    ax.set_ylabel("Score")
    ax.set_xlim(min(thresholds) - 0.02, max(thresholds) + 0.02)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend(fontsize=9)
    ax.set_title("Stage 1 detector: metrics vs. score threshold")
    fig.tight_layout()
    return savefig(fig, out_path)


# ---------------------------------------------------------------------------
# Plot 3 — Column density vs FN rate
# ---------------------------------------------------------------------------

def _group_columns(gt_boxes: list, overlap_ratio: float = 0.20) -> list[int]:
    """
    Assign each GT box a column-size (density) value.
    gt_boxes: list of [x1, y1, x2, y2] in pixel coords.
    """
    if not gt_boxes:
        return []
    boxes = np.array(gt_boxes, dtype=np.float32)  # (N, 4)
    x1, x2 = boxes[:, 0], boxes[:, 2]
    widths  = (x2 - x1).clip(min=1.0)
    n = len(boxes)

    # Union-find for column grouping
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pi] = pj

    for i in range(n):
        for j in range(i + 1, n):
            # x-overlap check
            overlap = min(x2[i], x2[j]) - max(x1[i], x1[j])
            min_w   = min(widths[i], widths[j])
            if overlap / min_w >= overlap_ratio:
                union(i, j)

    # Count column sizes
    from collections import Counter
    roots = [find(i) for i in range(n)]
    col_sizes = Counter(roots)
    return [col_sizes[find(i)] for i in range(n)]


def _iou_matrix_np(gt_boxes: list, pred_boxes: list) -> np.ndarray:
    """IoU matrix: (len(gt), len(pred))."""
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
    union = ag[:, None] + ap[None, :] - inter
    return inter / (union + 1e-6)


def plot_column_density_fn(
    all_gt_boxes: list,
    all_pred_boxes: list,
    iou_threshold: float = 0.5,
    out_path: "str | Path" = "column_density_fn.png",
) -> Path:
    """
    2-panel bar chart: FN rate and GT character count per column-density bucket.
    """
    BUCKETS = [(1, 1, "1\n(isolated)"), (2, 4, "2–4\n(sparse)"),
               (5, 10, "5–10\n(medium)"), (11, 9999, "11+\n(dense)")]

    bucket_fn    = [0] * len(BUCKETS)
    bucket_total = [0] * len(BUCKETS)

    for gt_boxes, pred_boxes in zip(all_gt_boxes, all_pred_boxes):
        if not gt_boxes:
            continue
        densities = _group_columns(gt_boxes)
        iou_mat   = _iou_matrix_np(gt_boxes, pred_boxes)
        # A GT box is FN if its max IoU with any pred < iou_threshold
        is_fn = (iou_mat.max(axis=1) < iou_threshold) if iou_mat.size > 0 \
                else np.ones(len(gt_boxes), dtype=bool)

        for i, (density, fn) in enumerate(zip(densities, is_fn)):
            for b_idx, (lo, hi, _) in enumerate(BUCKETS):
                if lo <= density <= hi:
                    bucket_total[b_idx] += 1
                    if fn:
                        bucket_fn[b_idx] += 1
                    break

    labels    = [b[2] for b in BUCKETS]
    fn_rates  = [bucket_fn[i] / max(1, bucket_total[i]) * 100 for i in range(len(BUCKETS))]
    colors    = ["#4C72B0", "#55A868", "#C44E52", "#DD8452"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    bars1 = ax1.bar(labels, fn_rates, color=colors, alpha=BAR_ALPHA)
    for bar, rate in zip(bars1, fn_rates):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{rate:.1f}%", ha="center", va="bottom", fontsize=9)
    ax1.set_ylabel("False-negative rate (%)")
    ax1.set_xlabel("Column density (characters per column)")
    ax1.set_title("FN rate by column density")
    ax1.grid(axis="y", alpha=GRID_ALPHA)
    ax1.set_ylim(0, max(fn_rates) * 1.25 + 1)

    bars2 = ax2.bar(labels, bucket_total, color=colors, alpha=BAR_ALPHA)
    for bar, tot in zip(bars2, bucket_total):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 f"{tot:,}", ha="center", va="bottom", fontsize=9)
    ax2.set_ylabel("GT character count")
    ax2.set_xlabel("Column density")
    ax2.set_title("GT characters per density bucket")
    ax2.grid(axis="y", alpha=GRID_ALPHA)

    fig.suptitle(
        f"Column-density analysis  (IoU threshold={iou_threshold:.2f}  "
        f"total GT={sum(bucket_total):,}  total FN={sum(bucket_fn):,})",
        fontsize=10,
    )
    fig.tight_layout()
    return savefig(fig, out_path)


# ---------------------------------------------------------------------------
# Plot 4 — SAM2 / gap-fill contribution
# ---------------------------------------------------------------------------

def plot_sam2_contribution(
    metrics_no_gapfill: dict,
    metrics_gapfill: dict,
    sam2_metrics: "dict | None",
    out_path: "str | Path" = "sam2_contribution.png",
) -> Path:
    """
    Grouped bar chart comparing recall and precision across 2 or 3 conditions.
    """
    conditions = [
        ("Stage 1\n(no gap-fill)",         metrics_no_gapfill),
        ("Stage 1\n+ gap-fill",             metrics_gapfill),
    ]
    if sam2_metrics is not None:
        conditions.append(("Stage 1\n+ gap-fill\n+ SAM2", sam2_metrics))

    n = len(conditions)
    labels    = [c[0] for c in conditions]
    recalls   = [c[1]["recall"]    for c in conditions]
    precisions= [c[1]["precision"] for c in conditions]
    f1s       = [c[1]["f1"]        for c in conditions]

    x = np.arange(n)
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - width, recalls,    width, label="Recall",    color="#4C72B0", alpha=BAR_ALPHA)
    b2 = ax.bar(x,         precisions, width, label="Precision", color="#55A868", alpha=BAR_ALPHA)
    b3 = ax.bar(x + width, f1s,        width, label="F1",        color="#C44E52", alpha=BAR_ALPHA)

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.set_title("Coverage contribution: Stage 1 → gap-fill → SAM2 proposals")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=GRID_ALPHA)
    fig.tight_layout()
    return savefig(fig, out_path)


# ---------------------------------------------------------------------------
# Plot 5 — Learning curves
# ---------------------------------------------------------------------------

_EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)/\d+\s+Train:\s+([\d.]+)\s+\(heat=([\d.]+),\s*bbox=([\d.]+)\)"
    r"\s+Val:\s+([\d.]+)\s+\(heat=([\d.]+),\s*bbox=([\d.]+)\)"
)


def _parse_learning_log(log_file: "str | Path") -> list[dict]:
    """Parse all Epoch summary lines; group consecutive runs."""
    records = []
    with open(log_file, encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = _EPOCH_RE.search(line)
            if m:
                records.append({
                    "epoch":      int(m.group(1)),
                    "train_loss": float(m.group(2)),
                    "train_heat": float(m.group(3)),
                    "train_bbox": float(m.group(4)),
                    "val_loss":   float(m.group(5)),
                    "val_heat":   float(m.group(6)),
                    "val_bbox":   float(m.group(7)),
                })
    return records


def _split_runs(records: list[dict]) -> list[list[dict]]:
    """Split into contiguous training runs whenever epoch resets to 1."""
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


def plot_learning_curves(
    log_file: "str | Path",
    out_path: "str | Path" = "learning_curves.png",
) -> "Path | None":
    """
    Parse train_stage1.log and plot train/val loss curves.
    Returns None if log file is missing or has no matching lines.
    """
    log_file = Path(log_file)
    if not log_file.exists():
        print(f"Learning curve: log file not found ({log_file}), skipping.")
        return None

    records = _parse_learning_log(log_file)
    if not records:
        print(f"Learning curve: no epoch summary lines found in {log_file}, skipping.")
        return None

    runs = _split_runs(records)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.tab10.colors
    global_epoch = 0

    for run_idx, run in enumerate(runs):
        epochs      = [global_epoch + r["epoch"] for r in run]
        train_loss  = [r["train_loss"] for r in run]
        val_loss    = [r["val_loss"]   for r in run]
        train_heat  = [r["train_heat"] for r in run]
        train_bbox  = [r["train_bbox"] for r in run]
        val_heat    = [r["val_heat"]   for r in run]
        val_bbox    = [r["val_bbox"]   for r in run]

        c_t = colors[run_idx % len(colors)]
        c_v = colors[(run_idx + 5) % len(colors)]

        label_sfx = f" (run {run_idx + 1})" if len(runs) > 1 else ""
        ax1.plot(epochs, train_loss, "-",  color=c_t, linewidth=1.5, label=f"Train{label_sfx}")
        ax1.plot(epochs, val_loss,   "--", color=c_v, linewidth=1.5, label=f"Val{label_sfx}")

        ax2.plot(epochs, train_heat, "-",  color=c_t, linewidth=1.2, alpha=0.8, label=f"Train heat{label_sfx}")
        ax2.plot(epochs, train_bbox, ":",  color=c_t, linewidth=1.2, alpha=0.8, label=f"Train bbox{label_sfx}")
        ax2.plot(epochs, val_heat,   "--", color=c_v, linewidth=1.2, alpha=0.8, label=f"Val heat{label_sfx}")
        ax2.plot(epochs, val_bbox,   "-.", color=c_v, linewidth=1.2, alpha=0.8, label=f"Val bbox{label_sfx}")

        # Mark resume points with a vertical line
        if run_idx > 0:
            for ax in [ax1, ax2]:
                ax.axvline(epochs[0], color="grey", linestyle=":", linewidth=0.8, alpha=0.6)

        global_epoch = epochs[-1]

    # Mark best val loss epoch globally
    all_val = [(global_ep_offset + r["epoch"], r["val_loss"])
               for run_idx, run in enumerate(runs)
               for r in run
               for global_ep_offset in [sum(len(runs[k]) for k in range(run_idx))]]
    if all_val:
        best_ep, best_val = min(all_val, key=lambda x: x[1])
        for ax in [ax1, ax2]:
            ax.axvline(best_ep, color="crimson", linestyle="--", linewidth=1.0,
                       alpha=0.7, label=f"Best val epoch ({best_ep})")

    for ax, title in [(ax1, "Total loss"), (ax2, "Heat + bbox sub-losses")]:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.grid(True, alpha=GRID_ALPHA)
        ax.legend(fontsize=7, ncol=2)

    fig.suptitle("Stage 1 training curves", fontsize=11)
    fig.tight_layout()
    return savefig(fig, out_path)
