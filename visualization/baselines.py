"""Published external baselines for the thesis comparison section
(pipeline_analysis.txt items 8.1 / 8.2).

IMPORTANT — task mismatch caveat:
Two genuinely different tasks get called "Kuzushiji recognition" in the
literature:
  1. Page-level detection + recognition: find characters on a full page
     AND classify them (your Stage 1 -> Stage 2 pipeline; KuroNet).
  2. Isolated single-character classification: given an already-cropped
     character image, output its label (Kuzushiji-MNIST/49/Kanji; DKNet).
Task 2 is strictly easier — it never has to find characters, only label
ones already handed to it. A 98% F1 on task 2 is not evidence of being
"better than" a 90% F1 on task 1. `is_comparable_to` on each entry exists
so comparison code can enforce this rather than a human having to
remember it every time this table gets copied into a chart.

Sources (verify page/table numbers against the actual PDFs before citing
in the thesis — these were gathered via web search, not a full paper read):
  - Clanuwat, Lamb, Kitamoto. "KuroNet: Pre-Modern Japanese Kuzushiji
    Character Recognition with Deep Learning." ICDAR 2019.
  - Clanuwat, Bober-Irizar, Kitamoto, Lamb, Yamamoto, Ha. "Deep Learning
    for Classical Japanese Literature." NeurIPS 2018 Workshop on Machine
    Learning for Creativity and Design. arXiv:1812.01718. Introduces
    Kuzushiji-MNIST / Kuzushiji-49 / Kuzushiji-Kanji.
  - DKNet: "Deep Kuzushiji Characters Recognition Network."
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

from visualization.common import BAR_ALPHA, GRID_ALPHA, savefig

PIPELINE_F1 = "pipeline_f1"                 # page-level detection+recognition F1
CLASSIFICATION_ACC = "stage2_classification_acc"  # isolated single-char classification


@dataclass(frozen=True)
class Baseline:
    name: str
    paper: str
    task: str
    metric: str
    value: float           # 0-1 fraction
    is_comparable_to: str  # PIPELINE_F1 | CLASSIFICATION_ACC | "none"
    note: str = ""


BASELINES: list[Baseline] = [
    Baseline(
        name="KuroNet (median book)",
        paper="Clanuwat et al., ICDAR 2019",
        task="page-level detection + recognition",
        metric="F1",
        value=0.8824,
        is_comparable_to=PIPELINE_F1,
        note="Reported example-page F1; only 15% of evaluated books scored "
             "F1 in 90-100%, so this is not a uniform book-wide number — "
             "treat as a single reference point, not a tight benchmark.",
    ),
    Baseline(
        name="Kuzushiji-49 (isolated char classification)",
        paper="Clanuwat et al., NeurIPS 2018 workshop, arXiv:1812.01718",
        task="isolated pre-cropped single-character classification",
        metric="test accuracy",
        value=0.9721,
        is_comparable_to=CLASSIFICATION_ACC,
        note="Benchmark dataset paper, not a detector — no page-level F1 "
             "reported. Task is strictly easier than page-level detection.",
    ),
    Baseline(
        name="DKNet",
        paper="Deep Kuzushiji Characters Recognition Network",
        task="isolated pre-cropped single-character classification (Kuzushiji-49)",
        metric="F1",
        value=0.9795,
        is_comparable_to=CLASSIFICATION_ACC,
        note="Also reports ~88.5%/92.5% F1 in a different val/test split; "
             "not a page-level detector — do not compare to Stage 1.",
    ),
]


def compare_to_baselines(
    your_metrics: dict[str, float],
    checkpoint_name: str = "your model",
    out_path: "str | Path" = "baseline_comparison.png",
) -> Path:
    """Bar chart: your metrics vs. published baselines, split into a
    "comparable" panel (same task) and a clearly separate greyed-out
    "reference only" panel (different task) so the two are never implied
    to be head-to-head.

    your_metrics: e.g. {"Stage 1 F1": 0.9088, "Pipeline F1": ..., "Stage 2 top-1": 0.8726}
    Each key is matched against baselines by `is_comparable_to`: keys
    containing "f1" or "pipeline" are compared against PIPELINE_F1 entries;
    everything else against CLASSIFICATION_ACC entries.
    """
    comparable = [b for b in BASELINES if b.is_comparable_to == PIPELINE_F1]
    reference_only = [b for b in BASELINES if b.is_comparable_to != PIPELINE_F1]

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1, 1]}
    )

    # Left panel: genuinely comparable (page-level F1)
    your_pipeline_vals = {k: v for k, v in your_metrics.items()
                           if "f1" in k.lower() or "pipeline" in k.lower()}
    left_names = list(your_pipeline_vals.keys()) + [b.name for b in comparable]
    left_vals = list(your_pipeline_vals.values()) + [b.value for b in comparable]
    left_colors = ["#4C72B0"] * len(your_pipeline_vals) + ["#DD8452"] * len(comparable)
    ax_l.barh(left_names, left_vals, color=left_colors, alpha=BAR_ALPHA)
    for i, v in enumerate(left_vals):
        ax_l.text(v, i, f" {v:.3f}", va="center", fontsize=9)
    ax_l.set_xlim(0, 1.08)
    ax_l.set_title(f"Comparable: page-level detection + recognition F1\n({checkpoint_name} vs. KuroNet)")
    ax_l.grid(axis="x", alpha=GRID_ALPHA)

    # Right panel: reference only (different task — greyed out)
    other_vals = {k: v for k, v in your_metrics.items() if k not in your_pipeline_vals}
    right_names = list(other_vals.keys()) + [b.name for b in reference_only]
    right_vals = list(other_vals.values()) + [b.value for b in reference_only]
    right_colors = (["#4C72B0"] * len(other_vals)
                     + ["#BBBBBB"] * len(reference_only))
    ax_r.barh(right_names, right_vals, color=right_colors, alpha=BAR_ALPHA)
    for i, v in enumerate(right_vals):
        ax_r.text(v, i, f" {v:.3f}", va="center", fontsize=9)
    ax_r.set_xlim(0, 1.08)
    ax_r.set_title(
        "Reference only — DIFFERENT task\n"
        "(isolated single-character classification, not page-level detection)"
    )
    ax_r.grid(axis="x", alpha=GRID_ALPHA)

    fig.suptitle("Baseline comparison — see visualization/baselines.py for task-comparability notes")
    fig.tight_layout()
    return savefig(fig, out_path)
