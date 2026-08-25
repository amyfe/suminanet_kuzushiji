"""
sweep_reading_order_divergence.py
-----------------------------------
Compares ROIReadingOrder.sort_single()'s geometrically re-sorted character
order against each annotation file's own raw (dedup'd) label order, for
every page in assets/data/annotations/. Annotators record characters in
reading order already; this quantifies how often the geometric re-sort
diverges from that already-correct order instead of confirming it.

This reproduces the corpus sweep that motivated removing the sort_single()
call from KuzushijiDataset.__getitem__ (see THESIS_SUPERVISOR_NOTES.tex and
utils/__init__.py's __getitem__ comment) and adding the clustering-based
orientation-misclassification guard in
model/suminanet/roi/roi_ordering.py's infer_reading_orientation_from_boxes.
Baseline (before either fix): 43.8% of all 5344 annotation files diverge
(35.4% of the train split specifically); 53.4% of diverging pages are
"severe" (>=50% of positions differ), 29.1% "minor" (<10%).

Since KuzushijiDataset no longer calls sort_single() at all, this script's
purpose going forward is (a) a historical/reproducible record of that
baseline for the thesis, and (b) a regression check for any future change to
ROIReadingOrder's column/row clustering or orientation detection -- rerun
after such a change and compare the divergence-rate breakdown, especially
the orientation-attributable subset (pages where sort_single's chosen
orientation differs from what it would have been before the guard).

Usage:
    python scripts/onetime_scripts/sweep_reading_order_divergence.py
    python scripts/onetime_scripts/sweep_reading_order_divergence.py --split train
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import DATA_DIR
from model.suminanet.roi.roi_ordering import ROIReadingOrder, infer_reading_orientation_from_boxes


def dedupe(boxes: list, labels: list) -> tuple[list, list]:
    seen: set[tuple] = set()
    boxes_out, labels_out = [], []
    for box, lbl in zip(boxes, labels):
        key = (tuple(box), lbl)
        if key in seen:
            continue
        seen.add(key)
        boxes_out.append(box)
        labels_out.append(lbl)
    return boxes_out, labels_out


def divergence_fraction(raw_labels: list, sorted_labels: list) -> float:
    n = len(raw_labels)
    if n == 0:
        return 0.0
    n_diff = sum(1 for a, b in zip(raw_labels, sorted_labels) if a != b)
    return n_diff / n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", type=str, default=None, choices=[None, "train", "val"],
                    help="Restrict to a splits/<split>.txt file's stems. Default: all annotation files.")
    args = p.parse_args()

    ann_dir = DATA_DIR / "annotations"
    files = sorted(ann_dir.glob("*.json"))

    if args.split is not None:
        split_file = DATA_DIR / "splits" / f"{args.split}.txt"
        valid_ids = {
            (ln[:-5] if ln.endswith(".json") else ln)
            for ln in (l.strip() for l in split_file.read_text().splitlines())
            if ln
        }
        files = [f for f in files if f.stem in valid_ids]

    roi_order = ROIReadingOrder()

    n_total = 0
    n_diverged = 0
    n_severe = 0    # >=50% positions differ
    n_moderate = 0  # 10-50%
    n_minor = 0     # <10%, >0%

    for f in files:
        ann = json.loads(f.read_text(encoding="utf-8"))
        raw_boxes, raw_labels = dedupe(ann.get("boxes", []), ann.get("labels", []))
        if len(raw_boxes) == 0:
            continue
        n_total += 1

        orientation = infer_reading_orientation_from_boxes(raw_boxes)
        boxes_t = torch.tensor(raw_boxes, dtype=torch.float32)
        mask = torch.ones((boxes_t.size(0),), dtype=torch.bool)
        _, _, sort_idx, _col_ids = roi_order.sort_single(boxes_t, mask, orientation)
        sorted_labels = [raw_labels[i] for i in sort_idx.detach().cpu().tolist()]

        frac = divergence_fraction(raw_labels, sorted_labels)
        if frac > 0:
            n_diverged += 1
            if frac >= 0.5:
                n_severe += 1
            elif frac >= 0.1:
                n_moderate += 1
            else:
                n_minor += 1

    print(f"Checked {n_total} annotation files"
          + (f" (split={args.split})" if args.split else " (all)"))
    print(f"Diverged from raw annotation order: {n_diverged} ({100*n_diverged/max(1,n_total):.1f}%)")
    if n_diverged:
        print(f"  severe   (>=50% positions differ): {n_severe} ({100*n_severe/n_diverged:.1f}% of diverged)")
        print(f"  moderate (10-50%):                 {n_moderate} ({100*n_moderate/n_diverged:.1f}% of diverged)")
        print(f"  minor    (<10%):                   {n_minor} ({100*n_minor/n_diverged:.1f}% of diverged)")


if __name__ == "__main__":
    main()
