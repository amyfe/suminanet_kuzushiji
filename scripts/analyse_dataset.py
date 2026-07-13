"""Dataset introduction analysis for thesis (section 7.5.5).

Scans all annotation JSON files, computes character frequencies per split,
and produces 6 publication-ready plots.

Usage:
    python scripts/analyse_dataset.py [--out-dir results/dataset_analysis]
    python scripts/analyse_dataset.py --top-n 50 --rare-thresh 50
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from visualization.dataset import (
    plot_chars_per_image,
    plot_class_imbalance,
    plot_script_distribution,
    plot_split_overview,
    plot_top_characters,
    plot_zipf_curve,
)


def _load_dataset_stats(assets_root: Path):
    """
    Single pass over all annotation JSON files.
    Returns:
      label_counts       : Counter (all splits combined)
      split_label_counts : {'train': Counter, 'val': Counter}
      chars_per_image    : list[int] (all images)
      split_cpi          : {'train': list[int], 'val': list[int]}
      split_image_counts : {'train': int, ...}
    """
    ann_dir    = assets_root / "data" / "annotations"
    splits_dir = assets_root / "data" / "splits"

    # Load split membership (strip .json suffix if present)
    split_ids: dict[str, set[str]] = {}
    for s in ["train", "val"]:
        split_file = splits_dir / f"{s}.txt"
        ids: set[str] = set()
        if split_file.exists():
            for line in split_file.read_text().splitlines():
                line = line.strip()
                if line.endswith(".json"):
                    line = line[:-5]
                if line:
                    ids.add(line)
        split_ids[s] = ids

    all_ann_files = sorted(ann_dir.glob("*.json"))
    total = len(all_ann_files)
    print(f"  Scanning {total} annotation files …")

    label_counts: collections.Counter = collections.Counter()
    split_label_counts = {s: collections.Counter() for s in ["train", "val"]}
    chars_per_image: list[int] = []
    split_cpi: dict[str, list[int]] = {s: [] for s in ["train", "val"]}
    split_image_counts = {s: 0 for s in ["train", "val"]}

    for i, f in enumerate(all_ann_files):
        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{total} …")
        d = json.load(open(f, encoding="utf-8"))
        labels = d.get("labels", [])
        n = len(labels)
        label_counts.update(labels)
        chars_per_image.append(n)

        stem = f.stem
        for s, ids in split_ids.items():
            if stem in ids:
                split_label_counts[s].update(labels)
                split_cpi[s].append(n)
                split_image_counts[s] += 1

    return label_counts, split_label_counts, chars_per_image, split_cpi, split_image_counts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dataset introduction visualizations for thesis")
    p.add_argument("--assets-root", default="assets",
                   help="Root directory containing data/annotations and data/splits")
    p.add_argument("--out-dir", default="results/dataset_analysis")
    p.add_argument("--top-n",       type=int, default=40,
                   help="Number of top characters to show in the bar chart")
    p.add_argument("--rare-thresh", type=int, default=50,
                   help="Characters with fewer occurrences are classified as 'rare'")
    return p.parse_args()


def main() -> None:
    args     = parse_args()
    assets   = Path(args.assets_root)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Dataset analysis]  assets={assets}  out={out_dir}")

    # ------------------------------------------------------------------
    # 1. Scan annotation files
    # ------------------------------------------------------------------
    label_counts, split_lc, chars_per_image, split_cpi, split_imgs = \
        _load_dataset_stats(assets)

    all_chars  = sum(label_counts.values())
    all_unique = len(label_counts)

    print(f"  Total images     : {len(chars_per_image):,}")
    print(f"  Total characters : {all_chars:,}")
    print(f"  Unique classes   : {all_unique:,}")
    for s in ["train", "val"]:
        c = split_lc[s]
        print(f"  {s:5s}: images={split_imgs[s]:,}  chars={sum(c.values()):,}  unique={len(c):,}")

    # ------------------------------------------------------------------
    # 2. Build split stats dict for overview plot
    # ------------------------------------------------------------------
    split_stats = {
        s: {
            "images": split_imgs[s],
            "chars":  sum(split_lc[s].values()),
            "unique": len(split_lc[s]),
        }
        for s in ["train", "val"]
    }

    # ------------------------------------------------------------------
    # 3. Generate plots
    # ------------------------------------------------------------------
    plots = {}

    print("  Plotting dataset split overview …")
    plots["split_overview"] = plot_split_overview(
        split_stats,
        out_path=out_dir / "split_overview.png",
    )

    print("  Plotting Zipf frequency curve …")
    plots["zipf_curve"] = plot_zipf_curve(
        label_counts,
        out_path=out_dir / "zipf_curve.png",
    )

    print("  Plotting script type distribution …")
    plots["script_distribution"] = plot_script_distribution(
        label_counts,
        split_label_counts=split_lc,
        out_path=out_dir / "script_distribution.png",
    )

    print("  Plotting class imbalance histogram …")
    plots["class_imbalance"] = plot_class_imbalance(
        label_counts,
        rare_thresh=args.rare_thresh,
        out_path=out_dir / "class_imbalance.png",
    )

    print(f"  Plotting top-{args.top_n} characters …")
    plots["top_characters"] = plot_top_characters(
        label_counts,
        top_n=args.top_n,
        rare_thresh=args.rare_thresh,
        out_path=out_dir / "top_characters.png",
    )

    print("  Plotting characters-per-image distribution …")
    plots["chars_per_image"] = plot_chars_per_image(
        chars_per_image,
        split_chars_per_image=split_cpi,
        out_path=out_dir / "chars_per_image.png",
    )

    # ------------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------------
    n_hapax = sum(1 for v in label_counts.values() if v == 1)
    n_rare  = sum(1 for v in label_counts.values() if v < args.rare_thresh)
    top10   = label_counts.most_common(10)
    top10_cov = sum(v for _, v in top10) / all_chars * 100

    import numpy as np
    freqs = sorted(label_counts.values(), reverse=True)
    cum   = np.cumsum(freqs)
    k100_cov  = cum[99]  / all_chars * 100 if len(freqs) >= 100  else None
    k500_cov  = cum[499] / all_chars * 100 if len(freqs) >= 500  else None
    k1000_cov = cum[999] / all_chars * 100 if len(freqs) >= 1000 else None

    print("\n" + "=" * 62)
    print("Dataset statistics for thesis")
    print(f"  Total images           : {len(chars_per_image):,}")
    print(f"  Total instances        : {all_chars:,}")
    print(f"  Unique character classes: {all_unique:,}")
    print(f"  Hapax legomena (freq=1): {n_hapax:,}")
    print(f"  Rare classes (< {args.rare_thresh:3d})   : {n_rare:,}")
    print(f"  Top-10 coverage        : {top10_cov:.1f}%")
    if k100_cov:  print(f"  Top-100 coverage       : {k100_cov:.1f}%")
    if k500_cov:  print(f"  Top-500 coverage       : {k500_cov:.1f}%")
    if k1000_cov: print(f"  Top-1000 coverage      : {k1000_cov:.1f}%")
    print(f"  Chars/image: median={int(np.median(chars_per_image))}  "
          f"mean={np.mean(chars_per_image):.0f}  "
          f"max={max(chars_per_image)}")
    print("Files written:")
    for k, p in plots.items():
        if p:
            print(f"  [{k}]  {p}")
    print("=" * 62)


if __name__ == "__main__":
    main()
