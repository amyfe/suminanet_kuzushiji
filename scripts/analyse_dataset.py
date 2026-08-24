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

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from visualization.dataset import (
    PAPER_BG_COLOR,
    _script,
    plot_augmentation_comparison,
    plot_char_variant_gallery,
    plot_chars_per_image,
    plot_class_imbalance,
    plot_copy_paste_augmentation,
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


def _find_rare_char_example(
    assets_root: Path,
    rare_chars: set[str],
    pad: int = 350,
):
    """
    Scan annotations for the first image containing a rare character.
    Returns (image_path, crop_box, boxes, labels) where crop_box is padded
    around that character so a few neighbouring glyphs stay visible for
    context; boxes/labels are the full GT for that page (original scale).
    Returns (None, None, None, None) if no rare character is found.
    """
    ann_dir = assets_root / "data" / "annotations"
    for f in sorted(ann_dir.glob("*.json")):
        d = json.load(open(f, encoding="utf-8"))
        labels = d.get("labels", [])
        boxes = d.get("boxes", d.get("bboxes", []))
        img_rel = d.get("image_path")
        if not labels or not boxes or not img_rel:
            continue
        for lbl, box in zip(labels, boxes):
            if lbl not in rare_chars:
                continue
            img_path = assets_root / "data" / img_rel
            if not img_path.exists():
                continue
            with Image.open(img_path) as im:
                w, h = im.size
            x1, y1, x2, y2 = box
            crop_box = (
                max(0, int(x1) - pad),
                max(0, int(y1) - pad),
                min(w, int(x2) + pad),
                min(h, int(y2) + pad),
            )
            return img_path, crop_box, boxes, labels
    return None, None, None, None


def _build_rare_kanji_crop_db(
    assets_root: Path,
    rare_kanji_chars: set[str],
    min_crops: int = 10,
    max_files_scanned: int = 2000,
) -> dict:
    """
    Build a small crop database of rare-kanji glyphs for the copy-paste demo.
    Scans annotations (only loading an image when the annotation actually
    contains a rare kanji) until at least `min_crops` crops are collected or
    `max_files_scanned` candidate files have been read, whichever comes first.
    """
    import cv2

    ann_dir = assets_root / "data" / "annotations"
    db: dict[str, list] = collections.defaultdict(list)
    scanned = 0

    for f in sorted(ann_dir.glob("*.json")):
        if scanned >= max_files_scanned or sum(len(v) for v in db.values()) >= min_crops:
            break
        d = json.load(open(f, encoding="utf-8"))
        labels = d.get("labels", [])
        boxes = d.get("boxes", d.get("bboxes", []))
        img_rel = d.get("image_path")
        if not labels or not boxes or not img_rel:
            continue
        if not any(lbl in rare_kanji_chars for lbl in labels):
            continue
        scanned += 1

        img_path = assets_root / "data" / img_rel
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        for lbl, box in zip(labels, boxes):
            if lbl not in rare_kanji_chars:
                continue
            x1, y1, x2, y2 = (int(round(v)) for v in box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            db[lbl].append(img[y1:y2, x1:x2].copy())

    return dict(db)


def _build_char_variant_crop_db(
    assets_root: Path,
    target_labels: set[str],
    max_per_char: int = 20,
    max_files_scanned: int = 3000,
    margin_frac: float = 0.25,
) -> dict:
    """
    Build a database of glyph-crop variants for a fixed set of target
    character labels (e.g. the most common hiragana) — same annotation-scan
    pattern as _build_rare_kanji_crop_db, but for characters common enough
    that a partial scan of the corpus already yields plenty of examples.
    Stops once every target label has max_per_char crops, or
    max_files_scanned annotation files have been read, whichever is first.
    Crops are stored as RGB uint8 arrays, in the source page's original
    color (no grayscale/inversion) — see plot_char_variant_gallery.

    Each crop gets a blank margin around the tight ground-truth box, filled
    with PAPER_BG_COLOR (matching the gallery canvas — see
    plot_char_variant_gallery) rather than the real neighboring page
    content, sized as margin_frac * max(box_w, box_h) on every side — so
    very thin/elongated boxes (e.g. cursive し, which is often annotated as
    a single tall stroke) get visible breathing room without pulling in
    adjacent characters' strokes.
    """
    ann_dir = assets_root / "data" / "annotations"
    db: dict[str, list] = collections.defaultdict(list)
    scanned = 0

    for f in sorted(ann_dir.glob("*.json")):
        if scanned >= max_files_scanned:
            break
        if all(len(db[lbl]) >= max_per_char for lbl in target_labels):
            break
        d = json.load(open(f, encoding="utf-8"))
        labels = d.get("labels", [])
        boxes = d.get("boxes", d.get("bboxes", []))
        img_rel = d.get("image_path")
        if not labels or not boxes or not img_rel:
            continue
        if not any(lbl in target_labels and len(db[lbl]) < max_per_char for lbl in labels):
            continue
        scanned += 1

        img_path = assets_root / "data" / img_rel
        if not img_path.exists():
            continue
        with Image.open(img_path) as im:
            img = im.convert("RGB")
            w, h = img.size
            for lbl, box in zip(labels, boxes):
                if lbl not in target_labels or len(db[lbl]) >= max_per_char:
                    continue
                x1, y1, x2, y2 = (int(round(v)) for v in box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                tight = img.crop((x1, y1, x2, y2))
                bw, bh = x2 - x1, y2 - y1
                margin = max(3, round(margin_frac * max(bw, bh)))
                padded = Image.new("RGB", (bw + 2 * margin, bh + 2 * margin), color=PAPER_BG_COLOR)
                padded.paste(tight, (margin, margin))
                db[lbl].append(np.array(padded))

    return dict(db)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dataset introduction visualizations for thesis")
    p.add_argument("--assets-root", default="assets",
                   help="Root directory containing data/annotations and data/splits")
    p.add_argument("--out-dir", default="results/dataset_analysis")
    p.add_argument("--top-n",       type=int, default=40,
                   help="Number of top characters to show in the bar chart")
    p.add_argument("--rare-thresh", type=int, default=50,
                   help="Characters with fewer occurrences are classified as 'rare'")
    p.add_argument("--aug-image", default=None,
                   help="Image to use for the augmentation comparison plot. "
                        "Defaults to the first image found containing a rare character.")
    p.add_argument("--aug-crop", default=None,
                   help="Crop box 'x1,y1,x2,y2' applied before augmenting (default: auto, "
                        "centered on a rare character). Use 'none' to disable cropping.")
    p.add_argument("--skip-aug-demo", action="store_true",
                   help="Skip the classic-vs-rare augmentation comparison plot")
    p.add_argument("--copy-paste-n", type=int, default=3,
                   help="Number of rare-kanji crops to paste in the copy-paste demo plot")
    p.add_argument("--skip-copy-paste-demo", action="store_true",
                   help="Skip the rare-kanji copy-paste augmentation demo plot")
    p.add_argument("--n-variant-chars", type=int, default=20,
                   help="Number of characters shown in the handwriting-variant gallery, "
                        "picked by descending frequency among hiragana (ignored if "
                        "--variant-chars is given)")
    p.add_argument("--variant-chars", default=None,
                   help="Comma-separated characters to show in the handwriting-variant "
                        "gallery instead of the top-N-by-frequency hiragana, e.g. "
                        "'あ,い,う,え,お'")
    p.add_argument("--variant-crops-per-char", type=int, default=20,
                   help="Max example crops collected per character for the "
                        "handwriting-variant gallery")
    p.add_argument("--skip-variant-gallery", action="store_true",
                   help="Skip the handwriting-variant gallery plot")
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

    kanji_counts = collections.Counter(
        {ch: c for ch, c in label_counts.items() if _script(ch) == "Kanji"}
    )
    if kanji_counts:
        print(f"  Plotting top-{args.top_n} Kanji characters …")
        plots["top_characters_kanji"] = plot_top_characters(
            kanji_counts,
            top_n=args.top_n,
            rare_thresh=args.rare_thresh,
            out_path=out_dir / "top_characters_kanji.png",
            title=f"Top-{args.top_n} most frequent Kanji characters in the dataset",
        )

    print("  Plotting characters-per-image distribution …")
    plots["chars_per_image"] = plot_chars_per_image(
        chars_per_image,
        split_chars_per_image=split_cpi,
        out_path=out_dir / "chars_per_image.png",
    )

    rare_chars = {ch for ch, cnt in label_counts.items() if cnt < args.rare_thresh}
    example_img_path = example_boxes = example_labels = example_crop_box = None
    if not args.aug_image or not args.skip_copy_paste_demo:
        print("  Locating an example image with a rare character …")
        example_img_path, example_crop_box, example_boxes, example_labels = \
            _find_rare_char_example(assets, rare_chars)

    if not args.skip_aug_demo:
        if args.aug_image:
            img_path = Path(args.aug_image)
            crop_box = None
        else:
            img_path, crop_box = example_img_path, example_crop_box

        if args.aug_crop == "none":
            crop_box = None
        elif args.aug_crop:
            crop_box = tuple(int(v) for v in args.aug_crop.split(","))

        if img_path is None:
            print("  Skipping augmentation comparison plot (no rare-character image found).")
        else:
            print(f"  Plotting classic-vs-rare augmentation comparison ({img_path.name}) …")
            plots["augmentation_comparison"] = plot_augmentation_comparison(
                img_path,
                crop_box=crop_box,
                out_path=out_dir / "augmentation_comparison.png",
            )

    if not args.skip_copy_paste_demo:
        rare_kanji = {ch for ch in rare_chars if _script(ch) == "Kanji"}
        if example_img_path is None or not rare_kanji:
            print("  Skipping copy-paste demo (no rare-kanji example image found).")
        else:
            print("  Building a small rare-kanji crop database …")
            crop_db = _build_rare_kanji_crop_db(assets, rare_kanji)
            if not crop_db:
                print("  Skipping copy-paste demo (could not collect any rare-kanji crops).")
            else:
                print(f"  Plotting rare-kanji copy-paste demo ({example_img_path.name}) …")
                plots["copy_paste_augmentation"] = plot_copy_paste_augmentation(
                    example_img_path,
                    boxes=example_boxes,
                    labels=example_labels,
                    rare_chars=rare_kanji,
                    crop_db=crop_db,
                    n_paste=args.copy_paste_n,
                    out_path=out_dir / "copy_paste_augmentation.png",
                )

    if not args.skip_variant_gallery:
        if args.variant_chars:
            variant_labels = {f"U+{ord(c):04X}" for c in args.variant_chars.split(",") if c.strip()}
        else:
            hiragana_by_freq = [ch for ch, _ in label_counts.most_common() if _script(ch) == "Hiragana"]
            variant_labels = set(hiragana_by_freq[:args.n_variant_chars])

        if not variant_labels:
            print("  Skipping handwriting-variant gallery (no target characters resolved).")
        else:
            print(f"  Collecting handwriting variants for {len(variant_labels)} characters …")
            variant_crop_db = _build_char_variant_crop_db(
                assets, variant_labels, max_per_char=args.variant_crops_per_char,
            )
            if not variant_crop_db:
                print("  Skipping handwriting-variant gallery (could not collect any crops).")
            else:
                print(f"  Plotting handwriting-variant gallery ({len(variant_crop_db)} characters) …")
                plots["char_variant_gallery"] = plot_char_variant_gallery(
                    variant_crop_db, out_path=out_dir / "char_variant_gallery.png",
                )

    # ------------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------------
    n_hapax = sum(1 for v in label_counts.values() if v == 1)
    n_rare  = sum(1 for v in label_counts.values() if v < args.rare_thresh)
    top10   = label_counts.most_common(10)
    top10_cov = sum(v for _, v in top10) / all_chars * 100

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
