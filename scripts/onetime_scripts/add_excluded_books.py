"""
add_excluded_books.py
----------------------
One-time backfill for pipeline_analysis.txt item 9.4.

The 4 books below were previously fully skipped via config.EXCLUDE_BOOKS, so
every character that only ever appears in them was invisible to the model.
This script processes those 4 books, but instead of excluding them wholesale
it holds out only a handful of pages per book as a genuine test set (the
pages listed in config.EXCLUDE_PAGES), and adds the rest to train/val.

config.EXCLUDE_PAGES is the source of truth for which pages are held out —
prepare_codh_annotations.py also reads it and skips those same pages, so a
future from-scratch rebuild (wipe assets/data, rerun prepare_codh_annotations
then this script) reproduces the same held-out test set. If a book has no
entry in EXCLUDE_PAGES yet (e.g. a new book being added for the first time),
this script falls back to picking 3-5 pages deterministically via
config.SEED and prints them so they can be added to EXCLUDE_PAGES.

Safe to re-run: per-image idempotent (skips any page that already has an
annotation JSON, whether written by this script or by
prepare_codh_annotations.py). Unlike prepare_codh_annotations.main(), this
MERGES into the existing label2id.json / id2label.json instead of
overwriting them, and appends to splits/*.txt instead of rewriting them.

Usage:
    python scripts/onetime_scripts/add_excluded_books.py
"""

import json
import random
import shutil
import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import DATA_DIR, DATA_PREPROCESSED_DIR, EXCLUDE_PAGES, SEED

BOOKS = ["200021925", "200022050", "200025191", "umgy00000"]

OUTPUT_ANNOT_DIR = DATA_DIR / "annotations"
OUTPUT_LABELS = DATA_DIR / "label2id.json"
OUTPUT_ID2LABEL = DATA_DIR / "id2label.json"
SPLITS_DIR = DATA_DIR / "splits"

REQUIRED_COLS = {"Unicode", "Image", "X", "Y", "Width", "Height"}


def process_image(book_dir: Path, book_name: str, img_name_base: str, group: pd.DataFrame):
    """Build one annotation JSON + copy the image. Returns (stem, filename, labels) or None."""
    annot_file = OUTPUT_ANNOT_DIR / f"{img_name_base}.json"
    if annot_file.exists():
        # Already processed (by this script or prepare_codh_annotations.py) — skip.
        return None

    img_dir = book_dir / "images"
    matches = list(img_dir.glob(f"{img_name_base}.*"))
    if not matches:
        print(f"  [warn] image not found for {img_name_base}, skipping")
        return None
    img_path = matches[0]

    dest_img_dir = DATA_DIR / book_name / "images"
    dest_img_dir.mkdir(parents=True, exist_ok=True)
    dest_img_path = dest_img_dir / img_path.name
    if not dest_img_path.exists():
        shutil.copy2(img_path, dest_img_path)

    try:
        with Image.open(dest_img_path) as im:
            img_w, img_h = im.size
    except Exception:
        print(f"  [warn] could not open {dest_img_path}, skipping")
        return None

    boxes, labels = [], []
    for _, row in group.iterrows():
        x_min = max(0, int(row["X"]))
        y_min = max(0, int(row["Y"]))
        x_max = min(img_w - 1, x_min + int(row["Width"]))
        y_max = min(img_h - 1, y_min + int(row["Height"]))
        if x_max <= x_min or y_max <= y_min:
            continue
        boxes.append([x_min, y_min, x_max, y_max])
        labels.append(str(row["Unicode"]).strip())

    if not boxes:
        print(f"  [warn] no valid boxes for {img_name_base}, skipping")
        return None

    annot = {
        "boxes": boxes,
        "labels": labels,
        "image_path": f"{book_name}/images/{img_path.name}",
    }
    with open(annot_file, "w", encoding="utf-8") as f:
        json.dump(annot, f, ensure_ascii=False, indent=2)

    return annot_file.stem, img_path.name, set(labels)


def main():
    rng = random.Random(SEED)

    all_new_labels = set()
    train_stems, val_stems, test_stems = [], [], []
    unconfigured_held_out_report = {}

    for book_name in BOOKS:
        book_dir = DATA_PREPROCESSED_DIR / book_name

        csv_files = list(book_dir.glob("*_coordinate.csv"))
        if not csv_files:
            print(f"[warn] no coordinate CSV for {book_name}, skipping book")
            continue
        df = pd.read_csv(csv_files[0])
        if not REQUIRED_COLS.issubset(df.columns):
            raise ValueError(f"Missing required columns in {csv_files[0].name}")

        unique_images = sorted(df["Image"].astype(str).str.strip().unique())

        configured = EXCLUDE_PAGES.get(book_name)
        if configured:
            held_out = {Path(f).stem for f in configured}
            print(f"[book] {book_name}: {len(unique_images)} pages, "
                  f"{len(held_out)} held out per config.EXCLUDE_PAGES")
        else:
            k = 5 if len(unique_images) >= 10 else 3
            held_out = set(rng.sample(unique_images, k))
            print(f"[book] {book_name}: {len(unique_images)} pages, holding out {k} as test "
                  f"(not yet in config.EXCLUDE_PAGES — add the printed list below)")

        book_nonheld_stems = []
        book_held_files = []
        for img_name_base in tqdm(unique_images, desc=book_name):
            group = df[df["Image"].astype(str).str.strip() == img_name_base]
            result = process_image(book_dir, book_name, img_name_base, group)
            if result is None:
                continue
            stem, filename, labels = result
            all_new_labels.update(labels)
            if img_name_base in held_out:
                test_stems.append(stem)
                book_held_files.append(filename)
            else:
                book_nonheld_stems.append(stem)

        rng.shuffle(book_nonheld_stems)
        split_idx = int(len(book_nonheld_stems) * 0.9)
        train_stems.extend(book_nonheld_stems[:split_idx])
        val_stems.extend(book_nonheld_stems[split_idx:])
        if not configured and book_held_files:
            unconfigured_held_out_report[book_name] = sorted(book_held_files)

    if not all_new_labels and not train_stems and not test_stems:
        print("Nothing to do.")
        return

    # Merge labels into existing label2id.json (never overwrite existing ids)
    with open(OUTPUT_LABELS, "r", encoding="utf-8") as f:
        label2id = json.load(f)
    next_id = max(label2id.values(), default=-1) + 1
    added = 0
    for label in sorted(all_new_labels):
        if label not in label2id:
            label2id[label] = next_id
            next_id += 1
            added += 1
    with open(OUTPUT_LABELS, "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)

    id2label = {str(v): k for k, v in label2id.items()}
    with open(OUTPUT_ID2LABEL, "w", encoding="utf-8") as f:
        json.dump(id2label, f, ensure_ascii=False, indent=2)

    # Append (never rewrite) split files
    def append_split(name, stems):
        path = SPLITS_DIR / f"{name}.txt"
        with open(path, "a", encoding="utf-8") as f:
            for stem in stems:
                f.write(stem + ".json\n")

    append_split("train", train_stems)
    append_split("val", val_stems)
    append_split("test", test_stems)

    print("\n" + "=" * 60)
    print(f"New unique labels added: {added} (label2id now has {len(label2id)})")
    print(f"train.txt +{len(train_stems)}  val.txt +{len(val_stems)}  test.txt +{len(test_stems)}")
    if unconfigured_held_out_report:
        print("\nThese books had no config.EXCLUDE_PAGES entry — add them so future "
              "rebuilds hold out the same pages:")
        for book_name, files in unconfigured_held_out_report.items():
            print(f"  {book_name!r}: {files!r},")


if __name__ == "__main__":
    main()
