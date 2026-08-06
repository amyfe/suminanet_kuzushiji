"""
---------------------------
Konvertiert CODH Kuzushiji Datenstruktur in ein einheitliches JSON-Format:

data_preprocessed/
  ├── 100241706/
  │   ├── images/
  │   ├── characters/
  │   └── 100241706_coordinate.csv
  ├── 100241707/
  └── ...

Erzeugt:
  data/annotations/<image>.json
  data/label2id.json
"""

import pandas as pd
import json
import random
import shutil
import sys
from pathlib import Path
from PIL import Image

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, DATA_PREPROCESSED_DIR, DATA_ZIP_DIR, EXCLUDE_BOOKS, EXCLUDE_PAGES
from tqdm import tqdm

OUTPUT_ANNOT_DIR = DATA_DIR / "annotations"
OUTPUT_ANNOT_DIR.mkdir(exist_ok=True, parents=True)
OUTPUT_IMAGES_DIR = DATA_DIR  # images will be copied into DATA_DIR/<book_id>/images
OUTPUT_LABELS = DATA_DIR / "label2id.json"
OUTPUT_ID2LABEL = DATA_DIR / "id2label.json"
SPLITS_DIR = DATA_DIR / "splits"
SPLITS_DIR.mkdir(exist_ok=True, parents=True)

def unzip_file(zip_path: Path, extract_to: Path):
    """Entpackt eine ZIP-Datei in das angegebene Verzeichnis."""
    import zipfile

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def process_book(book_dir: Path):
    """Liest eine einzelne *_coordinate.csv und erzeugt JSON-Dateien für jedes Bild."""
    csv_files = list(book_dir.glob("*_coordinate.csv"))
    if not csv_files:
        print(f"⚠️ No coordinate CSV found in {book_dir}, skipping.")
        return set()

    csv_path = csv_files[0]
    df = pd.read_csv(csv_path)
    print(f"📄 Processing {csv_path.name} with {len(df)} entries.")

    required_cols = {"Unicode", "Image", "X", "Y", "Width", "Height"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing important columns in {csv_path.name}")

    all_labels = set()
    # Keyed by book_dir.name (not the per-image book_id heuristic below, which
    # mis-derives for books like umgy00000 whose filenames don't match the dir name).
    excluded_stems = {Path(f).stem for f in EXCLUDE_PAGES.get(book_dir.name, [])}

    for _, row in df.iterrows():
        img_name_base = str(row["Image"]).strip()
        if img_name_base in excluded_stems:
            # Reserved as a held-out test page — added separately by
            # scripts/onetime_scripts/add_excluded_books.py.
            continue
        img_dir = book_dir / "images"
        # flexible Extension suchen
        matches = list(img_dir.glob(f"{img_name_base}.*"))
        if not matches:
            print(f"⚠️ Image not found: {img_dir / img_name_base}, skipping.")
            continue
        img_path = matches[0]

        # Copy image into DATA_DIR/<book_id>/images/
        # Extract base book_id (e.g., "hnsd00000" from "hnsd00000_0001_1.jpg")
        # Split by underscore and take first parts that are alphabetic/numeric prefix
        parts = img_path.name.split("_")
        # Find the base book_id by removing numeric suffixes like _0001, _0002, etc.
        book_id = parts[0]
        
        # If filename has pattern like "hnsd00000_0001_1.jpg", keep just "hnsd00000"
        # by checking if subsequent parts are 4-digit numbers (like 0001, 0002)
        if len(parts) > 1 and parts[1].isdigit() and len(parts[1]) == 4:
            book_id = parts[0]  # Already correct
        elif len(parts) > 1:
            # Otherwise, might need to combine: e.g., if it's numeric_id_variant
            # For numeric-only book IDs, just take first part
            book_id = parts[0]
        dest_img_dir = OUTPUT_IMAGES_DIR / book_id / "images"
        dest_img_dir.mkdir(parents=True, exist_ok=True)
        dest_img_path = dest_img_dir / img_path.name
        if not dest_img_path.exists():
            shutil.copy2(img_path, dest_img_path)

        # load image to get size and clamp boxes
        try:
            with Image.open(dest_img_path) as im:
                img_w, img_h = im.size
        except Exception:
            print(f"⚠️ Could not open image {dest_img_path}, skipping.")
            continue

        x_min = int(row["X"])
        y_min = int(row["Y"])
        width = int(row["Width"])
        height = int(row["Height"])
        x_max = x_min + width
        y_max = y_min + height

        # clamp to image bounds
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(img_w - 1, x_max)
        y_max = min(img_h - 1, y_max)
        if x_max <= x_min or y_max <= y_min:
            # skip degenerate boxes
            continue

        label = str(row["Unicode"]).strip()
        all_labels.add(label)

        # JSON-Datei pro Bild (flat annotations dir)
        annot_file = OUTPUT_ANNOT_DIR / (img_path.name.replace(img_path.suffix, ".json"))
        if annot_file.exists():
            with open(annot_file, "r", encoding="utf-8") as f:
                annot = json.load(f)
        else:
            annot = {"boxes": [], "labels": [], "image_path": None}

        annot["boxes"].append([x_min, y_min, x_max, y_max])
        annot["labels"].append(label)
        # store relative image path from DATA_DIR root: <book_id>/images/<image>
        annot["image_path"] = str((book_id + "/images/" + img_path.name))

        with open(annot_file, "w", encoding="utf-8") as f:
            json.dump(annot, f, ensure_ascii=False, indent=2)

    return all_labels


def main():
    #Get all data from zip files if any
    # zip_files = list(DATA_ZIP_DIR.glob("*.zip"))
    # for zip_file in zip_files:
    #     print(f"Unzipping {zip_file} to {DATA_PREPROCESSED_DIR}...")
    #     unzip_file(zip_file, DATA_PREPROCESSED_DIR)
    
    all_labels = set()
    
    # Check which books are already processed by looking at existing folders in DATA_DIR
    processed_books = set()
    if OUTPUT_IMAGES_DIR.exists():
        # Look for existing book folders (those with /images subdirectory)
        for item in OUTPUT_IMAGES_DIR.iterdir():
            if item.is_dir() and (item / "images").exists():
                processed_books.add(item.name)
    
    # Alle Buch-Ordner durchsuchen
    book_dirs = [d for d in DATA_PREPROCESSED_DIR.iterdir() if d.is_dir()]
    print(f"📚 Found {len(book_dirs)} book directories.")
    print(f"📋 Already processed: {len(processed_books)} books")
    
    # Filter to only unprocessed books
    unprocessed_dirs = [d for d in book_dirs if d.name not in processed_books and d.name not in EXCLUDE_BOOKS]
    print(f"⏳ To process: {len(unprocessed_dirs)} books")

    for book_dir in tqdm(unprocessed_dirs, desc="Processing books"):
        book_labels = process_book(book_dir)
        all_labels.update(book_labels)

    # Label-Mapping speichern
    label2id = {label: idx for idx, label in enumerate(sorted(all_labels))}
    with open(OUTPUT_LABELS, "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)

    # also save id2label for convenience
    id2label = {str(v): k for k, v in label2id.items()}
    with open(OUTPUT_ID2LABEL, "w", encoding="utf-8") as f:
        json.dump(id2label, f, ensure_ascii=False, indent=2)

    # create a simple train/val split
    ann_files = sorted([p.name for p in OUTPUT_ANNOT_DIR.glob("*.json")])
    random.seed(42)
    random.shuffle(ann_files)
    train_frac = 0.9
    split_idx = int(len(ann_files) * train_frac)
    train_list = ann_files[:split_idx]
    val_list = ann_files[split_idx:]

    with open(SPLITS_DIR / "train.txt", "w", encoding="utf-8") as f:
        for n in train_list:
            f.write(n + "\n")
    with open(SPLITS_DIR / "val.txt", "w", encoding="utf-8") as f:
        for n in val_list:
            f.write(n + "\n")

    print(f"✅ Wrote splits: {len(train_list)} train, {len(val_list)} val to {SPLITS_DIR}")

    print(f"\n✅ Saved {len(all_labels)} unique labels to: {OUTPUT_LABELS}")
    print(f"✅ Annotations saved to: {OUTPUT_ANNOT_DIR}")


if __name__ == "__main__":
    main()
