"""
cleanup_annotations.py
----------------------
Deletes annotations that:
1. Start with letters (e.g., "brsk00000")
2. Belong to EXCLUDE_BOOKS from config.py
"""

import sys
from pathlib import Path
import json
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, EXCLUDE_BOOKS

OUTPUT_ANNOT_DIR = DATA_DIR / "annotations"
OUTPUT_LABELS = DATA_DIR / "label2id.json"
OUTPUT_ID2LABEL = DATA_DIR / "id2label.json"
SPLITS_DIR = DATA_DIR / "splits"

def is_numeric_book_id(filename: str) -> bool:
    """Check if book_id (first part before first underscore) is numeric."""
    book_id = filename.split("_")[0].replace(".json", "")
    return book_id.isdigit()

def should_delete(ann_filename: str) -> tuple:
    """
    Returns (should_delete: bool, reason: str)
    """
    book_id = ann_filename.split("_")[0].replace(".json", "")
    
    # Check if starts with letters
    if not book_id[0].isdigit():
        return True, f"starts with letter ({book_id})"
    
    # Check if in EXCLUDE_BOOKS
    if book_id in EXCLUDE_BOOKS:
        return True, f"in EXCLUDE_BOOKS ({book_id})"
    
    return False, ""

def cleanup():
    """Delete unwanted annotations."""
    
    if not OUTPUT_ANNOT_DIR.exists():
        print(f"❌ Annotations directory not found: {OUTPUT_ANNOT_DIR}")
        return
    
    ann_files = sorted(list(OUTPUT_ANNOT_DIR.glob("*.json")))
    print(f"📊 Total annotation files: {len(ann_files)}")
    
    to_delete = []
    for ann_file in ann_files:
        should_del, reason = should_delete(ann_file.name)
        if should_del:
            to_delete.append((ann_file, reason))
    
    if not to_delete:
        print("✅ No annotations to delete (all are numeric and not excluded)")
        return
    
    print(f"\n🗑️  Deleting {len(to_delete)} annotations...")
    
    # Delete files
    deleted_count = 0
    for ann_file, reason in tqdm(to_delete, desc="Deleting"):
        try:
            ann_file.unlink()
            deleted_count += 1
        except Exception as e:
            print(f"⚠️  Failed to delete {ann_file.name}: {e}")
    
    print(f"\n✅ Deleted {deleted_count} annotation files")
    
    # Rebuild labels from remaining annotations
    print("\n🔄 Rebuilding label mappings...")
    remaining_files = sorted(list(OUTPUT_ANNOT_DIR.glob("*.json")))
    all_labels = set()
    
    for ann_file in remaining_files:
        with open(ann_file, "r", encoding="utf-8") as f:
            ann = json.load(f)
            labels = ann.get("labels", [])
            all_labels.update(labels)
    
    # Preserve existing IDs where possible
    if OUTPUT_LABELS.exists():
        with open(OUTPUT_LABELS, "r", encoding="utf-8") as f:
            old_label2id = json.load(f)
    else:
        old_label2id = {}
    
    # Keep only labels that still exist in annotations
    label2id = {label: idx for label, idx in old_label2id.items() if label in all_labels}
    
    # Add any new labels (shouldn't happen, but just in case)
    max_id = max(label2id.values()) if label2id else -1
    for label in sorted(all_labels):
        if label not in label2id:
            max_id += 1
            label2id[label] = max_id
    
    with open(OUTPUT_LABELS, "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)
    
    id2label = {str(v): k for k, v in label2id.items()}
    with open(OUTPUT_ID2LABEL, "w", encoding="utf-8") as f:
        json.dump(id2label, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Updated labels: {len(label2id)} unique characters")
    
    # Rebuild splits (remove deleted files from splits)
    print("\n🔄 Rebuilding train/val splits...")
    remaining_filenames = set(f.stem + ".json" for f in remaining_files)
    
    if (SPLITS_DIR / "train.txt").exists():
        with open(SPLITS_DIR / "train.txt", "r") as f:
            train_list = [line.strip() for line in f if line.strip() in remaining_filenames]
        with open(SPLITS_DIR / "train.txt", "w") as f:
            for name in train_list:
                f.write(name + "\n")
        print(f"   Train split: {len(train_list)} files")
    
    if (SPLITS_DIR / "val.txt").exists():
        with open(SPLITS_DIR / "val.txt", "r") as f:
            val_list = [line.strip() for line in f if line.strip() in remaining_filenames]
        with open(SPLITS_DIR / "val.txt", "w") as f:
            for name in val_list:
                f.write(name + "\n")
        print(f"   Val split: {len(val_list)} files")
    
    print(f"\n✅ Cleanup complete!")
    print(f"   Remaining annotations: {len(remaining_files)}")
    print(f"   Remaining labels: {len(label2id)}")

if __name__ == "__main__":
    cleanup()
