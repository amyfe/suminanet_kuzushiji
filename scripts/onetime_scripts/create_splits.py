"""Create train/validation splits for the dataset."""
import random
from pathlib import Path

SEED = 42
random.seed(SEED)

def create_splits(data_dir, train_ratio=0.8):
    """
    Create train/val splits by splitting annotation files.
    
    Args:
        data_dir: Path to data directory
        train_ratio: Ratio of training data (default 0.8 for 80/20 split)
    """
    data_dir = Path(data_dir)
    annotations_dir = data_dir / "annotations"
    splits_dir = data_dir / "splits"
    
    # Get all annotation files
    ann_files = sorted(list(annotations_dir.glob("*.json")))
    
    if len(ann_files) == 0:
        raise FileNotFoundError(f"No annotation files found in {annotations_dir}")
    
    print(f"Found {len(ann_files)} annotation files")
    
    # Shuffle annotations
    random.shuffle(ann_files)
    
    # Split
    split_idx = int(len(ann_files) * train_ratio)
    train_files = ann_files[:split_idx]
    val_files = ann_files[split_idx:]
    
    print(f"Train: {len(train_files)} files ({len(train_files)/len(ann_files)*100:.1f}%)")
    print(f"Val: {len(val_files)} files ({len(val_files)/len(ann_files)*100:.1f}%)")
    
    # Create splits directory
    splits_dir.mkdir(exist_ok=True)
    
    # Write train split
    train_path = splits_dir / "train.txt"
    with open(train_path, 'w') as f:
        for ann_file in train_files:
            f.write(f"{ann_file.stem}\n")
    print(f"Created: {train_path}")
    
    # Write val split
    val_path = splits_dir / "val.txt"
    with open(val_path, 'w') as f:
        for ann_file in val_files:
            f.write(f"{ann_file.stem}\n")
    print(f"Created: {val_path}")
    
    print(f"\nSplits created successfully!")
    print(f"To use splits, update your dataset initialization:")
    print(f"  train_dataset = KuzushijiDataset(..., split='train')")
    print(f"  val_dataset = KuzushijiDataset(..., split='val')")

if __name__ == "__main__":
    # Use relative path to data directory
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "assets" / "data"
    create_splits(data_dir, train_ratio=0.8)
