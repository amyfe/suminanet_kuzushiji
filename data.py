"""Dataset and dataloader utilities.
This is a minimal example: you will need to adapt loaders to your annotation
format (images, box annotations, per-box class ids, reading order sequences).
"""
import random
from pathlib import Path
from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

from config import DATA_DIR, BATCH_SIZE, NUM_WORKERS, SEED

random.seed(SEED)

def validate_items(items):
    """Check that all items have required keys and valid paths."""
    required_keys = ["image_path", "boxes", "labels"]
    for i, item in enumerate(items):
        for k in required_keys:
            if k not in item:
                raise KeyError(f"Item {i} is missing required key: '{k}'. Item: {item}")
        if not Path(item["image_path"]).exists():
            raise FileNotFoundError(f"Image path does not exist for item {i}: {item['image_path']}")