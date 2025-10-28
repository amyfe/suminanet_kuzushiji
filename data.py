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

# class PageDataset(Dataset):
#     """Loads full-page images and annotation metadata.

#     Expected annotation format (per item):
#       {
#         "image_path": str,
#         "boxes": [ [x1,y1,x2,y2], ... ],
#         "labels": [ int, ... ],
#         "sequences": [list of label ids in reading order]  # optional
#       }
#     """
#     def __init__(self, items: List[Dict], transforms=None):
#         validate_items(items)
#         self.items = items
#         self.transforms = transforms or T.Compose([
#             T.Resize((1024, 768)),
#             T.ToTensor(),
#         ])

#     def __len__(self):
#         return len(self.items)

#     def __getitem__(self, idx):
#         meta = self.items[idx]
#         try:
#             img = Image.open(meta["image_path"]).convert("RGB")
#         except Exception as e:
#             raise RuntimeError(f"Failed to load image for item {idx}: {meta['image_path']}. Error: {e}")
#         img_t = self.transforms(img)

#         # convert boxes/labels -> tensors
#         boxes = torch.tensor(meta.get("boxes", []), dtype=torch.float32)
#         labels = torch.tensor(meta.get("labels", []), dtype=torch.long)
#         sequence = torch.tensor(meta.get("sequence", []), dtype=torch.long) if meta.get("sequence") is not None else None

#         return {
#             "image": img_t,
#             "boxes": boxes,
#             "labels": labels,
#             "sequence": sequence,
#             "meta": meta,
#         }


# def make_dataloader(items, batch_size=BATCH_SIZE, shuffle=True):
#     print(f"Creating dataloader with {len(items)} items, batch_size={batch_size}, shuffle={shuffle}")
#     ds = PageDataset(items)
#     print(f"Dataset page size: {len(ds)} samples")
#     return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=NUM_WORKERS, collate_fn=collate_fn)


# def collate_fn(batch):
#     # simple collate, keep variable-length boxes as list
#     images = torch.stack([b['image'] for b in batch], dim=0)
#     boxes = [b['boxes'] for b in batch]
#     labels = [b['labels'] for b in batch]
#     sequences = [b['sequence'] for b in batch]
#     metas = [b['meta'] for b in batch]
#     return {
#         'images': images,
#         'boxes': boxes,
#         'labels': labels,
#         'sequences': sequences,
#         'metas': metas,
#     }