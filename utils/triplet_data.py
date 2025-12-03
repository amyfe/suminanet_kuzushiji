import json
import random
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset

class TripletDataset(Dataset):
    """
    Triplet-Dataset basierend auf deinen CODH-Annotations.
    Erwartet:
      data/
        annotations/*.json
        splits/train.txt
        <bookid>/images/<image>.jpg
    """
    def __init__(self, data_dir, split="train", transform=None):
        self.data_dir = Path(data_dir)
        self.annot_dir = self.data_dir / "annotations"
        self.transform = transform

        # Splits einlesen
        split_file = self.data_dir / "splits" / f"{split}.txt"
        with open(split_file, "r", encoding="utf-8") as f:
            self.annot_files = [line.strip() for line in f]

        # Annotationen laden
        self.samples = []
        self.label_to_indices = {}

        for fname in self.annot_files:
            ann_path = self.annot_dir / fname
            with open(ann_path, "r", encoding="utf-8") as f:
                ann = json.load(f)

            img_path = self.data_dir / ann["image_path"]

            for box, label in zip(ann["boxes"], ann["labels"]):
                entry = {
                    "img_path": img_path,
                    "label": label,
                    "box": box
                }
                self.samples.append(entry)

                self.label_to_indices.setdefault(label, []).append(len(self.samples) - 1)

        self.labels = list(self.label_to_indices.keys())

    def __len__(self):
        return len(self.samples)

    def load_patch(self, entry):
        img = Image.open(entry["img_path"]).convert("RGB")
        x1, y1, x2, y2 = entry["box"]
        patch = img.crop((x1, y1, x2, y2))

        if self.transform:
            patch = self.transform(patch)

        return patch

    def __getitem__(self, idx):
        anchor_entry = self.samples[idx]
        anchor_label = anchor_entry["label"]

        # === Positive Sample ===
        pos_list = self.label_to_indices[anchor_label]
        pos_idx = idx
        while pos_idx == idx:  # kein identischer Patch
            pos_idx = random.choice(pos_list)
        positive_entry = self.samples[pos_idx]

        # === Negative Sample ===
        neg_label = random.choice([l for l in self.labels if l != anchor_label])
        neg_idx = random.choice(self.label_to_indices[neg_label])
        negative_entry = self.samples[neg_idx]

        # Croppen + transform
        anchor = self.load_patch(anchor_entry)
        positive = self.load_patch(positive_entry)
        negative = self.load_patch(negative_entry)

        return anchor, positive, negative, anchor_label
