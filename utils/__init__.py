import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from pathlib import Path


def build_label_mapping(ann_dir):
    ann_dir = Path(ann_dir)
    labels = set()
    for p in ann_dir.glob("*.json"):
        with open(p, "r", encoding="utf-8") as f:
            ann = json.load(f)
        for lbl in ann.get("labels", []):
            labels.add(lbl)
    labels = sorted(labels)
    label2id = {l:i for i,l in enumerate(labels)}
    id2label = {i:l for l,i in label2id.items()}
    return label2id, id2label

class KuzushijiDataset(Dataset):
    """
    Dataset für Demo-/Kuzushiji-Daten.
    Erwartet Struktur:
    data/
        images/
            *.jpg
        annotations/
            *.json
    """
    def __init__(self, root_dir, transform=None, label2id=None):
        self.root_dir = root_dir
        self.transform = transform or T.Compose([T.ToTensor()])
        self.images_dir = os.path.join(root_dir, "images")
        self.anns_dir = os.path.join(root_dir, "annotations")
        self.images = sorted(os.listdir(self.images_dir))
        if label2id is None:
            self.label2id, self.id2label = build_label_mapping(self.anns_dir)
        else:
            self.label2id = label2id or {}
            self.id2label = {v:k for k,v in self.label2id.items()}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        ann_path = os.path.join(self.anns_dir, img_name.replace(".jpg", ".json"))

        # Bild laden
        image = Image.open(img_path).convert("RGB")

        # JSON-Annotation laden
        with open(ann_path, "r", encoding="utf-8") as f:
            ann = json.load(f)
        boxes = torch.tensor(ann["boxes"], dtype=torch.float32)  # [N,4]
        labels = ann["labels"]  # Liste von Strings

        # Optionale Transformation
        if self.transform:
            image = self.transform(image)

        # Tensor für das Bild (C,H,W)
        image = T.ToTensor()(image)

        sample = {
            "image": image,
            "boxes": boxes,
            "labels": labels,
            "raw_labels": [self.id2label[int(i)] for i in labels] if len(labels)>0 else []
        }
        return sample
