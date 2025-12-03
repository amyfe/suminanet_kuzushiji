import json
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class KuzushijiDataset(Dataset):
    """
    Kuzushiji OCR Dataset.
    
    Erwartete Struktur:
    
    root/
        100241706/
            images/*.jpg
        annotations/*.json
        splits/train.txt  (optional)
    """
    def __init__(self, root_dir, vocab=None, use_sequences=True, transform=None, resize=(512,512)):
        self.root_dir = Path(root_dir)
        self.annotations_dir = self.root_dir / "annotations"
        self.resize_to = resize
        self.use_sequences = use_sequences
        self.vocab = vocab

        self.transform = transform or T.Compose([
            T.ToTensor()
        ])

        # ---------------------------
        # Sammle alle Annotationen
        # ---------------------------
        self.ann_files = sorted(list(self.annotations_dir.glob("*.json")))
        
        # Precompute image paths: safer than reconstructing later
        self.items = []
        for ann_file in self.ann_files:
            with open(ann_file, "r", encoding="utf-8") as f:
                ann = json.load(f)

            img_path = ann.get("image_path")
            if img_path is None:
                # fallback: derive based on your structure
                fn = ann_file.stem + ".jpg"
                book_id = fn.split("_")[0]
                img_path = f"{book_id}/images/{fn}"

            self.items.append({
                "ann_file": ann_file,
                "img_path": self.root_dir / img_path
            })

    def __len__(self):
        return len(self.items)

    # ----------------------------------------------------------
    # Main loader
    # ----------------------------------------------------------
    def __getitem__(self, idx):
        item = self.items[idx]
        ann_file = item["ann_file"]
        img_path = item["img_path"]

        # Load image
        image = Image.open(img_path).convert("RGB")

        with open(ann_file, "r", encoding="utf-8") as f:
            ann = json.load(f)

        boxes = ann.get("boxes", [])
        labels = ann.get("labels", [])
        orientation = ann.get("orientation", "horizontal")

        # ---------------------------
        # Resize + rescale boxes
        # ---------------------------
        orig_w, orig_h = image.size
        new_w, new_h = self.resize_to

        if (orig_w, orig_h) != (new_w, new_h):
            image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
            sx = new_w / orig_w
            sy = new_h / orig_h
            boxes = [[x1*sx, y1*sy, x2*sx, y2*sy] for (x1,y1,x2,y2) in boxes]

        # ---------------------------
        # OCR CRITICAL: Sort boxes in reading order
        # ---------------------------
        # Sort by y1 first, then x1
        if len(boxes) > 0:
            sorted_indices = sorted(
                range(len(boxes)),
                key=lambda i: (boxes[i][1], boxes[i][0])
            )

            boxes = [boxes[i] for i in sorted_indices]
            labels = [labels[i] for i in sorted_indices]

        # Convert to tensors
        boxes = torch.tensor(boxes, dtype=torch.float32)
        label_ids = None
        if self.vocab is not None:
            label_ids = [self.vocab.char2id.get(l, self.vocab.unk_id) for l in labels]

        image = self.transform(image)

        sample = {
            "image": image,
            "boxes": boxes,
            "labels": torch.tensor(label_ids, dtype=torch.long) if label_ids else None,
            "raw_labels": labels,
            "orientation": orientation,
        }

        # ---------------------------
        # Sequence for ATTENTION Decoder
        # ---------------------------
        if self.use_sequences and self.vocab is not None:
            seq = labels  # already sorted
            seq_ids = self.vocab.encode(seq, add_sos=True, add_eos=True)

            sample["text_ids"] = torch.tensor(seq_ids, dtype=torch.long)
            sample["text_length"] = len(seq_ids)
            sample["text_chars"] = seq

        return sample
