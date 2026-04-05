import json
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from config import IMAGE_SIZE
from model.kuronet.roi.roi_ordering import infer_reading_orientation_from_boxes


class KuzushijiDataset(Dataset):
    """
    Kuzushiji OCR Dataset.
    
    Erwartete Struktur:
    
    root/
        100241706/
            images/*.jpg
        annotations/*.json
        splits/train.txt  (optional)
            splits/val.txt    (optional)
    """
    def __init__(self, root_dir, vocab=None, use_sequences=True, transform=None, resize=None, split=None):
        """
        Args:
            root_dir: Path to data directory
            vocab: VocabManager instance
            use_sequences: Whether to return text sequences
            transform: Image transforms (None for default based on split)
            resize: Target image size (height, width). Defaults to config.IMAGE_SIZE when None.
            split: 'train', 'val', or None (use all data)
        """
        self.root_dir = Path(root_dir)
        self.annotations_dir = self.root_dir / "annotations"
        self.resize_to = resize if resize is not None else IMAGE_SIZE
        self.use_sequences = use_sequences
        self.vocab = vocab
        self.split = split

        if transform is None:
            # Separate transforms for train vs val: validation should NOT have augmentation
            if split == 'train':
                # Training: use augmentation
                self.transform = T.Compose([
                    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                    # T.RandomAffine(degrees=2, translate=(0.05, 0.05), scale=(0.95, 1.05)), boxes not tranformed yet
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                # Validation/test: no augmentation, only normalize
                self.transform = T.Compose([
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transform

        # ---------------------------
        # Sammle alle Annotationen
        # ---------------------------
        self.ann_files = sorted(list(self.annotations_dir.glob("*.json")))
        
        # Filter by split if specified
        if split is not None:
            split_file = self.root_dir / "splits" / f"{split}.txt"
            if not split_file.exists():
                raise FileNotFoundError(f"Expected split file at {split_file} but it does not exist. Refusing to fall back to full dataset.")

            with open(split_file, 'r') as f:
                lines = [ln.strip() for ln in f.read().splitlines() if ln.strip()]
            # Strip optional .json suffix from ids to match annotation stems
            valid_ids = set([ln[:-5] if ln.endswith('.json') else ln for ln in lines])

            self.ann_files = [f for f in self.ann_files if f.stem in valid_ids]
            print(f"Loaded {split} split: {len(self.ann_files)} files")

            if len(self.ann_files) == 0:
                raise ValueError(
                    f"Split file {split_file} produced zero matches against annotations."
                    " Ensure IDs match annotation filenames (with or without .json suffix)."
                )
        
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
        # Reading direction is inferred from box layout (annotation orientation is not required).
        orientation = infer_reading_orientation_from_boxes(boxes)

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
        if len(boxes) > 0:
            if orientation == "vertical":
                # Traditional Japanese: right-to-left columns, top-to-bottom within columns
                # Sort by -x (right to left), then by y (top to bottom)
                sorted_indices = sorted(
                    range(len(boxes)),
                    key=lambda i: (-boxes[i][0], boxes[i][1])
                )
            else:  # horizontal
                # Left-to-right, top-to-bottom
                # Sort by y first (top to bottom), then x (left to right)
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
