import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from config import AVG_GT_PER_IMAGE, IMAGE_SIZE, SUMINANET_COPY_PASTE_PROB, SAM2_MASKS_DIR, SAM2_PREPROCESSING
from model.suminanet.roi.roi_ordering import infer_reading_orientation_from_boxes, ROIReadingOrder


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
    def __init__(self, root_dir, vocab=None, use_sequences=True, transform=None, resize=None,
                 split=None, rare_chars: Optional[FrozenSet[str]] = None):
        """
        Args:
            root_dir: Path to data directory
            vocab: VocabManager instance
            use_sequences: Whether to return text sequences
            transform: Image transforms (None for default based on split)
            resize: Target image size (height, width). Defaults to config.IMAGE_SIZE when None.
            split: 'train', 'val', or None (use all data)
            rare_chars: Optional frozenset of character labels considered rare.
                        When set and the image contains ≥1 rare character, a stronger
                        augmentation transform is applied (Option A).
        """
        self.root_dir = Path(root_dir)
        self.annotations_dir = self.root_dir / "annotations"
        self.resize_to = resize if resize is not None else IMAGE_SIZE
        self.use_sequences = use_sequences
        self.vocab = vocab
        self.split = split
        self.rare_chars = rare_chars  # frozenset[str] or None

        if transform is None:
            # Separate transforms for train vs val: validation should NOT have augmentation
            if split == 'train':
                self.transform = T.Compose([
                    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform = T.Compose([
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transform

        # Stronger pixel-level augmentation for images containing rare characters (Option A).
        # Only active when rare_chars is provided and split == 'train'.
        if rare_chars and split == 'train':
            from utils.char_augmentation import build_rare_char_transform
            self.rare_transform: Optional[T.Compose] = build_rare_char_transform()
        else:
            self.rare_transform = None

        # Copy-paste crop database — built after self.items is populated below.
        # Set to None here; assigned in _build_copy_paste_db() called at the end of __init__.
        self.copy_paste_db: Optional[Dict[str, List[np.ndarray]]] = None

        # Cached once per dataset instance — do not instantiate fresh on every __getitem__ call.
        self.roi_order = ROIReadingOrder()

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
                "ann": ann,
                "img_path": self.root_dir / img_path,
                "n_gt": len(ann.get("boxes", [])),   # exact GT count for per-image density filter
            })

        # Option C: build copy-paste crop database (train only, requires cv2).
        if rare_chars and split == 'train':
            self.copy_paste_db = self._build_copy_paste_db(rare_chars, max_per_char=100)

    def _build_copy_paste_db(
        self,
        rare_chars: FrozenSet[str],
        max_per_char: int = 100,
    ) -> Optional[Dict[str, List[np.ndarray]]]:
        """
        Build a per-character crop database at target resolution for copy-paste augmentation.

        Crops are extracted at the same scale as the resized training images so they
        can be pasted directly without further scaling. Only images containing at least
        one rare character are loaded, keeping init time low.

        Returns None if cv2 is not available.
        """
        try:
            import cv2
        except ImportError:
            return None

        db: Dict[str, List[np.ndarray]] = defaultdict(list)
        new_w, new_h = self.resize_to

        for item in self.items:
            ann = item["ann"]
            labels = ann.get("labels", [])
            boxes = ann.get("boxes", [])

            if not any(lbl in rare_chars for lbl in labels):
                continue

            try:
                img_pil = Image.open(item["img_path"]).convert("RGB")
            except Exception:
                continue

            orig_w, orig_h = img_pil.size
            if (orig_w, orig_h) != (new_w, new_h):
                img_pil = img_pil.resize((new_w, new_h), Image.Resampling.BILINEAR)
                sx = new_w / orig_w
                sy = new_h / orig_h
                boxes_sc = [[x1 * sx, y1 * sy, x2 * sx, y2 * sy] for (x1, y1, x2, y2) in boxes]
            else:
                boxes_sc = list(boxes)

            # PIL RGB → numpy BGR (cv2 convention used by copy_paste_rare_chars)
            img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

            for lbl, box in zip(labels, boxes_sc):
                if lbl not in rare_chars:
                    continue
                if len(db[lbl]) >= max_per_char:
                    continue
                x1, y1, x2, y2 = (int(round(v)) for v in box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(new_w, x2), min(new_h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                db[lbl].append(img_bgr[y1:y2, x1:x2].copy())

        if not db:
            return None

        result = dict(db)
        print(
            f"[augmentation] copy-paste db: {len(result)} rare chars, "
            f"{sum(len(v) for v in result.values())} total crops"
        )
        return result

    def __len__(self):
        return len(self.items)

    # ----------------------------------------------------------
    # Main loader
    # ----------------------------------------------------------
    def __getitem__(self, idx):
        item = self.items[idx]
        img_path = item["img_path"]
        ann = item["ann"]

        # Load image
        image = Image.open(img_path).convert("RGB")

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
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            mask = torch.ones((boxes_tensor.size(0),), dtype=torch.bool)
            _, _, sort_idx, _col_ids = self.roi_order.sort_single(boxes_tensor, mask, orientation)
            sorted_indices = sort_idx.detach().cpu().tolist()
            boxes = [boxes[i] for i in sorted_indices]
            labels = [labels[i] for i in sorted_indices]

        # Option C: copy-paste rare character crops (40% of training steps).
        # Applied on the resized PIL image before the tensor transform so that
        # crops are at the correct 512×512 scale. New boxes are appended in-place;
        # IoU-based GT matching in the loss is order-independent so no re-sort needed.
        if self.copy_paste_db is not None and random.random() < SUMINANET_COPY_PASTE_PROB:
            try:
                import cv2
                from utils.char_augmentation import copy_paste_rare_chars
                img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                img_bgr, boxes, labels = copy_paste_rare_chars(
                    image=img_bgr,
                    boxes=boxes,
                    labels=labels,
                    crop_db=self.copy_paste_db,
                    rare_chars=self.rare_chars if self.rare_chars is not None else frozenset(),
                    n_paste=2,
                )
                image = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            except Exception as exc:
                import warnings
                warnings.warn(f"copy_paste_rare_chars failed (non-fatal): {exc}", stacklevel=2)

        # Convert to tensors
        boxes = torch.tensor(boxes, dtype=torch.float32)
        label_ids = None
        if self.vocab is not None:
            label_ids = [self.vocab.char2id.get(l, self.vocab.unk_id) for l in labels]

        # Option A: apply stronger augmentation if image contains a rare character
        if self.rare_transform is not None and self.rare_chars and any(
            lbl in self.rare_chars for lbl in labels
        ):
            image = self.rare_transform(image)
        else:
            image = self.transform(image)

        # SAM2 illustration mask — loaded when SAM2_PREPROCESSING=True.
        # Shape: (1, H, W) float32 tensor at IMAGE_SIZE, values in [0, 1].
        # Zero tensor if file missing. Old bool masks are upcast cleanly by .float().
        illus_mask = None
        if SAM2_PREPROCESSING:
            mask_path = Path(SAM2_MASKS_DIR) / f"{img_path.stem}.npy"
            if mask_path.exists():
                arr = np.load(mask_path)
                illus_mask = torch.from_numpy(arr).float().unsqueeze(0)  # (1, H, W) float32
            else:
                h_t, w_t = self.resize_to
                illus_mask = torch.zeros((1, h_t, w_t), dtype=torch.float32)

        n_gt = self.items[idx]["n_gt"]
        sample = {
            "image": image,
            "boxes": boxes,
            "labels": torch.tensor(label_ids, dtype=torch.long) if label_ids else None,
            "raw_labels": labels,
            "orientation": orientation,
            "image_stem": img_path.stem,
            # Per-image GT count for adaptive spatial density filter; falls back to global
            # AVG_GT_PER_IMAGE when n_gt is 0 (illustrated/blank pages).
            "avg_gt_per_image": float(n_gt) if n_gt > 0 else AVG_GT_PER_IMAGE,
        }
        if illus_mask is not None:
            sample["illus_mask"] = illus_mask

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
