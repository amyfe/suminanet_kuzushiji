"""
Utilities for handling class imbalance in Kuzushiji character recognition.

Provides:
  - compute_char_frequencies: count occurrences per character in training annotations
  - get_rare_chars: return frozenset of characters below a frequency threshold
  - compute_sample_weights: per-image weights for WeightedRandomSampler (Option B)
  - build_rare_char_transform: stronger augmentation transform for rare-char images (Option A)
  - build_rare_char_crop_db: prebuilt database of (crop_array, char_label) for rare chars
  - copy_paste_rare_chars: paste rare-char crops onto an image, updating GT boxes/labels
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np
import torchvision.transforms as T


def compute_char_frequencies(ann_files: List[Path]) -> Counter:
    """Count how often each character label appears across the given annotation files."""
    counter: Counter = Counter()
    for f in ann_files:
        data = json.loads(Path(f).read_text(encoding="utf-8"))
        for lbl in data.get("labels", []):
            counter[lbl] += 1
    return counter


def get_rare_chars(char_counter: Counter, threshold: int) -> FrozenSet[str]:
    """Return characters that appear fewer than `threshold` times."""
    return frozenset(ch for ch, cnt in char_counter.items() if cnt < threshold)


def compute_sample_weights(
    items: list,
    char_counter: Counter,
    threshold: int,
    boost_scale: float = 3.0,
) -> List[float]:
    """
    Compute a per-image sampling weight for WeightedRandomSampler (Option B).

    Images containing rare characters (freq < threshold) receive a higher weight
    so they are sampled more often per epoch.  The boost grows as frequency drops:

        boost_i = sqrt(threshold / max(1, freq(char_i)))

    The total image weight is 1 + boost_scale * sum(boosts), so images with no rare
    characters get weight=1 (unchanged sampling rate) and images rich in rare
    characters can get weights up to ~10×.

    Args:
        items: list of dataset items, each with ann['labels']
        char_counter: Counter from compute_char_frequencies (training split only)
        threshold: characters below this frequency are considered rare
        boost_scale: scalar applied to the summed per-char boost per image
    """
    weights: List[float] = []
    for item in items:
        labels = item["ann"].get("labels", [])
        boost = 0.0
        for lbl in labels:
            freq = char_counter.get(lbl, 0)
            if freq < threshold:
                boost += math.sqrt(threshold / max(1, freq))
        weights.append(1.0 + boost_scale * boost)
    return weights


def build_rare_char_crop_db(
    ann_files: List[Path],
    image_dir: Path,
    rare_chars: FrozenSet[str],
    max_per_char: int = 200,
) -> Dict[str, List[np.ndarray]]:
    """
    Build a database of raw image crops for every rare character.

    For each annotation file, loads the image and crops each bounding box
    whose label is in rare_chars.  Crops are stored as uint8 HWC numpy arrays.

    Args:
        ann_files:     annotation JSON files (training split only)
        image_dir:     root directory containing the page images
        rare_chars:    frozenset of character labels considered rare
        max_per_char:  cap stored crops per character to avoid memory blow-up

    Returns:
        dict mapping char_label -> list of uint8 HWC crops
    """
    try:
        import cv2
    except ImportError as e:
        raise ImportError("build_rare_char_crop_db requires opencv-python.") from e

    db: Dict[str, List[np.ndarray]] = defaultdict(list)

    for ann_path in ann_files:
        data = json.loads(ann_path.read_text(encoding="utf-8"))
        labels = data.get("labels", [])
        boxes  = data.get("bboxes", data.get("boxes", []))
        if not labels or not boxes:
            continue

        # Derive image path: annotation stem matches image filename
        img_name = ann_path.stem
        img_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = image_dir / (img_name + ext)
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        for lbl, box in zip(labels, boxes):
            if lbl not in rare_chars:
                continue
            if len(db[lbl]) >= max_per_char:
                continue

            x1, y1, x2, y2 = (int(round(v)) for v in box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            crop = img[y1:y2, x1:x2]
            db[lbl].append(crop)

    return dict(db)


def copy_paste_rare_chars(
    image: np.ndarray,                        # uint8 HWC BGR
    boxes: List[List[float]],                 # [[x1,y1,x2,y2], ...] existing GT
    labels: List[str],                        # existing GT labels
    crop_db: Dict[str, List[np.ndarray]],     # from build_rare_char_crop_db
    rare_chars: FrozenSet[str],
    n_paste: int = 3,
    iou_thresh: float = 0.05,
    max_attempts: int = 15,
) -> Tuple[np.ndarray, List[List[float]], List[str]]:
    """
    Paste n_paste randomly selected rare-char crops onto image at empty positions.

    Each crop is placed at a random location that does not significantly overlap
    any existing GT box (IoU < iou_thresh).  Successful pastes are added to
    boxes and labels so the model sees them as supervised characters.

    Args:
        image:        input image (modified in-place copy)
        boxes:        existing GT boxes [[x1,y1,x2,y2], ...]
        labels:       existing GT labels
        crop_db:      prebuilt {label: [crop_array, ...]}
        rare_chars:   set of rare character labels (only paste these)
        n_paste:      number of crops to paste per call
        iou_thresh:   maximum allowable IoU with any existing box
        max_attempts: position-search attempts per crop before giving up

    Returns:
        (augmented_image, new_boxes, new_labels)
    """
    img = image.copy()
    img_h, img_w = img.shape[:2]
    new_boxes  = [list(b) for b in boxes]
    new_labels = list(labels)

    available = [c for c in crop_db if c in rare_chars and crop_db[c]]
    if not available:
        return img, new_boxes, new_labels

    for _ in range(n_paste):
        lbl   = random.choice(available)
        crop  = random.choice(crop_db[lbl]).copy()
        ch, cw = crop.shape[:2]
        if cw <= 0 or ch <= 0 or cw > img_w or ch > img_h:
            continue

        placed = False
        for _attempt in range(max_attempts):
            px = random.randint(0, img_w - cw)
            py = random.randint(0, img_h - ch)
            candidate = [float(px), float(py), float(px + cw), float(py + ch)]

            # Check IoU with every existing box
            ok = True
            for eb in new_boxes:
                ix1 = max(candidate[0], eb[0])
                iy1 = max(candidate[1], eb[1])
                ix2 = min(candidate[2], eb[2])
                iy2 = min(candidate[3], eb[3])
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                if inter == 0.0:
                    continue
                area_c = cw * ch
                area_e = (eb[2] - eb[0]) * (eb[3] - eb[1])
                iou = inter / (area_c + area_e - inter + 1e-6)
                if iou > iou_thresh:
                    ok = False
                    break

            if ok:
                img[py:py + ch, px:px + cw] = crop
                new_boxes.append(candidate)
                new_labels.append(lbl)
                placed = True
                break

        if not placed:
            continue  # silently skip if no valid position found

    return img, new_boxes, new_labels


def build_rare_char_transform() -> T.Compose:
    """
    Stronger augmentation transform for images containing rare characters.

    Replaces the standard train transform with a stronger version:
      - More aggressive ColorJitter (brightness/contrast ±0.4 vs ±0.2)
      - GaussianBlur — mimics ink bleeding / uneven pressure on old paper
      - RandomGrayscale — simulates ink fading on aged paper

    These are pixel-level transforms and do NOT require box-coordinate updates.
    """
    return T.Compose([
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.05),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        T.RandomGrayscale(p=0.10),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
