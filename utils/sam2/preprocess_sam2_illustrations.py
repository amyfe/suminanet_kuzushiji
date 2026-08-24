"""Offline SAM2-based illustration mask generation for Kuzushiji Stage-1 training.

What this produces
------------------
For each training image, a binary boolean mask (IMAGE_SIZE resolution) that
marks illustration pixels — regions SAM2 identified as large non-text areas.

These masks are NOT applied to the images.  Instead, during training, cells
inside the illustration mask that have no annotated character receive an
upweighted focal loss (SAM2_HARD_NEG_WEIGHT × normal weight).  This teaches
the detector to suppress false positive activations inside illustrations while
still being exposed to full, unmodified Edo page images at training time.

How illustration regions are identified
----------------------------------------
A SAM2 mask is flagged as an illustration when ALL of:
  1. pixel area >= SAM2_ILLUS_MIN_AREA
  2. SAM2_ILLUS_AREA_THRESH <= frac <= SAM2_ILLUS_AREA_MAX_THRESH
     (not tiny character-sized, not the full-page parchment background)
  3. No GT character box centre falls inside the mask
     (definitive: text regions always contain annotated characters)

Installation
------------
  pip install git+https://github.com/facebookresearch/sam2.git
  wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt \\
       -P checkpoints/sam2/

Usage
-----
  python preprocess_sam2_illustrations.py [--splits train val] [--overwrite]

Output
------
  assets/data_sam2_masks/<image_stem>.npy   — float32 array shape IMAGE_SIZE, values in [0, 1]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

# Allow running as `python utils/sam2/preprocess_sam2_illustrations.py` (not
# just `python -m utils.sam2.preprocess_sam2_illustrations`) — Python only
# puts this script's own directory on sys.path, not the repo root, so the
# `config` import below fails without this.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import (
    DATA_DIR,
    DEVICE,
    IMAGE_SIZE,
    SAM2_CHECKPOINT,
    SAM2_ILLUS_AREA_MAX_THRESH,
    SAM2_ILLUS_AREA_THRESH,
    SAM2_ILLUS_AREA_SOFT_MARGIN,
    SAM2_ILLUS_MIN_AREA,
    SAM2_ILLUS_PRED_IOU_THRESH,
    SAM2_ILLUS_STABILITY_THRESH,
    SAM2_MASKS_DIR,
)
from utils.letterbox import letterbox_pil


# ---------------------------------------------------------------------------
# SAM2 loader
# ---------------------------------------------------------------------------

def _load_mask_generator(checkpoint: str, points_per_side: int = 32):
    try:
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    except ImportError as exc:
        raise ImportError(
            "SAM2 is required.  Install with:\n"
            "  pip install git+https://github.com/facebookresearch/sam2.git\n"
            "Download checkpoint:\n"
            "  wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
            "sam2.1_hiera_large.pt -P checkpoints/sam2/"
        ) from exc

    model = build_sam2("configs/sam2.1/sam2.1_hiera_l", checkpoint, device=DEVICE)
    return SAM2AutomaticMaskGenerator(
        model,
        points_per_side=points_per_side,
        pred_iou_thresh=SAM2_ILLUS_PRED_IOU_THRESH,
        stability_score_thresh=SAM2_ILLUS_STABILITY_THRESH,
        min_mask_region_area=SAM2_ILLUS_MIN_AREA,
    )


# ---------------------------------------------------------------------------
# Image + annotation collection
# ---------------------------------------------------------------------------

def _collect_entries(data_dir: Path, splits: List[str]) -> List[dict]:
    """
    Return one dict per image:
      {
        "img_path":  Path,
        "stem":      str,          
        "gt_boxes":  list[list],   # [[x1,y1,x2,y2], ...] original pixel coords
        "orig_size": (W, H),
      }
    """
    split_dir = data_dir / "splits"
    results: List[dict] = []

    for split in splits:
        split_file = split_dir / f"{split}.txt"
        if not split_file.exists():
            print(f"WARNING: split file {split_file} not found — skipping")
            continue
        names = {l.strip() for l in split_file.read_text().splitlines() if l.strip()}
        ann_dir = data_dir / "annotations"
        for ann_name in sorted(names):
            ann_path = ann_dir / ann_name
            if not ann_path.exists():
                continue
            try:
                ann = json.loads(ann_path.read_text())
                rel = ann.get("image_path") or ann.get("img_path") or ann.get("path")
                if not rel:
                    continue
                img_path = data_dir / rel
                if not img_path.exists():
                    continue
                with Image.open(img_path) as pil:
                    orig_w, orig_h = pil.size
                results.append({
                    "img_path":  img_path,
                    "stem":      img_path.stem,
                    "gt_boxes":  ann.get("boxes", []),
                    "orig_size": (orig_w, orig_h),
                })
            except Exception:
                pass

    return results


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _mask_contains_gt(
    seg: np.ndarray,            # (H, W) bool — at image resolution
    gt_boxes: list,             # [[x1,y1,x2,y2], ...] original image coords
    img_w: int,
    img_h: int,
    orig_w: int,
    orig_h: int,
    min_overlap_frac: float = 0.05,
) -> bool:
    """Return True if any GT character box overlaps the mask.

    Two checks are applied per box (either suffices):
    1. Center-point: fast single-pixel test (handles normal-sized characters).
    2. Area overlap: fraction of the GT bounding box covered by the mask >= min_overlap_frac.
       Needed for large calligraphic characters whose bounding-box center often falls on
       whitespace between ink strokes, causing the center-point test to miss the overlap.
    """
    if not gt_boxes:
        return False
    sx = img_w / orig_w
    sy = img_h / orig_h
    for (x1, y1, x2, y2) in gt_boxes:
        # Check 1: center point (original fast check)
        cx = max(0, min(img_w - 1, int((x1 + x2) * 0.5 * sx)))
        cy = max(0, min(img_h - 1, int((y1 + y2) * 0.5 * sy)))
        if seg[cy, cx]:
            return True
        # Check 2: bounding-box area overlap
        bx1 = max(0, int(x1 * sx))
        by1 = max(0, int(y1 * sy))
        bx2 = min(img_w, int(x2 * sx) + 1)
        by2 = min(img_h, int(y2 * sy) + 1)
        if bx2 > bx1 and by2 > by1:
            box_area = (bx2 - bx1) * (by2 - by1)
            overlap = int(seg[by1:by2, bx1:bx2].sum())
            if overlap > 0 and overlap / box_area >= min_overlap_frac:
                return True
    return False


def build_illustration_mask(
    image_rgb: np.ndarray,
    masks_data: list,
    gt_boxes: list,
    orig_size: tuple,
    area_thresh: float = SAM2_ILLUS_AREA_THRESH,
    area_max_thresh: float = SAM2_ILLUS_AREA_MAX_THRESH,
    area_soft_margin: float = SAM2_ILLUS_AREA_SOFT_MARGIN,
    min_area: int = SAM2_ILLUS_MIN_AREA,
    min_overlap_frac: float = 0.05,
) -> np.ndarray:
    """
    Build a (H, W) float32 mask with illustration confidence in [0, 1].

    Each SAM2 mask that is not a character region is included with a soft
    sigmoid weight based on its area fraction relative to the thresholds.
    Borderline masks (near area_thresh or area_max_thresh) get intermediate
    weight rather than being committed to a hard binary 0/1 decision.

    Weight formula:
        low_w  = sigmoid((frac - area_thresh)      / area_soft_margin)
        high_w = sigmoid((area_max_thresh - frac)  / area_soft_margin)
        weight = low_w * high_w   ∈ [0, 1]
    """
    h, w = image_rgb.shape[:2]
    total_pixels = h * w
    orig_w, orig_h = orig_size
    combined = np.zeros((h, w), dtype=np.float32)

    for mask_info in masks_data:
        area = int(mask_info["area"])
        if area < min_area:
            continue
        frac = float(area) / total_pixels

        # Soft sigmoid weight — tapers to 0 at both thresholds instead of a hard cutoff
        low_w  = 1.0 / (1.0 + np.exp(-(frac - area_thresh)      / area_soft_margin))
        high_w = 1.0 / (1.0 + np.exp(-(area_max_thresh - frac)  / area_soft_margin))
        weight = float(low_w * high_w)
        if weight < 0.05:   # effectively zero — skip for efficiency
            continue

        seg: np.ndarray = mask_info["segmentation"]
        if _mask_contains_gt(seg, gt_boxes, w, h, orig_w, orig_h, min_overlap_frac):
            continue
        combined = np.maximum(combined, seg.astype(np.float32) * weight)

    return combined


# ---------------------------------------------------------------------------
# Public API — called automatically by train_stage1.py
# ---------------------------------------------------------------------------

def run_preprocessing(
    splits: List[str],
    *,
    sam2_checkpoint: str | None = None,
    masks_dir: Path | None = None,
    area_thresh: float | None = None,
    area_max_thresh: float | None = None,
    min_area: int | None = None,
    min_overlap_frac: float = 0.05,
    points_per_side: int = 32,
    overwrite: bool = False,
) -> None:
    
    sam2_checkpoint = sam2_checkpoint or str(SAM2_CHECKPOINT)
    masks_dir    = Path(masks_dir or SAM2_MASKS_DIR)
    area_thresh     = area_thresh     if area_thresh     is not None else SAM2_ILLUS_AREA_THRESH
    area_max_thresh = area_max_thresh if area_max_thresh is not None else SAM2_ILLUS_AREA_MAX_THRESH
    min_area        = min_area        if min_area        is not None else SAM2_ILLUS_MIN_AREA

    masks_dir.mkdir(parents=True, exist_ok=True)
    train_h, train_w = IMAGE_SIZE

    data_dir = Path(DATA_DIR)
    entries = _collect_entries(data_dir, splits)
    if not entries:
        print("[SAM2] No images found — check splits and DATA_DIR.")
        return

    pending = [e for e in entries if not (masks_dir / f"{e['stem']}.npy").exists() or overwrite]
    if not pending:
        print(f"[SAM2] Mask cache complete ({len(entries)} images in {masks_dir}).")
        return

    print(
        f"[SAM2] Building illustration masks: {len(pending)}/{len(entries)} images pending  "
        f"(area {area_thresh:.0%}–{area_max_thresh:.0%}, GT-box filter on)"
    )
    generator = _load_mask_generator(sam2_checkpoint, points_per_side)

    n_processed = 0
    n_with_illus = 0

    for entry in tqdm(pending, desc="[SAM2]"):
        try:
            pil_img = Image.open(entry["img_path"]).convert("RGB")
        except Exception as exc:
            print(f"[SAM2] WARNING: could not open {entry['img_path']}: {exc}")
            continue

        img_rgb = np.array(pil_img)
        with torch.no_grad():
            masks_data = generator.generate(img_rgb)

        illus_mask = build_illustration_mask(
            image_rgb=img_rgb, 
            masks_data=masks_data, 
            gt_boxes=entry["gt_boxes"], 
            orig_size=entry["orig_size"],
            area_thresh=area_thresh, 
            area_max_thresh=area_max_thresh, 
            min_area=min_area, 
            min_overlap_frac=min_overlap_frac,
        )

        # Letterbox to training resolution using nearest-neighbour, matching
        # KuzushijiDataset's letterboxing so masks stay pixel-aligned with images.
        if illus_mask.max() > 0.01:
            pil_mask, _, _ = letterbox_pil(
                Image.fromarray(illus_mask, mode="F"),
                train_w,
                resample=Image.Resampling.NEAREST,
                fill=0.0,
            )
            illus_mask_resized = np.array(pil_mask, dtype=np.float32)
            n_with_illus += 1
        else:
            illus_mask_resized = np.zeros((train_h, train_w), dtype=np.float32)

        np.save(masks_dir / f"{entry['stem']}.npy", illus_mask_resized)
        n_processed += 1

    skipped = len(entries) - len(pending)
    print(
        f"[SAM2] Done. Processed {n_processed} | Skipped {skipped} | "
        f"Images with illustrations: {n_with_illus}/{n_processed}"
    )


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build SAM2 illustration mask cache for hard-negative training"
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--sam2_checkpoint", default=str(SAM2_CHECKPOINT))
    parser.add_argument("--masks_dir", default=str(SAM2_MASKS_DIR))
    parser.add_argument("--area_thresh",     type=float, default=SAM2_ILLUS_AREA_THRESH)
    parser.add_argument("--area_max_thresh", type=float, default=SAM2_ILLUS_AREA_MAX_THRESH)
    parser.add_argument("--min_area",          type=int,   default=SAM2_ILLUS_MIN_AREA)
    parser.add_argument("--min_overlap_frac",  type=float, default=0.05,
                        help="Min fraction of GT bounding-box area that must overlap a SAM2 "
                             "mask for it to be treated as a text region (default: 0.05).")
    parser.add_argument("--points_per_side", type=int,   default=32)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run_preprocessing(
        splits=args.splits,
        sam2_checkpoint=args.sam2_checkpoint,
        masks_dir=Path(args.masks_dir),
        area_thresh=args.area_thresh,
        area_max_thresh=args.area_max_thresh,
        min_area=args.min_area,
        min_overlap_frac=args.min_overlap_frac,
        points_per_side=args.points_per_side,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
