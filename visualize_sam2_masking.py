"""Visual verification for SAM2 illustration masks.

Two modes:

  --from_cache  (fast, no SAM2)
      Loads the saved .npy mask from SAM2_MASKS_DIR and overlays it on the
      original image.  Use this to verify the preprocessing output.

  default  (slow, runs SAM2)
      Re-runs SAM2 AutomaticMaskGenerator live.  Use this to tune thresholds
      before running the full preprocessing job.

Also:

  --stats
      Prints a summary across the whole cached dataset (no image output).

Usage
-----
  # Verify a cached mask — instant, no GPU needed
  python visualize_sam2_masking.py path/to/image.jpg --from_cache

  # Re-run SAM2 live to tune thresholds
  python visualize_sam2_masking.py path/to/image.jpg --area_thresh 0.08

  # Dataset-wide cache stats
  python visualize_sam2_masking.py --stats
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from config import (
    DATA_DIR,
    DEVICE,
    IMAGE_SIZE,
    SAM2_CHECKPOINT,
    SAM2_ILLUS_AREA_MAX_THRESH,
    SAM2_ILLUS_AREA_THRESH,
    SAM2_ILLUS_MIN_AREA,
    SAM2_ILLUS_PRED_IOU_THRESH,
    SAM2_ILLUS_STABILITY_THRESH,
    SAM2_MASKS_DIR,
)
from preprocess_sam2_illustrations import _mask_contains_gt

_RED  = (220, 50,  50,  140)   # illustration
_BLUE = (50,  150, 220, 80)    # text region (skipped by GT filter)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_sam2(checkpoint: str):
    try:
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    except ImportError as exc:
        raise ImportError(
            "SAM2 not installed.  Run:\n"
            "  pip install git+https://github.com/facebookresearch/sam2.git"
        ) from exc
    model = build_sam2("configs/sam2.1/sam2.1_hiera_l", checkpoint, device=DEVICE)
    return SAM2AutomaticMaskGenerator(
        model,
        points_per_side=32,
        pred_iou_thresh=SAM2_ILLUS_PRED_IOU_THRESH,
        stability_score_thresh=SAM2_ILLUS_STABILITY_THRESH,
        min_mask_region_area=SAM2_ILLUS_MIN_AREA,
    )


def _find_annotation(img_path: Path):
    import json
    ann_dir = Path(DATA_DIR) / "annotations"
    for ann_path in ann_dir.glob("*.json"):
        try:
            ann = json.loads(ann_path.read_text())
            rel = ann.get("image_path") or ann.get("img_path") or ""
            if Path(rel).name == img_path.name:
                return ann
        except Exception:
            pass
    return None


def _add_label(img_arr: np.ndarray, text: str) -> np.ndarray:
    pil = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(pil)
    draw.text((11, 11), text, fill=(0, 0, 0))
    draw.text((10, 10), text, fill=(255, 255, 100))
    return np.array(pil)


def _save_outputs(img_rgb: np.ndarray, illus_mask: np.ndarray,
                  out_dir: Path, stem: str, label: str) -> None:
    """Overlay the boolean mask on the image and write PNGs."""
    h, w = img_rgb.shape[:2]

    # Resize mask to image size if needed (cache is at IMAGE_SIZE)
    if illus_mask.shape != (h, w):
        illus_mask = np.array(
            Image.fromarray(illus_mask).resize((w, h), Image.NEAREST), dtype=bool
        )

    overlay = Image.fromarray(img_rgb).convert("RGBA")
    colour_arr = np.zeros((h, w, 4), dtype=np.uint8)
    colour_arr[illus_mask] = _RED
    overlay = Image.alpha_composite(overlay, Image.fromarray(colour_arr, "RGBA"))
    overlay_arr = np.array(overlay.convert("RGB"))

    frac = illus_mask.mean() * 100
    n_regions = _count_regions(illus_mask)
    overlay_arr = _add_label(overlay_arr,
        f"{label} — {frac:.1f}% masked, ~{n_regions} region(s)")

    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = out_dir / f"{stem}_overlay.png"
    Image.fromarray(overlay_arr).save(overlay_path)
    print(f"Saved: {overlay_path}")
    print(f"  Masked pixels: {frac:.1f}%  |  Approx regions: {n_regions}")


def _count_regions(mask: np.ndarray) -> int:
    """Rough connected-component count without scipy dependency."""
    try:
        from scipy import ndimage
        labeled, n = ndimage.label(mask)
        return n
    except ImportError:
        # Fallback: count transitions from 0→1 in each row
        return max(1, int(np.diff(mask.any(axis=1).astype(int)).clip(min=0).sum()))


# ---------------------------------------------------------------------------
# From-cache mode
# ---------------------------------------------------------------------------

def verify_from_cache(img_path: Path, out_dir: Path) -> None:
    stem = img_path.stem
    mask_path = Path(SAM2_MASKS_DIR) / f"{stem}.npy"

    if not mask_path.exists():
        print(f"No cached mask found at {mask_path}")
        print("Run  python preprocess_sam2_illustrations.py  first.")
        return

    illus_mask = np.load(mask_path)   # (H_train, W_train) bool
    img_rgb = np.array(Image.open(img_path).convert("RGB"))

    any_masked = illus_mask.any()
    print(f"Cache mask shape: {illus_mask.shape}  |  Has illustration: {any_masked}")
    _save_outputs(img_rgb, illus_mask, out_dir, stem, "Cached mask")


# ---------------------------------------------------------------------------
# Live SAM2 mode
# ---------------------------------------------------------------------------

def verify_live(img_path: Path, out_dir: Path, args) -> None:
    import torch

    ann = _find_annotation(img_path)
    gt_boxes = ann.get("boxes", []) if ann else []
    print(f"GT boxes found: {len(gt_boxes)}")

    pil_img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = pil_img.size
    img_rgb = np.array(pil_img)
    h, w = img_rgb.shape[:2]

    generator = _load_sam2(args.sam2_checkpoint)
    print("Running SAM2 ...")
    with torch.no_grad():
        masks_data = generator.generate(img_rgb)

    total_px = h * w
    all_areas = sorted([(m["area"] / total_px, m["area"]) for m in masks_data], reverse=True)
    print(f"Total masks: {len(masks_data)}")
    print(f"Largest 5 (fraction, px²): {all_areas[:5]}")

    in_window = [
        m for m in masks_data
        if m["area"] >= args.min_area
        and args.area_thresh <= m["area"] / total_px <= args.area_max_thresh
    ]
    n_text  = sum(1 for m in in_window
                  if gt_boxes and _mask_contains_gt(
                      m["segmentation"], gt_boxes, w, h, orig_w, orig_h))
    n_illus = len(in_window) - n_text
    n_bg    = sum(1 for m in masks_data if m["area"] / total_px > args.area_max_thresh)

    print(f"In area window: {len(in_window)}  →  illustration={n_illus}  text(skipped)={n_text}  bg={n_bg}")

    # Build combined masks for output
    illus_combined = np.zeros((h, w), dtype=bool)
    text_combined  = np.zeros((h, w), dtype=bool)
    for m in in_window:
        seg = m["segmentation"]
        has_text = gt_boxes and _mask_contains_gt(seg, gt_boxes, w, h, orig_w, orig_h)
        if has_text:
            text_combined |= seg
        else:
            illus_combined |= seg

    # Overlay: red=illustration, blue=text regions that were skipped
    overlay = Image.fromarray(img_rgb).convert("RGBA")
    for mask, colour in [(illus_combined, _RED), (text_combined, _BLUE)]:
        if mask.any():
            arr = np.zeros((h, w, 4), dtype=np.uint8)
            arr[mask] = colour
            overlay = Image.alpha_composite(overlay, Image.fromarray(arr, "RGBA"))

    overlay_arr = np.array(overlay.convert("RGB"))
    overlay_arr = _add_label(overlay_arr,
        f"Red=illustration({n_illus})  Blue=text-skipped({n_text})  BG-excl({n_bg})")

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = img_path.stem
    out_path = out_dir / f"{stem}_live_overlay.png"
    Image.fromarray(overlay_arr).save(out_path)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Stats mode
# ---------------------------------------------------------------------------

def print_stats() -> None:
    masks_dir = Path(SAM2_MASKS_DIR)
    if not masks_dir.exists():
        print(f"Cache directory not found: {masks_dir}")
        print("Run  python preprocess_sam2_illustrations.py  first.")
        return

    npy_files = list(masks_dir.glob("*.npy"))
    if not npy_files:
        print(f"No .npy mask files found in {masks_dir}")
        return

    fracs = []
    n_with_illus = 0
    for f in npy_files:
        mask = np.load(f)
        frac = float(mask.mean())
        fracs.append(frac)
        if frac > 0:
            n_with_illus += 1

    fracs = np.array(fracs)
    print(f"Cache: {masks_dir}")
    print(f"Total masks: {len(npy_files)}")
    print(f"Images with illustration regions: {n_with_illus} ({100*n_with_illus/len(npy_files):.1f}%)")
    print(f"Masked area — mean: {fracs.mean()*100:.2f}%  "
          f"median: {np.median(fracs)*100:.2f}%  "
          f"max: {fracs.max()*100:.2f}%")

    suspicious = [(f.stem, frac) for f, frac in zip(npy_files, fracs) if frac > 0.40]
    if suspicious:
        print(f"\nWARNING: {len(suspicious)} image(s) have >40% masked — check these:")
        for stem, frac in sorted(suspicious, key=lambda x: -x[1])[:10]:
            print(f"  {stem}  {frac*100:.1f}%")
        print("  Run:  python visualize_sam2_masking.py <image_path> --from_cache")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Verify SAM2 illustration masks")
    parser.add_argument("image", nargs="?", help="Path to input image (omit when using --stats)")
    parser.add_argument("--from_cache", action="store_true",
                        help="Load saved .npy mask — fast, no SAM2 needed")
    parser.add_argument("--stats", action="store_true",
                        help="Print dataset-wide cache statistics")
    parser.add_argument("--sam2_checkpoint", default=str(SAM2_CHECKPOINT))
    parser.add_argument("--area_thresh",     type=float, default=SAM2_ILLUS_AREA_THRESH)
    parser.add_argument("--area_max_thresh", type=float, default=SAM2_ILLUS_AREA_MAX_THRESH)
    parser.add_argument("--min_area",        type=int,   default=SAM2_ILLUS_MIN_AREA)
    parser.add_argument("--out_dir",         default="debug_vis")
    args = parser.parse_args()

    if args.stats:
        print_stats()
        return

    if not args.image:
        parser.error("Provide an image path, or use --stats for dataset-wide summary.")

    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(img_path)
    out_dir = Path(args.out_dir)

    if args.from_cache:
        verify_from_cache(img_path, out_dir)
    else:
        verify_live(img_path, out_dir, args)


if __name__ == "__main__":
    main()
