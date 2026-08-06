"""Visualize ROIReadingOrder's column clustering + reading-order sequencing,
and the resulting column-vs-block break classification (the fix from the
translation-pipeline comparison), on a real page.

Draws, on top of the source image:
  - each character box outlined in a color keyed to its detected column id
  - a small reading-order rank number above each box (0-indexed)
  - a path connecting consecutive boxes in reading order:
      green  = normal step, no break
      orange = column wrap only (no content break)
      red (thick, with a circle) = genuine detected block boundary

Uses the same sample page as scripts/compare_translation_pipelines.py by default.

Usage:
  python scripts/visualize_reading_order.py
  python scripts/visualize_reading_order.py --annotation assets/data/annotations/<other>.json
"""
from __future__ import annotations

import argparse
import colorsys
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from PIL import Image, ImageDraw, ImageFont

from config import DATA_DIR
from model.kuronet.roi.roi_ordering import ROIReadingOrder
from model.translation.translation import _detect_block_breaks, _detect_column_breaks

SAMPLE_ANNOTATION = ROOT / "assets" / "data" / "annotations" / "200021660_00003_1.json"
OUTPUT_DIR = ROOT / "results" / "reading_order_viz"


def _column_color(col_id: int, num_cols: int) -> tuple[int, int, int]:
    """Distinct, stable color per column id via evenly-spaced hues."""
    hue = (col_id % max(1, num_cols)) / max(1, num_cols)
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def _load_font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--annotation", type=Path, default=SAMPLE_ANNOTATION)
    p.add_argument("--orientation", type=str, default="vertical", choices=["vertical", "horizontal"])
    p.add_argument("--gap-factor", type=float, default=2.5, help="Fallback distance-heuristic gap factor (only used when col_id is unavailable -- irrelevant here, kept for parity with translation.py's default).")
    args = p.parse_args()

    ann = json.loads(args.annotation.read_text(encoding="utf-8"))
    img_path = Path(DATA_DIR) / ann["image_path"]
    boxes = ann["boxes"]
    if not boxes:
        raise ValueError(f"{args.annotation} has no boxes to visualize.")

    image = Image.open(img_path).convert("RGB")

    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    mask = torch.ones((boxes_tensor.size(0),), dtype=torch.bool)
    sorted_boxes, _, _sort_idx, col_ids = ROIReadingOrder().sort_single(boxes_tensor, mask, args.orientation)

    sorted_boxes_list = sorted_boxes.tolist()
    col_ids_list = col_ids.tolist()
    num_cols = len(set(col_ids_list))

    # Build chars already in reading-order sequence -- _detect_*_breaks expect
    # chars[i]/chars[i+1] to be adjacent in reading order, which sorted_boxes/
    # col_ids already are (no permutation-inversion needed here, unlike
    # compare_translation_pipelines.py which needs original annotation order).
    chars = [{"box": b, "col_id": c} for b, c in zip(sorted_boxes_list, col_ids_list)]
    column_breaks = _detect_column_breaks(chars)
    block_breaks = _detect_block_breaks(chars, args.gap_factor)

    draw = ImageDraw.Draw(image)
    font = _load_font(22)

    centers = []
    for rank, (box, col_id) in enumerate(zip(sorted_boxes_list, col_ids_list)):
        x1, y1, x2, y2 = box
        color = _column_color(col_id, num_cols)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        centers.append((cx, cy))
        draw.text((x1, max(0, y1 - 24)), str(rank), fill=color, font=font)

    # Reading-order path + break markers, drawn last so it sits on top.
    for i in range(1, len(centers)):
        p0, p1 = centers[i - 1], centers[i]
        if i in block_breaks:
            draw.line([p0, p1], fill=(255, 0, 0), width=9)
            mx, my = (p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0
            r = 14
            draw.ellipse([mx - r, my - r, mx + r, my + r], outline=(255, 0, 0), width=4)
        elif i in column_breaks:
            draw.line([p0, p1], fill=(38, 0, 255), width=6)
        else:
            draw.line([p0, p1], fill=(0, 200, 0), width=4)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{args.annotation.stem}_reading_order.png"
    image.save(out_path)

    print(f"Source image: {img_path}")
    print(f"Chars: {len(chars)}  Columns detected: {num_cols}")
    print(f"Column-wrap breaks (orange): {len(column_breaks)}")
    print(f"Genuine block breaks (red): {len(block_breaks)}  at reading-order indices {sorted(block_breaks)}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
