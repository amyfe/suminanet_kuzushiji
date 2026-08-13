"""Visualize ROIReadingOrder's column clustering + reading-order sequencing,
comparing ground-truth boxes against the actual app inference pipeline
(detector + SuminaNet + ROIReadingOrder, run exactly as /api/transcribe does),
on a real page.

Draws two panels side by side:
  - GROUND TRUTH:   annotation boxes -> ROIReadingOrder (orientation forced)
  - MODEL (APP):    run_inference() on the raw image, same code path and
                     defaults as app/backend/app.py's /api/transcribe

Each panel shows, on top of the source image:
  - each character box outlined in a color keyed to its detected column id
  - a small reading-order rank number above each box (0-indexed)
  - a solid "column line" through the mean position of each detected
    column/row, spanning its extent -- makes over-/under-segmentation of
    columns on dense pages immediately visible
  - a path connecting consecutive boxes in reading order:
      green  = normal step, no break
      orange = column wrap only (no content break)
      red (thick, with a circle) = genuine detected block boundary

This is the tool for answering "does the app's column clustering on dense
text actually match reality?" -- the GT panel uses clean, hand-labeled boxes
so its column split is the reference; the MODEL panel uses whatever the
detector actually produced, warts and all.

Uses the same sample page as scripts/compare_translation_pipelines.py by default.

Usage:
  python scripts/visualize_reading_order.py
  python scripts/visualize_reading_order.py --annotation assets/data/annotations/<other>.json
  python scripts/visualize_reading_order.py --skip-model   # GT panel only, no model load
"""
from __future__ import annotations

import argparse
import colorsys
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from PIL import Image, ImageDraw, ImageFont

from config import DATA_DIR, SUMINANET_BG_SCORE_GATE, SUMINANET_CER_SCORE_THRESH
from model.suminanet.roi.roi_ordering import ROIReadingOrder
from utils.training_helpers.helper_translation import _detect_block_breaks, _detect_column_breaks

SAMPLE_ANNOTATION = ROOT / "assets" / "data" / "annotations" / "200022050_00005_2.json"
OUTPUT_DIR = ROOT / "results" / "reading_order_viz"


def _column_color(col_id: int, num_cols: int) -> tuple[int, int, int]:
    """Distinct, stable color per column id via evenly-spaced hues."""
    if col_id < 0:
        return (128, 128, 128)
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


def _draw_column_lines(
    draw: ImageDraw.ImageDraw,
    boxes_list: list,
    col_ids_list: list,
    orientation: str,
    num_cols: int,
) -> None:
    """Draw a solid line through the mean position of each detected
    column (vertical) / row (horizontal), spanning its own box extent.

    This is the literal geometry ROIReadingOrder's column clustering
    produced -- a merged/split column on dense text shows up directly as a
    too-wide gap between lines, or two lines sitting on top of each other.
    """
    groups: dict[int, list] = defaultdict(list)
    for box, cid in zip(boxes_list, col_ids_list):
        if cid < 0:
            continue
        groups[cid].append(box)

    pad = 10
    for cid, boxes in groups.items():
        color = _column_color(cid, num_cols)
        if orientation == "vertical":
            mean_x = sum((b[0] + b[2]) / 2.0 for b in boxes) / len(boxes)
            y_min = min(b[1] for b in boxes)
            y_max = max(b[3] for b in boxes)
            draw.line([(mean_x, y_min - pad), (mean_x, y_max + pad)], fill=color, width=2)
        else:
            mean_y = sum((b[1] + b[3]) / 2.0 for b in boxes) / len(boxes)
            x_min = min(b[0] for b in boxes)
            x_max = max(b[2] for b in boxes)
            draw.line([(x_min - pad, mean_y), (x_max + pad, mean_y)], fill=color, width=2)


def _render_panel(
    source: Image.Image,
    boxes_list: list,
    col_ids_list: list,
    chars: list,
    orientation: str,
    gap_factor: float,
    label: str,
    font: ImageFont.ImageFont,
) -> tuple[Image.Image, int, set, set]:
    """Render one annotated panel (boxes, ranks, column lines, reading-order
    path with break classification) and a header bar with summary stats.

    Returns (panel_image, num_cols, column_breaks, block_breaks).
    """
    panel = source.copy()
    draw = ImageDraw.Draw(panel)
    num_cols = len(set(c for c in col_ids_list if c >= 0)) or 1

    column_breaks = _detect_column_breaks(chars)
    block_breaks = _detect_block_breaks(chars, gap_factor)

    centers = []
    for rank, (box, col_id) in enumerate(zip(boxes_list, col_ids_list)):
        x1, y1, x2, y2 = box
        color = _column_color(col_id, num_cols)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        centers.append((cx, cy))
        draw.text((x1, max(0, y1 - 24)), str(rank), fill=color, font=font)

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

    _draw_column_lines(draw, boxes_list, col_ids_list, orientation, num_cols)

    header_h = 44
    header = Image.new("RGB", (panel.width, header_h), (20, 20, 20))
    hd = ImageDraw.Draw(header)
    hd.text(
        (10, 10),
        f"{label}   chars={len(chars)}  cols={num_cols}  orientation={orientation}  "
        f"col-wraps={len(column_breaks)}  block-breaks={len(block_breaks)}",
        fill=(255, 255, 255),
        font=font,
    )
    combined = Image.new("RGB", (panel.width, panel.height + header_h))
    combined.paste(header, (0, 0))
    combined.paste(panel, (0, header_h))
    return combined, num_cols, column_breaks, block_breaks


def _run_gt_panel(image: Image.Image, boxes: list, orientation: str, gap_factor: float, font):
    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    mask = torch.ones((boxes_tensor.size(0),), dtype=torch.bool)
    sorted_boxes, _, _sort_idx, col_ids = ROIReadingOrder().sort_single(boxes_tensor, mask, orientation)

    boxes_list = sorted_boxes.tolist()
    col_ids_list = col_ids.tolist()
    chars = [{"box": b, "col_id": c} for b, c in zip(boxes_list, col_ids_list)]

    return _render_panel(image, boxes_list, col_ids_list, chars, orientation, gap_factor, "GROUND TRUTH", font)


def _run_model_panel(
    image: Image.Image,
    img_path: Path,
    args: argparse.Namespace,
    gap_factor: float,
    font,
):
    """Run the exact code path app/backend/app.py's /api/transcribe uses:
    load_image() letterbox -> run_inference() -> _unletterbox_boxes() back
    into the same original-image pixel space the GT boxes are already in.
    """
    from model.translation.infer import (
        _DEFAULT_SUMINANET_CKPT,
        _unletterbox_boxes,
        load_image,
        load_suminanet,
        run_inference,
    )
    from train_stage2_suminanet import load_vocab

    ckpt = Path(args.ckpt) if args.ckpt else _DEFAULT_SUMINANET_CKPT
    print(f"Loading vocab + SuminaNet checkpoint: {ckpt}")
    vocab = load_vocab()
    model = load_suminanet(ckpt, vocab)

    image_tensor, orig_size, scale, pad = load_image(img_path)

    print("Running app inference pipeline (detector + SuminaNet + ROIReadingOrder)...")
    result = run_inference(
        model,
        image_tensor,
        vocab,
        orientation=args.model_orientation,
        score_thresh=args.score_thresh,
        bg_score_gate=args.bg_score_gate,
        gap_factor=args.model_gap_factor,
        col_gap_factor=args.model_col_gap_factor,
    )
    _unletterbox_boxes(result["chars"], orig_size, scale, pad)

    chars = result["chars"]
    boxes_list = [c["box"] for c in chars]
    col_ids_list = [c["col_id"] for c in chars]

    panel, num_cols, column_breaks, block_breaks = _render_panel(
        image, boxes_list, col_ids_list, chars, result["orientation"], gap_factor, "MODEL (APP) PREDICTION", font
    )
    return panel, num_cols, column_breaks, block_breaks, len(chars)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--annotation", type=Path, default=SAMPLE_ANNOTATION)
    p.add_argument("--orientation", type=str, default="vertical", choices=["vertical", "horizontal"],
                   help="Orientation forced for the GROUND TRUTH panel.")
    p.add_argument("--gap-factor", type=float, default=2.5,
                   help="Block-break distance-heuristic gap factor, applied identically to "
                        "both panels for a fair comparison.")
    p.add_argument("--skip-model", action="store_true",
                   help="Skip loading/running the model -- render the GT panel only.")
    p.add_argument("--ckpt", type=str, default=None,
                   help="SuminaNet checkpoint for the MODEL panel (default: same as the app).")
    p.add_argument("--model-orientation", type=str, default="auto",
                   choices=["auto", "vertical", "horizontal"],
                   help="Orientation passed to run_inference() for the MODEL panel. "
                        "'auto' matches what the app actually sends (the frontend never "
                        "overrides orientation).")
    p.add_argument("--score-thresh", type=float, default=SUMINANET_CER_SCORE_THRESH,
                   help="Min ROI refine score to keep a char (matches app default).")
    p.add_argument("--bg-score-gate", type=float, default=SUMINANET_BG_SCORE_GATE)
    p.add_argument("--model-gap-factor", type=float, default=1.8,
                   help="run_inference()'s internal column gap-filling threshold (vertical).")
    p.add_argument("--model-col-gap-factor", type=float, default=1.5,
                   help="run_inference()'s internal column gap-filling threshold (horizontal).")
    args = p.parse_args()

    ann = json.loads(args.annotation.read_text(encoding="utf-8"))
    img_path = Path(DATA_DIR) / ann["image_path"]
    boxes = ann["boxes"]
    if not boxes:
        raise ValueError(f"{args.annotation} has no boxes to visualize.")

    image = Image.open(img_path).convert("RGB")
    font = _load_font(22)

    gt_panel, gt_num_cols, gt_col_breaks, gt_block_breaks = _run_gt_panel(
        image, boxes, args.orientation, args.gap_factor, font
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = args.annotation.stem

    print(f"Source image: {img_path}")
    print(f"[GT]    chars={len(boxes)}  cols={gt_num_cols}  col-wraps={len(gt_col_breaks)}  "
          f"block-breaks={len(gt_block_breaks)} at {sorted(gt_block_breaks)}")

    if args.skip_model:
        out_path = OUTPUT_DIR / f"{stem}_reading_order_gt.png"
        gt_panel.save(out_path)
        print(f"Saved: {out_path}")
        return

    model_panel, model_num_cols, model_col_breaks, model_block_breaks, n_model_chars = _run_model_panel(
        image, img_path, args, args.gap_factor, font
    )

    gap = 20
    width = gt_panel.width + gap + model_panel.width
    height = max(gt_panel.height, model_panel.height)
    compare = Image.new("RGB", (width, height), (255, 255, 255))
    compare.paste(gt_panel, (0, 0))
    compare.paste(model_panel, (gt_panel.width + gap, 0))

    out_path = OUTPUT_DIR / f"{stem}_reading_order_compare.png"
    compare.save(out_path)

    print(f"[MODEL] chars={n_model_chars}  cols={model_num_cols}  col-wraps={len(model_col_breaks)}  "
          f"block-breaks={len(model_block_breaks)} at {sorted(model_block_breaks)}")
    print(f"Column count difference (model - GT): {model_num_cols - gt_num_cols}")
    print(f"Char count difference (model - GT): {n_model_chars - len(boxes)}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
