"""Qualitative page examples for the thesis (pipeline_analysis.txt 7.5.3):
auto-select 2 representative validation pages per category (8 total), render
a correct/incorrect character overlay for each, and persist an English
translation of each page's transcription.

Page selection uses results/checkpoint_compare/<ckpt-name>/per_image_metrics.csv
(built by utils/validation/validate_suminanet.py) to pick, per category:
  clean_success      - highest pipeline_f1 among pages with >= --min-gt chars
  failure_case        - lowest  pipeline_f1 among pages with >= --min-gt chars
  dense_columns        - highest GT character count
  illustration_heavy  - highest SAM2 illustration-mask coverage

For each page: runs the real app inference path (model.translation.infer,
same code as /api/transcribe), IoU-matches predictions against GT boxes to
classify each GT character as correct / misclassified / missed, and renders
green/red/blue/orange boxes on the original page image (green=correct,
red=missed — no prediction at all overlapped this GT box, blue=misclassified
— a box was found here but the predicted character is wrong, orange=a box
was predicted but doesn't correspond to a real character). GT (box, label)
pairs that are exact duplicates in the annotation file — a known data
artifact affecting ~14% of assets/data/annotations/*.json — are dropped
before matching/rendering (see dedupe_gt()).

For every page (regardless of --skip-translation) a stage-2 transcription
example is also saved as JSON: the ground-truth text (assembled in the same
reading order SuminaNet uses) alongside the predicted transcription and their
CER, so the two can be compared directly. Unless --skip-translation is set,
the translation pipeline (real, billed Anthropic/OpenRouter API call) is also
run on the page's transcription and saved as JSON — including the same
gt_text/pred_text pair — there is no reference translation to score against
(see pipeline_analysis.txt), so this only persists output for future use, it
does not compute BLEU/BERT.

Usage:
    python scripts/visualize_qualitative_examples.py
    python scripts/visualize_qualitative_examples.py --skip-translation
    python scripts/visualize_qualitative_examples.py --min-gt 10
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_iou

load_dotenv(ROOT / ".env")

from config import DATA_DIR, SAM2_MASKS_DIR, STAGE2_REFINE_POS_IOU, SUMINANET_BG_SCORE_GATE, SUMINANET_CER_SCORE_THRESH
from model.suminanet.roi.roi_ordering import ROIReadingOrder, infer_reading_orientation_from_boxes
from utils.text_normalization import unicode_token_to_char

DEFAULT_CKPT = ROOT / "checkpoints" / "E_final_countdown" / "suminanet_recognizer" / "suminanet_best.pt"
DEFAULT_PER_IMAGE_CSV = ROOT / "results" / "checkpoint_compare" / "EfficientNet" / "per_image_metrics.csv"
OUTPUT_DIR = ROOT / "results" / "qualitative_examples"

_COLORS = {
    "correct":       (0, 170, 0),    # green  — GT correctly recognized
    "missed":        (220, 30, 30),  # red    — no prediction at all overlapped this GT box
    "misclassified": (40, 110, 230), # blue   — a box was found here, but the predicted character is wrong
    "extra":         (255, 150, 0),  # orange — a box was predicted but doesn't correspond to a real character
}
_WIDTHS = {"correct": 6, "missed": 10, "misclassified": 9, "extra": 8}


def _load_font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Page selection
# ---------------------------------------------------------------------------

def select_representative_pages(
    per_image_csv: Path, min_gt: int, n_per_category: int = 2,
) -> dict[str, list[dict]]:
    with open(per_image_csv, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{per_image_csv} has no rows.")
    for r in rows:
        r["gt"] = int(r["gt"])
        r["pipeline_f1"] = float(r["pipeline_f1"])

    eligible = [r for r in rows if r["gt"] >= min_gt] or rows
    selected = {
        "clean_success": sorted(eligible, key=lambda r: -r["pipeline_f1"])[:n_per_category],
        "failure_case":  sorted(eligible, key=lambda r: r["pipeline_f1"])[:n_per_category],
        "dense_columns": sorted(rows, key=lambda r: -r["gt"])[:n_per_category],
    }

    masks_dir = Path(SAM2_MASKS_DIR)
    illus_frac: dict[str, float] = {}
    for r in rows:
        mask_path = masks_dir / f"{r['image_stem']}.npy"
        if mask_path.exists():
            illus_frac[r["image_stem"]] = float(np.load(mask_path).mean())
    if illus_frac:
        illus_stems = sorted(illus_frac, key=illus_frac.get, reverse=True)[:n_per_category]
        selected["illustration_heavy"] = [
            next(r for r in rows if r["image_stem"] == s) for s in illus_stems
        ]
    else:
        print("[warn] No SAM2 masks found — skipping illustration_heavy category.")

    return selected


# ---------------------------------------------------------------------------
# Matching predictions to GT
# ---------------------------------------------------------------------------

def match_predictions_to_gt(
    gt_boxes: list, gt_chars: list[str], pred_chars: list[dict],
) -> tuple[list[str], list[int]]:
    """Returns (gt_status per GT box: "correct"/"misclassified"/"missed",
    extra_pred_indices: predicted boxes with no matching GT)."""
    n_gt, n_pred = len(gt_boxes), len(pred_chars)
    if n_gt == 0:
        return [], list(range(n_pred))

    gt_boxes_t = torch.tensor(gt_boxes, dtype=torch.float32)
    if n_pred == 0:
        return ["missed"] * n_gt, []

    pred_boxes_t = torch.tensor([c["box"] for c in pred_chars], dtype=torch.float32)
    iou = box_iou(gt_boxes_t, pred_boxes_t)  # (n_gt, n_pred)

    gt_status: list[str] = []
    matched_pred: set[int] = set()
    # Greedy: process GT boxes in descending order of their best available IoU,
    # so two GT boxes competing for the same prediction resolve sensibly.
    order = sorted(range(n_gt), key=lambda i: -float(iou[i].max()) if n_pred else 0)
    status_by_gt = ["missed"] * n_gt
    for i in order:
        row = iou[i].clone()
        for j in matched_pred:
            row[j] = -1.0
        best_j = int(row.argmax())
        best_iou = float(row[best_j])
        if best_iou >= STAGE2_REFINE_POS_IOU:
            matched_pred.add(best_j)
            pred_char = pred_chars[best_j]["char"]
            status_by_gt[i] = "correct" if pred_char == gt_chars[i] else "misclassified"
    gt_status = status_by_gt

    extra_idx = [j for j in range(n_pred) if j not in matched_pred]
    return gt_status, extra_idx


def dedupe_gt(gt_boxes: list, gt_chars: list[str]) -> tuple[list, list[str]]:
    """Some annotation files in assets/data/annotations/ contain every (box, label)
    pair exactly twice (a data-prep artifact, not a real double-annotated character:
    e.g. brsk001_*, 200015843_* — 732/5344 files affected as of 2026-08, all/almost-all
    of their boxes duplicated). An un-matchable duplicate GT box always renders as
    "missed" and, at identical coordinates, its thicker outline paints over the
    genuinely-correct twin underneath it — drop exact (box, label) duplicates so
    qualitative examples (and page selection, which uses raw GT counts) aren't
    skewed by this."""
    seen: set[tuple] = set()
    boxes_out: list = []
    chars_out: list[str] = []
    for box, ch in zip(gt_boxes, gt_chars):
        key = (tuple(box), ch)
        if key in seen:
            continue
        seen.add(key)
        boxes_out.append(box)
        chars_out.append(ch)
    return boxes_out, chars_out


def gt_reading_order_indices(gt_boxes: list) -> list[int]:
    """Indices into gt_boxes/gt_chars in the same reading order SuminaNet uses
    for its own predictions (ROIReadingOrder), so rank 0 is the first character
    read, rank 1 the second, etc."""
    if not gt_boxes:
        return []
    orientation = infer_reading_orientation_from_boxes(gt_boxes)
    boxes_t = torch.tensor(gt_boxes, dtype=torch.float32)
    mask = torch.ones((boxes_t.size(0),), dtype=torch.bool)
    _, _, sort_idx, _ = ROIReadingOrder().sort_single(boxes_t, mask, orientation)
    return sort_idx.detach().cpu().tolist()


def gt_text_in_reading_order(gt_boxes: list, gt_chars: list[str], order: list[int]) -> str:
    """Assembles the ground-truth characters into a single string using `order`
    (from gt_reading_order_indices), so gt_text is directly comparable to
    result["transcription"]."""
    return "".join(gt_chars[i] for i in order)


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def sequence_cer(pred_text: str, gt_text: str) -> float:
    return _edit_distance(pred_text, gt_text) / max(1, len(gt_text))


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_overlay(
    image: Image.Image,
    gt_boxes: list, gt_status: list[str],
    pred_chars: list[dict], extra_idx: list[int],
    category: str, stem: str, pipeline_f1: float,
    font: ImageFont.ImageFont,
) -> Image.Image:
    panel = image.copy()
    draw = ImageDraw.Draw(panel)

    for box, status in zip(gt_boxes, gt_status):
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=_COLORS[status], width=_WIDTHS[status])
    for j in extra_idx:
        x1, y1, x2, y2 = pred_chars[j]["box"]
        draw.rectangle([x1, y1, x2, y2], outline=_COLORS["extra"], width=_WIDTHS["extra"])

    counts = {k: gt_status.count(k) for k in ("correct", "misclassified", "missed")}
    header_h = 60
    header = Image.new("RGB", (panel.width, header_h), (20, 20, 20))
    hd = ImageDraw.Draw(header)
    hd.text((10, 6), f"{category}   {stem}   pipeline_f1={pipeline_f1:.3f}",
            fill=(255, 255, 255), font=font)
    legend = (
        f"green=correct({counts['correct']})  "
        f"red=missed({counts['missed']})  "
        f"blue=misclassified({counts['misclassified']})  "
        f"orange=extra_found_not_a_character({len(extra_idx)})"
    )
    hd.text((10, 32), legend, fill=(220, 220, 220), font=font)

    combined = Image.new("RGB", (panel.width, panel.height + header_h))
    combined.paste(header, (0, 0))
    combined.paste(panel, (0, header_h))
    return combined


def render_gt_only(
    image: Image.Image,
    gt_boxes: list, order: list[int],
    category: str, stem: str,
    font: ImageFont.ImageFont, rank_font: ImageFont.ImageFont,
) -> Image.Image:
    """GT boxes only (no predictions), numbered by reading-order rank so the
    order used for gt_text can be visually cross-checked against the actual
    box layout — adjacent ranks alternate color so a reading-order mistake
    (rank N+1 jumping somewhere unexpected) is easy to spot."""
    panel = image.copy()
    draw = ImageDraw.Draw(panel)
    rank_colors = [(0, 140, 230), (230, 0, 160)]

    for rank, i in enumerate(order):
        x1, y1, x2, y2 = gt_boxes[i]
        color = rank_colors[rank % 2]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1 + 2, y1 + 2), str(rank), fill=color, font=rank_font)

    header_h = 60
    header = Image.new("RGB", (panel.width, header_h), (20, 20, 20))
    hd = ImageDraw.Draw(header)
    hd.text((10, 6), f"{category}   {stem}   GT boxes only (deduped) — n={len(gt_boxes)}",
            fill=(255, 255, 255), font=font)
    hd.text((10, 32), "numbers = reading-order rank (color alternates per rank) — should match gt_text order",
            fill=(220, 220, 220), font=font)

    combined = Image.new("RGB", (panel.width, panel.height + header_h))
    combined.paste(header, (0, 0))
    combined.paste(panel, (0, header_h))
    return combined


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--per-image-csv", type=Path, default=DEFAULT_PER_IMAGE_CSV,
                   help="per_image_metrics.csv from validate_suminanet.py, used for page selection")
    p.add_argument("--min-gt", type=int, default=20,
                   help="Minimum GT char count for clean_success/failure_case candidates")
    p.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--skip-translation", action="store_true",
                   help="Skip the translation-persistence step (no API calls)")
    args = p.parse_args()

    from model.translation.infer import load_image, load_suminanet, run_inference, _unletterbox_boxes
    from train_stage2_suminanet import load_vocab

    print(f"Selecting representative pages from {args.per_image_csv} (min_gt={args.min_gt})...")
    selected = select_representative_pages(args.per_image_csv, args.min_gt)
    for category, rows in selected.items():
        for row in rows:
            print(f"  {category:<20} {row['image_stem']:<28} gt={row['gt']:<5} pipeline_f1={row['pipeline_f1']:.3f}")

    print(f"\nLoading vocab + SuminaNet checkpoint: {args.ckpt}")
    vocab = load_vocab()
    model = load_suminanet(args.ckpt, vocab)
    font = _load_font(20)
    rank_font = _load_font(14)

    translator = None
    if not args.skip_translation:
        from model.translation.translation import EdoPeriodTranslationPipeline
        translator = EdoPeriodTranslationPipeline()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for category, rows in selected.items():
        for row in rows:
            stem = row["image_stem"]
            ann_path = DATA_DIR / "annotations" / f"{stem}.json"
            ann = json.loads(ann_path.read_text(encoding="utf-8"))
            img_path = DATA_DIR / ann["image_path"]
            raw_boxes = ann["boxes"]
            raw_chars = [unicode_token_to_char(t) for t in ann["labels"]]
            gt_boxes, gt_chars = dedupe_gt(raw_boxes, raw_chars)
            if len(gt_boxes) != len(raw_boxes):
                print(f"  [warn] {stem}: dropped {len(raw_boxes) - len(gt_boxes)} duplicate GT "
                      f"(box, label) pairs (annotation data artifact)")

            print(f"\n[{category}] {stem}  ({len(gt_boxes)} GT chars)")
            image = Image.open(img_path).convert("RGB")
            if gt_boxes:
                xs = [c for box in gt_boxes for c in (box[0], box[2])]
                ys = [c for box in gt_boxes for c in (box[1], box[3])]
                print(f"  GT box extent: x=[{min(xs):.0f}, {max(xs):.0f}]  "
                      f"y=[{min(ys):.0f}, {max(ys):.0f}]  vs image size {image.size}")
            image_tensor, orig_size, scale, pad = load_image(img_path)
            result = run_inference(
                model, image_tensor, vocab,
                score_thresh=SUMINANET_CER_SCORE_THRESH, bg_score_gate=SUMINANET_BG_SCORE_GATE,
            )
            _unletterbox_boxes(result["chars"], orig_size, scale, pad)
            pred_chars = result["chars"]

            gt_status, extra_idx = match_predictions_to_gt(gt_boxes, gt_chars, pred_chars)
            overlay = render_overlay(
                image, gt_boxes, gt_status, pred_chars, extra_idx,
                category, stem, row["pipeline_f1"], font,
            )
            img_out = args.out_dir / f"{category}_{stem}_overlay.png"
            overlay.save(img_out)
            print(f"  Overlay saved: {img_out}")

            order = gt_reading_order_indices(gt_boxes)
            gt_text = gt_text_in_reading_order(gt_boxes, gt_chars, order)
            pred_text = result["transcription"]
            cer = sequence_cer(pred_text, gt_text)

            preview_n = min(10, len(order))
            preview = ", ".join(f"{r}:{gt_chars[i]}" for r, i in enumerate(order[:preview_n]))
            print(f"  Reading-order preview (rank:char): {preview}{' ...' if len(order) > preview_n else ''}")

            gt_only = render_gt_only(image, gt_boxes, order, category, stem, font, rank_font)
            gt_only_out = args.out_dir / f"{category}_{stem}_gt_only.png"
            gt_only.save(gt_only_out)
            print(f"  GT-only plot saved: {gt_only_out}")

            # Stage 2 (transcription-only) example: original vs. transcribed text,
            # no translation call — saved for every page regardless of --skip-translation.
            transcription_record = {
                "category": category,
                "image_stem": stem,
                "gt_text": gt_text,
                "pred_text": pred_text,
                "cer": cer,
            }
            trans_txt_out = args.out_dir / f"{category}_{stem}_transcription.json"
            trans_txt_out.write_text(
                json.dumps(transcription_record, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"  Transcription comparison saved: {trans_txt_out}  (CER={cer:.4f})")

            if translator is not None:
                print("  Requesting translation (billed API call)...")
                try:
                    translation = translator.translate_text(
                        classical_text=result["transcription"],
                        chars=result["chars"],
                        combined=True,
                        lang="en",
                    )
                except Exception as e:
                    print(f"  [ERROR] Translation failed: {e}")
                    translation = {"error": str(e), "transcription": result["transcription"]}
                translation["gt_text"] = gt_text
                translation["pred_text"] = pred_text
                translation["cer"] = cer
                trans_out = args.out_dir / f"{category}_{stem}_translation.json"
                trans_out.write_text(json.dumps(translation, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"  Translation saved: {trans_out}")

    print(f"\nAll outputs written under: {args.out_dir}")


if __name__ == "__main__":
    main()
