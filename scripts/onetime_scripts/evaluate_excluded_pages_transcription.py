"""
evaluate_excluded_pages_transcription.py
------------------------------------------
Runs the real SuminaNet inference pipeline (detector + SuminaNet +
ROIReadingOrder, same code path as model.translation.infer.run_inference /
the app's /api/transcribe) on the 20 pages held out via config.EXCLUDE_PAGES,
and compares each page's predicted transcription -- which does go through
ROIReadingOrder, same as live inference -- against its ground-truth
transcription. GT text is taken directly from the annotation file's own
label order (see gt_text_for_page) rather than re-sorted geometrically;
annotators record characters in reading order already, and re-deriving it
via ROIReadingOrder was found to corrupt that order on some pages instead
of confirming it (see that function's docstring). This is the same GT
construction export_excluded_page_transcriptions.py now uses.

These pages were genuinely never trained/validated on (see
add_excluded_books.py), so this is the closest thing to a true test-set
transcription evaluation.

For each page, prints predicted vs. GT text and per-page CER, and writes a
combined JSON report. Drops exact (box, label) duplicate GT annotations
before assembling GT text -- a known data-prep artifact affecting ~14% of
assets/data/annotations/*.json (see dedupe_gt in
scripts/visualize_qualitative_examples.py); none of these 20 pages are
currently affected, but this keeps behavior consistent if that changes.

Usage:
    python scripts/onetime_scripts/evaluate_excluded_pages_transcription.py
    python scripts/onetime_scripts/evaluate_excluded_pages_transcription.py --ckpt checkpoints/E_final_countdown/suminanet_recognizer/suminanet_best.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from config import DATA_DIR, EXCLUDE_PAGES, SUMINANET_BG_SCORE_GATE, SUMINANET_CER_SCORE_THRESH, WEBSITE_CHECKPOINT_DIR
from utils.text_normalization import unicode_token_to_char

OUTPUT_DIR = ROOT / "results" / "excluded_pages_eval"


def dedupe_gt(gt_boxes: list, gt_chars: list[str]) -> tuple[list, list[str]]:
    """Drop exact (box, label) duplicate GT pairs -- see module docstring."""
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


def gt_text_for_page(stem: str) -> tuple[str, str]:
    """Returns (gt_text, image_path_relative_to_DATA_DIR).

    Uses the raw annotation array order directly rather than re-deriving
    reading order via ROIReadingOrder.sort_single(). Annotators recorded
    characters in reading order at labeling time, and empirically that raw
    order is already correct (verified against 3 independently-checked
    pages). Re-sorting geometrically was found to *corrupt* this
    already-correct order on some pages instead: sort_single's column/row
    clustering is a statistical heuristic (see ROIReadingOrder's own
    docstrings on the ~38% ordering-violation rate on dense pages) tuned for
    noisy model *predictions*, which have no inherent order of their own --
    ground truth doesn't have that problem and shouldn't pay that risk.
    Confirmed failures from the geometric approach: a short horizontal
    caption (2-character sub-column groups misordered) and at least one
    dense vertical page (451 boxes, plausible-looking group count but still
    scrambled). See THESIS_SUPERVISOR_NOTES.tex and the memory note on this
    session's data-quality findings for the full investigation.
    """
    ann_path = DATA_DIR / "annotations" / f"{stem}.json"
    ann = json.loads(ann_path.read_text(encoding="utf-8"))
    raw_boxes = ann["boxes"]
    raw_chars = [unicode_token_to_char(t) for t in ann["labels"]]
    _, gt_chars = dedupe_gt(raw_boxes, raw_chars)
    gt_text = "".join(gt_chars)
    return gt_text, ann["image_path"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, default=Path(WEBSITE_CHECKPOINT_DIR))
    p.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    args = p.parse_args()

    from model.translation.infer import _unletterbox_boxes, load_image, load_suminanet, run_inference
    from train_stage2_suminanet import load_vocab

    print(f"Loading vocab + SuminaNet checkpoint: {args.ckpt}")
    vocab = load_vocab()
    model = load_suminanet(args.ckpt, vocab)

    stems = sorted(
        Path(f).stem
        for files in EXCLUDE_PAGES.values()
        for f in files
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    cers = []

    for stem in stems:
        ann_path = DATA_DIR / "annotations" / f"{stem}.json"
        if not ann_path.exists():
            print(f"  [warn] no annotation JSON for {stem}, skipping")
            continue

        gt_text, image_rel_path = gt_text_for_page(stem)
        img_path = DATA_DIR / image_rel_path

        image_tensor, orig_size, scale, pad = load_image(img_path)
        result = run_inference(
            model, image_tensor, vocab,
            score_thresh=SUMINANET_CER_SCORE_THRESH, bg_score_gate=SUMINANET_BG_SCORE_GATE,
        )
        _unletterbox_boxes(result["chars"], orig_size, scale, pad)
        pred_text = result["transcription"]

        cer = sequence_cer(pred_text, gt_text)
        cers.append(cer)
        records.append({
            "stem": stem,
            "gt_text": gt_text,
            "pred_text": pred_text,
            "cer": cer,
            "n_gt_chars": len(gt_text),
            "n_pred_chars": len(pred_text),
        })
        print(f"\n[{stem}] CER={cer:.4f}  gt={len(gt_text)} chars  pred={len(pred_text)} chars")
        print(f"  GT:   {gt_text}")
        print(f"  PRED: {pred_text}")

    mean_cer = sum(cers) / max(1, len(cers))
    print("\n" + "=" * 70)
    print(f"Mean CER over {len(cers)} excluded pages: {mean_cer:.4f}")

    out_json = args.out_dir / "excluded_pages_eval.json"
    out_json.write_text(
        json.dumps({"checkpoint": str(args.ckpt), "mean_cer": mean_cer, "pages": records},
                    ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
