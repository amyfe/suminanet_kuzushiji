"""
batch_infer.py — Run infer.py over a directory of images.

For each image found, writes:
  <out_dir>/<stem>/transcription.txt
  <out_dir>/<stem>/result.json

Optionally appends all transcriptions to a single combined document.

Usage
-----
  python batch_infer.py --images_dir assets/data/brsk001/images/ --out output/brsk001/
  python batch_infer.py --images_dir assets/data/ --out output/ --combined combined.txt
  python batch_infer.py --images_dir assets/data/ --out output/ --pattern "*.jpg" --workers 1
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import torch

from config import CHECKPOINT_DIR, KURONET_CER_SCORE_THRESH
from infer import load_image, load_kuronet, run_inference
from train_stage2_kuronet import load_vocab

_DEFAULT_KURONET_CKPT = CHECKPOINT_DIR / "stage2" / "kuronet_best.pt"
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def _find_images(images_dir: Path, pattern: str) -> list[Path]:
    files = sorted(images_dir.glob(pattern))
    return [f for f in files if f.suffix.lower() in _IMAGE_EXTENSIONS]


def process_one(
    image_path: Path,
    out_dir: Path,
    model,
    vocab,
    orientation: str,
    score_thresh: float,
    bg_score_gate: float,
) -> dict | None:
    page_dir = out_dir / image_path.stem
    page_dir.mkdir(parents=True, exist_ok=True)

    try:
        image_tensor, orig_size = load_image(image_path)
        result = run_inference(
            model, image_tensor, vocab,
            orientation=orientation,
            score_thresh=score_thresh,
            bg_score_gate=bg_score_gate,
        )

        # Scale boxes back to original coordinates
        orig_w, orig_h = orig_size
        from config import IMAGE_SIZE
        model_size = IMAGE_SIZE if isinstance(IMAGE_SIZE, int) else IMAGE_SIZE[0]
        sx, sy = orig_w / model_size, orig_h / model_size
        for c in result["chars"]:
            x1, y1, x2, y2 = c["box"]
            c["box"] = [round(x1*sx, 1), round(y1*sy, 1), round(x2*sx, 1), round(y2*sy, 1)]

        (page_dir / "transcription.txt").write_text(
            result["transcription"], encoding="utf-8"
        )

        json_out = {
            "image":               str(image_path),
            "orientation":         result["orientation"],
            "n_chars":             result["n_chars"],
            "transcription":       result["transcription"],
            "chars":               result["chars"],
            "image_size_original": list(orig_size),
        }
        (page_dir / "result.json").write_text(
            json.dumps(json_out, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return result

    except Exception:
        print(f"  [error] {image_path.name}:")
        traceback.print_exc()
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-transcribe Kuzushiji page images."
    )
    parser.add_argument("--images_dir", required=True,
                        help="Directory containing page images.")
    parser.add_argument("--out", required=True,
                        help="Output directory (one sub-folder per image).")
    parser.add_argument("--kuronet_ckpt", default=str(_DEFAULT_KURONET_CKPT),
                        help="KuroNet checkpoint path.")
    parser.add_argument("--combined", default=None,
                        help="If set, write all transcriptions to this single file.")
    parser.add_argument("--pattern", default="**/*",
                        help="Glob pattern for image files (default: '**/*').")
    parser.add_argument("--orientation", default="auto",
                        choices=["auto", "vertical", "horizontal", "other"],
                        help="Reading orientation (default: auto).")
    parser.add_argument("--score_thresh", type=float,
                        default=KURONET_CER_SCORE_THRESH,
                        help="Min ROI quality score to include a char.")
    parser.add_argument("--bg_score_gate", type=float, default=0.0,
                        help="Score gate for BG suppression (0 = disabled).")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process at most N images (0 = all).")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    out_dir    = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = _find_images(images_dir, args.pattern)
    if not images:
        print(f"No images found in {images_dir} with pattern '{args.pattern}'")
        return

    if args.limit > 0:
        images = images[:args.limit]

    print(f"Found {len(images)} images.")
    print("Loading vocab and model...")
    vocab = load_vocab()
    model = load_kuronet(args.kuronet_ckpt, vocab)
    print("Model ready.\n")

    combined_lines: list[str] = []
    n_ok = n_err = 0

    for i, img_path in enumerate(images):
        print(f"[{i+1}/{len(images)}] {img_path.name} ...", end=" ", flush=True)
        result = process_one(
            img_path, out_dir, model, vocab,
            args.orientation, args.score_thresh, args.bg_score_gate,
        )
        if result is not None:
            n_ok += 1
            preview = result["transcription"][:40]
            print(f"{result['n_chars']} chars  '{preview}'")
            if args.combined is not None:
                combined_lines.append(f"# {img_path.name}\n{result['transcription']}\n")
        else:
            n_err += 1
            print("FAILED")

    if args.combined and combined_lines:
        combined_path = Path(args.combined)
        combined_path.write_text("\n".join(combined_lines), encoding="utf-8")
        print(f"\nCombined transcription saved: {combined_path}")

    print(f"\nDone: {n_ok} OK, {n_err} errors.")


if __name__ == "__main__":
    main()
