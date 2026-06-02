"""
infer.py — Run the full two-stage Kuzushiji pipeline on a single image.

Outputs
-------
  <out_dir>/transcription.txt  — assembled text in reading order
  <out_dir>/result.json        — structured output: chars, boxes, scores, orientation

Usage
-----
  python infer.py --image path/to/page.jpg
  python infer.py --image page.jpg --kuronet_ckpt checkpoints/stage2/kuronet_best.pt --out out/
  python infer.py --image page.jpg --orientation horizontal --score_thresh 0.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch
import torchvision.transforms as T
from PIL import Image

from config import (
    CHECKPOINT_DIR,
    DEVICE,
    IMAGE_SIZE,
    KURONET_CER_SCORE_THRESH,
)
from model.kuronet.kuronet_recognizer import KuroNetRecognizer
from model.kuronet.roi.roi_ordering import infer_reading_orientation_from_boxes
from train_stage2_kuronet import build_kuronet_model, load_vocab
from utils.training_helpers.helper_stage2 import _normalize_orientation_label
from utils.text_normalization import unicode_token_to_char

_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_DEFAULT_KURONET_CKPT = CHECKPOINT_DIR / "stage2" / "kuronet_best.pt"


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _align_compile_keys(model_state: dict, ckpt_state: dict) -> dict:
    """
    Handle torch.compile key mismatch between model and checkpoint.

    torch.compile wraps submodules and renames their parameters with an
    '_orig_mod.' infix (e.g. 'encoder.conv_stem' → 'encoder._orig_mod.conv_stem').
    Checkpoints may be saved from compiled or uncompiled models; align them.
    """
    model_keys  = set(model_state.keys())
    ckpt_keys   = set(ckpt_state.keys())
    if model_keys == ckpt_keys:
        return ckpt_state  # already aligned

    # Detect direction: does ckpt have _orig_mod but model doesn't?
    ckpt_has_orig  = any("_orig_mod" in k for k in ckpt_keys)
    model_has_orig = any("_orig_mod" in k for k in model_keys)

    out = {}
    for k, v in ckpt_state.items():
        if ckpt_has_orig and not model_has_orig:
            new_k = k.replace("._orig_mod.", ".").replace("_orig_mod.", "")
        elif model_has_orig and not ckpt_has_orig:
            new_k = k
            parts = k.split(".")
            for i in range(1, len(parts)):
                candidate = ".".join(parts[:i]) + "._orig_mod." + ".".join(parts[i:])
                if candidate in model_keys:
                    new_k = candidate
                    break
        else:
            new_k = k
        out[new_k] = v
    return out


def load_kuronet(kuronet_ckpt: str | Path, vocab) -> torch.nn.Module:
    model = build_kuronet_model(vocab)
    ckpt = torch.load(kuronet_ckpt, map_location=DEVICE)
    state = ckpt.get("model_state_dict", ckpt)
    state = _align_compile_keys(model.state_dict(), state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    truly_missing   = [k for k in missing   if "_orig_mod" not in k]
    truly_unexpected = [k for k in unexpected if "_orig_mod" not in k]
    if truly_missing:
        print(f"  [warn] {len(truly_missing)} missing keys (first: {truly_missing[0]})")
    if truly_unexpected:
        print(f"  [warn] {len(truly_unexpected)} unexpected keys (first: {truly_unexpected[0]})")
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image(image_path: str | Path) -> tuple[torch.Tensor, tuple[int, int]]:
    """Load and resize image to IMAGE_SIZE. Returns (1,3,H,W) tensor and (orig_W, orig_H)."""
    img = Image.open(image_path).convert("RGB")
    orig_size = img.size  # PIL: (W, H)
    size = IMAGE_SIZE if isinstance(IMAGE_SIZE, int) else IMAGE_SIZE[0]
    img = img.resize((size, size), Image.LANCZOS)
    tensor = _TRANSFORM(img).unsqueeze(0)
    return tensor, orig_size


# ---------------------------------------------------------------------------
# Orientation inference (from detected boxes, no GT required)
# ---------------------------------------------------------------------------

def _infer_orientation(model: KuroNetRecognizer, image_tensor: torch.Tensor) -> str:
    """
    Infer reading orientation from detected proposal boxes.
    Uses the model's roi_boxes output (no private method access needed).
    """
    with torch.no_grad():
        outputs = model(image_tensor, orientations=["vertical"])
    # roi_boxes: (1, T, 4) — all detector proposals, unordered
    roi_boxes = outputs["roi_boxes"][0]          # (T, 4)
    roi_mask  = outputs["roi_mask"][0].bool()    # (T,)
    valid_boxes = roi_boxes[roi_mask]            # (N, 4)
    return _normalize_orientation_label(infer_reading_orientation_from_boxes(valid_boxes))


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def run_inference(
    model: KuroNetRecognizer,
    image_tensor: torch.Tensor,
    vocab,
    orientation: str = "auto",
    score_thresh: float = KURONET_CER_SCORE_THRESH,
    bg_score_gate: float = 0.0,
) -> dict:
    """
    Run the full KuroNet forward pass on one image.

    Returns
    -------
    dict with keys:
      transcription : str — full text in reading order
      chars         : list of {char, box:[x1,y1,x2,y2], score}  (IMAGE_SIZE coords)
      orientation   : str
      n_chars       : int
    """
    image_tensor = image_tensor.to(DEVICE)

    if orientation == "auto":
        orientation = _infer_orientation(model, image_tensor)

    with torch.no_grad():
        outputs = model(image_tensor, orientations=[orientation])

    char_logits   = outputs["char_logits"]    # (1, T, V) — sorted order
    ordered_mask  = outputs["ordered_mask"]   # (1, T)    — sorted order
    ordered_boxes = outputs["ordered_boxes"]  # (1, T, 4) — sorted order, refined coords
    sort_indices  = outputs["sort_indices"]   # (1, T)    — sorted -> original index
    refine_scores = outputs["refine_scores"]  # (1, T)    — unordered quality scores

    b = 0
    mask_b = ordered_mask[b]  # (T,)
    si_b   = sort_indices[b]  # (T,)

    # Apply score threshold: reorder refine_scores into sorted space for filtering
    if score_thresh > 0.0 and mask_b.any():
        valid_si     = si_b[mask_b]
        scores_ord   = torch.sigmoid(refine_scores[b].index_select(0, valid_si))
        keep         = scores_ord >= score_thresh
        valid_pos    = mask_b.nonzero(as_tuple=True)[0][keep]
    else:
        valid_pos = mask_b.nonzero(as_tuple=True)[0]

    if valid_pos.numel() == 0:
        return {"transcription": "", "chars": [], "orientation": orientation, "n_chars": 0}

    logits_b = char_logits[b, valid_pos]      # (N, V)
    pred_ids: List[int] = logits_b.argmax(dim=-1).tolist()

    # BG score gate: suppress high-quality proposals predicted as BG
    if model.bg_id is not None and bg_score_gate > 0.0:
        orig_pos   = si_b[valid_pos]
        roi_quality = torch.sigmoid(refine_scores[b].index_select(0, orig_pos))
        suppressed: List[int] = []
        for i, p_id in enumerate(pred_ids):
            if p_id == model.bg_id and float(roi_quality[i]) > bg_score_gate:
                lgt = logits_b[i].clone()
                lgt[model.bg_id] = float("-inf")
                suppressed.append(int(lgt.argmax().item()))
            else:
                suppressed.append(p_id)
        pred_ids = suppressed

    # Get boxes and quality scores in sorted order
    boxes_sorted  = ordered_boxes[b, valid_pos].cpu()         # (N, 4)
    orig_pos_for_score = si_b[valid_pos]
    scores_sorted = torch.sigmoid(
        refine_scores[b].index_select(0, orig_pos_for_score)
    ).cpu()

    # Compute median box area to derive a furigana filter threshold.
    # Furigana are typically ≤25% of the median main-character area; filtering
    # them avoids duplicate outputs and reading-order noise on annotated pages.
    if boxes_sorted.size(0) > 0:
        areas = (boxes_sorted[:, 2] - boxes_sorted[:, 0]) * (boxes_sorted[:, 3] - boxes_sorted[:, 1])
        median_area = float(areas.median())
        min_area = median_area * 0.25
    else:
        min_area = 0.0

    # Build per-char records, skipping BG tokens and furigana-sized boxes
    chars_out: list[dict] = []
    for i, p_id in enumerate(pred_ids):
        if model.bg_id is not None and p_id == model.bg_id:
            continue
        decoded = vocab.decode([p_id], remove_special=True)
        if not decoded:
            continue
        ch = unicode_token_to_char(decoded[0])
        x1, y1, x2, y2 = boxes_sorted[i].tolist()
        if (x2 - x1) * (y2 - y1) < min_area:
            continue  # skip furigana / tiny noise boxes
        chars_out.append({
            "char":  ch,
            "box":   [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "score": round(float(scores_sorted[i]), 4),
        })

    transcription = "".join(c["char"] for c in chars_out)

    return {
        "transcription": transcription,
        "chars":         chars_out,
        "orientation":   orientation,
        "n_chars":       len(chars_out),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe a Kuzushiji page image using the two-stage OCR model."
    )
    parser.add_argument("--image", required=True,
                        help="Path to input image (JPG/PNG).")
    parser.add_argument("--kuronet_ckpt", default=str(_DEFAULT_KURONET_CKPT),
                        help=f"KuroNet checkpoint (default: {_DEFAULT_KURONET_CKPT}).")
    parser.add_argument("--out", default="output",
                        help="Output directory (default: output/).")
    parser.add_argument("--orientation", default="auto",
                        choices=["auto", "vertical", "horizontal", "other"],
                        help="Reading orientation. 'auto' infers from detected boxes (default).")
    parser.add_argument("--score_thresh", type=float,
                        default=KURONET_CER_SCORE_THRESH,
                        help="Min ROI refine score to include a char (0 = keep all).")
    parser.add_argument("--bg_score_gate", type=float, default=0.0,
                        help="Score gate for BG→char suppression (0 = disabled).")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading vocab...")
    vocab = load_vocab()

    print(f"Loading KuroNet: {args.kuronet_ckpt}")
    model = load_kuronet(args.kuronet_ckpt, vocab)
    print("Model ready.")

    print(f"Loading image: {args.image}")
    image_tensor, orig_size = load_image(args.image)

    print("Running inference...")
    result = run_inference(
        model, image_tensor, vocab,
        orientation=args.orientation,
        score_thresh=args.score_thresh,
        bg_score_gate=args.bg_score_gate,
    )

    # Scale boxes from IMAGE_SIZE coords back to original image coordinates
    orig_w, orig_h = orig_size
    model_size = IMAGE_SIZE if isinstance(IMAGE_SIZE, int) else IMAGE_SIZE[0]
    sx = orig_w / model_size
    sy = orig_h / model_size
    for c in result["chars"]:
        x1, y1, x2, y2 = c["box"]
        c["box"] = [round(x1*sx, 1), round(y1*sy, 1), round(x2*sx, 1), round(y2*sy, 1)]

    # --- Save transcription.txt ---
    txt_path = out_dir / "transcription.txt"
    txt_path.write_text(result["transcription"], encoding="utf-8")

    # --- Save result.json ---
    json_out = {
        "image":              str(args.image),
        "orientation":        result["orientation"],
        "n_chars":            result["n_chars"],
        "transcription":      result["transcription"],
        "chars":              result["chars"],
        "image_size_original": list(orig_size),
        "image_size_model":   model_size,
        "kuronet_ckpt":       str(args.kuronet_ckpt),
        "score_thresh":       args.score_thresh,
    }
    json_path = out_dir / "result.json"
    json_path.write_text(
        json.dumps(json_out, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n{'='*60}")
    print(f"Orientation : {result['orientation']}")
    print(f"Chars found : {result['n_chars']}")
    print(f"Preview     : {result['transcription'][:80]}")
    print(f"{'='*60}")
    print(f"Saved: {txt_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
