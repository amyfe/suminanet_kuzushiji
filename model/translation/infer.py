"""
infer.py — Run the full two-stage Kuzushiji pipeline on a single image.

Outputs
-------
  <out_dir>/transcription.txt  — assembled text in reading order
  <out_dir>/result.json        — structured output: chars, boxes, scores, orientation

Usage
-----
  python infer.py --image path/to/page.jpg
  python infer.py --image page.jpg --suminanet_ckpt checkpoints/suminanet_recognizer/suminanet_best.pt --out out/
  python infer.py --image page.jpg --orientation horizontal --score_thresh 0.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from torchvision.ops import nms

from config import (
    CHECKPOINT_DIR,
    DEVICE,
    IMAGE_SIZE,
    SUMINANET_BG_SCORE_GATE,
    SUMINANET_CER_SCORE_THRESH,
    SUMINANET_NUM_ALTERNATES,
    WEBSITE_CHECKPOINT_DIR
)
from model.suminanet.suminanet_recognizer import SuminaNetRecognizer
from model.suminanet.roi.roi_ordering import infer_reading_orientation_from_boxes
from train_stage2_suminanet import build_suminanet_model, load_vocab
from utils.detection_utils import _extract_all_peaks
from utils.training_helpers.helper_stage2 import _normalize_orientation_label
from utils.text_normalization import unicode_token_to_char

_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


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


def load_suminanet(suminanet_ckpt: str | Path, vocab) -> torch.nn.Module:
    model = build_suminanet_model(vocab, load_stage1_weights=False)
    ckpt = torch.load(suminanet_ckpt, map_location=DEVICE)
    state = ckpt.get("model_state_dict", ckpt)
    state = _align_compile_keys(model.state_dict(), state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    truly_missing    = [k for k in missing   if "_orig_mod" not in k]
    truly_unexpected = [k for k in unexpected if "_orig_mod" not in k]
    if len(truly_missing) > 10:
        print(f"  [ERROR] {len(truly_missing)} missing keys — checkpoint may be wrong model "
              f"architecture. First: {truly_missing[0]}. Proceeding anyway.")
    elif truly_missing:
        print(f"  [warn] {len(truly_missing)} missing keys (first: {truly_missing[0]})")
    if truly_unexpected:
        print(f"  [warn] {len(truly_unexpected)} unexpected keys (first: {truly_unexpected[0]})")
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Image loading — letterboxed to preserve aspect ratio
# ---------------------------------------------------------------------------

def load_image(
    image_path: str | Path,
) -> tuple[torch.Tensor, tuple[int, int], float, tuple[int, int]]:
    """
    Load image and letterbox to IMAGE_SIZE.

    Returns
    -------
    tensor   : (1, 3, H, W) normalised tensor
    orig_size: (orig_W, orig_H) in pixels
    scale    : uniform scale factor applied to the original image
    pad      : (pad_x, pad_y) — pixel offset of the image inside the canvas
    """
    img = Image.open(image_path).convert("RGB")
    orig_size = img.size  # PIL: (W, H)
    size = IMAGE_SIZE if isinstance(IMAGE_SIZE, int) else IMAGE_SIZE[0]
    orig_w, orig_h = orig_size
    scale  = size / max(orig_w, orig_h)
    new_w  = round(orig_w * scale)
    new_h  = round(orig_h * scale)
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)
    pad_x  = (size - new_w) // 2
    pad_y  = (size - new_h) // 2
    canvas = Image.new("RGB", (size, size), (128, 128, 128))
    canvas.paste(img_resized, (pad_x, pad_y))
    tensor = _TRANSFORM(canvas).unsqueeze(0)
    return tensor, orig_size, scale, (pad_x, pad_y)


def _unletterbox_boxes(
    chars: list[dict],
    orig_size: tuple[int, int],
    scale: float,
    pad: tuple[int, int],
) -> None:
    """Convert boxes in-place from letterboxed model coords to original image coords."""
    orig_w, orig_h = orig_size
    pad_x, pad_y   = pad
    for c in chars:
        x1, y1, x2, y2 = c["box"]
        x1 = max(0.0, min((x1 - pad_x) / scale, float(orig_w)))
        y1 = max(0.0, min((y1 - pad_y) / scale, float(orig_h)))
        x2 = max(0.0, min((x2 - pad_x) / scale, float(orig_w)))
        y2 = max(0.0, min((y2 - pad_y) / scale, float(orig_h)))
        c["box"] = [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]


# ---------------------------------------------------------------------------
# Column-gap filling helpers (no SAM2 — pure geometry)
# ---------------------------------------------------------------------------

def _cluster_columns(
    boxes: torch.Tensor,          # (N, 4) xyxy
    col_gap_factor: float = 1.5,
) -> List[List[int]]:
    """Group proposal indices into vertical text columns by x-centre proximity."""
    n = boxes.size(0)
    if n == 0:
        return []
    cx     = ((boxes[:, 0] + boxes[:, 2]) * 0.5).numpy()
    widths = (boxes[:, 2] - boxes[:, 0]).numpy()
    thresh = max(float(np.median(widths)) * col_gap_factor, 5.0)
    order  = np.argsort(cx)
    scx    = cx[order]
    cols: List[List[int]] = []
    cur: List[int] = [int(order[0])]
    for k in range(1, len(order)):
        if scx[k] - scx[k - 1] > thresh:
            cols.append(cur)
            cur = []
        cur.append(int(order[k]))
    cols.append(cur)
    return cols


def _gap_proposals(
    col_boxes: torch.Tensor,           # (M, 4) xyxy — one column
    gap_factor: float = 1.8,
    max_fill: int = 8,
    gap_score: float = 0.40,
    raw_boxes: Optional[List[List[float]]] = None,   # all Stage 1 raw peaks
    raw_scores: Optional[List[float]] = None,
    lower_thresh: float = 0.35,
) -> Tuple[List[List[float]], List[float]]:
    """Return (boxes, scores) for missing characters inside one column.

    If raw_boxes/raw_scores are provided, searches actual subthreshold Stage 1 peaks
    in each gap region (score >= lower_thresh) before falling back to geometric synthesis.
    """
    m = col_boxes.size(0)
    if m < 2:
        return [], []
    cy     = ((col_boxes[:, 1] + col_boxes[:, 3]) * 0.5).numpy()
    h_vals = (col_boxes[:, 3] - col_boxes[:, 1]).numpy()
    w_vals = (col_boxes[:, 2] - col_boxes[:, 0]).numpy()
    cx_vals= ((col_boxes[:, 0] + col_boxes[:, 2]) * 0.5).numpy()
    h_med  = float(np.median(h_vals))
    w_med  = float(np.median(w_vals))
    cx_med = float(np.median(cx_vals))
    if h_med < 2.0:
        return [], []

    # Pre-filter raw peak candidates to those above lower_thresh and in column x-range
    use_raw = (raw_boxes is not None and raw_scores is not None and len(raw_boxes) > 0)
    cand_cy = 0
    cand_sc = []
    cand_boxes = []
    if use_raw:
        cx_lo = float(col_boxes[:, 0].min().item()) - w_med * 0.5
        cx_hi = float(col_boxes[:, 2].max().item()) + w_med * 0.5
        raw_arr = np.asarray(raw_boxes, dtype=np.float32)
        raw_sc  = np.asarray(raw_scores, dtype=np.float32)
        raw_cx  = 0.5 * (raw_arr[:, 0] + raw_arr[:, 2])
        raw_cy  = 0.5 * (raw_arr[:, 1] + raw_arr[:, 3])
        col_mask = (raw_sc >= lower_thresh) & (raw_cx >= cx_lo) & (raw_cx <= cx_hi)
        has_cands = col_mask.any()
        if has_cands:
            cand_boxes = raw_arr[col_mask]
            cand_sc    = raw_sc[col_mask]
            cand_cy    = raw_cy[col_mask]
    else:
        has_cands = False

    order     = np.argsort(cy)
    sorted_cy = cy[order]
    boxes_out:  List[List[float]] = []
    scores_out: List[float]       = []

    for k in range(len(order) - 1):
        gap = sorted_cy[k + 1] - sorted_cy[k]
        if gap <= gap_factor * h_med:
            continue
        n_miss = min(int(round(gap / h_med)) - 1, max_fill)
        if n_miss <= 0:
            continue
        y_lo = sorted_cy[k]     + h_med * 0.3
        y_hi = sorted_cy[k + 1] - h_med * 0.3

        if has_cands:
            # Prefer actual heatmap peaks over synthesis
            in_gap = (cand_cy >= y_lo) & (cand_cy <= y_hi)
            if in_gap.any():
                top_order = np.argsort(cand_sc[in_gap])[::-1][:n_miss]
                for ti in top_order:
                    b = cand_boxes[in_gap][ti].tolist()
                    boxes_out.append(b)
                    scores_out.append(float(cand_sc[in_gap][ti]))
                continue  # gap handled by real peaks; skip synthesis

        # Fallback: geometric synthesis
        for j in range(1, n_miss + 1):
            fill_cy = sorted_cy[k] + j * (gap / (n_miss + 1))
            x1 = cx_med - w_med * 0.5
            y1 = fill_cy - h_med * 0.5
            x2 = cx_med + w_med * 0.5
            y2 = fill_cy + h_med * 0.5
            boxes_out.append([x1, y1, x2, y2])
            scores_out.append(gap_score)

    return boxes_out, scores_out


def _augment_with_column_gaps(
    boxes: torch.Tensor,                           # (N, 4) valid coarse boxes (xyxy)
    scores: torch.Tensor,                          # (N,)  valid coarse scores
    gap_factor: float = 1.8,
    col_gap_factor: float = 1.5,
    gap_score: float = 0.40,
    raw_boxes: Optional[List[List[float]]] = None, # Stage 1 subthreshold peak candidates
    raw_scores: Optional[List[float]] = None,
    lower_thresh: float = 0.35,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cluster proposals into columns, find y-gaps, return augmented (boxes, scores).

    When raw_boxes/raw_scores are provided (actual Stage 1 subthreshold peaks), each gap
    is filled with real heatmap candidates instead of geometrically synthesized boxes.
    Falls back to synthesis only when no real candidate exists for a given gap.
    """
    if boxes.size(0) < 4:
        return boxes, scores
    columns = _cluster_columns(boxes, col_gap_factor)
    extra_boxes: List[List[float]] = []
    extra_scores: List[float] = []
    for col_idx in columns:
        if len(col_idx) < 2:
            continue
        new_b, new_s = _gap_proposals(
            boxes[col_idx], gap_factor, gap_score=gap_score,
            raw_boxes=raw_boxes, raw_scores=raw_scores, lower_thresh=lower_thresh,
        )
        extra_boxes.extend(new_b)
        extra_scores.extend(new_s)
    if not extra_boxes:
        return boxes, scores
    extra_t = torch.tensor(extra_boxes,  dtype=torch.float32)
    extra_s = torch.tensor(extra_scores, dtype=torch.float32)
    return torch.cat([boxes, extra_t], dim=0), torch.cat([scores, extra_s], dim=0)


# ---------------------------------------------------------------------------
# Orientation inference — Stage 1 only (fast, no ROI/classifier)
# ---------------------------------------------------------------------------

def _infer_orientation_from_s1(
    model: SuminaNetRecognizer,
    image_tensor: torch.Tensor,
) -> tuple[str, list, list]:
    """
    Infer reading orientation using Stage 1 peaks only (backbone + detector).

    Much faster than a full two-stage pass. Returns orientation string plus
    the raw Stage 1 (boxes, scores) so the caller can reuse them for gap-filling
    without running the backbone a second time.
    """
    with torch.no_grad():
        s1_feats = model.backbone(image_tensor)
        s1_out   = model.detector(s1_feats)
        hm_probs = torch.sigmoid(s1_out["heatmap"])
        _, _, Hf, Wf = hm_probs.shape   # heatmap feature-map size, not backbone output size

    raw_boxes, raw_scores = _extract_all_peaks(
        hm_probs, s1_out["bbox"], (Hf, Wf), IMAGE_SIZE,
        top_k=500, min_box_size=model.det_min_box_size,
    )

    if raw_boxes:
        boxes_t = torch.tensor(raw_boxes, dtype=torch.float32)
        orientation = _normalize_orientation_label(
            infer_reading_orientation_from_boxes(boxes_t)
        )
    else:
        orientation = "vertical"  # safe default when no proposals found

    return orientation, raw_boxes, raw_scores


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def run_inference(
    model: SuminaNetRecognizer,
    image_tensor: torch.Tensor,
    vocab,
    orientation: str = "auto",
    score_thresh: float = SUMINANET_CER_SCORE_THRESH,
    bg_score_gate: float = SUMINANET_BG_SCORE_GATE,
    fill_gaps: bool = True,
    gap_factor: float = 1.8,
    col_gap_factor: float = 1.5,
) -> dict:
    """
    Run the full SuminaNet forward pass on one image.

    Returns
    -------
    dict with keys:
      transcription : str — full text in reading order
      chars         : list of {char, box:[x1,y1,x2,y2], score, col_id}  (letterboxed IMAGE_SIZE coords)
                      col_id: column/row membership from ROIReadingOrder, reindexed into
                      reading order -- lets a consumer tell a mere column wrap apart from
                      a genuine text-block boundary (see translation.py's block-break logic)
      orientation   : str
      n_chars       : int
    """
    image_tensor = image_tensor.to(DEVICE)

    # Stage 1: backbone + detector (always run; needed for orientation and gap-filling)
    raw_s1_boxes: Optional[List[List[float]]] = None
    raw_s1_scores: Optional[List[float]] = None

    if orientation == "auto" or fill_gaps:
        orientation_s1, raw_s1_boxes, raw_s1_scores = _infer_orientation_from_s1(
            model, image_tensor
        )
        if orientation == "auto":
            orientation = orientation_s1
        if not fill_gaps:
            raw_s1_boxes  = None  # discard peaks if gap-filling is disabled
            raw_s1_scores = None

    # Full two-stage forward pass
    with torch.no_grad():
        outputs = model(image_tensor, orientations=[orientation])

    # Gap-filling: find missing characters in column gaps, then re-run stage 2
    if fill_gaps:
        roi_boxes  = outputs["roi_boxes"][0].cpu()       # (T, 4) — padded
        roi_scores = outputs["roi_scores"][0].cpu()      # (T,)
        roi_mask   = outputs["roi_mask"][0].bool().cpu() # (T,)
        valid_boxes  = roi_boxes[roi_mask]
        valid_scores = roi_scores[roi_mask]
        aug_boxes, aug_scores = _augment_with_column_gaps(
            valid_boxes, valid_scores,
            gap_factor=gap_factor,
            col_gap_factor=col_gap_factor,
            raw_boxes=raw_s1_boxes, raw_scores=raw_s1_scores,
        )
        if aug_boxes.size(0) > valid_boxes.size(0):      # gaps were found
            # Gap-filled boxes (real sub-threshold peaks or synthesized geometric
            # placeholders) are not deduplicated by _augment_with_column_gaps, unlike
            # every other proposal set the model sees (extract_coarse_proposals always
            # applies NMS). Re-apply it here so the second pass sees a proposal density
            # consistent with what the refinement/classification heads were trained on.
            keep = nms(aug_boxes, aug_scores, iou_threshold=model.det_nms_iou)
            aug_boxes  = aug_boxes[keep]
            aug_scores = aug_scores[keep]

            # Re-derive orientation from the complete augmented box set before second pass.
            # The initial orientation was inferred from the incomplete first-pass detections;
            # the full set may support a more accurate determination.
            new_orientation = _normalize_orientation_label(
                infer_reading_orientation_from_boxes(aug_boxes)
            )
            if new_orientation != orientation:
                orientation = new_orientation
            with torch.no_grad():
                outputs = model(
                    image_tensor,
                    orientations=[orientation],
                    coarse_boxes_list=[aug_boxes.to(DEVICE)],
                    coarse_scores_list=[aug_scores.to(DEVICE)],
                )

    char_logits   = outputs["char_logits"]    # (1, T, V) — sorted order
    ordered_mask  = outputs["ordered_mask"]   # (1, T)    — sorted order
    ordered_boxes = outputs["ordered_boxes"]  # (1, T, 4) — sorted order, refined coords
    sort_indices  = outputs["sort_indices"]   # (1, T)    — sorted -> original index
    refine_scores = outputs["refine_scores"]  # (1, T)    — unordered quality scores

    b = 0
    mask_b = ordered_mask[b]  # (T,)
    si_b   = sort_indices[b]  # (T,)

    # Apply refine-score threshold
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

    # BG score gate: relabel high-quality proposals predicted as BG to the best
    # non-BG class. A high refine_score means the ROI is a well-localized real
    # character, so a BG prediction there is more likely a misclassification
    # (e.g. the し→BG confusion) than a correct background suppression.
    if model.bg_id is not None and bg_score_gate > 0.0:
        orig_pos    = si_b[valid_pos]
        roi_quality = torch.sigmoid(refine_scores[b].index_select(0, orig_pos))
        for i, p_id in enumerate(pred_ids):
            if p_id == model.bg_id and float(roi_quality[i]) > bg_score_gate:
                lgt = logits_b[i].clone()
                lgt[model.bg_id] = float("-inf")
                pred_ids[i] = int(lgt.argmax().item())

    # Get boxes and quality scores in sorted order
    boxes_sorted  = ordered_boxes[b, valid_pos].cpu()         # (N, 4)
    orig_pos_for_score = si_b[valid_pos]
    scores_sorted = torch.sigmoid(
        refine_scores[b].index_select(0, orig_pos_for_score)
    ).cpu()

    # Isolation/furigana masks from ROIReadingOrder (same signal training routes
    # strong BG-loss through): isolated-and-not-furigana groups are likely
    # illustration/FP regions; furigana groups are real small kana glosses and
    # must never be dropped just for being small. Sorted order, so index with
    # the same valid_pos used for boxes_sorted/pred_ids — no si_b remap needed.
    isolation_valid = outputs["isolation_mask"][b, valid_pos]
    furigana_valid  = outputs["furigana_mask"][b, valid_pos]
    col_ids_valid   = outputs["col_ids"][b, valid_pos]

    # Build per-char records, skipping BG tokens and isolated non-furigana boxes
    chars_out: list[dict] = []
    for i, p_id in enumerate(pred_ids):
        if model.bg_id is not None and p_id == model.bg_id:
            continue
        if bool(isolation_valid[i]) and not bool(furigana_valid[i]):
            continue  # likely illustration/FP region, not a real character
        decoded = vocab.decode([p_id], remove_special=True)
        if not decoded:
            continue
        ch = unicode_token_to_char(decoded[0])
        x1, y1, x2, y2 = boxes_sorted[i].tolist()

        # Candidate list for this ROI: the chosen pick plus up to
        # SUMINANET_NUM_ALTERNATES runner-up classes, so the frontend can show
        # "other candidates" for low-confidence chars without ever losing the
        # model's own top pick from the list (it's always alternates[0]).
        probs = logits_b[i].softmax(dim=-1)
        topk_probs, topk_ids = probs.topk(SUMINANET_NUM_ALTERNATES + 2)
        alternates: list[dict] = [{
            "char":   ch,
            "prob":   round(float(probs[p_id]), 3),
            "chosen": True,
        }]
        for alt_prob, alt_id in zip(topk_probs.tolist(), topk_ids.tolist()):
            if len(alternates) >= SUMINANET_NUM_ALTERNATES + 1:
                break
            if alt_id == p_id or (model.bg_id is not None and alt_id == model.bg_id):
                continue
            alt_decoded = vocab.decode([alt_id], remove_special=True)
            if not alt_decoded:
                continue
            alternates.append({
                "char":   unicode_token_to_char(alt_decoded[0]),
                "prob":   round(alt_prob, 3),
                "chosen": False,
            })

        chars_out.append({
            "char":       ch,
            "box":        [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "score":      round(float(scores_sorted[i]), 4),
            "alternates": alternates,
            "col_id":     int(col_ids_valid[i]),
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
    parser.add_argument("--suminanet_ckpt", default=str(WEBSITE_CHECKPOINT_DIR),
                        help=f"SuminaNet checkpoint (default: {WEBSITE_CHECKPOINT_DIR}).")
    parser.add_argument("--out", default="output",
                        help="Output directory (default: output/).")
    parser.add_argument("--orientation", default="auto",
                        choices=["auto", "vertical", "horizontal", "other"],
                        help="Reading orientation. 'auto' infers from detected boxes (default).")
    parser.add_argument("--score_thresh", type=float,
                        default=SUMINANET_CER_SCORE_THRESH,
                        help="Min ROI refine score to include a char (0 = keep all).")
    parser.add_argument("--bg_score_gate", type=float, default=SUMINANET_BG_SCORE_GATE,
                        help="Quality gate for BG relabeling: proposals above this score "
                             "predicted as BG are relabeled to the best non-BG class "
                             "(0 = disabled).")
    parser.add_argument("--gap_factor", type=float, default=1.8,
                        help="Vertical gap threshold as multiple of median char height "
                             "to trigger gap-filling (default 1.8). Increase for sparser text.")
    parser.add_argument("--col_gap_factor", type=float, default=1.5,
                        help="Horizontal gap threshold as multiple of median char width "
                             "to split columns (default 1.5). Decrease for tightly-spaced columns.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading vocab...")
    vocab = load_vocab()

    print(f"Loading SuminaNet: {args.suminanet_ckpt}")
    model = load_suminanet(args.suminanet_ckpt, vocab)
    print("Model ready.")

    print(f"Loading image: {args.image}")
    image_tensor, orig_size, scale, pad = load_image(args.image)

    print("Running inference...")
    result = run_inference(
        model, image_tensor, vocab,
        orientation=args.orientation,
        score_thresh=args.score_thresh,
        bg_score_gate=args.bg_score_gate,
        gap_factor=args.gap_factor,
        col_gap_factor=args.col_gap_factor,
    )

    # Map boxes from letterboxed model coords → original image coords (with clamping)
    _unletterbox_boxes(result["chars"], orig_size, scale, pad)

    # --- Save transcription.txt ---
    txt_path = out_dir / "transcription.txt"
    txt_path.write_text(result["transcription"], encoding="utf-8")

    # --- Save result.json ---
    model_size = IMAGE_SIZE if isinstance(IMAGE_SIZE, int) else IMAGE_SIZE[0]
    json_out = {
        "image":               str(args.image),
        "orientation":         result["orientation"],
        "n_chars":             result["n_chars"],
        "transcription":       result["transcription"],
        "chars":               result["chars"],
        "image_size_original": list(orig_size),
        "image_size_model":    model_size,
        "suminanet_ckpt":        str(args.suminanet_ckpt),
        "score_thresh":        args.score_thresh,
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
