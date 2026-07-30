"""Offline SAM2-based proposal refinement and FN recovery for Kuzushiji training.

Why this exists
---------------
The KuroNet coverage is capped at ~84% because the IoU-matching threshold (0.45)
drops detector proposals whose boxes don't align tightly enough with the GT characters.
SAM2 with bounding-box prompts produces pixel-precise masks that, when converted to
tight bounding boxes, consistently achieve higher IoU with GT — closing that gap without
any detector retraining.

Additionally, three FN-recovery modes are available:

  --fill_gaps      Column-gap filling: Kuzushiji text runs in vertical columns. The
                   detector finds most characters per column but misses small/faint ones.
                   This mode clusters found proposals into columns, locates y-gaps larger
                   than gap_factor × median_char_height, and uses SAM2 point prompts at
                   the predicted missing-character positions to recover them.

  --sub_thresh     Sub-threshold peaks: Extracts heatmap peaks between fn_score_low and
                   the main score_thresh, feeds them as additional SAM2 box prompts.
                   Targets characters the detector "saw" but wasn't confident enough about.

  --recover_frames Frame/cartouche detection: Many manuscripts have text inside
                   rectangular frames (seals, title boxes, signatures) that are suppressed
                   by the spatial density filter. Uses SAM2AutomaticMaskGenerator to find
                   rectangular regions, then re-runs the detector at higher resolution
                   inside suspicious frames with too few proposals.

Pipeline per image (base mode)
--------------------------------
  detector coarse boxes  (N, 4)
        ↓ SAM2 box-prompt (one call per box, batched)
  refined masks  (N, H, W)
        ↓ mask → tight bbox
  refined boxes  (N, 4)   ← same N, tighter bounds

Installation
------------
  pip install git+https://github.com/facebookresearch/sam2.git
  # Download model weights:
  wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt \
       -P checkpoints/sam2/

Usage
-----
  python preprocess_sam2_proposals.py \\
      --sam2_checkpoint checkpoints/sam2/sam2.1_hiera_large.pt \\
      --output_dir     assets/sam2_proposals \\
      [--splits train val] \\
      [--score_thresh 0.50] \\
      [--top_k 500] \\
      [--fill_gaps] [--gap_factor 1.8] [--col_gap_factor 1.5] \\
      [--sub_thresh] [--fn_score_low 0.10] [--fn_top_k 600] \\
      [--recover_frames]
     CUDA_VISIBLE_DEVICES=1 python preprocess_sam2_proposals.py     --splits val --fill_gaps --overwrite     --output_dir assets/sam2_proposals_fn

The script saves one .pt file per image:
  assets/sam2_proposals/<image_stem>.pt
  → dict {"boxes": FloatTensor(N,4), "scores": FloatTensor(N)}
  where boxes are in image-pixel xyxy coordinates (same convention as detector output).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torchvision.ops
import torchvision.transforms as T
from PIL import Image
RESAMPLE_LANCZOS = getattr(Image, "Resampling", Image).LANCZOS
from tqdm import tqdm

from config import (
    BACKBONE_BASE_FEATURES,
    BACKBONE_TYPE,
    CHECKPOINT_DIR,
    DATA_DIR,
    DEVICE,
    DET_NMS_IOU,
    DET_SCORE_THRESH,
    DET_TOP_K,
    IMAGE_SIZE,
    SAM2_CHECKPOINT,
    SAM2_PROPOSALS_DIR,
)
from model.kuronet import DetectorHead, build_backbone
from model.kuronet.detection.proposal_utils import extract_coarse_proposals
from utils.vocab import VocabManager


# ---------------------------------------------------------------------------
# Helpers — detector & SAM2 loading
# ---------------------------------------------------------------------------

def _load_detector(ckpt_path: "str | Path | None" = None) -> tuple:
    """Load frozen Stage-1 detector. Returns (backbone, detector)."""
    if ckpt_path is None:
        ckpt_path = CHECKPOINT_DIR / "stage1_detection" / "detector_best.pt"
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Stage 1 checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    backbone_type = ckpt.get("backbone_type", BACKBONE_TYPE)

    ann_files = sorted((Path(DATA_DIR) / "annotations").glob("*.json"))
    vocab = VocabManager.from_annotations(ann_files)

    backbone = build_backbone(backbone_type, BACKBONE_BASE_FEATURES, pretrained=False).to(DEVICE)
    detector = DetectorHead(
        in_ch=BACKBONE_BASE_FEATURES,
        num_classes=vocab.vocab_size,
        dropout_rate=0.0,
        predict_boxes=True,
        predict_classes=False,
    ).to(DEVICE)

    state_key = "backbone_state_dict" if "backbone_state_dict" in ckpt else "unet_state_dict"
    backbone.load_state_dict(ckpt[state_key])
    detector.load_state_dict(ckpt["detector_state_dict"])

    backbone.eval()
    detector.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    for p in detector.parameters():
        p.requires_grad_(False)

    print(f"Loaded Stage-1 detector from {ckpt_path} (backbone={backbone_type})")
    return backbone, detector


def _mask_to_tight_box(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    """Convert a binary mask (H, W) to a tight xyxy bounding box in pixel coords."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return float(x_min), float(y_min), float(x_max), float(y_max)


def _load_sam2(checkpoint: str) -> tuple:
    """Load SAM2 image predictor and the underlying model. Returns (predictor, sam2_model)."""
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as e:
        raise ImportError(
            "SAM2 is required for this script.\n"
            "Install: pip install git+https://github.com/facebookresearch/sam2.git\n"
            "Download checkpoint:\n"
            "  wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/"
            "sam2.1_hiera_large.pt -P checkpoints/sam2/"
        ) from e

    # build_sam2 uses Hydra and expects a config name relative to the sam2
    # package (e.g. "configs/sam2.1/sam2.1_hiera_l"), not an absolute path.
    sam2_model = build_sam2("configs/sam2.1/sam2.1_hiera_l", checkpoint, device=DEVICE)
    predictor = SAM2ImagePredictor(sam2_model)
    return predictor, sam2_model


# ---------------------------------------------------------------------------
# Pass 1: SAM2 box-prompt refinement (existing behaviour)
# ---------------------------------------------------------------------------

def _refine_boxes_with_sam2(
    predictor,
    coarse_boxes: torch.Tensor,      # (N, 4) xyxy float — image already set on predictor
) -> torch.Tensor:
    """
    Refine coarse detector boxes using SAM2 box-prompt segmentation.

    Returns refined boxes (N, 4) xyxy float on CPU.
    Boxes that SAM2 fails to segment fall back to the original coarse box.
    Image must already be set via predictor.set_image() before calling.
    """
    n = coarse_boxes.size(0)
    if n == 0:
        return coarse_boxes.cpu()

    boxes_np = coarse_boxes.cpu().numpy()       # (N, 4) float32
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=boxes_np,                           # SAM2 expects (N, 4) xyxy
        multimask_output=False,
    )
    # masks: (N, 1, H, W) bool

    refined = coarse_boxes.clone().cpu()
    for i in range(n):
        mask_i = masks[i, 0]                    # (H, W) bool
        bbox = _mask_to_tight_box(mask_i)
        if bbox is not None:
            refined[i] = torch.tensor(bbox, dtype=torch.float32)

    return refined


# ---------------------------------------------------------------------------
# Pass 2: Column-gap filling
# ---------------------------------------------------------------------------

def _cluster_into_columns(
    boxes: torch.Tensor,          # (N, 4) xyxy
    col_gap_factor: float = 1.5,
) -> List[List[int]]:
    """Group proposal indices into vertical text columns by x-center proximity."""
    n = boxes.size(0)
    if n == 0:
        return []

    cx = ((boxes[:, 0] + boxes[:, 2]) * 0.5).cpu().numpy()
    widths = (boxes[:, 2] - boxes[:, 0]).cpu().numpy()
    median_w = float(np.median(widths)) if n > 0 else 20.0
    gap_thresh = max(median_w * col_gap_factor, 5.0)

    order = np.argsort(cx)
    sorted_cx = cx[order]

    columns: List[List[int]] = []
    current: List[int] = [int(order[0])]
    for k in range(1, len(order)):
        if sorted_cx[k] - sorted_cx[k - 1] > gap_thresh:
            columns.append(current)
            current = []
        current.append(int(order[k]))
    columns.append(current)

    return columns


def _find_gap_candidates(
    col_boxes: torch.Tensor,      # (M, 4) xyxy — proposals in one column
    gap_factor: float = 1.8,
    max_fill: int = 8,
) -> List[Tuple[float, float, float, float]]:
    """
    Find vertical gaps in a single text column and return (cx, cy, w, h) candidates
    for the estimated missing characters.
    """
    m = col_boxes.size(0)
    if m < 2:
        return []

    cy = ((col_boxes[:, 1] + col_boxes[:, 3]) * 0.5).cpu().numpy()
    heights = (col_boxes[:, 3] - col_boxes[:, 1]).cpu().numpy()
    widths  = (col_boxes[:, 2] - col_boxes[:, 0]).cpu().numpy()
    cx_vals = ((col_boxes[:, 0] + col_boxes[:, 2]) * 0.5).cpu().numpy()

    h_med  = float(np.median(heights))
    w_med  = float(np.median(widths))
    cx_med = float(np.median(cx_vals))

    if h_med < 2.0:
        return []

    order = np.argsort(cy)
    sorted_cy = cy[order]

    candidates: List[Tuple[float, float, float, float]] = []
    for k in range(len(order) - 1):
        y_lo = sorted_cy[k]
        y_hi = sorted_cy[k + 1]
        gap = y_hi - y_lo
        if gap > gap_factor * h_med:
            n_missing = min(int(round(gap / h_med)) - 1, max_fill)
            for m_idx in range(1, n_missing + 1):
                fill_cy = y_lo + m_idx * (gap / (n_missing + 1))
                candidates.append((cx_med, fill_cy, w_med, h_med))

    return candidates


def _recover_with_point_prompts(
    predictor,
    centers: List[Tuple[float, float, float, float]],  # (cx, cy, w, h)
    fn_score: float,
    min_area: float,
    max_area: float,
    max_aspect: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Use SAM2 point prompts at candidate centres to recover missed characters.
    Image must already be set via predictor.set_image() before calling.
    """
    if not centers:
        return torch.zeros((0, 4)), torch.zeros((0,))

    n = len(centers)
    # (N, 1, 2) / (N, 1): one foreground point per candidate, batched into a
    # single predict() call — mirrors the batched (N, 4) box prompts used in
    # _refine_boxes_with_sam2.
    point_coords = np.array(
        [[[cx, cy]] for (cx, cy, _w, _h) in centers], dtype=np.float32
    )
    point_labels = np.ones((n, 1), dtype=np.int32)

    try:
        masks, sam_scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=None,
            multimask_output=True,
        )
    except Exception:
        return torch.zeros((0, 4)), torch.zeros((0,))

    # SAM2ImagePredictor.predict() does masks.squeeze(0) internally, so a
    # single-candidate batch collapses (1, 3, H, W) -> (3, H, W). Restore the
    # per-candidate dimension so indexing below is uniform for any N.
    if masks.ndim == 3:
        masks = masks[None, ...]
        sam_scores = sam_scores[None, ...]
    # masks: (N, 3, H, W) bool, sam_scores: (N, 3)

    recovered_boxes: List[List[float]] = []
    recovered_scores: List[float] = []

    for i in range(n):
        best_idx = int(np.argmax(sam_scores[i]))
        mask = masks[i, best_idx]  # (H, W) bool

        bbox = _mask_to_tight_box(mask)
        if bbox is None:
            continue

        x1, y1, x2, y2 = bbox
        bw = x2 - x1
        bh = y2 - y1
        if bw < 1.0 or bh < 1.0:
            continue
        area = bw * bh
        aspect = max(bw, bh) / (min(bw, bh) + 1e-3)

        if area < min_area or area > max_area or aspect > max_aspect:
            continue

        recovered_boxes.append([x1, y1, x2, y2])
        recovered_scores.append(fn_score)

    if not recovered_boxes:
        return torch.zeros((0, 4)), torch.zeros((0,))

    return (
        torch.tensor(recovered_boxes, dtype=torch.float32),
        torch.tensor(recovered_scores, dtype=torch.float32),
    )


def _fill_column_gaps(
    predictor,
    pass1_boxes: torch.Tensor,     # (N, 4) xyxy — already SAM2-refined pass-1 proposals
    gap_factor: float,
    col_gap_factor: float,
    fn_col_score: float,
    fn_min_area: float,
    fn_max_area: float,
    fn_max_aspect: float,
    max_fill: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Column-gap FN recovery: find y-gaps in vertical text columns, prompt SAM2."""
    if pass1_boxes.size(0) < 4:
        return torch.zeros((0, 4)), torch.zeros((0,))

    columns = _cluster_into_columns(pass1_boxes, col_gap_factor)
    all_candidates: List[Tuple[float, float, float, float]] = []

    for col_indices in columns:
        if len(col_indices) < 2:
            continue
        col_boxes = pass1_boxes[col_indices]
        candidates = _find_gap_candidates(col_boxes, gap_factor, max_fill)
        all_candidates.extend(candidates)

    return _recover_with_point_prompts(
        predictor, all_candidates, fn_col_score,
        fn_min_area, fn_max_area, fn_max_aspect,
    )


# ---------------------------------------------------------------------------
# Pass 3: Sub-threshold peak recovery
# ---------------------------------------------------------------------------

def _recover_subthreshold(
    predictor,
    det_out: dict,
    h_img: int,
    w_img: int,
    fn_score_low: float,
    fn_score_high: float,
    fn_top_k: int,
    pass1_boxes: torch.Tensor,
    fn_min_area: float,
    fn_max_area: float,
    fn_max_aspect: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract sub-threshold heatmap peaks, remove those inside pass-1 boxes,
    feed the rest as SAM2 box prompts, filter by geometry.
    """
    boxes_list, scores_list = extract_coarse_proposals(
        heat_logits=det_out["heatmap"],
        bbox_reg=det_out["bbox"],
        image_size=(h_img, w_img),
        score_thresh=fn_score_low,
        top_k=fn_top_k,
        nms_iou=DET_NMS_IOU,
        min_size=2.0,
    )

    all_boxes  = boxes_list[0].cpu()
    all_scores = scores_list[0].cpu()

    # Keep only sub-threshold candidates
    sub_mask = all_scores < fn_score_high
    sub_boxes  = all_boxes[sub_mask]
    sub_scores = all_scores[sub_mask]

    if sub_boxes.size(0) == 0:
        return torch.zeros((0, 4)), torch.zeros((0,))

    # Remove candidates whose centre falls inside any pass-1 box
    if pass1_boxes.size(0) > 0:
        sub_cx = (sub_boxes[:, 0] + sub_boxes[:, 2]) * 0.5
        sub_cy = (sub_boxes[:, 1] + sub_boxes[:, 3]) * 0.5
        keep_flags = []
        for k in range(sub_boxes.size(0)):
            inside = (
                (sub_cx[k] >= pass1_boxes[:, 0]) & (sub_cx[k] <= pass1_boxes[:, 2]) &
                (sub_cy[k] >= pass1_boxes[:, 1]) & (sub_cy[k] <= pass1_boxes[:, 3])
            ).any()
            keep_flags.append(not bool(inside))
        keep_idx = torch.tensor(
            [k for k, f in enumerate(keep_flags) if f], dtype=torch.long
        )
        if keep_idx.numel() == 0:
            return torch.zeros((0, 4)), torch.zeros((0,))
        sub_boxes  = sub_boxes[keep_idx]
        sub_scores = sub_scores[keep_idx]

    # SAM2 box-prompt refine on these candidates
    refined = _refine_boxes_with_sam2(predictor, sub_boxes)

    # Geometry filter
    bw = refined[:, 2] - refined[:, 0]
    bh = refined[:, 3] - refined[:, 1]
    area = bw * bh
    aspect = torch.max(bw, bh) / (torch.min(bw, bh) + 1e-3)
    geom_ok = (area >= fn_min_area) & (area <= fn_max_area) & (aspect <= fn_max_aspect)

    return refined[geom_ok], sub_scores[geom_ok]


# ---------------------------------------------------------------------------
# Pass 4: Frame / cartouche detection
# ---------------------------------------------------------------------------

def _load_sam2_mask_generator(sam2_model, points_per_side: int = 16):
    """Create a SAM2AutomaticMaskGenerator from an already-loaded SAM2 model."""
    try:
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    except ImportError as e:
        raise ImportError("SAM2 automatic_mask_generator not found. Ensure sam2 is installed from source.") from e

    return SAM2AutomaticMaskGenerator(
        sam2_model,
        points_per_side=points_per_side,
        pred_iou_thresh=0.80,
        stability_score_thresh=0.85,
        min_mask_region_area=2000,
    )


def _recover_framed_text(
    mask_generator,
    backbone,
    detector,
    image_rgb: np.ndarray,
    to_tensor_norm,
    h_img: int,
    w_img: int,
    pass1_boxes: torch.Tensor,
    pass1_scores: torch.Tensor,
    score_thresh: float,
    top_k: int,
    frame_min_area: float = 2000.0,
    frame_max_area: float = 50000.0,
    frame_fill_thr: float = 0.70,
    median_char_area: float = 617.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Detect rectangular frames (cartouches, seals, title boxes) using SAM2 auto-segmentation,
    then re-run the detector at higher resolution inside frames that have too few proposals.
    """
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        masks_data = mask_generator.generate(image_rgb)
    if not masks_data:
        return torch.zeros((0, 4)), torch.zeros((0,))

    all_recovered_boxes: List[torch.Tensor] = []
    all_recovered_scores: List[torch.Tensor] = []

    for mask_info in masks_data:
        area = float(mask_info["area"])
        if area < frame_min_area or area > frame_max_area:
            continue

        # SAM2 bbox is (x, y, w, h)
        fx, fy, fw, fh = mask_info["bbox"]
        tight_area = fw * fh
        if tight_area < 1.0:
            continue

        # Rectangularity: how much of the tight bbox the mask fills
        fill_ratio = area / tight_area
        if fill_ratio < frame_fill_thr:
            continue

        # Aspect ratio
        aspect = max(fw, fh) / (min(fw, fh) + 1e-3)
        if aspect > 3.0:
            continue

        x1f, y1f, x2f, y2f = fx, fy, fx + fw, fy + fh

        # Count existing proposals already inside this frame region
        if pass1_boxes.size(0) > 0:
            inside = (
                (pass1_boxes[:, 0] >= x1f) & (pass1_boxes[:, 2] <= x2f) &
                (pass1_boxes[:, 1] >= y1f) & (pass1_boxes[:, 3] <= y2f)
            )
            existing_count = int(inside.sum())
        else:
            existing_count = 0

        expected_count = max(1, int(area / max(median_char_area, 1.0) * 0.3))
        if existing_count >= expected_count:
            continue

        # Crop to frame region and upscale to 512×512 for higher-resolution detection
        x1c = max(0, int(x1f))
        y1c = max(0, int(y1f))
        x2c = min(w_img, int(x2f))
        y2c = min(h_img, int(y2f))
        if x2c <= x1c or y2c <= y1c:
            continue

        crop_rgb = image_rgb[y1c:y2c, x1c:x2c]
        crop_pil = Image.fromarray(crop_rgb).resize((512, 512), Image.LANCZOS)
        crop_t = to_tensor_norm(crop_pil).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            feats = backbone(crop_t)
            det_out = detector(feats)
            boxes_list, scores_list = extract_coarse_proposals(
                heat_logits=det_out["heatmap"],
                bbox_reg=det_out["bbox"],
                image_size=(512, 512),
                score_thresh=score_thresh,
                top_k=top_k,
                nms_iou=DET_NMS_IOU,
                min_size=2.0,
            )

        crop_boxes = boxes_list[0].cpu()
        if crop_boxes.size(0) == 0:
            continue

        # Map from 512×512 crop space back to full image coordinates
        cw = float(x2c - x1c)
        ch = float(y2c - y1c)
        scale_x = cw / 512.0
        scale_y = ch / 512.0

        full_boxes = crop_boxes.clone()
        full_boxes[:, 0] = crop_boxes[:, 0] * scale_x + x1c
        full_boxes[:, 1] = crop_boxes[:, 1] * scale_y + y1c
        full_boxes[:, 2] = crop_boxes[:, 2] * scale_x + x1c
        full_boxes[:, 3] = crop_boxes[:, 3] * scale_y + y1c

        all_recovered_boxes.append(full_boxes)
        all_recovered_scores.append(scores_list[0].cpu())

    if not all_recovered_boxes:
        return torch.zeros((0, 4)), torch.zeros((0,))

    return torch.cat(all_recovered_boxes, dim=0), torch.cat(all_recovered_scores, dim=0)


# ---------------------------------------------------------------------------
# Image list
# ---------------------------------------------------------------------------

def _collect_images(data_dir: Path, splits: List[str]) -> List[Path]:
    """Return all image paths for the requested splits (train/val/test)."""
    split_dir = data_dir / "splits"
    images: List[Path] = []

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
                img_rel = ann.get("img_path") or ann.get("image_path") or ann.get("path")
                if img_rel:
                    img_path = data_dir / img_rel
                    if img_path.exists():
                        images.append(img_path)
            except Exception:
                pass

    return images


# ---------------------------------------------------------------------------
# Shared processing loop (used by main() and run_preprocessing())
# ---------------------------------------------------------------------------

def _run_processing_loop(
    backbone,
    detector,
    predictor,
    mask_generator,
    images: List[Path],
    output_dir: Path,
    args,
) -> None:
    """Per-image SAM2 refinement + FN-recovery loop."""
    to_tensor_norm = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    h_img, w_img = IMAGE_SIZE

    skipped = 0
    proposals_per_image: List[int] = []
    fn_recovered_per_image: List[int] = []

    for img_path in tqdm(images, desc="SAM2 refinement"):
        out_path = output_dir / f"{img_path.stem}.pt"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            pil_img = Image.open(img_path).convert("RGB").resize(
                (w_img, h_img), RESAMPLE_LANCZOS
            )
        except Exception as e:
            print(f"WARNING: could not open {img_path}: {e}")
            continue

        img_rgb = np.array(pil_img)
        img_t = to_tensor_norm(img_rgb)
        img_t = img_t.unsqueeze(0).to(DEVICE)

        # Restore Stage 1 to GPU (no-op on first iteration; cheap .to() on subsequent ones)
        backbone.to(DEVICE)
        detector.to(DEVICE)

        # --- Detector forward pass ---
        with torch.no_grad():
            feats   = backbone(img_t)
            det_out = detector(feats)
            boxes_list, scores_list = extract_coarse_proposals(
                heat_logits=det_out["heatmap"],
                bbox_reg=det_out["bbox"],
                image_size=(h_img, w_img),
                score_thresh=args.score_thresh,
                top_k=args.top_k,
                nms_iou=DET_NMS_IOU,
                min_size=2.0,
            )

        coarse_boxes  = boxes_list[0].cpu()
        coarse_scores = scores_list[0].cpu()
        del img_t, feats  # release image tensor and backbone activations immediately

        # Offload Stage 1 weights to CPU before SAM2 to reclaim VRAM and reduce
        # allocator fragmentation.  sub_thresh and recover_frames need backbone/det_out
        # on GPU later in the same iteration, so skip offload in those modes.
        _offload_s1 = (
            not getattr(args, "sub_thresh", False)
            and not getattr(args, "recover_frames", False)
        )
        if _offload_s1:
            backbone.cpu()
            detector.cpu()
            torch.cuda.empty_cache()

        fn_boxes_all:  List[torch.Tensor] = []
        fn_scores_all: List[torch.Tensor] = []

        # --- SAM2 image encoding + Pass 1-3 (every predictor.predict() call for
        # this image shares one inference_mode/autocast scope, per SAM2 docs).
        # Stage 1 backbone/detector forward passes stay outside — they don't need
        # SAM2's autocast dtype and already ran (or, for Pass 4, run separately below).
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictor.set_image(img_rgb)

            # --- Pass 1: Box-prompt refinement ---
            if coarse_boxes.size(0) == 0:
                refined_boxes = coarse_boxes
            else:
                refined_boxes = _refine_boxes_with_sam2(predictor, coarse_boxes)
            refined_scores = coarse_scores

            # --- Pass 2: Column-gap filling (with second pass for newly recovered chars) ---
            if args.fill_gaps and refined_boxes.size(0) > 0:
                gap_boxes, gap_scores = _fill_column_gaps(
                    predictor,
                    refined_boxes,
                    gap_factor=args.gap_factor,
                    col_gap_factor=args.col_gap_factor,
                    fn_col_score=args.fn_col_score,
                    fn_min_area=args.fn_min_area,
                    fn_max_area=args.fn_max_area,
                    fn_max_aspect=args.fn_max_aspect,
                )
                if gap_boxes.size(0) > 0:
                    fn_boxes_all.append(gap_boxes)
                    fn_scores_all.append(gap_scores)
                    # Second pass: re-check gaps in the augmented column (catches cases where
                    # SAM2 only partially fills a multi-char gap, leaving a new gap between
                    # the recovered positions that was not visible in the first pass).
                    combined = torch.cat([refined_boxes, gap_boxes], dim=0)
                    gap_boxes2, gap_scores2 = _fill_column_gaps(
                        predictor,
                        combined,
                        gap_factor=args.gap_factor,
                        col_gap_factor=args.col_gap_factor,
                        fn_col_score=args.fn_col_score,
                        fn_min_area=args.fn_min_area,
                        fn_max_area=args.fn_max_area,
                        fn_max_aspect=args.fn_max_aspect,
                    )
                    if gap_boxes2.size(0) > 0:
                        fn_boxes_all.append(gap_boxes2)
                        fn_scores_all.append(gap_scores2)

            # --- Pass 3: Sub-threshold peaks ---
            if args.sub_thresh:
                sub_boxes, sub_scores = _recover_subthreshold(
                    predictor,
                    det_out,
                    h_img, w_img,
                    fn_score_low=args.fn_score_low,
                    fn_score_high=args.score_thresh,
                    fn_top_k=args.fn_top_k,
                    pass1_boxes=refined_boxes,
                    fn_min_area=args.fn_min_area,
                    fn_max_area=args.fn_max_area,
                    fn_max_aspect=args.fn_max_aspect,
                )
                if sub_boxes.size(0) > 0:
                    fn_boxes_all.append(sub_boxes)
                    fn_scores_all.append(sub_scores)

        # --- Pass 4: Frame/cartouche recovery ---
        if args.recover_frames and mask_generator is not None:
            frame_boxes, frame_scores = _recover_framed_text(
                mask_generator,
                backbone,
                detector,
                img_rgb,
                to_tensor_norm,
                h_img, w_img,
                pass1_boxes=refined_boxes,
                pass1_scores=refined_scores,
                score_thresh=args.score_thresh,
                top_k=args.top_k,
                frame_min_area=args.frame_min_area,
                frame_max_area=args.frame_max_area,
                frame_fill_thr=args.frame_fill_thr,
            )
            if frame_boxes.size(0) > 0:
                fn_boxes_all.append(frame_boxes)
                fn_scores_all.append(frame_scores)

        # --- Merge & deduplicate ---
        n_fn_recovered = sum(b.size(0) for b in fn_boxes_all)

        if fn_boxes_all:
            all_boxes  = torch.cat([refined_boxes]  + fn_boxes_all,  dim=0)
            all_scores = torch.cat([refined_scores] + fn_scores_all, dim=0)
            if all_boxes.size(0) > 1:
                keep = torchvision.ops.nms(all_boxes, all_scores, iou_threshold=args.fn_nms_iou)
                all_boxes  = all_boxes[keep]
                all_scores = all_scores[keep]
        else:
            all_boxes  = refined_boxes
            all_scores = refined_scores

        torch.save({"boxes": all_boxes, "scores": all_scores}, out_path)
        proposals_per_image.append(int(all_boxes.size(0)))
        fn_recovered_per_image.append(n_fn_recovered)

    avg_proposals = sum(proposals_per_image) / max(1, len(proposals_per_image))
    avg_fn = sum(fn_recovered_per_image) / max(1, len(fn_recovered_per_image))
    print(
        f"\nDone.  Processed {len(proposals_per_image)} images | "
        f"Skipped {skipped} (already exist)\n"
        f"Avg proposals/image: {avg_proposals:.1f}  "
        f"Avg FN-recovered/image: {avg_fn:.1f}\n"
        f"Refined proposals saved to: {output_dir}\n\n"
        f"To use in training, pass --sam2_proposals {output_dir} to train_stage2_kuronet.py"
    )


# ---------------------------------------------------------------------------
# Callable API (mirrors preprocess_sam2_illustrations.run_preprocessing)
# ---------------------------------------------------------------------------

def run_preprocessing(
    detector_ckpt: "str | Path | None" = None,
    output_dir: "str | Path | None" = None,
    splits: "list[str] | tuple[str, ...]" = ("train", "val"),
    *,
    overwrite: bool = False,
    fill_gaps: bool = True,
    sub_thresh: bool = False,
    score_thresh: "float | None" = None,
    top_k: "int | None" = None,
    gap_factor: float = 1.8,
    col_gap_factor: float = 1.5,
    fn_col_score: float = 0.40,
    fn_min_area: float = 50.0,
    fn_max_area: float = 4000.0,
    fn_max_aspect: float = 4.0,
    fn_nms_iou: float = 0.3,
    fn_score_low: float = 0.10,
    fn_top_k: int = 600,
) -> None:
    """
    Build (or extend) the SAM2 proposal cache for the given splits.

    Skips already-cached images unless overwrite=True.
    Called automatically at end of Stage 1 training when SAM2_PROPOSALS_PREPROCESSING=True
    in config.py.  Frame/cartouche recovery is not available here; use the CLI directly
    for that mode (requires a separate mask generator and additional VRAM).
    """
    import argparse as _argparse

    # Reduce CUDA allocator fragmentation so SAM2's large contiguous allocations
    # don't OOM when the heap is fragmented by the preceding Stage 1 forward pass.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    out_dir = Path(output_dir or SAM2_PROPOSALS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(DATA_DIR)
    images = _collect_images(data_dir, list(splits))
    if not images:
        print("[SAM2 proposals] No images found — check splits and DATA_DIR.")
        return
    pending = [p for p in images if not (out_dir / f"{p.stem}.pt").exists() or overwrite]
    if not pending:
        print(f"[SAM2 proposals] Cache complete ({len(images)} images in {out_dir}).")
        return
    print(f"[SAM2 proposals] {len(pending)}/{len(images)} images pending in {list(splits)}")

    backbone, detector = _load_detector(ckpt_path=detector_ckpt)
    predictor, _ = _load_sam2(str(SAM2_CHECKPOINT))

    ns = _argparse.Namespace(
        overwrite=overwrite,
        fill_gaps=fill_gaps,
        sub_thresh=sub_thresh,
        score_thresh=score_thresh if score_thresh is not None else DET_SCORE_THRESH,
        top_k=top_k if top_k is not None else DET_TOP_K,
        gap_factor=gap_factor,
        col_gap_factor=col_gap_factor,
        fn_col_score=fn_col_score,
        fn_min_area=fn_min_area,
        fn_max_area=fn_max_area,
        fn_max_aspect=fn_max_aspect,
        fn_nms_iou=fn_nms_iou,
        fn_score_low=fn_score_low,
        fn_top_k=fn_top_k,
        recover_frames=False,
        frame_min_area=2000.0,
        frame_max_area=50000.0,
        frame_fill_thr=0.70,
    )

    _run_processing_loop(backbone, detector, predictor, None, pending, out_dir, ns)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Offline SAM2 proposal refinement + FN recovery")
    parser.add_argument(
        "--sam2_checkpoint",
        default="checkpoints/sam2/sam2.1_hiera_large.pt",
        help="Path to SAM2 model checkpoint (.pt)",
    )
    parser.add_argument(
        "--output_dir",
        default="assets/sam2_proposals",
        help="Directory where per-image refined proposal .pt files are saved",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Dataset splits to process",
    )
    parser.add_argument(
        "--score_thresh",
        type=float,
        default=DET_SCORE_THRESH,
        help="Detector score threshold for initial proposals",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=DET_TOP_K,
        help="Detector top-K proposals per image",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-process images that already have a .pt file",
    )

    # --- Column-gap filling ---
    parser.add_argument("--fill_gaps", action="store_true",
                        help="Enable column-gap FN recovery (Approach 1)")
    parser.add_argument("--gap_factor", type=float, default=1.8,
                        help="Gap/h_med ratio to declare a missing character")
    parser.add_argument("--col_gap_factor", type=float, default=1.5,
                        help="x-distance/median_width ratio to split into separate columns")
    parser.add_argument("--fn_col_score", type=float, default=0.40,
                        help="Score assigned to gap-filled proposals")

    # --- Sub-threshold peaks ---
    parser.add_argument("--sub_thresh", action="store_true",
                        help="Enable sub-threshold peak recovery (Approach 2)")
    parser.add_argument("--fn_score_low", type=float, default=0.10,
                        help="Min heatmap confidence for sub-threshold candidates")
    parser.add_argument("--fn_top_k", type=int, default=600,
                        help="Max sub-threshold candidates per image")

    # --- Frame/cartouche recovery ---
    parser.add_argument("--recover_frames", action="store_true",
                        help="Enable frame/cartouche text recovery (Approach 3)")
    parser.add_argument("--frame_min_area", type=float, default=2000.0,
                        help="Min SAM2 mask area to consider as a frame (px²)")
    parser.add_argument("--frame_max_area", type=float, default=50000.0,
                        help="Max frame mask area (px²)")
    parser.add_argument("--frame_fill_thr", type=float, default=0.70,
                        help="Min mask_area/bbox_area ratio (rectangularity threshold)")

    # --- Shared geometry filter ---
    parser.add_argument("--fn_min_area", type=float, default=50.0,
                        help="Min accepted mask area for FN candidates (px²)")
    parser.add_argument("--fn_max_area", type=float, default=4000.0,
                        help="Max accepted mask area for FN candidates (px²)")
    parser.add_argument("--fn_max_aspect", type=float, default=4.0,
                        help="Max width/height ratio for FN candidates")
    parser.add_argument("--fn_nms_iou", type=float, default=0.3,
                        help="NMS IoU threshold for deduplication across all proposal sets")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(DATA_DIR)
    images = _collect_images(data_dir, args.splits)
    if not images:
        print("No images found. Check --splits and DATA_DIR in config.py.")
        return
    print(f"Found {len(images)} images across splits: {args.splits}")

    # Load detector (always needed)
    backbone, detector = _load_detector()

    # Load SAM2
    predictor, sam2_model = _load_sam2(args.sam2_checkpoint)

    # Load mask generator only if frame recovery is requested (shares sam2_model)
    mask_generator = None
    if args.recover_frames:
        mask_generator = _load_sam2_mask_generator(sam2_model, points_per_side=16)
        print("SAM2AutomaticMaskGenerator loaded for frame recovery.")

    _run_processing_loop(backbone, detector, predictor, mask_generator, images, output_dir, args)


if __name__ == "__main__":
    main()
