"""Validate Stage 2 sequence recognition with Stage 1 box proposals.

This script evaluates the current ROI-based Stage 2 architecture:

    image -> Stage 1 UNet/detector -> boxes (GT or predicted)
         -> ROISequenceEncoder -> ROIContextEncoder -> attention decoder

It provides three complementary views:
1) Stage 2 sequence quality (CER / exact match) under a chosen box source.
2) Teacher-forced Stage 2 validation losses.
3) Detection-to-text proxy using predicted Stage 1 boxes matched to GT labels.

Box source modes for Stage 2 decoding:
- "gt"   : decode using GT boxes (oracle upper bound)
- "pred" : decode using Stage 1 predicted boxes (realistic end-to-end mode)
- "both" : decode both ways for direct comparison
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

# Add parent directory to path so we can import config and utils
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    CHECKPOINT_DIR,
    CONTEXT_HIDDEN_DIM,
    DATA_DIR,
    DEVICE,
    IMAGE_SIZE,
    ROI_EMBED_DIM,
    ROI_BOX_LOSS_WEIGHT,
    ROI_POOL_SIZE,
    STAGE2_READING_ORDER_POLICY,
    USE_ROI_ATTENTION,
)
from model.kuronet import DetectorHead, EncoderWrapper, ROIContextEncoder, ROISequenceEncoder, UNet
from model.kuronet.decoder.attention import SeqDecoderAttention
from utils import KuzushijiDataset
from utils.detection_utils import compute_roi_box_loss
from utils.text_normalization import render_tokens
from utils.training_helpers import collate_fn
from utils.vocab import VocabManager
from validate_stage1 import compute_detection_metrics, compute_iou_batch, extract_boxes_from_heatmap

def _edit_distance(a: str, b: str) -> int:
    """Classic DP edit distance."""
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
            curr.append(min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + cost,
            ))
        prev = curr
    return prev[-1]


def _sort_boxes_reading_order(boxes: Sequence[Sequence[float]], orientation: str) -> List[List[float]]:
    if boxes is None or len(boxes) == 0:
        return []
    if orientation == "vertical":
        idx = sorted(range(len(boxes)), key=lambda i: (-boxes[i][0], boxes[i][1]))
    else:
        idx = sorted(range(len(boxes)), key=lambda i: (boxes[i][1], boxes[i][0]))
    return [[float(v) for v in boxes[i]] for i in idx]


def _infer_reading_orientation_from_boxes(boxes: Sequence[Sequence[float]]) -> str:
    """Infer reading direction from nearest-neighbor center distances; default vertical."""
    if boxes is None or len(boxes) == 0:
        return "vertical"
    centers = torch.tensor(
        [[0.5 * (float(b[0]) + float(b[2])), 0.5 * (float(b[1]) + float(b[3]))] for b in boxes],
        dtype=torch.float32,
    )
    dx = torch.cdist(centers[:, :1], centers[:, :1], p=1)
    dy = torch.cdist(centers[:, 1:2], centers[:, 1:2], p=1)
    inf = torch.tensor(float("inf"), dtype=torch.float32)
    dx = dx + torch.eye(dx.size(0), dtype=torch.float32) * inf
    dy = dy + torch.eye(dy.size(0), dtype=torch.float32) * inf
    mean_min_dx = float(dx.min(dim=1).values.mean().item())
    mean_min_dy = float(dy.min(dim=1).values.mean().item())
    return "vertical" if mean_min_dy <= mean_min_dx else "horizontal"


def _resolve_sort_orientation(
    orientation_hint: Optional[str],
    boxes: Sequence[Sequence[float]],
    reading_order_policy: str,
) -> str:
    policy = str(reading_order_policy or "annotation").strip().lower()
    if policy == "inferred":
        return _infer_reading_orientation_from_boxes(boxes)
    if policy == "auto":
        if orientation_hint in ("vertical", "horizontal"):
            return orientation_hint
        return _infer_reading_orientation_from_boxes(boxes)
    # default: annotation
    if orientation_hint in ("vertical", "horizontal"):
        return orientation_hint
    return "vertical"

def _build_boxes_for_stage2_validation(
    images: torch.Tensor,
    boxes_batch: Optional[Sequence[Any]],
    orientations: Optional[Sequence[str]],
    use_gt_boxes: bool,
    unet: UNet,
    detector: DetectorHead,
    confidence: float,
    top_k: int,
    nms_iou: float,
    reading_order_policy: str,
) -> List[torch.Tensor]:
    """
    Build sorted box sequences for ROI-based Stage 2 validation.

    If use_gt_boxes=True:
        use GT boxes from the dataset and sort explicitly.

    If use_gt_boxes=False:
        run Stage 1 detector, decode predicted boxes, then sort them.
    """
    if use_gt_boxes:
        out: List[torch.Tensor] = []
        for i, b in enumerate(boxes_batch or []):
            if b is None or b.numel() == 0:
                out.append(torch.empty((0, 4), dtype=torch.float32, device=DEVICE))
                continue

            boxes_i = b.detach().cpu().tolist()
            orientation_hint = orientations[i] if orientations is not None and i < len(orientations) else None
            sort_orientation = _resolve_sort_orientation(orientation_hint, boxes_i, reading_order_policy)
            boxes_i = _sort_boxes_reading_order(boxes_i, sort_orientation)
            out.append(torch.tensor(boxes_i, dtype=torch.float32, device=DEVICE))
        return out

    with torch.no_grad():
        features = unet(images)
        det_out = detector(features)
        heat_probs = torch.sigmoid(det_out["heatmap"])
        bbox_reg = det_out["bbox"]
        _, _, hf, wf = features.shape

        out: List[torch.Tensor] = []
        for i in range(images.size(0)):
            pred_boxes_i, _, _ = extract_boxes_from_heatmap(
                heatmap_probs=heat_probs[i:i + 1],
                bbox_reg=bbox_reg[i:i + 1],
                confidence_thresh=confidence,
                output_size=(hf, wf),
                image_size=IMAGE_SIZE,
                top_k=top_k,
                nms_iou=nms_iou,
                min_box_size=4.0,
                debug=False,
            )

            orientation_hint = orientations[i] if orientations is not None and i < len(orientations) else None
            sort_orientation = _resolve_sort_orientation(orientation_hint, pred_boxes_i, reading_order_policy)
            pred_boxes_i = _sort_boxes_reading_order(pred_boxes_i, sort_orientation)

            if len(pred_boxes_i) == 0:
                out.append(torch.empty((0, 4), dtype=torch.float32, device=DEVICE))
            else:
                out.append(torch.tensor(pred_boxes_i, dtype=torch.float32, device=DEVICE))

        return out
    
def _decode_text_from_ids(ids: Sequence[int], vocab: VocabManager) -> str:
    chars = vocab.decode([int(x) for x in ids], remove_special=True)
    return render_tokens(chars)


def _truncate_at_eos(ids: Sequence[int], eos_id: int) -> List[int]:
    out: List[int] = []
    for tok in ids:
        out.append(int(tok))
        if int(tok) == eos_id:
            break
    return out


def _denormalize_image_tensor(image_tensor: torch.Tensor) -> np.ndarray:
    """Convert normalized tensor (C,H,W) to uint8 RGB image."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = (img * std + mean) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _resolve_plot_fonts(font_size: int, font_path: Optional[str] = None) -> List[Any]:
    """Load a prioritized list of fonts for per-character fallback rendering."""
    candidates: List[Path] = []
    if font_path:
        candidates.append(Path(font_path))

    # Prioritize broad CJK coverage fonts first (especially useful for historical text).
    candidates.extend([
        Path("/usr/share/fonts/truetype/hanazono/HanaMinA.ttf"),
        Path("/usr/share/fonts/truetype/hanazono/HanaMinB.ttf"),
        Path("/usr/share/fonts/opentype/hanazono/HanaMinA.otf"),
        Path("/usr/share/fonts/opentype/hanazono/HanaMinB.otf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttf"),
        Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
        Path("/usr/share/fonts/truetype/fonts-japanese-gothic.ttf"),
        Path("/usr/share/fonts/truetype/takao-gothic/TakaoPGothic.ttf"),
        Path("/usr/share/fonts/truetype/ipafont-gothic/ipag.ttf"),
    ])

    # Also query system fontconfig for Japanese-capable fonts.
    try:
        proc = subprocess.run(
            ["fc-list", ":lang=ja", "file"],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            for line in proc.stdout.splitlines():
                p = line.strip()
                if p:
                    candidates.append(Path(p))
    except Exception:
        pass

    seen: set[str] = set()
    fonts: List[Any] = []
    for p in candidates:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)

        if p.exists():
            try:
                fnt = ImageFont.truetype(str(p), size=font_size)
                fonts.append(fnt)
            except Exception:
                continue

    # Put fonts with likely CJK support first.
    fonts.sort(key=lambda f: 0 if _font_supports_cjk(f) else 1)
    if fonts:
        return fonts

    return [ImageFont.load_default()]


def _font_supports_cjk(font: Any) -> bool:
    """Heuristic check for Japanese glyph support in the selected font."""
    probe_chars = ["あ", "日", "の"]
    masks = []
    try:
        for ch in probe_chars:
            m = font.getmask(ch)
            masks.append((m.size, bytes(m)))
    except Exception:
        return False

    # If all probe chars map to the exact same bitmap, this is likely a missing-glyph fallback.
    return not (masks[0] == masks[1] == masks[2])


def _font_supports_char(font: Any, ch: str) -> bool:
    if not ch or ch.isspace():
        return True
    try:
        m = font.getmask(ch)
        ref1 = font.getmask("\uFFFD")
        ref2 = font.getmask("\u25A1")
        sig = (m.size, bytes(m))
        return sig != (ref1.size, bytes(ref1)) and sig != (ref2.size, bytes(ref2))
    except Exception:
        return False


def _pick_font_for_char(ch: str, fonts: Sequence[Any]) -> Any:
    for f in fonts:
        if _font_supports_char(f, ch):
            return f
    return fonts[0]


def _draw_text_with_fallback(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    text: str,
    fonts: Sequence[Any],
    fill,
) -> Tuple[int, int]:
    cursor: int = x
    max_h: int = 0
    for ch in text:
        f = _pick_font_for_char(ch, fonts)
        draw.text((cursor, y), ch, fill=fill, font=f)
        try:
            bb = draw.textbbox((cursor, y), ch, font=f)
            w = int(max(1, bb[2] - bb[0]))
            h = int(max(1, bb[3] - bb[1]))
        except Exception:
            w = int(max(1, int(0.6 * getattr(f, "size", 14))))
            h = int(max(1, int(getattr(f, "size", 14))))
        cursor += w
        max_h = max(max_h, h)
    return int(cursor - x), int(max_h)


def _safe_truncate_text(text: Optional[str], max_chars: int = 160) -> str:
    if text is None:
        return ""
    return text if len(text) <= max_chars else text[:max_chars] + " ..."


def _wrap_text_by_chars(text: str, width: int = 40) -> List[str]:
    """
    Wrap text by character count.
    Works better for Japanese than whitespace-based wrapping.
    """
    if not text:
        return [""]
    return [text[i:i + width] for i in range(0, len(text), width)]


def _draw_multiline_text(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    label: str,
    text: str,
    font: Any,
    fill=(235, 235, 235),
    label_fill=(255, 220, 120),
    wrap_width: int = 42,
    line_gap: int = 4,
) -> int:
    """
    Draw labeled multiline text and return new y-position after the block.
    """
    lines = _wrap_text_by_chars(text, width=wrap_width)

    fonts = font if isinstance(font, list) else [font]

    # label line
    _, line_h = _draw_text_with_fallback(draw, x, y, label, fonts, label_fill)
    if line_h <= 0:
        line_h = max(16, getattr(fonts[0], "size", 14) + 2)

    y += line_h + line_gap

    for line in lines:
        _, curr_h = _draw_text_with_fallback(draw, x, y, line, fonts, fill)
        if curr_h <= 0:
            curr_h = line_h
        y += curr_h + line_gap

    return y + 4


def _draw_stage2_box_character_plot(
    image_tensor: torch.Tensor,
    gt_boxes: Sequence[Sequence[float]],
    pred_boxes: Sequence[Sequence[float]],
    stage2_pred_text: str,
    proxy_pred_text: str,
    gt_text: str,
    orientation: str,
    save_path: Path,
    font_path: Optional[str],
    font_size: int,
    stage2_cer: Optional[float] = None,
    proxy_cer: Optional[float] = None,
    show_box_indices: bool = True,
    max_index_labels: int = 30,
    max_text_chars: int = 160,
) -> None:
    """
    Cleaner visualization:
    - left: original
    - right: GT and predicted boxes
    - bottom: GT / Stage2 / Proxy text summary

    Important:
    - no per-box character labels anymore
    - optional small index labels only
    - text rendered separately for readability
    """
    rgb = _denormalize_image_tensor(image_tensor)
    original_img = Image.fromarray(rgb).convert("RGB")
    overlay_img = original_img.copy().convert("RGBA")
    draw = ImageDraw.Draw(overlay_img, "RGBA")
    fonts = _resolve_plot_fonts(font_size=font_size, font_path=font_path)
    font = fonts[0]

    # --- draw GT boxes (green)
    for b in gt_boxes:
        x1, y1, x2, y2 = [float(v) for v in b]
        draw.rectangle([x1, y1, x2, y2], outline=(40, 220, 70, 220), width=1)

    # --- draw predicted boxes (red)
    for i, b in enumerate(pred_boxes):
        x1, y1, x2, y2 = [float(v) for v in b]
        draw.rectangle([x1, y1, x2, y2], outline=(230, 45, 45, 235), width=2)

        if show_box_indices and i < max_index_labels:
            idx_label = str(i)
            tx = int(x1)
            ty = max(0, int(y1) - (font_size + 4))

            try:
                tb = draw.textbbox((tx, ty), idx_label, font=font)
                tw = tb[2] - tb[0]
                th = tb[3] - tb[1]
            except Exception:
                tw = max(10, int(0.6 * font_size * len(idx_label)) + 4)
                th = font_size + 2

            draw.rectangle([tx, ty, tx + tw + 4, ty + th + 2], fill=(0, 0, 0, 160))
            draw.text((tx + 2, ty + 1), idx_label, fill=(255, 255, 255, 245), font=font)

    # Header
    header = f"ori={orientation} | gt_boxes={len(gt_boxes)} | pred_boxes={len(pred_boxes)}"
    try:
        hb = draw.textbbox((0, 0), header, font=font)
        hw = hb[2] - hb[0]
        hh = hb[3] - hb[1]
    except Exception:
        hw = int(0.6 * font_size * len(header)) + 8
        hh = font_size + 4

    draw.rectangle([0, 0, min(overlay_img.width, hw + 10), hh + 6], fill=(0, 0, 0, 170))
    draw.text((4, 3), header, fill=(255, 245, 120, 255), font=font)

    overlay_img = overlay_img.convert("RGB")

    # --- prepare text block
    gt_text_disp = _safe_truncate_text(gt_text, max_chars=max_text_chars)
    stage2_text_disp = _safe_truncate_text(stage2_pred_text, max_chars=max_text_chars)
    proxy_text_disp = _safe_truncate_text(proxy_pred_text, max_chars=max_text_chars)

    # Layout sizes
    gap = 10
    top_label_h = 22
    img_panel_w = original_img.width + gap + overlay_img.width
    img_panel_h = max(original_img.height, overlay_img.height) + top_label_h

    # Estimate text block height dynamically
    dummy = Image.new("RGB", (10, 10))
    ddraw = ImageDraw.Draw(dummy)
    try:
        line_h = ddraw.textbbox((0, 0), "Ag", font=font)[3] - ddraw.textbbox((0, 0), "Ag", font=font)[1]
    except Exception:
        line_h = max(16, font_size + 2)

    text_wrap_width = 42
    gt_lines = _wrap_text_by_chars(f"{gt_text_disp}", width=text_wrap_width)
    s2_lines = _wrap_text_by_chars(f"{stage2_text_disp}", width=text_wrap_width)
    px_lines = _wrap_text_by_chars(f"{proxy_text_disp}", width=text_wrap_width)

    metrics_lines = [
        f"Stage2 CER: {stage2_cer:.4f}" if stage2_cer is not None else "Stage2 CER: n/a",
        f"Proxy CER : {proxy_cer:.4f}" if proxy_cer is not None else "Proxy CER : n/a",
    ]

    text_block_h = (
        20
        + (1 + len(gt_lines)) * (line_h + 4)
        + (1 + len(s2_lines)) * (line_h + 4)
        + (1 + len(px_lines)) * (line_h + 4)
        + len(metrics_lines) * (line_h + 4)
        + 20
    )

    panel_w = int(img_panel_w)
    panel_h = int(img_panel_h + gap + text_block_h)
    panel = Image.new("RGB", (panel_w, panel_h), color=(22, 22, 22))

    # paste images
    panel.paste(original_img, (0, top_label_h))
    panel.paste(overlay_img, (original_img.width + gap, top_label_h))

    panel_draw = ImageDraw.Draw(panel)
    panel_draw.text((4, 2), "Original", fill=(230, 230, 230), font=font)
    panel_draw.text((original_img.width + gap + 4, 2), "GT + Predicted Boxes", fill=(230, 230, 230), font=font)

    # text area background
    text_y0 = img_panel_h + gap
    panel_draw.rectangle(
        [0, text_y0, panel_w, panel_h],
        fill=(28, 28, 28)
    )

    # Draw text blocks
    x0 = 10
    y = text_y0 + 10

    y = _draw_multiline_text(
        panel_draw, x0, y,
        label="GT:",
        text=gt_text_disp,
        font=fonts,
        fill=(235, 235, 235),
        label_fill=(120, 255, 140),
        wrap_width=text_wrap_width,
    )

    y = _draw_multiline_text(
        panel_draw, x0, y,
        label="Stage2:",
        text=stage2_text_disp,
        font=fonts,
        fill=(255, 230, 150),
        label_fill=(255, 210, 80),
        wrap_width=text_wrap_width,
    )

    y = _draw_multiline_text(
        panel_draw, x0, y,
        label="Proxy:",
        text=proxy_text_disp,
        font=fonts,
        fill=(180, 220, 255),
        label_fill=(120, 190, 255),
        wrap_width=text_wrap_width,
    )

    for line in metrics_lines:
        panel_draw.text((x0, y), line, fill=(210, 210, 210), font=font)
        y += line_h + 4

    save_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(save_path)


def _align_pred_boxes_to_gt_labels(
    pred_boxes: Sequence[Sequence[float]],
    gt_boxes: Sequence[Sequence[float]],
    gt_label_ids: Sequence[int],
    vocab: VocabManager,
    iou_thr: float,
    orientation: str,
) -> str:
    """Proxy text: assign each predicted box a GT label via best unmatched IoU."""
    if len(pred_boxes) == 0:
        return ""

    pred_sorted = _sort_boxes_reading_order(pred_boxes, orientation)
    gt_arr = [list(map(float, b)) for b in gt_boxes]
    used_gt = set()
    pred_text_ids: List[int] = []

    for pbox in pred_sorted:
        if len(gt_arr) == 0:
            pred_text_ids.append(vocab.unk_id)
            continue

        ious = compute_iou_batch(pbox, gt_arr)
        best_j = int(ious.argmax()) if len(ious) else -1
        best_iou = float(ious[best_j]) if best_j >= 0 else 0.0

        if best_j >= 0 and best_iou >= iou_thr and best_j not in used_gt:
            used_gt.add(best_j)
            pred_text_ids.append(int(gt_label_ids[best_j]))
        else:
            pred_text_ids.append(vocab.unk_id)

    return _decode_text_from_ids(pred_text_ids, vocab)


@dataclass
class EvalSample:
    index: int
    gt_text: str
    stage2_pred_text: str
    stage2_cer: float
    proxy_pred_text: str
    proxy_cer: float
    gt_boxes: int
    pred_boxes: int
    orientation: str


def _resolve_stage1_ckpt(path: Optional[str]) -> Path:
    if path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Stage 1 checkpoint not found: {p}")
        return p

    best_ckpt = CHECKPOINT_DIR / "stage1_detection" / "detector_best.pt"
    if best_ckpt.exists():
        return best_ckpt

    epoch_ckpts = sorted((CHECKPOINT_DIR / "stage1_detection").glob("detector_epoch*.pt"))
    if not epoch_ckpts:
        raise FileNotFoundError(
            f"No Stage 1 checkpoint found under {CHECKPOINT_DIR / 'stage1_detection'}"
        )
    return epoch_ckpts[-1]


def _resolve_stage2_ckpt(path: Optional[str]) -> Path:
    if path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Stage 2 checkpoint not found: {p}")
        return p

    ckpts = sorted((CHECKPOINT_DIR / "stage2_sequence").glob("sequence_epoch*.pt"))
    if not ckpts:
        raise FileNotFoundError(
            f"No Stage 2 checkpoint found under {CHECKPOINT_DIR / 'stage2_sequence'}"
        )
    return ckpts[-1]


def _build_stage2_decoder(
    vocab_size: int,
    state_dict: Dict[str, torch.Tensor],
    use_attn_centroid_boxes: bool = True,
) -> SeqDecoderAttention:
    embed_dim = int(state_dict["embed.weight"].shape[1])
    hidden_dim = int(state_dict["rnn.weight_hh_l0"].shape[1])
    rnn_in = int(state_dict["rnn.weight_ih_l0"].shape[1])
    enc_dim = int(rnn_in - embed_dim)
    use_roi_attention = any(k.startswith("box_head.") for k in state_dict.keys())

    decoder = SeqDecoderAttention(
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        enc_dim=enc_dim,
        num_layers=1,
        init_from_encoder=True,
        sampling_method="argmax",
        use_roi_attention=use_roi_attention,
        use_attn_centroid_boxes=use_attn_centroid_boxes,
    )
    return decoder

def _compute_stage2_loss_metrics(
    encoder: EncoderWrapper,
    decoder: SeqDecoderAttention,
    dataloader: DataLoader,
    vocab: VocabManager,
    roi_sequence_encoder: Optional[ROISequenceEncoder],
    context_encoder: Optional[ROIContextEncoder],
    use_gt_boxes: bool,
    unet: UNet,
    detector: DetectorHead,
    confidence: float,
    top_k: int,
    nms_iou: float,
    reading_order_policy: str,
    ctc_head: Optional[nn.Module] = None,
    aux_ctc_weight: float = 0.0,
) -> Dict[str, float]:
    """
    Compute teacher-forced Stage 2 validation losses using the chosen box source.
    """
    ce_loss = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)

    seq_sum = 0.0
    box_sum = 0.0
    ctc_sum = 0.0
    total_sum = 0.0
    n_batches = 0

    decode_len_sum = 0.0
    gt_len_sum = 0.0
    total_sequences = 0

    ctc_blank_id = vocab.vocab_size  # matches training when ctc_head output = vocab+1
    ctc_loss_fn = nn.CTCLoss(blank=ctc_blank_id, zero_infinity=True) if (ctc_head is not None and aux_ctc_weight > 0.0) else None

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(DEVICE)
            text_ids = batch.get("text_ids", None)
            text_ids_present = batch.get("text_ids_present", None)
            orientations = batch.get("orientations", None)
            boxes_batch = batch.get("boxes", None)

            if text_ids is None or text_ids_present is None:
                continue

            valid_idx = text_ids_present.nonzero(as_tuple=False).squeeze(1)
            if valid_idx.numel() == 0:
                continue

            images = images.index_select(0, valid_idx.to(images.device))
            text_ids = text_ids.to(DEVICE)
            valid_idx_cpu = valid_idx.detach().cpu().tolist()

            if orientations is not None and len(orientations) >= max(valid_idx_cpu) + 1:
                orientations = [orientations[i] for i in valid_idx_cpu]
            else:
                orientations = ["horizontal"] * len(valid_idx_cpu)

            if boxes_batch is not None and len(boxes_batch) >= max(valid_idx_cpu) + 1:
                boxes_batch = [boxes_batch[i] for i in valid_idx_cpu]
            else:
                boxes_batch = [None for _ in range(images.size(0))]

            gt_boxes_for_loss: List[torch.Tensor] = []
            for b in boxes_batch:
                if b is None:
                    gt_boxes_for_loss.append(torch.empty((0, 4), dtype=torch.float32, device=DEVICE))
                else:
                    gt_boxes_for_loss.append(b.to(DEVICE, dtype=torch.float32))

            input_seq = text_ids[:, :-1]
            targets = text_ids[:, 1:]

            if roi_sequence_encoder is not None and context_encoder is not None:
                boxes_for_encoder = _build_boxes_for_stage2_validation(
                    images=images,
                    boxes_batch=boxes_batch,
                    orientations=orientations,
                    use_gt_boxes=use_gt_boxes,
                    unet=unet,
                    detector=detector,
                    confidence=confidence,
                    top_k=top_k,
                    nms_iou=nms_iou,
                    reading_order_policy=reading_order_policy,
                )

                feats_2d = encoder(images, return_2d=True)
                enc_roi, roi_mask = roi_sequence_encoder(feats_2d, boxes_for_encoder, image_size=IMAGE_SIZE)
                enc_outputs, enc_mask = context_encoder(enc_roi, roi_mask)

                # decode-length stats
                for i in range(len(boxes_for_encoder)):
                    decode_len = int(boxes_for_encoder[i].size(0)) + 2
                    decode_len_sum += decode_len
                    gt_len_sum += float((text_ids[i] != vocab.pad_id).sum().item())
                    total_sequences += 1
            else:
                orientation = orientations[0] if orientations else "horizontal"
                enc_outputs, enc_mask = encoder(images, orientation=orientation)

            decoder_output = decoder(
                input_seq=input_seq,
                enc_outputs=enc_outputs,
                enc_mask=enc_mask,
                targets=targets,
                teacher_forcing_ratio=1.0,
                eos_id=vocab.eos_id,
                image_size=IMAGE_SIZE,
            )

            predicted_boxes = None
            if len(decoder_output) == 4:
                logits, _, _, predicted_boxes = decoder_output
            else:
                logits, _, _ = decoder_output

            bsz, _, vocab_dim = logits.shape
            loss_seq = ce_loss(logits.reshape(-1, vocab_dim), targets.reshape(-1))

            if USE_ROI_ATTENTION and predicted_boxes is not None and len(gt_boxes_for_loss) == bsz:
                loss_box = compute_roi_box_loss(
                    predicted_boxes,
                    gt_boxes_for_loss,
                    reduction="mean",
                    iou_weight=0.0,
                    use_x_only=True,
                    coord_scale=float(IMAGE_SIZE[1]),
                )
            else:
                loss_box = torch.tensor(0.0, device=DEVICE)

            if ctc_loss_fn is not None:
                # ctc_loss_fn only exists when ctc_head is configured.
                ctc_head_module = cast(nn.Module, ctc_head)
                ctc_logits = ctc_head_module(enc_outputs)
                ctc_log_probs = ctc_logits.float().log_softmax(dim=-1).permute(1, 0, 2).contiguous()
                ctc_input_lengths = enc_mask.sum(dim=1).clamp(min=1).to(dtype=torch.long)

                targets_ctc = []
                target_lengths = []
                keep_indices = []

                for i in range(text_ids.size(0)):
                    ids = text_ids[i]
                    ids = ids[ids != vocab.pad_id]
                    if ids.numel() == 0:
                        continue
                    if ids[0].item() == vocab.sos_id:
                        ids = ids[1:]
                    if ids.numel() > 0 and ids[-1].item() == vocab.eos_id:
                        ids = ids[:-1]
                    if ids.numel() == 0:
                        continue
                    if ids.numel() > int(ctc_input_lengths[i].item()):
                        continue

                    targets_ctc.append(ids)
                    target_lengths.append(ids.numel())
                    keep_indices.append(i)

                if len(targets_ctc) == 0:
                    loss_ctc = torch.tensor(0.0, device=DEVICE)
                else:
                    targets_concat = torch.cat(targets_ctc).to(DEVICE)
                    target_lengths = torch.tensor(target_lengths, dtype=torch.long, device=DEVICE)
                    keep_indices = torch.tensor(keep_indices, dtype=torch.long, device=DEVICE)

                    loss_ctc = ctc_loss_fn(
                        ctc_log_probs[:, keep_indices, :],
                        targets_concat,
                        ctc_input_lengths[keep_indices],
                        target_lengths,
                    )
            else:
                loss_ctc = torch.tensor(0.0, device=DEVICE)

            loss_total = loss_seq + aux_ctc_weight * loss_ctc + ROI_BOX_LOSS_WEIGHT * loss_box

            seq_sum += float(loss_seq.item())
            box_sum += float(loss_box.item())
            ctc_sum += float(loss_ctc.item())
            total_sum += float(loss_total.item())
            n_batches += 1

    denom = max(1, n_batches)
    seq_denom = max(1, total_sequences)

    return {
        "mean_total_loss": total_sum / denom,
        "mean_seq_loss": seq_sum / denom,
        "mean_ctc_loss": ctc_sum / denom,
        "mean_weighted_ctc_loss": aux_ctc_weight * (ctc_sum / denom),
        "mean_box_loss": box_sum / denom,
        "mean_weighted_box_loss": ROI_BOX_LOSS_WEIGHT * (box_sum / denom),
        "mean_decode_len": decode_len_sum / seq_denom,
        "mean_gt_len": gt_len_sum / seq_denom,
        "num_batches": n_batches,
    }

def validate_stage2_e2e(
    stage1_ckpt_path: Optional[str],
    stage2_ckpt_path: Optional[str],
    split: str,
    num_samples: Optional[int],
    batch_size: int,
    confidence: float,
    top_k: int,
    nms_iou: float,
    det_iou_thr: float,
    text_iou_thr: float,
    max_decode_len: int,
    job_id: Optional[str],
    save_plots: bool,
    num_plot_samples: int,
    plot_font_path: Optional[str],
    plot_font_size: int,
    stage2_decode_boxes: str,
    reading_order_policy: str,
) -> Dict[str, Any]:
    stage1_ckpt = _resolve_stage1_ckpt(stage1_ckpt_path)
    stage2_ckpt = _resolve_stage2_ckpt(stage2_ckpt_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    resolved_job_id = job_id or os.environ.get("SLURM_JOB_ID") or os.environ.get("JOB_ID")
    run_tag = f"{timestamp}_job{resolved_job_id}" if resolved_job_id else timestamp
    out_dir = CHECKPOINT_DIR / "stage2_validation" / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"

    ann_files = sorted(list((Path(DATA_DIR) / "annotations").glob("*.json")))
    if len(ann_files) == 0:
        raise FileNotFoundError(f"No annotation files found in {Path(DATA_DIR) / 'annotations'}")

    vocab = VocabManager.from_annotations(ann_files)
    pad_id = vocab.pad_id

    dataset = KuzushijiDataset(
        Path(DATA_DIR),
        vocab=vocab,
        use_sequences=True,
        resize=IMAGE_SIZE,
        split=split,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: collate_fn(b, pad_id),
        pin_memory=(DEVICE == "cuda"),
    )

    # Stage 1 models
    unet = UNet(in_channels=3, base_features=32).to(DEVICE)
    detector = DetectorHead(
        in_ch=32,
        num_classes=vocab.vocab_size,
        dropout_rate=0.0,
        predict_classes=False,
    ).to(DEVICE)

    ckpt1 = torch.load(stage1_ckpt, map_location=DEVICE)
    required_stage1 = {"unet_state_dict", "detector_state_dict"}
    missing_stage1 = required_stage1 - set(ckpt1.keys())
    if missing_stage1:
        raise KeyError(f"Stage 1 checkpoint missing keys: {sorted(missing_stage1)}")

    unet.load_state_dict(ckpt1["unet_state_dict"], strict=True)
    detector.load_state_dict(ckpt1["detector_state_dict"], strict=True)
    unet.eval()
    detector.eval()

    # Stage 2 models
    encoder = EncoderWrapper(backbone=unet, in_channels=32, enc_dim=256).to(DEVICE)
    roi_sequence_encoder: Optional[ROISequenceEncoder] = None
    context_encoder: Optional[ROIContextEncoder] = None

    ckpt2 = torch.load(stage2_ckpt, map_location=DEVICE)
    stage2_use_gt_boxes_from_ckpt = ckpt2.get("stage2_use_gt_boxes", None)
    required_stage2 = {"encoder_state_dict", "decoder_state_dict"}
    missing_stage2 = required_stage2 - set(ckpt2.keys())
    if missing_stage2:
        raise KeyError(f"Stage 2 checkpoint missing keys: {sorted(missing_stage2)}")

    stage2_vocab = ckpt2.get("vocab", None)
    vocab_warning = None
    if isinstance(stage2_vocab, dict) and len(stage2_vocab) != vocab.vocab_size:
        vocab_warning = (
            f"Stage 2 checkpoint vocab size ({len(stage2_vocab)}) differs from current vocab size "
            f"({vocab.vocab_size})."
        )
        print(f"[WARN] {vocab_warning}")

    decoder = _build_stage2_decoder(
        vocab_size=vocab.vocab_size,
        state_dict=ckpt2["decoder_state_dict"],
        use_attn_centroid_boxes=bool(ckpt2.get("stage2_use_attn_centroid_boxes", True)),
            ).to(DEVICE)
    encoder.load_state_dict(ckpt2["encoder_state_dict"], strict=True)


    ctc_head = None
    if "ctc_head_state_dict" in ckpt2:
        # infer output dim from checkpoint
        ctc_out_dim = int(ckpt2["ctc_head_state_dict"]["weight"].shape[0])
        ctc_in_dim = int(ckpt2["ctc_head_state_dict"]["weight"].shape[1])
        ctc_head = nn.Linear(ctc_in_dim, ctc_out_dim).to(DEVICE)
        ctc_head.load_state_dict(ckpt2["ctc_head_state_dict"], strict=True)
        ctc_head.eval()

    if "roi_sequence_encoder_state_dict" in ckpt2 and "context_encoder_state_dict" in ckpt2:
        roi_pool_size = tuple(ckpt2.get("roi_pool_size", ROI_POOL_SIZE))
        roi_embed_dim = int(ckpt2.get("roi_embed_dim", ROI_EMBED_DIM))
        context_hidden_dim = int(ckpt2.get("context_hidden_dim", CONTEXT_HIDDEN_DIM))

        roi_sequence_encoder = ROISequenceEncoder(
            in_dim=256,
            roi_size=roi_pool_size,
            out_dim=roi_embed_dim,
        ).to(DEVICE)
        context_encoder = ROIContextEncoder(
            in_dim=roi_embed_dim,
            hidden_dim=context_hidden_dim,
            out_dim=context_hidden_dim,
        ).to(DEVICE)
        assert roi_sequence_encoder is not None and context_encoder is not None
        try:
            roi_sequence_encoder.load_state_dict(ckpt2["roi_sequence_encoder_state_dict"], strict=True)
            context_encoder.load_state_dict(ckpt2["context_encoder_state_dict"], strict=True)
        except RuntimeError as exc:
            print(f"[WARN] ROI module strict load failed, retrying non-strict: {exc}")
            roi_sequence_encoder.load_state_dict(ckpt2["roi_sequence_encoder_state_dict"], strict=False)
            context_encoder.load_state_dict(ckpt2["context_encoder_state_dict"], strict=False)

    decoder.load_state_dict(ckpt2["decoder_state_dict"], strict=True)
    encoder.eval()
    decoder.eval()
    if roi_sequence_encoder is not None:
        roi_sequence_encoder.eval()
    if context_encoder is not None:
        context_encoder.eval()

    loss_metrics_pred = None
    loss_metrics_gt = None

    if stage2_decode_boxes in ("pred", "both"):
        loss_metrics_pred = _compute_stage2_loss_metrics(
            encoder=encoder,
            decoder=decoder,
            dataloader=dataloader,
            vocab=vocab,
            roi_sequence_encoder=roi_sequence_encoder,
            context_encoder=context_encoder,
            use_gt_boxes=False,
            unet=unet,
            detector=detector,
            confidence=confidence,
            top_k=top_k,
            nms_iou=nms_iou,
            reading_order_policy=reading_order_policy,
            ctc_head=ctc_head,
            aux_ctc_weight=float(ckpt2.get("stage2_aux_ctc_weight", 0.0)),
        )

    if stage2_decode_boxes in ("gt", "both"):
        loss_metrics_gt = _compute_stage2_loss_metrics(
            encoder=encoder,
            decoder=decoder,
            dataloader=dataloader,
            vocab=vocab,
            roi_sequence_encoder=roi_sequence_encoder,
            context_encoder=context_encoder,
            use_gt_boxes=True,
            unet=unet,
            detector=detector,
            confidence=confidence,
            top_k=top_k,
            nms_iou=nms_iou,
            reading_order_policy=reading_order_policy,
            ctc_head=ctc_head,
            aux_ctc_weight=float(ckpt2.get("stage2_aux_ctc_weight", 0.0)),
        )

    if stage2_decode_boxes == "pred":
        loss_metrics = loss_metrics_pred
    elif stage2_decode_boxes == "gt":
        loss_metrics = loss_metrics_gt
    else:
        loss_metrics = {
            "pred": loss_metrics_pred,
            "gt": loss_metrics_gt,
        }

    # Collect metrics
    total_samples = 0
    active_decode_modes = ["pred", "gt"] if stage2_decode_boxes == "both" else [stage2_decode_boxes]
    primary_decode_mode = "pred" if stage2_decode_boxes == "both" else stage2_decode_boxes
    stage2_stats: Dict[str, Dict[str, float]] = {
        mode: {"exact": 0.0, "cer_sum": 0.0} for mode in active_decode_modes
    }
    proxy_exact = 0
    proxy_cer_sum = 0.0

    all_gt_boxes: List[List[List[float]]] = []
    all_pred_boxes: List[List[List[float]]] = []

    sample_rows: List[EvalSample] = []

    pbar = tqdm(dataloader, desc="Stage2 E2E Validation")
    plots_written = 0

    def _decode_stage2_text(
        img_i: torch.Tensor,
        decode_boxes_i: Sequence[Sequence[float]],
        orientation: str,
    ) -> Tuple[str, int]:
        if roi_sequence_encoder is not None and context_encoder is not None:
            decode_boxes_tensor = (
                torch.tensor(decode_boxes_i, dtype=torch.float32, device=DEVICE)
                if len(decode_boxes_i) > 0
                else torch.empty((0, 4), dtype=torch.float32, device=DEVICE)
            )
            feat_2d_i = encoder(img_i, return_2d=True)
            roi_seq_i, roi_mask_i = roi_sequence_encoder(
                feat_2d_i,
                [decode_boxes_tensor],
                image_size=IMAGE_SIZE,
            )
            enc_seq, enc_mask = context_encoder(roi_seq_i, roi_mask_i)
        else:
            enc_seq, enc_mask = encoder(img_i, orientation=orientation)

        decode_len = len(decode_boxes_i) + 2
        decode_len = max(2, min(max_decode_len, decode_len))

        decoder_out = decoder(
            input_seq=None,
            enc_outputs=enc_seq,
            enc_mask=enc_mask,
            targets=None,
            teacher_forcing_ratio=0.0,
            sos_id=vocab.sos_id,
            eos_id=vocab.eos_id,
            max_len=decode_len,
            image_size=IMAGE_SIZE,
        )

        if len(decoder_out) == 4:
            logits, _, _, _ = decoder_out
        else:
            logits, _, _ = decoder_out

        pred_ids = logits[0].argmax(dim=-1).detach().cpu().tolist()
        pred_ids = _truncate_at_eos(pred_ids, vocab.eos_id)
        pred_text = _decode_text_from_ids(pred_ids, vocab)
        return pred_text, len(pred_ids)

    with torch.no_grad():
        for batch in pbar:
            images = batch["image"].to(DEVICE)
            boxes_batch = batch.get("boxes", [])
            labels_batch = batch.get("labels", [])
            orientations = batch.get("orientations", ["horizontal"] * images.size(0))
            text_ids = batch["text_ids"]
            text_ids_present = batch.get("text_ids_present", None)

            if text_ids is None or text_ids_present is None:
                continue

            valid_idx = text_ids_present.nonzero(as_tuple=False).squeeze(1)
            if valid_idx.numel() == 0:
                continue

            # Keep alignment with collate behavior.
            images = images.index_select(0, valid_idx.to(images.device))
            text_ids = text_ids.to(DEVICE)
            valid_idx_list = valid_idx.detach().cpu().tolist()

            if len(orientations) >= max(valid_idx_list) + 1:
                orientations = [orientations[i] for i in valid_idx_list]
            else:
                orientations = ["horizontal"] * len(valid_idx_list)

            gt_boxes_valid = [boxes_batch[i] for i in valid_idx_list]
            gt_labels_valid = [labels_batch[i] for i in valid_idx_list]

            features = unet(images)
            det_out = detector(features)
            heat_probs = torch.sigmoid(det_out["heatmap"])
            bbox_reg = det_out["bbox"]
            _, _, hf, wf = features.shape

            for i in range(images.size(0)):
                sample_index = total_samples
                orientation = orientations[i] if i < len(orientations) else "horizontal"
                gt_boxes_i = gt_boxes_valid[i]
                gt_labels_i = gt_labels_valid[i]

                gt_boxes_list = gt_boxes_i.detach().cpu().numpy().tolist() if gt_boxes_i is not None else []
                gt_label_ids = gt_labels_i.detach().cpu().tolist() if gt_labels_i is not None else []

                pred_boxes_i, _, _ = extract_boxes_from_heatmap(
                    heatmap_probs=heat_probs[i:i + 1],
                    bbox_reg=bbox_reg[i:i + 1],
                    confidence_thresh=confidence,
                    output_size=(hf, wf),
                    image_size=IMAGE_SIZE,
                    top_k=top_k,
                    nms_iou=nms_iou,
                    min_box_size=4.0,
                    debug=False,
                )

                sort_orientation = _resolve_sort_orientation(orientation, pred_boxes_i, reading_order_policy)
                pred_boxes_i = _sort_boxes_reading_order(pred_boxes_i, sort_orientation)

                all_gt_boxes.append(gt_boxes_list)
                all_pred_boxes.append(pred_boxes_i)

                # Ground-truth text from sequence ids.
                gt_seq = text_ids[i].detach().cpu().tolist()
                gt_text = _decode_text_from_ids(gt_seq, vocab)

                img_i = images[i:i + 1]

                stage2_outputs: Dict[str, Tuple[str, float, int]] = {}
                denom = max(1, len(gt_text))

                for mode in active_decode_modes:
                    if mode == "gt":
                        gt_sort_orientation = _resolve_sort_orientation(orientation, gt_boxes_list, reading_order_policy)
                        decode_boxes_i = _sort_boxes_reading_order(gt_boxes_list, gt_sort_orientation)
                        decode_orientation = gt_sort_orientation
                    else:
                        decode_boxes_i = pred_boxes_i
                        decode_orientation = sort_orientation

                    mode_text, mode_pred_len = _decode_stage2_text(img_i, decode_boxes_i, decode_orientation)
                    mode_dist = _edit_distance(mode_text, gt_text)
                    mode_cer = mode_dist / denom

                    stage2_stats[mode]["cer_sum"] += mode_cer
                    stage2_stats[mode]["exact"] += float(mode_text == gt_text)

                    # optional length stats
                    stage2_stats[mode].setdefault("pred_len_sum", 0.0)
                    stage2_stats[mode].setdefault("gt_len_sum", 0.0)
                    stage2_stats[mode]["pred_len_sum"] += float(mode_pred_len)
                    stage2_stats[mode]["gt_len_sum"] += float(len(gt_text))

                    stage2_outputs[mode] = (mode_text, mode_cer, mode_pred_len)

                stage2_pred_text, stage2_cer, stage2_pred_len = stage2_outputs[primary_decode_mode]

                # Detection-to-text proxy (box matching -> labels).
                proxy_pred_text = _align_pred_boxes_to_gt_labels(
                    pred_boxes=pred_boxes_i,
                    gt_boxes=gt_boxes_list,
                    gt_label_ids=gt_label_ids,
                    vocab=vocab,
                    iou_thr=text_iou_thr,
                    orientation=sort_orientation,
                )

                proxy_dist = _edit_distance(proxy_pred_text, gt_text)
                proxy_cer = proxy_dist / denom

                total_samples += 1
                proxy_cer_sum += proxy_cer
                proxy_exact += int(proxy_pred_text == gt_text)

                sample_rows.append(
                    EvalSample(
                        index=sample_index,
                        gt_text=gt_text,
                        stage2_pred_text=stage2_pred_text,
                        stage2_cer=stage2_cer,
                        proxy_pred_text=proxy_pred_text,
                        proxy_cer=proxy_cer,
                        gt_boxes=len(gt_boxes_list),
                        pred_boxes=len(pred_boxes_i),
                        orientation=orientation,
                    )
                )

                if save_plots and plots_written < num_plot_samples:
                    plot_name = f"sample_{sample_index:05d}.png"
                    _draw_stage2_box_character_plot(
                        image_tensor=images[i],
                        gt_boxes=gt_boxes_list,
                        pred_boxes=pred_boxes_i,
                        stage2_pred_text=stage2_pred_text,
                        proxy_pred_text=proxy_pred_text,
                        gt_text=gt_text,
                        orientation=sort_orientation,
                        save_path=plots_dir / plot_name,
                        font_path=plot_font_path,
                        font_size=plot_font_size,
                        stage2_cer=stage2_cer,
                        proxy_cer=proxy_cer,
                        show_box_indices=False,
                        max_index_labels=15,
                        max_text_chars=160,
                    )
                    plots_written += 1

                if num_samples is not None and total_samples >= num_samples:
                    break

            pbar.set_postfix({
                "samples": total_samples,
                "stage2_cer": f"{(stage2_stats[primary_decode_mode]['cer_sum'] / max(1, total_samples)):.3f}",
                "proxy_cer": f"{(proxy_cer_sum / max(1, total_samples)):.3f}",
            })

            if num_samples is not None and total_samples >= num_samples:
                break

    det_metrics = compute_detection_metrics(all_gt_boxes, all_pred_boxes, iou_threshold=det_iou_thr)

    metrics: Dict[str, Any] = {
            "run": {
                "stage1_checkpoint": str(stage1_ckpt),
                "stage2_checkpoint": str(stage2_ckpt),
                "split": split,
                "num_samples": total_samples,
                "device": DEVICE,
                "job_id": resolved_job_id,
            },
            "configuration": {
                "confidence": confidence,
                "top_k": top_k,
                "nms_iou": nms_iou,
                "detection_iou_threshold": det_iou_thr,
                "text_alignment_iou_threshold": text_iou_thr,
                "max_decode_len": max_decode_len,
                "reading_order_policy": reading_order_policy,
                "stage2_use_gt_boxes_from_checkpoint": stage2_use_gt_boxes_from_ckpt,
                "stage2_decode_boxes": stage2_decode_boxes,
            },
            "architecture_notes": {
                "stage2_checkpoint_uses_roi_sequence": bool(
                    roi_sequence_encoder is not None and context_encoder is not None
                ),
                "stage2_decoder_consumes_box_sequences": bool(
                    roi_sequence_encoder is not None and context_encoder is not None
                ),
                "note": (
                    "When ROI sequence modules are present, Stage 2 consumes GT or predicted boxes "
                    "as an ordered ROI token sequence before contextual encoding and decoding. "
                    "Legacy fallback without ROI modules uses the older encoder sequence path."
                ),
                "current_decode_box_source": stage2_decode_boxes,
            },
            "stage1_detection": det_metrics,
            "stage2_sequence": {
                "exact_match": stage2_stats[primary_decode_mode]["exact"] / max(1, total_samples),
                "mean_cer": stage2_stats[primary_decode_mode]["cer_sum"] / max(1, total_samples),
                "mean_pred_len": stage2_stats[primary_decode_mode].get("pred_len_sum", 0.0) / max(1, total_samples),
                "mean_gt_len": stage2_stats[primary_decode_mode].get("gt_len_sum", 0.0) / max(1, total_samples),
            },
            "stage2_validation_losses": loss_metrics,
            "detection_to_text_proxy": {
                "exact_match": proxy_exact / max(1, total_samples),
                "mean_cer": proxy_cer_sum / max(1, total_samples),
            },
            "warnings": {
                "vocab_mismatch": vocab_warning,
                "stage2_box_source_parity": None,
            },
            "artifacts": {
                "plots_dir": str(plots_dir) if save_plots else None,
                "num_plots_written": plots_written,
            },
        }
    if stage2_decode_boxes == "both":
        metrics["stage2_sequence_compare"] = {
            "pred": {
                "exact_match": stage2_stats["pred"]["exact"] / max(1, total_samples),
                "mean_cer": stage2_stats["pred"]["cer_sum"] / max(1, total_samples),
                "mean_pred_len": stage2_stats["pred"].get("pred_len_sum", 0.0) / max(1, total_samples),
                "mean_gt_len": stage2_stats["pred"].get("gt_len_sum", 0.0) / max(1, total_samples),
            },
            "gt": {
                "exact_match": stage2_stats["gt"]["exact"] / max(1, total_samples),
                "mean_cer": stage2_stats["gt"]["cer_sum"] / max(1, total_samples),
                "mean_pred_len": stage2_stats["gt"].get("pred_len_sum", 0.0) / max(1, total_samples),
                "mean_gt_len": stage2_stats["gt"].get("gt_len_sum", 0.0) / max(1, total_samples),
            },
        }

    parity_payload = {
        "stage2_use_gt_boxes": stage2_use_gt_boxes_from_ckpt,
        "stage2_decode_boxes": stage2_decode_boxes,
        "reading_order_policy": reading_order_policy,
        "stage2_checkpoint": str(stage2_ckpt),
        "stage1_checkpoint": str(stage1_ckpt),
        "job_id": resolved_job_id,
    }
    parity_warning = None
    if stage2_use_gt_boxes_from_ckpt is None:
        parity_warning = "stage2_use_gt_boxes missing in stage2 checkpoint metadata"
    elif bool(stage2_use_gt_boxes_from_ckpt) and stage2_decode_boxes == "pred":
        parity_warning = "checkpoint trained with GT boxes but validator decoding uses predicted boxes"
    elif (not bool(stage2_use_gt_boxes_from_ckpt)) and stage2_decode_boxes == "gt":
        parity_warning = "checkpoint trained with predicted boxes but validator decoding uses GT boxes"

    parity_payload["warning"] = parity_warning
    metrics["warnings"]["stage2_box_source_parity"] = parity_warning

    parity_path = out_dir / "stage2_mode_parity.json"
    with open(parity_path, "w", encoding="utf-8") as f:
        json.dump(parity_payload, f, ensure_ascii=False, indent=2)

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    samples_path = out_dir / "samples.jsonl"
    with open(samples_path, "w", encoding="utf-8") as f:
        for row in sample_rows:
            f.write(json.dumps(row.__dict__, ensure_ascii=False) + "\n")

    print("\n" + "=" * 72)
    print("STAGE2 END-TO-END VALIDATION")
    print("=" * 72)
    print(f"samples: {total_samples}")
    print(
        "stage2 sequence -> "
        f"exact={metrics['stage2_sequence']['exact_match']:.4f}, "
        f"cer={metrics['stage2_sequence']['mean_cer']:.4f}, "
        f"pred_len={metrics['stage2_sequence']['mean_pred_len']:.2f}, "
        f"gt_len={metrics['stage2_sequence']['mean_gt_len']:.2f}"
    )
    if stage2_decode_boxes == "both":
        print(
            "stage2 compare -> "
            f"pred_cer={metrics['stage2_sequence_compare']['pred']['mean_cer']:.4f}, "
            f"gt_cer={metrics['stage2_sequence_compare']['gt']['mean_cer']:.4f}, "
            f"pred_len(pred)={metrics['stage2_sequence_compare']['pred']['mean_pred_len']:.2f}, "
            f"pred_len(gt)={metrics['stage2_sequence_compare']['gt']['mean_pred_len']:.2f}"
        )

    if stage2_decode_boxes == "both":
        print(
            "stage2 losses(pred) -> "
            f"total={metrics['stage2_validation_losses']['pred']['mean_total_loss']:.4f}, "
            f"seq={metrics['stage2_validation_losses']['pred']['mean_seq_loss']:.4f}, "
            f"ctc={metrics['stage2_validation_losses']['pred']['mean_ctc_loss']:.4f}, "
            f"box={metrics['stage2_validation_losses']['pred']['mean_box_loss']:.4f}"
        )
        print(
            "stage2 losses(gt) -> "
            f"total={metrics['stage2_validation_losses']['gt']['mean_total_loss']:.4f}, "
            f"seq={metrics['stage2_validation_losses']['gt']['mean_seq_loss']:.4f}, "
            f"ctc={metrics['stage2_validation_losses']['gt']['mean_ctc_loss']:.4f}, "
            f"box={metrics['stage2_validation_losses']['gt']['mean_box_loss']:.4f}"
        )
    else:
        print(
            "stage2 losses -> "
            f"total={metrics['stage2_validation_losses']['mean_total_loss']:.4f}, "
            f"seq={metrics['stage2_validation_losses']['mean_seq_loss']:.4f}, "
            f"ctc={metrics['stage2_validation_losses']['mean_ctc_loss']:.4f}, "
            f"box={metrics['stage2_validation_losses']['mean_box_loss']:.4f}, "
            f"wctc={metrics['stage2_validation_losses']['mean_weighted_ctc_loss']:.4f}, "
            f"wbox={metrics['stage2_validation_losses']['mean_weighted_box_loss']:.4f}"
        )

    print(
        "box-text proxy -> "
        f"exact={metrics['detection_to_text_proxy']['exact_match']:.4f}, "
        f"cer={metrics['detection_to_text_proxy']['mean_cer']:.4f}"
    )
    print(
        "stage1 detection -> "
        f"precision={det_metrics['precision']:.4f}, "
        f"recall={det_metrics['recall']:.4f}, "
        f"f1={det_metrics['f1']:.4f}"
    )
    print(f"saved metrics: {metrics_path}")
    print(f"saved mode parity: {parity_path}")
    print(f"saved samples: {samples_path}")
    if save_plots:
        print(f"saved plots: {plots_dir} ({plots_written})")
    if parity_warning:
        print(f"[WARN] {parity_warning}")
    print("=" * 72 + "\n")

    return metrics


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate Stage 2 end-to-end with Stage 1 predictions")
    p.add_argument("--stage1_ckpt", type=str, default=None)
    p.add_argument("--stage2_ckpt", type=str, default=None)
    p.add_argument("--split", type=str, default="val", choices=["train", "val"])
    p.add_argument("--num_samples", type=int, default=0, help="0 means full split")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--confidence", type=float, default=0.5)
    p.add_argument("--top_k", type=int, default=300)
    p.add_argument("--nms_iou", type=float, default=0.5)
    p.add_argument("--det_iou_thr", type=float, default=0.5)
    p.add_argument("--text_iou_thr", type=float, default=0.5)
    p.add_argument("--max_decode_len", type=int, default=256)
    p.add_argument("--job_id", type=str, default=None)
    p.add_argument("--save_plots", action="store_true", help="Save per-sample box-character visualizations")
    p.add_argument("--num_plot_samples", type=int, default=40, help="How many sample plots to save")
    p.add_argument("--plot_font_path", type=str, default=None, help="Optional path to a TTF/TTC font with Japanese glyph support")
    p.add_argument("--plot_font_size", type=int, default=12, help="Font size for plot labels")
    p.add_argument("--stage2_decode_boxes", type=str, default="both", choices=["pred", "gt", "both"], help="Which boxes Stage 2 decoder should consume during free decoding")
    p.add_argument("--reading_order_policy", type=str, default=STAGE2_READING_ORDER_POLICY, choices=["annotation", "inferred", "auto"], help="How to choose box sorting orientation")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    validate_stage2_e2e(
        stage1_ckpt_path=args.stage1_ckpt,
        stage2_ckpt_path=args.stage2_ckpt,
        split=args.split,
        num_samples=None if args.num_samples == 0 else args.num_samples,
        batch_size=args.batch_size,
        confidence=args.confidence,
        top_k=args.top_k,
        nms_iou=args.nms_iou,
        det_iou_thr=args.det_iou_thr,
        text_iou_thr=args.text_iou_thr,
        max_decode_len=args.max_decode_len,
        job_id=args.job_id,
        save_plots=args.save_plots,
        num_plot_samples=args.num_plot_samples,
        plot_font_path=args.plot_font_path,
        plot_font_size=args.plot_font_size,
        stage2_decode_boxes=args.stage2_decode_boxes,
        reading_order_policy=args.reading_order_policy,
    )
