"""Shared training helper functions for stage training and Optuna runs."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from config import STAGE1_AVG_GT_PER_IMAGE, STAGE1_BBOX_WEIGHT, STAGE1_DIOU_WEIGHT, STAGE1_FOCAL_POS_THRESHOLD, STAGE1_HEATMAP_SIGMA, STAGE1_SIGMA_FLOOR, STAGE1_SIGMA_CEIL, STAGE1_SIGMA_SCALE, STAGE1_FOCAL_ALPHA, STAGE1_FOCAL_GAMMA, STAGE1_POS_WEIGHT
from utils.detection_utils import build_detection_targets
from utils.focal_loss import focal_loss_heatmap


def prune_existing_checkpoints(ckpt_dir: Path, old_name: str = "checkpoint_old.pt") -> None:
    """At start: keep the newest non-best checkpoint, rename it to old_name."""
    ckpts = [p for p in ckpt_dir.glob("*.pt") if not p.name.endswith("_best.pt")]
    ckpts = sorted(ckpts, key=lambda p: p.stat().st_mtime)
    if not ckpts:
        return
    newest = ckpts[-1]
    for p in ckpts[:-1]:
        try:
            p.unlink()
        except Exception as exc:  # best-effort cleanup
            print(f"Warning: could not delete {p}: {exc}")
    target = ckpt_dir / old_name
    if newest == target:
        return
    if target.exists():
        try:
            target.unlink()
        except Exception:
            pass
    try:
        newest.rename(target)
        print(f"Preserved latest checkpoint as {target.name}")
    except Exception as exc:
        print(f"Warning: could not rename {newest} to {target}: {exc}")


def prune_to_keep_last_n(ckpt_dir: Path, keep: int = 2, exclude: str = "checkpoint_old.pt") -> None:
    """Keep only the newest N non-best checkpoints (excluding preserved files)."""
    ckpts = [p for p in ckpt_dir.glob("*.pt") if p.name != exclude and not p.name.endswith("_best.pt")]
    ckpts = sorted(ckpts, key=lambda p: p.stat().st_mtime)
    if len(ckpts) <= keep:
        return
    to_delete = ckpts[:-keep]
    for p in to_delete:
        try:
            p.unlink()
        except Exception as exc:
            print(f"Warning: could not delete old checkpoint {p}: {exc}")

def check_backbone_type(resume_ckpt: str | None, backbone_type: str) -> bool:
    """We have (for now) 2 different backbone types (ResNet18 and EfficientNet-B0). 
    If resuming from a checkpoint, we must ensure the backbone type matches the checkpoint. 
    If it doesn't, we rebuild the model with the checkpoint's backbone type to avoid shape mismatches."""
    if resume_ckpt is not None:
        _peek = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
        ckpt_backbone_type = _peek.get("backbone_type", None)
        if ckpt_backbone_type is not None and ckpt_backbone_type != backbone_type:
            print(
                f"⚠  Checkpoint backbone_type='{ckpt_backbone_type}' differs from "
                f"config BACKBONE_TYPE='{backbone_type}'. "
                f"Building '{ckpt_backbone_type}' to match the checkpoint."
            )
            return False
        del _peek
    return True

def data_info_startup(images, features, outputs, gt_heat, gt_bbox, gt_bbox_mask, Hf, Wf):
    print("\n[TRAIN DEBUG]")
    print("images.shape:", tuple(images.shape))
    print("features.shape:", tuple(features.shape))
    print("output_size:", (Hf, Wf))
    print("image_size:", tuple(images.shape[-2:]))
    print("gt_heat min/max/mean:",
        gt_heat.min().item(),
        gt_heat.max().item(),
        gt_heat.mean().item())
    print("num bbox supervised cells:", int(gt_bbox_mask.sum().item()))

    if gt_bbox_mask.sum() > 0:
        gt_bbox_pos = gt_bbox.permute(0, 2, 3, 1)[gt_bbox_mask]
        pred_bbox_pos = outputs["bbox"].permute(0, 2, 3, 1)[gt_bbox_mask]

        print("gt_bbox mean [dx,dy,bw,bh]:",
            gt_bbox_pos.mean(dim=0).detach().cpu().tolist())
        print("gt_bbox min  [dx,dy,bw,bh]:",
            gt_bbox_pos.min(dim=0).values.detach().cpu().tolist())
        print("gt_bbox max  [dx,dy,bw,bh]:",
            gt_bbox_pos.max(dim=0).values.detach().cpu().tolist())

        print("pred_bbox mean [dx,dy,bw,bh]:",
            pred_bbox_pos.mean(dim=0).detach().cpu().tolist())
        print("pred_bbox min  [dx,dy,bw,bh]:",
            pred_bbox_pos.min(dim=0).values.detach().cpu().tolist())
        print("pred_bbox max  [dx,dy,bw,bh]:",
            pred_bbox_pos.max(dim=0).values.detach().cpu().tolist())

def collate_fn(batch, pad_id):
    """
    batch: list of samples from KuzushijiDataset
    Each sample contains: image (Tensor), text_ids (optional Tensor), text_length, boxes, labels
    Returns dict with images, text_ids_padded, text_lengths, boxes, labels

    NOTE: If a sample lacks text_ids, we still include boxes/labels for that image,
    but set its text_ids to None. This ensures 1-to-1 correspondence between batch indices.
    """
    images = torch.stack([b["image"] for b in batch], dim=0)

    # Boxes and labels for detection
    boxes = [b.get("boxes", torch.empty((0, 4))) for b in batch]
    labels = [b.get("labels", torch.empty((0,), dtype=torch.long)) for b in batch]
    orientations = [b.get("orientation", "horizontal") for b in batch]
    image_stems = [b.get("image_stem", "") for b in batch]

    # Text sequences - keep batch alignment by inserting a short all-pad placeholder
    # for samples without transcripts. This preserves B-way correspondence between
    # images, boxes, labels, and decoder supervision tensors.
    text_ids_list = []
    text_lengths_list = []
    text_ids_present = []
    for b in batch:
        has_text_ids = "text_ids" in b and b["text_ids"] is not None
        text_ids_present.append(has_text_ids)
        if has_text_ids:
            text_ids_list.append(b["text_ids"])
            text_lengths_list.append(len(b["text_ids"]))
        else:
            text_ids_list.append(torch.full((2,), int(pad_id), dtype=torch.long))
            text_lengths_list.append(0)

    text_padded = nn.utils.rnn.pad_sequence(text_ids_list, batch_first=True, padding_value=pad_id)
    text_lengths = torch.tensor(text_lengths_list, dtype=torch.long)

    result = {
        "image": images,
        "text_ids": text_padded,
        "text_lengths": text_lengths,
        "boxes": boxes,
        "labels": labels,
        "orientations": orientations,
        "image_stems": image_stems,
        "text_ids_present": torch.tensor(text_ids_present, dtype=torch.bool),
    }

    if "illus_mask" in batch[0]:
        result["illus_mask"] = torch.stack([b["illus_mask"] for b in batch], dim=0)

    result["avg_gt_per_image"] = [
        b.get("avg_gt_per_image", STAGE1_AVG_GT_PER_IMAGE) for b in batch
    ]

    return result


def scheduled_teacher_forcing(epoch, total_epochs, start=1.0, end=0.2, schedule="exp"):
    if schedule == "linear":
        return max(end, start - (start - end) * (epoch / max(1, (total_epochs - 1))))
    decay = 0.97 ** epoch
    return max(end, start * decay)


def masked_bbox_smoothl1_loss(
    pred_bbox: torch.Tensor,  # (B,4,H,W)
    gt_bbox: torch.Tensor,  # (B,4,H,W)
    gt_bbox_mask: torch.Tensor,  # (B,H,W) boolean mask
) -> torch.Tensor:
    """Compute bbox loss only where objects exist."""
    if gt_bbox_mask.sum() == 0:
        return pred_bbox.new_tensor(0.0)

    # select positives -> (Npos,4)
    pred_pos = pred_bbox.permute(0, 2, 3, 1)[gt_bbox_mask]
    gt_pos = gt_bbox.permute(0, 2, 3, 1)[gt_bbox_mask]

    return F.smooth_l1_loss(pred_pos, gt_pos, reduction="mean")


def masked_bbox_diou_loss(
    pred_bbox: torch.Tensor,      # (B, 4, H, W) — deltas (dx, dy, bw, bh) in grid units
    gt_bbox: torch.Tensor,        # (B, 4, H, W) — same format
    gt_bbox_mask: torch.Tensor,   # (B, H, W) bool
    image_size: tuple[int, int],  # (img_h, img_w)
) -> torch.Tensor:
    """
    DIoU loss on positive cells.

    Bboxes are stored as (dx, dy, bw, bh) in grid-unit deltas:
        cx_abs = (col + dx) * stride_w,  w_abs = bw * stride_w
        cy_abs = (row + dy) * stride_h,  h_abs = bh * stride_h

    DIoU = 1 - IoU + d² / c²  (d = centre distance, c = enclosing-box diagonal).
    Directly optimises IoU, which the validation metric (IoU ≥ 0.5) measures,
    plus a centre-distance penalty that SmoothL1 in delta space only approximates.
    """
    if gt_bbox_mask.sum() == 0:
        return pred_bbox.new_tensor(0.0)

    B, _, H, W = pred_bbox.shape
    img_h, img_w = image_size
    stride_h = img_h / H
    stride_w = img_w / W

    # Grid coordinates (row, col) — broadcast over batch
    rows = torch.arange(H, device=pred_bbox.device, dtype=pred_bbox.dtype).view(1, H, 1).expand(B, H, W)
    cols = torch.arange(W, device=pred_bbox.device, dtype=pred_bbox.dtype).view(1, 1, W).expand(B, H, W)

    def _to_xyxy(bbox: torch.Tensor):
        cx = (cols + bbox[:, 0]) * stride_w
        cy = (rows + bbox[:, 1]) * stride_h
        w  = bbox[:, 2] * stride_w
        h  = bbox[:, 3] * stride_h
        return cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5

    px1, py1, px2, py2 = _to_xyxy(pred_bbox)
    gx1, gy1, gx2, gy2 = _to_xyxy(gt_bbox)

    # Select positive cells
    px1 = px1[gt_bbox_mask]; py1 = py1[gt_bbox_mask]
    px2 = px2[gt_bbox_mask]; py2 = py2[gt_bbox_mask]
    gx1 = gx1[gt_bbox_mask]; gy1 = gy1[gt_bbox_mask]
    gx2 = gx2[gt_bbox_mask]; gy2 = gy2[gt_bbox_mask]

    # Intersection
    ix1 = torch.maximum(px1, gx1); iy1 = torch.maximum(py1, gy1)
    ix2 = torch.minimum(px2, gx2); iy2 = torch.minimum(py2, gy2)
    inter = (ix2 - ix1).clamp_min(0) * (iy2 - iy1).clamp_min(0)

    p_area = (px2 - px1).clamp_min(0) * (py2 - py1).clamp_min(0)
    g_area = (gx2 - gx1).clamp_min(0) * (gy2 - gy1).clamp_min(0)
    union  = (p_area + g_area - inter).clamp_min(1e-6)
    iou    = inter / union

    # Centre-distance penalty
    pcx = (px1 + px2) * 0.5; pcy = (py1 + py2) * 0.5
    gcx = (gx1 + gx2) * 0.5; gcy = (gy1 + gy2) * 0.5
    d_sq = (pcx - gcx) ** 2 + (pcy - gcy) ** 2

    # Enclosing-box diagonal
    ex1 = torch.minimum(px1, gx1); ey1 = torch.minimum(py1, gy1)
    ex2 = torch.maximum(px2, gx2); ey2 = torch.maximum(py2, gy2)
    c_sq = ((ex2 - ex1) ** 2 + (ey2 - ey1) ** 2).clamp_min(1e-6)

    diou = 1.0 - iou + d_sq / c_sq
    return diou.mean()


def validate_detector(unet, detector, dataloader, device, use_mixed_precision, bbox_radius=0):
    unet.eval()
    detector.eval()

    total_loss = 0.0
    total_heat = 0.0
    total_bbox = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validating", leave=False)):
        images = batch["image"].to(device)
        boxes = [
            b.to(device) if b.numel() > 0 else torch.empty((0, 4), device=device)
            for b in batch.get("boxes", [])
        ]
        labels = [
            l.to(device)
            if (l is not None and l.numel() > 0)
            else torch.empty((0,), dtype=torch.long, device=device)
            for l in batch.get("labels", [])
        ]

        with torch.amp.autocast(device_type="cuda", enabled=use_mixed_precision):
            features = unet(images)
            _, _, hf, wf = features.shape
            gt_heat, gt_bbox, gt_bbox_mask, _ = build_detection_targets(
                boxes,
                labels,
                output_size=(hf, wf),
                image_size=tuple(images.shape[-2:]),
                device=device,
                sigma=STAGE1_HEATMAP_SIGMA,
                sigma_floor=STAGE1_SIGMA_FLOOR,
                sigma_ceil=STAGE1_SIGMA_CEIL,
                sigma_scale=STAGE1_SIGMA_SCALE,
                bbox_radius=bbox_radius,
            )

            features_shared = detector.shared(features)
            heat_logits = detector.heatmap(features_shared)
            bbox_reg = detector.bbox(features_shared)

            loss_heatmap = focal_loss_heatmap(
                heat_logits,
                gt_heat,
                alpha=STAGE1_FOCAL_ALPHA,
                gamma=STAGE1_FOCAL_GAMMA,
                pos_weight=STAGE1_POS_WEIGHT,
                pos_threshold=STAGE1_FOCAL_POS_THRESHOLD,
            )
            loss_bbox = masked_bbox_smoothl1_loss(bbox_reg, gt_bbox, gt_bbox_mask)
            loss_diou = masked_bbox_diou_loss(
                bbox_reg, gt_bbox, gt_bbox_mask,
                image_size=tuple(images.shape[-2:]),
            )
            loss = loss_heatmap + STAGE1_BBOX_WEIGHT * loss_bbox + STAGE1_DIOU_WEIGHT * loss_diou

            if batch_idx == 0:
                print("\n[VAL DEBUG]")
                print("images.shape:", tuple(images.shape))
                print("features.shape:", tuple(features.shape))
                print(
                    "gt_heat min/max/mean:",
                    gt_heat.min().item(),
                    gt_heat.max().item(),
                    gt_heat.mean().item(),
                )
                print("num bbox supervised cells:", int(gt_bbox_mask.sum().item()))
                if gt_bbox_mask.sum() > 0:
                    gt_bbox_pos = gt_bbox.permute(0, 2, 3, 1)[gt_bbox_mask]
                    pred_bbox_pos = bbox_reg.permute(0, 2, 3, 1)[gt_bbox_mask]
                    print("gt_bbox mean [dx,dy,bw,bh]:", gt_bbox_pos.mean(dim=0).detach().cpu().tolist())
                    print("pred_bbox mean [dx,dy,bw,bh]:", pred_bbox_pos.mean(dim=0).detach().cpu().tolist())

        total_loss += float(loss.item())
        total_heat += float(loss_heatmap.item())
        total_bbox += float(loss_bbox.item())
        num_batches += 1

    denom = max(1, num_batches)
    return (
        total_loss / denom,
        total_heat / denom,
        total_bbox / denom,
    )
