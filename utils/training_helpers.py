"""Shared training helper functions for stage training and Optuna runs."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from config import BBOX_WEIGHT, DETECTOR_HEATMAP_SIGMA, FOCAL_ALPHA, FOCAL_GAMMA, POS_WEIGHT
from utils.detection_utils import build_detection_targets
from utils.focal_loss import focal_loss_heatmap


def prune_existing_checkpoints(ckpt_dir: Path, old_name: str = "checkpoint_old.pt") -> None:
    """At start: keep only the newest checkpoint, rename it to old_name."""
    ckpts = sorted(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
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
    """Keep only the newest N checkpoints (excluding a preserved old file)."""
    ckpts = [p for p in ckpt_dir.glob("*.pt") if p.name != exclude]
    ckpts = sorted(ckpts, key=lambda p: p.stat().st_mtime)
    if len(ckpts) <= keep:
        return
    to_delete = ckpts[:-keep]
    for p in to_delete:
        try:
            p.unlink()
        except Exception as exc:
            print(f"Warning: could not delete old checkpoint {p}: {exc}")


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

    # Text sequences - fill missing with None instead of filtering
    text_ids_list = []
    text_lengths_list = []
    for b in batch:
        if "text_ids" in b and b["text_ids"] is not None:
            text_ids_list.append(b["text_ids"])
            text_lengths_list.append(len(b["text_ids"]))
        else:
            text_ids_list.append(None)
            text_lengths_list.append(0)

    # Pad sequences (skip None entries)
    valid_text_ids = [t for t in text_ids_list if t is not None]
    if len(valid_text_ids) == 0:
        text_padded = None
        text_lengths = torch.tensor(text_lengths_list, dtype=torch.long)
    else:
        text_padded = nn.utils.rnn.pad_sequence(valid_text_ids, batch_first=True, padding_value=pad_id)
        text_lengths = torch.tensor(text_lengths_list, dtype=torch.long)

    return {
        "image": images,
        "text_ids": text_padded,
        "text_lengths": text_lengths,
        "boxes": boxes,
        "labels": labels,
        "orientations": orientations,
        "text_ids_present": torch.tensor([t is not None for t in text_ids_list], dtype=torch.bool),
    }


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
                sigma=DETECTOR_HEATMAP_SIGMA,
                bbox_radius=bbox_radius,
            )

            features_shared = detector.shared(features)
            heat_logits = detector.heatmap(features_shared)
            bbox_reg = detector.bbox(features_shared)

            loss_heatmap = focal_loss_heatmap(
                heat_logits,
                gt_heat,
                alpha=FOCAL_ALPHA,
                gamma=FOCAL_GAMMA,
                pos_weight=POS_WEIGHT,
            )
            loss_bbox = masked_bbox_smoothl1_loss(bbox_reg, gt_bbox, gt_bbox_mask)
            loss = loss_heatmap + BBOX_WEIGHT * loss_bbox

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
