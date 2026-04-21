"""Training script for KuroNetRecognizer (per-ROI character classifier).

Single-phase training: detect boxes, refine, sort by reading order,
classify each ROI as a Kuzushiji character.

No teacher forcing, no free decoding, no pointer mechanism.

Usage:
    python train_kuronet.py
    python train_kuronet.py --phase-a-ckpt checkpoints/stage2_hybrid_phaseA/stage2_hybrid_best.pt
    python train_kuronet.py --resume checkpoints/kuronet_recognizer/kuronet_best.pt
    python train_kuronet.py --epochs 30 --lr 1e-4
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Optional

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    DATA_DIR,
    DEVICE,
    FREEZE_BACKBONE,
    FREEZE_DETECTOR,
    GRAD_CLIP,
    GRADIENT_ACCUMULATION_STEPS,
    IMAGE_SIZE,
    KURONET_CHECKPOINT_DIR,
    KURONET_CLASSIFIER_HIDDEN,
    KURONET_CER_SCORE_THRESH,
    KURONET_ENABLE_TQDM,
    KURONET_EPOCHS,
    KURONET_GRAD_ACCUM_STEPS,
    KURONET_LAMBDA_BOX,
    KURONET_LAMBDA_CHAR,
    KURONET_LAMBDA_DELTA,
    KURONET_LAMBDA_SCORE,
    KURONET_LOG_PREDICTIONS,
    KURONET_LR,
    KURONET_LR_ETA_MIN,
    KURONET_PREDICTION_SAMPLES,
    KURONET_PROGRESS_POSTFIX_N,
    KURONET_ROI_SIZE,
    KURONET_USE_CONTEXT,
    KURONET_VALIDATION_BATCHES,
    KURONET_WEIGHT_DECAY,
    NUM_WORKERS,
    STAGE2_CONTEXT_HIDDEN_DIM,
    STAGE2_CONTEXT_NUM_LAYERS,
    DET_MIN_BOX_SIZE,
    DET_NMS_IOU,
    DET_SCORE_THRESH,
    DET_TOP_K,
    STAGE2_DROPOUT_RATE,
    STAGE2_PROJ_DIM,
    STAGE2_REFINE_HIDDEN_DIM,
    STAGE2_REFINE_NEG_IOU,
    STAGE2_REFINE_POS_IOU,
    STAGE2_REFINE_POS_WEIGHT,
    STAGE2_ROI_FEAT_DIM,
    STAGE2_TOKEN_DIM,
    STAGE2_TOKEN_HIDDEN_DIM,
    STAGE2_TOKEN_USE_SCORE_BRANCH,
    USE_MIXED_PRECISION,
)

from model.kuronet import UNet, DetectorHead
from model.kuronet.kuronet_recognizer import KuroNetRecognizer

from utils import KuzushijiDataset
from utils.stage2_losses import (
    aux_classification_loss,
    delta_regression_loss,
    refine_score_bce_loss,
    smooth_l1_box_loss,
)
from utils.stage2_targets import build_refinement_targets
from utils.training_helpers.helper_stage1 import (
    collate_fn,
    prune_to_keep_last_n,
)
from utils.training_helpers.helper_stage2 import (
    _load_compatible_state_dict,
    _normalize_orientation_label,
    reorder_by_sort_indices,
)
from utils.text_normalization import render_tokens
from utils.vocab import VocabManager


# ---------------------------------------------------------------------------
# Vocabulary + data loaders (reused unchanged from train_stage2.py)
# ---------------------------------------------------------------------------

def load_vocab() -> VocabManager:
    ann_files = sorted(list((Path(DATA_DIR) / "annotations").glob("*.json")))
    if len(ann_files) == 0:
        raise FileNotFoundError(f"No annotation files found in {Path(DATA_DIR) / 'annotations'}")
    return VocabManager.from_annotations(ann_files)


def build_dataloaders(vocab: VocabManager):
    pad_id = vocab.pad_id

    train_dataset = KuzushijiDataset(
        Path(DATA_DIR),
        vocab=vocab,
        use_sequences=True,
        resize=IMAGE_SIZE,
        split="train",
    )
    val_dataset = KuzushijiDataset(
        Path(DATA_DIR),
        vocab=vocab,
        use_sequences=True,
        resize=IMAGE_SIZE,
        split="val",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=lambda b: collate_fn(b, pad_id),
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=lambda b: collate_fn(b, pad_id),
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=NUM_WORKERS > 0,
    )

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_kuronet_model(
    vocab: VocabManager,
    phase_a_ckpt: Optional[str | Path] = None,
) -> KuroNetRecognizer:
    """
    Build KuroNetRecognizer.

    Loads backbone + detector weights from Stage 1 checkpoint.
    Optionally warm-starts shared ROI pipeline from Phase A checkpoint.
    """
    vocab_size = vocab.vocab_size

    stage1_ckpt = CHECKPOINT_DIR / "stage1_detection" / "detector_best.pt"
    if not stage1_ckpt.exists():
        raise FileNotFoundError(f"Stage 1 checkpoint not found: {stage1_ckpt}")

    backbone = UNet(in_channels=3, base_features=32).to(DEVICE)
    detector = DetectorHead(
        in_ch=32,
        num_classes=vocab_size,
        dropout_rate=STAGE2_DROPOUT_RATE,
        predict_boxes=True,
        predict_classes=False,
    ).to(DEVICE)

    ckpt = torch.load(stage1_ckpt, map_location=DEVICE)
    backbone.load_state_dict(ckpt["unet_state_dict"])
    detector.load_state_dict(ckpt["detector_state_dict"])
    print(f"Loaded Stage 1 weights from {stage1_ckpt}")

    model = KuroNetRecognizer(
        backbone=backbone,
        detector=detector,
        backbone_out_channels=32,
        vocab_size=vocab_size,

        proj_dim=STAGE2_PROJ_DIM,
        roi_size=KURONET_ROI_SIZE,
        roi_feat_dim=STAGE2_ROI_FEAT_DIM,
        refine_hidden_dim=STAGE2_REFINE_HIDDEN_DIM,
        token_dim=STAGE2_TOKEN_DIM,
        token_hidden_dim=STAGE2_TOKEN_HIDDEN_DIM,
        token_use_score_branch=STAGE2_TOKEN_USE_SCORE_BRANCH,

        use_context=KURONET_USE_CONTEXT,
        context_hidden_dim=STAGE2_CONTEXT_HIDDEN_DIM,
        context_num_layers=STAGE2_CONTEXT_NUM_LAYERS,

        classifier_hidden_dim=KURONET_CLASSIFIER_HIDDEN,

        det_score_thresh=DET_SCORE_THRESH,
        det_top_k=DET_TOP_K,
        det_nms_iou=DET_NMS_IOU,
        det_min_box_size=DET_MIN_BOX_SIZE,

        dropout=STAGE2_DROPOUT_RATE,
    ).to(DEVICE)

    if FREEZE_BACKBONE:
        for p in model.backbone.parameters():
            p.requires_grad = False
        model.backbone.eval()

    if FREEZE_DETECTOR:
        for p in model.detector.parameters():
            p.requires_grad = False
        model.detector.eval()

    # Warm-start from Phase A checkpoint (shared ROI pipeline weights)
    if phase_a_ckpt is not None:
        phase_a_ckpt = Path(phase_a_ckpt)
        if not phase_a_ckpt.exists():
            print(f"WARNING: Phase A checkpoint not found: {phase_a_ckpt} — skipping warm start.")
        else:
            ckpt_a = torch.load(phase_a_ckpt, map_location=DEVICE)
            state = ckpt_a.get("model_state_dict", ckpt_a)
            _load_compatible_state_dict(model, state)
            print(f"Warm-started from Phase A checkpoint: {phase_a_ckpt}")

    return model


def set_trainable_modules(model: KuroNetRecognizer) -> None:
    """Freeze backbone + detector; train all other components."""
    for p in model.parameters():
        p.requires_grad = True

    if FREEZE_BACKBONE:
        for p in model.backbone.parameters():
            p.requires_grad = False

    if FREEZE_DETECTOR:
        for p in model.detector.parameters():
            p.requires_grad = False


def get_trainable_params(model: KuroNetRecognizer):
    return [p for p in model.parameters() if p.requires_grad]


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def compute_kuronet_loss(
    outputs: dict,
    refine_targets: dict,
) -> tuple[torch.Tensor, dict]:
    """
    Single-phase KuroNet loss.

    Character classification loss is computed on sorted (reading-order) features,
    so labels/masks are reordered via sort_indices before calling aux_classification_loss.

    Box, delta, and score losses are computed in original (pre-sort) order.
    """
    pos_mask   = refine_targets["refine_pos_mask"]      # (B, T) original order
    neg_mask   = refine_targets["refine_neg_mask"]      # (B, T)
    ignore_mask = refine_targets["refine_ignore_mask"]  # (B, T)
    gt_labels  = refine_targets["matched_gt_labels"]    # (B, T) original order
    gt_boxes   = refine_targets["matched_gt_boxes"]     # (B, T, 4)
    gt_deltas  = refine_targets["target_deltas"]        # (B, T, 4)

    # --- Box regression losses (original order) ---
    loss_box = smooth_l1_box_loss(
        pred_boxes=outputs["refined_boxes"],
        target_boxes=gt_boxes,
        pos_mask=pos_mask,
    )

    loss_delta = delta_regression_loss(
        pred_deltas=outputs["box_deltas"],
        target_deltas=gt_deltas,
        pos_mask=pos_mask,
    )

    loss_score = refine_score_bce_loss(
        refine_scores=outputs["refine_scores"],
        pos_mask=pos_mask,
        neg_mask=neg_mask,
        ignore_mask=ignore_mask,
        pos_weight=STAGE2_REFINE_POS_WEIGHT,
    )

    # --- Character classification loss (sorted order) ---
    sort_indices = outputs.get("sort_indices", None)

    if sort_indices is not None:
        gt_labels_sorted = reorder_by_sort_indices(gt_labels, sort_indices)
        pos_mask_sorted  = reorder_by_sort_indices(pos_mask.long(), sort_indices).bool()
    else:
        gt_labels_sorted = gt_labels
        pos_mask_sorted  = pos_mask

    loss_char = aux_classification_loss(
        aux_logits=outputs["char_logits"],
        target_labels=gt_labels_sorted,
        pos_mask=pos_mask_sorted,
        ignore_index=-1,
    )

    total = (
        KURONET_LAMBDA_CHAR  * loss_char
        + KURONET_LAMBDA_BOX   * loss_box
        + KURONET_LAMBDA_DELTA * loss_delta
        + KURONET_LAMBDA_SCORE * loss_score
    )

    return total, {
        "loss_char":  float(loss_char.item()),
        "loss_box":   float(loss_box.item()),
        "loss_delta": float(loss_delta.item()),
        "loss_score": float(loss_score.item()),
    }


# ---------------------------------------------------------------------------
# Accuracy helpers
# ---------------------------------------------------------------------------

def _top_k_accuracy(
    logits: torch.Tensor,   # (B, T, V)
    labels: torch.Tensor,   # (B, T)
    mask: torch.Tensor,     # (B, T) bool
    k: int = 1,
) -> float:
    valid = mask & labels.ne(-1)
    if not valid.any():
        return 0.0
    logits_v = logits[valid]     # (N, V)
    labels_v = labels[valid]     # (N,)
    topk = logits_v.topk(k, dim=-1).indices  # (N, k)
    correct = (topk == labels_v.unsqueeze(-1)).any(dim=-1)
    return float(correct.float().mean().item())


def _ids_to_text(ids: list[int], vocab: VocabManager) -> str:
    return render_tokens(vocab.decode(ids, remove_special=True))


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


def _compute_assembled_cer(
    outputs: dict,
    gt_labels_sorted: torch.Tensor,   # (B, T) in sorted order
    pos_mask_sorted: torch.Tensor,     # (B, T) in sorted order
    vocab: VocabManager,
    score_thresh: float = 0.0,
) -> float:
    """
    Assemble per-ROI argmax predictions in reading order and compute CER vs GT.

    score_thresh: if > 0, filters proposals by sigmoid(refine_score) before
    assembling the prediction string, removing false-positive proposals that
    would otherwise add insertion errors to the CER.
    """
    char_logits   = outputs["char_logits"]    # (B, T, V)
    ordered_mask  = outputs["ordered_mask"]   # (B, T) sorted order
    refine_scores = outputs.get("refine_scores")   # (B, T) original order
    sort_indices  = outputs.get("sort_indices")    # (B, T)
    bsz = char_logits.size(0)

    total_cer = 0.0
    count = 0

    for b in range(bsz):
        # Build GT text from sorted GT labels (positive ROIs only)
        valid_b = pos_mask_sorted[b] & gt_labels_sorted[b].ne(-1)
        gt_ids = gt_labels_sorted[b][valid_b].tolist()
        gt_text = _ids_to_text(gt_ids, vocab)

        if len(gt_text) == 0:
            continue

        # Determine valid prediction positions
        if score_thresh > 0.0 and refine_scores is not None and sort_indices is not None:
            # Map sorted mask positions back to original order to get scores
            si = sort_indices[b]                                     # (T,)
            sorted_positions = ordered_mask[b].nonzero(as_tuple=True)[0]  # sorted indices
            orig_positions   = si[sorted_positions]                  # original indices
            scores = torch.sigmoid(refine_scores[b][orig_positions]) # (N_valid,)
            keep = scores >= score_thresh
            valid_positions = sorted_positions[keep]
        else:
            valid_positions = ordered_mask[b].nonzero(as_tuple=True)[0]

        pred_ids  = char_logits[b, valid_positions].argmax(dim=-1).tolist()
        pred_text = _ids_to_text(pred_ids, vocab)

        cer = _edit_distance(pred_text, gt_text) / max(1, len(gt_text))
        total_cer += cer
        count += 1

    return total_cer / count if count > 0 else 0.0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_kuronet(
    model: KuroNetRecognizer,
    val_loader: DataLoader,
    vocab: VocabManager,
    max_batches: Optional[int] = None,
) -> dict:
    """
    Validation loop for KuroNetRecognizer.

    Returns:
        top1_acc, top5_acc, coverage, assembled_cer, avg_iou,
        avg_proposals_per_image, avg_gt_per_image
    """
    model.eval()

    total_loss = 0.0
    loss_parts: dict[str, float] = {"loss_char": 0., "loss_box": 0., "loss_delta": 0., "loss_score": 0.}

    top1_sum = 0.0
    top5_sum = 0.0
    cer_sum = 0.0
    iou_sum = 0.0
    pos_sum = 0.0
    props_sum = 0.0
    gt_sum = 0.0
    n_batches = 0

    # For confusion analysis
    error_counter: Counter = Counter()
    pred_counter: Counter = Counter()

    # For prediction examples
    examples: list[dict] = []

    pad_id = vocab.pad_id

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images         = batch["image"].to(DEVICE, non_blocking=True)
            boxes_list     = [b.to(DEVICE, dtype=torch.float32) for b in batch["boxes"]]
            gt_labels_list = [l.to(DEVICE, dtype=torch.long) for l in batch["labels"]]
            orientations   = [
                _normalize_orientation_label(o) for o in batch["orientations"]
            ]

            outputs = model(images, orientations)

            # Build refinement targets in original box order
            refine_targets = build_refinement_targets(
                coarse_boxes=outputs["roi_boxes"],
                roi_mask=outputs["roi_mask"],
                gt_boxes_list=boxes_list,
                gt_labels_list=gt_labels_list,
                pos_iou_thresh=STAGE2_REFINE_POS_IOU,
                neg_iou_thresh=STAGE2_REFINE_NEG_IOU,
            )

            loss, parts = compute_kuronet_loss(outputs, refine_targets)
            total_loss += float(loss.item())
            for k in loss_parts:
                loss_parts[k] += parts[k]

            # Reorder labels/mask to sorted order for classification metrics
            sort_indices = outputs.get("sort_indices", None)
            gt_labels   = refine_targets["matched_gt_labels"]
            pos_mask    = refine_targets["refine_pos_mask"]

            if sort_indices is not None:
                gt_labels_s = reorder_by_sort_indices(gt_labels, sort_indices)
                pos_mask_s  = reorder_by_sort_indices(pos_mask.long(), sort_indices).bool()
            else:
                gt_labels_s = gt_labels
                pos_mask_s  = pos_mask

            top1_sum += _top_k_accuracy(outputs["char_logits"], gt_labels_s, pos_mask_s, k=1)
            top5_sum += _top_k_accuracy(outputs["char_logits"], gt_labels_s, pos_mask_s, k=5)
            cer_sum  += _compute_assembled_cer(outputs, gt_labels_s, pos_mask_s, vocab,
                                              score_thresh=KURONET_CER_SCORE_THRESH)

            # Coverage: proportion of GT chars matched by a positive ROI
            for b in range(images.size(0)):
                n_gt   = boxes_list[b].size(0)
                n_pos  = int(refine_targets["refine_pos_mask"][b].sum().item())
                n_prop = int(outputs["roi_mask"][b].sum().item())
                pos_sum += n_pos
                props_sum += n_prop
                gt_sum    += n_gt

            # Average IoU on positives
            iou_b = refine_targets["matched_iou"]
            pm_b  = refine_targets["refine_pos_mask"]
            if pm_b.any():
                iou_sum += float(iou_b[pm_b].mean().item())

            # Confusion pairs (on first sample of each batch)
            valid_b = pos_mask_s[0] & gt_labels_s[0].ne(-1)
            if valid_b.any():
                pred_ids = outputs["char_logits"][0, valid_b].argmax(dim=-1).tolist()
                true_ids = gt_labels_s[0][valid_b].tolist()
                for pid, tid in zip(pred_ids, true_ids):
                    pred_counter[pid] += 1
                    if pid != tid:
                        error_counter[(tid, pid)] += 1

            # Log prediction examples
            if len(examples) < KURONET_PREDICTION_SAMPLES and KURONET_LOG_PREDICTIONS:
                valid_pred = outputs["ordered_mask"][0]
                pred_ids_ex = outputs["char_logits"][0, valid_pred].argmax(dim=-1).tolist()
                pred_text = _ids_to_text(pred_ids_ex, vocab)

                valid_gt = pos_mask_s[0] & gt_labels_s[0].ne(-1)
                gt_ids_ex = gt_labels_s[0][valid_gt].tolist()
                gt_text = _ids_to_text(gt_ids_ex, vocab)

                examples.append({"pred": pred_text[:120], "gt": gt_text[:120]})

            n_batches += 1

    if n_batches == 0:
        return {}

    # Aggregate
    avg = lambda x: x / n_batches
    n_images = n_batches * BATCH_SIZE

    avg_props     = props_sum / max(1, n_images)
    avg_pos_img   = pos_sum   / max(1, n_images)
    avg_gt_img    = gt_sum    / max(1, n_images)
    det_recall    = avg_pos_img / max(1e-6, avg_gt_img)
    det_precision = avg_pos_img / max(1e-6, avg_props)
    det_f1        = 2 * det_precision * det_recall / max(1e-6, det_precision + det_recall)

    metrics = {
        "val_loss":    avg(total_loss),
        "top1_acc":    avg(top1_sum),
        "top5_acc":    avg(top5_sum),
        "assembled_cer": avg(cer_sum),
        "coverage":       det_recall,
        "det_precision":  det_precision,
        "det_f1":         det_f1,
        "avg_proposals_per_image": avg_props,
        "avg_pos_per_image":       avg_pos_img,
        "avg_gt_per_image":        avg_gt_img,
        "avg_iou_on_positives":    avg(iou_sum),
        **{f"val_{k}": avg(v) for k, v in loss_parts.items()},
    }

    # Print summary
    print(
        f"Val | loss={metrics['val_loss']:.4f}"
        f" (char={metrics['val_loss_char']:.4f}"
        f", box={metrics['val_loss_box']:.4f}"
        f", delta={metrics['val_loss_delta']:.4f}"
        f", score={metrics['val_loss_score']:.4f})"
    )
    print(
        f"Val | top1={metrics['top1_acc']:.4f}"
        f"  top5={metrics['top5_acc']:.4f}"
        f"  CER={metrics['assembled_cer']:.4f}"
        f"  recall={metrics['coverage']:.4f}"
        f"  prec={metrics['det_precision']:.4f}"
        f"  F1={metrics['det_f1']:.4f}"
        f"  IoU+={metrics['avg_iou_on_positives']:.4f}"
        f"  props/img={metrics['avg_proposals_per_image']:.1f}"
        f"  gt/img={metrics['avg_gt_per_image']:.1f}"
    )

    # Top predictions and errors
    top_preds = [
        {"token": _ids_to_text([tid], vocab), "count": cnt}
        for tid, cnt in pred_counter.most_common(8)
    ]
    top_errors = [
        {
            "gt": _ids_to_text([gt], vocab),
            "pred": _ids_to_text([pr], vocab),
            "count": cnt,
        }
        for (gt, pr), cnt in error_counter.most_common(8)
    ]
    print(f"Top predictions: {top_preds}")
    print(f"Top errors:      {top_errors}")

    if examples:
        for i, ex in enumerate(examples):
            print(f"[VAL PRED] sample={i} | PRED: {ex['pred']} | GT: {ex['gt']}")

    return metrics


# ---------------------------------------------------------------------------
# Model score for checkpoint selection
# ---------------------------------------------------------------------------

def select_model_score(metrics: dict) -> float:
    """
    Composite score for best-checkpoint selection.

    Rewards both per-ROI accuracy and coverage (recall) equally.
    Penalises high assembled CER.
    """
    top1  = float(metrics.get("top1_acc", 0.0))
    cov   = float(metrics.get("coverage", 0.0))
    cer   = float(metrics.get("assembled_cer", 1.0))
    return top1 * cov - 0.3 * cer


# ---------------------------------------------------------------------------
# Training epoch
# ---------------------------------------------------------------------------

def train_epoch(
    model: KuroNetRecognizer,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    vocab: VocabManager,
    epoch: int,
    grad_accum_steps: int,
) -> dict:
    model.train()
    if FREEZE_BACKBONE:
        model.backbone.eval()
    if FREEZE_DETECTOR:
        model.detector.eval()

    total_loss = 0.0
    loss_parts: dict[str, float] = {"loss_char": 0., "loss_box": 0., "loss_delta": 0., "loss_score": 0.}
    top1_sum = 0.0
    n_batches = 0

    pad_id = vocab.pad_id
    optimizer.zero_grad()

    bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch}",
        disable=not KURONET_ENABLE_TQDM,
        dynamic_ncols=True,
    )

    for batch_idx, batch in enumerate(bar):
        images       = batch["image"].to(DEVICE, non_blocking=True)
        boxes_list   = [b.to(DEVICE, dtype=torch.float32) for b in batch["boxes"]]
        gt_labels_list = [l.to(DEVICE, dtype=torch.long) for l in batch["labels"]]
        orientations = [
            _normalize_orientation_label(o) for o in batch["orientations"]
        ]

        with torch.cuda.amp.autocast(enabled=USE_MIXED_PRECISION):
            outputs = model(images, orientations)

            refine_targets = build_refinement_targets(
                coarse_boxes=outputs["roi_boxes"],
                roi_mask=outputs["roi_mask"],
                gt_boxes_list=boxes_list,
                gt_labels_list=gt_labels_list,
                pos_iou_thresh=STAGE2_REFINE_POS_IOU,
                neg_iou_thresh=STAGE2_REFINE_NEG_IOU,
            )

            loss, parts = compute_kuronet_loss(outputs, refine_targets)
            loss = loss / grad_accum_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += float(loss.item()) * grad_accum_steps
        for k in loss_parts:
            loss_parts[k] += parts[k]

        # Quick train accuracy (top-1 on positives in sorted order)
        with torch.no_grad():
            si = outputs.get("sort_indices", None)
            gt_labels  = refine_targets["matched_gt_labels"]
            pos_mask   = refine_targets["refine_pos_mask"]
            if si is not None:
                gt_labels_s = reorder_by_sort_indices(gt_labels, si)
                pos_mask_s  = reorder_by_sort_indices(pos_mask.long(), si).bool()
            else:
                gt_labels_s, pos_mask_s = gt_labels, pos_mask
            top1_sum += _top_k_accuracy(outputs["char_logits"], gt_labels_s, pos_mask_s, k=1)

        n_batches += 1

        if (batch_idx + 1) % KURONET_PROGRESS_POSTFIX_N == 0:
            bar.set_postfix({
                "loss": f"{total_loss / n_batches:.4f}",
                "char": f"{loss_parts['loss_char'] / n_batches:.4f}",
                "top1": f"{top1_sum / n_batches:.3f}",
            })

    if n_batches == 0:
        return {}

    avg = lambda x: x / n_batches
    return {
        "train_loss":      avg(total_loss),
        "train_top1":      avg(top1_sum),
        **{f"train_{k}": avg(v) for k, v in loss_parts.items()},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train KuroNet per-ROI classifier")
    parser.add_argument("--phase-a-ckpt", type=str, default=None,
                        help="Path to Stage 2 Phase A checkpoint for warm-start")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to KuroNet checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=KURONET_EPOCHS)
    parser.add_argument("--lr", type=float, default=KURONET_LR)
    parser.add_argument("--weight-decay", type=float, default=KURONET_WEIGHT_DECAY)
    parser.add_argument("--grad-accum", type=int, default=KURONET_GRAD_ACCUM_STEPS)
    args = parser.parse_args()

    print("=" * 70)
    print("KURONET RECOGNIZER TRAINING (per-ROI classifier)")
    print("=" * 70)

    KURONET_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    vocab = load_vocab()
    train_loader, val_loader = build_dataloaders(vocab)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # Build model
    model = build_kuronet_model(
        vocab=vocab,
        phase_a_ckpt=args.phase_a_ckpt,
    )

    start_epoch = 1
    best_score  = -float("inf")

    # Resume from checkpoint
    if args.resume is not None:
        resume_path = Path(args.resume)
        if resume_path.exists():
            ckpt = torch.load(resume_path, map_location=DEVICE)
            model.load_state_dict(ckpt["model_state_dict"])
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_score  = float(ckpt.get("best_score", -float("inf")))
            print(f"Resumed from {resume_path} (epoch {start_epoch - 1})")
        else:
            print(f"WARNING: resume checkpoint not found: {resume_path}")

    set_trainable_modules(model)
    trainable_params = get_trainable_params(model)
    print(f"Trainable params: {sum(p.numel() for p in trainable_params):,}")
    print("=" * 70)

    optimizer = optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=KURONET_LR_ETA_MIN,
        last_epoch=start_epoch - 2,  # -2 so first step lands at epoch 1 LR
    )
    scaler = torch.cuda.amp.GradScaler(enabled=USE_MIXED_PRECISION)

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            vocab=vocab,
            epoch=epoch,
            grad_accum_steps=args.grad_accum,
        )

        print(
            f"Epoch {epoch}/{args.epochs}"
            f" | Train loss={train_metrics.get('train_loss', 0):.4f}"
            f" (char={train_metrics.get('train_loss_char', 0):.4f}"
            f", box={train_metrics.get('train_loss_box', 0):.4f}"
            f", delta={train_metrics.get('train_loss_delta', 0):.4f}"
            f", score={train_metrics.get('train_loss_score', 0):.4f})"
            f" | train_top1={train_metrics.get('train_top1', 0):.4f}"
        )

        val_metrics = validate_kuronet(
            model=model,
            val_loader=val_loader,
            vocab=vocab,
            max_batches=KURONET_VALIDATION_BATCHES,
        )

        score = select_model_score(val_metrics)
        ckpt_state = {
            "epoch":            epoch,
            "model_state_dict": model.state_dict(),
            "best_score":       best_score,
            "val_metrics":      val_metrics,
            "train_metrics":    train_metrics,
        }

        # Save epoch checkpoint
        epoch_path = KURONET_CHECKPOINT_DIR / f"kuronet_epoch{epoch}.pt"
        torch.save(ckpt_state, epoch_path)

        if score > best_score:
            best_score = score
            best_path  = KURONET_CHECKPOINT_DIR / "kuronet_best.pt"
            torch.save(ckpt_state, best_path)
            print(f"✅ saved best: kuronet_best.pt (score={score:.4f})")

        # Keep last 3 epoch checkpoints
        prune_to_keep_last_n(KURONET_CHECKPOINT_DIR, keep=3)

        scheduler.step()

    print("=" * 70)
    print(f"TRAINING COMPLETE")
    print(f"Best checkpoint: {KURONET_CHECKPOINT_DIR / 'kuronet_best.pt'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
