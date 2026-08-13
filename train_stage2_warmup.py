# train_stage2_hybrid.py

from __future__ import annotations

from functools import partial
from pathlib import Path
from collections import Counter
from typing import Optional

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    AVG_GT_PER_IMAGE,
    BACKBONE_BASE_FEATURES,
    DATA_DIR,
    DENSITY_FACTOR,
    DENSITY_GRID,
    DEVICE,
    STAGE2_BATCH_SIZE,
    FREEZE_BACKBONE,
    FREEZE_DETECTOR,
    NUM_WORKERS,
    NUM_EPOCHS,
    LR,
    STAGE2_CONTEXT_HIDDEN_DIM,
    STAGE2_CONTEXT_MODE,
    STAGE2_CONTEXT_NUM_LAYERS,
    DET_MIN_BOX_SIZE,
    DET_NMS_IOU,
    DET_SCORE_THRESH,
    DET_TOP_K,
    STAGE2_TRAIN_PROP_STATS_EVERY_N_STEPS,
    STAGE2_ENABLE_TQDM,
    STAGE2_PROGRESS_POSTFIX_EVERY_N_STEPS,
    STAGE2_PROJ_DIM,
    STAGE2_REFINE_HIDDEN_DIM,
    STAGE2_ROI_FEAT_DIM,
    STAGE2_ROI_SIZE,
    STAGE2_TOKEN_HIDDEN_DIM,
    STAGE2_TOKEN_DIM,
    STAGE2_TOKEN_USE_SCORE_BRANCH,
    STAGE2_USE_AUX_HEAD,
    WEIGHT_DECAY,
    IMAGE_SIZE,
    CHECKPOINT_DIR,
    USE_MIXED_PRECISION,
    GRADIENT_ACCUMULATION_STEPS,
    STAGE2_DROPOUT_RATE,
    STAGE2_REFINE_POS_IOU,
    STAGE2_REFINE_NEG_IOU,
    STAGE2_REFINE_POS_WEIGHT,
    STAGE2_USE_HUNGARIAN,
    VALIDATION_BATCHES,
    STAGE2_DEBUG_BATCH_STATS,
    STAGE2_DEBUG_AUX_ALIGNMENT,
    STAGE2_DEBUG_AUX_ALIGNMENT_LIMIT,
    WARMUP_EPOCHS,
    WARMUP_LAMBDA_BOX,
    WARMUP_LAMBDA_DELTA,
    WARMUP_LAMBDA_SCORE,
    WARMUP_LAMBDA_AUX,
    BACKBONE_TYPE,
)

from model.suminanet import DetectorHead, build_backbone
from model.suminanet.hybrid_recognizer import HybridSuminaNetRecognizer

from utils import KuzushijiDataset
from utils.training_helpers.helper_stage1 import (
    collate_fn,
    prune_existing_checkpoints,
    prune_to_keep_last_n,
)
from utils.vocab import VocabManager
from utils.stage2_targets import build_refinement_targets
from utils.stage2_losses import (
    smooth_l1_box_loss,
    delta_regression_loss,
    refine_score_bce_loss,
    aux_classification_loss,
)

from utils.training_helpers.helper_stage2 import (
    _load_compatible_state_dict,
    _normalize_orientation_label,
    reorder_by_sort_indices,
)
from utils.training_helpers.logging_stage2 import (
    _debug_aux_alignment,
    _finalize_aux_branch_stats,
    _init_aux_branch_stats,
    _update_refinement_epoch_stats,
    log_stage2_batch_debug,
)


def get_warmup_settings(overrides: Optional[dict] = None) -> dict:
    overrides = overrides or {}
    return {
        "epochs": int(WARMUP_EPOCHS),
        "lambda_box": float(WARMUP_LAMBDA_BOX),
        "lambda_delta": float(WARMUP_LAMBDA_DELTA),
        "lambda_score": float(WARMUP_LAMBDA_SCORE),
        "lambda_aux": float(WARMUP_LAMBDA_AUX),
        "train_context_encoder": True,
        "use_context_aux_for_loss": True,
        "raw_aux_weight": 0.0,
        "context_aux_weight": 1.0,
    }


def _select_active_aux_summary(
    aux_without_ctx: dict,
    aux_with_ctx: dict,
    use_context_aux_for_loss: bool,
) -> dict:
    return aux_with_ctx if use_context_aux_for_loss else aux_without_ctx


def _select_model_score(val_metrics: dict) -> float:
    prop = val_metrics["proposal_summary"]
    aux_summary = prop["aux_summary"]
    active_aux = aux_summary["with_context_encode"]
    return (
        2.0 * float(prop["unique_coverage_ratio"])
        + 1.0 * float(active_aux["top1"])
        + 0.5 * float(active_aux["top5"])
        + 0.25 * float(prop["avg_matched_iou_on_positives"])
        - 0.02 * float(prop["avg_negatives_per_image"])
    )


def set_trainable_modules_for_phase(model: HybridSuminaNetRecognizer) -> None:
    for name, p in model.named_parameters():
        if name.startswith("backbone.") or name.startswith("detector.") or name.startswith("aux_head."):
            continue
        p.requires_grad = False

    for module in [model.feature_projector, model.roi_pool, model.roi_refine, model.roi_tokens]:
        for p in module.parameters():
            p.requires_grad = True

    if hasattr(model.roi_pool, "aux_head") and model.roi_pool.aux_head is not None:
        for p in model.roi_pool.aux_head.parameters():
            p.requires_grad = True

    if hasattr(model, "aux_head_context") and model.aux_head_context is not None:
        for p in model.aux_head_context.parameters():
            p.requires_grad = True

    for p in model.context_encoder.parameters():
        p.requires_grad = True



def _compute_phase_aux_loss(
    outputs: dict,
    refine_targets: dict,
    phase_settings: dict,
) -> tuple[torch.Tensor, dict]:
    zero = outputs["refined_boxes"].new_tensor(0.0)

    matched_labels = refine_targets["matched_gt_labels"]
    pos_mask = refine_targets["refine_pos_mask"]

    raw_aux_weight = float(phase_settings.get("raw_aux_weight", 0.0))
    context_aux_weight = float(phase_settings.get("context_aux_weight", 0.0))

    loss_aux_raw = zero
    loss_aux_ctx = zero

    if outputs.get("aux_logits", None) is not None and raw_aux_weight > 0.0:
        loss_aux_raw = aux_classification_loss(
            aux_logits=outputs["aux_logits"],
            target_labels=matched_labels,
            pos_mask=pos_mask,
            ignore_index=-1,
        )

    if outputs.get("aux_logits_with_context", None) is not None and context_aux_weight > 0.0:
        sort_indices = outputs.get("sort_indices", None)
        if sort_indices is not None:
            matched_labels_ctx = reorder_by_sort_indices(matched_labels, sort_indices)
            pos_mask_ctx = reorder_by_sort_indices(pos_mask.long(), sort_indices).bool()
        else:
            matched_labels_ctx = matched_labels
            pos_mask_ctx = pos_mask

        loss_aux_ctx = aux_classification_loss(
            aux_logits=outputs["aux_logits_with_context"],
            target_labels=matched_labels_ctx,
            pos_mask=pos_mask_ctx,
            ignore_index=-1,
        )

    total_aux = raw_aux_weight * loss_aux_raw + context_aux_weight * loss_aux_ctx
    return total_aux, {
        "loss_aux_raw": loss_aux_raw,
        "loss_aux_ctx": loss_aux_ctx,
    }


def load_vocab() -> VocabManager:
    ann_files = sorted(list((Path(DATA_DIR) / "annotations").glob("*.json")))
    if len(ann_files) == 0:
        raise FileNotFoundError(f"No annotation files found in {Path(DATA_DIR) / 'annotations'}")
    return VocabManager.from_annotations(ann_files)


def build_stage2_dataloaders(vocab: VocabManager):
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
        batch_size=STAGE2_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=partial(collate_fn, pad_id=pad_id),
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=NUM_WORKERS > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=STAGE2_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=partial(collate_fn, pad_id=pad_id),
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=NUM_WORKERS > 0,
    )

    return train_loader, val_loader


def build_stage2_model(
    detector_ckpt_path: str | Path,
    vocab: VocabManager,
    overrides: Optional[dict] = None,
) -> HybridSuminaNetRecognizer:
    overrides = overrides or {}
    vocab_size = vocab.vocab_size

    det_score_thresh = float(overrides.get("det_score_thresh", DET_SCORE_THRESH))
    det_top_k = int(overrides.get("det_top_k", DET_TOP_K))
    det_nms_iou = float(overrides.get("det_nms_iou", DET_NMS_IOU))
    det_min_box_size = float(overrides.get("det_min_box_size", DET_MIN_BOX_SIZE))

    density_grid = int(overrides.get("density_grid", DENSITY_GRID))
    density_factor = float(overrides.get("density_factor", DENSITY_FACTOR))
    avg_gt_per_image = int(overrides.get("avg_gt_per_image", AVG_GT_PER_IMAGE))

    token_dim = int(overrides.get("token_dim", STAGE2_TOKEN_DIM))
    token_hidden_dim = int(overrides.get("token_hidden_dim", STAGE2_TOKEN_HIDDEN_DIM))
    token_use_score_branch = bool(overrides.get("token_use_score_branch", STAGE2_TOKEN_USE_SCORE_BRANCH))

    context_hidden_dim = int(overrides.get("context_hidden_dim", STAGE2_CONTEXT_HIDDEN_DIM))
    context_num_layers = int(overrides.get("context_num_layers", STAGE2_CONTEXT_NUM_LAYERS))
    context_mode = str(overrides.get("context_mode", STAGE2_CONTEXT_MODE))

    checkpoint = torch.load(detector_ckpt_path, map_location=DEVICE)
    backbone_type = checkpoint.get("backbone_type", BACKBONE_TYPE)
    backbone = build_backbone(backbone_type, BACKBONE_BASE_FEATURES, pretrained=False).to(DEVICE)
    detector = DetectorHead(
        in_ch=BACKBONE_BASE_FEATURES,
        num_classes=vocab_size,
        dropout_rate=STAGE2_DROPOUT_RATE,
        predict_boxes=True,
        predict_classes=False,
    ).to(DEVICE)

    state_key = "backbone_state_dict" if "backbone_state_dict" in checkpoint else "unet_state_dict"
    backbone.load_state_dict(checkpoint[state_key])
    detector.load_state_dict(checkpoint["detector_state_dict"])

    model = HybridSuminaNetRecognizer(
        backbone=backbone,
        detector=detector,
        backbone_out_channels=BACKBONE_BASE_FEATURES,
        vocab_size=vocab_size,

        proj_dim=STAGE2_PROJ_DIM,
        roi_size=STAGE2_ROI_SIZE,
        roi_feat_dim=STAGE2_ROI_FEAT_DIM,
        refine_hidden_dim=STAGE2_REFINE_HIDDEN_DIM,
        token_dim=token_dim,
        token_hidden_dim=token_hidden_dim,
        token_use_score_branch=token_use_score_branch,
        context_hidden_dim=context_hidden_dim,
        context_num_layers=context_num_layers,
        context_mode=context_mode,

        det_score_thresh=det_score_thresh,
        det_top_k=det_top_k,
        det_nms_iou=det_nms_iou,
        det_min_box_size=det_min_box_size,

        density_grid=density_grid,
        density_factor=density_factor,
        avg_gt_per_image=avg_gt_per_image,

        use_aux_head=STAGE2_USE_AUX_HEAD,
        dropout=STAGE2_DROPOUT_RATE,
    ).to(DEVICE, memory_format=torch.channels_last) # type: ignore

    if FREEZE_BACKBONE:
        for p in model.backbone.parameters():
            p.requires_grad = False
        model.backbone.eval()

    if FREEZE_DETECTOR:
        for p in model.detector.parameters():
            p.requires_grad = False
        model.detector.eval()

    return model


def get_trainable_parameters(model: HybridSuminaNetRecognizer):
    return [p for p in model.parameters() if p.requires_grad]


def move_gt_lists_to_device(gt_list, dtype=None):
    out = []
    for x in gt_list:
        if x is None:
            if dtype is None:
                out.append(torch.empty((0,), device=DEVICE))
            else:
                if dtype == torch.long:
                    out.append(torch.empty((0,), device=DEVICE, dtype=torch.long))
                else:
                    out.append(torch.empty((0, 4), device=DEVICE, dtype=dtype))
            continue

        if x.numel() == 0:
            out.append(x.to(device=DEVICE, non_blocking=True))
        else:
            out.append(x.to(device=DEVICE, dtype=dtype, non_blocking=True) if dtype is not None else x.to(device=DEVICE, non_blocking=True))
    return out


def _compute_losses(outputs: dict, refine_targets: dict, phase_settings: dict) -> dict:
    loss_box = smooth_l1_box_loss(
        pred_boxes=outputs["refined_boxes"],
        target_boxes=refine_targets["matched_gt_boxes"],
        pos_mask=refine_targets["refine_pos_mask"],
    )
    loss_delta = delta_regression_loss(
        pred_deltas=outputs["box_deltas"],
        target_deltas=refine_targets["target_deltas"],
        pos_mask=refine_targets["refine_pos_mask"],
    )
    loss_score = refine_score_bce_loss(
        refine_scores=outputs["refine_scores"],
        pos_mask=refine_targets["refine_pos_mask"],
        neg_mask=refine_targets["refine_neg_mask"],
        ignore_mask=refine_targets["refine_ignore_mask"],
        pos_weight=STAGE2_REFINE_POS_WEIGHT,
    )
    aux_total, aux_parts = _compute_phase_aux_loss(outputs, refine_targets, phase_settings)

    lbox = phase_settings["lambda_box"]
    ldelta = phase_settings["lambda_delta"]
    lscore = phase_settings["lambda_score"]
    laux = phase_settings["lambda_aux"]

    loss_total = lbox * loss_box + ldelta * loss_delta + lscore * loss_score + laux * aux_total

    return {
        "loss_total": loss_total,
        "loss_box": loss_box,
        "loss_delta": loss_delta,
        "loss_score": loss_score,
        "loss_aux": laux * aux_total,
        "loss_aux_raw": aux_parts["loss_aux_raw"],
        "loss_aux_ctx": aux_parts["loss_aux_ctx"],
    }


def _build_proposal_stats_dict() -> dict:
    return {
        "images": 0,
        "images_with_zero_valid_props": 0,
        "proposals": 0,
        "positives": 0,
        "negatives": 0,
        "ignores": 0,
        "gt_tokens": 0,
        "unique_gt_matched": 0,
        "duplicate_positive_matches": 0,
        "matched_iou_sum": 0.0,
        "matched_iou_count": 0,
        "score_logit_sum": 0.0,
        "score_logit_sq_sum": 0.0,
        "score_prob_sum": 0.0,
        "score_prob_sq_sum": 0.0,
        "score_count": 0,
        "ordering_primary_mono_sum": 0.0,
        "ordering_primary_viol_sum": 0.0,
        "ordering_weight_sum": 0.0,
        "aux_without_context_encode": _init_aux_branch_stats(),
        "aux_with_context_encode": _init_aux_branch_stats(),
    }


def _finalize_proposal_summary(stats: dict, vocab: VocabManager, phase_settings: dict) -> dict:
    n_img = max(1, stats["images"])
    n_pos = stats["positives"]

    aux_without_ctx = _finalize_aux_branch_stats(
        stats["aux_without_context_encode"],
        vocab=vocab,
        available=stats["aux_without_context_encode"]["total"] > 0,
    )
    aux_with_ctx = _finalize_aux_branch_stats(
        stats["aux_with_context_encode"],
        vocab=vocab,
        available=stats["aux_with_context_encode"]["total"] > 0,
    )
    active_aux = _select_active_aux_summary(
        aux_without_ctx=aux_without_ctx,
        aux_with_ctx=aux_with_ctx,
        use_context_aux_for_loss=phase_settings["use_context_aux_for_loss"],
    )
    return {
        "images": stats["images"],
        "avg_proposals_per_image": stats["proposals"] / n_img,
        "avg_positives_per_image": n_pos / n_img,
        "avg_gt_tokens_per_image": stats["gt_tokens"] / n_img,
        "positive_coverage_ratio": n_pos / max(1, stats["gt_tokens"]),
        "unique_coverage_ratio": stats["unique_gt_matched"] / max(1, stats["gt_tokens"]),
        "avg_unique_gt_matched_per_image": stats["unique_gt_matched"] / n_img,
        "avg_duplicate_positive_matches_per_image": stats["duplicate_positive_matches"] / n_img,
        "duplicate_positive_rate": stats["duplicate_positive_matches"] / max(1, n_pos),
        "images_with_zero_valid_props_ratio": stats["images_with_zero_valid_props"] / n_img,
        "positive_precision_proxy": n_pos / max(1, stats["proposals"]),
        "avg_negatives_per_image": stats["negatives"] / n_img,
        "avg_ignores_per_image": stats["ignores"] / n_img,
        "avg_matched_iou_on_positives": stats["matched_iou_sum"] / max(1, stats["matched_iou_count"]),
        "ordering_primary_monotonic_fraction": stats["ordering_primary_mono_sum"] / max(1.0, stats["ordering_weight_sum"]),
        "ordering_primary_violation_fraction": stats["ordering_primary_viol_sum"] / max(1.0, stats["ordering_weight_sum"]),
        "refine_score_logit_mean": stats["score_logit_sum"] / max(1, stats["score_count"]),
        "refine_score_logit_std": (
            max(
                0.0,
                stats["score_logit_sq_sum"] / max(1, stats["score_count"])
                - (stats["score_logit_sum"] / max(1, stats["score_count"])) ** 2,
            )
        ) ** 0.5,
        "refine_score_prob_mean": stats["score_prob_sum"] / max(1, stats["score_count"]),
        "refine_score_prob_std": (
            max(
                0.0,
                stats["score_prob_sq_sum"] / max(1, stats["score_count"])
                - (stats["score_prob_sum"] / max(1, stats["score_count"])) ** 2,
            )
        ) ** 0.5,
        "aux_accuracy_on_positives": active_aux["top1"],
        "aux_top5_on_positives": active_aux["top5"],
        "active_aux_branch": "with_context_encode" if phase_settings["use_context_aux_for_loss"] else "without_context_encode",
        "aux_summary": {
            "without_context_encode": aux_without_ctx,
            "with_context_encode": aux_with_ctx,
        },
    }


@torch.no_grad()
def validate_stage2(
    model: HybridSuminaNetRecognizer,
    val_loader: DataLoader,
    vocab: VocabManager,
    phase_settings: Optional[dict] = None,
    max_batches: Optional[int] = None,
):
    model.eval()
    if FREEZE_BACKBONE:
        model.backbone.eval()
    if FREEZE_DETECTOR:
        model.detector.eval()
    if phase_settings is None:
        phase_settings = get_warmup_settings()

    total_loss = 0.0
    total_box = 0.0
    total_delta = 0.0
    total_score = 0.0
    total_aux = 0.0
    n_batches = 0

    proposal_stats = _build_proposal_stats_dict()
    val_orientation_counts: Counter[str] = Counter()

    for batch_idx, batch in enumerate(
        tqdm(
            val_loader,
            desc="Stage2 Validation",
            leave=False,
            disable=not bool(STAGE2_ENABLE_TQDM),
        )
    ):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        images = batch["image"].to(DEVICE, non_blocking=True, memory_format=torch.channels_last)
        text_ids = batch["text_ids"]
        if text_ids is None:
            continue

        orientations = batch["orientations"]
        val_orientation_counts.update(_normalize_orientation_label(x) for x in orientations)

        gt_boxes_list = move_gt_lists_to_device(batch["boxes"], dtype=torch.float32)
        gt_labels_list = move_gt_lists_to_device(batch["labels"], dtype=torch.long)

        with torch.cuda.amp.autocast(enabled=USE_MIXED_PRECISION and str(DEVICE).startswith("cuda"), dtype=torch.float16):
            outputs = model.encode_images(images=images, orientations=orientations)

            refine_targets = build_refinement_targets(
                coarse_boxes=outputs["roi_boxes"],
                roi_mask=outputs["roi_mask"],
                gt_boxes_list=gt_boxes_list,
                gt_labels_list=gt_labels_list,
                pos_iou_thresh=STAGE2_REFINE_POS_IOU,
                neg_iou_thresh=STAGE2_REFINE_NEG_IOU,
                use_hungarian=STAGE2_USE_HUNGARIAN,
            )

            if STAGE2_DEBUG_BATCH_STATS:
                log_stage2_batch_debug(
                    phase="val",
                    epoch=None,
                    batch_idx=batch_idx,
                    roi_mask=outputs["roi_mask"],
                    refine_targets=refine_targets,
                )

            losses = _compute_losses(outputs, refine_targets, phase_settings)

        if STAGE2_DEBUG_AUX_ALIGNMENT and batch_idx == 0:
            _debug_aux_alignment(
                outputs=outputs,
                refine_targets=refine_targets,
                vocab=vocab,
                limit=int(STAGE2_DEBUG_AUX_ALIGNMENT_LIMIT),
            )

        total_loss += float(losses["loss_total"].item())
        total_box += float(losses["loss_box"].item())
        total_delta += float(losses["loss_delta"].item())
        total_score += float(losses["loss_score"].item())
        total_aux += float(losses["loss_aux"].item())
        n_batches += 1

        _update_refinement_epoch_stats(proposal_stats, outputs, refine_targets, gt_labels_list)

    denom = max(1, n_batches)
    return {
        "val_loss": total_loss / denom,
        "val_box": total_box / denom,
        "val_delta": total_delta / denom,
        "val_score": total_score / denom,
        "val_aux": total_aux / denom,
        "proposal_summary": {
            **_finalize_proposal_summary(proposal_stats, vocab, phase_settings),
            "orientation_counts": {
                "horizontal": int(val_orientation_counts.get("horizontal", 0)),
                "vertical": int(val_orientation_counts.get("vertical", 0)),
                "other": int(val_orientation_counts.get("other", 0)),
            },
        },
    }


def train_stage2_hybrid(
    detector_ckpt_path: str | Path,
    num_epochs: int = NUM_EPOCHS,
    lr: Optional[float] = None,
    checkpoint_dir: Optional[str | Path] = None,
    resume_model_ckpt: Optional[str | Path] = None,
    model_overrides: Optional[dict] = None,
    val_max_batches: Optional[int] = None,
):
    if lr is None:
        lr = LR

    if checkpoint_dir is None:
        checkpoint_dir = CHECKPOINT_DIR / "stage2_hybrid"
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    model_overrides = model_overrides or {}
    phase_settings = get_warmup_settings(overrides=model_overrides)
    prune_existing_checkpoints(checkpoint_dir)

    vocab = load_vocab()
    train_loader, val_loader = build_stage2_dataloaders(vocab)
    model = build_stage2_model(
        detector_ckpt_path=detector_ckpt_path,
        vocab=vocab,
        overrides=model_overrides,
    )
    if resume_model_ckpt is not None:
        resume_model_ckpt = Path(resume_model_ckpt)
        if not resume_model_ckpt.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_model_ckpt}")
        resume_ckpt = torch.load(resume_model_ckpt, map_location=DEVICE)
        _load_compatible_state_dict(model, resume_ckpt["model_state_dict"])
        print(f"Loaded resume checkpoint: {resume_model_ckpt}")
    set_trainable_modules_for_phase(model)
    trainable_params = get_trainable_parameters(model)
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters found for Stage 2 hybrid model.")

    optimizer = optim.AdamW(
        trainable_params,
        lr=lr,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6,
    )

    # fp16 mixed precision (Turing GPUs have no bf16 tensor core path); GradScaler
    # guards against fp16 underflow during backward.
    scaler = torch.cuda.amp.GradScaler(enabled=USE_MIXED_PRECISION and str(DEVICE).startswith("cuda"))

    best_val = None
    best_val_metrics = None

    print("=" * 70)
    print("WARMUP TRAINING (ROI pipeline pre-training for SuminaNet)")
    print("=" * 70)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    print(f"Trainable params: {sum(p.numel() for p in trainable_params):,}")
    print(f"Frozen backbone: {FREEZE_BACKBONE} | Frozen detector: {FREEZE_DETECTOR}")
    print("=" * 70)

    for epoch in range(num_epochs):
        model.train()
        if FREEZE_BACKBONE:
            model.backbone.eval()
        if FREEZE_DETECTOR:
            model.detector.eval()

        optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        total_box = 0.0
        total_delta = 0.0
        total_score = 0.0
        total_aux = 0.0
        n_batches = 0

        train_proposal_stats = _build_proposal_stats_dict()
        train_orientation_counts: Counter[str] = Counter()
        train_prop_stats_every_n_steps = max(1, int(model_overrides.get("train_prop_stats_every_n_steps", STAGE2_TRAIN_PROP_STATS_EVERY_N_STEPS)))

        pbar = tqdm(
            train_loader,
            desc=f"Stage2 Epoch {epoch+1}/{num_epochs}",
            mininterval=10.0,
            disable=not bool(STAGE2_ENABLE_TQDM),
        )

        for step, batch in enumerate(pbar):
            images = batch["image"].to(DEVICE, non_blocking=True, memory_format=torch.channels_last)
            text_ids = batch["text_ids"]
            if text_ids is None:
                continue

            orientations = batch["orientations"]
            train_orientation_counts.update(_normalize_orientation_label(x) for x in orientations)
            gt_boxes_list = move_gt_lists_to_device(batch["boxes"], dtype=torch.float32)
            gt_labels_list = move_gt_lists_to_device(batch["labels"], dtype=torch.long)

            with torch.cuda.amp.autocast(enabled=USE_MIXED_PRECISION and str(DEVICE).startswith("cuda"), dtype=torch.float16):
                outputs = model.encode_images(images=images, orientations=orientations)

                refine_targets = build_refinement_targets(
                    coarse_boxes=outputs["roi_boxes"],
                    roi_mask=outputs["roi_mask"],
                    gt_boxes_list=gt_boxes_list,
                    gt_labels_list=gt_labels_list,
                    pos_iou_thresh=STAGE2_REFINE_POS_IOU,
                    neg_iou_thresh=STAGE2_REFINE_NEG_IOU,
                    use_hungarian=STAGE2_USE_HUNGARIAN,
                )

                if STAGE2_DEBUG_BATCH_STATS:
                    log_stage2_batch_debug(
                        phase="train",
                        epoch=epoch,
                        batch_idx=step,
                        roi_mask=outputs["roi_mask"],
                        refine_targets=refine_targets,
                    )

                losses = _compute_losses(outputs, refine_targets, phase_settings)
                loss = losses["loss_total"] / GRADIENT_ACCUMULATION_STEPS

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            total_loss += float(losses["loss_total"].item())
            total_box += float(losses["loss_box"].item())
            total_delta += float(losses["loss_delta"].item())
            total_score += float(losses["loss_score"].item())
            total_aux += float(losses["loss_aux"].item())
            n_batches += 1

            if (step % train_prop_stats_every_n_steps) == 0:
                _update_refinement_epoch_stats(train_proposal_stats, outputs, refine_targets, gt_labels_list)

            if bool(STAGE2_ENABLE_TQDM) and (step % max(1, int(STAGE2_PROGRESS_POSTFIX_EVERY_N_STEPS)) == 0):
                pbar.set_postfix({
                    "loss": total_loss / max(1, n_batches),
                    "box": total_box / max(1, n_batches),
                    "delta": total_delta / max(1, n_batches),
                    "score": total_score / max(1, n_batches),
                    "aux": total_aux / max(1, n_batches),
                })

        if n_batches > 0 and (n_batches % GRADIENT_ACCUMULATION_STEPS) != 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()

        train_proposal_summary = _finalize_proposal_summary(train_proposal_stats, vocab, phase_settings)

        train_metrics = {
            "train_loss": total_loss / max(1, n_batches),
            "train_box": total_box / max(1, n_batches),
            "train_delta": total_delta / max(1, n_batches),
            "train_score": total_score / max(1, n_batches),
            "train_aux": total_aux / max(1, n_batches),
            "proposal_summary": {
                **train_proposal_summary,
                "orientation_counts": {
                    "horizontal": int(train_orientation_counts.get("horizontal", 0)),
                    "vertical": int(train_orientation_counts.get("vertical", 0)),
                    "other": int(train_orientation_counts.get("other", 0)),
                },
            },
        }

        val_metrics = validate_stage2(
            model=model,
            val_loader=val_loader,
            vocab=vocab,
            phase_settings=phase_settings,
            max_batches=val_max_batches,
        )

        print(
            f"\nEpoch {epoch+1}/{num_epochs} | "
            f"Train loss={train_metrics['train_loss']:.4f} "
            f"(box={train_metrics['train_box']:.4f}, "
            f"delta={train_metrics['train_delta']:.4f}, "
            f"score={train_metrics['train_score']:.4f}, "
            f"aux={train_metrics['train_aux']:.4f}) | "
            f"Val loss={val_metrics['val_loss']:.4f} "
            f"(box={val_metrics['val_box']:.4f}, "
            f"delta={val_metrics['val_delta']:.4f}, "
            f"score={val_metrics['val_score']:.4f}, "
            f"aux={val_metrics['val_aux']:.4f})"
        )

        val_prop = val_metrics["proposal_summary"]
        print(
            "Key val metrics | "
            f"uniq_cov={val_prop['unique_coverage_ratio']:.3f}, "
            f"aux_top1={val_prop['aux_accuracy_on_positives']:.3f}, "
            f"IoU+={val_prop['avg_matched_iou_on_positives']:.3f}"
        )

        current_score = _select_model_score(val_metrics)
        is_best = best_val is None or current_score > best_val
        if is_best:
            best_val = current_score
            best_val_metrics = val_metrics

        ckpt = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "vocab_size": vocab.vocab_size,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "is_best": is_best,
            "stage2_config": {
                "proj_dim": STAGE2_PROJ_DIM,
                "roi_size": STAGE2_ROI_SIZE,
                "roi_feat_dim": STAGE2_ROI_FEAT_DIM,
                "refine_hidden_dim": STAGE2_REFINE_HIDDEN_DIM,
                "token_dim": STAGE2_TOKEN_DIM,
                "token_hidden_dim": STAGE2_TOKEN_HIDDEN_DIM,
                "token_use_score_branch": STAGE2_TOKEN_USE_SCORE_BRANCH,
                "context_hidden_dim": STAGE2_CONTEXT_HIDDEN_DIM,
                "context_num_layers": STAGE2_CONTEXT_NUM_LAYERS,
                "context_mode": STAGE2_CONTEXT_MODE,
                "det_score_thresh": float(model.det_score_thresh),
                "det_top_k": int(model.det_top_k),
                "det_nms_iou": float(model.det_nms_iou),
                "det_min_box_size": float(model.det_min_box_size),
                "freeze_backbone": FREEZE_BACKBONE,
                "freeze_detector": FREEZE_DETECTOR,
            },
            "model_overrides": model_overrides or {},
        }

        epoch_path = checkpoint_dir / f"warmup_epoch{epoch+1}.pt"
        torch.save(ckpt, epoch_path)

        if is_best:
            torch.save(ckpt, checkpoint_dir / "warmup_best.pt")
            print(f"Saved best: warmup_best.pt (score={current_score:.4f})")

        prune_to_keep_last_n(checkpoint_dir, keep=2, exclude="checkpoint_old.pt")

    print("\n" + "=" * 70)
    print("WARMUP TRAINING COMPLETE")
    print(f"Best checkpoint: {checkpoint_dir / 'warmup_best.pt'}")
    print("=" * 70 + "\n")

    return {
        "model": model,
        "best_val_score": float(best_val) if best_val is not None else None,
        "best_val_metrics": best_val_metrics,
    }


def main():
    best_ckpt = CHECKPOINT_DIR / "stage1_detection" / "detector_best.pt"
    last_ckpt = CHECKPOINT_DIR / "stage1_detection" / f"detector_epoch{NUM_EPOCHS}.pt"

    detector_ckpt = best_ckpt if best_ckpt.exists() else last_ckpt
    if not detector_ckpt.exists():
        raise FileNotFoundError(
            f"No Stage-1 detector checkpoint found. Expected one of:\n"
            f"  - {best_ckpt}\n"
            f"  - {last_ckpt}"
        )

    phase_settings = get_warmup_settings()
    checkpoint_dir = CHECKPOINT_DIR / "suminanet_warmup"

    train_stage2_hybrid(
        detector_ckpt_path=detector_ckpt,
        num_epochs=phase_settings["epochs"],
        lr=LR,
        checkpoint_dir=checkpoint_dir,
        val_max_batches=VALIDATION_BATCHES,
    )


if __name__ == "__main__":
    main()
