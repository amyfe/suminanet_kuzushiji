# train_stage2_hybrid.py

from __future__ import annotations

from pathlib import Path
from collections import Counter
from typing import Optional

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    DATA_DIR,
    DEVICE,
    BATCH_SIZE,
    FREEZE_BACKBONE,
    FREEZE_DETECTOR,
    NUM_WORKERS,
    NUM_EPOCHS,
    LR,
    STAGE2_ACTION_ALIGNMENT_FULL_EPOCH,
    STAGE2_ACTION_ALIGNMENT_START_EPOCH,
    STAGE2_CONTEXT_HIDDEN_DIM,
    STAGE2_CONTEXT_NUM_LAYERS,
    STAGE2_DECODER_EMBED_DIM,
    STAGE2_DECODER_HIDDEN_DIM,
    STAGE2_DECODER_STOP_POS_WEIGHT,
    DET_MIN_BOX_SIZE,
    DET_NMS_IOU,
    DET_SCORE_THRESH,
    DET_TOP_K,
    STAGE2_LAMBDA_ACTION,
    STAGE2_LAMBDA_FREE_PREFIX,
    STAGE2_LAMBDA_STOP,
    STAGE2_FREE_PREFIX_EVERY_N_STEPS,
    STAGE2_TRAIN_FREE_DECODE_BATCHES,
    STAGE2_TRAIN_DECODER_STATS_EVERY_N_STEPS,
    STAGE2_TRAIN_DECODER_STATS_MAX_SAMPLES,
    STAGE2_TRAIN_PROP_STATS_EVERY_N_STEPS,
    STAGE2_TRAIN_FREE_DIAG_MAX_LEN,
    STAGE2_VAL_FREE_DECODE_BATCHES,
    STAGE2_FREE_RECOVERY_WINDOW,
    STAGE2_FREE_RECOVERY_WEIGHT,
    STAGE2_FREE_PREFIX_MIN_TOKENS,
    STAGE2_USE_CHUNKED_FREE_DECODE,
    STAGE2_LOG_PREDICTIONS,
    STAGE2_PREDICTION_SAMPLES,
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
    STAGE2_DECODER_LABEL_SMOOTHING,
    STAGE2_DECODER_EOS_WEIGHT,
    STAGE2_ACTION_AUX_WEIGHT,
    VALIDATION_BATCHES,
    STAGE2_TF_SCHEDULE,
    STAGE2_VAL_MAX_DECODE_LEN,
    STAGE2_DEBUG_BATCH_STATS,
    STAGE2_DEBUG_AUX_ALIGNMENT,
    STAGE2_DEBUG_AUX_ALIGNMENT_LIMIT,
    STAGE2_PHASE,
    STAGE2_PHASE_A_EPOCHS,
    STAGE2_PHASE_A_LAMBDA_BOX,
    STAGE2_PHASE_A_LAMBDA_DELTA,
    STAGE2_PHASE_A_LAMBDA_SCORE,
    STAGE2_PHASE_A_LAMBDA_AUX,
    STAGE2_PHASE_A_LAMBDA_DECODER,
    STAGE2_PHASE_A_TF_START,
    STAGE2_PHASE_A_TF_END,
    STAGE2_PHASE_B_EPOCHS,
    STAGE2_PHASE_B_LAMBDA_BOX,
    STAGE2_PHASE_B_LAMBDA_DELTA,
    STAGE2_PHASE_B_LAMBDA_SCORE,
    STAGE2_PHASE_B_LAMBDA_AUX,
    STAGE2_PHASE_B_LAMBDA_DECODER,
    STAGE2_PHASE_B_TF_START,
    STAGE2_PHASE_B_TF_END,
    STAGE2_PHASE_B_FREE_REPETITION_PENALTY,
    STAGE2_PHASE_B_FREE_BLOCK_IMMEDIATE_REPEAT,
    STAGE2_PHASE_B_FREE_MIN_STEPS,
    STAGE2_PHASE_B_FREE_MAX_STALL_STEPS,
    STAGE2_PHASE_B_FREE_FORCE_STOP_ON_STALL,
    STAGE2_PHASE_B_FREE_STOP_THRESHOLD,
    STAGE2_PHASE_B_FREE_MIN_PROGRESS_FOR_EOS,
    STAGE2_PHASE_B_FREE_MIN_COVERAGE_FOR_EOS,
    STAGE2_PHASE_B_TRAIN_FREE_REPETITION_PENALTY,
    STAGE2_PHASE_B_TRAIN_FREE_BLOCK_IMMEDIATE_REPEAT,
    STAGE2_PHASE_B_TRAIN_FREE_MIN_STEPS,
    STAGE2_PHASE_B_TRAIN_FREE_MAX_STALL_STEPS,
    STAGE2_PHASE_B_TRAIN_FREE_FORCE_STOP_ON_STALL,
    STAGE2_PHASE_B_TRAIN_FREE_STOP_THRESHOLD,
    STAGE2_PHASE_B_TRAIN_FREE_MIN_PROGRESS_FOR_EOS,
    STAGE2_PHASE_B_TRAIN_FREE_MIN_COVERAGE_FOR_EOS,
    STAGE2_USE_CHUNKED_FREE_TRAINING,
    STAGE2_FREE_TEACHER_PREFIX_STEPS,
    STAGE2_LAMBDA_STOP_FREE_SCALE,
    STAGE2_LAMBDA_ACTION_FREE_SCALE,
    
)

from model.kuronet import UNet, DetectorHead
from model.kuronet.hybrid_recognizer import HybridKuroNetRecognizer

from utils import KuzushijiDataset
from utils.training_helpers.helper_stage1 import (
    collate_fn,
    prune_existing_checkpoints,
    prune_to_keep_last_n,
    scheduled_teacher_forcing,
)
from utils.vocab import VocabManager
from utils.stage2_targets import (
    build_refinement_targets,
    build_decoder_targets,
)
from utils.stage2_losses import build_decoder_roi_targets_from_bias, compute_stage2_total_loss, compute_free_decoder_prefix_losses
from utils.stage2_losses import (
    build_decoder_roi_targets_monotonic,
    build_decoder_roi_targets_from_alignment,
)
from utils.stage2_losses import aux_classification_loss

from utils.training_helpers.helper_stage2 import (
    _decode_free_by_reading_units,
    _detach_encoded_for_free_decode,
    _get_action_class_weights,
    _load_compatible_state_dict,
    _normalize_orientation_label,
    _prepare_chunked_free_training_batch,
    compute_simple_token_accuracy,
    greedy_decode_from_logits,
    reorder_by_sort_indices,
)
from utils.training_helpers.logging_stage2 import (
    _debug_aux_alignment,
    _finalize_aux_branch_stats,
    _finalize_decoder_epoch_stats,
    _init_aux_branch_stats,
    _init_decoder_epoch_stats,
    _update_decoder_epoch_stats,
    _update_refinement_epoch_stats,
    log_stage2_batch_debug,
)
from utils.text_normalization import render_tokens

def get_phase_settings(phase: str, overrides: Optional[dict] = None) -> dict:
    overrides = overrides or {}
    phase = phase.upper()
    if phase == "A":
        settings = {
            "name": "A",
            "epochs": int(STAGE2_PHASE_A_EPOCHS),
            "lambda_box": float(STAGE2_PHASE_A_LAMBDA_BOX),
            "lambda_delta": float(STAGE2_PHASE_A_LAMBDA_DELTA),
            "lambda_score": float(STAGE2_PHASE_A_LAMBDA_SCORE),
            "lambda_aux": float(STAGE2_PHASE_A_LAMBDA_AUX),
            "lambda_decoder": float(STAGE2_PHASE_A_LAMBDA_DECODER),
            "tf_start": float(STAGE2_PHASE_A_TF_START),
            "tf_end": float(STAGE2_PHASE_A_TF_END),
            "tf_schedule": STAGE2_TF_SCHEDULE,
            "train_context_encoder": True,
            "log_free_decoder": False,
            "use_context_aux_for_loss": True,
            "raw_aux_weight": 0.0,
            "context_aux_weight": 1.0,
            "decode_constraints": None,
            "free_prefix_every_n_steps": 1,
            "train_free_decode_batches": 0,
            "val_free_decode_batches": 0,
        }
    elif phase == "B":
        settings = {
            "name": "B",
            "epochs": int(STAGE2_PHASE_B_EPOCHS),
            "lambda_box": float(STAGE2_PHASE_B_LAMBDA_BOX),
            "lambda_delta": float(STAGE2_PHASE_B_LAMBDA_DELTA),
            "lambda_score": float(STAGE2_PHASE_B_LAMBDA_SCORE),
            "lambda_aux": float(STAGE2_PHASE_B_LAMBDA_AUX),
            "lambda_decoder": float(STAGE2_PHASE_B_LAMBDA_DECODER),
            "lambda_free_prefix": float(STAGE2_LAMBDA_FREE_PREFIX),
            "tf_start": float(STAGE2_PHASE_B_TF_START),
            "tf_end": float(STAGE2_PHASE_B_TF_END),
            "tf_schedule": STAGE2_TF_SCHEDULE,
            "train_context_encoder": True,
            "log_free_decoder": True,
            "use_context_aux_for_loss": True,
            "raw_aux_weight": 0.5,
            "context_aux_weight": 0.5,
            "decode_constraints": {
                "repetition_penalty": float(STAGE2_PHASE_B_FREE_REPETITION_PENALTY),
                "block_immediate_repeat": bool(STAGE2_PHASE_B_FREE_BLOCK_IMMEDIATE_REPEAT),
                "min_steps": int(STAGE2_PHASE_B_FREE_MIN_STEPS),
                "max_stall_steps": int(STAGE2_PHASE_B_FREE_MAX_STALL_STEPS),
                "force_stop_at_last_pointer": True,
                "force_stop_on_stall": bool(STAGE2_PHASE_B_FREE_FORCE_STOP_ON_STALL),
                "require_stop_for_eos": False,
                "stop_threshold": float(STAGE2_PHASE_B_FREE_STOP_THRESHOLD),
                "eos_bonus_scale": 0.0,
                "min_progress_for_eos": float(STAGE2_PHASE_B_FREE_MIN_PROGRESS_FOR_EOS),
                "min_coverage_for_eos": float(STAGE2_PHASE_B_FREE_MIN_COVERAGE_FOR_EOS),
                "aux_bias_topk": 0,
            },
            "train_decode_constraints": {
                "repetition_penalty": float(STAGE2_PHASE_B_TRAIN_FREE_REPETITION_PENALTY),
                "block_immediate_repeat": bool(STAGE2_PHASE_B_TRAIN_FREE_BLOCK_IMMEDIATE_REPEAT),
                "min_steps": int(STAGE2_PHASE_B_TRAIN_FREE_MIN_STEPS),
                "max_stall_steps": int(STAGE2_PHASE_B_TRAIN_FREE_MAX_STALL_STEPS),
                "force_stop_at_last_pointer": True,
                "force_stop_on_stall": bool(STAGE2_PHASE_B_TRAIN_FREE_FORCE_STOP_ON_STALL),
                "require_stop_for_eos": False,
                "stop_threshold": float(STAGE2_PHASE_B_TRAIN_FREE_STOP_THRESHOLD),
                "eos_bonus_scale": 0.0,
                "min_progress_for_eos": float(STAGE2_PHASE_B_TRAIN_FREE_MIN_PROGRESS_FOR_EOS),
                "min_coverage_for_eos": float(STAGE2_PHASE_B_TRAIN_FREE_MIN_COVERAGE_FOR_EOS),
                "aux_bias_topk": 0,
            },
            "free_prefix_every_n_steps": int(overrides.get("free_prefix_every_n_steps", STAGE2_FREE_PREFIX_EVERY_N_STEPS)),
            "train_free_decode_batches": int(overrides.get("train_free_decode_batches", STAGE2_TRAIN_FREE_DECODE_BATCHES)),
            "val_free_decode_batches": int(overrides.get("val_free_decode_batches", STAGE2_VAL_FREE_DECODE_BATCHES)),
            "free_stop_loss_scale": float(overrides.get("free_stop_loss_scale", STAGE2_LAMBDA_STOP_FREE_SCALE)),
        }

        dc = settings["decode_constraints"]
        train_dc = settings["train_decode_constraints"]
        if "repetition_penalty" in overrides:
            dc["repetition_penalty"] = float(overrides["repetition_penalty"])
        if "block_immediate_repeat" in overrides:
            dc["block_immediate_repeat"] = bool(overrides["block_immediate_repeat"])
        if "min_steps" in overrides:
            dc["min_steps"] = int(overrides["min_steps"])
        if "train_repetition_penalty" in overrides:
            train_dc["repetition_penalty"] = float(overrides["train_repetition_penalty"])
        if "train_block_immediate_repeat" in overrides:
            train_dc["block_immediate_repeat"] = bool(overrides["train_block_immediate_repeat"])
        if "train_min_steps" in overrides:
            train_dc["min_steps"] = int(overrides["train_min_steps"])

        return settings
    else:
        raise ValueError(f"Unsupported phase '{phase}'. Use 'A' or 'B'.")
    return settings


def _strip_special_tokens(ids: list[int], pad_id: int, sos_id: int, eos_id: int) -> list[int]:
    out: list[int] = []
    for token_id in ids:
        token_id = int(token_id)
        if token_id in (pad_id, sos_id):
            continue
        if token_id == eos_id:
            break
        out.append(token_id)
    return out
    


def _ids_to_text(ids: list[int], vocab: VocabManager) -> str:
    return render_tokens(vocab.decode(ids, remove_special=True))

def _select_active_aux_summary(
    aux_without_ctx: dict,
    aux_with_ctx: dict,
    use_context_aux_for_loss: bool,
) -> dict:
    return aux_with_ctx if use_context_aux_for_loss else aux_without_ctx

def _select_model_score(val_metrics: dict, phase: str) -> float:
    phase = phase.upper()
    prop = val_metrics["proposal_summary"]
    aux_summary = prop["aux_summary"]
    decoder_summary = val_metrics["decoder_summary"]["teacher_forcing"]

    active_aux = aux_summary["with_context_encode"]
    if phase == "A":
        return (
            2.0 * float(prop["unique_coverage_ratio"])
            + 1.0 * float(active_aux["top1"])
            + 0.5 * float(active_aux["top5"])
            + 0.25 * float(prop["avg_matched_iou_on_positives"])
            - 0.02 * float(prop["avg_negatives_per_image"])
        )

    if phase == "B":
        free_summary = val_metrics["decoder_summary"].get("free_decoding", None)
        free_samples = int(free_summary.get("samples", 0)) if free_summary is not None else 0
        if free_summary is None or free_samples <= 0:
            # Fallback for runs where free-decoding diagnostics are disabled.
            # Avoid rewarding missing FREE metrics as if they were perfect.
            return (
                -1.10 * float(decoder_summary["mean_cer"])
                + 0.40 * float(decoder_summary["eos_hit_fraction"])
                - 0.20 * abs(float(decoder_summary["mean_length_ratio"]) - 1.0)
                + 0.20 * float(active_aux["top1"])
                + 0.10 * float(prop["unique_coverage_ratio"])
            )

        free_len_ratio = float(free_summary["mean_length_ratio"]) if free_summary is not None else 1.0
        free_dom = float(free_summary.get("mean_dominant_share", 0.0))
        return (
            -1.25 * float(free_summary["mean_cer"])
            + 0.90 * float(free_summary["eos_hit_fraction"])
            - 0.40 * abs(free_len_ratio - 1.0)
            - 0.10 * free_dom
            + 0.20 * float(active_aux["top1"])
            + 0.10 * float(prop["unique_coverage_ratio"])

        )

    raise ValueError(f"Unsupported phase: {phase}")

def set_trainable_modules_for_phase(model: HybridKuroNetRecognizer, phase: str) -> None:
    phase = phase.upper()

    # default: everything off except backbone/detector policy handled elsewhere
    for name, p in model.named_parameters():
        if name.startswith("backbone.") or name.startswith("detector.") or name.startswith("aux_head."):
            continue
        p.requires_grad = False

    # always train ROI pipeline
    modules_phase_a = [
        model.feature_projector,
        model.roi_pool,
        model.roi_refine,
        model.roi_tokens,
    ]

    for module in modules_phase_a:
        for p in module.parameters():
            p.requires_grad = True

    phase_settings = get_phase_settings(phase)
    if phase == "A":
        # optional: train context encoder too, but I would start without it
        if hasattr(model.roi_pool, "aux_head") and model.roi_pool.aux_head is not None:
            for p in model.roi_pool.aux_head.parameters():
                p.requires_grad = True

        if hasattr(model, "aux_head_context") and model.aux_head_context is not None:
            for p in model.aux_head_context.parameters():
                p.requires_grad = bool(phase_settings["train_context_encoder"])

        # keep context and decoder frozen in first pass
        for p in model.context_encoder.parameters():
            p.requires_grad = bool(phase_settings["train_context_encoder"])
        for p in model.decoder.parameters():
            p.requires_grad = False

    elif phase == "B":
        for p in model.context_encoder.parameters():
            p.requires_grad = bool(phase_settings["train_context_encoder"])
        for p in model.decoder.parameters():
            p.requires_grad = True

        if hasattr(model.roi_pool, "aux_head") and model.roi_pool.aux_head is not None:
            for p in model.roi_pool.aux_head.parameters():
                p.requires_grad = True
        if hasattr(model, "aux_head_context") and model.aux_head_context is not None:
            for p in model.aux_head_context.parameters():
                p.requires_grad = True
    else:
        raise ValueError(f"Unsupported phase '{phase}'")


def _compute_phase_aux_loss(
    outputs: dict,
    refine_targets: dict,
    phase_settings: dict,
) -> tuple[torch.Tensor, dict]:
    """
    Phase-aware auxiliary classification loss.
    Supports raw aux, context aux, or a weighted combination.

    Returns:
        total_aux_loss
        aux_parts = {
            "loss_aux_raw": ...,
            "loss_aux_ctx": ...,
        }
    """
    device = outputs["refined_boxes"].device
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


def build_stage2_model(
    detector_ckpt_path: str | Path,
    vocab: VocabManager,
    overrides: Optional[dict] = None,
) -> HybridKuroNetRecognizer:
    overrides = overrides or {}
    vocab_size = vocab.vocab_size

    det_score_thresh = float(overrides.get("det_score_thresh", DET_SCORE_THRESH))
    det_top_k = int(overrides.get("det_top_k", DET_TOP_K))
    det_nms_iou = float(overrides.get("det_nms_iou", DET_NMS_IOU))
    det_min_box_size = float(overrides.get("det_min_box_size", DET_MIN_BOX_SIZE))

    token_dim = int(overrides.get("token_dim", STAGE2_TOKEN_DIM))
    token_hidden_dim = int(overrides.get("token_hidden_dim", STAGE2_TOKEN_HIDDEN_DIM))
    token_use_score_branch = bool(overrides.get("token_use_score_branch", STAGE2_TOKEN_USE_SCORE_BRANCH))

    context_hidden_dim = int(overrides.get("context_hidden_dim", STAGE2_CONTEXT_HIDDEN_DIM))
    context_num_layers = int(overrides.get("context_num_layers", STAGE2_CONTEXT_NUM_LAYERS))

    backbone = UNet(in_channels=3, base_features=32).to(DEVICE)
    detector = DetectorHead(
        in_ch=32,
        num_classes=vocab_size,
        dropout_rate=STAGE2_DROPOUT_RATE,
        predict_boxes=True,
        predict_classes=False,
    ).to(DEVICE)

    checkpoint = torch.load(detector_ckpt_path, map_location=DEVICE)
    backbone.load_state_dict(checkpoint["unet_state_dict"])
    detector.load_state_dict(checkpoint["detector_state_dict"])

    model = HybridKuroNetRecognizer(
        backbone=backbone,
        detector=detector,
        backbone_out_channels=32,
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
        decoder_embed_dim=STAGE2_DECODER_EMBED_DIM,
        decoder_hidden_dim=STAGE2_DECODER_HIDDEN_DIM,

        det_score_thresh=det_score_thresh,
        det_top_k=det_top_k,
        det_nms_iou=det_nms_iou,
        det_min_box_size=det_min_box_size,

        use_aux_head=STAGE2_USE_AUX_HEAD,
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

    if "bias_scale" in overrides:
        with torch.no_grad():
            model.decoder.bias_scale.fill_(float(overrides["bias_scale"]))

    return model


def get_trainable_parameters(model: HybridKuroNetRecognizer):
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



@torch.no_grad()
def validate_stage2(
    model: HybridKuroNetRecognizer,
    val_loader: DataLoader,
    vocab: VocabManager,
    phase: str = "A",
    max_batches: Optional[int] = None,
    runtime_overrides: Optional[dict] = None,
    current_epoch: Optional[int] = None,
    total_epochs: Optional[int] = None,
):
    model.eval()
    if FREEZE_BACKBONE:
        model.backbone.eval()
    if FREEZE_DETECTOR:
        model.detector.eval()
    runtime_overrides = runtime_overrides or {}
    phase_settings = get_phase_settings(phase, overrides=runtime_overrides)
    lambda_stop = float(runtime_overrides.get("lambda_stop", STAGE2_LAMBDA_STOP))
    decoder_eos_weight = float(runtime_overrides.get("decoder_eos_weight", STAGE2_DECODER_EOS_WEIGHT))
    val_max_decode_len = int(runtime_overrides.get("val_max_decode_len", STAGE2_VAL_MAX_DECODE_LEN))
    val_free_decode_batches = int(phase_settings.get("val_free_decode_batches", STAGE2_VAL_FREE_DECODE_BATCHES))
    total_loss = 0.0
    total_box = 0.0
    total_delta = 0.0
    total_score = 0.0
    total_aux = 0.0
    total_decoder = 0.0
    total_acc = 0.0
    total_action = 0.0
    val_action_alignment_mix = 0.0
    n_batches = 0
    total_stop = 0.0

    proposal_stats = {
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

    decoder_stats_tf = _init_decoder_epoch_stats(example_limit=3, pointer_example_limit=3)
    decoder_stats_free = _init_decoder_epoch_stats(example_limit=3, pointer_example_limit=3)
    val_orientation_counts: Counter[str] = Counter()

    pad_id = vocab.pad_id
    sos_id = vocab.sos_id
    eos_id = vocab.eos_id

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
        images = batch["image"].to(DEVICE, non_blocking=True)
        text_ids = batch["text_ids"]
        if text_ids is None:
            continue
        text_ids = text_ids.to(DEVICE, non_blocking=True)
        orientations = batch["orientations"]
        val_orientation_counts.update(_normalize_orientation_label(x) for x in orientations)

        gt_boxes_list = move_gt_lists_to_device(batch["boxes"], dtype=torch.float32)
        gt_labels_list = move_gt_lists_to_device(batch["labels"], dtype=torch.long)

        dec_targets = build_decoder_targets(
            text_ids=text_ids,
            pad_id=pad_id,
        )

        encoded = model.encode_images(
            images=images,
            orientations=orientations,
        )
        decoder_token_bias = encoded.get("decoder_token_bias", None)
        if decoder_token_bias is not None:
            oracle_pointer_positions = build_decoder_roi_targets_from_bias(
                target_tokens=dec_targets["target_tokens"],
                target_mask=dec_targets["target_mask"],
                enc_mask=encoded.get("context_mask", None),
                eos_id=int(eos_id),
                encoder_token_bias=decoder_token_bias,
                aux_weight=0.35,
            )
        else:
            oracle_pointer_positions = build_decoder_roi_targets_monotonic(
                target_tokens=dec_targets["target_tokens"],
                target_mask=dec_targets["target_mask"],
                enc_mask=encoded.get("context_mask", None),
                eos_id=int(eos_id),
            )

        outputs = model.decode_from_encoded(
            encoded,
            input_seq=dec_targets["decoder_inputs"],
            targets=dec_targets["target_tokens"],
            teacher_forcing_ratio=1.0,   # oder 1.0 für echte teacher-forced diagnostics
            sos_id=sos_id,
            eos_id=eos_id,
            max_len=None,
            oracle_pointer_positions=oracle_pointer_positions,
        )

        refine_targets = build_refinement_targets(
            coarse_boxes=outputs["roi_boxes"],
            roi_mask=outputs["roi_mask"],
            gt_boxes_list=gt_boxes_list,
            gt_labels_list=gt_labels_list,
            pos_iou_thresh=STAGE2_REFINE_POS_IOU,
            neg_iou_thresh=STAGE2_REFINE_NEG_IOU,
        )

        if STAGE2_DEBUG_BATCH_STATS:
            log_stage2_batch_debug(
                phase="val",
                epoch=None,
                batch_idx=batch_idx,
                roi_mask=outputs["roi_mask"],
                refine_targets=refine_targets,
            )

        base_losses = compute_stage2_total_loss(
                refined_boxes=outputs["refined_boxes"],
                box_deltas=outputs["box_deltas"],
                refine_scores=outputs["refine_scores"],
                aux_logits=None,
                decoder_logits=outputs["decoder_logits"],
                stop_logits=outputs.get("stop_logits", None),
                action_logits=outputs.get("action_logits", None),
                attn_weights=outputs.get("attn_weights", None),
                enc_mask=outputs.get("context_mask", None),
                encoder_token_bias=outputs.get("decoder_token_bias", None),

                matched_gt_boxes=refine_targets["matched_gt_boxes"],
                target_deltas=refine_targets["target_deltas"],
                matched_gt_labels=refine_targets["matched_gt_labels"],
                refine_pos_mask=refine_targets["refine_pos_mask"],
                refine_neg_mask=refine_targets["refine_neg_mask"],
                refine_ignore_mask=refine_targets["refine_ignore_mask"],

                target_tokens=dec_targets["target_tokens"],
                target_mask=dec_targets["target_mask"],

                lambda_box=phase_settings["lambda_box"],
                lambda_delta=phase_settings["lambda_delta"],
                lambda_score=phase_settings["lambda_score"],
                lambda_aux=0.0,
                lambda_decoder=phase_settings["lambda_decoder"],
                lambda_stop=lambda_stop,
                lambda_action= STAGE2_LAMBDA_ACTION ,
                action_aux_weight=STAGE2_ACTION_AUX_WEIGHT,
                action_class_weights=_get_action_class_weights(
                    device=outputs["decoder_logits"].device,
                    dtype=outputs["decoder_logits"].dtype,
                ),
                current_epoch=(0 if current_epoch is None else int(current_epoch)),
                total_epochs=(
                    int(phase_settings.get("epochs", 1))
                    if total_epochs is None
                    else int(total_epochs)
                ),
                action_alignment_start_epoch=STAGE2_ACTION_ALIGNMENT_START_EPOCH,
                action_alignment_full_epoch=STAGE2_ACTION_ALIGNMENT_FULL_EPOCH,
                decoder_stop_pos_weight=STAGE2_DECODER_STOP_POS_WEIGHT,
                refine_pos_weight=STAGE2_REFINE_POS_WEIGHT,
                decoder_label_smoothing=STAGE2_DECODER_LABEL_SMOOTHING,
                decoder_eos_id=eos_id,
                decoder_eos_weight=decoder_eos_weight,
            )

        aux_total, aux_parts = _compute_phase_aux_loss(
            outputs=outputs,
            refine_targets=refine_targets,
            phase_settings=phase_settings,
        )

        losses = dict(base_losses)
        losses["loss_aux"] = phase_settings["lambda_aux"] * aux_total
        losses["loss_aux_raw"] = aux_parts["loss_aux_raw"]
        losses["loss_aux_ctx"] = aux_parts["loss_aux_ctx"]
        losses["loss_total"] = base_losses["loss_total"] + losses["loss_aux"]
        val_action_alignment_mix = float(losses.get("action_alignment_mix", 0.0))

        if STAGE2_DEBUG_AUX_ALIGNMENT and batch_idx == 0:
            _debug_aux_alignment(
                outputs=outputs,
                refine_targets=refine_targets,
                vocab=vocab,
                limit=int(STAGE2_DEBUG_AUX_ALIGNMENT_LIMIT),
            )

        decoder_active = phase_settings["lambda_decoder"] > 0.0

        if decoder_active:
            pred_ids = greedy_decode_from_logits(outputs["decoder_logits"])
            batch_acc = compute_simple_token_accuracy(
                pred_ids=pred_ids,
                tgt_ids=dec_targets["target_tokens"],
                tgt_mask=dec_targets["target_mask"],
            )
        else:
            pred_ids = None
            batch_acc = 0.0

        total_loss += float(losses["loss_total"].item())
        total_box += float(losses["loss_box"].item())
        total_delta += float(losses["loss_delta"].item())
        total_score += float(losses["loss_score"].item())
        total_aux += float(losses["loss_aux"].item())
        total_decoder += float(losses["loss_decoder"].item())
        total_stop += float(losses.get("loss_stop", torch.tensor(0.0)).item())
        total_action += float(losses.get("loss_action", torch.tensor(0.0)).item())
        total_acc += batch_acc
        n_batches += 1

        _update_refinement_epoch_stats(proposal_stats, outputs, refine_targets, gt_labels_list)
        if pred_ids is not None:
            _update_decoder_epoch_stats(
                decoder_stats_tf,
                pred_ids,
                dec_targets["target_tokens"],
                vocab,
                action_logits_batch=outputs.get("action_logits", None),
                pointer_positions_batch=outputs.get("pointer_positions", None),
                valid_mask_batch=dec_targets["target_mask"],
                action_targets_batch=losses.get("action_targets", None),
            )
        if phase_settings["log_free_decoder"] and batch_idx < val_free_decode_batches:
            free_decoded = _decode_free_by_reading_units(
                model,
                encoded,
                orientations,
                vocab,
                sos_id=sos_id,
                eos_id=eos_id,
                max_len=val_max_decode_len,
                decode_constraints=phase_settings.get("decode_constraints"),
            )
            _update_decoder_epoch_stats(
                decoder_stats_free,
                free_decoded["pred_ids"],
                text_ids,
                vocab,
                action_logits_batch=free_decoded.get("action_logits", None),
                pointer_positions_batch=free_decoded.get("pointer_positions", None),
                valid_mask_batch=free_decoded.get("valid_mask", None),
                action_targets_batch=None,
            )
            if bool(STAGE2_LOG_PREDICTIONS) and batch_idx == 0:
                sample_limit = max(1, int(STAGE2_PREDICTION_SAMPLES))
                pred_free = free_decoded["pred_ids"].detach().cpu()
                pred_tf = pred_ids.detach().cpu() if pred_ids is not None else None
                gt_ids = text_ids.detach().cpu()
                for s in range(min(sample_limit, pred_free.size(0))):
                    gt_list = _strip_special_tokens(gt_ids[s].tolist(), pad_id, sos_id, eos_id)
                    free_list = _strip_special_tokens(pred_free[s].tolist(), pad_id, sos_id, eos_id)
                    free_text = _ids_to_text(free_list, vocab)
                    gt_text = _ids_to_text(gt_list, vocab)
                    msg = f"[VAL PRED] sample={s} | FREE: {free_text} | GT: {gt_text}"
                    if pred_tf is not None:
                        tf_list = _strip_special_tokens(pred_tf[s].tolist(), pad_id, sos_id, eos_id)
                        tf_text = _ids_to_text(tf_list, vocab)
                        msg = f"[VAL PRED] sample={s} | TF: {tf_text} | FREE: {free_text} | GT: {gt_text}"
                    print(msg)
        
    denom = max(1, n_batches)
    proposal_denom_images = max(1, proposal_stats["images"])
    proposal_pos = proposal_stats["positives"]
    proposal_neg = proposal_stats["negatives"]
    proposal_ign = proposal_stats["ignores"]
    aux_without_ctx = _finalize_aux_branch_stats(
        proposal_stats["aux_without_context_encode"],
        vocab=vocab,
        available=proposal_stats["aux_without_context_encode"]["total"] > 0,
    )
    aux_with_ctx = _finalize_aux_branch_stats(
        proposal_stats["aux_with_context_encode"],
        vocab=vocab,
        available=proposal_stats["aux_with_context_encode"]["total"] > 0,
    )
    active_aux = _select_active_aux_summary(
        aux_without_ctx=aux_without_ctx,
        aux_with_ctx=aux_with_ctx,
        use_context_aux_for_loss=phase_settings["use_context_aux_for_loss"],
    )
    return {
        "val_loss": total_loss / denom,
        "val_box": total_box / denom,
        "val_delta": total_delta / denom,
        "val_score": total_score / denom,
        "val_aux": total_aux / denom,
        "val_decoder": total_decoder / denom,
        "val_token_acc": total_acc / denom,
        "val_stop": total_stop / denom,
        "val_action": total_action / denom,
        "val_action_alignment_mix": val_action_alignment_mix if n_batches > 0 else 0.0,
        "proposal_summary": {
            "images": proposal_stats["images"],
            "orientation_counts": {
                "horizontal": int(val_orientation_counts.get("horizontal", 0)),
                "vertical": int(val_orientation_counts.get("vertical", 0)),
                "other": int(val_orientation_counts.get("other", 0)),
            },
            "avg_proposals_per_image": proposal_stats["proposals"] / proposal_denom_images,
            "avg_positives_per_image": proposal_pos / proposal_denom_images,
            "avg_gt_tokens_per_image": proposal_stats["gt_tokens"] / proposal_denom_images,
            "positive_coverage_ratio": proposal_pos / max(1, proposal_stats["gt_tokens"]),
            "unique_coverage_ratio": proposal_stats["unique_gt_matched"] / max(1, proposal_stats["gt_tokens"]),
            "avg_unique_gt_matched_per_image": proposal_stats["unique_gt_matched"] / proposal_denom_images,
            "avg_duplicate_positive_matches_per_image": proposal_stats["duplicate_positive_matches"] / proposal_denom_images,
            "duplicate_positive_rate": proposal_stats["duplicate_positive_matches"] / max(1, proposal_pos),
            "images_with_zero_valid_props_ratio": proposal_stats["images_with_zero_valid_props"] / proposal_denom_images,
            "positive_precision_proxy": proposal_pos / max(1, proposal_stats["proposals"]),
            "avg_negatives_per_image": proposal_neg / proposal_denom_images,
            "avg_ignores_per_image": proposal_ign / proposal_denom_images,
            "avg_matched_iou_on_positives": proposal_stats["matched_iou_sum"] / max(1, proposal_stats["matched_iou_count"]),
            "ordering_primary_monotonic_fraction": proposal_stats["ordering_primary_mono_sum"] / max(1.0, proposal_stats["ordering_weight_sum"]),
            "ordering_primary_violation_fraction": proposal_stats["ordering_primary_viol_sum"] / max(1.0, proposal_stats["ordering_weight_sum"]),
            "refine_score_logit_mean": proposal_stats["score_logit_sum"] / max(1, proposal_stats["score_count"]),
            "refine_score_logit_std": (
                max(
                    0.0,
                    proposal_stats["score_logit_sq_sum"] / max(1, proposal_stats["score_count"])
                    - (proposal_stats["score_logit_sum"] / max(1, proposal_stats["score_count"])) ** 2,
                )
            ) ** 0.5,
            "refine_score_prob_mean": proposal_stats["score_prob_sum"] / max(1, proposal_stats["score_count"]),
            "refine_score_prob_std": (
                max(
                    0.0,
                    proposal_stats["score_prob_sq_sum"] / max(1, proposal_stats["score_count"])
                    - (proposal_stats["score_prob_sum"] / max(1, proposal_stats["score_count"])) ** 2,
                )
            ) ** 0.5,
            "aux_accuracy_on_positives": active_aux["top1"],
            "aux_top5_on_positives": active_aux["top5"],
            "active_aux_branch": "with_context_encode" if phase_settings["use_context_aux_for_loss"] else "without_context_encode",
            "aux_summary": {
                "without_context_encode": aux_without_ctx,
                "with_context_encode": aux_with_ctx,
            },
        },
        "decoder_summary": {
            "teacher_forcing": _finalize_decoder_epoch_stats(decoder_stats_tf, vocab),
            "free_decoding": _finalize_decoder_epoch_stats(decoder_stats_free, vocab),
        },
    }


def train_stage2_hybrid(
    detector_ckpt_path: str | Path,
    num_epochs: int = NUM_EPOCHS,
    lr: Optional[float] = None,
    checkpoint_dir: Optional[str | Path] = None,
    phase: str = "A",
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
    phase_settings = get_phase_settings(phase, overrides=model_overrides)
    lambda_stop = float(model_overrides.get("lambda_stop", STAGE2_LAMBDA_STOP))
    decoder_eos_weight = float(model_overrides.get("decoder_eos_weight", STAGE2_DECODER_EOS_WEIGHT))
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
    set_trainable_modules_for_phase(model, phase)
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

    scaler = torch.cuda.amp.GradScaler() if USE_MIXED_PRECISION and str(DEVICE).startswith("cuda") else None

    pad_id = vocab.pad_id
    sos_id = vocab.sos_id
    eos_id = vocab.eos_id

    best_val = None
    best_val_metrics = None

    print("=" * 70)
    print("STAGE 2 HYBRID TRAINING (OPTION C)")
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

        # tf_ratio = scheduled_teacher_forcing(
        #     epoch=epoch,
        #     total_epochs=num_epochs,
        #     start=STAGE2_TF_START,
        #     end=STAGE2_TF_END,
        #     schedule=STAGE2_TF_SCHEDULE,
        # )
        tf_ratio = scheduled_teacher_forcing(
            epoch=epoch,
            total_epochs=num_epochs,
            start=phase_settings["tf_start"],
            end=phase_settings["tf_end"],
            schedule=phase_settings["tf_schedule"],
        )
        optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        total_box = 0.0
        total_delta = 0.0
        total_score = 0.0
        total_aux = 0.0
        total_decoder = 0.0
        total_decoder_free = 0.0
        total_acc = 0.0
        total_action = 0.0
        total_action_free = 0.0
        train_action_alignment_mix = 0.0
        n_batches = 0
        total_stop = 0.0
        total_stop_free = 0.0
        total_free_prefix_tokens = 0.0
        total_free_control_tokens = 0.0
        total_free_target_tokens = 0.0
        total_free_prefix_sequences = 0.0
        total_free_recovery_tokens = 0.0
        total_free_align_ratio = 0.0
        total_free_align_samples = 0.0

        train_proposal_stats = {
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
        train_orientation_counts: Counter[str] = Counter()
        decoder_stats_tf = _init_decoder_epoch_stats(example_limit=3, pointer_example_limit=3)
        decoder_stats_free = _init_decoder_epoch_stats(example_limit=3, pointer_example_limit=3)
        train_free_eval_batches = int(phase_settings.get("train_free_decode_batches", STAGE2_TRAIN_FREE_DECODE_BATCHES))
        free_prefix_every_n_steps = max(1, int(phase_settings.get("free_prefix_every_n_steps", STAGE2_FREE_PREFIX_EVERY_N_STEPS)))
        train_decoder_stats_every_n_steps = max(1, int(model_overrides.get("train_decoder_stats_every_n_steps", STAGE2_TRAIN_DECODER_STATS_EVERY_N_STEPS)))
        train_decoder_stats_max_samples = int(model_overrides.get("train_decoder_stats_max_samples", STAGE2_TRAIN_DECODER_STATS_MAX_SAMPLES))
        train_prop_stats_every_n_steps = max(1, int(model_overrides.get("train_prop_stats_every_n_steps", STAGE2_TRAIN_PROP_STATS_EVERY_N_STEPS)))
        train_free_diag_max_len = max(16, int(model_overrides.get("train_free_diag_max_len", STAGE2_TRAIN_FREE_DIAG_MAX_LEN)))

        pbar = tqdm(
            train_loader,
            desc=f"Stage2 Epoch {epoch+1}/{num_epochs}",
            mininterval=10.0,
            disable=not bool(STAGE2_ENABLE_TQDM),
        )

        for step, batch in enumerate(pbar):
            images = batch["image"].to(DEVICE, non_blocking=True)
            text_ids = batch["text_ids"]
            if text_ids is None:
                continue
            text_ids = text_ids.to(DEVICE, non_blocking=True)

            orientations = batch["orientations"]
            train_orientation_counts.update(_normalize_orientation_label(x) for x in orientations)
            gt_boxes_list = move_gt_lists_to_device(batch["boxes"], dtype=torch.float32)
            gt_labels_list = move_gt_lists_to_device(batch["labels"], dtype=torch.long)

            dec_targets = build_decoder_targets(
                text_ids=text_ids,
                pad_id=pad_id,
            )

            with torch.cuda.amp.autocast(enabled=USE_MIXED_PRECISION and str(DEVICE).startswith("cuda")):
                encoded = model.encode_images(
                    images=images,
                    orientations=orientations,
                )
                decoder_token_bias = encoded.get("decoder_token_bias", None)
                if decoder_token_bias is not None:
                    oracle_pointer_positions = build_decoder_roi_targets_from_bias(
                        target_tokens=dec_targets["target_tokens"],
                        target_mask=dec_targets["target_mask"],
                        enc_mask=encoded.get("context_mask", None),
                        eos_id=int(eos_id),
                        encoder_token_bias=decoder_token_bias,
                        aux_weight=0.35,
                    )
                else:
                    oracle_pointer_positions = build_decoder_roi_targets_monotonic(
                        target_tokens=dec_targets["target_tokens"],
                        target_mask=dec_targets["target_mask"],
                        enc_mask=encoded.get("context_mask", None),
                        eos_id=int(eos_id),
                    )

                outputs = model.decode_from_encoded(
                    encoded,
                    input_seq=dec_targets["decoder_inputs"],
                    targets=dec_targets["target_tokens"],
                    teacher_forcing_ratio=tf_ratio,
                    sos_id=sos_id,
                    eos_id=eos_id,
                    max_len=None,
                    oracle_pointer_positions=oracle_pointer_positions,
                )
                refine_targets = build_refinement_targets(
                    coarse_boxes=outputs["roi_boxes"],
                    roi_mask=outputs["roi_mask"],
                    gt_boxes_list=gt_boxes_list,
                    gt_labels_list=gt_labels_list,
                    pos_iou_thresh=STAGE2_REFINE_POS_IOU,
                    neg_iou_thresh=STAGE2_REFINE_NEG_IOU,
                )

                if STAGE2_DEBUG_BATCH_STATS:
                    log_stage2_batch_debug(
                        phase="train",
                        epoch=epoch,
                        batch_idx=step,
                        roi_mask=outputs["roi_mask"],
                        refine_targets=refine_targets,
                    )

                base_losses = compute_stage2_total_loss(
                    refined_boxes=outputs["refined_boxes"],
                    box_deltas=outputs["box_deltas"],
                    refine_scores=outputs["refine_scores"],
                    aux_logits=None,
                    decoder_logits=outputs["decoder_logits"],
                    stop_logits=outputs.get("stop_logits", None),
                    action_logits=outputs.get("action_logits", None),
                    attn_weights=outputs.get("attn_weights", None),
                    enc_mask=outputs.get("context_mask", None),
                    encoder_token_bias=outputs.get("decoder_token_bias", None),

                    matched_gt_boxes=refine_targets["matched_gt_boxes"],
                    target_deltas=refine_targets["target_deltas"],
                    matched_gt_labels=refine_targets["matched_gt_labels"],
                    refine_pos_mask=refine_targets["refine_pos_mask"],
                    refine_neg_mask=refine_targets["refine_neg_mask"],
                    refine_ignore_mask=refine_targets["refine_ignore_mask"],

                    target_tokens=dec_targets["target_tokens"],
                    target_mask=dec_targets["target_mask"],

                    lambda_box=phase_settings["lambda_box"],
                    lambda_delta=phase_settings["lambda_delta"],
                    lambda_score=phase_settings["lambda_score"],
                    lambda_aux=0.0,
                    lambda_decoder=phase_settings["lambda_decoder"],
                    lambda_stop=lambda_stop,
                    lambda_action= STAGE2_LAMBDA_ACTION ,
                    action_aux_weight=STAGE2_ACTION_AUX_WEIGHT,
                    action_class_weights=_get_action_class_weights(
                        device=outputs["decoder_logits"].device,
                        dtype=outputs["decoder_logits"].dtype,
                    ),
                    current_epoch=epoch,
                    total_epochs=num_epochs,
                    action_alignment_start_epoch=STAGE2_ACTION_ALIGNMENT_START_EPOCH,
                    action_alignment_full_epoch=STAGE2_ACTION_ALIGNMENT_FULL_EPOCH,
                    refine_pos_weight=STAGE2_REFINE_POS_WEIGHT,
                    decoder_label_smoothing=STAGE2_DECODER_LABEL_SMOOTHING,
                    decoder_eos_id=eos_id,
                    decoder_eos_weight=decoder_eos_weight,
                    decoder_stop_pos_weight=STAGE2_DECODER_STOP_POS_WEIGHT,
                )

                aux_total, aux_parts = _compute_phase_aux_loss(
                    outputs=outputs,
                    refine_targets=refine_targets,
                    phase_settings=phase_settings,
                )
                free_loss_dict = {
                    "loss_decoder_free": base_losses["loss_total"].new_tensor(0.0),
                    "loss_stop_free": base_losses["loss_total"].new_tensor(0.0),
                    "loss_action_free": base_losses["loss_total"].new_tensor(0.0),
                    "loss_total_free": base_losses["loss_total"].new_tensor(0.0),
                    "prefix_decoder_tokens": base_losses["loss_total"].new_tensor(0, dtype=torch.long),
                    "prefix_control_tokens": base_losses["loss_total"].new_tensor(0, dtype=torch.long),
                    "prefix_target_tokens": base_losses["loss_total"].new_tensor(0, dtype=torch.long),
                    "prefix_sequence_count": base_losses["loss_total"].new_tensor(0, dtype=torch.long),
                    "recovery_decoder_tokens": base_losses["loss_total"].new_tensor(0, dtype=torch.long),
                }
                free_outputs = None

                if (
                    phase.upper() == "B"
                    and float(phase_settings.get("lambda_free_prefix", 0.0)) > 0.0
                    and (step % free_prefix_every_n_steps == 0)
                ):
                    encoded_free = _detach_encoded_for_free_decode(encoded)
                    free_targets = dec_targets
                    free_encoded_for_decode = encoded_free
                    free_input_seq = None
                    free_teacher_prefix_steps = 0
                    if bool(STAGE2_USE_CHUNKED_FREE_TRAINING):
                        roi_targets_align = None
                        align_ratio = None
                        attn_for_alignment = outputs.get("attn_weights", None)
                        if attn_for_alignment is not None:
                            roi_targets_mono = build_decoder_roi_targets_monotonic(
                                target_tokens=dec_targets["target_tokens"],
                                target_mask=dec_targets["target_mask"],
                                enc_mask=encoded_free.get("context_mask", None),
                                eos_id=int(eos_id),
                            )
                            roi_targets_align = build_decoder_roi_targets_from_alignment(
                                target_tokens=dec_targets["target_tokens"],
                                target_mask=dec_targets["target_mask"],
                                attn_weights=attn_for_alignment,
                                enc_mask=encoded_free.get("context_mask", None),
                                eos_id=int(eos_id),
                                encoder_token_bias=encoded_free.get("decoder_token_bias", None),
                                aux_weight=0.35,
                            )
                            valid_align = dec_targets["target_mask"].bool()
                            if valid_align.any():
                                align_ratio = (
                                    roi_targets_align.ne(roi_targets_mono)[valid_align]
                                    .float()
                                    .mean()
                                    .item()
                                )
                        chunk_free_batch = _prepare_chunked_free_training_batch(
                            encoded_free,
                            dec_targets,
                            orientations,
                            eos_id=int(eos_id),
                            pad_id=int(vocab.pad_id),
                            roi_targets=roi_targets_align,
                        )
                        if align_ratio is not None:
                            total_free_align_ratio += float(align_ratio)
                            total_free_align_samples += 1.0
                        if chunk_free_batch is not None:
                            free_encoded_for_decode = chunk_free_batch["encoded"]
                            free_targets = {
                                "target_tokens": chunk_free_batch["target_tokens"],
                                "target_mask": chunk_free_batch["target_mask"],
                            }
                            free_input_seq = chunk_free_batch["decoder_inputs"]
                            free_teacher_prefix_steps = int(STAGE2_FREE_TEACHER_PREFIX_STEPS)

                    free_outputs = model.decode_from_encoded(
                        free_encoded_for_decode,
                        targets=None,
                        teacher_forcing_ratio=0.0,
                        teacher_prefix_steps=free_teacher_prefix_steps,
                        input_seq=free_input_seq,
                        sos_id=sos_id,
                        eos_id=eos_id,
                        max_len=int(free_targets["target_tokens"].size(1)),
                        decode_constraints=phase_settings.get("train_decode_constraints", phase_settings.get("decode_constraints")),
                        force_full_rollout=True,
                    )

                    free_loss_dict = compute_free_decoder_prefix_losses(
                        free_decoder_logits=free_outputs["decoder_logits"],
                        free_pred_ids=free_outputs.get("emitted_token_ids", None),
                        free_stop_logits=free_outputs.get("stop_logits", None),
                        free_action_logits=free_outputs.get("action_logits", None),
                        free_attn_weights=free_outputs.get("attn_weights", None),
                        target_tokens=free_targets["target_tokens"],
                        target_mask=free_targets["target_mask"],
                        enc_mask=free_outputs.get("context_mask", None),
                        encoder_token_bias=free_outputs.get("decoder_token_bias", None),
                        eos_id=int(eos_id),
                        label_smoothing=STAGE2_DECODER_LABEL_SMOOTHING,
                        eos_weight=decoder_eos_weight,
                        stop_pos_weight=STAGE2_DECODER_STOP_POS_WEIGHT,
                        min_prefix_len=int(model_overrides.get("free_prefix_min_tokens", STAGE2_FREE_PREFIX_MIN_TOKENS)),
                        lambda_stop=lambda_stop,
                        lambda_action=float(model_overrides.get("lambda_action", STAGE2_LAMBDA_ACTION)),
                        action_aux_weight=0.35,
                        action_class_weights=None,
                        current_epoch=epoch,
                        total_epochs=num_epochs,
                        action_alignment_start_epoch=2,
                        action_alignment_full_epoch=6,
                        action_degenerate_same_fraction_threshold=0.90,
                        action_degenerate_zero_fraction_threshold=0.90,
                        recovery_window=int(model_overrides.get("free_recovery_window", STAGE2_FREE_RECOVERY_WINDOW)),
                        recovery_weight=float(model_overrides.get("free_recovery_weight", STAGE2_FREE_RECOVERY_WEIGHT)),
                    )

                losses = dict(base_losses)
                free_prefix_weight = float(phase_settings.get("lambda_free_prefix", 0.0))
                free_stop_scale = float(phase_settings.get("free_stop_loss_scale", 1.0))
                losses["loss_decoder_free"] = free_prefix_weight * free_loss_dict["loss_decoder_free"]
                losses["loss_stop_free"] = free_prefix_weight * free_stop_scale * float(lambda_stop) * free_loss_dict["loss_stop_free"]
                free_action_scale = float(model_overrides.get("lambda_action_free_scale", STAGE2_LAMBDA_ACTION_FREE_SCALE))
                losses["loss_action_free"] = (
                    free_prefix_weight
                    * free_action_scale
                    * float(model_overrides.get("lambda_action", STAGE2_LAMBDA_ACTION))
                    * free_loss_dict["loss_action_free"]
                )
                losses["loss_total_free"] = (
                    losses["loss_decoder_free"]
                    + losses["loss_stop_free"]
                    + losses["loss_action_free"]
                )
                losses["loss_aux"] = phase_settings["lambda_aux"] * aux_total
                losses["loss_aux_raw"] = aux_parts["loss_aux_raw"]
                losses["loss_aux_ctx"] = aux_parts["loss_aux_ctx"]
                losses["loss_total"] = base_losses["loss_total"] + losses["loss_aux"] + losses["loss_total_free"]
                train_action_alignment_mix = float(losses.get("action_alignment_mix", 0.0))

                if (step % train_prop_stats_every_n_steps) == 0:
                    _update_refinement_epoch_stats(train_proposal_stats, outputs, refine_targets, gt_labels_list)

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

            decoder_active = phase_settings["lambda_decoder"] > 0.0

            if decoder_active:
                pred_ids = greedy_decode_from_logits(outputs["decoder_logits"])
                batch_acc = compute_simple_token_accuracy(
                    pred_ids=pred_ids,
                    tgt_ids=dec_targets["target_tokens"],
                    tgt_mask=dec_targets["target_mask"],
                )
            else:
                pred_ids = None
                batch_acc = 0.0

            should_update_train_decoder_stats = (step % train_decoder_stats_every_n_steps) == 0
            if pred_ids is not None and should_update_train_decoder_stats:
                _update_decoder_epoch_stats(
                    decoder_stats_tf,
                    pred_ids,
                    dec_targets["target_tokens"],
                    vocab,
                    action_logits_batch=outputs.get("action_logits", None),
                    pointer_positions_batch=outputs.get("pointer_positions", None),
                    valid_mask_batch=dec_targets["target_mask"],
                    action_targets_batch=losses.get("action_targets", None),
                    max_samples=train_decoder_stats_max_samples,
                )

            should_log_train_free = (
                bool(phase_settings.get("log_free_decoder", False))
                and train_free_eval_batches > 0
                and step < train_free_eval_batches
            )
            if should_log_train_free:
                with torch.no_grad():
                    free_decoded = _decode_free_by_reading_units(
                        model,
                        encoded,
                        orientations,
                        vocab,
                        sos_id=sos_id,
                        eos_id=eos_id,
                        max_len=min(train_free_diag_max_len, STAGE2_VAL_MAX_DECODE_LEN),
                        decode_constraints=phase_settings.get("train_decode_constraints", phase_settings.get("decode_constraints")),
                    )
                    _update_decoder_epoch_stats(
                        decoder_stats_free,
                        free_decoded["pred_ids"],
                        text_ids,
                        vocab,
                        action_logits_batch=free_decoded.get("action_logits", None),
                        pointer_positions_batch=free_decoded.get("pointer_positions", None),
                        valid_mask_batch=free_decoded.get("valid_mask", None),
                        action_targets_batch=None,
                        max_samples=train_decoder_stats_max_samples,
                    )

            total_loss += float(losses["loss_total"].item())
            total_box += float(losses["loss_box"].item())
            total_delta += float(losses["loss_delta"].item())
            total_score += float(losses["loss_score"].item())
            total_aux += float(losses["loss_aux"].item())
            total_decoder += float(losses["loss_decoder"].item())
            total_decoder_free += float(losses.get("loss_decoder_free", torch.tensor(0.0)).item())
            total_action += float(losses.get("loss_action", torch.tensor(0.0)).item())
            total_action_free += float(losses.get("loss_action_free", torch.tensor(0.0)).item())
            total_acc += batch_acc
            n_batches += 1
            total_stop += float(losses.get("loss_stop", torch.tensor(0.0)).item())
            total_stop_free += float(losses.get("loss_stop_free", torch.tensor(0.0)).item())
            total_free_prefix_tokens += float(free_loss_dict.get("prefix_decoder_tokens", 0))
            total_free_control_tokens += float(free_loss_dict.get("prefix_control_tokens", 0))
            total_free_target_tokens += float(free_loss_dict.get("prefix_target_tokens", 0))
            total_free_prefix_sequences += float(free_loss_dict.get("prefix_sequence_count", 0))
            total_free_recovery_tokens += float(free_loss_dict.get("recovery_decoder_tokens", 0))

            if bool(STAGE2_ENABLE_TQDM) and (step % max(1, int(STAGE2_PROGRESS_POSTFIX_EVERY_N_STEPS)) == 0):
                pbar.set_postfix({
                    "loss": total_loss / max(1, n_batches),
                    "dec": total_decoder / max(1, n_batches),
                    "fdec": total_decoder_free / max(1, n_batches),
                    "stop": total_stop / max(1, n_batches),
                    "fstop": total_stop_free / max(1, n_batches),
                    "box": total_box / max(1, n_batches),
                    "delta": total_delta / max(1, n_batches),
                    "acc": total_acc / max(1, n_batches),
                    "tf": tf_ratio,
                    "act": total_action / max(1, n_batches),
                    "fact": total_action_free / max(1, n_batches),
                    "fpfx": total_free_prefix_tokens / max(1.0, total_free_prefix_sequences),
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

        train_aux_without_ctx = _finalize_aux_branch_stats(
            train_proposal_stats["aux_without_context_encode"],
            vocab=vocab,
            available=train_proposal_stats["aux_without_context_encode"]["total"] > 0,
        )
        train_aux_with_ctx = _finalize_aux_branch_stats(
            train_proposal_stats["aux_with_context_encode"],
            vocab=vocab,
            available=train_proposal_stats["aux_with_context_encode"]["total"] > 0,
        )
        train_active_aux = _select_active_aux_summary(
            aux_without_ctx=train_aux_without_ctx,
            aux_with_ctx=train_aux_with_ctx,
            use_context_aux_for_loss=phase_settings["use_context_aux_for_loss"],
        )

        train_metrics = {
            "train_loss": total_loss / max(1, n_batches),
            "train_box": total_box / max(1, n_batches),
            "train_delta": total_delta / max(1, n_batches),
            "train_score": total_score / max(1, n_batches),
            "train_aux": total_aux / max(1, n_batches),
            "train_decoder": total_decoder / max(1, n_batches),
            "train_decoder_free": total_decoder_free / max(1, n_batches),
            "train_token_acc": total_acc / max(1, n_batches),
            "train_stop": total_stop / max(1, n_batches),
            "train_stop_free": total_stop_free / max(1, n_batches),
            "train_action": total_action / max(1, n_batches),
            "train_action_free": total_action_free / max(1, n_batches),
            "train_action_alignment_mix": train_action_alignment_mix if n_batches > 0 else 0.0,
            "train_free_prefix_len": total_free_prefix_tokens / max(1.0, total_free_prefix_sequences),
            "train_free_control_len": total_free_control_tokens / max(1.0, total_free_prefix_sequences),
            "train_free_prefix_ratio": total_free_prefix_tokens / max(1.0, total_free_target_tokens),
            "train_free_control_ratio": total_free_control_tokens / max(1.0, total_free_target_tokens),
            "train_free_recovery_ratio": total_free_recovery_tokens / max(1.0, total_free_target_tokens),
            "train_free_align_ratio": total_free_align_ratio / max(1.0, total_free_align_samples),
            "decoder_summary": {
                "teacher_forcing": _finalize_decoder_epoch_stats(decoder_stats_tf, vocab),
                "free_decoding": _finalize_decoder_epoch_stats(decoder_stats_free, vocab),
            },
            "proposal_summary": {
                "images": train_proposal_stats["images"],
                "orientation_counts": {
                    "horizontal": int(train_orientation_counts.get("horizontal", 0)),
                    "vertical": int(train_orientation_counts.get("vertical", 0)),
                    "other": int(train_orientation_counts.get("other", 0)),
                },
                "avg_proposals_per_image": train_proposal_stats["proposals"] / max(1, train_proposal_stats["images"]),
                "avg_positives_per_image": train_proposal_stats["positives"] / max(1, train_proposal_stats["images"]),
                "avg_gt_tokens_per_image": train_proposal_stats["gt_tokens"] / max(1, train_proposal_stats["images"]),
                "positive_coverage_ratio": train_proposal_stats["positives"] / max(1, train_proposal_stats["gt_tokens"]),
                "unique_coverage_ratio": train_proposal_stats["unique_gt_matched"] / max(1, train_proposal_stats["gt_tokens"]),
                "avg_unique_gt_matched_per_image": train_proposal_stats["unique_gt_matched"] / max(1, train_proposal_stats["images"]),
                "avg_duplicate_positive_matches_per_image": train_proposal_stats["duplicate_positive_matches"] / max(1, train_proposal_stats["images"]),
                "duplicate_positive_rate": train_proposal_stats["duplicate_positive_matches"] / max(1, train_proposal_stats["positives"]),
                "images_with_zero_valid_props_ratio": train_proposal_stats["images_with_zero_valid_props"] / max(1, train_proposal_stats["images"]),
                "positive_precision_proxy": train_proposal_stats["positives"] / max(1, train_proposal_stats["proposals"]),
                "avg_negatives_per_image": train_proposal_stats["negatives"] / max(1, train_proposal_stats["images"]),
                "avg_ignores_per_image": train_proposal_stats["ignores"] / max(1, train_proposal_stats["images"]),
                "avg_matched_iou_on_positives": train_proposal_stats["matched_iou_sum"] / max(1, train_proposal_stats["matched_iou_count"]),
                "ordering_primary_monotonic_fraction": train_proposal_stats["ordering_primary_mono_sum"] / max(1.0, train_proposal_stats["ordering_weight_sum"]),
                "ordering_primary_violation_fraction": train_proposal_stats["ordering_primary_viol_sum"] / max(1.0, train_proposal_stats["ordering_weight_sum"]),
                "refine_score_logit_mean": train_proposal_stats["score_logit_sum"] / max(1, train_proposal_stats["score_count"]),
                "refine_score_logit_std": (
                    max(
                        0.0,
                        train_proposal_stats["score_logit_sq_sum"] / max(1, train_proposal_stats["score_count"])
                        - (train_proposal_stats["score_logit_sum"] / max(1, train_proposal_stats["score_count"])) ** 2,
                    )
                ) ** 0.5,
                "refine_score_prob_mean": train_proposal_stats["score_prob_sum"] / max(1, train_proposal_stats["score_count"]),
                "refine_score_prob_std": (
                    max(
                        0.0,
                        train_proposal_stats["score_prob_sq_sum"] / max(1, train_proposal_stats["score_count"])
                        - (train_proposal_stats["score_prob_sum"] / max(1, train_proposal_stats["score_count"])) ** 2,
                    )
                ) ** 0.5,
                "aux_accuracy_on_positives": train_active_aux["top1"],
                "aux_top5_on_positives": train_active_aux["top5"],
                "active_aux_branch": "with_context_encode" if phase_settings["use_context_aux_for_loss"] else "without_context_encode",
                "aux_summary": {
                    "without_context_encode": train_aux_without_ctx,
                    "with_context_encode": train_aux_with_ctx,
                },
            },
        }

        val_metrics = validate_stage2(
            model=model,
            val_loader=val_loader,
            vocab=vocab,
            phase=phase,
            max_batches=val_max_batches,
            runtime_overrides=model_overrides,
            current_epoch=epoch,
            total_epochs=num_epochs,
        )

        print(
            f"\nEpoch {epoch+1}/{num_epochs} | "
            f"Train loss={train_metrics['train_loss']:.4f} "
            f"(dec={train_metrics['train_decoder']:.4f}, "
            f"fdec={train_metrics['train_decoder_free']:.4f}, "
            f"stop={train_metrics['train_stop']:.4f}, "
            f"fstop={train_metrics['train_stop_free']:.4f}, "
            f"act={train_metrics['train_action']:.4f}, "
            f"fact={train_metrics['train_action_free']:.4f}, "
            f"box={train_metrics['train_box']:.4f}, "
            f"delta={train_metrics['train_delta']:.4f}, "
            f"score={train_metrics['train_score']:.4f}, "
            f"aux={train_metrics['train_aux']:.4f}, "
            f"acc={train_metrics['train_token_acc']:.4f}) | "
            f"Val loss={val_metrics['val_loss']:.4f} "
            f"(dec={val_metrics['val_decoder']:.4f}, "
            f"stop={val_metrics['val_stop']:.4f}, "
            f"box={val_metrics['val_box']:.4f}, "
            f"delta={val_metrics['val_delta']:.4f}, "
            f"score={val_metrics['val_score']:.4f}, "
            f"aux={val_metrics['val_aux']:.4f}, "
            f"acc={val_metrics['val_token_acc']:.4f})"
        )

        train_prop = train_metrics["proposal_summary"]
        val_prop = val_metrics["proposal_summary"]
        train_ori = train_prop["orientation_counts"]
        val_ori = val_prop["orientation_counts"]
        train_aux_wo = train_prop["aux_summary"]["without_context_encode"]
        train_aux_w = train_prop["aux_summary"]["with_context_encode"]
        val_aux_wo = val_prop["aux_summary"]["without_context_encode"]
        val_aux_w = val_prop["aux_summary"]["with_context_encode"]
        train_dec_tf = train_metrics["decoder_summary"]["teacher_forcing"]
        train_dec_free = train_metrics["decoder_summary"]["free_decoding"]
        val_dec_tf = val_metrics["decoder_summary"]["teacher_forcing"]
        val_dec_free = val_metrics["decoder_summary"]["free_decoding"]

        print(
            "Orientation summary | "
            f"train H/V/O={train_ori['horizontal']}/{train_ori['vertical']}/{train_ori['other']} | "
            f"val H/V/O={val_ori['horizontal']}/{val_ori['vertical']}/{val_ori['other']}"
        )

        print(
            "Proposal summary | "
            f"train avg props/img={train_prop['avg_proposals_per_image']:.2f}, "
            f"pos/img={train_prop['avg_positives_per_image']:.2f}, "
            f"gt/img={train_prop['avg_gt_tokens_per_image']:.2f}, "
            f"cov={train_prop['positive_coverage_ratio']:.3f}, "
            f"uniq_cov={train_prop['unique_coverage_ratio']:.3f}, "
            f"dup+={train_prop['duplicate_positive_rate']:.3f}, "
            f"zero/img={train_prop['images_with_zero_valid_props_ratio']:.3f}, "
            f"neg/img={train_prop['avg_negatives_per_image']:.2f}, "
            f"ign/img={train_prop['avg_ignores_per_image']:.2f}, "
            f"mean IoU+={train_prop['avg_matched_iou_on_positives']:.3f}, "
            f"ord_mono={train_prop['ordering_primary_monotonic_fraction']:.3f}, "
            f"prec_proxy={train_prop['positive_precision_proxy']:.3f}, "
            f"aux acc+={train_prop['aux_accuracy_on_positives']:.3f} | "
            f"val avg props/img={val_prop['avg_proposals_per_image']:.2f}, "
            f"pos/img={val_prop['avg_positives_per_image']:.2f}, "
            f"gt/img={val_prop['avg_gt_tokens_per_image']:.2f}, "
            f"cov={val_prop['positive_coverage_ratio']:.3f}, "
            f"uniq_cov={val_prop['unique_coverage_ratio']:.3f}, "
            f"dup+={val_prop['duplicate_positive_rate']:.3f}, "
            f"zero/img={val_prop['images_with_zero_valid_props_ratio']:.3f}, "
            f"neg/img={val_prop['avg_negatives_per_image']:.2f}, "
            f"ign/img={val_prop['avg_ignores_per_image']:.2f}, "
            f"mean IoU+={val_prop['avg_matched_iou_on_positives']:.3f}, "
            f"ord_mono={val_prop['ordering_primary_monotonic_fraction']:.3f}, "
            f"prec_proxy={val_prop['positive_precision_proxy']:.3f}, "
            f"aux acc+={val_prop['aux_accuracy_on_positives']:.3f}, "
            f"aux top5+={val_prop['aux_top5_on_positives']:.3f}"
        )
        print(
            "Proposal score calibration | "
            f"train logits mean/std={train_prop['refine_score_logit_mean']:.3f}/{train_prop['refine_score_logit_std']:.3f}, "
            f"train prob mean/std={train_prop['refine_score_prob_mean']:.3f}/{train_prop['refine_score_prob_std']:.3f} | "
            f"val logits mean/std={val_prop['refine_score_logit_mean']:.3f}/{val_prop['refine_score_logit_std']:.3f}, "
            f"val prob mean/std={val_prop['refine_score_prob_mean']:.3f}/{val_prop['refine_score_prob_std']:.3f}"
        )
        print(
            "Aux summary (without context encode) | "
            f"train top1={train_aux_wo['top1']:.3f}, top5={train_aux_wo['top5']:.3f}, n={train_aux_wo['total']} | "
            f"val top1={val_aux_wo['top1']:.3f}, top5={val_aux_wo['top5']:.3f}, n={val_aux_wo['total']}"
        )
        print(f"Aux top predictions (val, without context encode): {val_aux_wo['top_predictions']}")
        print(f"Aux top errors (val, without context encode): {val_aux_wo['top_errors']}")

        if val_aux_w["available"] or train_aux_w["available"]:
            print(
                "Aux summary (with context encode) | "
                f"train top1={train_aux_w['top1']:.3f}, top5={train_aux_w['top5']:.3f}, n={train_aux_w['total']} | "
                f"val top1={val_aux_w['top1']:.3f}, top5={val_aux_w['top5']:.3f}, n={val_aux_w['total']}"
            )
            print(f"Aux top predictions (val, with context encode): {val_aux_w['top_predictions']}")
            print(f"Aux top errors (val, with context encode): {val_aux_w['top_errors']}")
        print(
            "Decoder train TF | "
            f"EOS hit={train_dec_tf['eos_hit_fraction']:.3f}, "
            f"len_ratio={train_dec_tf['mean_length_ratio']:.3f}, "
            f"CER={train_dec_tf['mean_cer']:.4f}, "
            f"exact={train_dec_tf['exact_match_fraction']:.3f}, "
            f"action stay/adv/stop="
            f"{train_dec_tf['action_pred_distribution']['fractions']['stay']:.2f}/"
            f"{train_dec_tf['action_pred_distribution']['fractions']['advance']:.2f}/"
            f"{train_dec_tf['action_pred_distribution']['fractions']['stop']:.2f}, "
            f"action_match={train_dec_tf['action_match_fraction']:.3f}, "
            f"ptr adv/stay/reg="
            f"{train_dec_tf['pointer_summary']['advance_fraction']:.2f}/"
            f"{train_dec_tf['pointer_summary']['stay_fraction']:.2f}/"
            f"{train_dec_tf['pointer_summary']['regress_fraction']:.2f}"
        )
        print(
            "Decoder train FREE | "
            f"EOS hit={train_dec_free['eos_hit_fraction']:.3f}, "
            f"len_ratio={train_dec_free['mean_length_ratio']:.3f}, "
            f"CER={train_dec_free['mean_cer']:.4f}, "
            f"exact={train_dec_free['exact_match_fraction']:.3f}, "
            f"prefix_len={train_metrics['train_free_prefix_len']:.2f}, "
            f"prefix_ratio={train_metrics['train_free_prefix_ratio']:.3f}, "
            f"rec_ratio={train_metrics['train_free_recovery_ratio']:.3f}, "
            f"align_ratio={train_metrics['train_free_align_ratio']:.3f}, "
            f"action stay/adv/stop="
            f"{train_dec_free['action_pred_distribution']['fractions']['stay']:.2f}/"
            f"{train_dec_free['action_pred_distribution']['fractions']['advance']:.2f}/"
            f"{train_dec_free['action_pred_distribution']['fractions']['stop']:.2f}, "
            f"ptr adv/stay/reg="
            f"{train_dec_free['pointer_summary']['advance_fraction']:.2f}/"
            f"{train_dec_free['pointer_summary']['stay_fraction']:.2f}/"
            f"{train_dec_free['pointer_summary']['regress_fraction']:.2f}"
        )
        print(
            "Decoder val TF | "
            f"EOS hit={val_dec_tf['eos_hit_fraction']:.3f}, "
            f"len_ratio={val_dec_tf['mean_length_ratio']:.3f}, "
            f"CER={val_dec_tf['mean_cer']:.4f}, "
            f"exact={val_dec_tf['exact_match_fraction']:.3f}, "
            f"action stay/adv/stop="
            f"{val_dec_tf['action_pred_distribution']['fractions']['stay']:.2f}/"
            f"{val_dec_tf['action_pred_distribution']['fractions']['advance']:.2f}/"
            f"{val_dec_tf['action_pred_distribution']['fractions']['stop']:.2f}, "
            f"action_match={val_dec_tf['action_match_fraction']:.3f}, "
            f"ptr adv/stay/reg="
            f"{val_dec_tf['pointer_summary']['advance_fraction']:.2f}/"
            f"{val_dec_tf['pointer_summary']['stay_fraction']:.2f}/"
            f"{val_dec_tf['pointer_summary']['regress_fraction']:.2f}"
        )
        print(
            "Decoder val FREE | "
            f"EOS hit={val_dec_free['eos_hit_fraction']:.3f}, "
            f"len_ratio={val_dec_free['mean_length_ratio']:.3f}, "
            f"CER={val_dec_free['mean_cer']:.4f}, "
            f"exact={val_dec_free['exact_match_fraction']:.3f}, "
            f"action stay/adv/stop="
            f"{val_dec_free['action_pred_distribution']['fractions']['stay']:.2f}/"
            f"{val_dec_free['action_pred_distribution']['fractions']['advance']:.2f}/"
            f"{val_dec_free['action_pred_distribution']['fractions']['stop']:.2f}, "
            f"ptr adv/stay/reg="
            f"{val_dec_free['pointer_summary']['advance_fraction']:.2f}/"
            f"{val_dec_free['pointer_summary']['stay_fraction']:.2f}/"
            f"{val_dec_free['pointer_summary']['regress_fraction']:.2f}"
        )
        print(
            "Action diag | "
            f"train TF pred={train_dec_tf['action_pred_distribution']['fractions']['stay']:.2f}/"
            f"{train_dec_tf['action_pred_distribution']['fractions']['advance']:.2f}/"
            f"{train_dec_tf['action_pred_distribution']['fractions']['stop']:.2f}, "
            f"target={train_dec_tf['action_target_distribution']['fractions']['stay']:.2f}/"
            f"{train_dec_tf['action_target_distribution']['fractions']['advance']:.2f}/"
            f"{train_dec_tf['action_target_distribution']['fractions']['stop']:.2f}, "
            f"match={train_dec_tf['action_match_fraction']:.3f} | "
            f"val TF pred={val_dec_tf['action_pred_distribution']['fractions']['stay']:.2f}/"
            f"{val_dec_tf['action_pred_distribution']['fractions']['advance']:.2f}/"
            f"{val_dec_tf['action_pred_distribution']['fractions']['stop']:.2f}, "
            f"target={val_dec_tf['action_target_distribution']['fractions']['stay']:.2f}/"
            f"{val_dec_tf['action_target_distribution']['fractions']['advance']:.2f}/"
            f"{val_dec_tf['action_target_distribution']['fractions']['stop']:.2f}, "
            f"match={val_dec_tf['action_match_fraction']:.3f}"
        )
        print(
            "Action diag FREE | "
            f"train pred={train_dec_free['action_pred_distribution']['fractions']['stay']:.2f}/"
            f"{train_dec_free['action_pred_distribution']['fractions']['advance']:.2f}/"
            f"{train_dec_free['action_pred_distribution']['fractions']['stop']:.2f}, "
            f"ptr={train_dec_free['pointer_summary']['advance_fraction']:.2f}/"
            f"{train_dec_free['pointer_summary']['stay_fraction']:.2f}/"
            f"{train_dec_free['pointer_summary']['regress_fraction']:.2f} | "
            f"val pred={val_dec_free['action_pred_distribution']['fractions']['stay']:.2f}/"
            f"{val_dec_free['action_pred_distribution']['fractions']['advance']:.2f}/"
            f"{val_dec_free['action_pred_distribution']['fractions']['stop']:.2f}, "
            f"ptr={val_dec_free['pointer_summary']['advance_fraction']:.2f}/"
            f"{val_dec_free['pointer_summary']['stay_fraction']:.2f}/"
            f"{val_dec_free['pointer_summary']['regress_fraction']:.2f}"
        )
        print(f"Pointer examples (val FREE): {val_dec_free['pointer_examples']}")

        current_score = _select_model_score(val_metrics, phase)
        # Composite model score is defined such that higher is better for both phases.
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
                "decoder_embed_dim": STAGE2_DECODER_EMBED_DIM,
                "decoder_hidden_dim": STAGE2_DECODER_HIDDEN_DIM,
                "det_score_thresh": float(model.det_score_thresh),
                "det_top_k": int(model.det_top_k),
                "det_nms_iou": float(model.det_nms_iou),
                "det_min_box_size": float(model.det_min_box_size),
                "freeze_backbone": FREEZE_BACKBONE,
                "freeze_detector": FREEZE_DETECTOR,
            },
            "model_overrides": model_overrides or {},
        }

        epoch_path = checkpoint_dir / f"stage2_hybrid_epoch{epoch+1}.pt"
        torch.save(ckpt, epoch_path)

        if is_best:
            torch.save(ckpt, checkpoint_dir / "stage2_hybrid_best.pt")
            print(f"✅ saved best: stage2_hybrid_best.pt (val={val_metrics['val_loss']:.4f})")

        prune_to_keep_last_n(checkpoint_dir, keep=2, exclude="checkpoint_old.pt")

    print("\n" + "=" * 70)
    print("STAGE 2 HYBRID TRAINING COMPLETE")
    print(f"Best checkpoint: {checkpoint_dir / 'stage2_hybrid_best.pt'}")
    print("=" * 70 + "\n")

    return {
        "model": model,
        "best_val_loss": float(best_val) if best_val is not None else None,
        "best_val_metrics": best_val_metrics,
    }


def main():
    selected_phase = STAGE2_PHASE.upper()
    phase_settings = get_phase_settings(selected_phase)

    best_ckpt = CHECKPOINT_DIR / "stage1_detection" / "detector_best.pt"
    last_ckpt = CHECKPOINT_DIR / "stage1_detection" / f"detector_epoch{NUM_EPOCHS}.pt"

    detector_ckpt = best_ckpt if best_ckpt.exists() else last_ckpt
    if not detector_ckpt.exists():
        raise FileNotFoundError(
            f"No Stage-1 detector checkpoint found. Expected one of:\n"
            f"  - {best_ckpt}\n"
            f"  - {last_ckpt}"
        )

    checkpoint_dir = CHECKPOINT_DIR / f"stage2_hybrid_phase_changed_3_{selected_phase}"
    resume_model_ckpt = None
    if selected_phase == "B":
        print("Phase B selected: will attempt to resume from Phase A best checkpoint.")
        phase_a_best = CHECKPOINT_DIR / "stage2_hybrid_phaseA" / "stage2_hybrid_best.pt"
        if phase_a_best.exists():
            resume_model_ckpt = phase_a_best
        else:
            raise FileNotFoundError(
                "Phase B requested but no Phase-A best checkpoint found at:\n"
                f"  - {phase_a_best}\n"
                "Run Phase A first."
            )

    train_stage2_hybrid(
        detector_ckpt_path=detector_ckpt,
        num_epochs=phase_settings["epochs"],
        lr=LR,
        checkpoint_dir=checkpoint_dir,
        phase=selected_phase,
        resume_model_ckpt=resume_model_ckpt,
        val_max_batches=VALIDATION_BATCHES,
    )


if __name__ == "__main__":
    main()
