"""Training script for SuminaNetRecognizer (per-ROI character classifier).

Single-phase training: detect boxes, refine, sort by reading order,
classify each ROI as a Kuzushiji character.

No teacher forcing, no free decoding, no pointer mechanism.

Usage:
    python train_suminanet.py
    python train_suminanet.py --warmup-ckpt checkpoints/suminanet_warmup/warmup_best.pt
    python train_suminanet.py --resume checkpoints/suminanet_recognizer/suminanet_best.pt
    python train_suminanet.py --epochs 30 --lr 1e-4
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import sys
import time
from collections import Counter
from functools import partial
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from config import (
    AVG_GT_PER_IMAGE,
    BACKBONE_BASE_FEATURES,
    STAGE2_BATCH_SIZE,
    CHECKPOINT_DIR,
    DATA_DIR,
    DENSITY_FACTOR,
    DENSITY_GRID,
    DEVICE,
    FREEZE_BACKBONE,
    FREEZE_DETECTOR,
    GRAD_CLIP,
    IMAGE_SIZE,
    SUMINANET_CHECKPOINT_DIR,
    SUMINANET_CLASSIFIER_HIDDEN,
    SUMINANET_CER_SCORE_THRESH,
    SUMINANET_LAMBDA_SCRIPT,
    SUMINANET_EARLY_STOPPING_PATIENCE,
    SUMINANET_ENABLE_TQDM,
    SUMINANET_EPOCHS,
    SUMINANET_GRAD_ACCUM_STEPS,
    SUMINANET_FOCAL_GAMMA,
    SUMINANET_RARE_CHAR_THRESH,
    SUMINANET_HARD_NEG_WEIGHT,
    SUMINANET_HARD_NEG_TOP_K,
    SUMINANET_HARD_NEG_START_EPOCH,
    SUMINANET_BG_WEIGHT,
    SUMINANET_STRONG_BG_WEIGHT,
    SUMINANET_LAMBDA_BOX,
    SUMINANET_LAMBDA_CHAR,
    SUMINANET_LAMBDA_DELTA,
    SUMINANET_LAMBDA_SCORE,
    SUMINANET_LOG_PREDICTIONS,
    SUMINANET_LR,
    SUMINANET_LR_ETA_MIN,
    SUMINANET_PREDICTION_SAMPLES,
    SUMINANET_PROGRESS_POSTFIX_N,
    STAGE2_ROI_SIZE,
    SUMINANET_ROI_POOL_OUTPUT_SIZE,
    SUMINANET_RESIDUAL_SCALE_INIT,
    SUMINANET_USE_CONTEXT,
    SUMINANET_CONTEXT_BLOCK_GAP_FACTOR,
    SUMINANET_USE_CROP_ENCODER,
    SUMINANET_CROP_ENCODER_SIZE,
    SUMINANET_CROP_ENCODER_CHUNK_SIZE,
    SUMINANET_FREEZE_CROP_ENCODER,
    SUMINANET_FREEZE_CROP_ENCODER_AFTER,
    SUMINANET_VALIDATION_BATCHES,
    SUMINANET_WEIGHT_DECAY,
    NUM_WORKERS,
    STAGE2_CONTEXT_HIDDEN_DIM,
    STAGE2_CONTEXT_NUM_LAYERS,
    STAGE2_CONTEXT_MODE,
    DET_MIN_BOX_SIZE,
    DET_NMS_IOU,
    DET_TOP_K,
    SUMINANET_DET_SCORE_THRESH,
    STAGE2_DROPOUT_RATE,
    STAGE2_PROJ_DIM,
    STAGE2_REFINE_HIDDEN_DIM,
    STAGE2_REFINE_NEG_IOU,
    STAGE2_REFINE_POS_IOU,
    STAGE2_REFINE_POS_WEIGHT,
    STAGE2_USE_HUNGARIAN,
    STAGE2_ROI_FEAT_DIM,
    STAGE2_TOKEN_DIM,
    STAGE2_TOKEN_HIDDEN_DIM,
    STAGE2_TOKEN_USE_SCORE_BRANCH,
    USE_MIXED_PRECISION,
    BACKBONE_TYPE,
)

from model.suminanet import DetectorHead, build_backbone
from model.suminanet.suminanet_recognizer import SuminaNetRecognizer

from utils import KuzushijiDataset
from utils.char_augmentation import compute_char_frequencies, compute_sample_weights, get_rare_chars
from utils.stage2_losses import (
    background_classification_loss,
    focal_classification_loss,
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
from utils.text_normalization import render_tokens, unicode_token_to_char
from utils.vocab import VocabManager


def _atomic_save(obj, path):
    """Write checkpoint atomically: save to .tmp then rename, so crashes never leave a corrupt file."""
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Vocabulary + data loaders (reused unchanged from train_stage2.py)
# ---------------------------------------------------------------------------

def load_vocab() -> VocabManager:
    ann_files = sorted(list((Path(DATA_DIR) / "annotations").glob("*.json")))
    if len(ann_files) == 0:
        raise FileNotFoundError(f"No annotation files found in {Path(DATA_DIR) / 'annotations'}")
    return VocabManager.from_annotations(ann_files)


def build_dataloaders(vocab: VocabManager, world_size: int = 1):
    pad_id = vocab.pad_id

    # Compute training-split character frequencies for Options A and B.
    ann_dir = Path(DATA_DIR) / "annotations"
    train_split_file = Path(DATA_DIR) / "splits" / "train.txt"
    if train_split_file.exists():
        train_names = set(l.strip() for l in train_split_file.read_text().splitlines() if l.strip())
        train_ann_files = [f for f in sorted(ann_dir.glob("*.json")) if f.name in train_names]
    else:
        train_ann_files = sorted(ann_dir.glob("*.json"))

    char_freq = compute_char_frequencies(train_ann_files)
    rare_chars = get_rare_chars(char_freq, threshold=SUMINANET_RARE_CHAR_THRESH)
    print(
        f"[augmentation] rare chars (freq < {SUMINANET_RARE_CHAR_THRESH}): "
        f"{len(rare_chars)} classes  ({len(rare_chars)/max(1,len(char_freq))*100:.1f}% of vocab)"
    )

    train_dataset = KuzushijiDataset(
        Path(DATA_DIR),
        vocab=vocab,
        use_sequences=True,
        resize=IMAGE_SIZE,
        split="train",
        rare_chars=rare_chars,
    )
    val_dataset = KuzushijiDataset(
        Path(DATA_DIR),
        vocab=vocab,
        use_sequences=True,
        resize=IMAGE_SIZE,
        split="val",
    )

    sample_weights = compute_sample_weights(
        items=train_dataset.items,
        char_counter=char_freq,
        threshold=SUMINANET_RARE_CHAR_THRESH,
        boost_scale=1.5,
    )
    # Under DDP, each rank gets its own independent WeightedRandomSampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset) // world_size,
        replacement=True,
    )
    print(
        f"[augmentation] WeightedRandomSampler: "
        f"min_weight={min(sample_weights):.2f}  max_weight={max(sample_weights):.2f}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=STAGE2_BATCH_SIZE,
        sampler=sampler,         # replaces shuffle=True
        num_workers=NUM_WORKERS,
        collate_fn=partial(collate_fn, pad_id=pad_id),
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=STAGE2_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=partial(collate_fn, pad_id=pad_id),
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=NUM_WORKERS > 0,
    )

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_suminanet_model(
    vocab: VocabManager,
    warmup_ckpt: Optional[str | Path] = None,
    load_stage1_weights: bool = True,
    backbone_type_override: Optional[str] = None,
    context_mode_override: Optional[str] = None,
    device: str | torch.device = DEVICE,
) -> SuminaNetRecognizer:
    """
    Build SuminaNetRecognizer.

    Loads backbone + detector weights from Stage 1 checkpoint (unless
    load_stage1_weights=False, e.g. at inference/eval, where a full SuminaNet
    checkpoint is loaded on top right after and would overwrite them anyway).
    Optionally warm-starts shared ROI pipeline from warmup checkpoint.

    backbone_type_override, if given, takes precedence over both the Stage 1
    checkpoint's saved backbone_type and the global config.BACKBONE_TYPE --
    needed when building a model to receive a SuminaNet checkpoint whose
    backbone architecture differs from whatever's currently configured (e.g.
    validating an archived 'unet' checkpoint while config.py is set to
    'efficientnet_b2'). The resolved value is stashed on the returned model
    as `.backbone_type` so callers loading a state dict on top can verify it
    matches (see _load_compatible_state_dict's ckpt_backbone_type guard).

    context_mode_override, if given, takes precedence over the global
    config.STAGE2_CONTEXT_MODE -- needed when building a model to receive a
    SuminaNet checkpoint trained with a different context_mode than whatever
    is currently configured (e.g. validating an archived 'bigru' checkpoint
    while config.py is set to 'gru'). Without this, _load_compatible_state_dict
    raises rather than silently corrupting the mismatched RNN layer (see its
    ckpt_context_mode guard) -- so the failure mode is a hard error, not wrong
    numbers, but building with the matching mode from the start avoids the
    error entirely.

    device defaults to the global config.DEVICE but can be overridden -- e.g.
    to build a CPU-resident copy alongside the GPU one without ever touching
    CUDA (see app.py's per-request CPU fallback on CUDA OOM).
    """
    vocab_size = vocab.vocab_size

    stage1_ckpt = CHECKPOINT_DIR / "stage1_detection" / "detector_best.pt"
    if not stage1_ckpt.exists():
        raise FileNotFoundError(f"Stage 1 checkpoint not found: {stage1_ckpt}")
    ckpt = torch.load(stage1_ckpt, map_location=device)
    if backbone_type_override is not None:
        backbone_type = backbone_type_override
    elif load_stage1_weights:
        backbone_type = ckpt.get("backbone_type", BACKBONE_TYPE)
    else:
        backbone_type = BACKBONE_TYPE

    backbone = build_backbone(backbone_type, BACKBONE_BASE_FEATURES, pretrained=False).to(device)
    detector = DetectorHead(
        in_ch=BACKBONE_BASE_FEATURES,
        num_classes=vocab_size,
        dropout_rate=STAGE2_DROPOUT_RATE,
        predict_boxes=True,
        predict_classes=False,
    ).to(device)

    if load_stage1_weights:
        state_key = "backbone_state_dict" if "backbone_state_dict" in ckpt else "unet_state_dict"
        backbone.load_state_dict(ckpt[state_key])
        detector.load_state_dict(ckpt["detector_state_dict"])
        print(f"Loaded Stage 1 weights from {stage1_ckpt} (backbone={backbone_type})")
    else:
        print(f"Skipping Stage 1 checkpoint read (backbone={backbone_type}); "
              f"weights will be loaded from the SuminaNet checkpoint instead.")

    bg_id = vocab.bg_id if hasattr(vocab, "bg_id") and vocab.BG_TOKEN in vocab.char2id else None

    model = SuminaNetRecognizer(
        backbone=backbone,
        detector=detector,
        backbone_out_channels=BACKBONE_BASE_FEATURES,
        vocab_size=vocab_size,

        proj_dim=STAGE2_PROJ_DIM,
        roi_size=STAGE2_ROI_SIZE,
        roi_pool_output_size=SUMINANET_ROI_POOL_OUTPUT_SIZE,
        roi_feat_dim=STAGE2_ROI_FEAT_DIM,
        refine_hidden_dim=STAGE2_REFINE_HIDDEN_DIM,
        residual_scale_init=SUMINANET_RESIDUAL_SCALE_INIT,
        token_dim=STAGE2_TOKEN_DIM,
        token_hidden_dim=STAGE2_TOKEN_HIDDEN_DIM,
        token_use_score_branch=STAGE2_TOKEN_USE_SCORE_BRANCH,

        use_context=SUMINANET_USE_CONTEXT,
        context_hidden_dim=STAGE2_CONTEXT_HIDDEN_DIM,
        context_num_layers=STAGE2_CONTEXT_NUM_LAYERS,
        context_mode=context_mode_override if context_mode_override is not None else STAGE2_CONTEXT_MODE,
        context_block_gap_factor=SUMINANET_CONTEXT_BLOCK_GAP_FACTOR,

        classifier_hidden_dim=SUMINANET_CLASSIFIER_HIDDEN,

        det_score_thresh=SUMINANET_DET_SCORE_THRESH,
        det_top_k=DET_TOP_K,
        det_nms_iou=DET_NMS_IOU,
        det_min_box_size=DET_MIN_BOX_SIZE,

        density_grid=DENSITY_GRID,
        density_factor=DENSITY_FACTOR,
        avg_gt_per_image=AVG_GT_PER_IMAGE,

        dropout=STAGE2_DROPOUT_RATE,
        bg_id=bg_id,

        use_crop_encoder=SUMINANET_USE_CROP_ENCODER,
        crop_encoder_size=SUMINANET_CROP_ENCODER_SIZE,
        freeze_crop_encoder=SUMINANET_FREEZE_CROP_ENCODER,
        crop_encoder_chunk_size=SUMINANET_CROP_ENCODER_CHUNK_SIZE,
    ).to(device)
    model.backbone_type = backbone_type

    if FREEZE_BACKBONE:
        for p in model.backbone.parameters():
            p.requires_grad = False
        model.backbone.eval()

    if FREEZE_DETECTOR:
        for p in model.detector.parameters():
            p.requires_grad = False
        model.detector.eval()

    if warmup_ckpt is not None and STAGE2_REFINE_HIDDEN_DIM != SUMINANET_CLASSIFIER_HIDDEN:
        print(
            f"WARNING: STAGE2_REFINE_HIDDEN_DIM ({STAGE2_REFINE_HIDDEN_DIM}) != "
            f"SUMINANET_CLASSIFIER_HIDDEN ({SUMINANET_CLASSIFIER_HIDDEN}) — "
            f"warmup aux_head_context → classifier weight transfer will be skipped."
        )

    # Warm-start from warmup checkpoint (shared ROI pipeline weights)
    if warmup_ckpt is not None:
        warmup_ckpt = Path(warmup_ckpt)
        if not warmup_ckpt.exists():
            print(f"WARNING: Warmup checkpoint not found: {warmup_ckpt} — skipping warm start.")
        else:
            ckpt_a = torch.load(warmup_ckpt, map_location=DEVICE)
            state = ckpt_a.get("model_state_dict", ckpt_a)
            ckpt_context_mode = ckpt_a.get("stage2_config", {}).get("context_mode")
            ckpt_vocab_hash = ckpt_a.get("vocab_hash")
            _load_compatible_state_dict(
                model, state,
                ckpt_context_mode=ckpt_context_mode,
                ckpt_vocab_hash=ckpt_vocab_hash,
                current_vocab_hash=vocab.content_hash(),
                ckpt_backbone_type=ckpt_a.get("backbone_type"),
            )

            # Warmup trained aux_head_context (Linear(384,256)→ReLU→Dropout→Linear(256,V))
            # _load_compatible_state_dict cannot map them because the key names differ, so copy the weights explicitly here.
            _AUX_TO_CLASSIFIER = {
                "classifier.0.weight": "aux_head_context.0.weight",
                "classifier.0.bias":   "aux_head_context.0.bias",
                "classifier.3.weight": "aux_head_context.3.weight",
                "classifier.3.bias":   "aux_head_context.3.bias",
            }
            n_copied = 0
            with torch.no_grad():
                for cls_key, ctx_key in _AUX_TO_CLASSIFIER.items():
                    if ctx_key not in state:
                        continue
                    dst = dict(model.named_parameters()).get(cls_key)
                    if dst is None:
                        continue
                    src = state[ctx_key]
                    if dst.shape != src.shape:
                        print(f"  classifier warm-start shape mismatch: {cls_key} {tuple(dst.shape)} vs {tuple(src.shape)}")
                        continue
                    dst.data.copy_(src)
                    n_copied += 1
            if n_copied == 4:
                print("Classifier fully warm-started from aux_head_context (4/4 tensors).")
            elif n_copied > 0:
                print(f"Classifier partially warm-started from aux_head_context ({n_copied}/4 tensors).")
            elif STAGE2_REFINE_HIDDEN_DIM != SUMINANET_CLASSIFIER_HIDDEN:
                print(
                    f"Classifier warm-start skipped: hidden dim mismatch "
                    f"({STAGE2_REFINE_HIDDEN_DIM} vs {SUMINANET_CLASSIFIER_HIDDEN}) — "
                    f"classifier randomly initialized."
                )
            else:
                print("WARNING: classifier warm-start from aux_head_context failed — classifier is randomly initialized.")

            print(f"Warm-started from warmup checkpoint: {warmup_ckpt}")

    return model


def set_trainable_modules(model: SuminaNetRecognizer) -> None:
    """Freeze backbone + detector; train all other components."""
    for p in model.parameters():
        p.requires_grad = True

    if FREEZE_BACKBONE:
        for p in model.backbone.parameters():
            p.requires_grad = False

    if FREEZE_DETECTOR:
        for p in model.detector.parameters():
            p.requires_grad = False


def get_trainable_params(model: SuminaNetRecognizer):
    return [p for p in model.parameters() if p.requires_grad]


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def build_kanji_vocab_weights(vocab: "VocabManager", kanji_weight: float = 1.8) -> torch.Tensor:
    """
    Build a per-vocabulary weight tensor for script-stratified focal loss.

    Kanji tokens receive kanji_weight (default 1.8×) while hiragana/katakana/other
    keep weight 1.0.  Following Clanuwat et al.: kanji are visually ambiguous and
    data-sparse, so they need proportionally larger gradient signal.

    The tensor is built once at training startup and cached on the target device.
    """
    weights = torch.ones(vocab.vocab_size, dtype=torch.float32)
    for id_, ch in vocab.id2char.items():
        if _char_to_script_type(ch) == _SCRIPT_KANJI:
            weights[id_] = kanji_weight
    return weights


def compute_suminanet_loss(
    outputs: dict,
    refine_targets: dict,
    bg_id: Optional[int] = None,
    hn_keys: Optional[torch.Tensor] = None,
    vocab: Optional["VocabManager"] = None,
    vocab_weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict]:
    """
    Single-phase SuminaNet loss.

    Character classification loss is computed on sorted (reading-order) features,
    so labels/masks are reordered via sort_indices before calling aux_classification_loss.

    Box, delta, and score losses are computed in original (pre-sort) order.

    When bg_id is provided, negative (FP) ROIs are additionally supervised to
    predict the background class, reducing insertion errors in the assembled CER.
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
        neg_mask_sorted  = reorder_by_sort_indices(neg_mask.long(), sort_indices).bool()
    else:
        gt_labels_sorted = gt_labels
        pos_mask_sorted  = pos_mask
        neg_mask_sorted  = neg_mask

    loss_char = focal_classification_loss(
        aux_logits=outputs["char_logits"],
        target_labels=gt_labels_sorted,
        pos_mask=pos_mask_sorted,
        gamma=SUMINANET_FOCAL_GAMMA,
        ignore_index=-1,
        vocab_weights=vocab_weights,
    )

    # Hard negative mining: extra CE penalty for previously-confused pairs.
    if hn_keys is not None and hn_keys.numel() > 0:
        logits_flat = outputs["char_logits"][pos_mask_sorted]   # (N_pos, V)
        labels_flat = gt_labels_sorted[pos_mask_sorted]          # (N_pos,)
        valid = labels_flat.ge(0)
        if valid.any():
            logits_valid = logits_flat[valid]
            gt_valid     = labels_flat[valid]
            pred_ids     = logits_valid.argmax(dim=-1)
            V            = logits_valid.size(-1)
            batch_keys   = gt_valid * V + pred_ids
            hn_mask      = torch.isin(batch_keys, hn_keys)
            if hn_mask.any():
                loss_hn   = F.cross_entropy(logits_valid[hn_mask], gt_valid[hn_mask])
                loss_char = loss_char + SUMINANET_HARD_NEG_WEIGHT * loss_hn

    # Background supervision: FP proposals learn to predict <BG>.
    if bg_id is not None:
        iso_mask  = outputs.get("isolation_mask", None)   # small/oversized FPs only
        furi_mask = outputs.get("furigana_mask",  None)   # furigana sub-columns — excluded
        if iso_mask is not None:
            not_furi = ~furi_mask if furi_mask is not None else torch.ones_like(iso_mask)
            iso_neg  = iso_mask & not_furi & neg_mask_sorted
            loss_bg = SUMINANET_STRONG_BG_WEIGHT * background_classification_loss(
                char_logits=outputs["char_logits"], neg_mask=iso_neg, bg_id=bg_id,
            )
        else:
            loss_bg = SUMINANET_BG_WEIGHT * background_classification_loss(
                char_logits=outputs["char_logits"],
                neg_mask=neg_mask_sorted,
                bg_id=bg_id,
            )
    else:
        loss_bg = loss_char.new_tensor(0.0)

    # --- Script-type auxiliary loss ---
    loss_script = loss_char.new_tensor(0.0)
    if SUMINANET_LAMBDA_SCRIPT > 0.0 and vocab is not None and "script_logits" in outputs:
        script_labels = _build_script_label_tensor(
            gt_labels=gt_labels_sorted,
            pos_mask=pos_mask_sorted,
            vocab=vocab,
        )
        # script_logits: (B, T, 4) in sorted order
        B, T, _ = outputs["script_logits"].shape
        sl_flat = outputs["script_logits"].reshape(B * T, 4)
        lbl_flat = script_labels.reshape(B * T)
        valid = lbl_flat >= 0
        if valid.any():
            loss_script = F.cross_entropy(sl_flat[valid], lbl_flat[valid])

    total = (
        SUMINANET_LAMBDA_CHAR   * loss_char
        + SUMINANET_LAMBDA_BOX   * loss_box
        + SUMINANET_LAMBDA_DELTA * loss_delta
        + SUMINANET_LAMBDA_SCORE * loss_score
        + loss_bg
        + SUMINANET_LAMBDA_SCRIPT * loss_script
    )

    return total, {
        "loss_char":   loss_char,
        "loss_box":    loss_box,
        "loss_delta":  loss_delta,
        "loss_score":  loss_score,
        "loss_bg":     loss_bg,
        "loss_script": loss_script,
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


# Script-type labels used by the auxiliary head.
# 0=hiragana  1=katakana  2=kanji  3=other
_SCRIPT_HIRAGANA = 0
_SCRIPT_KATAKANA = 1
_SCRIPT_KANJI    = 2
_SCRIPT_OTHER    = 3


def _char_to_script_type(ch: str) -> int:
    """Map a Unicode character to a script-type index.

    Annotation labels are stored as "U+XXXX" token strings — unicode_token_to_char
    converts them to the actual character before the range check.  NFC normalization
    then collapses NFKD voiced kana (base + combining dakuten) to a single codepoint.
    """
    import unicodedata
    if not ch:
        return _SCRIPT_OTHER
    ch = unicode_token_to_char(ch)          # "U+306E" → "の"
    ch_nfc = unicodedata.normalize("NFC", ch)
    cp = ord(ch_nfc[0])
    if 0x3041 <= cp <= 0x3096:      # hiragana block
        return _SCRIPT_HIRAGANA
    if 0x30A1 <= cp <= 0x30F6:      # katakana block
        return _SCRIPT_KATAKANA
    if (0x4E00 <= cp <= 0x9FFF       # CJK unified
            or 0x3400 <= cp <= 0x4DBF   # CJK extension A
            or 0xF900 <= cp <= 0xFAFF): # CJK compatibility
        return _SCRIPT_KANJI
    return _SCRIPT_OTHER


def _build_script_label_tensor(
    gt_labels: torch.Tensor,       # (B, T) char IDs
    pos_mask: torch.Tensor,        # (B, T) bool — only positives have a valid char
    vocab: VocabManager,
) -> torch.Tensor:
    """
    Build per-ROI script-type labels for the auxiliary script head.

    Returns (B, T) long tensor with values in {0,1,2,3}.
    Positions where pos_mask is False (or label == -1) are set to -1 so
    cross_entropy with ignore_index=-1 skips them.
    """
    B, T = gt_labels.shape
    out = gt_labels.new_full((B, T), -1)
    for b in range(B):
        for t in range(T):
            if not pos_mask[b, t]:
                continue
            lid = int(gt_labels[b, t].item())
            if lid < 0:
                continue
            ch = vocab.id2char.get(lid, "")
            out[b, t] = _char_to_script_type(ch)
    return out


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
    bsz = char_logits.size(0) if char_logits.size(0) > 0 else 1

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

def validate_suminanet(
    model: SuminaNetRecognizer,
    val_loader: DataLoader,
    vocab: VocabManager,
    max_batches: Optional[int] = None,
    vocab_weights: Optional[torch.Tensor] = None,
    sam2_dir: Optional[Path] = None,
    return_detailed: bool = False,
) -> dict:
    """
    Validation loop for SuminaNetRecognizer.

    Returns:
        top1_acc, top5_acc, coverage, assembled_cer, avg_iou,
        avg_proposals_per_image, avg_gt_per_image
    """
    model.eval()

    bg_id = vocab.bg_id if hasattr(vocab, "bg_id") and vocab.BG_TOKEN in vocab.char2id else None

    total_loss = 0.0
    loss_parts: dict[str, float] = {"loss_char": 0., "loss_box": 0., "loss_delta": 0., "loss_score": 0., "loss_bg": 0., "loss_script": 0.}

    top1_sum = 0.0
    top5_sum = 0.0
    cer_sum = 0.0
    iou_sum = 0.0
    pos_sum = 0.0
    props_sum = 0.0
    gt_sum = 0.0
    n_batches = 0
    n_images_tot = 0

    # For confusion analysis
    error_counter: Counter = Counter()
    pred_counter: Counter = Counter()
    gt_total_counter: Counter = Counter()

    # Detailed analysis accumulators (only used when return_detailed=True)
    per_image_cer_list: list[float] = []
    topk_in_gt_list: list[list[bool]] = []

    # Per-script accuracy (main classifier evaluated on GT-matched ROIs)
    # Keys: "hiragana", "katakana", "kanji", "other"
    _SCRIPT_NAMES = ["hiragana", "katakana", "kanji", "other"]
    script_correct: dict[str, int] = {s: 0 for s in _SCRIPT_NAMES}
    script_total:   dict[str, int] = {s: 0 for s in _SCRIPT_NAMES}

    # For prediction examples
    examples: list[dict] = []

    pad_id = vocab.pad_id

    # Coverage gap breakdown accumulators
    iou_02_sum = 0.0
    iou_02_gt  = 0

    # Ordering diagnostics accumulators
    order_mono_sum  = 0.0
    order_viol_sum  = 0.0
    order_diag_n    = 0

    # Isolation / furigana stats
    iso_neg_sum  = 0    # isolated negatives (get BG supervision)
    iso_pos_sum  = 0    # isolated positives (GT-matched but geometrically isolated)
    iso_pos_bg   = 0    # of those: predicted as BG → shows if model wrongly kills them
    furi_neg_sum = 0    # furigana negatives (no BG supervision)
    furi_pos_sum = 0    # furigana positives (GT-matched furigana)
    furi_pos_bg  = 0    # of those: predicted as BG

    # CER score-threshold audit: count tokens kept vs filtered per image
    cer_thresh_kept    = 0
    cer_thresh_filtered = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images         = batch["image"].to(DEVICE, non_blocking=True, memory_format=torch.channels_last)
            boxes_list     = [b.to(DEVICE, dtype=torch.float32) for b in batch["boxes"]]
            gt_labels_list = [l.to(DEVICE, dtype=torch.long) for l in batch["labels"]]
            orientations   = [
                _normalize_orientation_label(o) for o in batch["orientations"]
            ]

            with torch.cuda.amp.autocast(enabled=USE_MIXED_PRECISION, dtype=torch.float16):
                sam2_boxes, sam2_scores = None, None
                if sam2_dir is not None:
                    stems = batch.get("image_stems", [])
                    sam2_boxes, sam2_scores = _load_sam2_proposals(stems, sam2_dir, torch.device(DEVICE))
                outputs = model(images, orientations, sam2_boxes, sam2_scores)

                # Build refinement targets in original box order
                refine_targets = build_refinement_targets(
                    coarse_boxes=outputs["roi_boxes"],
                    roi_mask=outputs["roi_mask"],
                    gt_boxes_list=boxes_list,
                    gt_labels_list=gt_labels_list,
                    pos_iou_thresh=STAGE2_REFINE_POS_IOU,
                    neg_iou_thresh=STAGE2_REFINE_NEG_IOU,
                    ignore_label_ids=[vocab.unk_id],
                    use_hungarian=STAGE2_USE_HUNGARIAN,
                )

                loss, parts = compute_suminanet_loss(outputs, refine_targets, bg_id=bg_id, vocab=vocab,
                                                  vocab_weights=vocab_weights)
            total_loss += float(loss.item())
            for k in loss_parts:
                loss_parts[k] += float(parts[k].item())

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
                                              score_thresh=SUMINANET_CER_SCORE_THRESH)

            bsz = images.size(0)

            # Per-image CER (detailed mode only) — mirrors _compute_assembled_cer logic
            if return_detailed:
                _cl   = outputs["char_logits"]
                _om   = outputs["ordered_mask"]
                _rs   = outputs.get("refine_scores")
                _si   = outputs.get("sort_indices")
                for b in range(bsz):
                    _valid_b = pos_mask_s[b] & gt_labels_s[b].ne(-1)
                    _gt_text = _ids_to_text(gt_labels_s[b][_valid_b].tolist(), vocab)
                    if not _gt_text:
                        continue
                    if SUMINANET_CER_SCORE_THRESH > 0.0 and _rs is not None and _si is not None:
                        _sorted_pos = _om[b].nonzero(as_tuple=True)[0]
                        _orig_pos   = _si[b][_sorted_pos]
                        _scores     = torch.sigmoid(_rs[b][_orig_pos])
                        _sorted_pos = _sorted_pos[_scores >= SUMINANET_CER_SCORE_THRESH]
                    else:
                        _sorted_pos = _om[b].nonzero(as_tuple=True)[0]
                    _pred_text = _ids_to_text(_cl[b, _sorted_pos].argmax(dim=-1).tolist(), vocab)
                    per_image_cer_list.append(
                        _edit_distance(_pred_text, _gt_text) / max(1, len(_gt_text))
                    )

            # Coverage: proportion of GT chars matched by a positive ROI
            n_images_tot += bsz
            for b in range(bsz):
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

            # --- Coverage gap: Stage-1-equivalent recall at IoU > 0.2 ---
            # For each GT box, check whether ANY proposal has IoU > 0.2.
            # This mirrors the Stage 1 validation criterion and lets us measure how
            # much coverage is lost to the stricter IoU 0.45 positive threshold vs
            # how much is genuinely missed by the detector.
            roi_boxes_t = outputs["roi_boxes"]   # (B, T, 4) padded
            roi_mask_t  = outputs["roi_mask"]    # (B, T) bool
            for b in range(bsz):
                gt_b   = boxes_list[b].cpu()             # (G, 4)
                props_b = roi_boxes_t[b][roi_mask_t[b]].cpu()  # (P, 4)
                iou_02_gt += gt_b.size(0)
                if gt_b.size(0) == 0 or props_b.size(0) == 0:
                    continue
                # IoU matrix: (P, G)
                px1, py1, px2, py2 = props_b[:, 0], props_b[:, 1], props_b[:, 2], props_b[:, 3]
                gx1, gy1, gx2, gy2 = gt_b[:, 0], gt_b[:, 1], gt_b[:, 2], gt_b[:, 3]
                ix1 = torch.max(px1.unsqueeze(1), gx1.unsqueeze(0))   # (P, G)
                iy1 = torch.max(py1.unsqueeze(1), gy1.unsqueeze(0))
                ix2 = torch.min(px2.unsqueeze(1), gx2.unsqueeze(0))
                iy2 = torch.min(py2.unsqueeze(1), gy2.unsqueeze(0))
                inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)  # (P, G)
                area_p = (px2 - px1) * (py2 - py1)                    # (P,)
                area_g = (gx2 - gx1) * (gy2 - gy1)                    # (G,)
                union  = area_p.unsqueeze(1) + area_g.unsqueeze(0) - inter
                iou_pg = inter / (union + 1e-6)                        # (P, G)
                # A GT box is "covered at 0.2" if any proposal has IoU > 0.2
                covered = (iou_pg.max(dim=0).values > 0.2).sum().item()
                iou_02_sum += covered

            # --- CER score-threshold audit ---
            # Count how many ordered proposals survive vs are filtered by SUMINANET_CER_SCORE_THRESH
            if SUMINANET_CER_SCORE_THRESH > 0.0:
                rs = outputs.get("refine_scores")
                si = outputs.get("sort_indices")
                om = outputs["ordered_mask"]
                for b in range(bsz):
                    sorted_pos = om[b].nonzero(as_tuple=True)[0]
                    if sorted_pos.numel() == 0:
                        continue
                    if rs is not None and si is not None:
                        orig = si[b][sorted_pos]
                        scores = torch.sigmoid(rs[b][orig])
                        kept = int((scores >= SUMINANET_CER_SCORE_THRESH).sum().item())
                    else:
                        kept = sorted_pos.numel()
                    cer_thresh_kept     += kept
                    cer_thresh_filtered += sorted_pos.numel() - kept

            # Ordering diagnostics
            od = outputs.get("ordering_diagnostics")
            if od is not None:
                mono_t = od["primary_monotonic_fraction"]   # (B,)
                viol_t = od["primary_violation_fraction"]   # (B,)
                order_mono_sum += float(mono_t.mean().item())
                order_viol_sum += float(viol_t.mean().item())
                order_diag_n   += 1

            # Isolation / furigana stats (needs neg_mask in sorted order)
            neg_mask   = refine_targets["refine_neg_mask"]
            neg_mask_s = reorder_by_sort_indices(neg_mask.long(), sort_indices).bool() \
                         if sort_indices is not None else neg_mask
            iso_mask  = outputs.get("isolation_mask")
            furi_mask = outputs.get("furigana_mask")
            if iso_mask is not None and bg_id is not None:
                logits_flat = outputs["char_logits"]   # (B, T, V)
                # isolated
                m_iso_neg = iso_mask & neg_mask_s
                m_iso_pos = iso_mask & pos_mask_s
                iso_neg_sum += int(m_iso_neg.sum().item())
                iso_pos_sum += int(m_iso_pos.sum().item())
                if m_iso_pos.any():
                    preds_iso = logits_flat[m_iso_pos].argmax(dim=-1)
                    iso_pos_bg += int((preds_iso == bg_id).sum().item())
            if furi_mask is not None and bg_id is not None:
                logits_flat = outputs["char_logits"]
                m_furi_neg = furi_mask & neg_mask_s
                m_furi_pos = furi_mask & pos_mask_s
                furi_neg_sum += int(m_furi_neg.sum().item())
                furi_pos_sum += int(m_furi_pos.sum().item())
                if m_furi_pos.any():
                    preds_furi = logits_flat[m_furi_pos].argmax(dim=-1)
                    furi_pos_bg += int((preds_furi == bg_id).sum().item())

            # Confusion pairs + per-script accuracy
            for b in range(bsz):
                valid_b = pos_mask_s[b] & gt_labels_s[b].ne(-1)
                if valid_b.any():
                    logits_valid = outputs["char_logits"][b, valid_b]  # (N, V)
                    pred_ids = logits_valid.argmax(dim=-1).tolist()
                    true_ids = gt_labels_s[b][valid_b].tolist()
                    topk_ids_b = None
                    if return_detailed:
                        k5 = min(5, logits_valid.size(-1))
                        topk_ids_b = logits_valid.topk(k5, dim=-1).indices.tolist()  # (N, k5)
                    for i, (pid, tid) in enumerate(zip(pred_ids, true_ids)):
                        pred_counter[pid] += 1
                        gt_total_counter[tid] += 1
                        if pid != tid:
                            error_counter[(tid, pid)] += 1
                            if return_detailed and topk_ids_b is not None:
                                top5 = topk_ids_b[i]
                                topk_in_gt_list.append([tid in top5[:k] for k in range(1, 6)])
                        # Per-script accuracy bucket
                        ch = vocab.id2char.get(tid, "")
                        stype = _char_to_script_type(ch)
                        sname = _SCRIPT_NAMES[stype]
                        script_total[sname] += 1
                        if pid == tid:
                            script_correct[sname] += 1

            # Log prediction examples
            if len(examples) < SUMINANET_PREDICTION_SAMPLES and SUMINANET_LOG_PREDICTIONS:
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

    avg_props     = props_sum / max(1, n_images_tot)
    avg_pos_img   = pos_sum   / max(1, n_images_tot)
    avg_gt_img    = gt_sum    / max(1, n_images_tot)
    det_recall    = avg_pos_img / max(1e-6, avg_gt_img)
    det_precision = avg_pos_img / max(1e-6, avg_props)
    det_f1        = 2 * det_precision * det_recall / max(1e-6, det_precision + det_recall)

    # Coverage gap breakdown
    stage1_equiv_recall = iou_02_sum / max(1, iou_02_gt)
    iou_gap_pp = (stage1_equiv_recall - det_recall) * 100  # pp lost to IoU 0.2→0.45 threshold

    # CER threshold audit
    cer_thresh_total = cer_thresh_kept + cer_thresh_filtered
    cer_thresh_filter_rate = cer_thresh_filtered / max(1, cer_thresh_total)

    metrics = {
        "val_loss":    avg(total_loss),
        "top1_acc":    avg(top1_sum),
        "top5_acc":    avg(top5_sum),
        "assembled_cer": avg(cer_sum),
        "coverage":            det_recall,
        "stage1_equiv_recall": stage1_equiv_recall,
        "det_precision":       det_precision,
        "det_f1":              det_f1,
        "avg_proposals_per_image": avg_props,
        "avg_pos_per_image":       avg_pos_img,
        "avg_gt_per_image":        avg_gt_img,
        "avg_iou_on_positives":    avg(iou_sum),
        "cer_thresh_kept":         cer_thresh_kept,
        "cer_thresh_filtered":     cer_thresh_filtered,
        "cer_thresh_filter_rate":  cer_thresh_filter_rate,
        **{f"val_{k}": avg(v) for k, v in loss_parts.items()},
        **{
            f"acc_{s}": script_correct[s] / max(1, script_total[s])
            for s in _SCRIPT_NAMES
        },
        **{f"n_{s}": script_total[s] for s in _SCRIPT_NAMES},
    }

    # Print summary
    print(
        f"Val | loss={metrics['val_loss']:.4f}"
        f" (char={metrics['val_loss_char']:.4f}"
        f", box={metrics['val_loss_box']:.4f}"
        f", delta={metrics['val_loss_delta']:.4f}"
        f", score={metrics['val_loss_score']:.4f}"
        f", script={metrics['val_loss_script']:.4f})"
    )
    print(
        f"Val | top1={metrics['top1_acc']:.4f}"
        f"  top5={metrics['top5_acc']:.4f}"
        f"  CER={metrics['assembled_cer']:.4f}"
        f"  coverage={metrics['coverage']:.4f}"
        f"  prec={metrics['det_precision']:.4f}"
        f"  F1={metrics['det_f1']:.4f}"
        f"  IoU+={metrics['avg_iou_on_positives']:.4f}"
        f"  props/img={metrics['avg_proposals_per_image']:.1f}"
        f"  gt/img={metrics['avg_gt_per_image']:.1f}"
    )
    # Coverage gap breakdown
    print(
        f"Coverage gap | Stage-1-equiv recall (IoU>0.2): {stage1_equiv_recall:.4f}"
        f"  SuminaNet coverage (IoU>0.45): {det_recall:.4f}"
        f"  → IoU-threshold gap: {iou_gap_pp:.2f} pp"
    )
    print(
        f"CER thresh={SUMINANET_CER_SCORE_THRESH:.2f} audit |"
        f" kept={cer_thresh_kept}  filtered={cer_thresh_filtered}"
        f"  filter_rate={cer_thresh_filter_rate:.3f}"
        f"  (ROIs removed from CER string assembly)"
    )
    # Ordering diagnostics
    if order_diag_n > 0:
        avg_mono = order_mono_sum / order_diag_n
        avg_viol = order_viol_sum / order_diag_n
        print(
            f"Ordering | primary_mono={avg_mono:.3f}  primary_viol={avg_viol:.3f}"
            f"  (fraction of consecutive pairs in correct reading-axis order)"
        )

    # Isolation / furigana breakdown
    # Shows whether the neighbour-based geometry check is working and whether
    # furigana is being incorrectly suppressed toward <BG>.
    def _bg_pct(n_bg: int, n_total: int) -> str:
        return f"{100*n_bg/max(1,n_total):.1f}%" if n_total > 0 else "n/a"

    print(
        f"Isolation | iso_neg={iso_neg_sum}"
        f"  iso_pos={iso_pos_sum}(BG={_bg_pct(iso_pos_bg, iso_pos_sum)})"
        f"  furi_neg={furi_neg_sum}"
        f"  furi_pos={furi_pos_sum}(BG={_bg_pct(furi_pos_bg, furi_pos_sum)})"
        f"  [iso_neg→BG supervised; furi free]"
    )

    # Per-script accuracy on main classifier
    script_parts = "  ".join(
        f"{s}={script_correct[s]/max(1,script_total[s]):.3f}(n={script_total[s]})"
        for s in _SCRIPT_NAMES
    )
    print(f"Per-script acc | {script_parts}")

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

    # Optional detailed fields for thesis analysis scripts
    if return_detailed:
        metrics["error_counter"]    = dict(error_counter)
        metrics["gt_total_counter"] = dict(gt_total_counter)
        metrics["per_image_cer"]    = per_image_cer_list
        metrics["topk_in_gt"]       = topk_in_gt_list

    _ITERATION_MARK_IDS: set = {
        vocab.char2id[tok] for c in ("ゝ", "ヽ", "ゞ", "ヾ", "〱", "〲")
        if (tok := f"U+{ord(c):04X}") in vocab.char2id
    }

    def _is_kana(char_id: int) -> bool:
        ch = vocab.id2char.get(char_id, "")
        if not ch:
            return False
        try:
            cp = ord(unicode_token_to_char(ch)[0])
        except Exception:
            return False
        return (0x3041 <= cp <= 0x3096) or (0x30A1 <= cp <= 0x30F6)

    def _exclude_hard_neg(gt_id: int, pred_id: int) -> bool:
        if bg_id is not None and pred_id == bg_id:
            return True
        if gt_id in _ITERATION_MARK_IDS or pred_id in _ITERATION_MARK_IDS:
            return True
        # Exclude ALL intra-kana pairs (both characters are hiragana or katakana)
        if _is_kana(gt_id) and _is_kana(pred_id):
            return True
        return False

    # Scan up to 3× top-K candidates so filtering still yields SUMINANET_HARD_NEG_TOP_K pairs
    top_k_pairs = [
        (int(gt), int(pr))
        for (gt, pr), _ in error_counter.most_common(SUMINANET_HARD_NEG_TOP_K * 3)
        if not _exclude_hard_neg(int(gt), int(pr))
    ][:SUMINANET_HARD_NEG_TOP_K]
    metrics["hard_neg_pairs"] = top_k_pairs

    return metrics


# ---------------------------------------------------------------------------
# Model score for checkpoint selection
# ---------------------------------------------------------------------------

def select_model_score(metrics: dict) -> float:
    """
    Composite score for best-checkpoint selection.

    top1 * (1 - CER): rewards per-ROI accuracy AND low assembled CER jointly.
    Coverage (recall) is not included because it is fixed when the detector is frozen —
    multiplying by a constant factor would not change checkpoint ranking.
    """
    top1 = float(metrics.get("top1_acc", 0.0))
    cer  = float(metrics.get("assembled_cer", 1.0))
    return top1 * (1.0 - cer)


# ---------------------------------------------------------------------------
# Training epoch
# ---------------------------------------------------------------------------

def _load_sam2_proposals(
    image_stems: list[str],
    sam2_dir: Path,
    device: torch.device,
) -> tuple[list[torch.Tensor] | None, list[torch.Tensor] | None]:
    """
    Load pre-computed SAM2 proposal files for a batch.

    Returns (coarse_boxes_list, coarse_scores_list) where each element corresponds to
    one image.  Returns (None, None) when all files are missing (falls back to detector).
    """
    boxes_list:  list[torch.Tensor] = []
    scores_list: list[torch.Tensor] = []
    any_found = False

    for stem in image_stems:
        pt_path = sam2_dir / f"{stem}.pt"
        if pt_path.exists():
            data = torch.load(pt_path, map_location=device)
            boxes_list.append(data["boxes"].to(device, dtype=torch.float32))
            scores_list.append(data["scores"].to(device, dtype=torch.float32))
            any_found = True
        else:
            # Fallback: empty tensors → model will use its own detector for this image
            boxes_list.append(torch.zeros((0, 4), device=device, dtype=torch.float32))
            scores_list.append(torch.zeros((0,), device=device, dtype=torch.float32))

    if not any_found:
        return None, None
    return boxes_list, scores_list


def _build_hard_neg_keys(
    hard_neg_pairs: Optional[set],
    vocab_size: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Pre-encode hard negative pairs as int64 keys (gt * V + pred) for fast isin lookup."""
    if not hard_neg_pairs:
        return None
    V = vocab_size
    keys = [g * V + p for g, p in hard_neg_pairs if 0 <= g < V and 0 <= p < V]
    if not keys:
        return None
    return torch.tensor(keys, dtype=torch.long, device=device)


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

# Print a timing breakdown every this many batches (0 = disabled).
CHECK_TIMING_EVERY_N = 50


def _sync() -> float:
    """Synchronize CUDA and return current wall time."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def train_epoch(
    model: SuminaNetRecognizer,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    vocab: VocabManager,
    epoch: int,
    grad_accum_steps: int,
    hard_neg_pairs: Optional[set] = None,
    vocab_weights: Optional[torch.Tensor] = None,
    sam2_dir: Optional[Path] = None,
    is_distributed: bool = False,
    disable_tqdm: bool = False,
) -> dict:
    model.train()
    raw_model = model.module if is_distributed else model
    if FREEZE_BACKBONE:
        raw_model.backbone.eval()
    if FREEZE_DETECTOR:
        raw_model.detector.eval()

    bg_id = vocab.bg_id if hasattr(vocab, "bg_id") and vocab.BG_TOKEN in vocab.char2id else None

    # Pre-encode hard-neg pairs once per epoch (avoids rebuilding a list every batch step)
    hn_keys = _build_hard_neg_keys(hard_neg_pairs, vocab.vocab_size, torch.device(DEVICE))

    total_loss = 0.0
    loss_parts: dict[str, float] = {"loss_char": 0., "loss_box": 0., "loss_delta": 0., "loss_score": 0., "loss_bg": 0., "loss_script": 0.}
    top1_sum = 0.0
    n_batches = 0

    optimizer.zero_grad()

    bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch}",
        disable=disable_tqdm or not SUMINANET_ENABLE_TQDM,
        dynamic_ncols=True,
    )

    # Timing accumulators (wall-clock seconds, GPU-synchronised)
    t_data = t_transfer = t_forward = t_targets = t_loss = t_backward = t_step = t_metrics = 0.0
    t_batch_start = _sync()

    for batch_idx, batch in enumerate(bar):
        t0 = _sync()
        t_data += t0 - t_batch_start   # time the dataloader took to produce this batch

        images         = batch["image"].to(DEVICE, non_blocking=True, memory_format=torch.channels_last)
        boxes_list     = [b.to(DEVICE, dtype=torch.float32) for b in batch["boxes"]]
        gt_labels_list = [l.to(DEVICE, dtype=torch.long) for l in batch["labels"]]
        orientations   = [_normalize_orientation_label(o) for o in batch["orientations"]]
        t1 = _sync()
        t_transfer += t1 - t0

        with torch.cuda.amp.autocast(enabled=USE_MIXED_PRECISION, dtype=torch.float16):
            # If SAM2 proposals are available for this batch, bypass the frozen detector.
            sam2_boxes, sam2_scores = None, None
            if sam2_dir is not None:
                stems = batch.get("image_stems", [])
                sam2_boxes, sam2_scores = _load_sam2_proposals(stems, sam2_dir, torch.device(DEVICE))
            outputs = model(images, orientations, sam2_boxes, sam2_scores)
            t2 = _sync()
            t_forward += t2 - t1

            refine_targets = build_refinement_targets(
                coarse_boxes=outputs["roi_boxes"],
                roi_mask=outputs["roi_mask"],
                gt_boxes_list=boxes_list,
                gt_labels_list=gt_labels_list,
                pos_iou_thresh=STAGE2_REFINE_POS_IOU,
                neg_iou_thresh=STAGE2_REFINE_NEG_IOU,
                ignore_label_ids=[vocab.unk_id],
                use_hungarian=STAGE2_USE_HUNGARIAN,
            )
            t3 = _sync()
            t_targets += t3 - t2

            # No ROI proposals survived for any sample in this batch (e.g. an
            # aggressive det_score_thresh filtered everything out) -> every loss
            # term below would fall back to a disconnected zero tensor with no
            # grad_fn, crashing backward(). Nothing to supervise.
            batch_empty = not refine_targets["refine_pos_mask"].any() and not refine_targets["refine_neg_mask"].any()

            if batch_empty and not is_distributed:
                # Single-process: skipping backward() here is just an
                # efficiency win, nothing else depends on every step running.
                t_batch_start = _sync()
                continue
            elif batch_empty:
                # Under DDP, skipping backward() here would desync ranks that
                # hit this on different batches (each rank draws data
                # independently), hanging the all-reduce for any rank that DID
                # call backward() this step. Substitute a zero-valued but
                # graph-connected loss touching every trainable parameter so
                # every rank always calls backward() the same number of times.
                loss = images.new_zeros(())
                for p in model.parameters():
                    if p.requires_grad:
                        loss = loss + p.sum() * 0.0
                parts = {k: loss.detach() for k in loss_parts}
            else:
                loss, parts = compute_suminanet_loss(
                    outputs, refine_targets, bg_id=bg_id, hn_keys=hn_keys, vocab=vocab,
                    vocab_weights=vocab_weights,
                )
            loss = loss / grad_accum_steps
            t4 = _sync()
            t_loss += t4 - t3

        sync_now = (batch_idx + 1) % grad_accum_steps == 0
        backward_ctx = (
            contextlib.nullcontext() if (sync_now or not is_distributed) else model.no_sync()
        )
        with backward_ctx:
            scaler.scale(loss).backward()
        t5 = _sync()
        t_backward += t5 - t4

        if sync_now:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        t6 = _sync()
        t_step += t6 - t5

        total_loss += float(loss.item()) * grad_accum_steps
        for k in loss_parts:
            loss_parts[k] += float(parts[k].item())
        t7 = _sync()
        t_metrics += t7 - t6

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
        t_batch_start = _sync()   # start timing next batch's data-load from here

        if CHECK_TIMING_EVERY_N > 0 and n_batches % CHECK_TIMING_EVERY_N == 0:
            n = n_batches
            total_t = t_data + t_transfer + t_forward + t_targets + t_loss + t_backward + t_step + t_metrics
            print(
                f"\n[TIMING] step {n_batches} | "
                f"total={total_t/n*1000:.0f}ms/batch  "
                f"data={t_data/n*1000:.0f}  "
                f"transfer={t_transfer/n*1000:.0f}  "
                f"forward={t_forward/n*1000:.0f}  "
                f"targets={t_targets/n*1000:.0f}  "
                f"loss={t_loss/n*1000:.0f}  "
                f"backward={t_backward/n*1000:.0f}  "
                f"optim={t_step/n*1000:.0f}  "
                f"metrics={t_metrics/n*1000:.0f}  (all ms)"
            )

        if (batch_idx + 1) % SUMINANET_PROGRESS_POSTFIX_N == 0:
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
    parser = argparse.ArgumentParser(description="Train SuminaNet per-ROI classifier")
    parser.add_argument("--warmup-ckpt", type=str, default=None,
                        help="Path to warmup checkpoint for ROI pipeline warm-start")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to SuminaNet checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=SUMINANET_EPOCHS)
    parser.add_argument("--lr", type=float, default=SUMINANET_LR)
    parser.add_argument("--weight-decay", type=float, default=SUMINANET_WEIGHT_DECAY)
    parser.add_argument("--grad-accum", type=int, default=SUMINANET_GRAD_ACCUM_STEPS)
    parser.add_argument(
        "--sam2_proposals",
        type=str,
        default=None,
        help=(
            "Directory of SAM2-refined proposal files produced by preprocess_sam2_proposals.py. "
            "When set, per-image pre-computed boxes replace the frozen detector's output, "
            "directly addressing the IoU-threshold coverage gap. "
            "Files must be named <image_stem>.pt and contain {boxes, scores} tensors."
        ),
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Ignore warmup_ckpt even if provided (for pure-resume runs)",
    )
    args = parser.parse_args()
    if getattr(args, "no_warmup", False):
        args.warmup_ckpt = None

    # --- Distributed setup (torchrun sets these env vars automatically) ---
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = world_size > 1
    if is_distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        if rank != 0:
            sys.stdout = open(os.devnull, "w")
            sys.stderr = open(os.devnull, "w")

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True, parents=True)
    logger = logging.getLogger(__name__)

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    logger.info("=" * 70)
    logger.info("SUMINANET RECOGNIZER TRAINING (per-ROI classifier)")
    logger.info("=" * 70)

    SUMINANET_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    vocab = load_vocab()
    train_loader, val_loader = build_dataloaders(vocab, world_size=world_size)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # Pre-compute per-class focal loss weights (Clanuwat-inspired: kanji needs more gradient)
    vocab_weights = build_kanji_vocab_weights(vocab, kanji_weight=1.8).to(DEVICE)

    # SAM2 refined proposals directory (optional)
    sam2_dir: Optional[Path] = Path(args.sam2_proposals) if args.sam2_proposals else None
    if sam2_dir is not None:
        if not sam2_dir.exists():
            print(f"WARNING: --sam2_proposals dir not found: {sam2_dir} — running detector as fallback")
            sam2_dir = None
        else:
            n_files = sum(1 for _ in sam2_dir.glob("*.pt"))
            print(f"SAM2 proposals: {sam2_dir}  ({n_files} files)")

    # Build model
    model = build_suminanet_model(
        vocab=vocab,
        warmup_ckpt=args.warmup_ckpt,
    )

    start_epoch  = 1
    best_score   = -float("inf")
    patience_ctr = 0

    # Resume from checkpoint
    if args.resume is not None:
        resume_path = Path(args.resume)
        if resume_path.exists():
            ckpt = torch.load(resume_path, map_location=DEVICE)
            _load_compatible_state_dict(
                model, ckpt["model_state_dict"],
                ckpt_context_mode=ckpt.get("context_mode"),
                ckpt_vocab_hash=ckpt.get("vocab_hash"),
                current_vocab_hash=vocab.content_hash(),
                ckpt_backbone_type=ckpt.get("backbone_type"),
            )
            start_epoch  = int(ckpt.get("epoch", 0)) + 1
            best_score   = float(ckpt.get("best_score", -float("inf")))
            patience_ctr = int(ckpt.get("patience_ctr", 0))
            print(f"Resumed from {resume_path} (epoch {start_epoch - 1})")
        else:
            print(f"WARNING: resume checkpoint not found: {resume_path}")

    set_trainable_modules(model)

    # If resuming past the crop-encoder freeze point, start already frozen instead
    # of waiting for the in-loop freeze trigger to catch up on the first resumed
    # epoch (set_trainable_modules() above unconditionally re-enables requires_grad
    # on every param, including the crop encoder, so this must run after it).
    if (
        SUMINANET_FREEZE_CROP_ENCODER_AFTER > 0
        and start_epoch > SUMINANET_FREEZE_CROP_ENCODER_AFTER
        and model.roi_crop_encoder is not None
        and not model.roi_crop_encoder.freeze_encoder
    ):
        for p in model.roi_crop_encoder.encoder.parameters():
            p.requires_grad_(False)
        model.roi_crop_encoder.encoder.eval()
        model.roi_crop_encoder.freeze_encoder = True
        print(
            f"Resuming at epoch {start_epoch} (past freeze-after epoch "
            f"{SUMINANET_FREEZE_CROP_ENCODER_AFTER}): crop encoder starts frozen."
        )

    trainable_params = get_trainable_params(model)
    print(f"Trainable params: {sum(p.numel() for p in trainable_params):,}")
    print("=" * 70)

    if is_distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    # Ensure optimizer param groups include 'initial_lr' so schedulers can be
    # constructed when resuming from a checkpoint.
    for pg in optimizer.param_groups:
        if "initial_lr" not in pg:
            pg["initial_lr"] = pg.get("lr", args.lr)
    # Warm restarts every 20 epochs so LR never stays near eta_min for long.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=20,
        T_mult=1,
        eta_min=SUMINANET_LR_ETA_MIN,
        last_epoch=start_epoch - 2,
    )
    # fp16 mixed precision (Turing GPUs have no bf16 tensor core path); GradScaler
    # guards against fp16 underflow during backward.
    scaler = torch.cuda.amp.GradScaler(enabled=USE_MIXED_PRECISION)

    hard_neg_path = SUMINANET_CHECKPOINT_DIR / "hard_neg_pairs.json"

    for epoch in range(start_epoch, args.epochs + 1):
        # Load confusion pairs saved by the previous epoch's validation
        hard_neg_pairs: Optional[set] = None
        if epoch >= SUMINANET_HARD_NEG_START_EPOCH and hard_neg_path.exists():
            with open(hard_neg_path) as f:
                raw = json.load(f)
                hard_neg_pairs = {(int(g), int(p)) for g, p in raw}
            print(f"Loaded {len(hard_neg_pairs)} hard-negative pairs from {hard_neg_path.name}")

        train_metrics = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            vocab=vocab,
            epoch=epoch,
            grad_accum_steps=args.grad_accum,
            hard_neg_pairs=hard_neg_pairs,
            vocab_weights=vocab_weights,
            sam2_dir=sam2_dir,
            is_distributed=is_distributed,
            disable_tqdm=is_distributed and rank != 0,
        )

        print(
            f"Epoch {epoch}/{args.epochs}"
            f" | Train loss={train_metrics.get('train_loss', 0):.4f}"
            f" (char={train_metrics.get('train_loss_char', 0):.4f}"
            f", bg={train_metrics.get('train_loss_bg', 0):.4f}"
            f", delta={train_metrics.get('train_loss_delta', 0):.4f}"
            f", score={train_metrics.get('train_loss_score', 0):.4f})"
            f" | train_top1={train_metrics.get('train_top1', 0):.4f}"
        )
        val_metrics = None
        if rank == 0:
            raw_model = model.module if is_distributed else model
            val_metrics = validate_suminanet(
                model=raw_model,
                val_loader=val_loader,
                vocab=vocab,
                max_batches=SUMINANET_VALIDATION_BATCHES,
                vocab_weights=vocab_weights,
                sam2_dir=sam2_dir,
            )

            # Persist top-K confusion pairs for the next epoch's hard negative mining
            new_pairs = val_metrics.pop("hard_neg_pairs", [])
            if new_pairs:
                with open(hard_neg_path, "w") as f:
                    json.dump(new_pairs, f)
                print(f"Saved {len(new_pairs)} hard-negative pairs → {hard_neg_path.name}")
                decoded = [
                    f"{unicode_token_to_char(vocab.id2char.get(g, '?'))}"
                    f"→{unicode_token_to_char(vocab.id2char.get(p, '?'))}"
                    for g, p in new_pairs
                ]
                print(f"  Mined pairs: {decoded}")

        raw_model = model.module if is_distributed else model
        if (
            SUMINANET_FREEZE_CROP_ENCODER_AFTER > 0
            and epoch >= SUMINANET_FREEZE_CROP_ENCODER_AFTER
            and raw_model.roi_crop_encoder is not None
            and not raw_model.roi_crop_encoder.freeze_encoder
        ):
            for p in raw_model.roi_crop_encoder.encoder.parameters():
                p.requires_grad_(False)
            raw_model.roi_crop_encoder.encoder.eval()
            raw_model.roi_crop_encoder.freeze_encoder = True
            n_frozen = sum(p.numel() for p in raw_model.roi_crop_encoder.encoder.parameters())
            print(
                f"[Epoch {epoch}] Crop encoder frozen "
                f"({n_frozen:,} EfficientNet-B0 params removed from backprop)"
           )
            if is_distributed:
                model = DistributedDataParallel(raw_model, device_ids=[local_rank], find_unused_parameters=True)

        if rank == 0 and val_metrics is not None:
            score   = select_model_score(val_metrics)
            is_best = score > best_score
            if is_best:
                best_score   = score
                patience_ctr = 0
            else:
                patience_ctr += 1

            raw_model = model.module if is_distributed else model
            ckpt_state = {
                "epoch":            epoch,
                "model_state_dict": raw_model.state_dict(),
                "best_score":       best_score,
                "patience_ctr":     patience_ctr,
                "val_metrics":      val_metrics,
                "train_metrics":    train_metrics,
                "context_mode":     STAGE2_CONTEXT_MODE,
                "vocab_size":       vocab.vocab_size,
                "vocab_hash":       vocab.content_hash(),
                "backbone_type":    raw_model.backbone_type,
            }

            # Save epoch checkpoint (atomic write: .tmp → final path, safe on crash)
            epoch_path = SUMINANET_CHECKPOINT_DIR / f"suminanet_epoch{epoch}.pt"
            _atomic_save(ckpt_state, epoch_path)

            if is_best:
                best_path = SUMINANET_CHECKPOINT_DIR / "suminanet_best.pt"
                _atomic_save(ckpt_state, best_path)
                print(f"✅ saved best: suminanet_best.pt (score={score:.4f})")

            # Keep last 2 epoch checkpoints + best
            prune_to_keep_last_n(SUMINANET_CHECKPOINT_DIR, keep=2)

            should_stop = (
                SUMINANET_EARLY_STOPPING_PATIENCE > 0
                and patience_ctr >= SUMINANET_EARLY_STOPPING_PATIENCE
            )
            if should_stop:
                print(
                    f"Early stopping: {patience_ctr} epochs without improvement. "
                    f"best={best_score:.4f}"
                )
        else:
            should_stop = False

        if is_distributed:
            payload = [should_stop]
            dist.broadcast_object_list(payload, src=0)
            should_stop = payload[0]

        if should_stop:
            break

        scheduler.step()

    print("=" * 70)
    print(f"TRAINING COMPLETE")
    print(f"Best checkpoint: {SUMINANET_CHECKPOINT_DIR / 'suminanet_best.pt'}")
    print("=" * 70)

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
