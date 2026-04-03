"""Two-stage training pipeline: Stage 1 (detection) + Stage 2 (classification)"""
import sys
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.cuda.amp as amp
from pathlib import Path
from tqdm import tqdm

from config import (
    DATA_DIR, DEVICE, BATCH_SIZE, DROPOUT_RATE, IMAGE_SIZE, NUM_EPOCHS, LR, NUM_WORKERS, WEIGHT_DECAY,
    GRADIENT_ACCUMULATION_STEPS, CHECKPOINT_DIR, USE_MIXED_PRECISION, DETECTOR_HEATMAP_SIGMA,
    FOCAL_ALPHA, FOCAL_GAMMA, POS_WEIGHT, BBOX_WEIGHT, USE_ROI_ATTENTION, ROI_BOX_LOSS_WEIGHT,
    STAGE2_USE_CTC_WARMUP,
    STAGE2_AUX_CTC_WEIGHT,
    STAGE2_AR_LOSS_WEIGHT,
    STAGE2_AR_TF_MIN,
    STAGE2_AR_REFINEMENT_ENABLE,
    STAGE2_AR_REFINEMENT_STRENGTH,
    STAGE2_USE_CTC_PRIMARY,
    STAGE2_USE_ROI_AUX_CLASSIFIER,
    STAGE2_ROI_AUX_GT_ONLY,
    STAGE2_ROI_AUX_LOSS_WEIGHT,
    STAGE2_USE_ROI_POSITIONAL_ENCODING,
    STAGE2_USE_GT_BOXES,
    STAGE2_CURRICULUM_ENABLE,
    STAGE2_CURRICULUM_GT_EPOCHS,
    STAGE2_GRAD_ACCUMULATION_STEPS,
    STAGE2_READING_ORDER_POLICY,
    STAGE2_USE_ATTN_CENTROID_BOXES,
    STAGE2_ARCH_GUARDRAIL_STRICT,
    ROI_POOL_SIZE,
    ROI_EMBED_DIM,
    CONTEXT_HIDDEN_DIM,
    STAGE2_DET_CONFIDENCE,
    STAGE2_DET_TOP_K,
    STAGE2_DET_NMS_IOU,
    STAGE2_VAL_MAX_DECODE_LEN,
    STAGE2_EOS_LOSS_WEIGHT,
    STAGE2_CTC_DECODE_TEMPERATURE,
    STAGE2_CTC_TIME_EXPAND_FACTOR,
)
from model.kuronet import UNet, DetectorHead, ROISequenceEncoder, ROIContextEncoder
from model.kuronet.encoder_wrapper import EncoderWrapper
from model.kuronet.decoder.attention import SeqDecoderAttention
from utils import KuzushijiDataset
from utils.detection_utils import build_detection_targets, compute_roi_box_loss
from utils.focal_loss import focal_loss_heatmap
from utils.text_normalization import render_tokens
from utils.training_helpers import (
    collate_fn,
    masked_bbox_smoothl1_loss,
    prune_existing_checkpoints,
    prune_to_keep_last_n,
    scheduled_teacher_forcing,
    validate_detector,
)
from utils.vocab import VocabManager
from validate_stage1 import extract_boxes_from_heatmap


def _sort_boxes_and_labels_reading_order(boxes, labels, orientation):
    if boxes is None or len(boxes) == 0:
        return [], [] if labels is not None else []

    idx = list(range(len(boxes)))
    if orientation == "vertical":
        idx = sorted(idx, key=lambda i: (-boxes[i][0], boxes[i][1]))
    else:
        idx = sorted(idx, key=lambda i: (boxes[i][1], boxes[i][0]))

    boxes_sorted = [[float(v) for v in boxes[i]] for i in idx]
    if labels is None:
        return boxes_sorted, None

    labels_sorted = [int(labels[i]) for i in idx]
    return boxes_sorted, labels_sorted


def _infer_reading_orientation_from_boxes(boxes):
    """Infer reading direction from nearest-neighbor center distances; default vertical."""
    if boxes is None or len(boxes) < 2:
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

def _rebuild_text_ids_from_sorted_labels(labels_sorted_batch, pad_id, sos_id, eos_id, device):
    """
    Build padded text_ids from sorted per-box labels.

    Each row becomes: [SOS] + labels + [EOS]
    """
    if labels_sorted_batch is None:
        return None

    seqs = []
    max_len = 0
    for labels_sorted in labels_sorted_batch:
        seq = [int(sos_id)] + [int(x) for x in labels_sorted] + [int(eos_id)]
        t = torch.tensor(seq, dtype=torch.long, device=device)
        seqs.append(t)
        max_len = max(max_len, t.numel())

    if len(seqs) == 0:
        return None

    text_ids_sorted = torch.full(
        (len(seqs), max_len),
        fill_value=int(pad_id),
        dtype=torch.long,
        device=device,
    )
    for i, t in enumerate(seqs):
        text_ids_sorted[i, : t.numel()] = t

    return text_ids_sorted

def _resolve_sort_orientation(orientation_hint, boxes, reading_order_policy):
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

def _build_boxes_for_encoder(
    images,
    boxes_batch,
    labels_batch,
    orientations,
    use_gt_boxes,
    unet,
    detector,
    reading_order_policy="annotation",
    pad_id=None,
    sos_id=None,
    eos_id=None,
):
    """
    Prepare box sequences for ROI encoder and keep GT labels aligned to ROI order.

    Returns:
        boxes_for_encoder: list[Tensor(N_i, 4)]
        labels_sorted_batch: list[list[int]] or None
        text_ids_sorted: Tensor(B, T) or None
    """
    if use_gt_boxes:
        out_boxes = []
        out_label_ids = []

        batch_size = len(boxes_batch) if boxes_batch is not None else 0

        for i in range(batch_size):
            b = boxes_batch[i]
            l = labels_batch[i] if labels_batch is not None and i < len(labels_batch) else None

            if b is None:
                out_boxes.append(torch.empty((0, 4), dtype=torch.float32, device=DEVICE))
                out_label_ids.append([])
                continue

            boxes_i = b.to(DEVICE, dtype=torch.float32)
            boxes_i_list = boxes_i.detach().cpu().tolist()

            if l is None:
                labels_i_list = []
            else:
                labels_i_list = [int(x) for x in l.detach().cpu().tolist()]

            orientation_hint = orientations[i] if orientations is not None and i < len(orientations) else None
            sort_orientation = _resolve_sort_orientation(
                orientation_hint,
                boxes_i_list,
                reading_order_policy,
            )

            boxes_sorted, labels_sorted = _sort_boxes_and_labels_reading_order(
                boxes_i_list,
                labels_i_list,
                sort_orientation,
            )

            if len(boxes_sorted) == 0:
                out_boxes.append(torch.empty((0, 4), dtype=torch.float32, device=DEVICE))
            else:
                out_boxes.append(torch.tensor(boxes_sorted, dtype=torch.float32, device=DEVICE))

            out_label_ids.append(labels_sorted if labels_sorted is not None else [])

        text_ids_sorted = None
        if pad_id is not None and sos_id is not None and eos_id is not None:
            text_ids_sorted = _rebuild_text_ids_from_sorted_labels(
                out_label_ids,
                pad_id=pad_id,
                sos_id=sos_id,
                eos_id=eos_id,
                device=DEVICE,
            )

        return out_boxes, out_label_ids, text_ids_sorted

    with torch.no_grad():
        features = unet(images)
        det_out = detector(features)
        heat_probs = torch.sigmoid(det_out["heatmap"])
        bbox_reg = det_out["bbox"]
        _, _, hf, wf = features.shape

        out_boxes = []
        for i in range(images.size(0)):
            pred_boxes_i, _, _ = extract_boxes_from_heatmap(
                heatmap_probs=heat_probs[i:i + 1],
                bbox_reg=bbox_reg[i:i + 1],
                confidence_thresh=STAGE2_DET_CONFIDENCE,
                output_size=(hf, wf),
                image_size=IMAGE_SIZE,
                top_k=STAGE2_DET_TOP_K,
                nms_iou=STAGE2_DET_NMS_IOU,
                min_box_size=4.0,
                debug=False,
            )

            orientation_hint = orientations[i] if orientations is not None and i < len(orientations) else None
            sort_orientation = _resolve_sort_orientation(
                orientation_hint,
                pred_boxes_i,
                reading_order_policy,
            )
            pred_boxes_i, _ = _sort_boxes_and_labels_reading_order(
                pred_boxes_i,
                None,
                sort_orientation,
            )

            if len(pred_boxes_i) == 0:
                out_boxes.append(torch.empty((0, 4), dtype=torch.float32, device=DEVICE))
            else:
                out_boxes.append(torch.tensor(pred_boxes_i, dtype=torch.float32, device=DEVICE))

        # no aligned labels available in detector mode
        return out_boxes, None, None


def _filter_stage2_batch(batch, device):
    """Filter and align Stage 2 batch tensors/lists exactly once."""
    images = batch["image"].to(device)
    text_ids = batch["text_ids"].to(device) if batch.get("text_ids", None) is not None else None
    boxes_batch = batch.get("boxes", None)
    labels_batch = batch.get("labels", None)
    orientations = batch.get("orientations", None)
    text_ids_present = batch.get("text_ids_present", None)

    if text_ids is None:
        return None

    if text_ids_present is not None:
        valid_idx = text_ids_present.to(device).nonzero(as_tuple=False).squeeze(1)
        if valid_idx.numel() == 0:
            return None

        images = images.index_select(0, valid_idx)
        text_ids = text_ids.index_select(0, valid_idx)
        valid_idx_cpu = valid_idx.detach().cpu().tolist()

        if orientations is not None:
            orientations = [orientations[i] for i in valid_idx_cpu]
        if boxes_batch is not None:
            boxes_batch = [boxes_batch[i] for i in valid_idx_cpu]
        if labels_batch is not None:
            labels_batch = [labels_batch[i] for i in valid_idx_cpu]

    return {
        "images": images,
        "text_ids": text_ids,
        "boxes_batch": boxes_batch,
        "labels_batch": labels_batch,
        "orientations": orientations,
    }

def _get_stage2_trainable_params(encoder, roi_sequence_encoder, context_encoder, decoder, ctc_head=None, roi_aux_head=None):
    params = []
    modules = [encoder, roi_sequence_encoder, context_encoder, decoder]
    if ctc_head is not None:
        modules.append(ctc_head)
    if roi_aux_head is not None:
        modules.append(roi_aux_head)

    for module in modules:
        params.extend([p for p in module.parameters() if p.requires_grad])

    return params


def _prepare_roi_aux_supervision_from_sorted_labels(roi_logits, labels_sorted_batch, roi_mask):
    """
    Supervise ROI logits directly with labels already sorted in the same order as ROI boxes.

    Args:
        roi_logits: (B, T_roi, V)
        labels_sorted_batch: list[list[int]] aligned to ROI order
        roi_mask: (B, T_roi)

    Returns:
        flat_logits, flat_targets or (None, None)
    """
    if labels_sorted_batch is None:
        return None, None

    logits_chunks = []
    target_chunks = []

    for i in range(len(labels_sorted_batch)):
        labels_sorted = labels_sorted_batch[i]
        if labels_sorted is None or len(labels_sorted) == 0:
            continue

        roi_len = int(roi_mask[i].sum().item())
        if roi_len <= 0:
            continue

        k = min(roi_len, len(labels_sorted), int(roi_logits.size(1)))
        if k <= 0:
            continue

        logits_chunks.append(roi_logits[i, :k, :])
        target_chunks.append(
            torch.tensor(labels_sorted[:k], dtype=torch.long, device=roi_logits.device)
        )

    if len(logits_chunks) == 0:
        return None, None

    flat_logits = torch.cat(logits_chunks, dim=0)
    flat_targets = torch.cat(target_chunks, dim=0)
    return flat_logits, flat_targets

def _expand_ctc_timesteps(enc_outputs, enc_mask, factor):
    """Expand encoder timesteps for CTC alignment slack by simple repeat-interleave."""
    f = max(1, int(factor))
    if f == 1:
        return enc_outputs, enc_mask

    enc_outputs_exp = enc_outputs.repeat_interleave(f, dim=1)
    enc_mask_exp = enc_mask.repeat_interleave(f, dim=1)
    return enc_outputs_exp, enc_mask_exp


def _build_ar_refinement_bias_from_ctc(ctc_logits, dec_len, blank_id, strength=0.2):
    """Build AR logit bias from CTC posterior by resampling time to decoder steps."""
    if ctc_logits is None or dec_len <= 0:
        return None

    # ctc_logits shape: (B, T_ctc, V+1) where class index blank_id is the blank token.
    ctc_probs = ctc_logits.float().softmax(dim=-1)
    ctc_non_blank = ctc_probs[..., :blank_id]  # (B, T_ctc, V)
    ctc_non_blank = ctc_non_blank.permute(0, 2, 1).contiguous()  # (B, V, T_ctc)
    ctc_non_blank = F.interpolate(
        ctc_non_blank,
        size=int(dec_len),
        mode="linear",
        align_corners=False,
    )
    ctc_non_blank = ctc_non_blank.permute(0, 2, 1).contiguous()  # (B, T_dec, V)
    return float(strength) * torch.log(ctc_non_blank.clamp_min(1e-8))


def _prepare_ctc_targets(text_ids, pad_id, sos_id, eos_id, input_lengths):
    """Build CTC targets per batch and keep only rows valid for CTC."""
    targets = []
    target_lengths = []
    keep_indices = []
    skipped_too_long = 0
    skipped_repeats_too_short = 0
    total_rows = int(text_ids.size(0))
    min_required_lengths = []
    input_minus_required = []

    for i in range(text_ids.size(0)):
        ids = text_ids[i]
        ids = ids[ids != pad_id]
        if ids.numel() == 0:
            continue

        if ids[0].item() == sos_id:
            ids = ids[1:]
        if ids.numel() > 0 and ids[-1].item() == eos_id:
            ids = ids[:-1]
        if ids.numel() == 0:
            continue

        # Basic CTC feasibility: target length cannot exceed input length.
        if ids.numel() > int(input_lengths[i].item()):
            skipped_too_long += 1
            continue

        # Strict CTC feasibility for repeated adjacent labels:
        # minimum input length must be target_len + num_adjacent_repeats.
        # Otherwise repeated characters are impossible to align.
        if ids.numel() > 1:
            adj_repeats = int((ids[1:] == ids[:-1]).sum().item())
        else:
            adj_repeats = 0
        min_required = int(ids.numel()) + adj_repeats
        if min_required > int(input_lengths[i].item()):
            skipped_repeats_too_short += 1
            continue

        targets.append(ids)
        target_lengths.append(ids.numel())
        keep_indices.append(i)
        min_required_lengths.append(float(min_required))
        input_minus_required.append(float(int(input_lengths[i].item()) - min_required))

    if len(targets) == 0:
        diag = {
            "total_rows": total_rows,
            "kept_rows": 0,
            "skipped_too_long": skipped_too_long,
            "skipped_repeats_too_short": skipped_repeats_too_short,
            "mean_input_len_kept": 0.0,
            "mean_target_len_kept": 0.0,
            "mean_min_required_len_kept": 0.0,
            "mean_input_minus_required_kept": 0.0,
        }
        return None, None, None, diag

    targets_concat = torch.cat(targets)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long, device=text_ids.device)
    keep_indices = torch.tensor(keep_indices, dtype=torch.long, device=text_ids.device)
    diag = {
        "total_rows": total_rows,
        "kept_rows": int(keep_indices.numel()),
        "skipped_too_long": skipped_too_long,
        "skipped_repeats_too_short": skipped_repeats_too_short,
        "mean_input_len_kept": float(input_lengths[keep_indices].float().mean().item()) if keep_indices.numel() > 0 else 0.0,
        "mean_target_len_kept": float(target_lengths.float().mean().item()) if target_lengths.numel() > 0 else 0.0,
        "mean_min_required_len_kept": float(sum(min_required_lengths) / max(1, len(min_required_lengths))),
        "mean_input_minus_required_kept": float(sum(input_minus_required) / max(1, len(input_minus_required))),
    }
    return targets_concat, target_lengths, keep_indices, diag


def _validate_stage2_guardrails(num_epochs):
    policy = str(STAGE2_READING_ORDER_POLICY).strip().lower()
    allowed = {"annotation", "inferred", "auto"}
    issues = []

    if policy not in allowed:
        issues.append(f"invalid STAGE2_READING_ORDER_POLICY={STAGE2_READING_ORDER_POLICY}")
    if not (isinstance(ROI_POOL_SIZE, tuple) and len(ROI_POOL_SIZE) == 2):
        issues.append(f"ROI_POOL_SIZE must be tuple(len=2), got {ROI_POOL_SIZE}")
    if int(STAGE2_CURRICULUM_GT_EPOCHS) < 0:
        issues.append(f"STAGE2_CURRICULUM_GT_EPOCHS must be >= 0, got {STAGE2_CURRICULUM_GT_EPOCHS}")
    if int(STAGE2_GRAD_ACCUMULATION_STEPS) < 1:
        issues.append(f"STAGE2_GRAD_ACCUMULATION_STEPS must be >= 1, got {STAGE2_GRAD_ACCUMULATION_STEPS}")

    warn_only = None
    if STAGE2_CURRICULUM_ENABLE and int(STAGE2_CURRICULUM_GT_EPOCHS) >= int(num_epochs):
        warn_only = (
            f"Curriculum GT epochs ({int(STAGE2_CURRICULUM_GT_EPOCHS)}) >= num_epochs ({int(num_epochs)}); "
            "run will stay in GT box mode and not switch to predicted boxes."
        )

    if issues and STAGE2_ARCH_GUARDRAIL_STRICT:
        raise ValueError("Stage2 architecture guardrail violation: " + " | ".join(issues))
    if issues:
        print("[WARN] Stage2 architecture guardrail issues: " + " | ".join(issues))
    if warn_only:
        print("[WARN] " + warn_only)

    return {
        "strict": bool(STAGE2_ARCH_GUARDRAIL_STRICT),
        "issues": issues,
        "warning": warn_only,
        "reading_order_policy": policy,
    }

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
            curr.append(min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + cost,
            ))
        prev = curr
    return prev[-1]

def _decode_text_from_ids(ids, vocab):
    chars = vocab.decode([int(x) for x in ids], remove_special=True)
    return render_tokens(chars)


def _truncate_at_eos(ids, eos_id):
    out = []
    for tok in ids:
        out.append(int(tok))
        if int(tok) == eos_id:
            break
    return out

def _ctc_greedy_decode(logits, blank_id=None, max_len=256, temperature=1.0):
    """
    Greedy CTC decoding: remove blanks and collapse consecutive duplicates.
    
    Args:
        logits: (T, V+1) raw logits or (B, T, V+1) batched logits
        blank_id: ID for blank token (default: last ID)
        max_len: max output length
    
    Returns:
        List of token IDs (single sequence) or list of lists (batch)
    """
    if logits.dim() == 2:
        if temperature > 0.0 and abs(float(temperature) - 1.0) > 1e-6:
            logits = logits / float(temperature)
        # Single sequence (T, V)
        pred_ids = logits.argmax(dim=-1).tolist()
        if blank_id is None:
            blank_id = logits.shape[-1] - 1
        
        # Remove blanks and collapse duplicates
        output = []
        prev_id = None
        for tok_id in pred_ids:
            if tok_id != blank_id and tok_id != prev_id:
                output.append(int(tok_id))
                if len(output) >= max_len:
                    break
            prev_id = tok_id
        return output
    
    elif logits.dim() == 3:
        if temperature > 0.0 and abs(float(temperature) - 1.0) > 1e-6:
            logits = logits / float(temperature)
        # Batch (B, T, V)
        batch_output = []
        if blank_id is None:
            blank_id = logits.shape[-1] - 1
        
        for sample_logits in logits:
            pred_ids = sample_logits.argmax(dim=-1).tolist()
            output = []
            prev_id = None
            for tok_id in pred_ids:
                if tok_id != blank_id and tok_id != prev_id:
                    output.append(int(tok_id))
                    if len(output) >= max_len:
                        break
                prev_id = tok_id
            batch_output.append(output)
        return batch_output
    
    else:
        raise ValueError(f"Expected logits to be 2D or 3D, got {logits.dim()}D")

def validate_sequence_stage(
    encoder,
    roi_sequence_encoder,
    context_encoder,
    decoder,
    ctc_head,
    dataloader,
    vocab,
    ce_loss,
    use_gt_boxes,
    unet,
    detector,
    reading_order_policy="annotation",
    max_decode_len=256,
    num_debug_samples=3,
    use_ctc_primary=False,
    ctc_decode_temperature=1.0,
):
    encoder.eval()
    roi_sequence_encoder.eval()
    context_encoder.eval()
    decoder.eval()
    ctc_head.eval()

    total_loss = 0.0
    total_cer = 0.0
    total_exact = 0
    total_samples = 0
    n_batches = 0
    total_pred_len = 0.0
    total_gt_len = 0.0
    total_max_decode_hits = 0
    debug_samples = []
    pred_token_counts = {}
    pred_len_bins = {"short": 0, "medium": 0, "long": 0}
    cer_len_buckets = {
        "short": {"cer_sum": 0.0, "count": 0},
        "medium": {"cer_sum": 0.0, "count": 0},
        "long": {"cer_sum": 0.0, "count": 0},
    }

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validate Stage2", leave=False):
            prepared = _filter_stage2_batch(batch, DEVICE)
            if prepared is None:
                continue

            images = prepared["images"]
            text_ids = prepared["text_ids"]
            boxes_batch = prepared["boxes_batch"]
            labels_batch = prepared["labels_batch"]
            orientations = prepared["orientations"]

            boxes_for_encoder, labels_sorted_batch, text_ids_sorted = _build_boxes_for_encoder(
                images=images,
                boxes_batch=boxes_batch,
                labels_batch=labels_batch,
                orientations=orientations,
                use_gt_boxes=use_gt_boxes,
                unet=unet,
                detector=detector,
                reading_order_policy=reading_order_policy,
                pad_id=vocab.pad_id,
                sos_id=vocab.sos_id,
                eos_id=vocab.eos_id,
            )

            if use_gt_boxes and text_ids_sorted is not None:
                seq_text_ids = text_ids_sorted
            else:
                seq_text_ids = text_ids

            input_seq = seq_text_ids[:, :-1]
            targets = seq_text_ids[:, 1:]

            feats_2d = encoder(images, return_2d=True)
            roi_seq, roi_mask = roi_sequence_encoder(feats_2d, boxes_for_encoder, image_size=IMAGE_SIZE)
            enc_outputs, enc_mask = context_encoder(roi_seq, roi_mask)

            if use_ctc_primary:
                enc_outputs_ctc, enc_mask_ctc = _expand_ctc_timesteps(
                    enc_outputs, enc_mask, STAGE2_CTC_TIME_EXPAND_FACTOR
                )
                ctc_logits = ctc_head(enc_outputs_ctc)  # (B, T, V+1)
                ctc_log_probs = ctc_logits.float().log_softmax(dim=-1).permute(1, 0, 2).contiguous()
                ctc_input_lengths = enc_mask_ctc.sum(dim=1).clamp(min=1).to(dtype=torch.long)

                targets_concat, target_lengths, keep_indices, _ = _prepare_ctc_targets(
                    text_ids=seq_text_ids,
                    pad_id=vocab.pad_id,
                    sos_id=vocab.sos_id,
                    eos_id=vocab.eos_id,
                    input_lengths=ctc_input_lengths,
                )

                ctc_blank_id = vocab.vocab_size
                ctc_loss_fn = nn.CTCLoss(blank=ctc_blank_id, zero_infinity=True)

                if targets_concat is not None:
                    loss = ctc_loss_fn(
                        ctc_log_probs[:, keep_indices, :],
                        targets_concat,
                        ctc_input_lengths[keep_indices],
                        target_lengths,
                    )
                else:
                    loss = torch.tensor(0.0, device=DEVICE)

                total_loss += float(loss.item())
                n_batches += 1

                pred_ids_batch = _ctc_greedy_decode(
                    ctc_logits,
                    blank_id=ctc_blank_id,
                    max_len=max_decode_len,
                    temperature=ctc_decode_temperature,
                )
                gt_ids_batch = seq_text_ids.detach().cpu().tolist()

            else:
                decoder_output = decoder(
                    input_seq=input_seq,
                    enc_outputs=enc_outputs,
                    enc_mask=enc_mask,
                    teacher_forcing_ratio=1.0,
                    targets=targets,
                    eos_id=vocab.eos_id,
                    image_size=(images.shape[2], images.shape[3]),
                )

                if len(decoder_output) == 4:
                    logits, _, _, _ = decoder_output
                else:
                    logits, _, _ = decoder_output

                B, T, V = logits.shape
                loss = ce_loss(logits.reshape(-1, V), targets.reshape(-1))
                total_loss += float(loss.item())
                n_batches += 1

                decoder_free = decoder(
                    input_seq=None,
                    enc_outputs=enc_outputs,
                    enc_mask=enc_mask,
                    teacher_forcing_ratio=0.0,
                    targets=None,
                    sos_id=vocab.sos_id,
                    eos_id=vocab.eos_id,
                    max_len=max_decode_len,
                    image_size=(images.shape[2], images.shape[3]),
                )

                if len(decoder_free) == 4:
                    free_logits, _, _, _ = decoder_free
                else:
                    free_logits, _, _ = decoder_free

                pred_ids_batch = free_logits.argmax(dim=-1).detach().cpu().tolist()
                gt_ids_batch = seq_text_ids.detach().cpu().tolist()

                for i in range(len(pred_ids_batch)):
                    pred_ids_batch[i] = _truncate_at_eos(pred_ids_batch[i], vocab.eos_id)

            for pred_ids, gt_ids in zip(pred_ids_batch, gt_ids_batch):
                pred_ids = [int(x) for x in pred_ids] if pred_ids else []
                gt_ids = [int(x) for x in gt_ids] if gt_ids else []

                gt_ids = _truncate_at_eos(gt_ids, vocab.eos_id)

                pred_text = _decode_text_from_ids(pred_ids, vocab)
                gt_text = _decode_text_from_ids(gt_ids, vocab)

                dist = _edit_distance(pred_text, gt_text)
                cer = dist / max(1, len(gt_text))

                total_cer += cer
                total_exact += int(pred_text == gt_text)
                total_pred_len += float(len(pred_ids))
                gt_len = len(gt_ids)
                total_gt_len += float(gt_len)

                if len(pred_ids) < 128:
                    pred_len_bins["short"] += 1
                elif len(pred_ids) < 256:
                    pred_len_bins["medium"] += 1
                else:
                    pred_len_bins["long"] += 1

                if gt_len < 128:
                    cer_bucket = "short"
                elif gt_len < 256:
                    cer_bucket = "medium"
                else:
                    cer_bucket = "long"
                cer_len_buckets[cer_bucket]["cer_sum"] += float(cer)
                cer_len_buckets[cer_bucket]["count"] += 1

                for tok in pred_ids:
                    tok_i = int(tok)
                    pred_token_counts[tok_i] = pred_token_counts.get(tok_i, 0) + 1

                if use_ctc_primary:
                    hit_max_decode = len(pred_ids) >= max_decode_len
                else:
                    hit_max_decode = (
                        len(pred_ids) >= max_decode_len and
                        (len(pred_ids) == 0 or int(pred_ids[-1]) != int(vocab.eos_id))
                    )
                total_max_decode_hits += int(hit_max_decode)
                total_samples += 1

                if num_debug_samples > 0:
                    sample = {
                        "cer": float(cer),
                        "pred_len": int(len(pred_ids)),
                        "gt_len": int(gt_len),
                        "hit_max_decode": bool(hit_max_decode),
                        "pred": pred_text,
                        "gt": gt_text,
                    }
                    if len(debug_samples) < num_debug_samples:
                        debug_samples.append(sample)
                    else:
                        replace_idx = random.randint(0, total_samples - 1)
                        if replace_idx < num_debug_samples:
                            debug_samples[replace_idx] = sample

    denom_batches = max(1, n_batches)
    denom_samples = max(1, total_samples)

    val_mean_pred_len = total_pred_len / denom_samples
    val_mean_gt_len = total_gt_len / denom_samples
    val_len_ratio = val_mean_pred_len / max(1e-6, val_mean_gt_len)

    total_pred_tokens = max(1, int(sum(pred_token_counts.values())))
    top1_token_share = 0.0
    if pred_token_counts:
        top1_token_share = max(pred_token_counts.values()) / float(total_pred_tokens)

    cer_by_len_bucket = {}
    for bucket_name, bucket_stats in cer_len_buckets.items():
        denom = max(1, int(bucket_stats["count"]))
        cer_by_len_bucket[bucket_name] = float(bucket_stats["cer_sum"]) / float(denom)

    return {
        "val_loss": total_loss / denom_batches,
        "val_cer": total_cer / denom_samples,
        "val_exact": total_exact / denom_samples,
        "val_samples": total_samples,
        "val_mean_pred_len": val_mean_pred_len,
        "val_mean_gt_len": val_mean_gt_len,
        "val_len_ratio": val_len_ratio,
        "val_max_decode_frac": total_max_decode_hits / denom_samples,
        "val_top1_token_share": top1_token_share,
        "val_pred_len_hist": pred_len_bins,
        "val_cer_by_len_bucket": cer_by_len_bucket,
        "debug_samples": debug_samples,
    }

def train_detector_stage(num_epochs=10, lr=None, checkpoint_dir=None, patience=3, val_split=None):
    """Stage 1: Train DetectorHead to localize characters.
    
    Args:
        num_epochs: Number of epochs for detector training
        lr: Learning rate (default from config)
        checkpoint_dir: Where to save checkpoints
        patience: Early stopping patience (0 = disable)
        val_split: Use predefined splits or not (None = use splits/train.txt and splits/val.txt)
    """
    if lr is None:
        lr = LR
    if checkpoint_dir is None:
        checkpoint_dir = CHECKPOINT_DIR / "stage1_detection"
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    # Build or load vocab
    ann_files = sorted(list((Path(DATA_DIR) / "annotations").glob("*.json")))
    if len(ann_files) == 0:
        raise FileNotFoundError(f"No annotation files found in {Path(DATA_DIR)/'annotations'}")
    vocab = VocabManager.from_annotations(ann_files)
    pad_id = vocab.pad_id

    # Clean up existing checkpoints: keep only newest, rename to checkpoint_old.pt
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    prune_existing_checkpoints(checkpoint_dir)

    # Dataset + loader - use pre-existing splits or random split if val_split provided
    if val_split is None:
        train_dataset = KuzushijiDataset(Path(DATA_DIR), vocab=vocab, use_sequences=True, resize=IMAGE_SIZE, split='train')
        val_dataset = KuzushijiDataset(Path(DATA_DIR), vocab=vocab, use_sequences=True, resize=IMAGE_SIZE, split='val')
    else:
        full_dataset = KuzushijiDataset(Path(DATA_DIR), vocab=vocab, use_sequences=True, resize=IMAGE_SIZE, split=None)
        if isinstance(val_split, float):
            if not (0.0 < val_split < 1.0):
                raise ValueError("val_split as float must be in (0, 1)")
            val_size = max(1, int(len(full_dataset) * val_split))
        elif isinstance(val_split, int):
            if val_split <= 0 or val_split >= len(full_dataset):
                raise ValueError("val_split as int must be in [1, len(dataset)-1]")
            val_size = val_split
        else:
            raise TypeError("val_split must be None, float, or int")
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                            collate_fn=lambda b: collate_fn(b, pad_id), pin_memory=True,
                            prefetch_factor=2, persistent_workers=NUM_WORKERS > 0)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                            collate_fn=lambda b: collate_fn(b, pad_id), pin_memory=True,
                            prefetch_factor=2, persistent_workers=NUM_WORKERS > 0)
    
    # Build model
    unet = UNet(in_channels=3, base_features=32).to(DEVICE)
    detector = DetectorHead(in_ch=32, num_classes=vocab.vocab_size, dropout_rate=DROPOUT_RATE, predict_classes=False).to(DEVICE)  # Disable class head to save memory
    print(f"✅ Model initialized with {vocab.vocab_size} classes (from vocab)")
    print(f"⚠️  Class prediction disabled to prevent OOM (Stage 1 focuses on spatial detection only)")
    # Optimizer
    optimizer = optim.Adam(
        list(unet.parameters()) + list(detector.parameters()),
        lr=lr, weight_decay=WEIGHT_DECAY
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    # Mixed precision (use new API to avoid deprecation warning)
    scaler = torch.amp.GradScaler(device='cuda') if USE_MIXED_PRECISION else None
    
    print(f"Stage 1: Training DetectorHead for {num_epochs} epochs (early stopping patience={patience})...")
    print(f"Training set: {len(train_dataset)} images | Validation set: {len(val_dataset)} images")
    bbox_radius = 0
    best_val = None
    patience_ctr = 0
    for epoch in range(num_epochs):
        # Training
        unet.train()
        detector.train()
        optimizer.zero_grad(set_to_none=True)
        
        total_loss = 0.0
        total_heat = 0.0
        total_bbox = 0.0
        n_batches = 0

        # Show progress bar with limited update frequency to reduce log spam
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", 
                   mininterval=30.0)  
        for step, batch in enumerate(pbar):
            images = batch['image'].to(DEVICE)
            boxes = [b.to(DEVICE) if b.numel() > 0 else torch.empty((0, 4), device=DEVICE) for b in batch.get('boxes', [])]
            labels = [l.to(DEVICE) if l.numel() > 0 else torch.empty((0,), dtype=torch.long, device=DEVICE) for l in batch.get('labels', [])]
            
            with torch.amp.autocast(device_type='cuda', enabled=USE_MIXED_PRECISION):
                # Single forward pass
                features = unet(images)  # (B, 32, H/8, W/8)
                outputs = detector(features)  # Returns dict with 'heatmap' (raw logits), 'bbox', etc.
                
                B, _, Hf, Wf = features.shape
                gt_heat, gt_bbox, gt_bbox_mask, gt_cls = build_detection_targets(
                    boxes,
                    labels,
                    output_size=(Hf, Wf),
                    image_size=tuple(images.shape[-2:]),
                    device=DEVICE,
                    sigma=DETECTOR_HEATMAP_SIGMA,
                    bbox_radius=bbox_radius,
                )
                # Compute losses with detailed tracking
                loss_heatmap = focal_loss_heatmap(
                    outputs["heatmap"], gt_heat, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, pos_weight=POS_WEIGHT
                )
                # Lower pos_thresh for bbox (include broader region around peak) and increase bbox weight
                loss_bbox = masked_bbox_smoothl1_loss(outputs["bbox"], gt_bbox, gt_bbox_mask)
                                
                # Balanced loss: heatmap for localization, bbox for accurate box dimensions
                loss = loss_heatmap + BBOX_WEIGHT * loss_bbox  
                loss = loss / GRADIENT_ACCUMULATION_STEPS
            
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Optimizer step with gradient accumulation
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(list(unet.parameters()) + list(detector.parameters()), 1.0)

                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
            
            # logging (unscaled)
            total_loss += float(loss.item() * GRADIENT_ACCUMULATION_STEPS)
            total_heat += float(loss_heatmap.item())
            total_bbox += float(loss_bbox.item())
            n_batches += 1

            pbar.set_postfix({
                "loss": total_loss / n_batches,
                "heat": total_heat / n_batches,
                "bbox": total_bbox / n_batches
            })
            
            if epoch == 0 and step == 0:
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
            # Clear cache periodically to prevent memory fragmentation
            if step % 50 == 0:
                torch.cuda.empty_cache()
        
        if n_batches > 0 and (n_batches % GRADIENT_ACCUMULATION_STEPS) != 0:
            if scaler is not None:
                scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                list(unet.parameters()) + list(detector.parameters()), 1.0
            )

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        torch.cuda.empty_cache()
        
        # Validation
        val_loss, val_heat, val_bbox = validate_detector(
            unet, detector, val_dataloader, DEVICE, USE_MIXED_PRECISION, bbox_radius=bbox_radius
        )
        train_loss = total_loss / max(1, n_batches)
        print(
            f"\nEpoch {epoch+1}/{num_epochs}  "
            f"Train: {train_loss:.4f} (heat={total_heat/max(1,n_batches):.4f}, bbox={total_bbox/max(1,n_batches):.4f})  "
            f"Val: {val_loss:.4f} (heat={val_heat:.4f}, bbox={val_bbox:.4f})"
        )

        # Early stopping check
        is_best = (best_val is None) or (val_loss < best_val)
        if is_best:
            best_val = val_loss
            patience_ctr = 0
        else:
            patience_ctr += 1
        
        ckpt = {
            "epoch": epoch + 1,
            "unet_state_dict": unet.state_dict(),
            "detector_state_dict": detector.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "vocab_size": vocab.vocab_size,
            "is_best": is_best,
        }

        epoch_path = checkpoint_dir / f"detector_epoch{epoch+1}.pt"
        torch.save(ckpt, epoch_path)

        if is_best:
            torch.save(ckpt, checkpoint_dir / "detector_best.pt")
            print(f"✅ saved best: detector_best.pt (val={val_loss:.4f})")

        # DISABLED: Early stopping to collect full training curves
        # if patience > 0 and patience_ctr >= patience:
        #     print(f"⏹️ early stop (no val improvement for {patience} epochs). best={best_val:.4f}")
        #     break    
    
    print(f"\n{'='*60}")
    print(f"Stage 1 training complete!")
    print(f"Best checkpoint: {checkpoint_dir / 'detector_best.pt'}")
    print(f"{'='*60}\n")
    
    return unet, detector


def train_sequence_stage(detector_ckpt_path, num_epochs=10, lr=None, checkpoint_dir=None, use_ctc_warmup=False):
    """Stage 2: Train Encoder + Decoder for sequence prediction with context.
    
    Uses detected boxes to guide attention and sequence prediction.
    Combines spatial detection (Stage 1) with contextual understanding.
    Architecture:
        image -> UNet backbone -> 2D projected features
             -> ROISequenceEncoder (ordered candidate boxes)
             -> ROIContextEncoder
             -> attention decoder

    Modes:
        Oracle mode:
            Stage 2 uses GT boxes from dataset
        Detector mode:
            Stage 2 uses Stage 1 predicted boxes
    Args:
        detector_ckpt_path: Path to detector checkpoint
        num_epochs: Number of epochs for sequence training
        lr: Learning rate (default from config)
        checkpoint_dir: Where to save checkpoints
        use_ctc_warmup: Whether to do CTC warmup before attention training
    """
    if lr is None:
        lr = LR
    if checkpoint_dir is None:
        checkpoint_dir = CHECKPOINT_DIR / "stage2_sequence_test2"
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    guardrail_info = _validate_stage2_guardrails(num_epochs=num_epochs)
    
    # Build or load vocab
    ann_files = sorted(list((Path(DATA_DIR) / "annotations").glob("*.json")))
    if len(ann_files) == 0:
        raise FileNotFoundError(f"No annotation files found in {Path(DATA_DIR)/'annotations'}")
    vocab = VocabManager.from_annotations(ann_files)
    pad_id = vocab.pad_id
    sos_id = vocab.sos_id
    eos_id = vocab.eos_id
    vocab_size = vocab.vocab_size

    # Dataset + loader
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

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=lambda b: collate_fn(b, pad_id),
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=NUM_WORKERS > 0,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=lambda b: collate_fn(b, pad_id),
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=NUM_WORKERS > 0,
    )
    
    print(f"Stage 2 datasets -> train: {len(train_dataset)} | val: {len(val_dataset)}")
    # Build and load detector (frozen for spatial guidance)
    unet = UNet(in_channels=3, base_features=32).to(DEVICE)
    detector = DetectorHead(in_ch=32, num_classes=vocab_size).to(DEVICE)
    
    checkpoint = torch.load(detector_ckpt_path, map_location=DEVICE)
    unet.load_state_dict(checkpoint['unet_state_dict'])
    detector.load_state_dict(checkpoint['detector_state_dict'])
    
    # Freeze detector - we only use it for guidance
    for p in unet.parameters():
        p.requires_grad = False
    for p in detector.parameters():
        p.requires_grad = False
    
    unet.eval()
    detector.eval()
    # -------------------------
    # Stage 2 modules
    # -------------------------

    encoder = EncoderWrapper(backbone=unet, in_channels=32, enc_dim=256).to(DEVICE)
    roi_sequence_encoder = ROISequenceEncoder(
        in_dim=256,
        roi_size=ROI_POOL_SIZE,
        out_dim=ROI_EMBED_DIM,
        use_geo_positional_encoding=STAGE2_USE_ROI_POSITIONAL_ENCODING,
    ).to(DEVICE)
    context_encoder = ROIContextEncoder(
        in_dim=ROI_EMBED_DIM,
        hidden_dim=CONTEXT_HIDDEN_DIM,
        out_dim=CONTEXT_HIDDEN_DIM,
    ).to(DEVICE)

    decoder = SeqDecoderAttention(
        embed_dim=64,
        hidden_dim=CONTEXT_HIDDEN_DIM,
        vocab_size=vocab_size,
        enc_dim=CONTEXT_HIDDEN_DIM,
        num_layers=1,
        init_from_encoder=True,
        sampling_method="argmax",
        use_roi_attention=USE_ROI_ATTENTION,
        use_attn_centroid_boxes=STAGE2_USE_ATTN_CENTROID_BOXES,
    ).to(DEVICE)

    ctc_blank_id = vocab_size
    ctc_head = nn.Linear(CONTEXT_HIDDEN_DIM, vocab_size + 1).to(DEVICE)
    roi_aux_head = nn.Linear(ROI_EMBED_DIM, vocab_size).to(DEVICE)
    
    # -------------------------
    # optimizer / scheduler / scaler
    # -------------------------

    optimizer = optim.AdamW(
        list(encoder.parameters())
        + list(roi_sequence_encoder.parameters())
        + list(context_encoder.parameters())
        + list(decoder.parameters())
        + list(ctc_head.parameters())
        + list(roi_aux_head.parameters()),
        lr=lr, 
        weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    amp_enabled = USE_MIXED_PRECISION and str(DEVICE).startswith("cuda")
    scaler = amp.GradScaler() if amp_enabled else None
    stage2_accum_steps = max(1, int(STAGE2_GRAD_ACCUMULATION_STEPS))
    
    # Losses
    ce_weight = torch.ones(vocab_size, dtype=torch.float32, device=DEVICE)
    ce_weight[eos_id] = float(STAGE2_EOS_LOSS_WEIGHT)
    ce_loss = nn.CrossEntropyLoss(ignore_index=pad_id, weight=ce_weight)
    roi_aux_ce_loss = nn.CrossEntropyLoss(ignore_index=pad_id)
    print(f"Stage2 EOS CE weight: {STAGE2_EOS_LOSS_WEIGHT:.2f}")
    trainable_params = _get_stage2_trainable_params(
    encoder, roi_sequence_encoder, context_encoder, decoder, ctc_head, roi_aux_head
    )

    run_config = {
        "stage2_use_gt_boxes": bool(STAGE2_USE_GT_BOXES),
        "stage2_decode_boxes": "n/a_training",
        "stage2_curriculum_enable": bool(STAGE2_CURRICULUM_ENABLE),
        "stage2_curriculum_gt_epochs": int(STAGE2_CURRICULUM_GT_EPOCHS),
        "stage2_grad_accumulation_steps": int(stage2_accum_steps),
        "stage2_reading_order_policy": STAGE2_READING_ORDER_POLICY,
        "stage2_use_attn_centroid_boxes": bool(STAGE2_USE_ATTN_CENTROID_BOXES),
        "stage2_arch_guardrail_strict": bool(STAGE2_ARCH_GUARDRAIL_STRICT),
        "stage2_arch_guardrail_issues": guardrail_info["issues"],
        "stage2_arch_guardrail_warning": guardrail_info["warning"],
        "stage2_val_max_decode_len": int(STAGE2_VAL_MAX_DECODE_LEN),
        "stage2_eos_loss_weight": float(STAGE2_EOS_LOSS_WEIGHT),
        "use_roi_attention": bool(USE_ROI_ATTENTION),
        "stage2_aux_ctc_weight": float(STAGE2_AUX_CTC_WEIGHT),
        "stage2_ar_refinement_enable": bool(STAGE2_AR_REFINEMENT_ENABLE),
        "stage2_ar_refinement_strength": float(STAGE2_AR_REFINEMENT_STRENGTH),
        "stage2_use_roi_positional_encoding": bool(STAGE2_USE_ROI_POSITIONAL_ENCODING),
        "stage2_use_roi_aux_classifier": bool(STAGE2_USE_ROI_AUX_CLASSIFIER),
        "stage2_roi_aux_gt_only": bool(STAGE2_ROI_AUX_GT_ONLY),
        "stage2_roi_aux_loss_weight": float(STAGE2_ROI_AUX_LOSS_WEIGHT),
        "stage2_ctc_time_expand_factor": int(STAGE2_CTC_TIME_EXPAND_FACTOR),
        "roi_box_loss_weight": float(ROI_BOX_LOSS_WEIGHT),
        "use_ctc_warmup": bool(use_ctc_warmup),
        "detector_checkpoint": str(detector_ckpt_path),
        "checkpoint_dir": str(checkpoint_dir),
    }
    run_config_path = checkpoint_dir / "stage2_train_run_config.json"
    with open(run_config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)
    print(f"Saved Stage2 run config: {run_config_path}")
    
    # CTC Warmup (optional)
    if use_ctc_warmup:
        print(f"Stage 2: CTC Warmup for 2 epochs...")
        ctc_loss_fn = nn.CTCLoss(blank=ctc_blank_id, zero_infinity=True)
        use_gt_boxes_warmup = True if STAGE2_CURRICULUM_ENABLE else STAGE2_USE_GT_BOXES
        
        for epoch in range(2):
            encoder.train()
            roi_sequence_encoder.train()
            context_encoder.train()
            decoder.train()
            ctc_head.train()
            roi_aux_head.train()
            pbar = tqdm(train_dataloader, desc=f"CTC Warmup {epoch+1}/2")
            optimizer.zero_grad(set_to_none=True)
            grad_counter = 0
            for batch in pbar:
                prepared = _filter_stage2_batch(batch, DEVICE)
                if prepared is None:
                    continue

                images = prepared["images"]
                text_ids = prepared["text_ids"]
                boxes_batch = prepared["boxes_batch"]
                labels_batch = prepared["labels_batch"]
                orientations = prepared["orientations"]

                boxes_for_encoder, labels_sorted_batch, text_ids_sorted = _build_boxes_for_encoder(
                    images=images,
                    boxes_batch=boxes_batch,
                    labels_batch=labels_batch,
                    orientations=orientations,
                    use_gt_boxes=use_gt_boxes_warmup,
                    unet=unet,
                    detector=detector,
                    reading_order_policy=STAGE2_READING_ORDER_POLICY,
                    pad_id=vocab.pad_id,
                    sos_id=vocab.sos_id,
                    eos_id=vocab.eos_id,
                )

                if use_gt_boxes_warmup and text_ids_sorted is not None:
                    seq_text_ids = text_ids_sorted
                else:
                    seq_text_ids = text_ids

                with amp.autocast(enabled=amp_enabled):
                    feats_2d = encoder(images, return_2d=True)
                    roi_seq, roi_mask = roi_sequence_encoder(feats_2d, boxes_for_encoder, image_size=IMAGE_SIZE)
                    enc_seq, enc_mask = context_encoder(roi_seq, roi_mask)
                    enc_seq_ctc, enc_mask_ctc = _expand_ctc_timesteps(
                        enc_seq, enc_mask, STAGE2_CTC_TIME_EXPAND_FACTOR
                    )

                    logits = ctc_head(enc_seq_ctc)
                    log_probs = logits.float().log_softmax(dim=-1).permute(1, 0, 2).contiguous()
                    input_lengths = enc_mask_ctc.sum(dim=1).clamp(min=1).to(dtype=torch.long)

                    targets_concat, target_lengths, keep_indices, _ = _prepare_ctc_targets(
                        text_ids=seq_text_ids,
                        pad_id=pad_id,
                        sos_id=sos_id,
                        eos_id=eos_id,
                        input_lengths=input_lengths,
                    )

                    if targets_concat is None:
                        continue

                    loss_raw = ctc_loss_fn(
                        log_probs[:, keep_indices, :],
                        targets_concat,
                        input_lengths[keep_indices],
                        target_lengths,
                    )
                    loss = loss_raw / stage2_accum_steps
                    if scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                grad_counter += 1

                if grad_counter % stage2_accum_steps == 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)

                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                pbar.set_postfix({"ctc_loss": float(loss_raw.item())})

            if grad_counter % stage2_accum_steps != 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
    
    # -------------------------
    # Main Stage 2 training
    # -------------------------
    print(f"Stage 2: Training ROI-based sequence model for {num_epochs} epochs...")
    print(f"Mode: {'Oracle (GT boxes)' if STAGE2_USE_GT_BOXES else 'Detector-guided (predicted boxes)'}")
    if STAGE2_CURRICULUM_ENABLE:
        print(f"Curriculum: GT boxes for first {int(STAGE2_CURRICULUM_GT_EPOCHS)} epochs, then predicted boxes")
    print(f"Loss strategy: {'CTC-PRIMARY (AR auxiliary)' if STAGE2_USE_CTC_PRIMARY else 'AR-PRIMARY (CTC auxiliary)'}")
    if STAGE2_USE_CTC_PRIMARY:
        print(f"  CTC weight (primary): {STAGE2_AUX_CTC_WEIGHT:.3f}")
        print(f"  AR weight (auxiliary): {STAGE2_AR_LOSS_WEIGHT:.3f}")
    else:
        print(f"  AR weight (primary): 1.0")
        print(f"  CTC weight (auxiliary): {STAGE2_AUX_CTC_WEIGHT:.3f}")
    print(f"ROI attention box loss: {'ON' if USE_ROI_ATTENTION else 'OFF'}")
    print(f"ROI geometric positional encoding: {'ON' if STAGE2_USE_ROI_POSITIONAL_ENCODING else 'OFF'}")
    print(f"ROI aux classifier: {'ON' if STAGE2_USE_ROI_AUX_CLASSIFIER else 'OFF'} (weight={STAGE2_ROI_AUX_LOSS_WEIGHT:.3f})")
    print(f"AR refinement from CTC posterior: {'ON' if STAGE2_AR_REFINEMENT_ENABLE else 'OFF'} (strength={STAGE2_AR_REFINEMENT_STRENGTH:.3f})")
    print(f"Grad accumulation (Stage2): {stage2_accum_steps}")

    ctc_loss_fn = nn.CTCLoss(blank=ctc_blank_id, zero_infinity=True)
    ctc_dominance_streak = 0

    best_val_cer = None

    curriculum_gt_epochs = max(0, int(STAGE2_CURRICULUM_GT_EPOCHS))

    def _use_gt_boxes_for_epoch(epoch_idx):
        if STAGE2_CURRICULUM_ENABLE:
            return epoch_idx < curriculum_gt_epochs
        return STAGE2_USE_GT_BOXES

    for epoch in range(num_epochs):
        encoder.train()
        roi_sequence_encoder.train()
        context_encoder.train()
        decoder.train()
        unet.eval()
        detector.eval()
        use_gt_boxes_epoch = _use_gt_boxes_for_epoch(epoch)
        epoch_box_source = "gt" if use_gt_boxes_epoch else "pred"

        tf_ratio = scheduled_teacher_forcing(epoch,
            num_epochs,
            start=0.9,
            end=0.0,
            schedule="linear")
        ar_tf_ratio = max(float(tf_ratio), float(STAGE2_AR_TF_MIN))

        print(
            f"Epoch {epoch+1}: box_source={epoch_box_source} | "
            f"reading_order_policy={STAGE2_READING_ORDER_POLICY} | "
            f"tf_ratio={tf_ratio:.3f} | "
            f"ar_tf_ratio={ar_tf_ratio:.3f}"
        )
        
        total_loss = 0.0
        total_seq_loss = 0.0
        total_ctc_loss = 0.0
        total_box_loss = 0.0
        total_roi_aux_loss = 0.0
        total_roi_aux_correct = 0.0
        total_roi_aux_count = 0.0
        total_roi_aux_coverage = 0.0
        total_ar_entropy = 0.0
        ctc_total_rows = 0
        ctc_kept_rows = 0
        ctc_skipped_too_long = 0
        ctc_skipped_repeats_too_short = 0
        ctc_input_len_sum = 0.0
        ctc_target_len_sum = 0.0
        ctc_required_len_sum = 0.0
        ctc_input_minus_required_sum = 0.0
        n_batches = 0
        optimizer.zero_grad(set_to_none=True)
        grad_counter = 0
        
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", mininterval=60.0)
        for batch in pbar:
            prepared = _filter_stage2_batch(batch, DEVICE)
            if prepared is None:
                continue

            images = prepared["images"]
            text_ids = prepared["text_ids"]
            boxes_batch = prepared["boxes_batch"]
            labels_batch = prepared["labels_batch"]
            orientations = prepared["orientations"]

            gt_boxes_for_loss = []
            if boxes_batch is not None:
                for b in boxes_batch:
                    if b is None:
                        gt_boxes_for_loss.append(torch.empty((0, 4), dtype=torch.float32, device=DEVICE))
                    else:
                        gt_boxes_for_loss.append(b.to(DEVICE, dtype=torch.float32))
            else:
                gt_boxes_for_loss = [torch.empty((0, 4), dtype=torch.float32, device=DEVICE) for _ in range(images.size(0))]

            boxes_for_encoder, labels_sorted_batch, text_ids_sorted = _build_boxes_for_encoder(
                images=images,
                boxes_batch=boxes_batch,
                labels_batch=labels_batch,
                orientations=orientations,
                use_gt_boxes=use_gt_boxes_epoch,
                unet=unet,
                detector=detector,
                reading_order_policy=STAGE2_READING_ORDER_POLICY,
                pad_id=vocab.pad_id,
                sos_id=vocab.sos_id,
                eos_id=vocab.eos_id,
            )

            if use_gt_boxes_epoch and text_ids_sorted is not None:
                seq_text_ids = text_ids_sorted
            else:
                seq_text_ids = text_ids

            input_seq = seq_text_ids[:, :-1]
            targets = seq_text_ids[:, 1:]

            with amp.autocast(enabled=amp_enabled):
                feats_2d = encoder(images, return_2d=True)
                roi_seq, roi_mask = roi_sequence_encoder(feats_2d, boxes_for_encoder, image_size=IMAGE_SIZE)
                enc_outputs, enc_mask = context_encoder(roi_seq, roi_mask)

                enc_outputs_ctc, enc_mask_ctc = _expand_ctc_timesteps(
                    enc_outputs, enc_mask, STAGE2_CTC_TIME_EXPAND_FACTOR
                )
                ctc_logits_for_refine = ctc_head(enc_outputs_ctc)  # (B, T_ctc, V+1)

                ar_logit_bias = None
                if STAGE2_AR_REFINEMENT_ENABLE:
                    ar_logit_bias = _build_ar_refinement_bias_from_ctc(
                        ctc_logits=ctc_logits_for_refine,
                        dec_len=int(input_seq.size(1)),
                        blank_id=ctc_blank_id,
                        strength=STAGE2_AR_REFINEMENT_STRENGTH,
                    )
                
                # Decoder forward pass (predicts sequence with context)
                decoder_output = decoder(
                    input_seq=input_seq,
                    enc_outputs=enc_outputs,
                    enc_mask=enc_mask,
                    teacher_forcing_ratio=ar_tf_ratio,
                    targets=targets,
                    eos_id=eos_id,
                    image_size=(images.shape[2], images.shape[3]),
                    token_logit_bias=ar_logit_bias,
                )
                
                # Handle both 3-tuple and 4-tuple output
                predicted_boxes = None
                if len(decoder_output) == 4:
                    logits, _, _, predicted_boxes = decoder_output
                else:
                    logits, _, _ = decoder_output
                
                B, T_dec, V = logits.shape
                
                # Compute box loss (independent of AR/CTC mode)
                if USE_ROI_ATTENTION and predicted_boxes is not None and len(gt_boxes_for_loss) == B:
                    loss_box = compute_roi_box_loss(
                        predicted_boxes,
                        gt_boxes_for_loss,
                        reduction='mean',
                        iou_weight=0.0,
                        use_x_only=True,
                        coord_scale=float(images.shape[3]),
                    )
                else:
                    loss_box = torch.tensor(0.0, device=DEVICE)

                roi_aux_enabled = bool(STAGE2_USE_ROI_AUX_CLASSIFIER) and (use_gt_boxes_epoch or not STAGE2_ROI_AUX_GT_ONLY)
                if roi_aux_enabled:
                    roi_aux_logits = roi_aux_head(roi_seq)
                    roi_aux_flat_logits, roi_aux_flat_targets = _prepare_roi_aux_supervision_from_sorted_labels(
                            roi_logits=roi_aux_logits,
                            labels_sorted_batch=labels_sorted_batch,
                            roi_mask=roi_mask,
                        )
                    if roi_aux_flat_logits is None:
                        loss_roi_aux = torch.tensor(0.0, device=DEVICE)
                    else:
                        loss_roi_aux = roi_aux_ce_loss(roi_aux_flat_logits, roi_aux_flat_targets)
                        with torch.no_grad():
                            if roi_aux_flat_targets is None:
                                roi_aux_target_count = 0
                            else:
                                roi_aux_target_count = int(roi_aux_flat_targets.numel())
                            roi_aux_preds = roi_aux_flat_logits.argmax(dim=-1)
                            if roi_aux_target_count > 0:
                                total_roi_aux_correct += float((roi_aux_preds == roi_aux_flat_targets).sum().item())
                                total_roi_aux_count += float(roi_aux_target_count)

                            # Coverage ratio over available ROI tokens in batch.
                            roi_tokens_in_batch = float(roi_mask.sum().item())
                            if roi_tokens_in_batch > 0.0 and roi_aux_target_count > 0:
                                total_roi_aux_coverage += float(roi_aux_target_count) / roi_tokens_in_batch
                else:
                    loss_roi_aux = torch.tensor(0.0, device=DEVICE)
                
                # Compute losses based on mode (CTC-primary vs AR-primary)
                if STAGE2_USE_CTC_PRIMARY:
                    # ============ CTC-PRIMARY MODE ============
                    # Compute CTC loss (always, as primary)
                    ctc_logits = ctc_logits_for_refine
                    ctc_log_probs = ctc_logits.float().log_softmax(dim=-1).permute(1, 0, 2).contiguous()
                    ctc_input_lengths = enc_mask_ctc.sum(dim=1).clamp(min=1).to(dtype=torch.long)
                    targets_concat, target_lengths, keep_indices, ctc_diag = _prepare_ctc_targets(
                        text_ids=seq_text_ids,
                        pad_id=pad_id,
                        sos_id=sos_id,
                        eos_id=eos_id,
                        input_lengths=ctc_input_lengths,
                    )
                    ctc_total_rows += int(ctc_diag["total_rows"])
                    ctc_kept_rows += int(ctc_diag["kept_rows"])
                    ctc_skipped_too_long += int(ctc_diag["skipped_too_long"])
                    ctc_skipped_repeats_too_short += int(ctc_diag["skipped_repeats_too_short"])
                    ctc_input_len_sum += float(ctc_diag["mean_input_len_kept"]) * int(ctc_diag["kept_rows"])
                    ctc_target_len_sum += float(ctc_diag["mean_target_len_kept"]) * int(ctc_diag["kept_rows"])
                    ctc_required_len_sum += float(ctc_diag["mean_min_required_len_kept"]) * int(ctc_diag["kept_rows"])
                    ctc_input_minus_required_sum += float(ctc_diag["mean_input_minus_required_kept"]) * int(ctc_diag["kept_rows"])
                    if targets_concat is None:
                        loss_ctc = torch.tensor(0.0, device=DEVICE)
                    else:
                        loss_ctc = ctc_loss_fn(
                            ctc_log_probs[:, keep_indices, :],
                            targets_concat,
                            ctc_input_lengths[keep_indices],
                            target_lengths,
                        )
                    
                    # AR loss is optional/auxiliary in CTC-primary mode
                    if STAGE2_AR_LOSS_WEIGHT > 0.0:
                        loss_seq = ce_loss(logits.reshape(-1, V), targets.reshape(-1))
                    else:
                        loss_seq = torch.tensor(0.0, device=DEVICE)

                    probs = torch.softmax(logits.float(), dim=-1)
                    token_entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)
                    total_ar_entropy += float(token_entropy.mean().item())
                    
                    # CTC-primary loss combination
                    loss_raw = (
                        STAGE2_AUX_CTC_WEIGHT * loss_ctc
                        + STAGE2_AR_LOSS_WEIGHT * loss_seq
                        + STAGE2_ROI_AUX_LOSS_WEIGHT * loss_roi_aux
                        + ROI_BOX_LOSS_WEIGHT * loss_box
                    )
                    
                else:
                    # ============ AR-PRIMARY MODE (Original) ============
                    # Sequence loss (AR decoder primary)
                    loss_seq = ce_loss(logits.reshape(-1, V), targets.reshape(-1))

                    probs = torch.softmax(logits.float(), dim=-1)
                    token_entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=-1)
                    total_ar_entropy += float(token_entropy.mean().item())

                    # CTC loss is optional/auxiliary in AR-primary mode
                    if STAGE2_AUX_CTC_WEIGHT > 0.0:
                        ctc_logits = ctc_logits_for_refine
                        ctc_log_probs = ctc_logits.float().log_softmax(dim=-1).permute(1, 0, 2).contiguous()
                        ctc_input_lengths = enc_mask_ctc.sum(dim=1).clamp(min=1).to(dtype=torch.long)
                        targets_concat, target_lengths, keep_indices, ctc_diag = _prepare_ctc_targets(
                            text_ids=seq_text_ids,
                            pad_id=pad_id,
                            sos_id=sos_id,
                            eos_id=eos_id,
                            input_lengths=ctc_input_lengths,
                        )
                        ctc_total_rows += int(ctc_diag["total_rows"])
                        ctc_kept_rows += int(ctc_diag["kept_rows"])
                        ctc_skipped_too_long += int(ctc_diag["skipped_too_long"])
                        ctc_skipped_repeats_too_short += int(ctc_diag["skipped_repeats_too_short"])
                        ctc_input_len_sum += float(ctc_diag["mean_input_len_kept"]) * int(ctc_diag["kept_rows"])
                        ctc_target_len_sum += float(ctc_diag["mean_target_len_kept"]) * int(ctc_diag["kept_rows"])
                        ctc_required_len_sum += float(ctc_diag["mean_min_required_len_kept"]) * int(ctc_diag["kept_rows"])
                        ctc_input_minus_required_sum += float(ctc_diag["mean_input_minus_required_kept"]) * int(ctc_diag["kept_rows"])
                        if targets_concat is None:
                            loss_ctc = torch.tensor(0.0, device=DEVICE)
                        else:
                            loss_ctc = ctc_loss_fn(
                                ctc_log_probs[:, keep_indices, :],
                                targets_concat,
                                ctc_input_lengths[keep_indices],
                                target_lengths,
                            )
                    else:
                        loss_ctc = torch.tensor(0.0, device=DEVICE)
                    
                    # AR-primary loss combination
                    loss_raw = (
                        loss_seq
                        + STAGE2_AUX_CTC_WEIGHT * loss_ctc
                        + STAGE2_ROI_AUX_LOSS_WEIGHT * loss_roi_aux
                        + ROI_BOX_LOSS_WEIGHT * loss_box
                    )
                
                loss = loss_raw / stage2_accum_steps
            
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            grad_counter += 1
            if grad_counter % stage2_accum_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            total_seq_loss += loss_seq.item()
            total_ctc_loss += loss_ctc.item()
            total_box_loss += loss_box.item()
            total_roi_aux_loss += loss_roi_aux.item()
            total_loss += loss_raw.item()
            n_batches += 1
            ctc_weight_for_log = STAGE2_AUX_CTC_WEIGHT if STAGE2_USE_CTC_PRIMARY else STAGE2_AUX_CTC_WEIGHT
            ar_weight_for_log = STAGE2_AR_LOSS_WEIGHT if STAGE2_USE_CTC_PRIMARY else 1.0
            pbar.set_postfix({
                "loss": total_loss / n_batches,
                "seq": total_seq_loss / n_batches,
                "ctc": total_ctc_loss / n_batches,
                "box": total_box_loss / n_batches,
                "roi_aux": total_roi_aux_loss / n_batches,
                "roi_aux_acc": (total_roi_aux_correct / total_roi_aux_count) if total_roi_aux_count > 0 else 0.0,
                "wctc": ctc_weight_for_log * (total_ctc_loss / n_batches),
                "war": ar_weight_for_log * (total_seq_loss / n_batches),
                "wbox": ROI_BOX_LOSS_WEIGHT * (total_box_loss / n_batches),
                "tf": tf_ratio,
                "ar_tf": ar_tf_ratio,
            })

        if grad_counter % stage2_accum_steps != 0:
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
        
        # -------------------------
        # validation
        # -------------------------
        val_metrics = validate_sequence_stage(
            encoder=encoder,
            roi_sequence_encoder=roi_sequence_encoder,
            context_encoder=context_encoder,
            decoder=decoder,
            ctc_head=ctc_head,
            dataloader=val_dataloader,
            vocab=vocab,
            ce_loss=ce_loss,
            use_gt_boxes=use_gt_boxes_epoch,
            unet=unet,
            detector=detector,
            reading_order_policy=STAGE2_READING_ORDER_POLICY,
            max_decode_len=STAGE2_VAL_MAX_DECODE_LEN,
            use_ctc_primary=STAGE2_USE_CTC_PRIMARY,
            ctc_decode_temperature=STAGE2_CTC_DECODE_TEMPERATURE,
        )

        train_loss_mean = total_loss / max(1, n_batches)
        train_seq_mean = total_seq_loss / max(1, n_batches)
        train_ctc_mean = total_ctc_loss / max(1, n_batches)
        train_box_mean = total_box_loss / max(1, n_batches)
        train_roi_aux_mean = total_roi_aux_loss / max(1, n_batches)
        train_roi_aux_acc = total_roi_aux_correct / max(1.0, total_roi_aux_count)
        train_roi_aux_coverage_mean = total_roi_aux_coverage / max(1, n_batches)
        train_wseq_mean = STAGE2_AR_LOSS_WEIGHT * train_seq_mean if STAGE2_USE_CTC_PRIMARY else train_seq_mean
        train_wctc_mean = STAGE2_AUX_CTC_WEIGHT * train_ctc_mean
        train_wroi_aux_mean = STAGE2_ROI_AUX_LOSS_WEIGHT * train_roi_aux_mean
        train_wbox_mean = ROI_BOX_LOSS_WEIGHT * train_box_mean
        train_ar_entropy_mean = total_ar_entropy / max(1, n_batches)
        ctc_kept_denom = max(1, ctc_kept_rows)
        ctc_total_denom = max(1, ctc_total_rows)
        ctc_skipped_frac = ctc_skipped_too_long / ctc_total_denom
        ctc_skipped_repeats_frac = ctc_skipped_repeats_too_short / ctc_total_denom
        ctc_input_len_mean = ctc_input_len_sum / ctc_kept_denom
        ctc_target_len_mean = ctc_target_len_sum / ctc_kept_denom
        ctc_required_len_mean = ctc_required_len_sum / ctc_kept_denom
        ctc_input_minus_required_mean = ctc_input_minus_required_sum / ctc_kept_denom

        if train_wctc_mean > 2.0 * train_wseq_mean:
            ctc_dominance_streak += 1
        else:
            ctc_dominance_streak = 0
        if ctc_dominance_streak >= 2:
            print(
                f"[WARN] Weighted CTC term dominates CE for {ctc_dominance_streak} epochs: "
                f"wctc={train_wctc_mean:.4f} vs wseq={train_wseq_mean:.4f}"
            )

        print(
            f"\nEpoch {epoch+1}/{num_epochs} | "
            f"train_loss={train_loss_mean:.4f} | "
            f"train_seq={train_seq_mean:.4f} | "
            f"train_ctc={train_ctc_mean:.4f} | "
            f"train_box={train_box_mean:.4f} | "
            f"train_roi_aux={train_roi_aux_mean:.4f} | "
            f"train_roi_aux_acc={train_roi_aux_acc:.4f} | "
            f"train_roi_aux_cov={train_roi_aux_coverage_mean:.3f} | "
            f"wseq={train_wseq_mean:.4f} | "
            f"wctc={train_wctc_mean:.4f} | "
            f"wroi_aux={train_wroi_aux_mean:.4f} | "
            f"wbox={train_wbox_mean:.4f} | "
            f"ar_entropy={train_ar_entropy_mean:.4f} | "
            f"ctc_skip_too_long={ctc_skipped_frac:.4f} | "
            f"ctc_skip_repeats_short={ctc_skipped_repeats_frac:.4f} | "
            f"ctc_in_len={ctc_input_len_mean:.2f} | "
            f"ctc_tgt_len={ctc_target_len_mean:.2f} | "
            f"ctc_min_req_len={ctc_required_len_mean:.2f} | "
            f"ctc_slack={ctc_input_minus_required_mean:.2f} | "
            f"val_loss={val_metrics['val_loss']:.4f} | "
            f"val_cer={val_metrics['val_cer']:.4f} | "
            f"val_exact={val_metrics['val_exact']:.4f} | "
            f"val_pred_len={val_metrics['val_mean_pred_len']:.2f} | "
            f"val_gt_len={val_metrics['val_mean_gt_len']:.2f} | "
            f"val_len_ratio={val_metrics['val_len_ratio']:.3f} | "
            f"val_max_decode_frac={val_metrics['val_max_decode_frac']:.3f} | "
            f"val_top1_token_share={val_metrics['val_top1_token_share']:.3f}"
        )

        print(
            "Validation length histogram: "
            f"short={val_metrics['val_pred_len_hist'].get('short', 0)} | "
            f"medium={val_metrics['val_pred_len_hist'].get('medium', 0)} | "
            f"long={val_metrics['val_pred_len_hist'].get('long', 0)}"
        )
        print(
            "Validation CER by GT length: "
            f"short={val_metrics['val_cer_by_len_bucket'].get('short', 0.0):.4f} | "
            f"medium={val_metrics['val_cer_by_len_bucket'].get('medium', 0.0):.4f} | "
            f"long={val_metrics['val_cer_by_len_bucket'].get('long', 0.0):.4f}"
        )

        debug_samples = val_metrics.get("debug_samples", [])
        if debug_samples:
            print("Validation random samples (GT | Pred | CER | pred_len/gt_len | max_decode_hit):")
            for i, sample in enumerate(debug_samples, start=1):
                print(
                    f"  [{i}] GT={sample['gt']} | "
                    f"Pred={sample['pred']} | "
                    f"CER={sample['cer']:.3f} | "
                    f"len={sample['pred_len']}/{sample['gt_len']} | "
                    f"max_decode_hit={int(sample['hit_max_decode'])}"
                )

        is_best = (best_val_cer is None) or (val_metrics["val_cer"] < best_val_cer)
        if is_best:
            best_val_cer = val_metrics["val_cer"]
        
        checkpoint_payload = {
            "epoch": epoch + 1,
            "encoder_state_dict": encoder.state_dict(),
            "roi_sequence_encoder_state_dict": roi_sequence_encoder.state_dict(),
            "context_encoder_state_dict": context_encoder.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "ctc_head_state_dict": ctc_head.state_dict(),
            "roi_aux_head_state_dict": roi_aux_head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss_mean,
            "train_seq_loss": train_seq_mean,
            "train_ctc_loss": train_ctc_mean,
            "train_box_loss": train_box_mean,
            "train_roi_aux_loss": train_roi_aux_mean,
            "train_roi_aux_acc": train_roi_aux_acc,
            "train_roi_aux_coverage": train_roi_aux_coverage_mean,
            "train_weighted_seq_loss": train_wseq_mean,
            "train_weighted_ctc_loss": train_wctc_mean,
            "train_weighted_roi_aux_loss": train_wroi_aux_mean,
            "train_weighted_box_loss": train_wbox_mean,
            "train_ar_entropy": train_ar_entropy_mean,
            "ctc_rows_total": ctc_total_rows,
            "ctc_rows_kept": ctc_kept_rows,
            "ctc_rows_skipped_too_long": ctc_skipped_too_long,
            "ctc_rows_skipped_repeats_too_short": ctc_skipped_repeats_too_short,
            "ctc_skipped_too_long_fraction": ctc_skipped_frac,
            "ctc_skipped_repeats_too_short_fraction": ctc_skipped_repeats_frac,
            "ctc_mean_input_len_kept": ctc_input_len_mean,
            "ctc_mean_target_len_kept": ctc_target_len_mean,
            "ctc_mean_min_required_len_kept": ctc_required_len_mean,
            "ctc_mean_input_minus_required_kept": ctc_input_minus_required_mean,
            "ctc_dominance_streak": ctc_dominance_streak,
            "val_loss": val_metrics["val_loss"],
            "val_cer": val_metrics["val_cer"],
            "val_exact": val_metrics["val_exact"],
            "val_mean_pred_len": val_metrics["val_mean_pred_len"],
            "val_mean_gt_len": val_metrics["val_mean_gt_len"],
            "val_len_ratio": val_metrics["val_len_ratio"],
            "val_max_decode_frac": val_metrics["val_max_decode_frac"],
            "val_top1_token_share": val_metrics["val_top1_token_share"],
            "val_pred_len_hist": val_metrics["val_pred_len_hist"],
            "val_cer_by_len_bucket": val_metrics["val_cer_by_len_bucket"],
            "use_roi_attention": USE_ROI_ATTENTION,
            "stage2_aux_ctc_weight": STAGE2_AUX_CTC_WEIGHT,
            "stage2_ar_loss_weight": STAGE2_AR_LOSS_WEIGHT,
            "stage2_ar_refinement_enable": bool(STAGE2_AR_REFINEMENT_ENABLE),
            "stage2_ar_refinement_strength": float(STAGE2_AR_REFINEMENT_STRENGTH),
            "stage2_ar_tf_min": STAGE2_AR_TF_MIN,
            "stage2_ctc_decode_temperature": STAGE2_CTC_DECODE_TEMPERATURE,
            "stage2_ctc_time_expand_factor": int(STAGE2_CTC_TIME_EXPAND_FACTOR),
            "stage2_use_roi_positional_encoding": bool(STAGE2_USE_ROI_POSITIONAL_ENCODING),
            "stage2_use_roi_aux_classifier": bool(STAGE2_USE_ROI_AUX_CLASSIFIER),
            "stage2_roi_aux_gt_only": bool(STAGE2_ROI_AUX_GT_ONLY),
            "stage2_roi_aux_loss_weight": float(STAGE2_ROI_AUX_LOSS_WEIGHT),
            "roi_box_loss_weight": ROI_BOX_LOSS_WEIGHT,
            "stage2_use_gt_boxes": STAGE2_USE_GT_BOXES,
            "stage2_use_gt_boxes_epoch": bool(use_gt_boxes_epoch),
            "stage2_epoch_box_source": epoch_box_source,
            "stage2_curriculum_enable": bool(STAGE2_CURRICULUM_ENABLE),
            "stage2_curriculum_gt_epochs": curriculum_gt_epochs,
            "stage2_grad_accumulation_steps": int(stage2_accum_steps),
            "stage2_reading_order_policy": STAGE2_READING_ORDER_POLICY,
            "stage2_use_attn_centroid_boxes": bool(STAGE2_USE_ATTN_CENTROID_BOXES),
            "stage2_arch_guardrail_strict": bool(STAGE2_ARCH_GUARDRAIL_STRICT),
            "stage2_arch_guardrail_issues": guardrail_info["issues"],
            "stage2_arch_guardrail_warning": guardrail_info["warning"],
            "roi_pool_size": ROI_POOL_SIZE,
            "roi_embed_dim": ROI_EMBED_DIM,
            "context_hidden_dim": CONTEXT_HIDDEN_DIM,
            "vocab": vocab.char2id,
            "is_best": is_best,
        }

        # Save checkpoint
        checkpoint_path = checkpoint_dir / f"sequence_epoch{epoch+1}.pt"
        torch.save(checkpoint_payload, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
        if is_best:
            torch.save(checkpoint_payload, checkpoint_dir / "sequence_best.pt")
            print(f"✅ saved best: {epoch + 1} (val_cer={val_metrics['val_cer']:.4f})")
        prune_to_keep_last_n(checkpoint_dir, keep=2, exclude="checkpoint_old.pt")
    
    return encoder, roi_sequence_encoder, context_encoder, decoder, ctc_head

def train(stage1 = True, stage2 = False, args=None):  
    print("="*60)
    print("STAGE 1: TRAINING DETECTOR (Spatial Localization)")
    print("="*60)
    if stage1 == True:
        unet, detector = train_detector_stage(num_epochs=NUM_EPOCHS, lr=None)
    
    if stage2 == True:
        print("="*60)
        print("STAGE 2: TRAINING SEQUENCE MODEL (Contextual Understanding)")
        print("="*60)
        # Use best checkpoint if it exists, otherwise use last epoch
        best_ckpt = CHECKPOINT_DIR / "stage1_detection" / "detector_best.pt"
        last_ckpt = CHECKPOINT_DIR / "stage1_detection" / f"detector_epoch{NUM_EPOCHS}.pt"
        detector_ckpt = best_ckpt if best_ckpt.exists() else last_ckpt
        
        encoder, roi_sequence_encoder, context_encoder, decoder, ctc_head = train_sequence_stage(
            detector_ckpt_path=detector_ckpt,
            num_epochs=NUM_EPOCHS,
            lr=None,
            use_ctc_warmup=STAGE2_USE_CTC_WARMUP,
        )
        
if __name__ == "__main__":
    train(stage1=False, stage2=True)  