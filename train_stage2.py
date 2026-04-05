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
    STAGE2_CONTEXT_HIDDEN_DIM,
    STAGE2_CONTEXT_NUM_LAYERS,
    STAGE2_DECODER_EMBED_DIM,
    STAGE2_DECODER_HIDDEN_DIM,
    STAGE2_DET_MIN_BOX_SIZE,
    STAGE2_DET_NMS_IOU,
    STAGE2_DET_SCORE_THRESH,
    STAGE2_DET_TOP_K,
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
    STAGE2_LAMBDA_BOX,
    STAGE2_LAMBDA_DELTA,
    STAGE2_LAMBDA_SCORE,
    STAGE2_LAMBDA_AUX,
    STAGE2_LAMBDA_DECODER,
    STAGE2_TF_START,
    STAGE2_TF_END,
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
    STAGE2_PHASE_A2_EPOCHS,
    STAGE2_PHASE_A2_LAMBDA_BOX,
    STAGE2_PHASE_A2_LAMBDA_DELTA,
    STAGE2_PHASE_A2_LAMBDA_SCORE,
    STAGE2_PHASE_A2_LAMBDA_AUX,
    STAGE2_PHASE_A2_LAMBDA_DECODER,
    STAGE2_PHASE_A2_TF_START,
    STAGE2_PHASE_A2_TF_END,
    STAGE2_PHASE_B_EPOCHS,
    STAGE2_PHASE_B_LAMBDA_BOX,
    STAGE2_PHASE_B_LAMBDA_DELTA,
    STAGE2_PHASE_B_LAMBDA_SCORE,
    STAGE2_PHASE_B_LAMBDA_AUX,
    STAGE2_PHASE_B_LAMBDA_DECODER,
    STAGE2_PHASE_B_TF_START,
    STAGE2_PHASE_B_TF_END,
    
)

from model.kuronet import UNet, DetectorHead
from model.kuronet.hybrid_recognizer import HybridKuroNetRecognizer

from utils import KuzushijiDataset
from utils.text_normalization import render_tokens
from utils.training_helpers import (
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
from utils.stage2_losses import compute_stage2_total_loss
from utils.stage2_losses import aux_classification_loss

def _decode_text_from_ids(ids, vocab):
    chars = vocab.decode([int(x) for x in ids], remove_special=True)
    return render_tokens(chars)

def _count_valid_rois_per_image(roi_mask: torch.Tensor) -> list[int]:
    return [int(v) for v in roi_mask.to(dtype=torch.long).sum(dim=1).tolist()]


def _count_mask_per_image(mask: torch.Tensor) -> list[int]:
    return [int(v) for v in mask.to(dtype=torch.long).sum(dim=1).tolist()]


def _load_compatible_state_dict(module: torch.nn.Module, state_dict: dict) -> None:
    """Load only tensors whose names and shapes match the current module."""
    current_state = module.state_dict()
    compatible_state = {}
    skipped_keys: list[str] = []

    for key, value in state_dict.items():
        if key in current_state and current_state[key].shape == value.shape:
            compatible_state[key] = value
        else:
            skipped_keys.append(key)

    missing_keys = [key for key in current_state.keys() if key not in compatible_state]
    module.load_state_dict(compatible_state, strict=False)

    print(
        f"Loaded resume checkpoint partially: {len(compatible_state)} tensors matched, "
        f"{len(skipped_keys)} skipped, {len(missing_keys)} left initialized."
    )
    if skipped_keys:
        preview = ", ".join(skipped_keys[:8])
        suffix = "..." if len(skipped_keys) > 8 else ""
        print(f"Skipped incompatible keys: {preview}{suffix}")


def _normalize_orientation_label(label: str) -> str:
    key = str(label).strip().lower()
    if key in {"horizontal", "h", "hor"}:
        return "horizontal"
    if key in {"vertical", "v", "ver"}:
        return "vertical"
    return "other"


def _valid_proposal_mask(boxes: torch.Tensor, roi_mask: torch.Tensor) -> torch.Tensor:
    return (
        roi_mask.bool()
        & boxes[..., 2].gt(boxes[..., 0])
        & boxes[..., 3].gt(boxes[..., 1])
    )

def get_phase_settings(phase: str) -> dict:
    phase = phase.upper()
    if phase == "A":
        return {
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
            "train_context_encoder": False,   # optional first variant
            "log_free_decoder": False,
            "use_context_aux_for_loss": False,
        }
    elif phase == "A2":
        return {
            "name": "A2",
            "epochs": int(STAGE2_PHASE_A2_EPOCHS),
            "lambda_box": float(STAGE2_PHASE_A2_LAMBDA_BOX),
            "lambda_delta": float(STAGE2_PHASE_A2_LAMBDA_DELTA),
            "lambda_score": float(STAGE2_PHASE_A2_LAMBDA_SCORE),
            "lambda_aux": float(STAGE2_PHASE_A2_LAMBDA_AUX),
            "lambda_decoder": float(STAGE2_PHASE_A2_LAMBDA_DECODER),
            "tf_start": float(STAGE2_PHASE_A2_TF_START),
            "tf_end": float(STAGE2_PHASE_A2_TF_END),
            "tf_schedule": STAGE2_TF_SCHEDULE,
            "train_context_encoder": True,
            "log_free_decoder": False,
            "use_context_aux_for_loss": True,
        }
    elif phase == "B":
        return {
            "name": "B",
            "epochs": int(STAGE2_PHASE_B_EPOCHS),
            "lambda_box": float(STAGE2_PHASE_B_LAMBDA_BOX),
            "lambda_delta": float(STAGE2_PHASE_B_LAMBDA_DELTA),
            "lambda_score": float(STAGE2_PHASE_B_LAMBDA_SCORE),
            "lambda_aux": float(STAGE2_PHASE_B_LAMBDA_AUX),
            "lambda_decoder": float(STAGE2_PHASE_B_LAMBDA_DECODER),
            "tf_start": float(STAGE2_PHASE_B_TF_START),
            "tf_end": float(STAGE2_PHASE_B_TF_END),
            "tf_schedule": STAGE2_TF_SCHEDULE,
            "train_context_encoder": True,
            "log_free_decoder": True,
            "use_context_aux_for_loss": False,
        }
    else:
        raise ValueError(f"Unsupported phase '{phase}'. Use 'A', 'A2' or 'B'.")
    
def set_trainable_modules_for_phase(model: HybridKuroNetRecognizer, phase: str) -> None:
    phase = phase.upper()

    # default: everything off except backbone/detector policy handled elsewhere
    for name, p in model.named_parameters():
        if name.startswith("backbone.") or name.startswith("detector."):
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
    if phase in {"A", "A2"}:
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


def reorder_by_sort_indices(values: torch.Tensor, sort_indices: torch.Tensor) -> torch.Tensor:
    """
    Reorder tensors by per-sample sort indices.

    values: (B, T) or (B, T, C)
    sort_indices: (B, T)
    """
    if values.dim() == 2:
        return values.gather(dim=1, index=sort_indices)
    if values.dim() == 3:
        expanded = sort_indices.unsqueeze(-1).expand(-1, -1, values.size(-1))
        return values.gather(dim=1, index=expanded)
    raise ValueError(f"Unsupported tensor rank for reorder_by_sort_indices: {values.dim()}")


def _aux_top1_accuracy(aux_logits: torch.Tensor, labels: torch.Tensor, pos_mask: torch.Tensor) -> float:
    valid = pos_mask & labels.ne(-1)
    if not valid.any():
        return 0.0
    pred = aux_logits.argmax(dim=-1)
    return float((pred[valid] == labels[valid]).float().mean().item())


def _debug_one_sample_alignment(
    outputs: dict,
    refine_targets: dict,
    vocab: VocabManager,
    sample_idx: int = 0,
    limit: int = 20,
) -> None:
    sort_idx = outputs["sort_indices"][sample_idx].detach().cpu()
    labels_unsorted = refine_targets["matched_gt_labels"][sample_idx].detach().cpu()
    pos_unsorted = refine_targets["refine_pos_mask"][sample_idx].detach().cpu()

    labels_sorted = labels_unsorted[sort_idx]
    pos_sorted = pos_unsorted[sort_idx]

    pred_unsorted = outputs["aux_logits"][sample_idx].argmax(dim=-1).detach().cpu()
    pred_ordered = None
    if outputs.get("aux_logits_ordered", None) is not None:
        pred_ordered = outputs["aux_logits_ordered"][sample_idx].argmax(dim=-1).detach().cpu()

    def decode_ids(ids: torch.Tensor):
        out = []
        for x in ids.tolist():
            x = int(x)
            if x < 0:
                out.append("<IGN>")
            else:
                txt = _ids_to_text([x], vocab)
                out.append(txt if txt else f"<{x}>")
        return out

    tqdm.write(f"[AUX ALIGN DEBUG] sample={sample_idx}")
    tqdm.write(f"sort_idx[:{limit}]      = {sort_idx[:limit].tolist()}")
    tqdm.write(f"labels_unsorted[:{limit}] = {labels_unsorted[:limit].tolist()}")
    tqdm.write(f"labels_sorted[:{limit}]   = {labels_sorted[:limit].tolist()}")
    tqdm.write(f"pos_unsorted[:{limit}]    = {pos_unsorted[:limit].tolist()}")
    tqdm.write(f"pos_sorted[:{limit}]      = {pos_sorted[:limit].tolist()}")
    tqdm.write(f"pred_aux_uns[:{limit}]    = {pred_unsorted[:limit].tolist()}")
    if pred_ordered is not None:
        tqdm.write(f"pred_aux_ord[:{limit}]    = {pred_ordered[:limit].tolist()}")
    tqdm.write(f"labels_unsorted txt   = {decode_ids(labels_unsorted[:limit])}")
    tqdm.write(f"labels_sorted txt     = {decode_ids(labels_sorted[:limit])}")
    tqdm.write(f"pred_aux_uns txt      = {decode_ids(pred_unsorted[:limit])}")
    if pred_ordered is not None:
        tqdm.write(f"pred_aux_ord txt      = {decode_ids(pred_ordered[:limit])}")


def _debug_aux_alignment(
    outputs: dict,
    refine_targets: dict,
    vocab: VocabManager,
    limit: int,
) -> None:
    aux_unsorted = outputs.get("aux_logits", None)
    sort_indices = outputs.get("sort_indices", None)
    if aux_unsorted is None or sort_indices is None:
        tqdm.write("[AUX ALIGN DEBUG] skipped: missing aux_logits or sort_indices")
        return

    labels_unsorted = refine_targets["matched_gt_labels"]
    pos_unsorted = refine_targets["refine_pos_mask"]

    labels_sorted = reorder_by_sort_indices(labels_unsorted, sort_indices)
    pos_sorted = reorder_by_sort_indices(pos_unsorted.long(), sort_indices).bool()

    loss_unsorted = aux_classification_loss(aux_unsorted, labels_unsorted, pos_unsorted)
    acc_unsorted = _aux_top1_accuracy(aux_unsorted, labels_unsorted, pos_unsorted)
    valid_uns = int((pos_unsorted & labels_unsorted.ne(-1)).sum().item())
    valid_sorted = int((pos_sorted & labels_sorted.ne(-1)).sum().item())
    aux_ordered = outputs.get("aux_logits_ordered", None)
    if aux_ordered is not None:
        loss_sorted = aux_classification_loss(aux_ordered, labels_sorted, pos_sorted)
        acc_sorted = _aux_top1_accuracy(aux_ordered, labels_sorted, pos_sorted)
        loss_mismatch = aux_classification_loss(aux_ordered, labels_unsorted, pos_unsorted)
        acc_mismatch = _aux_top1_accuracy(aux_ordered, labels_unsorted, pos_unsorted)

        tqdm.write(
            "[AUX ALIGN DEBUG] "
            f"loss uns={float(loss_unsorted.item()):.4f} | "
            f"loss sorted={float(loss_sorted.item()):.4f} | "
            f"loss mismatch={float(loss_mismatch.item()):.4f}"
            f"valid uns={valid_uns} | valid sorted={valid_sorted}"
        )
        tqdm.write(
            "[AUX ALIGN DEBUG] "
            f"acc uns={acc_unsorted:.4f} | "
            f"acc sorted={acc_sorted:.4f} | "
            f"acc mismatch={acc_mismatch:.4f}"
        )
    else:
        loss_sorted_reindexed = aux_classification_loss(aux_unsorted, labels_sorted, pos_sorted)
        acc_sorted_reindexed = _aux_top1_accuracy(aux_unsorted, labels_sorted, pos_sorted)
        tqdm.write(
            "[AUX ALIGN DEBUG] "
            f"loss uns={float(loss_unsorted.item()):.4f} | "
            f"loss sorted-reindexed={float(loss_sorted_reindexed.item()):.4f}"
            f"valid uns={valid_uns} | valid sorted={valid_sorted}"
        )
        tqdm.write(
            "[AUX ALIGN DEBUG] "
            f"acc uns={acc_unsorted:.4f} | "
            f"acc sorted-reindexed={acc_sorted_reindexed:.4f}"
        )

    _debug_one_sample_alignment(outputs, refine_targets, vocab=vocab, sample_idx=0, limit=limit)
    
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
    return _decode_text_from_ids(ids, vocab)


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


def _sequence_cer(pred_text: str, gt_text: str) -> float:
    return _edit_distance(pred_text, gt_text) / max(1, len(gt_text))


def _summarize_top_counter(counter: Counter, vocab: VocabManager, limit: int = 8) -> list[dict]:
    total = max(1, sum(counter.values()))
    summary = []
    for token_id, count in counter.most_common(limit):
        token_text = _ids_to_text([int(token_id)], vocab)
        summary.append({
            "token_id": int(token_id),
            "token": token_text if token_text else f"<{int(token_id)}>",
            "count": int(count),
            "share": float(count / total),
        })
    return summary


def _summarize_error_pairs(counter: Counter, vocab: VocabManager, limit: int = 8) -> list[dict]:
    summary = []
    for (gt_id, pred_id), count in counter.most_common(limit):
        gt_text = _ids_to_text([int(gt_id)], vocab)
        pred_text = _ids_to_text([int(pred_id)], vocab)
        summary.append({
            "gt_id": int(gt_id),
            "pred_id": int(pred_id),
            "gt": gt_text if gt_text else f"<{int(gt_id)}>",
            "pred": pred_text if pred_text else f"<{int(pred_id)}>",
            "count": int(count),
        })
    return summary


def _init_aux_branch_stats() -> dict:
    return {
        "total": 0,
        "correct_top1": 0,
        "correct_top5": 0,
        "pred_counter": Counter(),
        "error_pair_counter": Counter(),
    }


def _update_aux_branch_stats(
    branch_stats: dict,
    aux_logits: torch.Tensor,
    matched_labels: torch.Tensor,
    aux_valid: torch.Tensor,
) -> None:
    if not aux_valid.any():
        return

    valid_logits = aux_logits[aux_valid]
    valid_labels = matched_labels[aux_valid]
    if valid_logits.numel() == 0:
        return

    pred_labels = valid_logits.argmax(dim=-1)
    branch_stats["total"] += int(valid_labels.numel())
    branch_stats["correct_top1"] += int((pred_labels == valid_labels).sum().item())

    topk = min(5, int(valid_logits.size(-1)))
    topk_ids = valid_logits.topk(k=topk, dim=-1).indices
    top5_hits = topk_ids.eq(valid_labels.unsqueeze(1)).any(dim=1)
    branch_stats["correct_top5"] += int(top5_hits.sum().item())

    gt_list = valid_labels.detach().cpu().tolist()
    pred_list = pred_labels.detach().cpu().tolist()
    for gt_id, pred_id in zip(gt_list, pred_list):
        gt_id = int(gt_id)
        pred_id = int(pred_id)
        branch_stats["pred_counter"][pred_id] += 1
        if gt_id != pred_id:
            branch_stats["error_pair_counter"][(gt_id, pred_id)] += 1


def _finalize_aux_branch_stats(branch_stats: dict, vocab: VocabManager, available: bool) -> dict:
    total = int(branch_stats["total"])
    denom = max(1, total)
    return {
        "available": bool(available),
        "total": total,
        "top1": float(branch_stats["correct_top1"] / denom),
        "top5": float(branch_stats["correct_top5"] / denom),
        "top_predictions": _summarize_top_counter(branch_stats["pred_counter"], vocab),
        "top_errors": _summarize_error_pairs(branch_stats["error_pair_counter"], vocab),
    }


def _update_refinement_epoch_stats(stats: dict, outputs: dict, refine_targets: dict, gt_labels_list):
    proposal_mask = _valid_proposal_mask(outputs["roi_boxes"], outputs["roi_mask"])
    pos_mask = refine_targets["refine_pos_mask"] & proposal_mask
    neg_mask = refine_targets["refine_neg_mask"] & proposal_mask
    ign_mask = refine_targets["refine_ignore_mask"] & proposal_mask

    valid_props_per_image = proposal_mask.sum(dim=1)
    stats["images_with_zero_valid_props"] += int(valid_props_per_image.eq(0).sum().item())

    stats["images"] += int(outputs["roi_boxes"].size(0))
    stats["proposals"] += int(proposal_mask.sum().item())
    stats["positives"] += int(pos_mask.sum().item())
    stats["negatives"] += int(neg_mask.sum().item())
    stats["ignores"] += int(ign_mask.sum().item())
    stats["gt_tokens"] += int(sum(int(x.numel()) for x in gt_labels_list))

    matched_iou = refine_targets["matched_iou"]
    stats["matched_iou_sum"] += float(matched_iou[pos_mask].sum().item())
    stats["matched_iou_count"] += int(pos_mask.sum().item())

    matched_gt_index = refine_targets["matched_gt_index"]
    for b in range(int(outputs["roi_boxes"].size(0))):
        pos_idx_b = pos_mask[b]
        if not pos_idx_b.any():
            continue
        matched_gt_b = matched_gt_index[b][pos_idx_b]
        matched_gt_b = matched_gt_b[matched_gt_b.ge(0)]
        if matched_gt_b.numel() == 0:
            continue
        unique_gt_b = int(torch.unique(matched_gt_b).numel())
        pos_count_b = int(pos_idx_b.sum().item())
        stats["unique_gt_matched"] += unique_gt_b
        stats["duplicate_positive_matches"] += max(0, pos_count_b - unique_gt_b)

    refine_scores = outputs["refine_scores"]
    if refine_scores is not None:
        valid_scores = refine_scores[proposal_mask]
        if valid_scores.numel() > 0:
            score_prob = torch.sigmoid(valid_scores)
            stats["score_logit_sum"] += float(valid_scores.sum().item())
            stats["score_logit_sq_sum"] += float((valid_scores * valid_scores).sum().item())
            stats["score_prob_sum"] += float(score_prob.sum().item())
            stats["score_prob_sq_sum"] += float((score_prob * score_prob).sum().item())
            stats["score_count"] += int(valid_scores.numel())

    matched_labels = refine_targets["matched_gt_labels"]
    aux_valid = pos_mask & matched_labels.ne(-1)

    if outputs["aux_logits"] is not None:
        _update_aux_branch_stats(
            branch_stats=stats["aux_without_context_encode"],
            aux_logits=outputs["aux_logits"],
            matched_labels=matched_labels,
            aux_valid=aux_valid,
        )

    aux_with_context = outputs.get("aux_logits_with_context", None)
    if aux_with_context is not None:
        sort_indices = outputs.get("sort_indices", None)
        if sort_indices is not None:
            matched_labels_ctx = reorder_by_sort_indices(matched_labels, sort_indices)
            pos_mask_ctx = reorder_by_sort_indices(pos_mask.long(), sort_indices).bool()
        else:
            matched_labels_ctx = matched_labels
            pos_mask_ctx = pos_mask

        aux_valid_ctx = pos_mask_ctx & matched_labels_ctx.ne(-1)
        _update_aux_branch_stats(
            branch_stats=stats["aux_with_context_encode"],
            aux_logits=aux_with_context,
            matched_labels=matched_labels_ctx,
            aux_valid=aux_valid_ctx,
        )


def _update_decoder_epoch_stats(
    stats: dict,
    pred_ids_batch: torch.Tensor,
    text_ids_batch: torch.Tensor,
    vocab: VocabManager,
):
    for pred_ids, gt_ids in zip(pred_ids_batch.detach().cpu().tolist(), text_ids_batch.detach().cpu().tolist()):
        pred_tokens = _strip_special_tokens(pred_ids, vocab.pad_id, vocab.sos_id, vocab.eos_id)
        gt_tokens = _strip_special_tokens(gt_ids, vocab.pad_id, vocab.sos_id, vocab.eos_id)

        pred_text = _ids_to_text(pred_tokens, vocab)
        gt_text = _ids_to_text(gt_tokens, vocab)

        stats["samples"] += 1
        stats["pred_len_sum"] += len(pred_tokens)
        stats["gt_len_sum"] += len(gt_tokens)
        stats["cer_sum"] += _sequence_cer(pred_text, gt_text)
        stats["exact"] += int(pred_text == gt_text)

        if vocab.eos_id in pred_ids:
            stats["eos_hit"] += 1

        for token_id in pred_tokens:
            stats["pred_token_counter"][int(token_id)] += 1

        compare_len = min(len(pred_tokens), len(gt_tokens))
        for idx in range(compare_len):
            if pred_tokens[idx] != gt_tokens[idx]:
                stats["error_pair_counter"][(int(gt_tokens[idx]), int(pred_tokens[idx]))] += 1

        if len(stats["examples"]) < stats["example_limit"]:
            stats["examples"].append({
                "gt": gt_text,
                "pred": pred_text,
                "cer": _sequence_cer(pred_text, gt_text),
                "pred_len": len(pred_tokens),
                "gt_len": len(gt_tokens),
                "eos_hit": bool(vocab.eos_id in pred_ids),
            })


def _finalize_decoder_epoch_stats(stats: dict, vocab: VocabManager) -> dict:
    decoder_samples = max(1, stats["samples"])
    return {
        "samples": stats["samples"],
        "eos_hit_fraction": stats["eos_hit"] / decoder_samples,
        "mean_pred_len": stats["pred_len_sum"] / decoder_samples,
        "mean_gt_len": stats["gt_len_sum"] / decoder_samples,
        "mean_cer": stats["cer_sum"] / decoder_samples,
        "exact_match_fraction": stats["exact"] / decoder_samples,
        "top_tokens": _summarize_top_counter(stats["pred_token_counter"], vocab),
        "top_errors": _summarize_error_pairs(stats["error_pair_counter"], vocab),
        "examples": stats["examples"],
    }


def log_stage2_batch_debug(
    phase: str,
    epoch: Optional[int],
    batch_idx: int,
    roi_mask: torch.Tensor,
    refine_targets: dict,
):
    proposals_per_image = _count_valid_rois_per_image(roi_mask)
    positives_per_image = _count_mask_per_image(refine_targets["refine_pos_mask"])
    negatives_per_image = _count_mask_per_image(refine_targets["refine_neg_mask"])
    ignored_per_image = _count_mask_per_image(refine_targets["refine_ignore_mask"])

    epoch_text = f" epoch={epoch + 1}" if epoch is not None else ""
    tqdm.write(
        f"[{phase}] batch={batch_idx:04d}{epoch_text} | "
        f"proposals/img={proposals_per_image} | "
        f"pos/img={positives_per_image} | "
        f"neg/img={negatives_per_image} | "
        f"ign/img={ignored_per_image}"
    )








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

    det_score_thresh = float(overrides.get("det_score_thresh", STAGE2_DET_SCORE_THRESH))
    det_top_k = int(overrides.get("det_top_k", STAGE2_DET_TOP_K))
    det_nms_iou = float(overrides.get("det_nms_iou", STAGE2_DET_NMS_IOU))
    det_min_box_size = float(overrides.get("det_min_box_size", STAGE2_DET_MIN_BOX_SIZE))

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
            out.append(x.to(device=DEVICE))
        else:
            out.append(x.to(device=DEVICE, dtype=dtype) if dtype is not None else x.to(device=DEVICE))
    return out


def _strip_after_eos(ids: torch.Tensor, eos_id: int) -> torch.Tensor:
    """
    Cut sequence after first EOS (inclusive).
    """
    ids = ids.detach().cpu()
    eos_pos = (ids == eos_id).nonzero(as_tuple=False)
    if eos_pos.numel() == 0:
        return ids
    first = int(eos_pos[0].item())
    return ids[:first + 1]


def greedy_decode_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    logits: (B, T, V)
    returns: (B, T)
    """
    return logits.argmax(dim=-1)


def compute_simple_token_accuracy(
    pred_ids: torch.Tensor,
    tgt_ids: torch.Tensor,
    tgt_mask: torch.Tensor,
) -> float:
    """
    Position-wise token accuracy on masked positions.
    """
    valid = tgt_mask.bool()
    if valid.sum() == 0:
        return 0.0
    correct = (pred_ids[valid] == tgt_ids[valid]).float().mean()
    return float(correct.item())


@torch.no_grad()
def validate_stage2(
    model: HybridKuroNetRecognizer,
    val_loader: DataLoader,
    vocab: VocabManager,
    phase: str = "A",
    max_batches: Optional[int] = None,
):
    model.eval()
    if FREEZE_BACKBONE:
        model.backbone.eval()
    if FREEZE_DETECTOR:
        model.detector.eval()
    phase_settings = get_phase_settings(phase)
    total_loss = 0.0
    total_box = 0.0
    total_delta = 0.0
    total_score = 0.0
    total_aux = 0.0
    total_decoder = 0.0
    total_acc = 0.0
    n_batches = 0

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
        "aux_without_context_encode": _init_aux_branch_stats(),
        "aux_with_context_encode": _init_aux_branch_stats(),
    }

    decoder_stats_tf = {
        "samples": 0,
        "pred_len_sum": 0,
        "gt_len_sum": 0,
        "cer_sum": 0.0,
        "exact": 0,
        "eos_hit": 0,
        "pred_token_counter": Counter(),
        "error_pair_counter": Counter(),
        "examples": [],
        "example_limit": 3,
    }
    decoder_stats_free = {
        "samples": 0,
        "pred_len_sum": 0,
        "gt_len_sum": 0,
        "cer_sum": 0.0,
        "exact": 0,
        "eos_hit": 0,
        "pred_token_counter": Counter(),
        "error_pair_counter": Counter(),
        "examples": [],
        "example_limit": 3,
    }
    val_orientation_counts: Counter[str] = Counter()

    pad_id = vocab.pad_id
    sos_id = vocab.sos_id
    eos_id = vocab.eos_id

    for batch_idx, batch in enumerate(tqdm(val_loader, desc="Stage2 Validation", leave=False)):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        images = batch["image"].to(DEVICE)
        text_ids = batch["text_ids"]
        if text_ids is None:
            continue
        text_ids = text_ids.to(DEVICE)
        orientations = batch["orientations"]
        val_orientation_counts.update(_normalize_orientation_label(x) for x in orientations)

        gt_boxes_list = move_gt_lists_to_device(batch["boxes"], dtype=torch.float32)
        gt_labels_list = move_gt_lists_to_device(batch["labels"], dtype=torch.long)

        dec_targets = build_decoder_targets(
            text_ids=text_ids,
            pad_id=pad_id,
        )

        outputs = model(
            images=images,
            orientations=orientations,
            input_seq=dec_targets["decoder_inputs"],
            targets=dec_targets["target_tokens"],
            teacher_forcing_ratio=1.0,   # oder 1.0 für echte teacher-forced diagnostics
            sos_id=sos_id,
            eos_id=eos_id,
            max_len=None,
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

        losses = compute_stage2_total_loss(
            refined_boxes=outputs["refined_boxes"],
            box_deltas=outputs["box_deltas"],
            refine_scores=outputs["refine_scores"],
            aux_logits=outputs["aux_logits_with_context"] if phase_settings["use_context_aux_for_loss"] else outputs["aux_logits"],
            decoder_logits=outputs["decoder_logits"],

            matched_gt_boxes=refine_targets["matched_gt_boxes"],
            target_deltas=refine_targets["target_deltas"],
            matched_gt_labels=refine_targets["matched_gt_labels"],
            refine_pos_mask=refine_targets["refine_pos_mask"],
            refine_neg_mask=refine_targets["refine_neg_mask"],
            refine_ignore_mask=refine_targets["refine_ignore_mask"],

            aux_target_labels=reorder_by_sort_indices(refine_targets["matched_gt_labels"], outputs["sort_indices"]) if phase_settings["use_context_aux_for_loss"] else None,
            aux_pos_mask=reorder_by_sort_indices(refine_targets["refine_pos_mask"].long(), outputs["sort_indices"]).bool() if phase_settings["use_context_aux_for_loss"] else None,

            target_tokens=dec_targets["target_tokens"],
            target_mask=dec_targets["target_mask"],

            lambda_box=phase_settings["lambda_box"],
            lambda_delta=phase_settings["lambda_delta"],
            lambda_score=phase_settings["lambda_score"],
            lambda_aux=phase_settings["lambda_aux"],
            lambda_decoder=phase_settings["lambda_decoder"],
            refine_pos_weight=STAGE2_REFINE_POS_WEIGHT,
            decoder_label_smoothing=STAGE2_DECODER_LABEL_SMOOTHING,
            decoder_eos_id=eos_id,
            decoder_eos_weight=STAGE2_DECODER_EOS_WEIGHT,
        )

        if STAGE2_DEBUG_AUX_ALIGNMENT and batch_idx == 0:
            _debug_aux_alignment(
                outputs=outputs,
                refine_targets=refine_targets,
                vocab=vocab,
                limit=int(STAGE2_DEBUG_AUX_ALIGNMENT_LIMIT),
            )

        pred_ids = greedy_decode_from_logits(outputs["decoder_logits"])
        batch_acc = compute_simple_token_accuracy(
            pred_ids=pred_ids,
            tgt_ids=dec_targets["target_tokens"],
            tgt_mask=dec_targets["target_mask"],
        )

        total_loss += float(losses["loss_total"].item())
        total_box += float(losses["loss_box"].item())
        total_delta += float(losses["loss_delta"].item())
        total_score += float(losses["loss_score"].item())
        total_aux += float(losses["loss_aux"].item())
        total_decoder += float(losses["loss_decoder"].item())
        total_acc += batch_acc
        n_batches += 1

        _update_refinement_epoch_stats(proposal_stats, outputs, refine_targets, gt_labels_list)
        _update_decoder_epoch_stats(decoder_stats_tf, pred_ids, dec_targets["target_tokens"], vocab)

        if phase_settings["log_free_decoder"]:
            free_outputs = model(
                images=images,
                orientations=orientations,
                targets=None,
                teacher_forcing_ratio=0.0,
                input_seq=None,
                sos_id=sos_id,
                eos_id=eos_id,
                max_len=STAGE2_VAL_MAX_DECODE_LEN,
            )
        
            free_pred_ids = greedy_decode_from_logits(free_outputs["decoder_logits"])
            _update_decoder_epoch_stats(decoder_stats_free, free_pred_ids, text_ids, vocab)

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
    return {
        "val_loss": total_loss / denom,
        "val_box": total_box / denom,
        "val_delta": total_delta / denom,
        "val_score": total_score / denom,
        "val_aux": total_aux / denom,
        "val_decoder": total_decoder / denom,
        "val_token_acc": total_acc / denom,
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
            "aux_accuracy_on_positives": aux_without_ctx["top1"],
            "aux_top5_on_positives": aux_without_ctx["top5"],
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

    phase_settings = get_phase_settings(phase)
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
        total_acc = 0.0
        n_batches = 0

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
            "aux_without_context_encode": _init_aux_branch_stats(),
            "aux_with_context_encode": _init_aux_branch_stats(),
        }
        train_orientation_counts: Counter[str] = Counter()

        pbar = tqdm(train_loader, desc=f"Stage2 Epoch {epoch+1}/{num_epochs}", mininterval=10.0)

        for step, batch in enumerate(pbar):
            images = batch["image"].to(DEVICE)
            text_ids = batch["text_ids"]
            if text_ids is None:
                continue
            text_ids = text_ids.to(DEVICE)

            orientations = batch["orientations"]
            train_orientation_counts.update(_normalize_orientation_label(x) for x in orientations)
            gt_boxes_list = move_gt_lists_to_device(batch["boxes"], dtype=torch.float32)
            gt_labels_list = move_gt_lists_to_device(batch["labels"], dtype=torch.long)

            dec_targets = build_decoder_targets(
                text_ids=text_ids,
                pad_id=pad_id,
            )

            with torch.cuda.amp.autocast(enabled=USE_MIXED_PRECISION and str(DEVICE).startswith("cuda")):
                outputs = model(
                    images=images,
                    orientations=orientations,
                    input_seq=dec_targets["decoder_inputs"],
                    targets=dec_targets["target_tokens"],
                    teacher_forcing_ratio=tf_ratio,
                    sos_id=sos_id,
                    eos_id=eos_id,
                    max_len=None,
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

                losses = compute_stage2_total_loss(
                    refined_boxes=outputs["refined_boxes"],
                    box_deltas=outputs["box_deltas"],
                    refine_scores=outputs["refine_scores"],
                    aux_logits=outputs["aux_logits_with_context"] if phase_settings["use_context_aux_for_loss"] else outputs["aux_logits"],
                    decoder_logits=outputs["decoder_logits"],

                    matched_gt_boxes=refine_targets["matched_gt_boxes"],
                    target_deltas=refine_targets["target_deltas"],
                    matched_gt_labels=refine_targets["matched_gt_labels"],
                    refine_pos_mask=refine_targets["refine_pos_mask"],
                    refine_neg_mask=refine_targets["refine_neg_mask"],
                    refine_ignore_mask=refine_targets["refine_ignore_mask"],

                    aux_target_labels=reorder_by_sort_indices(refine_targets["matched_gt_labels"], outputs["sort_indices"]) if phase_settings["use_context_aux_for_loss"] else None,
                    aux_pos_mask=reorder_by_sort_indices(refine_targets["refine_pos_mask"].long(), outputs["sort_indices"]).bool() if phase_settings["use_context_aux_for_loss"] else None,

                    target_tokens=dec_targets["target_tokens"],
                    target_mask=dec_targets["target_mask"],

                    lambda_box=phase_settings["lambda_box"],
                    lambda_delta=phase_settings["lambda_delta"],
                    lambda_score=phase_settings["lambda_score"],
                    lambda_aux=phase_settings["lambda_aux"],
                    lambda_decoder=phase_settings["lambda_decoder"],
                    refine_pos_weight=STAGE2_REFINE_POS_WEIGHT,
                    decoder_label_smoothing=STAGE2_DECODER_LABEL_SMOOTHING,
                    decoder_eos_id=eos_id,
                    decoder_eos_weight=STAGE2_DECODER_EOS_WEIGHT,
                )

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

            pred_ids = greedy_decode_from_logits(outputs["decoder_logits"])
            batch_acc = compute_simple_token_accuracy(
                pred_ids=pred_ids,
                tgt_ids=dec_targets["target_tokens"],
                tgt_mask=dec_targets["target_mask"],
            )

            total_loss += float(losses["loss_total"].item())
            total_box += float(losses["loss_box"].item())
            total_delta += float(losses["loss_delta"].item())
            total_score += float(losses["loss_score"].item())
            total_aux += float(losses["loss_aux"].item())
            total_decoder += float(losses["loss_decoder"].item())
            total_acc += batch_acc
            n_batches += 1

            pbar.set_postfix({
                "loss": total_loss / max(1, n_batches),
                "dec": total_decoder / max(1, n_batches),
                "box": total_box / max(1, n_batches),
                "delta": total_delta / max(1, n_batches),
                "acc": total_acc / max(1, n_batches),
                "tf": tf_ratio,
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

        train_metrics = {
            "train_loss": total_loss / max(1, n_batches),
            "train_box": total_box / max(1, n_batches),
            "train_delta": total_delta / max(1, n_batches),
            "train_score": total_score / max(1, n_batches),
            "train_aux": total_aux / max(1, n_batches),
            "train_decoder": total_decoder / max(1, n_batches),
            "train_token_acc": total_acc / max(1, n_batches),
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
                "aux_accuracy_on_positives": train_aux_without_ctx["top1"],
                "aux_top5_on_positives": train_aux_without_ctx["top5"],
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
        )

        print(
            f"\nEpoch {epoch+1}/{num_epochs} | "
            f"Train loss={train_metrics['train_loss']:.4f} "
            f"(dec={train_metrics['train_decoder']:.4f}, "
            f"box={train_metrics['train_box']:.4f}, "
            f"delta={train_metrics['train_delta']:.4f}, "
            f"score={train_metrics['train_score']:.4f}, "
            f"aux={train_metrics['train_aux']:.4f}, "
            f"acc={train_metrics['train_token_acc']:.4f}) | "
            f"Val loss={val_metrics['val_loss']:.4f} "
            f"(dec={val_metrics['val_decoder']:.4f}, "
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
        if phase_settings["log_free_decoder"]:
            print(
                "Decoder summary | "
                f"TF EOS hit={val_dec_tf['eos_hit_fraction']:.3f}, "
                f"TF CER={val_dec_tf['mean_cer']:.4f}, "
                f"TF len={val_dec_tf['mean_pred_len']:.2f}/{val_dec_tf['mean_gt_len']:.2f} | "
                f"FREE EOS hit={val_dec_free['eos_hit_fraction']:.3f}, "
                f"FREE CER={val_dec_free['mean_cer']:.4f}, "
                f"FREE len={val_dec_free['mean_pred_len']:.2f}/{val_dec_free['mean_gt_len']:.2f}, "
                f"FREE exact={val_dec_free['exact_match_fraction']:.3f}"
            )
            print(f"Decoder top tokens (FREE): {val_dec_free['top_tokens']}")
            print(f"Decoder top errors (FREE): {val_dec_free['top_errors']}")
            if val_dec_free["examples"]:
                print("Decoder qualitative examples:")
                for idx, example in enumerate(val_dec_free["examples"], start=1):
                    print(
                        f"  [{idx}] CER={example['cer']:.3f} len={example['pred_len']}/{example['gt_len']} "
                        f"EOS={int(example['eos_hit'])} | GT={example['gt']} | Pred={example['pred']}"
                    )
        else:
            print(
                "Decoder summary | "
                f"TF EOS hit={val_dec_tf['eos_hit_fraction']:.3f}, "
                f"TF CER={val_dec_tf['mean_cer']:.4f}, "
                f"TF len={val_dec_tf['mean_pred_len']:.2f}/{val_dec_tf['mean_gt_len']:.2f}"
            )

        is_best = best_val is None or val_metrics["val_loss"] < best_val
        if is_best:
            best_val = val_metrics["val_loss"]
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

    checkpoint_dir = CHECKPOINT_DIR / f"stage2_hybrid_phase_test_{selected_phase}"
    resume_model_ckpt = None
    if selected_phase == "B":
        print("Phase B selected: will attempt to resume from Phase A2 best checkpoint, then Phase A best checkpoint.")
        phase_a2_best = CHECKPOINT_DIR / "stage2_hybrid_phaseA2" / "stage2_hybrid_best.pt"
        phase_a_best = CHECKPOINT_DIR / "stage2_hybrid_phaseA" / "stage2_hybrid_best.pt"
        if phase_a2_best.exists():
            resume_model_ckpt = phase_a2_best
        elif phase_a_best.exists():
            resume_model_ckpt = phase_a_best
        else:
            raise FileNotFoundError(
                "Phase B requested but no Phase-A2/Phase-A best checkpoint found at one of:\n"
                f"  - {phase_a2_best}\n"
                f"  - {phase_a_best}\n"
                "Run Phase A2 (preferred) or Phase A first."
            )

    train_stage2_hybrid(
        detector_ckpt_path=detector_ckpt,
        num_epochs=phase_settings["epochs"],
        lr=LR,
        checkpoint_dir=checkpoint_dir,
        phase=selected_phase,
        resume_model_ckpt=resume_model_ckpt,
    )


if __name__ == "__main__":
    main()