
from __future__ import annotations

from typing import Optional

import torch

# Phase B free-decoding was removed; these flags are kept as inline constants
# for the deprecated helper functions below that are no longer called.
_CHUNKED_FREE_DECODE = True
_CHUNKED_FREE_TRAINING = True
from model.suminanet.hybrid_recognizer import HybridSuminaNetRecognizer
from utils.stage2_losses import build_decoder_roi_targets_monotonic, build_decoder_roi_targets_from_bias
from utils.vocab import VocabManager

__all__ = [
    "_count_mask_per_image",
    "_decode_free_by_reading_units",
    "_detach_encoded_for_free_decode",
    "_get_action_class_weights",
    "_load_compatible_state_dict",
    "_normalize_orientation_label",
    "_prepare_chunked_free_training_batch",
    "_select_pred_ids",
    "_valid_proposal_mask",
    "compute_simple_token_accuracy",
    "greedy_decode_from_logits",
    "reorder_by_sort_indices",
]


def _load_compatible_state_dict(
    module: torch.nn.Module,
    state_dict: dict,
    *,
    ckpt_context_mode: Optional[str] = None,
    ckpt_vocab_hash: Optional[str] = None,
    current_vocab_hash: Optional[str] = None,
    ckpt_backbone_type: Optional[str] = None,
) -> None:
    """Load only tensors whose names and shapes match the current module.

    ckpt_context_mode, if given, is checked against module.context_encoder.mode
    and raises on mismatch. At STAGE2_CONTEXT_NUM_LAYERS=2, switching gru<->bigru
    changes the shape of exactly one RNN key (context_encoder.rnn.weight_ih_l1)
    while the rest coincidentally keep matching shapes, so the shape-based skip
    logic below would otherwise silently reinitialize that one layer instead of
    raising -- a partially-corrupted recurrent layer that trains/evals with no
    error.

    ckpt_vocab_hash/current_vocab_hash, if both given, are compared and raise on
    mismatch for the same reason: the vocab is rebuilt fresh from the annotation
    corpus on every run, so a classifier head trained against an older vocab could
    otherwise be silently reinitialized (different vocab_size) or, worse, silently
    kept with now-wrong char-to-ID semantics (same vocab_size, different mapping).

    ckpt_backbone_type, if given, is checked against module.backbone_type and
    raises on mismatch. 'unet' and 'efficientnet_b2' backbones share no key
    names or shapes at all, so every backbone.* tensor would land in
    skipped_keys and the backbone would silently stay randomly initialized
    instead of loading the checkpoint's trained weights -- the model still
    runs and produces output, just garbage, with no error anywhere.
    """
    current_mode = getattr(getattr(module, "context_encoder", None), "mode", None)
    if ckpt_context_mode is not None and current_mode is not None and ckpt_context_mode != current_mode:
        raise ValueError(
            f"Context-mode mismatch: checkpoint was trained with context_mode="
            f"'{ckpt_context_mode}' but the current model is configured with "
            f"STAGE2_CONTEXT_MODE='{current_mode}'. Loading anyway would silently "
            f"corrupt one RNN layer (gru/bigru coincidentally share most tensor "
            f"shapes). Set STAGE2_CONTEXT_MODE='{ckpt_context_mode}' to match this "
            f"checkpoint, or use a checkpoint trained with '{current_mode}'."
        )

    current_backbone_type = getattr(module, "backbone_type", None)
    if ckpt_backbone_type is not None and current_backbone_type is not None and ckpt_backbone_type != current_backbone_type:
        raise ValueError(
            f"Backbone-type mismatch: checkpoint was trained with backbone_type="
            f"'{ckpt_backbone_type}' but the current model was built with "
            f"backbone_type='{current_backbone_type}'. Loading anyway would silently "
            f"skip every backbone.* tensor (unet/efficientnet_b2 share no key names "
            f"or shapes), leaving the backbone randomly initialized instead of "
            f"raising. Build the model with backbone_type_override="
            f"'{ckpt_backbone_type}' (or config.BACKBONE_TYPE='{ckpt_backbone_type}') "
            f"to match this checkpoint."
        )

    if ckpt_vocab_hash is not None and current_vocab_hash is not None and ckpt_vocab_hash != current_vocab_hash:
        raise ValueError(
            f"Vocab mismatch: checkpoint was trained with vocab_hash='{ckpt_vocab_hash}' "
            f"but the current vocab (rebuilt from assets/data/annotations/) hashes to "
            f"'{current_vocab_hash}'. Loading anyway risks silently corrupting the "
            f"classifier head if vocab_size coincidentally matches (shape-based skip "
            f"logic would treat it as compatible), or leaving it randomly reinitialized "
            f"if not. Use the vocab this checkpoint was trained with, or retrain/fine-tune "
            f"from scratch against the current vocab."
        )

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


def _count_mask_per_image(mask: torch.Tensor) -> list[int]:
    if mask.dim() == 1:
        return [int(mask.to(dtype=torch.long).sum().item())]
    return [int(v) for v in mask.to(dtype=torch.long).sum(dim=1).tolist()]


def _get_action_class_weights(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor([2.0, 1.0, 4.0], device=device, dtype=dtype)


def _detach_encoded_for_free_decode(encoded: dict) -> dict:
    detached = dict(encoded)
    if "context_feats" in detached and torch.is_tensor(detached["context_feats"]):
        detached["context_feats"] = detached["context_feats"].detach()
    if "decoder_token_bias" in detached and torch.is_tensor(detached["decoder_token_bias"]):
        detached["decoder_token_bias"] = detached["decoder_token_bias"].detach()
    return detached


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


def greedy_decode_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    logits: (B, T, V)
    returns: (B, T)
    """
    return logits.argmax(dim=-1)


def _select_pred_ids(outputs: dict, prefer_emitted: bool = False) -> torch.Tensor:
    if prefer_emitted:
        emitted = outputs.get("emitted_token_ids", None)
        if emitted is not None:
            return emitted
    return greedy_decode_from_logits(outputs["decoder_logits"])


def _reading_unit_spans_from_sorted_boxes(
    boxes_sorted_valid: torch.Tensor,
    orientation: str,
    line_merge_thresh_ratio: float = 0.6,
) -> list[tuple[int, int]]:
    n = int(boxes_sorted_valid.size(0))
    if n <= 0:
        return []

    orientation = _normalize_orientation_label(orientation)
    if orientation == "horizontal":
        centers = (boxes_sorted_valid[:, 1] + boxes_sorted_valid[:, 3]) * 0.5
        scales = (boxes_sorted_valid[:, 3] - boxes_sorted_valid[:, 1]).clamp_min(1.0)
    else:
        centers = (boxes_sorted_valid[:, 0] + boxes_sorted_valid[:, 2]) * 0.5
        scales = (boxes_sorted_valid[:, 2] - boxes_sorted_valid[:, 0]).clamp_min(1.0)

    thresh = float((float(line_merge_thresh_ratio) * scales.median()).item())
    spans: list[tuple[int, int]] = []
    start = 0
    running_center = float(centers[0].item())
    running_count = 1

    for idx in range(1, n):
        center = float(centers[idx].item())
        if abs(center - running_center) <= thresh:
            running_center = (running_center * running_count + center) / float(running_count + 1)
            running_count += 1
        else:
            spans.append((start, idx))
            start = idx
            running_center = center
            running_count = 1

    spans.append((start, n))
    return spans


@torch.no_grad()
def _decode_free_by_reading_units(
    model: HybridSuminaNetRecognizer,
    encoded: dict,
    orientations: list[str],
    vocab: VocabManager,
    *,
    sos_id: int,
    eos_id: int,
    max_len: int,
    decode_constraints: Optional[dict],
    prefer_emitted: bool = True,
) -> dict:
    if not bool(_CHUNKED_FREE_DECODE):
        outputs = model.decode_from_encoded(
            encoded,
            targets=None,
            teacher_forcing_ratio=0.0,
            input_seq=None,
            sos_id=sos_id,
            eos_id=eos_id,
            max_len=max_len,
            decode_constraints=decode_constraints,
        )
        return {
            "pred_ids": _select_pred_ids(outputs, prefer_emitted=prefer_emitted),
            "action_logits": outputs.get("action_logits", None),
            "pointer_positions": outputs.get("pointer_positions", None),
            "valid_mask": None,
        }

    context_feats = encoded.get("context_feats", None)
    context_mask = encoded.get("context_mask", None)
    ordered_boxes = encoded.get("ordered_boxes", None)
    decoder_token_bias = encoded.get("decoder_token_bias", None)

    if (
        context_feats is None
        or context_mask is None
        or ordered_boxes is None
        or len(orientations) != int(context_feats.size(0))
    ):
        outputs = model.decode_from_encoded(
            encoded,
            targets=None,
            teacher_forcing_ratio=0.0,
            input_seq=None,
            sos_id=sos_id,
            eos_id=eos_id,
            max_len=max_len,
            decode_constraints=decode_constraints,
        )
        return {
            "pred_ids": _select_pred_ids(outputs, prefer_emitted=prefer_emitted),
            "action_logits": outputs.get("action_logits", None),
            "pointer_positions": outputs.get("pointer_positions", None),
            "valid_mask": None,
        }

    bsz, _, feat_dim = context_feats.shape
    device = context_feats.device

    chunk_feats: list[torch.Tensor] = []
    chunk_masks: list[torch.Tensor] = []
    chunk_biases: list[torch.Tensor] = []
    chunk_meta: list[tuple[int, int, int]] = []

    for b in range(bsz):
        valid_t = int(context_mask[b].sum().item())
        if valid_t <= 0:
            continue

        spans = _reading_unit_spans_from_sorted_boxes(
            ordered_boxes[b, :valid_t],
            orientations[b],
        )
        if not spans:
            spans = [(0, valid_t)]

        for start, end in spans:
            if end <= start:
                continue
            chunk_feats.append(context_feats[b, start:end])
            chunk_masks.append(context_mask[b, start:end])
            if decoder_token_bias is not None:
                chunk_biases.append(decoder_token_bias[b, start:end])
            chunk_meta.append((b, start, end))

    if not chunk_meta:
        pred_ids = torch.full((bsz, 1), vocab.pad_id, device=device, dtype=torch.long)
        return {
            "pred_ids": pred_ids,
            "action_logits": None,
            "pointer_positions": None,
            "valid_mask": None,
        }

    max_chunk_tokens = max(int(x.size(0)) for x in chunk_feats)
    chunk_bsz = len(chunk_meta)
    chunk_feat_batch = context_feats.new_zeros((chunk_bsz, max_chunk_tokens, feat_dim))
    chunk_mask_batch = torch.zeros((chunk_bsz, max_chunk_tokens), device=device, dtype=torch.bool)
    chunk_bias_batch = None
    if decoder_token_bias is not None:
        vocab_dim = int(decoder_token_bias.size(-1))
        chunk_bias_batch = decoder_token_bias.new_zeros((chunk_bsz, max_chunk_tokens, vocab_dim))

    for idx, feats in enumerate(chunk_feats):
        t = int(feats.size(0))
        chunk_feat_batch[idx, :t] = feats
        chunk_mask_batch[idx, :t] = chunk_masks[idx]
        if chunk_bias_batch is not None:
            chunk_bias_batch[idx, :t] = chunk_biases[idx]

    chunk_max_decode_len = min(int(max_len), max(16, 2 * max_chunk_tokens + 8))
    chunk_encoded = {
        "context_feats": chunk_feat_batch,
        "context_mask": chunk_mask_batch,
        "decoder_token_bias": chunk_bias_batch,
    }
    chunk_outputs = model.decode_from_encoded(
        chunk_encoded,
        targets=None,
        teacher_forcing_ratio=0.0,
        input_seq=None,
        sos_id=sos_id,
        eos_id=eos_id,
        max_len=chunk_max_decode_len,
        decode_constraints=decode_constraints,
    )

    chunk_pred_ids = _select_pred_ids(chunk_outputs, prefer_emitted=prefer_emitted)
    chunk_action_logits = chunk_outputs.get("action_logits", None)
    chunk_pointer_positions = chunk_outputs.get("pointer_positions", None)

    page_pred_lists: list[list[int]] = [[] for _ in range(bsz)]
    page_action_lists: list[list[torch.Tensor]] = [[] for _ in range(bsz)]
    page_pointer_lists: list[list[torch.Tensor]] = [[] for _ in range(bsz)]
    page_valid_lens = [0 for _ in range(bsz)]
    page_has_eos = [False for _ in range(bsz)]
    last_chunk_idx: dict[int, int] = {}
    for idx, (sample_idx, _start, _end) in enumerate(chunk_meta):
        last_chunk_idx[sample_idx] = idx

    for idx, (sample_idx, start, _end) in enumerate(chunk_meta):
        pred_row = [int(x) for x in chunk_pred_ids[idx].detach().cpu().tolist()]
        eos_pos = pred_row.index(int(eos_id)) if int(eos_id) in pred_row else -1
        raw_len = (eos_pos + 1) if eos_pos >= 0 else len(pred_row)

        pred_core = [
            tok for tok in pred_row[:raw_len]
            if tok not in (int(vocab.pad_id), int(vocab.sos_id), int(eos_id))
        ]
        page_pred_lists[sample_idx].extend(pred_core)

        if eos_pos >= 0 and last_chunk_idx.get(sample_idx, idx) == idx:
            page_has_eos[sample_idx] = True

        if chunk_action_logits is not None and chunk_pointer_positions is not None:
            act_slice = chunk_action_logits[idx, :raw_len].detach()
            ptr_slice = chunk_pointer_positions[idx, :raw_len].detach().to(dtype=torch.long) + int(start)
            page_action_lists[sample_idx].append(act_slice)
            page_pointer_lists[sample_idx].append(ptr_slice)
            page_valid_lens[sample_idx] += int(raw_len)

    pred_lengths = [
        len(tokens) + (1 if page_has_eos[sample_idx] else 0)
        for sample_idx, tokens in enumerate(page_pred_lists)
    ]
    pred_t_max = max(1, max(pred_lengths))
    page_pred_ids = torch.full((bsz, pred_t_max), vocab.pad_id, device=device, dtype=torch.long)
    for sample_idx, tokens in enumerate(page_pred_lists):
        seq = list(tokens)
        if page_has_eos[sample_idx]:
            seq.append(int(eos_id))
        if seq:
            page_pred_ids[sample_idx, :len(seq)] = torch.tensor(seq, device=device, dtype=torch.long)

    action_batch = None
    pointer_batch = None
    valid_mask_batch = None
    max_action_len = max(page_valid_lens) if page_valid_lens else 0
    if max_action_len > 0 and chunk_action_logits is not None and chunk_pointer_positions is not None:
        action_batch = chunk_action_logits.new_zeros((bsz, max_action_len, chunk_action_logits.size(-1)))
        pointer_batch = torch.zeros((bsz, max_action_len), device=device, dtype=torch.long)
        valid_mask_batch = torch.zeros((bsz, max_action_len), device=device, dtype=torch.bool)
        for sample_idx in range(bsz):
            if not page_action_lists[sample_idx]:
                continue
            act_cat = torch.cat(page_action_lists[sample_idx], dim=0)
            ptr_cat = torch.cat(page_pointer_lists[sample_idx], dim=0)
            t = int(act_cat.size(0))
            action_batch[sample_idx, :t] = act_cat
            pointer_batch[sample_idx, :t] = ptr_cat
            valid_mask_batch[sample_idx, :t] = True

    return {
        "pred_ids": page_pred_ids,
        "action_logits": action_batch,
        "pointer_positions": pointer_batch,
        "valid_mask": valid_mask_batch,
    }


def _prepare_chunked_free_training_batch(
    encoded: dict,
    dec_targets: dict,
    orientations: list[str],
    *,
    eos_id: int,
    pad_id: int,
    roi_targets: Optional[torch.Tensor] = None,
) -> Optional[dict]:
    if not bool(_CHUNKED_FREE_TRAINING):
        return None

    context_feats = encoded.get("context_feats", None)
    context_mask = encoded.get("context_mask", None)
    ordered_boxes = encoded.get("ordered_boxes", None)
    decoder_token_bias = encoded.get("decoder_token_bias", None)
    decoder_inputs = dec_targets.get("decoder_inputs", None)
    target_tokens = dec_targets.get("target_tokens", None)
    target_mask = dec_targets.get("target_mask", None)

    if (
        context_feats is None
        or context_mask is None
        or ordered_boxes is None
        or decoder_inputs is None
        or target_tokens is None
        or target_mask is None
        or len(orientations) != int(context_feats.size(0))
    ):
        return None

    bsz, _, feat_dim = context_feats.shape
    device = context_feats.device
    if roi_targets is None:
        if decoder_token_bias is not None:
            roi_targets = build_decoder_roi_targets_from_bias(
                target_tokens=target_tokens,
                target_mask=target_mask,
                enc_mask=context_mask,
                eos_id=int(eos_id),
                encoder_token_bias=decoder_token_bias,
                aux_weight=0.35,
            )
        else:
            roi_targets = build_decoder_roi_targets_monotonic(
                target_tokens=target_tokens,
                target_mask=target_mask,
                enc_mask=context_mask,
                eos_id=int(eos_id),
            )

    chunk_feats: list[torch.Tensor] = []
    chunk_masks: list[torch.Tensor] = []
    chunk_biases: list[torch.Tensor] = []
    chunk_decoder_inputs: list[torch.Tensor] = []
    chunk_target_tokens: list[torch.Tensor] = []
    chunk_target_masks: list[torch.Tensor] = []

    for b in range(bsz):
        valid_t = int(context_mask[b].sum().item())
        if valid_t <= 0:
            continue

        spans = _reading_unit_spans_from_sorted_boxes(
            ordered_boxes[b, :valid_t],
            orientations[b],
        )
        if not spans:
            spans = [(0, valid_t)]

        for start, end in spans:
            if end <= start:
                continue

            span_token_mask = (
                target_mask[b].bool()
                & roi_targets[b].ge(int(start))
                & roi_targets[b].lt(int(end))
            )
            token_positions = torch.nonzero(span_token_mask, as_tuple=False).squeeze(1)
            if token_positions.numel() <= 0:
                continue

            local_target_tokens = target_tokens[b, token_positions]
            local_target_mask = torch.ones_like(local_target_tokens, dtype=torch.bool)
            first_input_tok = decoder_inputs[b, token_positions[0]].view(1)
            if local_target_tokens.numel() > 1:
                local_decoder_inputs = torch.cat([first_input_tok, local_target_tokens[:-1]], dim=0)
            else:
                local_decoder_inputs = first_input_tok

            chunk_feats.append(context_feats[b, start:end])
            chunk_masks.append(context_mask[b, start:end])
            if decoder_token_bias is not None:
                chunk_biases.append(decoder_token_bias[b, start:end])
            chunk_decoder_inputs.append(local_decoder_inputs)
            chunk_target_tokens.append(local_target_tokens)
            chunk_target_masks.append(local_target_mask)

    if not chunk_feats:
        return None

    max_chunk_tokens = max(int(x.size(0)) for x in chunk_feats)
    max_chunk_decode_len = max(int(x.size(0)) for x in chunk_target_tokens)
    chunk_bsz = len(chunk_feats)

    chunk_feat_batch = context_feats.new_zeros((chunk_bsz, max_chunk_tokens, feat_dim))
    chunk_mask_batch = torch.zeros((chunk_bsz, max_chunk_tokens), device=device, dtype=torch.bool)
    chunk_bias_batch = None
    if decoder_token_bias is not None:
        vocab_dim = int(decoder_token_bias.size(-1))
        chunk_bias_batch = decoder_token_bias.new_zeros((chunk_bsz, max_chunk_tokens, vocab_dim))

    decoder_input_batch = torch.full(
        (chunk_bsz, max_chunk_decode_len),
        int(pad_id),
        device=device,
        dtype=torch.long,
    )
    target_token_batch = torch.full(
        (chunk_bsz, max_chunk_decode_len),
        int(pad_id),
        device=device,
        dtype=torch.long,
    )
    target_mask_batch = torch.zeros((chunk_bsz, max_chunk_decode_len), device=device, dtype=torch.bool)

    for idx, feats in enumerate(chunk_feats):
        t_enc = int(feats.size(0))
        t_dec = int(chunk_target_tokens[idx].size(0))
        chunk_feat_batch[idx, :t_enc] = feats
        chunk_mask_batch[idx, :t_enc] = chunk_masks[idx]
        decoder_input_batch[idx, :t_dec] = chunk_decoder_inputs[idx]
        target_token_batch[idx, :t_dec] = chunk_target_tokens[idx]
        target_mask_batch[idx, :t_dec] = chunk_target_masks[idx]
        if chunk_bias_batch is not None:
            chunk_bias_batch[idx, :t_enc] = chunk_biases[idx]

    return {
        "encoded": {
            "context_feats": chunk_feat_batch,
            "context_mask": chunk_mask_batch,
            "decoder_token_bias": chunk_bias_batch,
        },
        "decoder_inputs": decoder_input_batch,
        "target_tokens": target_token_batch,
        "target_mask": target_mask_batch,
    }


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
