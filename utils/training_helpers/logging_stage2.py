from __future__ import annotations

from collections import Counter
from typing import Optional

import torch
from tqdm import tqdm

from utils.stage2_losses import aux_classification_loss
from utils.text_normalization import render_tokens
from utils.training_helpers.helper_stage2 import (
	_count_mask_per_image,
	_valid_proposal_mask,
	reorder_by_sort_indices,
)
from utils.vocab import VocabManager

__all__ = [
	"_debug_aux_alignment",
	"_finalize_aux_branch_stats",
	"_finalize_decoder_epoch_stats",
	"_init_aux_branch_stats",
	"_init_decoder_epoch_stats",
	"_update_decoder_epoch_stats",
	"_update_refinement_epoch_stats",
	"log_stage2_batch_debug",
]


def _aux_top1_accuracy(aux_logits: torch.Tensor, labels: torch.Tensor, pos_mask: torch.Tensor) -> float:
	valid = pos_mask & labels.ne(-1)
	if not valid.any():
		return 0.0
	pred = aux_logits.argmax(dim=-1)
	return float((pred[valid] == labels[valid]).float().mean().item())


def _ids_to_text(ids: list[int], vocab: VocabManager) -> str:
	return render_tokens(vocab.decode(ids, remove_special=True))


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


def _dominant_token_share(token_ids: list[int]) -> float:
	if len(token_ids) == 0:
		return 0.0
	c = Counter(token_ids)
	return max(c.values()) / len(token_ids)


def _max_repeat_run(token_ids: list[int]) -> int:
	if len(token_ids) == 0:
		return 0
	best = 1
	curr = 1
	for i in range(1, len(token_ids)):
		if token_ids[i] == token_ids[i - 1]:
			curr += 1
			if curr > best:
				best = curr
		else:
			curr = 1
	return best


def _unique_token_ratio(token_ids: list[int]) -> float:
	if len(token_ids) == 0:
		return 0.0
	return len(set(token_ids)) / len(token_ids)


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

	def decode_ids(ids: torch.Tensor, show_id: bool = False):
		out = []
		for x in ids.tolist():
			x = int(x)
			if x < 0:
				out.append("<IGN>" if not show_id else f"{x}|<IGN>")
			else:
				txt = _ids_to_text([x], vocab)
				if show_id:
					out.append(f"{x}|{txt if txt else '?'}")
				else:
					out.append(txt if txt else f"<{x}>")
		return out

	tqdm.write(f"[AUX ALIGN DEBUG] sample={sample_idx}")
	tqdm.write(f"sort_idx[:{limit}]         = {sort_idx[:limit].tolist()}")
	tqdm.write(f"labels_unsorted[:{limit}]  = {labels_unsorted[:limit].tolist()}")
	tqdm.write(f"labels_sorted[:{limit}]    = {labels_sorted[:limit].tolist()}")
	tqdm.write(f"pos_unsorted[:{limit}]     = {pos_unsorted[:limit].tolist()}")
	tqdm.write(f"pos_sorted[:{limit}]       = {pos_sorted[:limit].tolist()}")
	tqdm.write(f"pred_aux_uns[:{limit}]     = {pred_unsorted[:limit].tolist()}")
	if pred_ordered is not None:
		tqdm.write(f"pred_aux_ord[:{limit}]     = {pred_ordered[:limit].tolist()}")
	tqdm.write(f"labels_unsorted (id|char)  = {decode_ids(labels_unsorted[:limit], show_id=True)}")
	tqdm.write(f"labels_sorted (id|char)    = {decode_ids(labels_sorted[:limit], show_id=True)}")
	tqdm.write(f"pred_aux_uns (id|char)     = {decode_ids(pred_unsorted[:limit], show_id=True)}")
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

	ordering_diag = outputs.get("ordering_diagnostics", None)
	if ordering_diag is not None:
		mono = ordering_diag.get("primary_monotonic_fraction", None)
		viol = ordering_diag.get("primary_violation_fraction", None)
		counts = ordering_diag.get("valid_counts", None)
		if mono is not None and viol is not None and counts is not None:
			counts_f = counts.to(dtype=torch.float32)
			stats["ordering_primary_mono_sum"] += float((mono.to(dtype=torch.float32) * counts_f).sum().item())
			stats["ordering_primary_viol_sum"] += float((viol.to(dtype=torch.float32) * counts_f).sum().item())
			stats["ordering_weight_sum"] += float(counts_f.sum().item())

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


def _effective_seq_len_from_prediction(pred_ids: list[int], eos_id: int) -> int:
	if int(eos_id) in pred_ids:
		return int(pred_ids.index(int(eos_id)) + 1)
	return len(pred_ids)


def _action_dist(counter: Counter) -> dict:
	names = {0: "stay", 1: "advance", 2: "stop"}
	total = max(1, int(sum(counter.values())))
	counts = {names[i]: int(counter.get(i, 0)) for i in (0, 1, 2)}
	fractions = {names[i]: float(counter.get(i, 0)) / float(total) for i in (0, 1, 2)}
	return {
		"counts": counts,
		"fractions": fractions,
	}


def _update_action_pointer_epoch_stats(
	stats: dict,
	pred_ids_batch: torch.Tensor,
	action_logits_batch: Optional[torch.Tensor],
	pointer_positions_batch: Optional[torch.Tensor],
	valid_mask_batch: Optional[torch.Tensor],
	action_targets_batch: Optional[torch.Tensor],
	eos_id: int,
) -> None:
	if action_logits_batch is None:
		return

	action_pred_batch = action_logits_batch.detach().argmax(dim=-1).cpu()
	pointer_cpu = pointer_positions_batch.detach().cpu() if pointer_positions_batch is not None else None
	valid_cpu = valid_mask_batch.detach().cpu().bool() if valid_mask_batch is not None else None
	target_cpu = action_targets_batch.detach().cpu() if action_targets_batch is not None else None
	pred_ids_cpu = pred_ids_batch.detach().cpu().tolist()

	if valid_cpu is not None:
		valid_flat = valid_cpu.reshape(-1)
		if bool(valid_flat.any()):
			action_pred_flat = action_pred_batch.reshape(-1)[valid_flat]
			pred_counts = torch.bincount(action_pred_flat, minlength=3)
			for cls_idx, count in enumerate(pred_counts.tolist()):
				stats["action_pred_counts"][int(cls_idx)] += int(count)

			if target_cpu is not None:
				action_target_flat = target_cpu.reshape(-1)[valid_flat]
				target_counts = torch.bincount(action_target_flat, minlength=3)
				for cls_idx, count in enumerate(target_counts.tolist()):
					stats["action_target_counts"][int(cls_idx)] += int(count)
				stats["action_match"] += int((action_pred_flat == action_target_flat).sum().item())
				stats["action_total"] += int(action_target_flat.numel())

	elif target_cpu is not None:
		action_pred_flat = action_pred_batch.reshape(-1)
		pred_counts = torch.bincount(action_pred_flat, minlength=3)
		for cls_idx, count in enumerate(pred_counts.tolist()):
			stats["action_pred_counts"][int(cls_idx)] += int(count)

	bsz, t_dec = action_pred_batch.shape
	for b in range(bsz):
		if valid_cpu is not None:
			seq_mask = valid_cpu[b]
		else:
			seq_len = min(_effective_seq_len_from_prediction(pred_ids_cpu[b], eos_id), t_dec)
			seq_mask = torch.zeros((t_dec,), dtype=torch.bool)
			seq_mask[:seq_len] = True

		if int(seq_mask.sum().item()) == 0:
			continue

		action_pred = action_pred_batch[b][seq_mask]

		if valid_cpu is None:
			pred_counts = torch.bincount(action_pred, minlength=3)
			for cls_idx, count in enumerate(pred_counts.tolist()):
				stats["action_pred_counts"][int(cls_idx)] += int(count)

		if target_cpu is not None and valid_cpu is None:
			action_target = target_cpu[b][seq_mask]
			target_counts = torch.bincount(action_target, minlength=3)
			for cls_idx, count in enumerate(target_counts.tolist()):
				stats["action_target_counts"][int(cls_idx)] += int(count)
			stats["action_match"] += int((action_pred == action_target).sum().item())
			stats["action_total"] += int(action_target.numel())

		if pointer_cpu is None:
			continue

		ptr = pointer_cpu[b][seq_mask].to(dtype=torch.long)
		if ptr.numel() == 0:
			continue

		stats["pointer_seq_count"] += 1
		stats["pointer_start_sum"] += float(ptr[0].item())
		stats["pointer_end_sum"] += float(ptr[-1].item())
		stats["pointer_span_sum"] += float((ptr[-1] - ptr[0]).item())

		if ptr.numel() > 1:
			diffs = ptr[1:] - ptr[:-1]
			stats["pointer_step_advance"] += int(diffs.gt(0).sum().item())
			stats["pointer_step_stay"] += int(diffs.eq(0).sum().item())
			stats["pointer_step_regress"] += int(diffs.lt(0).sum().item())
			stats["pointer_step_total"] += int(diffs.numel())

		if len(stats["pointer_examples"]) < int(stats["pointer_example_limit"]):
			stats["pointer_examples"].append({
				"start": int(ptr[0].item()),
				"end": int(ptr[-1].item()),
				"len": int(ptr.numel()),
				"trace_head": [int(x) for x in ptr[:24].tolist()],
			})


def _init_decoder_epoch_stats(example_limit: int = 3, pointer_example_limit: int = 3) -> dict:
	return {
		"samples": 0,
		"pred_len_sum": 0,
		"gt_len_sum": 0,
		"cer_sum": 0.0,
		"exact": 0,
		"eos_hit": 0,
		"dominant_share_sum": 0.0,
		"max_repeat_run_sum": 0.0,
		"unique_ratio_sum": 0.0,
		"length_ratio_sum": 0.0,
		"pred_token_counter": Counter(),
		"error_pair_counter": Counter(),
		"examples": [],
		"example_limit": int(example_limit),
		"action_pred_counts": Counter(),
		"action_target_counts": Counter(),
		"action_match": 0,
		"action_total": 0,
		"pointer_seq_count": 0,
		"pointer_start_sum": 0.0,
		"pointer_end_sum": 0.0,
		"pointer_span_sum": 0.0,
		"pointer_step_advance": 0,
		"pointer_step_stay": 0,
		"pointer_step_regress": 0,
		"pointer_step_total": 0,
		"pointer_examples": [],
		"pointer_example_limit": int(pointer_example_limit),
	}


def _update_decoder_epoch_stats(
	stats: dict,
	pred_ids_batch: torch.Tensor,
	text_ids_batch: torch.Tensor,
	vocab: VocabManager,
	action_logits_batch: Optional[torch.Tensor] = None,
	pointer_positions_batch: Optional[torch.Tensor] = None,
	valid_mask_batch: Optional[torch.Tensor] = None,
	action_targets_batch: Optional[torch.Tensor] = None,
):
	stats.setdefault("dominant_share_sum", 0.0)
	stats.setdefault("max_repeat_run_sum", 0.0)
	stats.setdefault("unique_ratio_sum", 0.0)
	stats.setdefault("length_ratio_sum", 0.0)

	for pred_ids, gt_ids in zip(pred_ids_batch.detach().cpu().tolist(), text_ids_batch.detach().cpu().tolist()):
		pred_tokens = _strip_special_tokens(pred_ids, vocab.pad_id, vocab.sos_id, vocab.eos_id)
		gt_tokens = _strip_special_tokens(gt_ids, vocab.pad_id, vocab.sos_id, vocab.eos_id)

		pred_text = _ids_to_text(pred_tokens, vocab)
		gt_text = _ids_to_text(gt_tokens, vocab)
		dominant_share = _dominant_token_share(pred_tokens)
		max_run = _max_repeat_run(pred_tokens)
		unique_ratio = _unique_token_ratio(pred_tokens)
		length_ratio = len(pred_tokens) / max(1, len(gt_tokens))

		stats["samples"] += 1
		stats["dominant_share_sum"] += dominant_share
		stats["max_repeat_run_sum"] += max_run
		stats["unique_ratio_sum"] += unique_ratio
		stats["length_ratio_sum"] += length_ratio
		stats["pred_len_sum"] += len(pred_tokens)
		stats["gt_len_sum"] += len(gt_tokens)
		stats["cer_sum"] += _sequence_cer(pred_text, gt_text)
		stats["exact"] += int(pred_text == gt_text)

		if vocab.eos_id in pred_ids:
			stats["eos_hit"] += 1

		stats["pred_token_counter"].update(int(token_id) for token_id in pred_tokens)

		compare_len = min(len(pred_tokens), len(gt_tokens))
		stats["error_pair_counter"].update(
			(int(gt_tok), int(pred_tok))
			for gt_tok, pred_tok in zip(gt_tokens[:compare_len], pred_tokens[:compare_len])
			if gt_tok != pred_tok
		)

		if len(stats["examples"]) < stats["example_limit"]:
			stats["examples"].append({
				"gt": gt_text,
				"pred": pred_text,
				"cer": _sequence_cer(pred_text, gt_text),
				"pred_len": len(pred_tokens),
				"gt_len": len(gt_tokens),
				"eos_hit": bool(vocab.eos_id in pred_ids),
			})

	_update_action_pointer_epoch_stats(
		stats=stats,
		pred_ids_batch=pred_ids_batch,
		action_logits_batch=action_logits_batch,
		pointer_positions_batch=pointer_positions_batch,
		valid_mask_batch=valid_mask_batch,
		action_targets_batch=action_targets_batch,
		eos_id=vocab.eos_id,
	)


def _finalize_decoder_epoch_stats(stats: dict, vocab: VocabManager) -> dict:
	decoder_samples = max(1, stats["samples"])
	pointer_seq_count = max(1, int(stats.get("pointer_seq_count", 0)))
	pointer_step_total = max(1, int(stats.get("pointer_step_total", 0)))
	action_total = max(1, int(stats.get("action_total", 0)))
	return {
		"samples": stats["samples"],
		"eos_hit_fraction": stats["eos_hit"] / decoder_samples,
		"mean_pred_len": stats["pred_len_sum"] / decoder_samples,
		"mean_gt_len": stats["gt_len_sum"] / decoder_samples,
		"mean_cer": stats["cer_sum"] / decoder_samples,
		"exact_match_fraction": stats["exact"] / decoder_samples,
		"mean_dominant_share": stats["dominant_share_sum"] / decoder_samples,
		"mean_max_repeat_run": stats["max_repeat_run_sum"] / decoder_samples,
		"mean_unique_ratio": stats["unique_ratio_sum"] / decoder_samples,
		"mean_length_ratio": stats["length_ratio_sum"] / decoder_samples,
		"action_pred_distribution": _action_dist(stats["action_pred_counts"]),
		"action_target_distribution": _action_dist(stats["action_target_counts"]),
		"action_match_fraction": float(stats.get("action_match", 0)) / float(action_total),
		"pointer_summary": {
			"mean_start": float(stats.get("pointer_start_sum", 0.0)) / float(pointer_seq_count),
			"mean_end": float(stats.get("pointer_end_sum", 0.0)) / float(pointer_seq_count),
			"mean_span": float(stats.get("pointer_span_sum", 0.0)) / float(pointer_seq_count),
			"advance_fraction": float(stats.get("pointer_step_advance", 0)) / float(pointer_step_total),
			"stay_fraction": float(stats.get("pointer_step_stay", 0)) / float(pointer_step_total),
			"regress_fraction": float(stats.get("pointer_step_regress", 0)) / float(pointer_step_total),
		},
		"pointer_examples": stats.get("pointer_examples", []),
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
	proposals_per_image = _count_mask_per_image(roi_mask)
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
