"""Validate Stage 2 hybrid model with concise prediction diagnostics.

This script reuses the validation logic from train_stage2.py and prints:
- loss breakdown (including action loss)
- proposal quality summary
- decoder summary (teacher forcing + free decoding)
- top token predictions and common confusion pairs
- a few GT vs prediction examples

Example:
    python validate_stage2.py \
        --phase B \
        --stage2-ckpt checkpoints/stage2_hybrid_phaseA/stage2_hybrid_best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from config import CHECKPOINT_DIR
from train_stage2 import (
    build_stage2_dataloaders,
    build_stage2_model,
    load_vocab,
    validate_stage2,
)


def _load_stage2_weights(model: torch.nn.Module, stage2_ckpt_path: Path) -> int:
    checkpoint = torch.load(stage2_ckpt_path, map_location="cpu")
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        epoch = int(checkpoint.get("epoch", -1))
    else:
        state_dict = checkpoint
        epoch = -1
    model.load_state_dict(state_dict, strict=True)
    return epoch


def _print_loss_summary(metrics: dict[str, Any]) -> None:
    print(
        "Loss summary | "
        f"val={metrics['val_loss']:.4f}, "
        f"dec={metrics['val_decoder']:.4f}, "
        f"stop={metrics['val_stop']:.4f}, "
        f"action={metrics['val_action']:.4f}, "
        f"box={metrics['val_box']:.4f}, "
        f"delta={metrics['val_delta']:.4f}, "
        f"score={metrics['val_score']:.4f}, "
        f"aux={metrics['val_aux']:.4f}, "
        f"token_acc={metrics['val_token_acc']:.4f}"
    )


def _print_proposal_summary(metrics: dict[str, Any]) -> None:
    prop = metrics["proposal_summary"]
    print(
        "Proposal summary | "
        f"props/img={prop['avg_proposals_per_image']:.2f}, "
        f"pos/img={prop['avg_positives_per_image']:.2f}, "
        f"gt/img={prop['avg_gt_tokens_per_image']:.2f}, "
        f"cov={prop['positive_coverage_ratio']:.3f}, "
        f"uniq_cov={prop['unique_coverage_ratio']:.3f}, "
        f"dup+={prop['duplicate_positive_rate']:.3f}, "
        f"zero/img={prop['images_with_zero_valid_props_ratio']:.3f}, "
        f"prec_proxy={prop['positive_precision_proxy']:.3f}, "
        f"mean IoU+={prop['avg_matched_iou_on_positives']:.3f}, "
        f"aux top1={prop['aux_accuracy_on_positives']:.3f}, "
        f"aux top5={prop['aux_top5_on_positives']:.3f}"
    )


def _print_decoder_summary(metrics: dict[str, Any]) -> None:
    tf = metrics["decoder_summary"]["teacher_forcing"]
    free = metrics["decoder_summary"]["free_decoding"]

    print(
        "Decoder TF | "
        f"EOS hit={tf['eos_hit_fraction']:.3f}, "
        f"len_ratio={tf['mean_length_ratio']:.3f}, "
        f"CER={tf['mean_cer']:.4f}, "
        f"len={tf['mean_pred_len']:.2f}/{tf['mean_gt_len']:.2f}, "
        f"exact={tf['exact_match_fraction']:.3f}, "
        f"action stay/adv/stop="
        f"{tf['action_pred_distribution']['fractions']['stay']:.2f}/"
        f"{tf['action_pred_distribution']['fractions']['advance']:.2f}/"
        f"{tf['action_pred_distribution']['fractions']['stop']:.2f}, "
        f"action_match={tf['action_match_fraction']:.3f}, "
        f"ptr adv/stay/reg="
        f"{tf['pointer_summary']['advance_fraction']:.2f}/"
        f"{tf['pointer_summary']['stay_fraction']:.2f}/"
        f"{tf['pointer_summary']['regress_fraction']:.2f}"
    )
    print(
        "Decoder free | "
        f"EOS hit={free['eos_hit_fraction']:.3f}, "
        f"len_ratio={free['mean_length_ratio']:.3f}, "
        f"CER={free['mean_cer']:.4f}, "
        f"len={free['mean_pred_len']:.2f}/{free['mean_gt_len']:.2f}, "
        f"exact={free['exact_match_fraction']:.3f}, "
        f"action stay/adv/stop="
        f"{free['action_pred_distribution']['fractions']['stay']:.2f}/"
        f"{free['action_pred_distribution']['fractions']['advance']:.2f}/"
        f"{free['action_pred_distribution']['fractions']['stop']:.2f}, "
        f"ptr adv/stay/reg="
        f"{free['pointer_summary']['advance_fraction']:.2f}/"
        f"{free['pointer_summary']['stay_fraction']:.2f}/"
        f"{free['pointer_summary']['regress_fraction']:.2f}"
    )


def _print_aux_predictions(metrics: dict[str, Any]) -> None:
    prop = metrics["proposal_summary"]
    aux_all = prop["aux_summary"]

    for branch_name in ("without_context_encode", "with_context_encode"):
        branch = aux_all[branch_name]
        if not branch.get("available", False):
            continue
        print(
            f"Aux branch={branch_name} | top1={branch['top1']:.3f}, "
            f"top5={branch['top5']:.3f}, n={branch['total']}"
        )
        print(f"Top predictions: {branch['top_predictions']}")
        print(f"Top errors: {branch['top_errors']}")


def _print_examples(metrics: dict[str, Any], max_examples: int) -> None:
    tf_examples = metrics["decoder_summary"]["teacher_forcing"].get("examples", [])
    free_examples = metrics["decoder_summary"]["free_decoding"].get("examples", [])
    free_pointer_examples = metrics["decoder_summary"]["free_decoding"].get("pointer_examples", [])

    if tf_examples:
        print("\nTeacher-forcing examples")
        for idx, ex in enumerate(tf_examples[:max_examples], start=1):
            print(
                f"[{idx}] CER={ex['cer']:.4f} | EOS={ex['eos_hit']} | "
                f"len pred/gt={ex['pred_len']}/{ex['gt_len']}"
            )
            print(f"  GT  : {ex['gt']}")
            print(f"  PRED: {ex['pred']}")

    if free_examples:
        print("\nFree-decoding examples")
        for idx, ex in enumerate(free_examples[:max_examples], start=1):
            print(
                f"[{idx}] CER={ex['cer']:.4f} | EOS={ex['eos_hit']} | "
                f"len pred/gt={ex['pred_len']}/{ex['gt_len']}"
            )
            print(f"  GT  : {ex['gt']}")
            print(f"  PRED: {ex['pred']}")

    if free_pointer_examples:
        print("\nFree-decoding pointer traces")
        for idx, ex in enumerate(free_pointer_examples[:max_examples], start=1):
            print(
                f"[{idx}] ptr start/end={ex['start']}/{ex['end']} | "
                f"len={ex['len']} | trace_head={ex['trace_head']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Stage-2 hybrid model and print key predictions.")
    parser.add_argument("--phase", type=str, default="B", choices=["A", "B"], help="Validation phase profile.")
    parser.add_argument(
        "--detector-ckpt",
        type=Path,
        default=CHECKPOINT_DIR / "stage1_detection" / "detector_best.pt",
        help="Stage-1 detector checkpoint used for model construction.",
    )
    parser.add_argument(
        "--stage2-ckpt",
        type=Path,
        default=CHECKPOINT_DIR / "stage2_hybrid_phase_B" / "stage2_hybrid_best.pt",
        help="Stage-2 checkpoint to validate.",
    )
    parser.add_argument("--max-batches", type=int, default=None, help="Optional limit for quick validation.")
    parser.add_argument("--val-max-decode-len", type=int, default=320, help="Max decode length for free decoding.")
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Optional path to store full validation metrics as JSON.",
    )
    parser.add_argument("--print-examples", type=int, default=3, help="How many decoder examples to print.")
    args = parser.parse_args()

    if not args.detector_ckpt.exists():
        raise FileNotFoundError(f"Detector checkpoint not found: {args.detector_ckpt}")
    if not args.stage2_ckpt.exists():
        raise FileNotFoundError(f"Stage2 checkpoint not found: {args.stage2_ckpt}")

    vocab = load_vocab()
    _, val_loader = build_stage2_dataloaders(vocab)

    model = build_stage2_model(
        detector_ckpt_path=args.detector_ckpt,
        vocab=vocab,
        overrides=None,
    )
    loaded_epoch = _load_stage2_weights(model, args.stage2_ckpt)

    runtime_overrides = {
        "val_max_decode_len": int(args.val_max_decode_len),
    }

    metrics = validate_stage2(
        model=model,
        val_loader=val_loader,
        vocab=vocab,
        phase=args.phase,
        max_batches=args.max_batches,
        runtime_overrides=runtime_overrides,
    )

    print("=" * 72)
    print(f"Stage2 validation done | ckpt={args.stage2_ckpt} | loaded_epoch={loaded_epoch}")
    print("=" * 72)
    _print_loss_summary(metrics)
    _print_proposal_summary(metrics)
    _print_decoder_summary(metrics)
    _print_aux_predictions(metrics)
    _print_examples(metrics, max_examples=max(0, int(args.print_examples)))

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with args.out_json.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"\nSaved metrics JSON to {args.out_json}")


if __name__ == "__main__":
    main()
