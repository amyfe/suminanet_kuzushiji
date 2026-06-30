from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path

import optuna
from optuna.trial import TrialState

import train_stage2_warmup
from config import CHECKPOINT_DIR, NUM_EPOCHS


def _resolve_detector_checkpoint() -> Path:
    best_ckpt = CHECKPOINT_DIR / "stage1_detection" / "detector_best.pt"
    last_ckpt = CHECKPOINT_DIR / "stage1_detection" / f"detector_epoch{NUM_EPOCHS}.pt"

    detector_ckpt = best_ckpt if best_ckpt.exists() else last_ckpt
    if not detector_ckpt.exists():
        raise FileNotFoundError(
            "No Stage-1 detector checkpoint found. Expected one of:\n"
            f"  - {best_ckpt}\n"
            f"  - {last_ckpt}"
        )
    return detector_ckpt


def objective(trial: optuna.Trial, args: argparse.Namespace) -> float:
    detector_ckpt = _resolve_detector_checkpoint()

    det_score_thresh = trial.suggest_float("det_score_thresh", 0.15, 0.40)
    det_top_k = trial.suggest_int("det_top_k", 320, 896, step=64)
    det_nms_iou = trial.suggest_float("det_nms_iou", 0.50, 0.80)
    det_min_box_size = trial.suggest_float("det_min_box_size", 0.5, 2.0)

    token_hidden_dim = trial.suggest_categorical("token_hidden_dim", [256, 384, 512])
    context_hidden_dim = trial.suggest_categorical("context_hidden_dim", [256, 384, 512])
    context_num_layers = trial.suggest_categorical("context_num_layers", [1, 2])

    model_overrides = {
        "det_score_thresh": det_score_thresh,
        "det_top_k": det_top_k,
        "det_nms_iou": det_nms_iou,
        "det_min_box_size": det_min_box_size,
        "token_hidden_dim": token_hidden_dim,
        "context_hidden_dim": context_hidden_dim,
        "context_num_layers": context_num_layers,
    }

    trial_ckpt_dir = Path(args.output_dir) / f"trial_{trial.number}"
    trial_ckpt_dir.mkdir(parents=True, exist_ok=True)

    run = train_stage2_warmup.train_stage2_hybrid(
        detector_ckpt_path=detector_ckpt,
        num_epochs=args.epochs,
        lr=args.lr,
        checkpoint_dir=trial_ckpt_dir,
        phase="A",
        resume_model_ckpt=None,
        model_overrides=model_overrides,
        val_max_batches=args.val_max_batches,
    )

    best_metrics = run.get("best_val_metrics") or {}
    prop = best_metrics.get("proposal_summary") or {}
    aux_ctx = (((prop.get("aux_summary") or {}).get("with_context_encode") or {}))

    unique_cov = float(prop.get("unique_coverage_ratio", 0.0))
    duplicate_rate = float(prop.get("duplicate_positive_rate", 0.0))
    neg_per_img = float(prop.get("avg_negatives_per_image", 0.0))
    mean_iou_pos = float(prop.get("avg_matched_iou_on_positives", 0.0))
    aux_ctx_top1 = float(aux_ctx.get("top1", 0.0))

    # Tradeoff objective: prioritize unique recall, then quality, lightly penalize duplicate/noisy proposals.
    objective_value = (
        unique_cov
        + args.weight_iou * mean_iou_pos
        + args.weight_aux_ctx * aux_ctx_top1
        - args.weight_duplicate * duplicate_rate
        - args.weight_negative * neg_per_img
    )

    trial.set_user_attr("val_loss", run.get("best_val_loss"))
    trial.set_user_attr("unique_coverage_ratio", unique_cov)
    trial.set_user_attr("positive_coverage_ratio", float(prop.get("positive_coverage_ratio", 0.0)))
    trial.set_user_attr("duplicate_positive_rate", duplicate_rate)
    trial.set_user_attr("avg_negatives_per_image", neg_per_img)
    trial.set_user_attr("avg_matched_iou_on_positives", mean_iou_pos)
    trial.set_user_attr("aux_with_context_top1", aux_ctx_top1)

    if args.cleanup_trial_dirs:
        shutil.rmtree(trial_ckpt_dir, ignore_errors=True)

    return float(objective_value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna search for Stage-2 A proposal + architecture tuning")
    parser.add_argument("--epochs", type=int, default=2, help="A epochs per trial")
    parser.add_argument("--n-trials", type=int, default=20, help="Number of Optuna trials")
    parser.add_argument("--global-max-trials", type=int, default=0, help="Shared cap across parallel workers (0 disables)")
    parser.add_argument("--timeout", type=int, default=0, help="Timeout in seconds (0 disables)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampler")
    parser.add_argument("--val-max-batches", type=int, default=80, help="Validation batches per epoch (0 => full)")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate; default uses config LR")

    parser.add_argument("--weight-iou", type=float, default=0.05, help="Objective weight for IoU quality")
    parser.add_argument("--weight-aux-ctx", type=float, default=0.02, help="Objective weight for context-aux top1")
    parser.add_argument("--weight-duplicate", type=float, default=0.15, help="Penalty weight for duplicate positives")
    parser.add_argument("--weight-negative", type=float, default=0.002, help="Penalty weight for negatives per image")

    parser.add_argument("--study-name", type=str, default="stage2_a_optuna")
    parser.add_argument("--storage", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--sampler", type=str, default="tpe", choices=["tpe", "random"])
    parser.add_argument("--cleanup-trial-dirs", action="store_true", help="Delete trial checkpoint dirs after metric extraction")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.output_dir:
        args.output_dir = str(CHECKPOINT_DIR / "optuna_stage2_a")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.storage:
        args.storage = f"sqlite:///{(out_dir / 'optuna_stage2_a.db').as_posix()}"

    if args.storage.startswith("sqlite:///"):
        Path(args.storage[len("sqlite:///"):]).parent.mkdir(parents=True, exist_ok=True)

    if args.val_max_batches <= 0:
        args.val_max_batches = None

    if args.sampler == "random":
        sampler = optuna.samplers.RandomSampler(seed=args.seed)
    else:
        sampler = optuna.samplers.TPESampler(seed=args.seed)

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1, interval_steps=1)

    for _attempt in range(10):
        try:
            study = optuna.create_study(
                study_name=args.study_name,
                storage=args.storage,
                load_if_exists=True,
                direction="maximize",
                sampler=sampler,
                pruner=pruner,
            )
            break
        except Exception as exc:
            if "already exists" in str(exc):
                time.sleep(random.uniform(1, 5))
            else:
                raise
    else:
        raise RuntimeError("Failed to create/load Optuna study after 10 retries")

    timeout = None if args.timeout <= 0 else args.timeout
    optimize_callbacks = []
    effective_n_trials = args.n_trials
    if args.global_max_trials > 0:
        optimize_callbacks.append(
            optuna.study.MaxTrialsCallback(
                n_trials=args.global_max_trials,
                states=(TrialState.COMPLETE, TrialState.PRUNED),
            )
        )
        effective_n_trials = None

    study.optimize(
        lambda trial: objective(trial, args),
        n_trials=effective_n_trials,
        timeout=timeout,
        callbacks=optimize_callbacks,
    )

    summary = {
        "study_name": study.study_name,
        "best_trial": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
        "best_trial_user_attrs": study.best_trial.user_attrs,
    }
    summary_path = out_dir / "best_params.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 70)
    print("Stage-2 Optuna search complete")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best objective: {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")
    print(f"Best attrs: {study.best_trial.user_attrs}")
    print(f"Saved summary: {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
