from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
import optuna
from optuna.trial import TrialState

from train_stage2 import train_stage2_hybrid
from config import CHECKPOINT_DIR, LR


def compute_length_objective(val_metrics: dict) -> float:
    free = val_metrics["decoder_summary"]["free_decoding"]

    free_cer = float(free["mean_cer"])
    free_len_ratio = float(free["mean_length_ratio"])
    free_eos = float(free["eos_hit_fraction"])
    free_dom = float(free.get("mean_dominant_share", free.get("mean_dominant_token_share", 1.0)))
    free_run = float(free.get("mean_max_repeat_run", 99.0))
    free_uniq = float(free.get("mean_unique_ratio", free.get("mean_unique_token_ratio", 0.0)))

    score = (
        1.5 * free_cer
        + 1.0 * abs(free_len_ratio - 1.0)
        + 0.5 * max(0.0, 0.4 - free_eos)
        + 0.25 * free_dom
        + 0.10 * max(0.0, free_run - 1.5)
        + 0.20 * max(0.0, 0.14 - free_uniq)
    )
    return score


def objective(
    trial: optuna.Trial,
    *,
    num_epochs: int,
    val_max_batches: int,
    trial_root: Path,
    detector_ckpt: Path,
    phase_a_ckpt: Path,
    lr: float,
) -> float:
    model_overrides = {
        "min_steps": trial.suggest_int("min_steps", 48, 144, step=24),
        "stop_threshold": trial.suggest_float("stop_threshold", 0.0, 0.2, step=0.05),
        "decoder_eos_weight": trial.suggest_float("decoder_eos_weight", 1.0, 3.0, step=0.25),
        "lambda_stop": trial.suggest_float("lambda_stop", 0.05, 0.5, step=0.05),
        "repetition_penalty": trial.suggest_float("repetition_penalty", 0.05, 0.25, step=0.05),
        "block_immediate_repeat": trial.suggest_categorical("block_immediate_repeat", [True, False]),
        "bias_scale": trial.suggest_float("bias_scale", 0.30, 0.60, step=0.05),
        "val_max_decode_len": trial.suggest_int("val_max_decode_len", 288, 384, step=32),
    }

    # eigener Trial-Ordner
    trial_dir = trial_root / f"trial_{trial.number:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    result = train_stage2_hybrid(
        detector_ckpt_path=detector_ckpt,
        num_epochs=int(num_epochs),
        lr=float(lr),
        checkpoint_dir=trial_dir,
        phase="B",
        resume_model_ckpt=phase_a_ckpt,
        model_overrides=model_overrides,
        val_max_batches=int(val_max_batches),
    )

    val_metrics = result["best_val_metrics"]
    if val_metrics is None:
        raise RuntimeError("No validation metrics returned from training.")

    free = val_metrics["decoder_summary"]["free_decoding"]

    # hartes Pruning / Bestrafung
    if float(free["eos_hit_fraction"]) < 0.02:
        return 999.0
    if float(free["mean_length_ratio"]) > 2.0:
        return 999.0
    if float(free.get("mean_max_repeat_run", 0.0)) > 3.0:
        return 999.0
    if float(free.get("mean_unique_ratio", free.get("mean_unique_token_ratio", 1.0))) < 0.10:
        return 999.0

    score = compute_length_objective(val_metrics)

    with open(trial_dir / "trial_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "trial_number": trial.number,
                "params": model_overrides,
                "score": score,
                "free_decoding": free,
                "teacher_forcing": val_metrics["decoder_summary"]["teacher_forcing"],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna sweep for Stage-2 Phase-B length control")
    parser.add_argument("--n-trials", type=int, default=30, help="Number of Optuna trials")
    parser.add_argument("--global-max-trials", type=int, default=0, help="Shared cap across parallel workers (0 disables)")
    parser.add_argument("--epochs", type=int, default=3, help="Epochs per trial")
    parser.add_argument("--val-max-batches", type=int, default=20, help="Validation batches per trial")
    parser.add_argument("--timeout", type=int, default=0, help="Timeout in seconds (0 disables)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for TPE sampler")
    parser.add_argument("--study-name", type=str, default="stage2_length_control")
    parser.add_argument("--storage", type=str, default="", help="Optuna storage URL (sqlite:///...)")
    parser.add_argument("--output-dir", type=str, default="", help="Directory for trial checkpoints/summaries")
    parser.add_argument("--detector-ckpt", type=str, default=str(CHECKPOINT_DIR / "stage1_detection" / "detector_best.pt"))
    parser.add_argument("--phase-a-ckpt", type=str, default=str(CHECKPOINT_DIR / "stage2_hybrid_phaseA" / "stage2_hybrid_best.pt"))
    parser.add_argument("--lr", type=float, default=float(LR), help="Learning rate override")
    return parser.parse_args()


def main():
    args = parse_args()

    detector_ckpt = Path(args.detector_ckpt)
    if not detector_ckpt.exists():
        raise FileNotFoundError(f"Detector checkpoint not found: {detector_ckpt}")

    phase_a_ckpt = Path(args.phase_a_ckpt)
    if not phase_a_ckpt.exists():
        raise FileNotFoundError(f"Phase-A checkpoint not found: {phase_a_ckpt}")

    output_dir = Path(args.output_dir) if args.output_dir else (CHECKPOINT_DIR / "optuna_stage2_length")
    output_dir.mkdir(parents=True, exist_ok=True)

    storage = args.storage
    if not storage:
        storage = f"sqlite:///{(output_dir / 'optuna_stage2_length.db').as_posix()}"

    def _objective(trial: optuna.Trial) -> float:
        return objective(
            trial,
            num_epochs=args.epochs,
            val_max_batches=args.val_max_batches,
            trial_root=output_dir,
            detector_ckpt=detector_ckpt,
            phase_a_ckpt=phase_a_ckpt,
            lr=args.lr,
        )

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1, interval_steps=1)

    study = None
    for _attempt in range(10):
        try:
            study = optuna.create_study(
                study_name=args.study_name,
                direction="minimize",
                sampler=sampler,
                pruner=pruner,
                storage=storage,
                load_if_exists=True,
            )
            break
        except Exception as exc:
            if "already exists" in str(exc):
                time.sleep(random.uniform(1.0, 5.0))
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

    study.optimize(_objective, n_trials=effective_n_trials, timeout=timeout, callbacks=optimize_callbacks)

    print("Best trial:")
    print(study.best_trial.number)
    print(study.best_trial.value)
    print(study.best_trial.params)


if __name__ == "__main__":
    main()