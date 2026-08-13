"""Optuna-based hyperparameter search for SuminaNet Recognizer.

Tunes loss weights, learning-rate, and the detector proposal threshold that
were previously round-number guesses. Each trial runs N short epochs so the
search stays tractable despite SuminaNet's long full-training time.

Objective: maximise  top1 * (1 - CER)  on the validation split.

Usage:
    python train_stage2_suminanet_optuna.py
    python train_stage2_suminanet_optuna.py --epochs 8 --n-trials 30
    python train_stage2_suminanet_optuna.py --warmup-ckpt checkpoints/suminanet_warmup/warmup_best.pt

EVALUATION DISCIPLINE: The Optuna objective uses the val split.  Val == test in
this repo.  Do not add new hyperparameter ranges based on what you observe in
test numbers after a full training run — that defeats the held-out set purpose.
Only use Optuna to explore genuinely new hyperparameter hypotheses.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

import optuna
import torch
import torch.optim as optim
from optuna.importance import get_param_importances
from optuna.trial import TrialState
from optuna.visualization.matplotlib import (
    plot_contour,
    plot_parallel_coordinate,
    plot_param_importances,
    plot_slice,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import train_stage2_suminanet as train_module
from gpu_cleanup import release_cuda_memory
from config import (
    CHECKPOINT_DIR,
    DEVICE,
    GRAD_CLIP,
    SUMINANET_CHECKPOINT_DIR,
    SUMINANET_GRAD_ACCUM_STEPS,
    SUMINANET_LR_ETA_MIN,
    NUM_WORKERS,
    USE_MIXED_PRECISION,
)


def objective(trial: optuna.Trial, args: argparse.Namespace) -> float:
    # ----------------------------------------------------------------
    # Sample hyperparameters
    # ----------------------------------------------------------------
    lr               = trial.suggest_float("lr",               1e-5, 5e-4, log=True)
    weight_decay     = trial.suggest_float("weight_decay",     1e-5, 5e-3, log=True)
    focal_gamma      = trial.suggest_float("focal_gamma",      0.5,  3.0)
    lambda_delta     = trial.suggest_float("lambda_delta",     0.2,  2.0)
    lambda_score     = trial.suggest_float("lambda_score",     0.05, 0.5)
    lambda_script    = trial.suggest_float("lambda_script",    0.05, 0.5)
    bg_base          = trial.suggest_float("bg_base",          0.05, 0.3)
    bg_ratio         = trial.suggest_float("bg_ratio",         2.0,  8.0)
    hard_neg_w       = trial.suggest_float("hard_neg_w",       1.0,  3.0)
    det_score_thresh = trial.suggest_float("det_score_thresh", 0.15, 0.45)

    # BG weight fusion: STRONG must always exceed BG — encode as (base, ratio).
    bg_strong = bg_base * bg_ratio

    # ----------------------------------------------------------------
    # Patch module-level loss hyperparameters so train_epoch / compute_suminanet_loss
    # pick up the trial values via Python's module-global lookup.
    # ----------------------------------------------------------------
    train_module.SUMINANET_FOCAL_GAMMA       = focal_gamma
    train_module.SUMINANET_LAMBDA_DELTA      = lambda_delta
    train_module.SUMINANET_LAMBDA_SCORE      = lambda_score
    train_module.SUMINANET_LAMBDA_SCRIPT     = lambda_script
    train_module.SUMINANET_BG_WEIGHT         = bg_base
    train_module.SUMINANET_STRONG_BG_WEIGHT  = bg_strong
    train_module.SUMINANET_HARD_NEG_WEIGHT   = hard_neg_w
    train_module.SUMINANET_DET_SCORE_THRESH  = det_score_thresh

    # ----------------------------------------------------------------
    # Build data, model, optimiser
    # ----------------------------------------------------------------
    # Predeclared so the `finally` block below can always `del` them, even if
    # an exception fires partway through construction.
    train_loader = val_loader = vocab_weights = None
    model = trainable_params = optimizer = scheduler = scaler = None
    try:
        vocab        = train_module.load_vocab()
        train_loader, val_loader = train_module.build_dataloaders(vocab)
        vocab_weights = train_module.build_kanji_vocab_weights(vocab, kanji_weight=1.8).to(DEVICE)

        model = train_module.build_suminanet_model(vocab, warmup_ckpt=args.warmup_ckpt)
        train_module.set_trainable_modules(model)
        trainable_params = train_module.get_trainable_params(model)

        optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=SUMINANET_LR_ETA_MIN,
        )
        scaler = torch.cuda.amp.GradScaler(enabled=USE_MIXED_PRECISION)

        # ----------------------------------------------------------------
        # Short trial training loop
        # ----------------------------------------------------------------
        best_score = -float("inf")
        patience   = 0
        val_metrics = {}

        for epoch in range(1, args.epochs + 1):
            train_module.train_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                scaler=scaler,
                vocab=vocab,
                epoch=epoch,
                grad_accum_steps=SUMINANET_GRAD_ACCUM_STEPS,
                hard_neg_pairs=None,
                vocab_weights=vocab_weights,
                sam2_dir=None,
            )
            scheduler.step()

            val_metrics = train_module.validate_suminanet(
                model=model,
                val_loader=val_loader,
                vocab=vocab,
                max_batches=args.val_max_batches or None,
                vocab_weights=vocab_weights,
            )

            score = train_module.select_model_score(val_metrics)
            trial.report(score, epoch)

            if trial.should_prune():
                raise optuna.TrialPruned()

            if score > best_score:
                best_score = score
                patience   = 0
            else:
                patience  += 1
                if patience >= args.early_stop:
                    break

        trial.set_user_attr("best_score",    best_score)
        trial.set_user_attr("top1_acc",      val_metrics.get("top1_acc", 0.0))
        trial.set_user_attr("assembled_cer", val_metrics.get("assembled_cer", 1.0))
        trial.set_user_attr("bg_strong",     bg_strong)

        return float(best_score)
    finally:
        # train_loader/val_loader use persistent_workers=True; deleting them
        # here also tears down their worker processes deterministically,
        # instead of waiting for this frame to be reclaimed (which a
        # propagating TrialPruned exception could otherwise delay).
        del train_loader, val_loader, vocab_weights
        del model, trainable_params, optimizer, scheduler, scaler
        release_cuda_memory()


def _save_matplotlib_plot(plot_obj, output_path: Path) -> None:
    figure = getattr(plot_obj, "figure", None)
    if figure is None and hasattr(plot_obj, "get_figure"):
        figure = plot_obj.get_figure()
    if figure is None:
        figure = plot_obj
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    try:
        import matplotlib.pyplot as plt
        plt.close(figure)
    except Exception:
        pass


def generate_optuna_png_plots(
    study: optuna.study.Study,
    output_dir: "str | Path",
    top_n: int = 4,
    max_contour_pairs: int = 3,
) -> list:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list = []

    try:
        importance_ax = plot_param_importances(study)
        p = output_dir / "param_importances.png"
        _save_matplotlib_plot(importance_ax, p)
        saved_paths.append(p)
    except Exception as exc:
        print(f"WARNING: could not create param importance plot: {exc}")

    important_params = list(get_param_importances(study).keys())[:max(0, top_n)]
    if important_params:
        try:
            slice_ax = plot_slice(study, params=important_params)
            p = output_dir / "param_slice_top.png"
            _save_matplotlib_plot(slice_ax, p)
            saved_paths.append(p)
        except Exception as exc:
            print(f"WARNING: could not create slice plot: {exc}")

        try:
            parallel_ax = plot_parallel_coordinate(study, params=important_params)
            p = output_dir / "parallel_coordinate_top.png"
            _save_matplotlib_plot(parallel_ax, p)
            saved_paths.append(p)
        except Exception as exc:
            print(f"WARNING: could not create parallel coordinate plot: {exc}")

    contour_params = important_params[:max(0, min(len(important_params), max_contour_pairs + 1))]
    if len(contour_params) >= 2:
        pair_count = 0
        for i in range(len(contour_params)):
            for j in range(i + 1, len(contour_params)):
                if pair_count >= max_contour_pairs:
                    break
                try:
                    contour_ax = plot_contour(study, params=[contour_params[i], contour_params[j]])
                    p = output_dir / f"contour_{contour_params[i]}_vs_{contour_params[j]}.png"
                    _save_matplotlib_plot(contour_ax, p)
                    saved_paths.append(p)
                    pair_count += 1
                except Exception as exc:
                    print(f"WARNING: contour plot failed: {exc}")
            if pair_count >= max_contour_pairs:
                break

    return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optuna search for SuminaNet loss weights and key thresholds"
    )
    parser.add_argument("--epochs",          type=int,   default=8,
                        help="Training epochs per trial (keep short; default 8)")
    parser.add_argument("--n-trials",        type=int,   default=30,
                        help="Number of Optuna trials")
    parser.add_argument("--global-max-trials", type=int, default=0,
                        help="Shared cap across parallel workers (0 disables)")
    parser.add_argument("--timeout",         type=int,   default=0,
                        help="Timeout in seconds (0 disables)")
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--val-max-batches", type=int,   default=80,
                        help="Validation batches per epoch (0 => full val set)")
    parser.add_argument("--early-stop",      type=int,   default=2,
                        help="Intra-trial early-stop patience (epochs without improvement)")
    parser.add_argument("--warmup-ckpt",     type=str,   default=None,
                        help="Warmup checkpoint for ROI pipeline warm-start (recommended)")

    parser.add_argument("--study-name",  type=str, default="suminanet_optuna")
    parser.add_argument("--storage",     type=str, default="")
    parser.add_argument("--output-dir",  type=str, default="")
    parser.add_argument("--sampler",     type=str, default="tpe",
                        choices=["tpe", "random"])
    parser.add_argument("--cleanup-trial-dirs", action="store_true",
                        help="Delete per-trial checkpoint dirs after metric extraction")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.output_dir:
        args.output_dir = str(CHECKPOINT_DIR / "optuna_suminanet")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.storage:
        args.storage = f"sqlite:///{(out_dir / 'optuna_suminanet.db').as_posix()}"

    if args.storage.startswith("sqlite:///"):
        Path(args.storage[len("sqlite:///"):]).parent.mkdir(parents=True, exist_ok=True)

    if args.val_max_batches <= 0:
        args.val_max_batches = None

    if args.sampler == "random":
        sampler = optuna.samplers.RandomSampler(seed=args.seed)
    else:
        sampler = optuna.samplers.TPESampler(seed=args.seed)

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2, interval_steps=1)

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
        "study_name":            study.study_name,
        "best_trial":            study.best_trial.number,
        "best_value":            study.best_value,
        "best_params":           study.best_params,
        "n_trials":              len(study.trials),
        "best_trial_user_attrs": study.best_trial.user_attrs,
    }
    summary_path = out_dir / "best_params.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Export PNG plots (non-fatal if optuna visualization is unavailable)
    try:
        plot_paths = generate_optuna_png_plots(study, out_dir)
        if plot_paths:
            print(f"Saved {len(plot_paths)} Optuna plot(s) to {out_dir}")
    except Exception as exc:
        print(f"WARNING: Optuna plot export failed (non-fatal): {exc}")

    print("=" * 70)
    print("SuminaNet Optuna search complete")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best objective (top1 × (1 - CER)): {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")
    print(f"Saved summary: {summary_path}")
    print("=" * 70)

    # Print config-ready lines for the best params
    p = study.best_params
    print("\n# Paste these into config.py:")
    print(f"SUMINANET_LR                = {p['lr']}")
    print(f"SUMINANET_WEIGHT_DECAY      = {p['weight_decay']}")
    print(f"SUMINANET_FOCAL_GAMMA       = {p['focal_gamma']}")
    print(f"SUMINANET_LAMBDA_DELTA      = {p['lambda_delta']}")
    print(f"SUMINANET_LAMBDA_SCORE      = {p['lambda_score']}")
    print(f"SUMINANET_LAMBDA_SCRIPT     = {p['lambda_script']}")
    print(f"SUMINANET_BG_WEIGHT         = {p['bg_base']}")
    print(f"SUMINANET_STRONG_BG_WEIGHT  = {p['bg_base'] * p['bg_ratio']:.6f}  # bg_base={p['bg_base']:.4f} × bg_ratio={p['bg_ratio']:.4f}")
    print(f"SUMINANET_HARD_NEG_WEIGHT   = {p['hard_neg_w']}")
    print(f"SUMINANET_DET_SCORE_THRESH  = {p['det_score_thresh']}")


if __name__ == "__main__":
    main()
