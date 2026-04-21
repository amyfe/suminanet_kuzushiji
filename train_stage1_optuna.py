"""Optuna-based hyperparameter search for Stage 1 detector training.

This script tunes the detection stage directly (UNet + DetectorHead) and keeps
your existing training pipeline untouched.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import optuna
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from optuna.trial import TrialState

from config import (
    CHECKPOINT_DIR,
    DATA_DIR,
    DEVICE,
    DET_MIN_BOX_SIZE,
    DET_NMS_IOU,
    DET_SCORE_THRESH,
    DET_TOP_K,
    GRADIENT_ACCUMULATION_STEPS,
    IMAGE_SIZE,
    NUM_WORKERS,
    USE_MIXED_PRECISION,
)
from model.kuronet import DetectorHead, UNet
from utils import KuzushijiDataset
from utils.detection_utils import build_detection_targets
from utils.focal_loss import focal_loss_heatmap
from utils.training_helpers.helper_stage1 import collate_fn, masked_bbox_smoothl1_loss
from utils.vocab import VocabManager
from validate_stage1 import compute_detection_metrics, extract_boxes_from_heatmap


AMP_ENABLED = USE_MIXED_PRECISION and str(DEVICE).startswith("cuda")
AUTOCAST_DEVICE = "cuda" if str(DEVICE).startswith("cuda") else "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dataloaders(vocab: VocabManager, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    pad_id = vocab.pad_id
    train_dataset = KuzushijiDataset(
        Path(DATA_DIR), vocab=vocab, use_sequences=True, resize=IMAGE_SIZE, split="train"
    )
    val_dataset = KuzushijiDataset(
        Path(DATA_DIR), vocab=vocab, use_sequences=True, resize=IMAGE_SIZE, split="val"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=lambda b: collate_fn(b, pad_id),
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=lambda b: collate_fn(b, pad_id),
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=NUM_WORKERS > 0,
    )
    return train_loader, val_loader


@torch.no_grad()
def validate_detector(
    unet: UNet,
    detector: DetectorHead,
    dataloader: DataLoader,
    heatmap_sigma: float,
    focal_alpha: float,
    focal_gamma: float,
    pos_weight: float,
    bbox_weight: float,
) -> float:
    unet.eval()
    detector.eval()

    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        images = batch["image"].to(DEVICE)
        boxes = [
            b.to(DEVICE) if b.numel() > 0 else torch.empty((0, 4), device=DEVICE)
            for b in batch.get("boxes", [])
        ]
        labels = [
            l.to(DEVICE)
            if (l is not None and l.numel() > 0)
            else torch.empty((0,), dtype=torch.long, device=DEVICE)
            for l in batch.get("labels", [])
        ]

        with torch.amp.autocast(device_type=AUTOCAST_DEVICE, enabled=AMP_ENABLED):
            features = unet(images)
            _, _, hf, wf = features.shape
            gt_heat, gt_bbox, gt_bbox_mask, _ = build_detection_targets(
                boxes,
                labels,
                output_size=(hf, wf),
                image_size=tuple(images.shape[-2:]),
                device=DEVICE,
                sigma=heatmap_sigma,
                bbox_radius=0,
            )

            outputs = detector(features)
            loss_heat = focal_loss_heatmap(
                outputs["heatmap"],
                gt_heat,
                alpha=focal_alpha,
                gamma=focal_gamma,
                pos_weight=pos_weight,
            )
            loss_bbox = masked_bbox_smoothl1_loss(outputs["bbox"], gt_bbox, gt_bbox_mask)
            loss = loss_heat + bbox_weight * loss_bbox

        total_loss += float(loss.item())
        num_batches += 1

    return total_loss / max(1, num_batches)


def objective(trial: optuna.Trial, args: argparse.Namespace) -> float:
    set_seed(args.seed + trial.number)

    # Search space from your candidate list.
    lr = trial.suggest_float("lr", 1e-5, 2e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [2, 4])
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    focal_alpha = trial.suggest_float("focal_alpha", 0.15, 0.45)
    focal_gamma = trial.suggest_float("focal_gamma", 1.0, 3.0)
    pos_weight = trial.suggest_float("pos_weight", 2.0, 12.0)
    bbox_weight = trial.suggest_float("bbox_weight", 0.1, 0.8)
    heatmap_sigma = trial.suggest_float("heatmap_sigma", 0.6, 1.4)

    ann_files = sorted(list((Path(DATA_DIR) / "annotations").glob("*.json")))
    if not ann_files:
        raise FileNotFoundError(f"No annotation files found in {Path(DATA_DIR) / 'annotations'}")

    vocab = VocabManager.from_annotations(ann_files)
    train_loader, val_loader = make_dataloaders(vocab, batch_size)

    unet = UNet(in_channels=3, base_features=32).to(DEVICE)
    detector = DetectorHead(
        in_ch=32,
        num_classes=vocab.vocab_size,
        dropout_rate=dropout_rate,
        predict_classes=False,
    ).to(DEVICE)

    optimizer = optim.Adam(
        list(unet.parameters()) + list(detector.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    scaler = torch.amp.GradScaler(device="cuda") if AMP_ENABLED else None

    best_val = float("inf")

    for epoch in range(args.epochs):
        unet.train()
        detector.train()

        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(
            train_loader,
            desc=f"Trial {trial.number} Epoch {epoch + 1}/{args.epochs}",
            leave=False,
            mininterval=args.pbar_mininterval,
            file=sys.stderr,
            disable=args.no_progress_bar,
        )

        for step, batch in enumerate(pbar):
            images = batch["image"].to(DEVICE)
            boxes = [
                b.to(DEVICE) if b.numel() > 0 else torch.empty((0, 4), device=DEVICE)
                for b in batch.get("boxes", [])
            ]
            labels = [
                l.to(DEVICE)
                if l.numel() > 0
                else torch.empty((0,), dtype=torch.long, device=DEVICE)
                for l in batch.get("labels", [])
            ]

            with torch.amp.autocast(device_type=AUTOCAST_DEVICE, enabled=AMP_ENABLED):
                features = unet(images)
                _, _, hf, wf = features.shape
                gt_heat, gt_bbox, gt_bbox_mask, _ = build_detection_targets(
                    boxes,
                    labels,
                    output_size=(hf, wf),
                    image_size=tuple(images.shape[-2:]),
                    device=DEVICE,
                    sigma=heatmap_sigma,
                    bbox_radius=0,
                )

                outputs = detector(features)
                loss_heat = focal_loss_heatmap(
                    outputs["heatmap"],
                    gt_heat,
                    alpha=focal_alpha,
                    gamma=focal_gamma,
                    pos_weight=pos_weight,
                )
                loss_bbox = masked_bbox_smoothl1_loss(outputs["bbox"], gt_bbox, gt_bbox_mask)
                loss = loss_heat + bbox_weight * loss_bbox
                loss = loss / GRADIENT_ACCUMULATION_STEPS

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

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

            pbar.set_postfix({"loss": float(loss.item() * GRADIENT_ACCUMULATION_STEPS)})

        # Flush remainder when number of batches is not divisible by accumulation steps.
        if len(train_loader) % GRADIENT_ACCUMULATION_STEPS != 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(list(unet.parameters()) + list(detector.parameters()), 1.0)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        val_loss = validate_detector(
            unet=unet,
            detector=detector,
            dataloader=val_loader,
            heatmap_sigma=heatmap_sigma,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            pos_weight=pos_weight,
            bbox_weight=bbox_weight,
        )
        best_val = min(best_val, val_loss)

        trial.report(val_loss, step=epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # Compute detection F1 on val set for recall-focused training mode.
    # Uses fixed inference params so trials are comparable on the same metric.
    if getattr(args, "optimize_f1", False):
        unet.eval()
        detector.eval()
        all_gt_boxes: list = []
        all_pred_boxes: list = []
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(DEVICE)
                gt_boxes = batch["boxes"][0].cpu().numpy().tolist()
                features = unet(images)
                outputs = detector(features)
                _, _, hf, wf = features.shape
                pred_boxes, _, _ = extract_boxes_from_heatmap(
                    heatmap_probs=torch.sigmoid(outputs["heatmap"]),
                    bbox_reg=outputs["bbox"],
                    confidence_thresh=DET_SCORE_THRESH,
                    output_size=(hf, wf),
                    image_size=IMAGE_SIZE,
                    top_k=DET_TOP_K,
                    nms_iou=DET_NMS_IOU,
                    min_box_size=DET_MIN_BOX_SIZE,
                    debug=False,
                )
                all_gt_boxes.append(gt_boxes)
                all_pred_boxes.append(pred_boxes)
        det_metrics = compute_detection_metrics(all_gt_boxes, all_pred_boxes, iou_threshold=0.5)
        return float(det_metrics["f1"])

    if args.save_checkpoints:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = out_dir / f"trial_{trial.number}_best.pt"
        torch.save(
            {
                "trial": trial.number,
                "best_val": best_val,
                "params": trial.params,
                "unet_state_dict": unet.state_dict(),
                "detector_state_dict": detector.state_dict(),
            },
            checkpoint_path,
        )

    return best_val


@torch.no_grad()
def objective_validation(trial: optuna.Trial, args: argparse.Namespace) -> float:
    # Extended ranges vs original: confidence down to 0.05 (was 0.2) to allow
    # high-recall operation; top_k up to 1500 (was 700) to match det_top_k=896+.
    confidence = trial.suggest_float("confidence", 0.05, 0.50)
    top_k = trial.suggest_int("top_k", 300, 1500, step=100)
    nms_iou = trial.suggest_float("nms_iou", 0.30, 0.70)
    min_box_size = trial.suggest_float("min_box_size", 1.0, 6.0)

    ann_files = sorted(list((Path(DATA_DIR) / "annotations").glob("*.json")))
    if not ann_files:
        raise FileNotFoundError(f"No annotation files found in {Path(DATA_DIR) / 'annotations'}")

    vocab = VocabManager.from_annotations(ann_files)
    dataset = KuzushijiDataset(
        Path(DATA_DIR),
        vocab=vocab,
        use_sequences=False,
        resize=IMAGE_SIZE,
        split=args.val_split,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    unet = UNet(in_channels=3, base_features=32).to(DEVICE)
    detector = DetectorHead(in_ch=32, num_classes=vocab.vocab_size, predict_classes=False).to(DEVICE)

    ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    unet.load_state_dict(ckpt["unet_state_dict"], strict=True)
    detector.load_state_dict(ckpt["detector_state_dict"], strict=True)
    unet.eval()
    detector.eval()

    all_gt_boxes = []
    all_pred_boxes = []

    max_samples = None if args.val_num_samples == 0 else args.val_num_samples
    for idx, batch in enumerate(loader):
        if max_samples is not None and idx >= max_samples:
            break

        images = batch["image"].to(DEVICE)
        gt_boxes = batch["boxes"][0].cpu().numpy().tolist()

        features = unet(images)
        outputs = detector(features)

        heatmap_probs = torch.sigmoid(outputs["heatmap"])
        bbox_reg = outputs["bbox"]
        _, _, hf, wf = features.shape

        pred_boxes, _, _ = extract_boxes_from_heatmap(
            heatmap_probs=heatmap_probs,
            bbox_reg=bbox_reg,
            confidence_thresh=confidence,
            output_size=(hf, wf),
            image_size=IMAGE_SIZE,
            top_k=top_k,
            nms_iou=nms_iou,
            min_box_size=min_box_size,
            debug=False,
        )
        all_gt_boxes.append(gt_boxes)
        all_pred_boxes.append(pred_boxes)

    metrics = compute_detection_metrics(all_gt_boxes, all_pred_boxes, iou_threshold=args.val_iou_thr)
    metric_value = float(metrics[args.val_metric])
    trial.set_user_attr("tp", int(metrics["tp"]))
    trial.set_user_attr("fp", int(metrics["fp"]))
    trial.set_user_attr("fn", int(metrics["fn"]))
    trial.report(metric_value, step=0)
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()
    return metric_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna search for Stage 1 detector training/validation")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "val"], help="train: tune stage1 training hparams, val: tune stage1 decode/validation params")
    parser.add_argument("--epochs", type=int, default=6, help="Epochs per trial")
    parser.add_argument("--n-trials", type=int, default=30, help="Number of Optuna trials")
    parser.add_argument(
        "--global-max-trials",
        type=int,
        default=0,
        help="Global trial cap across parallel workers (0 disables)",
    )
    parser.add_argument("--timeout", type=int, default=0, help="Timeout in seconds (0 = disabled)")
    parser.add_argument(
        "--pbar-mininterval",
        type=float,
        default=30.0,
        help="Minimum seconds between progress bar refreshes (reduces log size)",
    )
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="Disable tqdm progress bars entirely",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed base")
    parser.add_argument("--study-name", type=str, default="", help="Optuna study name (empty => mode-specific default)")
    parser.add_argument(
        "--storage",
        type=str,
        default="",
        help="Optuna storage URL (e.g., sqlite:///checkpoints/optuna_stage1/optuna.db)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Where to save trial artifacts",
    )
    parser.add_argument(
        "--save-checkpoints",
        action="store_true",
        help="Save a small best-model checkpoint for each completed trial",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="tpe",
        choices=["tpe", "random"],
        help="Optuna sampler",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(CHECKPOINT_DIR / "stage1_detection" / "detector_best.pt"),
        help="Detector checkpoint path (used in --mode val)",
    )
    parser.add_argument("--optimize-f1", action="store_true",
                        help="train mode: maximize detection F1 instead of minimizing val_loss (recall-focused)")
    parser.add_argument("--val-split", type=str, default="val", choices=["train", "val"], help="Validation split for --mode val")
    parser.add_argument("--val-num-samples", type=int, default=0, help="0 => full split in --mode val")
    parser.add_argument("--val-iou-thr", type=float, default=0.5, help="IoU threshold for TP/FP/FN in --mode val")
    parser.add_argument(
        "--val-metric",
        type=str,
        default="f1",
        choices=["f1", "precision", "recall"],
        help="Metric to maximize in --mode val",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.output_dir:
        args.output_dir = str(CHECKPOINT_DIR / ("optuna_stage1_val" if args.mode == "val" else "optuna_stage1_train"))
    if not args.storage:
        db_name = "optuna_val.db" if args.mode == "val" else "optuna_train.db"
        args.storage = f"sqlite:///{(Path(args.output_dir) / db_name).as_posix()}"
    if not args.study_name:
        args.study_name = "stage1_detector_optuna_val" if args.mode == "val" else "stage1_detector_optuna_train"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
                direction="maximize" if (args.mode == "val" or getattr(args, "optimize_f1", False)) else "minimize",
                sampler=sampler,
                pruner=pruner,
            )
            break
        except Exception as _e:
            if "already exists" in str(_e):
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
        # Let workers keep requesting new trials until the shared global cap is reached.
        effective_n_trials = None

    if args.mode == "val":
        if not Path(args.checkpoint).exists():
            raise FileNotFoundError(f"Validation mode requires checkpoint, not found: {args.checkpoint}")
        study.optimize(
            lambda trial: objective_validation(trial, args),
            n_trials=effective_n_trials,
            timeout=timeout,
            callbacks=optimize_callbacks,
        )
    else:
        study.optimize(
            lambda trial: objective(trial, args),
            n_trials=effective_n_trials,
            timeout=timeout,
            callbacks=optimize_callbacks,
        )

    summary = {
        "study_name": study.study_name,
        "mode": args.mode,
        "best_trial": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
    }
    summary_path = out_dir / "best_params.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 70)
    print(f"Optuna search complete (mode={args.mode})")
    print(f"Best trial: {study.best_trial.number}")
    label = args.val_metric if args.mode == "val" else "val_loss"
    print(f"Best {label}: {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")
    print(f"Saved summary: {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
