
import argparse
import logging
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from functools import partial
from pathlib import Path
from tqdm import tqdm

import torch.nn.functional as F

from config import (
    DATA_DIR, DEVICE, STAGE1_BATCH_SIZE, STAGE1_BBOX_WEIGHT, STAGE1_DIOU_WEIGHT, STAGE1_DROPOUT_RATE,
    STAGE1_FOCAL_POS_THRESHOLD, IMAGE_SIZE, NUM_EPOCHS, LR, NUM_WORKERS,
    STAGE1_FOCAL_ALPHA, STAGE1_FOCAL_GAMMA, STAGE1_HEATMAP_SIGMA, STAGE1_SIGMA_FLOOR, STAGE1_SIGMA_CEIL, STAGE1_SIGMA_SCALE,
    STAGE1_POS_WEIGHT, STAGE1_EARLY_STOPPING_PATIENCE, WEIGHT_DECAY,
    GRADIENT_ACCUMULATION_STEPS, CHECKPOINT_DIR, USE_MIXED_PRECISION, BACKBONE_BASE_FEATURES, BACKBONE_TYPE,
    SAM2_HARD_NEG_WEIGHT, SAM2_PREPROCESSING,
    SAM2_PROPOSALS_PREPROCESSING,
)
from model.suminanet import DetectorHead, build_backbone
from utils import KuzushijiDataset
from utils.detection_utils import build_detection_targets
from utils.focal_loss import focal_loss_heatmap
from utils.sam2.preprocess_sam2_illustrations import run_preprocessing
from utils.sam2.preprocess_sam2_proposals import run_preprocessing as _run_sam2_proposals

def _atomic_save(obj, path):
    """Write checkpoint atomically: save to .tmp then rename, so crashes never leave a corrupt file."""
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


from utils.training_helpers.helper_stage1 import (
    check_backbone_type,
    collate_fn,
    data_info_startup,
    masked_bbox_smoothl1_loss,
    masked_bbox_diou_loss,
    prune_to_keep_last_n,
    validate_detector,
)
from utils.vocab import VocabManager

def train_detector_stage(num_epochs=10, lr=None, checkpoint_dir=None, patience=3, val_split=None, resume_ckpt=None):
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
    
    # SAM2 illustration masking: build (or verify) the cache before loading data.
    # On first run this takes several minutes; subsequent runs exit in <1 s.
    if SAM2_PREPROCESSING:
        run_preprocessing(splits=["train", "val"] if val_split is None else ["train"])

    # Build or load vocab
    ann_files = sorted(list((Path(DATA_DIR) / "annotations").glob("*.json")))
    if len(ann_files) == 0:
        raise FileNotFoundError(f"No annotation files found in {Path(DATA_DIR)/'annotations'}")
    vocab = VocabManager.from_annotations(ann_files)
    pad_id = vocab.pad_id

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if resume_ckpt is None:
        # Clean up existing checkpoints only when starting fresh.
        prune_to_keep_last_n(checkpoint_dir, keep=2)

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
    
    train_dataloader = DataLoader(train_dataset, batch_size=STAGE1_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                            collate_fn=partial(collate_fn, pad_id=pad_id), pin_memory=True,
                            prefetch_factor=2, persistent_workers=NUM_WORKERS > 0)
    val_dataloader = DataLoader(val_dataset, batch_size=STAGE1_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                            collate_fn=partial(collate_fn, pad_id=pad_id), pin_memory=True,
                            prefetch_factor=2, persistent_workers=NUM_WORKERS > 0)
    
    effective_backbone_type = BACKBONE_TYPE
    if not check_backbone_type(resume_ckpt, effective_backbone_type):
        raise ValueError("Backbone type mismatch between checkpoint and current configuration. Please adjust BACKBONE_TYPE in config.py to match the checkpoint or use a different checkpoint.")
    # Build model. channels_last speeds up conv-heavy models on Ampere+/Blackwell/H200
    # tensor cores under AMP with no effect on training dynamics (layout only).
    backbone = build_backbone(effective_backbone_type, BACKBONE_BASE_FEATURES).to(DEVICE, memory_format=torch.channels_last)
    detector = DetectorHead(in_ch=BACKBONE_BASE_FEATURES, num_classes=vocab.vocab_size, dropout_rate=STAGE1_DROPOUT_RATE, predict_classes=False).to(DEVICE, memory_format=torch.channels_last)  # Disable class head to save memory
    print(f"✅ Model initialized: backbone={effective_backbone_type}, features={BACKBONE_BASE_FEATURES}, vocab={vocab.vocab_size} classes")
    # Optimizer
    optimizer = optim.Adam(
        list(backbone.parameters()) + list(detector.parameters()),
        lr=lr, weight_decay=WEIGHT_DECAY
    )
    
    # 2-epoch linear warmup then cosine annealing.
    _warmup = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=2
    )
    _cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, num_epochs - 2), eta_min=1e-6
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[_warmup, _cosine], milestones=[2]
    )

    # fp16 mixed precision (Turing GPUs have no bf16 tensor core path); GradScaler
    # guards against fp16 underflow during backward.
    scaler = torch.amp.GradScaler(device="cuda", enabled=USE_MIXED_PRECISION)

    best_val = None
    patience_ctr = 0
    start_epoch = 0

    if resume_ckpt is not None:
        ckpt = torch.load(resume_ckpt, map_location=DEVICE)
        backbone.load_state_dict(ckpt["backbone_state_dict"])
        detector.load_state_dict(ckpt["detector_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            try:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except Exception:
                print("Warning: scheduler state incompatible with current schedule — restarting scheduler.")
        start_epoch   = int(ckpt.get("epoch", 0))
        best_val      = ckpt.get("best_val", None)
        patience_ctr  = int(ckpt.get("patience_ctr", 0))
        print(
            f"▶ Resumed from {resume_ckpt}  "
            f"(epoch={start_epoch}, best_val={best_val}, patience_ctr={patience_ctr})"
        )

    print(f"Stage 1: Training DetectorHead for {num_epochs} epochs (early stopping patience={patience})...")
    print(f"Training set: {len(train_dataset)} images | Validation set: {len(val_dataset)} images")
    bbox_radius = 2
    for epoch in range(start_epoch, num_epochs):
        # Training
        backbone.train()
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
            images = batch['image'].to(DEVICE, memory_format=torch.channels_last)
            boxes = [b.to(DEVICE) if b.numel() > 0 else torch.empty((0, 4), device=DEVICE) for b in batch.get('boxes', [])]
            labels = [l.to(DEVICE) if l.numel() > 0 else torch.empty((0,), dtype=torch.long, device=DEVICE) for l in batch.get('labels', [])]

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=USE_MIXED_PRECISION):
                # Single forward pass
                features = backbone(images)  # (B, 32, H/8, W/8)
                outputs = detector(features)  # Returns dict with 'heatmap' (raw logits), 'bbox', etc.

                _, _, Hf, Wf = features.shape
                gt_heat, gt_bbox, gt_bbox_mask, _ = build_detection_targets(
                    boxes,
                    labels,
                    output_size=(Hf, Wf),
                    image_size=tuple(images.shape[-2:]),
                    device=DEVICE,
                    sigma=STAGE1_HEATMAP_SIGMA,
                    sigma_floor=STAGE1_SIGMA_FLOOR,
                    sigma_ceil=STAGE1_SIGMA_CEIL,
                    sigma_scale=STAGE1_SIGMA_SCALE,
                    bbox_radius=bbox_radius,
                )

                # Downsample illustration mask to feature-map resolution if present
                illus_mask_feat = None
                raw_illus = batch.get("illus_mask")
                if raw_illus is not None:
                    raw_illus = raw_illus.to(DEVICE)
                    if raw_illus.shape[-2:] != (Hf, Wf):
                        raw_illus = F.interpolate(
                            raw_illus.float(), size=(Hf, Wf), mode="nearest"
                        )
                    illus_mask_feat = raw_illus.bool()  # (B, 1, Hf, Wf)

                # Compute losses with detailed tracking
                loss_heatmap = focal_loss_heatmap(
                    outputs["heatmap"], gt_heat,
                    alpha=STAGE1_FOCAL_ALPHA, gamma=STAGE1_FOCAL_GAMMA,
                    pos_weight=STAGE1_POS_WEIGHT,
                    pos_threshold=STAGE1_FOCAL_POS_THRESHOLD,
                    illus_mask=illus_mask_feat,
                    illus_neg_weight=SAM2_HARD_NEG_WEIGHT,
                )
                loss_bbox = masked_bbox_smoothl1_loss(outputs["bbox"], gt_bbox, gt_bbox_mask)
                loss_diou = masked_bbox_diou_loss(
                    outputs["bbox"], gt_bbox, gt_bbox_mask,
                    image_size=tuple(images.shape[-2:]),
                )
                loss = loss_heatmap + STAGE1_BBOX_WEIGHT * loss_bbox + STAGE1_DIOU_WEIGHT * loss_diou  
                loss = loss / GRADIENT_ACCUMULATION_STEPS
            
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Optimizer step with gradient accumulation
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(list(backbone.parameters()) + list(detector.parameters()), 1.0)

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
                # Print data info for debugging on first batch of first epoch
                data_info_startup(images, features, outputs, gt_heat, gt_bbox, gt_bbox_mask, Hf, Wf)

        
        if n_batches > 0 and (n_batches % GRADIENT_ACCUMULATION_STEPS) != 0:
            if scaler is not None:
                scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                list(backbone.parameters()) + list(detector.parameters()), 1.0
            )

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        
        # Validation
        val_loss, val_heat, val_bbox = validate_detector(
            backbone, detector, val_dataloader, DEVICE, USE_MIXED_PRECISION, bbox_radius=bbox_radius
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
            "backbone_state_dict": backbone.state_dict(),
            "backbone_type": BACKBONE_TYPE,
            "unet_state_dict": backbone.state_dict(),  # backward-compat alias
            "detector_state_dict": detector.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val": best_val,
            "patience_ctr": patience_ctr,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "vocab_size": vocab.vocab_size,
            "is_best": is_best,
        }

        epoch_path = checkpoint_dir / f"detector_epoch{epoch+1}.pt"
        _atomic_save(ckpt, epoch_path)

        if is_best:
            _atomic_save(ckpt, checkpoint_dir / "detector_best.pt")
            print(f"✅ saved best: detector_best.pt (val={val_loss:.4f})")

        if patience > 0 and patience_ctr >= patience:
            print(f"Early stopping: no val improvement for {patience} epochs. best={best_val:.4f}")
            break
        prune_to_keep_last_n(CHECKPOINT_DIR, keep=2)
    
    print(f"\n{'='*60}")
    print(f"Stage 1 training complete!")
    print(f"Best checkpoint: {checkpoint_dir / 'detector_best.pt'}")
    print(f"{'='*60}\n")

    return backbone, detector

def train(args=None):
    parser = argparse.ArgumentParser(description="Stage 1: train detector head")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="CKPT",
        help="Path to a stage-1 checkpoint to resume from (e.g. checkpoints/stage1_detection/detector_epoch5.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
        help="Total number of training epochs (including already-completed ones when resuming)",
    )
    parsed = parser.parse_args(args)

    # Setup logging — tee stdout so every print() also lands in the log file
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True, parents=True)

    class _Tee:
        def __init__(self, *streams):
            self._streams = streams
        def write(self, data):
            for s in self._streams:
                s.write(data)
                s.flush()
        def flush(self):
            for s in self._streams:
                s.flush()

    _log_fh = open(log_dir / "train_stage1.log", "a")
    sys.stdout = _Tee(sys.__stdout__, _log_fh)
    sys.stderr = _Tee(sys.__stderr__, _log_fh)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'train_stage1.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("STAGE 1: TRAINING DETECTOR (Spatial Localization)")
    logger.info("=" * 60)
    train_detector_stage(
        num_epochs=parsed.epochs,
        lr=None,
        patience=STAGE1_EARLY_STOPPING_PATIENCE,
        resume_ckpt=Path(parsed.resume) if parsed.resume else None,
    )
    if SAM2_PROPOSALS_PREPROCESSING:
        _run_sam2_proposals(
            detector_ckpt=CHECKPOINT_DIR / "stage1_detection" / "detector_best.pt",
            splits=["train", "val"],
        )

if __name__ == "__main__":
    train()