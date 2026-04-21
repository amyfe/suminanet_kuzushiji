
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from config import (
    DATA_DIR, DEVICE, BATCH_SIZE, STAGE1_BBOX_WEIGHT, STAGE1_DROPOUT_RATE, STAGE1_FOCAL_POS_THRESHOLD, IMAGE_SIZE, NUM_EPOCHS, LR, NUM_WORKERS, STAGE1_FOCAL_ALPHA, STAGE1_FOCAL_GAMMA, STAGE1_HEATMAP_SIGMA, STAGE1_POS_WEIGHT, WEIGHT_DECAY,
    GRADIENT_ACCUMULATION_STEPS, CHECKPOINT_DIR, USE_MIXED_PRECISION
)
from model.kuronet import UNet, DetectorHead
from utils import KuzushijiDataset
from utils.detection_utils import build_detection_targets
from utils.focal_loss import focal_loss_heatmap
from utils.training_helpers.helper_stage1 import (
    collate_fn,
    masked_bbox_smoothl1_loss,
    prune_existing_checkpoints,
    validate_detector,
)
from utils.vocab import VocabManager

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
    detector = DetectorHead(in_ch=32, num_classes=vocab.vocab_size, dropout_rate=STAGE1_DROPOUT_RATE, predict_classes=False).to(DEVICE)  # Disable class head to save memory
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
    bbox_radius = 1
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
                    sigma=STAGE1_HEATMAP_SIGMA,
                    bbox_radius=bbox_radius,
                )
                # Compute losses with detailed tracking
                loss_heatmap = focal_loss_heatmap(
                    outputs["heatmap"], gt_heat,
                    alpha=STAGE1_FOCAL_ALPHA, gamma=STAGE1_FOCAL_GAMMA,
                    pos_weight=STAGE1_POS_WEIGHT,
                    pos_threshold=STAGE1_FOCAL_POS_THRESHOLD,
                )
                # Lower pos_thresh for bbox (include broader region around peak) and increase bbox weight
                loss_bbox = masked_bbox_smoothl1_loss(outputs["bbox"], gt_bbox, gt_bbox_mask)
                                
                # Balanced loss: heatmap for localization, bbox for accurate box dimensions
                loss = loss_heatmap + STAGE1_BBOX_WEIGHT * loss_bbox  
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

def train(args=None):  
    print("="*60)
    print("STAGE 1: TRAINING DETECTOR (Spatial Localization)")
    print("="*60)
    train_detector_stage(num_epochs=NUM_EPOCHS, lr=None)
        
if __name__ == "__main__":
    train()  