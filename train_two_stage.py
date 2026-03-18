"""Two-stage training pipeline: Stage 1 (detection) + Stage 2 (classification)"""
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.cuda.amp as amp
from pathlib import Path
from tqdm import tqdm

from config import (
    DATA_DIR, DEVICE, BATCH_SIZE, DROPOUT_RATE, IMAGE_SIZE, NUM_EPOCHS, LR, NUM_WORKERS, WEIGHT_DECAY,
    GRADIENT_ACCUMULATION_STEPS, CHECKPOINT_DIR, USE_MIXED_PRECISION, DETECTOR_HEATMAP_SIGMA,
    FOCAL_ALPHA, FOCAL_GAMMA, POS_WEIGHT, BBOX_WEIGHT
)
from model.kuronet import UNet, DetectorHead
from model.kuronet.encoder_wrapper import EncoderWrapper
from model.kuronet.decoder.attention import SeqDecoderAttention
from utils import KuzushijiDataset
from utils.detection_utils import build_detection_targets
from utils.focal_loss import focal_loss_heatmap
from utils.vocab import VocabManager
import torch.nn.functional as F


def prune_existing_checkpoints(ckpt_dir: Path, old_name: str = "checkpoint_old.pt"):
    """At start: keep only the newest checkpoint, rename it to old_name."""
    ckpts = sorted(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        return
    newest = ckpts[-1]
    for p in ckpts[:-1]:
        try:
            p.unlink()
        except Exception as exc:  # best-effort cleanup
            print(f"Warning: could not delete {p}: {exc}")
    target = ckpt_dir / old_name
    if newest == target:
        return
    if target.exists():
        try:
            target.unlink()
        except Exception:
            pass
    try:
        newest.rename(target)
        print(f"Preserved latest checkpoint as {target.name}")
    except Exception as exc:
        print(f"Warning: could not rename {newest} to {target}: {exc}")


def prune_to_keep_last_n(ckpt_dir: Path, keep: int = 2, exclude: str = "checkpoint_old.pt"):
    """Keep only the newest N checkpoints (excluding a preserved old file)."""
    ckpts = [p for p in ckpt_dir.glob("*.pt") if p.name != exclude]
    ckpts = sorted(ckpts, key=lambda p: p.stat().st_mtime)
    if len(ckpts) <= keep:
        return
    to_delete = ckpts[:-keep]
    for p in to_delete:
        try:
            p.unlink()
        except Exception as exc:
            print(f"Warning: could not delete old checkpoint {p}: {exc}")

def collate_fn(batch, pad_id):
    """
    batch: list of samples from KuzushijiDataset
    Each sample contains: image (Tensor), text_ids (optional Tensor), text_length, boxes, labels
    Returns dict with images, text_ids_padded, text_lengths, boxes, labels
    
    NOTE: If a sample lacks text_ids, we still include boxes/labels for that image,
    but set its text_ids to None. This ensures 1-to-1 correspondence between batch indices.
    """
    images = torch.stack([b["image"] for b in batch], dim=0)
    
    # Boxes and labels for detection
    boxes = [b.get("boxes", torch.empty((0, 4))) for b in batch]
    labels = [b.get("labels", torch.empty((0,), dtype=torch.long)) for b in batch]
    orientations = [b.get("orientation", "horizontal") for b in batch]
    
    # Text sequences - fill missing with None instead of filtering
    text_ids_list = []
    text_lengths_list = []
    for b in batch:
        if "text_ids" in b and b["text_ids"] is not None:
            text_ids_list.append(b["text_ids"])
            text_lengths_list.append(len(b["text_ids"]))
        else:
            text_ids_list.append(None)
            text_lengths_list.append(0)
    
    # Pad sequences (skip None entries)
    valid_text_ids = [t for t in text_ids_list if t is not None]
    if len(valid_text_ids) == 0:
        text_padded = None
        text_lengths = torch.tensor(text_lengths_list, dtype=torch.long)
    else:
        text_padded = nn.utils.rnn.pad_sequence(valid_text_ids, batch_first=True, padding_value=pad_id)
        text_lengths = torch.tensor(text_lengths_list, dtype=torch.long)
    
    return {
        "image": images, 
        "text_ids": text_padded, 
        "text_lengths": text_lengths,
        "boxes": boxes, 
        "labels": labels, 
        "orientations": orientations,
        "text_ids_present": torch.tensor([t is not None for t in text_ids_list], dtype=torch.bool)
    }

def scheduled_teacher_forcing(epoch, total_epochs, start=1.0, end=0.2, schedule="exp"):
    if schedule == "linear":
        return max(end, start - (start-end) * (epoch / max(1, (total_epochs - 1))))
    else:
        decay = 0.97 ** epoch
        return max(end, start * decay)


def masked_bbox_smoothl1_loss(
    pred_bbox: torch.Tensor,   # (B,4,H,W)
    gt_bbox: torch.Tensor,     # (B,4,H,W)
    gt_bbox_mask: torch.Tensor,  # (B,H,W) boolean mask
) -> torch.Tensor:
    """
    Compute bbox loss only where objects exist.
    We use gt_heatmap to decide positives (peaks / gaussian area).
    """
    if gt_bbox_mask.sum() == 0:
        return pred_bbox.new_tensor(0.0)

    # select positives -> (Npos,4)
    pred_pos = pred_bbox.permute(0, 2, 3, 1)[gt_bbox_mask]
    gt_pos = gt_bbox.permute(0, 2, 3, 1)[gt_bbox_mask]

    return F.smooth_l1_loss(pred_pos, gt_pos, reduction="mean")

def validate_detector(unet, detector, dataloader, DEVICE, USE_MIXED_PRECISION):
    unet.eval()
    detector.eval()
    
    total_loss = 0.0
    total_heat = 0.0
    total_bbox = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validating", leave=False)):
        images = batch['image'].to(DEVICE)
        boxes = [b.to(DEVICE) if b.numel() > 0 else torch.empty((0, 4), device=DEVICE) for b in batch.get('boxes', [])]
        labels = [l.to(DEVICE) if (l is not None and l.numel() > 0) else torch.empty((0,), dtype=torch.long, device=DEVICE) for l in batch.get('labels', [])]
        
        with torch.amp.autocast(device_type='cuda', enabled=USE_MIXED_PRECISION):
            features = unet(images)
            B, _, Hf, Wf = features.shape
            gt_heat, gt_bbox, gt_bbox_mask, gt_cls = build_detection_targets(
                boxes, labels, output_size=(Hf, Wf), image_size=tuple(images.shape[-2:]), device=DEVICE, sigma=DETECTOR_HEATMAP_SIGMA, bbox_radius=0
            )
            
            features_shared = detector.shared(features)
            heat_logits = detector.heatmap(features_shared)
            bbox_reg = detector.bbox(features_shared)
            
            loss_heatmap = focal_loss_heatmap(heat_logits, gt_heat, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, pos_weight=POS_WEIGHT)
            loss_bbox = masked_bbox_smoothl1_loss(bbox_reg, gt_bbox, gt_bbox_mask)
            loss = loss_heatmap + BBOX_WEIGHT * loss_bbox

            if batch_idx == 0:
                print("\n[VAL DEBUG]")
                print("images.shape:", tuple(images.shape))
                print("features.shape:", tuple(features.shape))
                print("gt_heat min/max/mean:",
                    gt_heat.min().item(),
                    gt_heat.max().item(),
                    gt_heat.mean().item())
                print("num bbox supervised cells:", int(gt_bbox_mask.sum().item()))
                if gt_bbox_mask.sum() > 0:
                    gt_bbox_pos = gt_bbox.permute(0, 2, 3, 1)[gt_bbox_mask]
                    pred_bbox_pos = bbox_reg.permute(0, 2, 3, 1)[gt_bbox_mask]
                    print("gt_bbox mean [dx,dy,bw,bh]:", gt_bbox_pos.mean(dim=0).detach().cpu().tolist())
                    print("pred_bbox mean [dx,dy,bw,bh]:", pred_bbox_pos.mean(dim=0).detach().cpu().tolist())

        
        total_loss += float(loss.item())
        total_heat += float(loss_heatmap.item())
        total_bbox += float(loss_bbox.item())
        num_batches += 1
    
    denom = max(1, num_batches)
    return (
        total_loss / denom,
        total_heat / denom,
        total_bbox / denom,
    )

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
    detector = DetectorHead(in_ch=32, num_classes=vocab.vocab_size, dropout_rate=DROPOUT_RATE, predict_classes=False).to(DEVICE)  # Disable class head to save memory
    print(f"✅ Model initialized with {vocab.vocab_size} classes (from vocab)")
    print(f"⚠️  Class prediction disabled to prevent OOM (Stage 1 focuses on spatial detection only)")
    ##DISCUSS: encoder = EncoderWrapper(backbone=unet, in_channels=32, enc_dim=256).to(DEVICE)
    ##DISCUSS: ctc_head = nn.Linear(256, vocab_size).to(DEVICE)
    ##DISCUSS: decoder = SeqDecoderAttention(embed_dim=64, hidden_dim=256, vocab_size=vocab_size,
                                #   enc_dim=256, num_layers=1, init_from_encoder=True,
                                #   sampling_method="multinomial", use_roi_attention=USE_ROI_ATTENTION).to(DEVICE)
    # Optimizer
    optimizer = optim.Adam(
        list(unet.parameters()) + list(detector.parameters()),
        lr=lr, weight_decay=WEIGHT_DECAY
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    # Mixed precision (use new API to avoid deprecation warning)
    scaler = torch.amp.GradScaler(device='cuda') if USE_MIXED_PRECISION else None
    
    ##DISCUSS: ce_loss = nn.CrossEntropyLoss(ignore_index=pad_id)
    ##DISCUSS: ctc_loss_fn = nn.CTCLoss(blank=pad_id, zero_infinity=True)
    ##DISCUSS: smoke_test((unet, encoder, decoder, ctc_head), dataset, vocab, DEVICE)
    ##DISCUSS CTC warmup 
    print(f"Stage 1: Training DetectorHead for {num_epochs} epochs (early stopping patience={patience})...")
    print(f"Training set: {len(train_dataset)} images | Validation set: {len(val_dataset)} images")
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
                    boxes, labels, output_size=(Hf, Wf), image_size=tuple(images.shape[-2:]), device=DEVICE, sigma=DETECTOR_HEATMAP_SIGMA
                )
                # Compute losses with detailed tracking
                loss_heatmap = focal_loss_heatmap(
                    outputs["heatmap"], gt_heat, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, pos_weight=POS_WEIGHT
                )
                # Lower pos_thresh for bbox (include broader region around peak) and increase bbox weight
                loss_bbox = masked_bbox_smoothl1_loss(outputs["bbox"], gt_bbox, gt_bbox_mask)
                                
                # Balanced loss: heatmap for localization, bbox for accurate box dimensions
                loss = loss_heatmap + BBOX_WEIGHT * loss_bbox  
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
            # Clear cache periodically to prevent memory fragmentation
            if step % 50 == 0:
                torch.cuda.empty_cache()
        
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
        torch.cuda.empty_cache()
        
        # Validation
        val_loss, val_heat, val_bbox = validate_detector(
            unet, detector, val_dataloader, DEVICE, USE_MIXED_PRECISION
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


def train_sequence_stage(detector_ckpt_path, num_epochs=10, lr=None, checkpoint_dir=None, use_ctc_warmup=True):
    """Stage 2: Train Encoder + Decoder for sequence prediction with context.
    
    Uses detected boxes to guide attention and sequence prediction.
    Combines spatial detection (Stage 1) with contextual understanding.
    
    Args:
        detector_ckpt_path: Path to detector checkpoint
        num_epochs: Number of epochs for sequence training
        lr: Learning rate (default from config)
        checkpoint_dir: Where to save checkpoints
        use_ctc_warmup: Whether to do CTC warmup before attention training
    """
    if lr is None:
        lr = LR
    if checkpoint_dir is None:
        checkpoint_dir = CHECKPOINT_DIR / "stage2_sequence"
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    # Build or load vocab
    ann_files = sorted(list((Path(DATA_DIR) / "annotations").glob("*.json")))
    if len(ann_files) == 0:
        raise FileNotFoundError(f"No annotation files found in {Path(DATA_DIR)/'annotations'}")
    vocab = VocabManager.from_annotations(ann_files)
    pad_id = vocab.pad_id
    sos_id = vocab.sos_id
    eos_id = vocab.eos_id
    vocab_size = vocab.vocab_size

    # Dataset + loader
    dataset = KuzushijiDataset(Path(DATA_DIR), vocab=vocab, use_sequences=True, resize=IMAGE_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                            collate_fn=lambda b: collate_fn(b, pad_id), pin_memory=True,
                            prefetch_factor=2, persistent_workers=NUM_WORKERS > 0)
    
    # Build and load detector (frozen for spatial guidance)
    unet = UNet(in_channels=3, base_features=32).to(DEVICE)
    detector = DetectorHead(in_ch=32, num_classes=3000).to(DEVICE)
    
    checkpoint = torch.load(detector_ckpt_path, map_location=DEVICE)
    unet.load_state_dict(checkpoint['unet_state_dict'])
    detector.load_state_dict(checkpoint['detector_state_dict'])
    
    # Freeze detector - we only use it for guidance
    for p in unet.parameters():
        p.requires_grad = False
    for p in detector.parameters():
        p.requires_grad = False
    
    # Build sequence models (trainable)
    encoder = EncoderWrapper(backbone=unet, in_channels=32, enc_dim=256).to(DEVICE)
    decoder = SeqDecoderAttention(
        embed_dim=64, hidden_dim=256, vocab_size=vocab_size,
        enc_dim=256, num_layers=1, init_from_encoder=True,
        sampling_method="multinomial", use_roi_attention=False 
    ).to(DEVICE)
    ctc_head = nn.Linear(256, vocab_size).to(DEVICE)
    
    # Optimizer - only train encoder, decoder, ctc_head (detector frozen)
    optimizer = optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()) + list(ctc_head.parameters()),
        lr=lr, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    # Mixed precision
    scaler = amp.GradScaler() if USE_MIXED_PRECISION else None
    
    # Losses
    ce_loss = nn.CrossEntropyLoss(ignore_index=pad_id)
    ctc_loss_fn = nn.CTCLoss(blank=pad_id, zero_infinity=True)
    
    # CTC Warmup (optional)
    if use_ctc_warmup:
        print(f"Stage 2: CTC Warmup for 2 epochs...")
        for epoch in range(2):
            encoder.train()
            total_loss = 0.0
            n_batches = 0
            
            pbar = tqdm(dataloader, desc=f"CTC Warmup {epoch+1}/2")
            for batch_idx, batch in enumerate(pbar):
                images = batch["image"].to(DEVICE)
                text_ids = batch["text_ids"].to(DEVICE) if batch["text_ids"] is not None else None
                if text_ids is None:
                    continue

                # collate_fn packs only valid text rows; keep images aligned to that subset
                text_ids_present = batch.get("text_ids_present", None)
                if text_ids_present is not None:
                    valid_idx = text_ids_present.to(DEVICE).nonzero(as_tuple=False).squeeze(1)
                    if valid_idx.numel() == 0:
                        continue
                    images = images.index_select(0, valid_idx)

                optimizer.zero_grad()
                
                with amp.autocast(enabled=USE_MIXED_PRECISION):
                    enc_seq, enc_mask = encoder(images, orientation="horizontal")
                    logits = ctc_head(enc_seq)  # (B, T, V)
                    log_probs = logits.log_softmax(dim=-1).permute(1,0,2).contiguous()  # (T,B,V)

                    # Prepare CTC targets (strip SOS/EOS/padding)
                    targets = []
                    target_lengths = []
                    for i in range(text_ids.size(0)):
                        ids = text_ids[i]
                        ids = ids[ids != pad_id]
                        if ids.numel() == 0:
                            continue
                        if ids[0].item() == sos_id:
                            ids = ids[1:]
                        if ids.numel() > 0 and ids[-1].item() == eos_id:
                            ids = ids[:-1]
                        if ids.numel() == 0:
                            continue
                        targets.append(ids)
                        target_lengths.append(ids.numel())
                    
                    if len(targets) == 0:
                        continue
                    
                    targets_concat = torch.cat(targets).to(DEVICE)
                    target_lengths = torch.tensor(target_lengths, dtype=torch.long, device=DEVICE)
                    input_lengths = torch.full((enc_seq.size(0),), fill_value=enc_seq.size(1), dtype=torch.long, device=DEVICE)

                    loss = ctc_loss_fn(log_probs, targets_concat, input_lengths, target_lengths)
                
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
                    optimizer.step()

                total_loss += loss.item()
                n_batches += 1
                pbar.set_postfix({"ctc_loss": total_loss / n_batches})
    
    # Main Attention Training with detected boxes as guidance
    print(f"Stage 2: Training Encoder + Decoder for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()
        unet.eval()  # Keep detector frozen
        detector.eval()
        
        tf_ratio = scheduled_teacher_forcing(epoch, num_epochs, start=1.0, end=0.1, schedule="exp")
        total_loss = 0.0
        total_seq_loss = 0.0
        n_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}",
                   mininterval=60.0)
        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(DEVICE)
            text_ids = batch["text_ids"].to(DEVICE) if batch["text_ids"] is not None else None
            orientations = batch.get("orientations", None)
            
            if text_ids is None:
                continue

            # collate_fn packs only valid text rows; keep image/orientation tensors aligned
            text_ids_present = batch.get("text_ids_present", None)
            if text_ids_present is not None:
                valid_idx = text_ids_present.to(DEVICE).nonzero(as_tuple=False).squeeze(1)
                if valid_idx.numel() == 0:
                    continue
                images = images.index_select(0, valid_idx)
                if orientations is not None:
                    valid_idx_cpu = valid_idx.detach().cpu().tolist()
                    orientations = [orientations[i] for i in valid_idx_cpu]

            input_seq = text_ids[:, :-1]
            targets = text_ids[:, 1:]

            optimizer.zero_grad()
            
            with amp.autocast(enabled=USE_MIXED_PRECISION):
                # Use batch orientation if consistent; fall back to horizontal
                if orientations is not None and len(orientations) > 0:
                    first_orientation = orientations[0]
                    all_same = all(o == first_orientation for o in orientations)
                    orientation = first_orientation if all_same else "horizontal"
                else:
                    orientation = "horizontal"

                # Get encoder sequence representation
                enc_outputs, enc_mask = encoder(images, orientation=orientation)
                
                # Decoder forward pass (predicts sequence with context)
                decoder_output = decoder(
                    input_seq=input_seq,
                    enc_outputs=enc_outputs,
                    enc_mask=enc_mask,
                    teacher_forcing_ratio=tf_ratio,
                    targets=targets,
                    eos_id=eos_id,
                    image_size=(images.shape[2], images.shape[3]),
                )
                
                # Handle both 3-tuple and 4-tuple output
                if len(decoder_output) == 4:
                    logits, hidden, attn, predicted_boxes = decoder_output
                else:
                    logits, hidden, attn = decoder_output
                
                B, T_dec, V = logits.shape
                
                # Sequence loss (contextual understanding)
                loss_seq = ce_loss(logits.reshape(-1, V), targets.reshape(-1))
                loss = loss_seq
            
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
                optimizer.step()
            
            total_seq_loss += loss_seq.item()
            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": total_loss / n_batches, "seq": total_seq_loss / n_batches})
        
        scheduler.step()
        
        # Save checkpoint
        checkpoint_path = checkpoint_dir / f"sequence_epoch{epoch+1}.pt"
        torch.save({
            'epoch': epoch + 1,
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'ctc_head_state_dict': ctc_head.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss / n_batches,
            'vocab': vocab.char2id,
        }, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
        prune_to_keep_last_n(checkpoint_dir, keep=2, exclude="checkpoint_old.pt")
    
    return encoder, decoder, ctc_head

def train(stage2=False, args=None):  
    print("="*60)
    print("STAGE 1: TRAINING DETECTOR (Spatial Localization)")
    print("="*60)
    unet, detector = train_detector_stage(num_epochs=NUM_EPOCHS, lr=None)
    
    if stage2 == True:
        print("="*60)
        print("STAGE 2: TRAINING SEQUENCE MODEL (Contextual Understanding)")
        print("="*60)
        # Use best checkpoint if it exists, otherwise use last epoch
        best_ckpt = CHECKPOINT_DIR / "stage1_detection" / "detector_best.pt"
        last_ckpt = CHECKPOINT_DIR / "stage1_detection" / f"detector_epoch{NUM_EPOCHS}.pt"
        detector_ckpt = best_ckpt if best_ckpt.exists() else last_ckpt
        
        encoder, decoder, ctc_head = train_sequence_stage(
            detector_ckpt_path=detector_ckpt,
            num_epochs=NUM_EPOCHS,
            lr=None,
            use_ctc_warmup=True
        )
        
if __name__ == "__main__":
    train(stage2=False)  # Run stage 1 only