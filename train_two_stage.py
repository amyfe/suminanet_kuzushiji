"""Two-stage training pipeline: Stage 1 (detection) + Stage 2 (classification)."""
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.cuda.amp as amp
from pathlib import Path
import json
from tqdm import tqdm
import argparse

from config import (
    DATA_DIR, DEVICE, BATCH_SIZE, IMAGE_SIZE, NUM_EPOCHS, LR, NUM_WORKERS, WEIGHT_DECAY,
    GRADIENT_ACCUMULATION_STEPS, CHECKPOINT_DIR, USE_MIXED_PRECISION
)
from model.kuronet import UNet, DetectorHead
from model.kuronet.encoder_wrapper import EncoderWrapper
from model.kuronet.decoder.attention import SeqDecoderAttention
from utils import KuzushijiDataset
from utils.detection_utils import build_detection_targets
from utils.focal_loss import focal_loss_heatmap
from utils.vocab import VocabManager

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
    """
    images = torch.stack([b["image"] for b in batch], dim=0)
    
    # Boxes and labels for detection
    boxes = [b.get("boxes", torch.empty((0, 4))) for b in batch]
    labels = [b.get("labels", torch.empty((0,), dtype=torch.long)) for b in batch]
    orientations = [b.get("orientation", "horizontal") for b in batch]
    
    # Text sequences
    seq_samples = [b for b in batch if ("text_ids" in b and b["text_ids"] is not None)]
    if len(seq_samples) == 0:
        return {"image": images, "text_ids": None, "text_lengths": torch.tensor([]), 
            "boxes": boxes, "labels": labels, "orientations": orientations}
    
    text_ids = [b["text_ids"] for b in seq_samples]
    text_lengths = torch.tensor([len(t) for t in text_ids], dtype=torch.long)
    text_padded = nn.utils.rnn.pad_sequence(text_ids, batch_first=True, padding_value=pad_id)
    
    return {"image": images, "text_ids": text_padded, "text_lengths": text_lengths,
        "boxes": boxes, "labels": labels, "orientations": orientations}

def scheduled_teacher_forcing(epoch, total_epochs, start=1.0, end=0.2, schedule="exp"):
    if schedule == "linear":
        return max(end, start - (start-end) * (epoch / max(1, (total_epochs - 1))))
    else:
        decay = 0.97 ** epoch
        return max(end, start * decay)


def validate_detector(unet, detector, dataloader, DEVICE, USE_MIXED_PRECISION):
    """Validate detector on full validation set. Returns average loss."""
    unet.eval()
    detector.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validating", leave=False)
        for batch in pbar:
            images = batch['image'].to(DEVICE)
            boxes = [b.to(DEVICE) if b.numel() > 0 else torch.empty((0, 4), device=DEVICE) for b in batch.get('boxes', [])]
            labels = [l.to(DEVICE) if l.numel() > 0 else torch.empty((0,), dtype=torch.long, device=DEVICE) for l in batch.get('labels', [])]
            
            with torch.amp.autocast(device_type='cuda', enabled=USE_MIXED_PRECISION):
                features = unet(images)
                outputs = detector(features)
                
                heatmap = outputs.get('heatmap')
                bbox_reg = outputs.get('bbox')
                cls_logits = outputs.get('cls')  # May be None
                
                B, _, H_feat, W_feat = features.shape
                output_size = (H_feat, W_feat)
                image_size = tuple(images.shape[-2:])
                
                gt_heatmap, gt_bbox, gt_cls = build_detection_targets(
                    boxes, labels, output_size, image_size, DEVICE
                )
                
                with torch.no_grad():
                    features_shared = detector.shared(features)
                heatmap_logits = detector.heatmap(features_shared)
                
                loss_heatmap = focal_loss_heatmap(
                    heatmap_logits, gt_heatmap, alpha=0.25, gamma=2.0, pos_weight=10.0
                )
                loss_bbox = torch.nn.SmoothL1Loss()(bbox_reg, gt_bbox)
                # Skip class loss (not computed)
                
                loss = loss_heatmap + 0.1 * loss_bbox
            
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"val_loss": total_loss / num_batches})
    
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    return avg_loss









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
    sos_id = vocab.sos_id
    eos_id = vocab.eos_id
    vocab_size = vocab.vocab_size

    # Clean up existing checkpoints: keep only newest, rename to checkpoint_old.pt
    ckpt_dir = Path(CHECKPOINT_DIR)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    prune_existing_checkpoints(ckpt_dir)

    # Dataset + loader - use pre-existing splits (splits/train.txt, splits/val.txt)
    train_dataset = KuzushijiDataset(Path(DATA_DIR), vocab=vocab, use_sequences=True, resize=IMAGE_SIZE, split='train')
    val_dataset = KuzushijiDataset(Path(DATA_DIR), vocab=vocab, use_sequences=True, resize=IMAGE_SIZE, split='val')
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                            collate_fn=lambda b: collate_fn(b, pad_id), pin_memory=True,
                            prefetch_factor=2, persistent_workers=NUM_WORKERS > 0)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                            collate_fn=lambda b: collate_fn(b, pad_id), pin_memory=True,
                            prefetch_factor=2, persistent_workers=NUM_WORKERS > 0)
    
    # Build model
    unet = UNet(in_channels=3, base_features=32).to(DEVICE)
    detector = DetectorHead(in_ch=32, num_classes=vocab.vocab_size, dropout_rate=0.3, predict_classes=False).to(DEVICE)  # Disable class head to save memory
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
    best_val_loss = None
    for epoch in range(num_epochs):
        # Training
        unet.train()
        detector.train()
        
        total_loss = 0
        loss_heatmap_total = 0
        loss_bbox_total = 0
        loss_cls_total = 0
        num_batches = 0
        num_labeled_pixels = 0
        num_total_pixels = 0
        
        # Show progress bar with limited update frequency to reduce log spam
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", 
                   mininterval=60.0)  # Update at most every 10 seconds
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(DEVICE)
            boxes = [b.to(DEVICE) if b.numel() > 0 else torch.empty((0, 4), device=DEVICE) for b in batch.get('boxes', [])]
            labels = [l.to(DEVICE) if l.numel() > 0 else torch.empty((0,), dtype=torch.long, device=DEVICE) for l in batch.get('labels', [])]
            
            with torch.amp.autocast(device_type='cuda', enabled=USE_MIXED_PRECISION):
                # Single forward pass - reuse detector outputs
                features = unet(images)  # (B, 32, H/8, W/8)
                outputs = detector(features)  # Returns dict with 'heatmap', 'bbox', 'cls'
                
                heatmap = outputs.get('heatmap')  # (B, 1, H, W) - sigmoid'd
                bbox_reg = outputs.get('bbox')    # (B, 4, H, W)
                cls_logits = outputs.get('cls')   # May be None if predict_classes=False
                
                # Build targets
                B, _, H_feat, W_feat = features.shape
                output_size = (H_feat, W_feat)
                image_size = tuple(images.shape[-2:])
                
                gt_heatmap, gt_bbox, gt_cls = build_detection_targets(
                    boxes, labels, output_size, image_size, DEVICE
                )
                
                # Get raw heatmap logits (recompute shared features for loss)
                with torch.no_grad():
                    features_shared = detector.shared(features)
                heatmap_logits = detector.heatmap(features_shared)  # Raw logits for focal loss
                
                # Compute losses with detailed tracking
                loss_heatmap = focal_loss_heatmap(
                    heatmap_logits, gt_heatmap, alpha=0.25, gamma=2.0, pos_weight=10.0
                )
                loss_bbox = torch.nn.SmoothL1Loss()(bbox_reg, gt_bbox)
                
                # Track labeled vs total pixels for statistics only
                if cls_logits is not None:
                    B_curr, _, H, W = cls_logits.shape
                    gt_cls_flat = gt_cls.reshape(-1)
                    valid_mask = gt_cls_flat >= 0
                    num_labeled = valid_mask.sum().item()
                    num_total = gt_cls_flat.numel()
                else:
                    # When class head is off, approximate labeled pixels from heatmap positives
                    hm_pos = (gt_heatmap > 0).sum().item()
                    num_labeled = hm_pos
                    num_total = gt_heatmap.numel()
                
                # Skip class loss: causes OOM with 4246 classes on flattened tensors
                # (99% of pixels are unlabeled anyway, class prediction is auxiliary)
                loss_cls = torch.tensor(0.0, device=DEVICE)
                
                # Heatmap-focused loss: spatial localization is the main goal for Stage 1
                loss = loss_heatmap + 0.1 * loss_bbox
                
                # Track for statistics BEFORE gradient accumulation scaling
                loss_heatmap_total += loss_heatmap.item()
                loss_bbox_total += loss_bbox.item() if isinstance(loss_bbox, torch.Tensor) and loss_bbox.numel() > 0 else 0
                loss_cls_total += loss_cls.item() if isinstance(loss_cls, torch.Tensor) and loss_cls.numel() > 0 else 0
                num_labeled_pixels += num_labeled
                num_total_pixels += num_total
                
                # Scale for gradient accumulation (after tracking)
                loss = loss / GRADIENT_ACCUMULATION_STEPS
            
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Optimizer step with gradient accumulation
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(list(unet.parameters()) + list(detector.parameters()), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(list(unet.parameters()) + list(detector.parameters()), 1.0)
                    optimizer.step()
                optimizer.zero_grad()
            
            # Accumulate unscaled loss for reporting
            total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            num_batches += 1
            pbar.set_postfix({"loss": total_loss / num_batches})
            
            # Print progress every 1000 batches when tqdm is disabled
            if not sys.stdout.isatty() and (batch_idx + 1) % 1000 == 0:
                avg_heat = loss_heatmap_total / num_batches
                avg_bbox = loss_bbox_total / num_batches
                avg_cls = loss_cls_total / num_batches if num_labeled_pixels > 0 else 0
                labeled_pct = 100.0 * num_labeled_pixels / max(1, num_total_pixels)
                print(f"Epoch {epoch+1}/{num_epochs} - Batch {batch_idx+1}/{len(train_dataloader)}")
                print(f"  Total Loss: {total_loss/num_batches:.4f} = Heatmap: {avg_heat:.4f} + Bbox: {avg_bbox*0.5:.4f} + Cls: {avg_cls*0.5:.4f}")
                print(f"  Labeled pixels: {num_labeled_pixels}/{num_total_pixels} ({labeled_pct:.1f}%)")
            
            # Clear cache periodically to prevent memory fragmentation
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
        
        scheduler.step()
        torch.cuda.empty_cache()
        
        # Validation
        val_loss = validate_detector(unet, detector, val_dataloader, DEVICE, USE_MIXED_PRECISION)
        train_loss = total_loss / num_batches
        
        # Compute average component losses
        avg_loss_heatmap = loss_heatmap_total / num_batches
        avg_loss_bbox = loss_bbox_total / num_batches
        avg_loss_cls = loss_cls_total / num_batches if num_labeled_pixels > 0 else 0
        labeled_pct = 100.0 * num_labeled_pixels / max(1, num_total_pixels)
        
        print(f"\nEpoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Components - Heatmap: {avg_loss_heatmap:.4f} | Bbox: {avg_loss_bbox:.4f} | Cls: {avg_loss_cls:.4f}")
        print(f"  Labeled pixels: {num_labeled_pixels}/{num_total_pixels} ({labeled_pct:.1f}%)")
        
        # Early stopping check
        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            is_best = True
        else:
            patience_counter += 1
            is_best = False
        
        # Save checkpoint
        checkpoint_path = checkpoint_dir / f"detector_epoch{epoch+1}.pt"
        try:
            torch.save({
                'epoch': epoch + 1,
                'unet_state_dict': unet.state_dict(),
                'detector_state_dict': detector.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                'val_loss': val_loss,
                'vocab_size': vocab.vocab_size,
                'is_best': is_best,
            }, checkpoint_path)
            
            if is_best:
                # Also save as best checkpoint
                best_path = checkpoint_dir / "detector_best.pt"
                torch.save({
                    'epoch': epoch + 1,
                    'unet_state_dict': unet.state_dict(),
                    'detector_state_dict': detector.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss,
                    'val_loss': val_loss,
                    'vocab_size': vocab.vocab_size,
                    'is_best': True,
                }, best_path)
                print(f"✅ Saved best checkpoint: {best_path} (Val Loss: {val_loss:.4f})")
            else:
                print(f"✅ Saved checkpoint: {checkpoint_path}")
            
            # Only prune after successful save
            prune_to_keep_last_n(checkpoint_dir, keep=2, exclude="detector_best.pt")
        except Exception as e:
            print(f"⚠️ Failed to save checkpoint: {e}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⏹️  Early stopping triggered! (Val loss did not improve for {patience} epochs)")
            print(f"Best val loss: {best_val_loss:.4f} at epoch {epoch+1-patience_counter}")
            break
    
    print(f"\n{'='*60}")
    print(f"Stage 1 training complete!")
    print(f"Final checkpoint: {checkpoint_path}")
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
    train(stage2=False)  # Run both stages