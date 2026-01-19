"""Two-stage training pipeline: Stage 1 (detection) + Stage 2 (classification)."""
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
from model.kuronet.classifier import build_glyph_classifier
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
    
    # Text sequences
    seq_samples = [b for b in batch if ("text_ids" in b and b["text_ids"] is not None)]
    if len(seq_samples) == 0:
        return {"image": images, "text_ids": None, "text_lengths": torch.tensor([]), 
                "boxes": boxes, "labels": labels}
    
    text_ids = [b["text_ids"] for b in seq_samples]
    text_lengths = torch.tensor([len(t) for t in text_ids], dtype=torch.long)
    text_padded = nn.utils.rnn.pad_sequence(text_ids, batch_first=True, padding_value=pad_id)
    
    return {"image": images, "text_ids": text_padded, "text_lengths": text_lengths,
            "boxes": boxes, "labels": labels}

def scheduled_teacher_forcing(epoch, total_epochs, start=1.0, end=0.2, schedule="exp"):
    if schedule == "linear":
        return max(end, start - (start-end) * (epoch / max(1, (total_epochs - 1))))
    else:
        decay = 0.97 ** epoch
        return max(end, start * decay)









def train_detector_stage(num_epochs=10, lr=None, checkpoint_dir=None):
    """Stage 1: Train DetectorHead to localize characters.
    
    Args:
        num_epochs: Number of epochs for detector training
        lr: Learning rate (default from config)
        checkpoint_dir: Where to save checkpoints
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

    # Dataset + loader
    dataset = KuzushijiDataset(Path(DATA_DIR), vocab=vocab, use_sequences=True, resize=IMAGE_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                            collate_fn=lambda b: collate_fn(b, pad_id), pin_memory=True)
    
    # Load data
    ##train_loader, val_loader = get_data_loaders(split_mode="train")
    
    # Build model
    unet = UNet(in_channels=3, base_features=32).to(DEVICE)
    detector = DetectorHead(in_ch=32, num_classes=3000).to(DEVICE)
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
    
    # Mixed precision
    scaler = amp.GradScaler() if USE_MIXED_PRECISION else None
    
    ##DISCUSS: ce_loss = nn.CrossEntropyLoss(ignore_index=pad_id)
    ##DISCUSS: ctc_loss_fn = nn.CTCLoss(blank=pad_id, zero_infinity=True)
    ##DISCUSS: smoke_test((unet, encoder, decoder, ctc_head), dataset, vocab, DEVICE)
    ##DISCUSS CTC warmup 
    print(f"Stage 1: Training DetectorHead for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Training
        unet.train()
        detector.train()
        
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(DEVICE)
            boxes = [b.to(DEVICE) if b.numel() > 0 else torch.empty((0, 4), device=DEVICE) for b in batch.get('boxes', [])]
            labels = [l.to(DEVICE) if l.numel() > 0 else torch.empty((0,), dtype=torch.long, device=DEVICE) for l in batch.get('labels', [])]
            
            if USE_MIXED_PRECISION:
                with amp.autocast(enabled=USE_MIXED_PRECISION):
                    # Forward pass
                    features = unet(images)  # (B, 32, H/8, W/8)
                    heatmap, bbox_reg, cls_logits = detector(features)  # Multi-task outputs
                    
                    # Build targets
                    B, _, H_feat, W_feat = features.shape
                    output_size = (H_feat, W_feat)
                    image_size = tuple(images.shape[-2:])
                    
                    gt_heatmap, gt_bbox, gt_cls = build_detection_targets(
                        boxes, labels, output_size, image_size, DEVICE
                    )
                    
                    # Compute loss with FocalLoss
                    loss_heatmap = focal_loss_heatmap(
                        heatmap, gt_heatmap, alpha=0.25, gamma=2.0, pos_weight=10.0
                    )
                    loss_bbox = torch.nn.SmoothL1Loss()(bbox_reg, gt_bbox)
                    loss_cls = torch.nn.CrossEntropyLoss(ignore_index=-1)(
                        cls_logits.permute(0, 2, 3, 1).reshape(-1, cls_logits.size(1)),
                        gt_cls.reshape(-1)
                    )
                    
                    loss = loss_heatmap + 0.5 * loss_bbox + 0.5 * loss_cls
                    loss = loss / GRADIENT_ACCUMULATION_STEPS
                
                if scaler is not None:
                    scaler.scale(loss).backward()
                    # Optimizer step with gradient accumulation
                    if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(list(unet.parameters()) + list(detector.parameters()), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    loss.backward()
                            
            else:
                # Forward pass
                features = unet(images)
                heatmap, bbox_reg, cls_logits = detector(features)
                
                # Build targets
                B, _, H_feat, W_feat = features.shape
                output_size = (H_feat, W_feat)
                image_size = tuple(images.shape[-2:])
                
                gt_heatmap, gt_bbox, gt_cls = build_detection_targets(
                    boxes, labels, output_size, image_size, DEVICE
                )
                
                # Compute loss
                loss_heatmap = focal_loss_heatmap(
                    heatmap, gt_heatmap, alpha=0.25, gamma=2.0, pos_weight=10.0
                )
                loss_bbox = torch.nn.SmoothL1Loss()(bbox_reg, gt_bbox)
                loss_cls = torch.nn.CrossEntropyLoss(ignore_index=-1)(
                    cls_logits.permute(0, 2, 3, 1).reshape(-1, cls_logits.size(1)),
                    gt_cls.reshape(-1)
                )
                
                loss = loss_heatmap + 0.5 * loss_bbox + 0.5 * loss_cls
                loss = loss / GRADIENT_ACCUMULATION_STEPS
                loss.backward()
                
                # Optimizer step with gradient accumulation
                if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(list(unet.parameters()) + list(detector.parameters()), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
            
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": total_loss / num_batches})
        
        scheduler.step()
        
        # Save checkpoint
        checkpoint_path = checkpoint_dir / f"detector_epoch{epoch+1}.pt"
        torch.save({
            'epoch': epoch + 1,
            'unet_state_dict': unet.state_dict(),
            'detector_state_dict': detector.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss / num_batches,
        }, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
        prune_to_keep_last_n(checkpoint_dir, keep=2, exclude="checkpoint_old.pt")
    
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
                            collate_fn=lambda b: collate_fn(b, pad_id), pin_memory=True)
    
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
        sampling_method="multinomial", use_roi_attention=False  # Don't use broken ROI attention
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
    
    # CTC Warmup (optional but recommended)
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
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(DEVICE)
            text_ids = batch["text_ids"].to(DEVICE) if batch["text_ids"] is not None else None
            
            if text_ids is None:
                continue

            input_seq = text_ids[:, :-1]
            targets = text_ids[:, 1:]

            optimizer.zero_grad()
            
            with amp.autocast(enabled=USE_MIXED_PRECISION):
                # Get encoder sequence representation
                enc_outputs, enc_mask = encoder(images, orientation="horizontal")
                
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
        encoder, decoder, ctc_head = train_sequence_stage(
            detector_ckpt_path=CHECKPOINT_DIR / "stage1_detection" / f"detector_epoch{NUM_EPOCHS}.pt",
            num_epochs=NUM_EPOCHS,
            lr=None,
            use_ctc_warmup=True
        )
        
if __name__ == "__main__":
    train(stage2=True)  # Run both stages