# train_seq.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
import math
import os

# ---- imports from your repo (adjust paths if needed) ----
from model.kuronet.unet import UNet
from model.kuronet.encoder_wrapper import EncoderWrapper
from model.kuronet import DetectorHead
from model.kuronet.decoder.attention import SeqDecoderAttention
from utils import KuzushijiDataset   # the corrected dataset
from utils.vocab import VocabManager
from utils.detection_utils import build_detection_targets, compute_detection_losses, compute_roi_box_loss
from utils.validation import validate
from config import (DEVICE, NUM_EPOCHS, LR, WEIGHT_DECAY, CHECKPOINT_DIR, DATA_DIR, BATCH_SIZE, NUM_WORKERS,
                    USE_DETECTOR_HEAD, USE_ROI_ATTENTION, DETECTION_LOSS_WEIGHT, ROI_BOX_LOSS_WEIGHT,
                    NUM_CLASSES, DETECTOR_HEATMAP_SIGMA, RUN_VALIDATION, VALIDATION_FREQ, VALIDATION_BATCHES)

# ---- Hyperparams you can tweak ----
CTC_WARMUP_EPOCHS = 5   # set 0 to disable
MAX_DECODING_LEN = 200  # fallback for decoder if needed
GRAD_CLIP = 1.0

# ---- Collate function (pads sequence with vocab.pad_id) ----
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

# ---- Scheduled teacher forcing ratio ----
def scheduled_teacher_forcing(epoch, total_epochs, start=1.0, end=0.1, schedule="exp"):
    if schedule == "linear":
        return max(end, start - (start-end) * (epoch / max(1, (total_epochs - 1))))
    else:
        decay = 0.97 ** epoch
        return max(end, start * decay)

# ---- Helper to prepare CTC targets (remove padding, SOS/EOS) ----
def prepare_ctc_targets(text_ids_batch, pad_id, sos_id, eos_id, device):
    """
    text_ids_batch: (B, T_padded)
    Returns:
      targets_concat: 1D tensor with concatenated target sequences
      target_lengths: 1D tensor len B_targets
    If no valid targets -> returns (None, None)
    """
    B, T = text_ids_batch.shape
    targets = []
    target_lengths = []
    for i in range(B):
        ids = text_ids_batch[i]
        # remove padding
        ids = ids[ids != pad_id]
        if ids.numel() == 0:
            continue
        # strip sos/eos if present
        if ids[0].item() == sos_id:
            ids = ids[1:]
        if ids.numel() > 0 and ids[-1].item() == eos_id:
            ids = ids[:-1]
        if ids.numel() == 0:
            continue
        targets.append(ids)
        target_lengths.append(ids.numel())
    if len(targets) == 0:
        return None, None
    targets_concat = torch.cat(targets).to(device)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long, device=device)
    return targets_concat, target_lengths

# ---- Smoke test utility: quick forward pass to validate shapes ----
def smoke_test(models, dataset, vocab, device, n_samples=2):
    print("Running smoke test (small forward pass) ...")
    unet, encoder, decoder, ctc_head = models
    # take n_samples
    loader = DataLoader(dataset, batch_size=n_samples, shuffle=False, collate_fn=lambda b: collate_fn(b, vocab.pad_id))
    batch = next(iter(loader))
    images = batch["image"].to(device)
    text_ids = batch["text_ids"].to(device) if batch["text_ids"] is not None else None

    with torch.no_grad():
        feats2d, _ = encoder.backbone(images), None
        # test encoder wrapper
        enc_seq, enc_mask = encoder(images, orientation="horizontal")
        print("enc_seq:", enc_seq.shape, "enc_mask:", enc_mask.shape)
        # test ctc head
        ctc_logits = ctc_head(enc_seq)  # (B, T, V)
        print("ctc_logits:", ctc_logits.shape)
        if text_ids is not None:
            input_seq = text_ids[:, :-1]
            targets = text_ids[:, 1:]
            decoder_out = decoder(input_seq=input_seq, enc_outputs=enc_seq, enc_mask=enc_mask,
                                   teacher_forcing_ratio=1.0, targets=targets, eos_id=vocab.eos_id)
            if len(decoder_out) == 4:
                logits, hidden, attn, predicted_boxes = decoder_out
                print("decoder logits:", logits.shape, "attn:", None if attn is None else attn.shape,
                      "boxes:", None if predicted_boxes is None else predicted_boxes.shape)
            else:
                logits, hidden, attn = decoder_out
                print("decoder logits:", logits.shape, "attn:", None if attn is None else attn.shape)
    print("smoke test finished successfully.\n")

# ---- Main training function ----
def train():
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
    dataset = KuzushijiDataset(Path(DATA_DIR), vocab=vocab, use_sequences=True, resize=(512,512))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                            collate_fn=lambda b: collate_fn(b, pad_id))

    # Instantiate models
    unet = UNet(in_channels=3, base_features=32).to(DEVICE)
    encoder = EncoderWrapper(backbone=unet, in_channels=32, enc_dim=256).to(DEVICE)
    decoder = SeqDecoderAttention(embed_dim=64, hidden_dim=256, vocab_size=vocab_size,
                                  enc_dim=256, num_layers=1, init_from_encoder=True,
                                  sampling_method="multinomial", use_roi_attention=USE_ROI_ATTENTION).to(DEVICE)
    ctc_head = nn.Linear(256, vocab_size).to(DEVICE)  # persistent CTC head
    
    # Optional detector head (Option 1)
    detector = None
    if USE_DETECTOR_HEAD:
        detector = DetectorHead(in_ch=32, num_classes=NUM_CLASSES, predict_boxes=True).to(DEVICE)
        print(f"Using DetectorHead for box prediction (Option 1)")
    
    # Report training mode
    if USE_ROI_ATTENTION:
        print(f"Using ROI attention for box prediction (Option 2)")
    if not USE_DETECTOR_HEAD and not USE_ROI_ATTENTION:
        print("No box prediction enabled (text-only training)")

    # Optimizer (include all model params)
    opt_params = list(encoder.parameters()) + list(decoder.parameters()) + list(ctc_head.parameters())
    if detector is not None:
        opt_params += list(detector.parameters())
    optimizer = optim.AdamW(opt_params, lr=LR, weight_decay=WEIGHT_DECAY)

    # Losses
    ce_loss = nn.CrossEntropyLoss(ignore_index=pad_id)
    ctc_loss_fn = nn.CTCLoss(blank=pad_id, zero_infinity=True)

    # Smoke test once
    smoke_test((unet, encoder, decoder, ctc_head), dataset, vocab, DEVICE)

    # Optional CTC warmup
    if CTC_WARMUP_EPOCHS > 0:
        print("=== Starting CTC warmup ===")
        for epoch in range(CTC_WARMUP_EPOCHS):
            encoder.train()
            total_loss = 0.0
            n_batches = 0
            for batch in dataloader:
                if n_batches % 10 == 0:
                    print(f"  CTC warmup epoch {epoch+1}, batch {n_batches}...")    
                images = batch["image"].to(DEVICE)
                text_ids = batch["text_ids"].to(DEVICE) if batch["text_ids"] is not None else None
                if text_ids is None:
                    continue

                optimizer.zero_grad()
                enc_seq, enc_mask = encoder(images, orientation="horizontal")
                logits = ctc_head(enc_seq)  # (B, T, V)
                log_probs = logits.log_softmax(dim=-1).permute(1,0,2).contiguous()  # (T,B,V)

                targets_concat, target_lengths = prepare_ctc_targets(text_ids, pad_id, sos_id, eos_id, DEVICE)
                if targets_concat is None:
                    continue
                input_lengths = torch.full((enc_seq.size(0),), fill_value=enc_seq.size(1), dtype=torch.long, device=DEVICE)

                loss = ctc_loss_fn(log_probs, targets_concat, input_lengths, target_lengths)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), GRAD_CLIP)
                torch.nn.utils.clip_grad_norm_(ctc_head.parameters(), GRAD_CLIP)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            if n_batches > 0:
                print(f"CTC epoch {epoch+1}/{CTC_WARMUP_EPOCHS} avg_loss={total_loss/n_batches:.4f}")
            else:
                print(f"CTC epoch {epoch+1}: no valid targets found")

    # Main attention training
    print("=== Starting attention training ===")
    for epoch in range(NUM_EPOCHS):
        encoder.train(); decoder.train()
        if detector is not None:
            detector.train()
        
        tf_ratio = scheduled_teacher_forcing(epoch, NUM_EPOCHS, start=1.0, end=0.1, schedule="exp")
        total_loss = 0.0
        total_seq_loss = 0.0
        total_det_loss = 0.0
        total_roi_loss = 0.0
        n_batches = 0
        
        for batch in dataloader:
            images = batch["image"].to(DEVICE)
            text_ids = batch["text_ids"].to(DEVICE) if batch["text_ids"] is not None else None
            boxes = batch["boxes"]
            labels = batch["labels"]
            
            if text_ids is None:
                continue

            input_seq = text_ids[:, :-1]
            targets = text_ids[:, 1:]

            optimizer.zero_grad()
            
            # Get UNet features for detection if needed
            feats2d = None
            if detector is not None:
                feats2d = unet(images)
            
            enc_outputs, enc_mask = encoder(images, orientation="horizontal")
            
            # Decoder forward (now returns 4-tuple with predicted_boxes)
            decoder_output = decoder(input_seq=input_seq, enc_outputs=enc_outputs, enc_mask=enc_mask,
                                    teacher_forcing_ratio=tf_ratio, targets=targets, eos_id=eos_id)
            
            # Handle both 3-tuple (old) and 4-tuple (new with predicted_boxes)
            if len(decoder_output) == 4:
                logits, hidden, attn, predicted_boxes = decoder_output
            else:
                logits, hidden, attn = decoder_output
                predicted_boxes = None
            
            B, T_dec, V = logits.shape
            
            # Sequence loss
            loss_seq = ce_loss(logits.reshape(-1, V), targets.reshape(-1))
            loss = loss_seq
            total_seq_loss += loss_seq.item()
            
            # Detection loss (Option 1: DetectorHead)
            if detector is not None and feats2d is not None:
                det_pred = detector(feats2d)
                H_out, W_out = det_pred['cls'].shape[2:]
                gt_heatmap, gt_bbox, gt_cls = build_detection_targets(
                    boxes, labels, (H_out, W_out), (512, 512), DEVICE, sigma=DETECTOR_HEATMAP_SIGMA
                )
                loss_det, (l_heat, l_bbox, l_cls) = compute_detection_losses(
                    det_pred, gt_heatmap, gt_bbox, gt_cls, weights=(1.0, 1.0, 1.0)
                )
                loss = loss + DETECTION_LOSS_WEIGHT * loss_det
                total_det_loss += loss_det.item()
            
            # ROI box loss (Option 2: attention-based boxes)
            if USE_ROI_ATTENTION and predicted_boxes is not None:
                # Convert boxes list to padded tensor
                gt_boxes_padded = []
                max_boxes = max(b.shape[0] for b in boxes)
                for box_tensor in boxes:
                    if box_tensor.numel() == 0:
                        box_tensor = torch.zeros((1, 4), device=DEVICE)
                    pad_size = max_boxes - box_tensor.shape[0]
                    if pad_size > 0:
                        box_tensor = torch.cat([
                            box_tensor.to(DEVICE),
                            torch.zeros((pad_size, 4), device=DEVICE)
                        ], dim=0)
                    gt_boxes_padded.append(box_tensor)
                gt_boxes_batch = torch.stack(gt_boxes_padded, dim=0)  # (B, N, 4)
                
                loss_roi = compute_roi_box_loss(predicted_boxes, gt_boxes_batch)
                loss = loss + ROI_BOX_LOSS_WEIGHT * loss_roi
                total_roi_loss += loss_roi.item()
            
            loss.backward()

            torch.nn.utils.clip_grad_norm_(encoder.parameters(), GRAD_CLIP)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), GRAD_CLIP)
            if detector is not None:
                torch.nn.utils.clip_grad_norm_(detector.parameters(), GRAD_CLIP)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = (total_loss / n_batches) if n_batches > 0 else 0.0
        avg_seq = (total_seq_loss / n_batches) if n_batches > 0 else 0.0
        avg_det = (total_det_loss / n_batches) if n_batches > 0 else 0.0
        avg_roi = (total_roi_loss / n_batches) if n_batches > 0 else 0.0
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} TF={tf_ratio:.3f} Total={avg_loss:.4f} "
              f"Seq={avg_seq:.4f} Det={avg_det:.4f} ROI={avg_roi:.4f}")
        
        # Run validation every VALIDATION_FREQ epochs (if enabled)
        if RUN_VALIDATION and (epoch + 1) % VALIDATION_FREQ == 0:
            print(f"  Running validation...")
            val_metrics = validate(
                encoder, decoder, detector, dataloader, vocab, DEVICE,
                use_detector_head=USE_DETECTOR_HEAD,
                use_roi_attention=USE_ROI_ATTENTION,
                detection_loss_weight=DETECTION_LOSS_WEIGHT,
                roi_box_loss_weight=ROI_BOX_LOSS_WEIGHT,
                detector_heatmap_sigma=DETECTOR_HEATMAP_SIGMA,
                max_batches=VALIDATION_BATCHES
            )
            print(f"  VAL Loss={val_metrics['loss']:.4f} CER={val_metrics['cer']:.4f} "
                  f"DetLoss={val_metrics['det_loss']:.4f} ROILoss={val_metrics['roi_loss']:.4f}")

        # Save checkpoint
        ckpt_dir = Path(CHECKPOINT_DIR)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"att_epoch{epoch+1}.pt"
        ckpt_dict = {
            'epoch': epoch+1,
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
            'ctc_head': ctc_head.state_dict(),
            'optimizer': optimizer.state_dict(),
            'vocab': vocab.char2id,
            'use_detector_head': USE_DETECTOR_HEAD,
            'use_roi_attention': USE_ROI_ATTENTION,
        }
        if detector is not None:
            ckpt_dict['detector'] = detector.state_dict()
        
        torch.save(ckpt_dict, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

    print("Training finished.")

if __name__ == "__main__":
    train()
