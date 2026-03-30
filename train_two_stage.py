"""Two-stage training pipeline: Stage 1 (detection) + Stage 2 (classification)"""
import sys
import json
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
    FOCAL_ALPHA, FOCAL_GAMMA, POS_WEIGHT, BBOX_WEIGHT, USE_ROI_ATTENTION, ROI_BOX_LOSS_WEIGHT,
    STAGE2_USE_CTC_WARMUP,
    STAGE2_AUX_CTC_WEIGHT,
    STAGE2_USE_GT_BOXES,
    STAGE2_CURRICULUM_ENABLE,
    STAGE2_CURRICULUM_GT_EPOCHS,
    STAGE2_GRAD_ACCUMULATION_STEPS,
    STAGE2_READING_ORDER_POLICY,
    STAGE2_USE_ATTN_CENTROID_BOXES,
    STAGE2_ARCH_GUARDRAIL_STRICT,
    ROI_POOL_SIZE,
    ROI_EMBED_DIM,
    CONTEXT_HIDDEN_DIM,
    STAGE2_DET_CONFIDENCE,
    STAGE2_DET_TOP_K,
    STAGE2_DET_NMS_IOU,
)
from model.kuronet import UNet, DetectorHead, ROISequenceEncoder, ROIContextEncoder
from model.kuronet.encoder_wrapper import EncoderWrapper
from model.kuronet.decoder.attention import SeqDecoderAttention
from utils import KuzushijiDataset
from utils.detection_utils import build_detection_targets, compute_roi_box_loss
from utils.focal_loss import focal_loss_heatmap
from utils.text_normalization import render_tokens
from utils.training_helpers import (
    collate_fn,
    masked_bbox_smoothl1_loss,
    prune_existing_checkpoints,
    prune_to_keep_last_n,
    scheduled_teacher_forcing,
    validate_detector,
)
from utils.vocab import VocabManager
from validate_stage1 import extract_boxes_from_heatmap


def _sort_boxes_reading_order(boxes, orientation):
    if boxes is None or len(boxes) == 0:
        return []
    if orientation == "vertical":
        idx = sorted(range(len(boxes)), key=lambda i: (-boxes[i][0], boxes[i][1]))
    else:
        idx = sorted(range(len(boxes)), key=lambda i: (boxes[i][1], boxes[i][0]))
    return [[float(v) for v in boxes[i]] for i in idx]


def _infer_reading_orientation_from_boxes(boxes):
    """Infer reading direction from nearest-neighbor center distances; default vertical."""
    if boxes is None or len(boxes) < 2:
        return "vertical"
    centers = torch.tensor(
        [[0.5 * (float(b[0]) + float(b[2])), 0.5 * (float(b[1]) + float(b[3]))] for b in boxes],
        dtype=torch.float32,
    )
    dx = torch.cdist(centers[:, :1], centers[:, :1], p=1)
    dy = torch.cdist(centers[:, 1:2], centers[:, 1:2], p=1)
    inf = torch.tensor(float("inf"), dtype=torch.float32)
    dx = dx + torch.eye(dx.size(0), dtype=torch.float32) * inf
    dy = dy + torch.eye(dy.size(0), dtype=torch.float32) * inf
    mean_min_dx = float(dx.min(dim=1).values.mean().item())
    mean_min_dy = float(dy.min(dim=1).values.mean().item())
    return "vertical" if mean_min_dy <= mean_min_dx else "horizontal"


def _resolve_sort_orientation(orientation_hint, boxes, reading_order_policy):
    policy = str(reading_order_policy or "annotation").strip().lower()
    if policy == "inferred":
        return _infer_reading_orientation_from_boxes(boxes)
    if policy == "auto":
        if orientation_hint in ("vertical", "horizontal"):
            return orientation_hint
        return _infer_reading_orientation_from_boxes(boxes)
    # default: annotation
    if orientation_hint in ("vertical", "horizontal"):
        return orientation_hint
    return "vertical"


def _build_boxes_for_encoder(images, boxes_batch, orientations, use_gt_boxes, unet, detector, reading_order_policy="annotation"):
    """Prepare box sequences for ROI encoder: GT boxes or Stage-1 predicted boxes."""
    if use_gt_boxes:
        out = []
        for b in boxes_batch or []:
            if b is None:
                out.append(torch.empty((0, 4), dtype=torch.float32, device=DEVICE))
            else:
                out.append(b.to(DEVICE, dtype=torch.float32))
        return out

    with torch.no_grad():
        features = unet(images)
        det_out = detector(features)
        heat_probs = torch.sigmoid(det_out["heatmap"])
        bbox_reg = det_out["bbox"]
        _, _, hf, wf = features.shape

        out = []
        for i in range(images.size(0)):
            pred_boxes_i, _, _ = extract_boxes_from_heatmap(
                heatmap_probs=heat_probs[i:i + 1],
                bbox_reg=bbox_reg[i:i + 1],
                confidence_thresh=STAGE2_DET_CONFIDENCE,
                output_size=(hf, wf),
                image_size=IMAGE_SIZE,
                top_k=STAGE2_DET_TOP_K,
                nms_iou=STAGE2_DET_NMS_IOU,
                min_box_size=4.0,
                debug=False,
            )
            orientation_hint = orientations[i] if orientations is not None and i < len(orientations) else None
            sort_orientation = _resolve_sort_orientation(orientation_hint, pred_boxes_i, reading_order_policy)
            pred_boxes_i = _sort_boxes_reading_order(pred_boxes_i, sort_orientation)
            if len(pred_boxes_i) == 0:
                out.append(torch.empty((0, 4), dtype=torch.float32, device=DEVICE))
            else:
                out.append(torch.tensor(pred_boxes_i, dtype=torch.float32, device=DEVICE))
        return out

def _get_stage2_trainable_params(encoder, roi_sequence_encoder, context_encoder, decoder, ctc_head=None):
    params = []
    modules = [encoder, roi_sequence_encoder, context_encoder, decoder]
    if ctc_head is not None:
        modules.append(ctc_head)

    for module in modules:
        params.extend([p for p in module.parameters() if p.requires_grad])

    return params


def _prepare_ctc_targets(text_ids, pad_id, sos_id, eos_id, input_lengths):
    """Build CTC targets per batch and keep only rows valid for CTC."""
    targets = []
    target_lengths = []
    keep_indices = []
    skipped_too_long = 0
    total_rows = int(text_ids.size(0))

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

        # Skip samples where CTC cannot align target to available ROI steps.
        if ids.numel() > int(input_lengths[i].item()):
            skipped_too_long += 1
            continue

        targets.append(ids)
        target_lengths.append(ids.numel())
        keep_indices.append(i)

    if len(targets) == 0:
        diag = {
            "total_rows": total_rows,
            "kept_rows": 0,
            "skipped_too_long": skipped_too_long,
            "mean_input_len_kept": 0.0,
            "mean_target_len_kept": 0.0,
        }
        return None, None, None, diag

    targets_concat = torch.cat(targets)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long, device=text_ids.device)
    keep_indices = torch.tensor(keep_indices, dtype=torch.long, device=text_ids.device)
    diag = {
        "total_rows": total_rows,
        "kept_rows": int(keep_indices.numel()),
        "skipped_too_long": skipped_too_long,
        "mean_input_len_kept": float(input_lengths[keep_indices].float().mean().item()) if keep_indices.numel() > 0 else 0.0,
        "mean_target_len_kept": float(target_lengths.float().mean().item()) if target_lengths.numel() > 0 else 0.0,
    }
    return targets_concat, target_lengths, keep_indices, diag


def _validate_stage2_guardrails(num_epochs):
    policy = str(STAGE2_READING_ORDER_POLICY).strip().lower()
    allowed = {"annotation", "inferred", "auto"}
    issues = []

    if policy not in allowed:
        issues.append(f"invalid STAGE2_READING_ORDER_POLICY={STAGE2_READING_ORDER_POLICY}")
    if not (isinstance(ROI_POOL_SIZE, tuple) and len(ROI_POOL_SIZE) == 2):
        issues.append(f"ROI_POOL_SIZE must be tuple(len=2), got {ROI_POOL_SIZE}")
    if int(STAGE2_CURRICULUM_GT_EPOCHS) < 0:
        issues.append(f"STAGE2_CURRICULUM_GT_EPOCHS must be >= 0, got {STAGE2_CURRICULUM_GT_EPOCHS}")
    if int(STAGE2_GRAD_ACCUMULATION_STEPS) < 1:
        issues.append(f"STAGE2_GRAD_ACCUMULATION_STEPS must be >= 1, got {STAGE2_GRAD_ACCUMULATION_STEPS}")

    warn_only = None
    if STAGE2_CURRICULUM_ENABLE and int(STAGE2_CURRICULUM_GT_EPOCHS) >= int(num_epochs):
        warn_only = (
            f"Curriculum GT epochs ({int(STAGE2_CURRICULUM_GT_EPOCHS)}) >= num_epochs ({int(num_epochs)}); "
            "run will stay in GT box mode and not switch to predicted boxes."
        )

    if issues and STAGE2_ARCH_GUARDRAIL_STRICT:
        raise ValueError("Stage2 architecture guardrail violation: " + " | ".join(issues))
    if issues:
        print("[WARN] Stage2 architecture guardrail issues: " + " | ".join(issues))
    if warn_only:
        print("[WARN] " + warn_only)

    return {
        "strict": bool(STAGE2_ARCH_GUARDRAIL_STRICT),
        "issues": issues,
        "warning": warn_only,
        "reading_order_policy": policy,
    }

def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + cost,
            ))
        prev = curr
    return prev[-1]

def _decode_text_from_ids(ids, vocab):
    chars = vocab.decode([int(x) for x in ids], remove_special=True)
    return render_tokens(chars)


def _truncate_at_eos(ids, eos_id):
    out = []
    for tok in ids:
        out.append(int(tok))
        if int(tok) == eos_id:
            break
    return out

def validate_sequence_stage(
    encoder,
    roi_sequence_encoder,
    context_encoder,
    decoder,
    dataloader,
    vocab,
    ce_loss,
    use_gt_boxes,
    unet,
    detector,
    reading_order_policy="annotation",
    max_decode_len=256,
):
    encoder.eval()
    roi_sequence_encoder.eval()
    context_encoder.eval()
    decoder.eval()

    total_loss = 0.0
    total_cer = 0.0
    total_exact = 0
    total_samples = 0
    n_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validate Stage2", leave=False):
            images = batch["image"].to(DEVICE)
            text_ids = batch["text_ids"].to(DEVICE) if batch["text_ids"] is not None else None
            orientations = batch.get("orientations", None)
            boxes_batch = batch.get("boxes", None)

            if text_ids is None:
                continue

            text_ids_present = batch.get("text_ids_present", None)
            if text_ids_present is not None:
                valid_idx = text_ids_present.to(DEVICE).nonzero(as_tuple=False).squeeze(1)
                if valid_idx.numel() == 0:
                    continue

                images = images.index_select(0, valid_idx)
                text_ids = text_ids.index_select(0, valid_idx)
                valid_idx_cpu = valid_idx.detach().cpu().tolist()

                if orientations is not None:
                    orientations = [orientations[i] for i in valid_idx_cpu]
                if boxes_batch is not None:
                    boxes_batch = [boxes_batch[i] for i in valid_idx_cpu]

            input_seq = text_ids[:, :-1]
            targets = text_ids[:, 1:]

            boxes_for_encoder = _build_boxes_for_encoder(
                images=images,
                boxes_batch=boxes_batch,
                orientations=orientations,
                use_gt_boxes=use_gt_boxes,
                unet=unet,
                detector=detector,
                reading_order_policy=reading_order_policy,
            )

            feats_2d = encoder(images, return_2d=True)
            roi_seq, roi_mask = roi_sequence_encoder(feats_2d, boxes_for_encoder, image_size=IMAGE_SIZE)
            enc_outputs, enc_mask = context_encoder(roi_seq, roi_mask)

            # teacher-forced validation loss
            decoder_output = decoder(
                input_seq=input_seq,
                enc_outputs=enc_outputs,
                enc_mask=enc_mask,
                teacher_forcing_ratio=1.0,
                targets=targets,
                eos_id=vocab.eos_id,
                image_size=(images.shape[2], images.shape[3]),
            )

            if len(decoder_output) == 4:
                logits, _, _, _ = decoder_output
            else:
                logits, _, _ = decoder_output

            B, T, V = logits.shape
            loss = ce_loss(logits.reshape(-1, V), targets.reshape(-1))
            total_loss += float(loss.item())
            n_batches += 1

            # free decoding for CER / exact match
            decoder_free = decoder(
                input_seq=None,
                enc_outputs=enc_outputs,
                enc_mask=enc_mask,
                teacher_forcing_ratio=0.0,
                targets=None,
                sos_id=vocab.sos_id,
                eos_id=vocab.eos_id,
                max_len=max_decode_len,
                image_size=(images.shape[2], images.shape[3]),
            )

            if len(decoder_free) == 4:
                free_logits, _, _, _ = decoder_free
            else:
                free_logits, _, _ = decoder_free

            pred_ids_batch = free_logits.argmax(dim=-1).detach().cpu().tolist()
            gt_ids_batch = text_ids.detach().cpu().tolist()

            for pred_ids, gt_ids in zip(pred_ids_batch, gt_ids_batch):
                pred_ids = _truncate_at_eos(pred_ids, vocab.eos_id)
                pred_text = _decode_text_from_ids(pred_ids, vocab)
                gt_text = _decode_text_from_ids(gt_ids, vocab)

                dist = _edit_distance(pred_text, gt_text)
                cer = dist / max(1, len(gt_text))

                total_cer += cer
                total_exact += int(pred_text == gt_text)
                total_samples += 1

    denom_batches = max(1, n_batches)
    denom_samples = max(1, total_samples)

    return {
        "val_loss": total_loss / denom_batches,
        "val_cer": total_cer / denom_samples,
        "val_exact": total_exact / denom_samples,
        "val_samples": total_samples,
    }

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
    bbox_radius = 0
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
                    sigma=DETECTOR_HEATMAP_SIGMA,
                    bbox_radius=bbox_radius,
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


def train_sequence_stage(detector_ckpt_path, num_epochs=10, lr=None, checkpoint_dir=None, use_ctc_warmup=False):
    """Stage 2: Train Encoder + Decoder for sequence prediction with context.
    
    Uses detected boxes to guide attention and sequence prediction.
    Combines spatial detection (Stage 1) with contextual understanding.
    Architecture:
        image -> UNet backbone -> 2D projected features
             -> ROISequenceEncoder (ordered candidate boxes)
             -> ROIContextEncoder
             -> attention decoder

    Modes:
        Oracle mode:
            Stage 2 uses GT boxes from dataset
        Detector mode:
            Stage 2 uses Stage 1 predicted boxes
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
    guardrail_info = _validate_stage2_guardrails(num_epochs=num_epochs)
    
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
    train_dataset = KuzushijiDataset(
        Path(DATA_DIR),
        vocab=vocab,
        use_sequences=True,
        resize=IMAGE_SIZE,
        split="train",
    )
    val_dataset = KuzushijiDataset(
        Path(DATA_DIR),
        vocab=vocab,
        use_sequences=True,
        resize=IMAGE_SIZE,
        split="val",
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=lambda b: collate_fn(b, pad_id),
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=NUM_WORKERS > 0,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=lambda b: collate_fn(b, pad_id),
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=NUM_WORKERS > 0,
    )
    
    print(f"Stage 2 datasets -> train: {len(train_dataset)} | val: {len(val_dataset)}")
    # Build and load detector (frozen for spatial guidance)
    unet = UNet(in_channels=3, base_features=32).to(DEVICE)
    detector = DetectorHead(in_ch=32, num_classes=vocab_size).to(DEVICE)
    
    checkpoint = torch.load(detector_ckpt_path, map_location=DEVICE)
    unet.load_state_dict(checkpoint['unet_state_dict'])
    detector.load_state_dict(checkpoint['detector_state_dict'])
    
    # Freeze detector - we only use it for guidance
    for p in unet.parameters():
        p.requires_grad = False
    for p in detector.parameters():
        p.requires_grad = False
    
    unet.eval()
    detector.eval()
    # -------------------------
    # Stage 2 modules
    # -------------------------

    encoder = EncoderWrapper(backbone=unet, in_channels=32, enc_dim=256).to(DEVICE)
    roi_sequence_encoder = ROISequenceEncoder(
        in_dim=256,
        roi_size=ROI_POOL_SIZE,
        out_dim=ROI_EMBED_DIM,
    ).to(DEVICE)
    context_encoder = ROIContextEncoder(
        in_dim=ROI_EMBED_DIM,
        hidden_dim=CONTEXT_HIDDEN_DIM,
        out_dim=CONTEXT_HIDDEN_DIM,
    ).to(DEVICE)

    decoder = SeqDecoderAttention(
        embed_dim=64,
        hidden_dim=256,
        vocab_size=vocab_size,
        enc_dim=CONTEXT_HIDDEN_DIM,
        num_layers=1,
        init_from_encoder=True,
        sampling_method="argmax",
        use_roi_attention=USE_ROI_ATTENTION,
        use_attn_centroid_boxes=STAGE2_USE_ATTN_CENTROID_BOXES,
    ).to(DEVICE)

    ctc_blank_id = vocab_size
    ctc_head = nn.Linear(CONTEXT_HIDDEN_DIM, vocab_size + 1).to(DEVICE)
    
    # -------------------------
    # optimizer / scheduler / scaler
    # -------------------------

    optimizer = optim.AdamW(
        list(encoder.parameters())
        + list(roi_sequence_encoder.parameters())
        + list(context_encoder.parameters())
        + list(decoder.parameters())
        + list(ctc_head.parameters()),
        lr=lr, 
        weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    amp_enabled = USE_MIXED_PRECISION and str(DEVICE).startswith("cuda")
    scaler = amp.GradScaler() if amp_enabled else None
    stage2_accum_steps = max(1, int(STAGE2_GRAD_ACCUMULATION_STEPS))
    
    # Losses
    ce_loss = nn.CrossEntropyLoss(ignore_index=pad_id)
    trainable_params = _get_stage2_trainable_params(
    encoder, roi_sequence_encoder, context_encoder, decoder, ctc_head
    )

    run_config = {
        "stage2_use_gt_boxes": bool(STAGE2_USE_GT_BOXES),
        "stage2_decode_boxes": "n/a_training",
        "stage2_curriculum_enable": bool(STAGE2_CURRICULUM_ENABLE),
        "stage2_curriculum_gt_epochs": int(STAGE2_CURRICULUM_GT_EPOCHS),
        "stage2_grad_accumulation_steps": int(stage2_accum_steps),
        "stage2_reading_order_policy": STAGE2_READING_ORDER_POLICY,
        "stage2_use_attn_centroid_boxes": bool(STAGE2_USE_ATTN_CENTROID_BOXES),
        "stage2_arch_guardrail_strict": bool(STAGE2_ARCH_GUARDRAIL_STRICT),
        "stage2_arch_guardrail_issues": guardrail_info["issues"],
        "stage2_arch_guardrail_warning": guardrail_info["warning"],
        "use_roi_attention": bool(USE_ROI_ATTENTION),
        "stage2_aux_ctc_weight": float(STAGE2_AUX_CTC_WEIGHT),
        "roi_box_loss_weight": float(ROI_BOX_LOSS_WEIGHT),
        "use_ctc_warmup": bool(use_ctc_warmup),
        "detector_checkpoint": str(detector_ckpt_path),
        "checkpoint_dir": str(checkpoint_dir),
    }
    run_config_path = checkpoint_dir / "stage2_train_run_config.json"
    with open(run_config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)
    print(f"Saved Stage2 run config: {run_config_path}")
    
    # CTC Warmup (optional)
    if use_ctc_warmup:
        print(f"Stage 2: CTC Warmup for 2 epochs...")
        ctc_loss_fn = nn.CTCLoss(blank=ctc_blank_id, zero_infinity=True)
        use_gt_boxes_warmup = True if STAGE2_CURRICULUM_ENABLE else STAGE2_USE_GT_BOXES
        
        for epoch in range(2):
            encoder.train()
            roi_sequence_encoder.train()
            context_encoder.train()         
            pbar = tqdm(train_dataloader, desc=f"CTC Warmup {epoch+1}/2")
            optimizer.zero_grad(set_to_none=True)
            grad_counter = 0
            for batch in pbar:
                images = batch["image"].to(DEVICE)
                text_ids = batch["text_ids"].to(DEVICE) if batch["text_ids"] is not None else None
                boxes_batch = batch.get("boxes", None)
                orientations = batch.get("orientations", None)
                if text_ids is None:
                    continue

                # collate_fn packs only valid text rows; keep images aligned to that subset
                text_ids_present = batch.get("text_ids_present", None)
                if text_ids_present is not None:
                    valid_idx = text_ids_present.to(DEVICE).nonzero(as_tuple=False).squeeze(1)
                    if valid_idx.numel() == 0:
                        continue
                    images = images.index_select(0, valid_idx)
                    valid_idx_cpu = valid_idx.detach().cpu().tolist()
                    if orientations is not None:
                        orientations = [orientations[i] for i in valid_idx_cpu]
                    if boxes_batch is not None:
                        boxes_batch = [boxes_batch[i] for i in valid_idx_cpu]
                
                boxes_for_encoder = _build_boxes_for_encoder(
                    images=images,
                    boxes_batch=boxes_batch,
                    orientations=orientations,
                    use_gt_boxes=use_gt_boxes_warmup,
                    unet=unet,
                    detector=detector,
                    reading_order_policy=STAGE2_READING_ORDER_POLICY,
                )

                with amp.autocast(enabled=amp_enabled):
                    feats_2d = encoder(images, return_2d=True)
                    roi_seq, roi_mask = roi_sequence_encoder(feats_2d, boxes_for_encoder, image_size=IMAGE_SIZE)
                    enc_seq, enc_mask = context_encoder(roi_seq, roi_mask)
                    logits = ctc_head(enc_seq)  # (B, T, V+1), last class is CTC blank
                    log_probs = logits.float().log_softmax(dim=-1).permute(1, 0, 2).contiguous()  # (T,B,V+1)
                    input_lengths = roi_mask.sum(dim=1).clamp(min=1).to(dtype=torch.long)

                    targets_concat, target_lengths, keep_indices, _ = _prepare_ctc_targets(
                        text_ids=text_ids,
                        pad_id=pad_id,
                        sos_id=sos_id,
                        eos_id=eos_id,
                        input_lengths=input_lengths,
                    )

                    if targets_concat is None:
                        continue

                    loss_raw = ctc_loss_fn(
                        log_probs[:, keep_indices, :],
                        targets_concat,
                        input_lengths[keep_indices],
                        target_lengths,
                    )
                    loss = loss_raw / stage2_accum_steps
                
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                grad_counter += 1
                if grad_counter % stage2_accum_steps == 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                pbar.set_postfix({"ctc_loss": float(loss_raw.item())})

            if grad_counter % stage2_accum_steps != 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
    
    # -------------------------
    # Main Stage 2 training
    # -------------------------
    print(f"Stage 2: Training ROI-based sequence model for {num_epochs} epochs...")
    print(f"Mode: {'Oracle (GT boxes)' if STAGE2_USE_GT_BOXES else 'Detector-guided (predicted boxes)'}")
    if STAGE2_CURRICULUM_ENABLE:
        print(f"Curriculum: GT boxes for first {int(STAGE2_CURRICULUM_GT_EPOCHS)} epochs, then predicted boxes")
    print(f"ROI attention box loss: {'ON' if USE_ROI_ATTENTION else 'OFF'}")
    print(f"Aux CTC weight: {STAGE2_AUX_CTC_WEIGHT:.3f}")
    print(f"Grad accumulation (Stage2): {stage2_accum_steps}")

    ctc_loss_fn = nn.CTCLoss(blank=ctc_blank_id, zero_infinity=True)
    ctc_dominance_streak = 0

    best_val_cer = None

    curriculum_gt_epochs = max(0, int(STAGE2_CURRICULUM_GT_EPOCHS))

    def _use_gt_boxes_for_epoch(epoch_idx):
        if STAGE2_CURRICULUM_ENABLE:
            return epoch_idx < curriculum_gt_epochs
        return STAGE2_USE_GT_BOXES

    for epoch in range(num_epochs):
        encoder.train()
        roi_sequence_encoder.train()
        context_encoder.train()
        decoder.train()
        unet.eval()
        detector.eval()
        use_gt_boxes_epoch = _use_gt_boxes_for_epoch(epoch)
        epoch_box_source = "gt" if use_gt_boxes_epoch else "pred"
        print(f"Epoch {epoch+1}: box_source={epoch_box_source} | reading_order_policy={STAGE2_READING_ORDER_POLICY}")
        
        tf_ratio = scheduled_teacher_forcing(epoch,
            num_epochs,
            start=1.0,
            end=0.1,
            schedule="exp")
        
        total_loss = 0.0
        total_seq_loss = 0.0
        total_ctc_loss = 0.0
        total_box_loss = 0.0
        ctc_total_rows = 0
        ctc_kept_rows = 0
        ctc_skipped_too_long = 0
        ctc_input_len_sum = 0.0
        ctc_target_len_sum = 0.0
        n_batches = 0
        optimizer.zero_grad(set_to_none=True)
        grad_counter = 0
        
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", mininterval=60.0)
        for batch in pbar:
            images = batch["image"].to(DEVICE)
            text_ids = batch["text_ids"].to(DEVICE) if batch["text_ids"] is not None else None
            orientations = batch.get("orientations", None)
            boxes_batch = batch.get("boxes", None)
            
            if text_ids is None:
                continue

            # collate_fn packs only valid text rows; keep image/orientation tensors aligned
            text_ids_present = batch.get("text_ids_present", None)
            if text_ids_present is not None:
                valid_idx = text_ids_present.to(DEVICE).nonzero(as_tuple=False).squeeze(1)
                if valid_idx.numel() == 0:
                    continue
                images = images.index_select(0, valid_idx)
                text_ids = text_ids.index_select(0, valid_idx)
                valid_idx_cpu = valid_idx.detach().cpu().tolist()

                if orientations is not None:
                    orientations = [orientations[i] for i in valid_idx_cpu]
                if boxes_batch is not None:
                    boxes_batch = [boxes_batch[i] for i in valid_idx_cpu]

            gt_boxes_for_loss = []
            if boxes_batch is not None:
                for b in boxes_batch:
                    if b is None:
                        gt_boxes_for_loss.append(torch.empty((0, 4), dtype=torch.float32, device=DEVICE))
                    else:
                        gt_boxes_for_loss.append(b.to(DEVICE, dtype=torch.float32))
            else:
                gt_boxes_for_loss = [torch.empty((0, 4), dtype=torch.float32, device=DEVICE) for _ in range(images.size(0))]

            boxes_for_encoder = _build_boxes_for_encoder(
                images=images,
                boxes_batch=boxes_batch,
                orientations=orientations,
                use_gt_boxes=use_gt_boxes_epoch,
                unet=unet,
                detector=detector,
                reading_order_policy=STAGE2_READING_ORDER_POLICY,
            )

            input_seq = text_ids[:, :-1]
            targets = text_ids[:, 1:]

            with amp.autocast(enabled=amp_enabled):
                feats_2d = encoder(images, return_2d=True)
                roi_seq, roi_mask = roi_sequence_encoder(feats_2d, boxes_for_encoder, image_size=IMAGE_SIZE)
                enc_outputs, enc_mask = context_encoder(roi_seq, roi_mask)
                
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
                predicted_boxes = None
                if len(decoder_output) == 4:
                    logits, _, _, predicted_boxes = decoder_output
                else:
                    logits, _, _ = decoder_output
                
                B, T_dec, V = logits.shape
                
                # Sequence loss (contextual understanding)
                loss_seq = ce_loss(logits.reshape(-1, V), targets.reshape(-1))

                if STAGE2_AUX_CTC_WEIGHT > 0.0:
                    ctc_logits = ctc_head(enc_outputs)  # (B, T_roi, V+1)
                    ctc_log_probs = ctc_logits.float().log_softmax(dim=-1).permute(1, 0, 2).contiguous()
                    ctc_input_lengths = enc_mask.sum(dim=1).clamp(min=1).to(dtype=torch.long)
                    targets_concat, target_lengths, keep_indices, ctc_diag = _prepare_ctc_targets(
                        text_ids=text_ids,
                        pad_id=pad_id,
                        sos_id=sos_id,
                        eos_id=eos_id,
                        input_lengths=ctc_input_lengths,
                    )
                    ctc_total_rows += int(ctc_diag["total_rows"])
                    ctc_kept_rows += int(ctc_diag["kept_rows"])
                    ctc_skipped_too_long += int(ctc_diag["skipped_too_long"])
                    ctc_input_len_sum += float(ctc_diag["mean_input_len_kept"]) * int(ctc_diag["kept_rows"])
                    ctc_target_len_sum += float(ctc_diag["mean_target_len_kept"]) * int(ctc_diag["kept_rows"])
                    if targets_concat is None:
                        loss_ctc = torch.tensor(0.0, device=DEVICE)
                    else:
                        loss_ctc = ctc_loss_fn(
                            ctc_log_probs[:, keep_indices, :],
                            targets_concat,
                            ctc_input_lengths[keep_indices],
                            target_lengths,
                        )
                else:
                    loss_ctc = torch.tensor(0.0, device=DEVICE)

                if USE_ROI_ATTENTION and predicted_boxes is not None and len(gt_boxes_for_loss) == B:
                    loss_box = compute_roi_box_loss(
                        predicted_boxes,
                        gt_boxes_for_loss,
                        reduction='mean',
                        iou_weight=0.0,
                        use_x_only=True,
                        coord_scale=float(images.shape[3]),
                    )
                else:
                    loss_box = torch.tensor(0.0, device=DEVICE)

                loss_raw = loss_seq + STAGE2_AUX_CTC_WEIGHT * loss_ctc + ROI_BOX_LOSS_WEIGHT * loss_box
                loss = loss_raw / stage2_accum_steps
            
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            grad_counter += 1
            if grad_counter % stage2_accum_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            total_seq_loss += loss_seq.item()
            total_ctc_loss += loss_ctc.item()
            total_box_loss += loss_box.item()
            total_loss += loss_raw.item()
            n_batches += 1
            pbar.set_postfix({
                "loss": total_loss / n_batches,
                "seq": total_seq_loss / n_batches,
                "ctc": total_ctc_loss / n_batches,
                "box": total_box_loss / n_batches,
                "wctc": STAGE2_AUX_CTC_WEIGHT * (total_ctc_loss / n_batches),
                "wbox": ROI_BOX_LOSS_WEIGHT * (total_box_loss / n_batches),
                "tf": tf_ratio,
            })

        if grad_counter % stage2_accum_steps != 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        scheduler.step()
        
        # -------------------------
        # validation
        # -------------------------
        val_metrics = validate_sequence_stage(
            encoder=encoder,
            roi_sequence_encoder=roi_sequence_encoder,
            context_encoder=context_encoder,
            decoder=decoder,
            dataloader=val_dataloader,
            vocab=vocab,
            ce_loss=ce_loss,
            use_gt_boxes=use_gt_boxes_epoch,
            unet=unet,
            detector=detector,
            reading_order_policy=STAGE2_READING_ORDER_POLICY,
            max_decode_len=256,
        )

        train_loss_mean = total_loss / max(1, n_batches)
        train_seq_mean = total_seq_loss / max(1, n_batches)
        train_ctc_mean = total_ctc_loss / max(1, n_batches)
        train_box_mean = total_box_loss / max(1, n_batches)
        train_wseq_mean = train_seq_mean
        train_wctc_mean = STAGE2_AUX_CTC_WEIGHT * train_ctc_mean
        train_wbox_mean = ROI_BOX_LOSS_WEIGHT * train_box_mean
        ctc_kept_denom = max(1, ctc_kept_rows)
        ctc_total_denom = max(1, ctc_total_rows)
        ctc_skipped_frac = ctc_skipped_too_long / ctc_total_denom
        ctc_input_len_mean = ctc_input_len_sum / ctc_kept_denom
        ctc_target_len_mean = ctc_target_len_sum / ctc_kept_denom

        if train_wctc_mean > 2.0 * train_wseq_mean:
            ctc_dominance_streak += 1
        else:
            ctc_dominance_streak = 0
        if ctc_dominance_streak >= 2:
            print(
                f"[WARN] Weighted CTC term dominates CE for {ctc_dominance_streak} epochs: "
                f"wctc={train_wctc_mean:.4f} vs wseq={train_wseq_mean:.4f}"
            )

        print(
            f"\nEpoch {epoch+1}/{num_epochs} | "
            f"train_loss={train_loss_mean:.4f} | "
            f"train_seq={train_seq_mean:.4f} | "
            f"train_ctc={train_ctc_mean:.4f} | "
            f"train_box={train_box_mean:.4f} | "
            f"wseq={train_wseq_mean:.4f} | "
            f"wctc={train_wctc_mean:.4f} | "
            f"wbox={train_wbox_mean:.4f} | "
            f"ctc_skip_too_long={ctc_skipped_frac:.4f} | "
            f"ctc_in_len={ctc_input_len_mean:.2f} | "
            f"ctc_tgt_len={ctc_target_len_mean:.2f} | "
            f"val_loss={val_metrics['val_loss']:.4f} | "
            f"val_cer={val_metrics['val_cer']:.4f} | "
            f"val_exact={val_metrics['val_exact']:.4f}"
        )

        is_best = (best_val_cer is None) or (val_metrics["val_cer"] < best_val_cer)
        if is_best:
            best_val_cer = val_metrics["val_cer"]
        
        checkpoint_payload = {
            "epoch": epoch + 1,
            "encoder_state_dict": encoder.state_dict(),
            "roi_sequence_encoder_state_dict": roi_sequence_encoder.state_dict(),
            "context_encoder_state_dict": context_encoder.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "ctc_head_state_dict": ctc_head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss_mean,
            "train_seq_loss": train_seq_mean,
            "train_ctc_loss": train_ctc_mean,
            "train_box_loss": train_box_mean,
            "train_weighted_seq_loss": train_wseq_mean,
            "train_weighted_ctc_loss": train_wctc_mean,
            "train_weighted_box_loss": train_wbox_mean,
            "ctc_rows_total": ctc_total_rows,
            "ctc_rows_kept": ctc_kept_rows,
            "ctc_rows_skipped_too_long": ctc_skipped_too_long,
            "ctc_skipped_too_long_fraction": ctc_skipped_frac,
            "ctc_mean_input_len_kept": ctc_input_len_mean,
            "ctc_mean_target_len_kept": ctc_target_len_mean,
            "ctc_dominance_streak": ctc_dominance_streak,
            "val_loss": val_metrics["val_loss"],
            "val_cer": val_metrics["val_cer"],
            "val_exact": val_metrics["val_exact"],
            "use_roi_attention": USE_ROI_ATTENTION,
            "stage2_aux_ctc_weight": STAGE2_AUX_CTC_WEIGHT,
            "roi_box_loss_weight": ROI_BOX_LOSS_WEIGHT,
            "stage2_use_gt_boxes": STAGE2_USE_GT_BOXES,
            "stage2_use_gt_boxes_epoch": bool(use_gt_boxes_epoch),
            "stage2_epoch_box_source": epoch_box_source,
            "stage2_curriculum_enable": bool(STAGE2_CURRICULUM_ENABLE),
            "stage2_curriculum_gt_epochs": curriculum_gt_epochs,
            "stage2_grad_accumulation_steps": int(stage2_accum_steps),
            "stage2_reading_order_policy": STAGE2_READING_ORDER_POLICY,
            "stage2_use_attn_centroid_boxes": bool(STAGE2_USE_ATTN_CENTROID_BOXES),
            "stage2_arch_guardrail_strict": bool(STAGE2_ARCH_GUARDRAIL_STRICT),
            "stage2_arch_guardrail_issues": guardrail_info["issues"],
            "stage2_arch_guardrail_warning": guardrail_info["warning"],
            "roi_pool_size": ROI_POOL_SIZE,
            "roi_embed_dim": ROI_EMBED_DIM,
            "context_hidden_dim": CONTEXT_HIDDEN_DIM,
            "vocab": vocab.char2id,
            "is_best": is_best,
        }

        # Save checkpoint
        checkpoint_path = checkpoint_dir / f"sequence_epoch{epoch+1}.pt"
        torch.save(checkpoint_payload, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
        if is_best:
            torch.save(checkpoint_payload, checkpoint_dir / "sequence_best.pt")
            print(f"✅ saved best: {epoch + 1} (val_cer={val_metrics['val_cer']:.4f})")
        prune_to_keep_last_n(checkpoint_dir, keep=2, exclude="checkpoint_old.pt")
    
    return encoder, roi_sequence_encoder, context_encoder, decoder, ctc_head

def train(stage1 = True, stage2 = False, args=None):  
    print("="*60)
    print("STAGE 1: TRAINING DETECTOR (Spatial Localization)")
    print("="*60)
    if stage1 == True:
        unet, detector = train_detector_stage(num_epochs=NUM_EPOCHS, lr=None)
    
    if stage2 == True:
        print("="*60)
        print("STAGE 2: TRAINING SEQUENCE MODEL (Contextual Understanding)")
        print("="*60)
        # Use best checkpoint if it exists, otherwise use last epoch
        best_ckpt = CHECKPOINT_DIR / "stage1_detection" / "detector_best.pt"
        last_ckpt = CHECKPOINT_DIR / "stage1_detection" / f"detector_epoch{NUM_EPOCHS}.pt"
        detector_ckpt = best_ckpt if best_ckpt.exists() else last_ckpt
        
        encoder, roi_sequence_encoder, context_encoder, decoder, ctc_head = train_sequence_stage(
            detector_ckpt_path=detector_ckpt,
            num_epochs=NUM_EPOCHS,
            lr=None,
            use_ctc_warmup=STAGE2_USE_CTC_WARMUP,
        )
        
if __name__ == "__main__":
    train(stage1=False, stage2=True)  