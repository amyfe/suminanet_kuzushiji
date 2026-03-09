"""Validation utilities for training."""
import torch
import torch.nn as nn
import torch.amp as amp
from utils.detection_utils import build_detection_targets, compute_detection_losses, compute_roi_align_loss
from config import IMAGE_SIZE


def compute_cer(predicted_ids, target_ids, vocab, pad_id, sos_id, eos_id):
    """
    Compute Character Error Rate between predicted and target sequences.
    Uses edit distance (Levenshtein distance).
    Returns: CER (0.0 = perfect, 1.0 = completely wrong), num_errors, num_chars
    """
    def remove_special_tokens(ids):
        """Remove padding, SOS, EOS tokens"""
        ids = ids[ids != pad_id]
        if ids.numel() > 0 and ids[0].item() == sos_id:
            ids = ids[1:]
        if ids.numel() > 0 and ids[-1].item() == eos_id:
            ids = ids[:-1]
        return ids.tolist()
    
    def edit_distance(s1, s2):
        """Levenshtein distance"""
        if len(s1) == 0:
            return len(s2)
        if len(s2) == 0:
            return len(s1)
        
        d = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]
        for i in range(len(s1) + 1):
            d[i][0] = i
        for j in range(len(s2) + 1):
            d[0][j] = j
        
        for i in range(1, len(s1) + 1):
            for j in range(1, len(s2) + 1):
                cost = 0 if s1[i-1] == s2[j-1] else 1
                d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + cost)
        
        return d[len(s1)][len(s2)]
    
    pred_clean = remove_special_tokens(predicted_ids.clone())
    targ_clean = remove_special_tokens(target_ids.clone())
    
    errors = edit_distance(pred_clean, targ_clean)
    total_chars = len(targ_clean)
    
    cer = errors / total_chars if total_chars > 0 else 0.0
    return cer, errors, total_chars


def validate(encoder, decoder, detector, dataloader, vocab, device, 
             use_detector_head, use_roi_attention, detection_loss_weight, roi_box_loss_weight,
             detector_heatmap_sigma, max_batches=None):
    """
    Run validation on the dataset and compute metrics.
    
    Args:
        encoder: Encoder model
        decoder: Decoder model
        detector: Optional DetectorHead
        dataloader: Validation dataloader
        vocab: VocabManager
        device: torch device
        use_detector_head: Whether to compute detection losses
        use_roi_attention: Whether to compute ROI losses
        detection_loss_weight: Weight for detection losses
        roi_box_loss_weight: Weight for ROI losses
        detector_heatmap_sigma: Gaussian sigma for detection targets
        max_batches: Maximum number of batches to validate on
        
    Returns:
        dict with keys: loss, cer, det_loss, roi_loss
    """
    encoder.eval()
    decoder.eval()
    if detector is not None:
        detector.eval()
    
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)
    
    total_loss = 0.0
    total_cer = 0.0
    total_det_loss = 0.0
    total_roi_loss = 0.0
    total_pred_chars = 0
    n_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break
            
            images = batch["image"].to(device)
            text_ids = batch["text_ids"].to(device) if batch["text_ids"] is not None else None
            boxes = batch["boxes"]
            labels = batch["labels"]
            
            if text_ids is None:
                continue

            text_ids_present = batch.get("text_ids_present", None)
            if text_ids_present is not None:
                valid_idx = text_ids_present.to(device).nonzero(as_tuple=False).squeeze(1)
                if valid_idx.numel() == 0:
                    continue
                images = images.index_select(0, valid_idx)
                valid_idx_list = valid_idx.detach().cpu().tolist()
                boxes = [boxes[i] for i in valid_idx_list]
                labels = [labels[i] for i in valid_idx_list]
            
            input_seq = text_ids[:, :-1]
            targets = text_ids[:, 1:]
            
            # Get UNet features for detection if needed
            feats2d = None
            if detector is not None and use_detector_head:
                feats2d = encoder.backbone(images)
            
            enc_outputs, enc_mask = encoder(images, orientation="horizontal")
            
            # Decoder forward (use teacher forcing=1.0 to match target length)
            # With TF=0.0, decoder stops early when predicting EOS, causing shape mismatch
            with amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
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
                logits, hidden, attn, predicted_boxes = decoder_output
            else:
                logits, hidden, attn = decoder_output
                predicted_boxes = None
            
            B, T_dec, V = logits.shape
            
            # Sequence loss and CER
            loss_seq = ce_loss_fn(logits.reshape(-1, V), targets.reshape(-1))
            total_loss += loss_seq.item()
            
            # Compute CER for each sample
            pred_ids = logits.argmax(dim=-1)  # (B, T_dec)
            for b in range(B):
                cer, _, n_chars = compute_cer(pred_ids[b], targets[b], vocab, vocab.pad_id, 
                                             vocab.sos_id, vocab.eos_id)
                total_cer += cer * n_chars
                total_pred_chars += n_chars
            
            # Detection loss (Option 1: DetectorHead)
            if detector is not None and feats2d is not None and use_detector_head:
                det_pred = detector(feats2d)
                if 'heatmap' in det_pred:
                    H_out, W_out = det_pred['heatmap'].shape[2:]
                    gt_heatmap, gt_bbox, gt_bbox_mask, gt_cls = build_detection_targets(
                        boxes, labels, (H_out, W_out), IMAGE_SIZE, device, sigma=detector_heatmap_sigma
                    )
                    loss_det, _ = compute_detection_losses(
                        det_pred, gt_heatmap, gt_bbox, gt_bbox_mask, gt_cls, weights=(1.0, 1.0, 1.0)
                    )
                    total_det_loss += loss_det.item()
            
            # ROI Align loss (Option 2: attention-based boxes with feature alignment)
            if use_roi_attention and predicted_boxes is not None:
                gt_boxes_padded = []
                gt_lengths = []
                max_boxes = max(b.shape[0] for b in boxes) if any(b.numel() > 0 for b in boxes) else 1
                for box_tensor in boxes:
                    orig_len = box_tensor.shape[0]
                    box_tensor = box_tensor.to(device)
                    if box_tensor.numel() == 0:
                        box_tensor = torch.zeros((1, 4), device=device)
                    pad_size = max_boxes - box_tensor.shape[0]
                    if pad_size > 0:
                        box_tensor = torch.cat([
                            box_tensor,
                            torch.zeros((pad_size, 4), device=device)
                        ], dim=0)
                    gt_lengths.append(orig_len)
                    gt_boxes_padded.append(box_tensor)
                gt_boxes_batch = torch.stack(gt_boxes_padded, dim=0)
                gt_lengths_tensor = torch.tensor(gt_lengths, device=device)
                
                loss_roi = compute_roi_align_loss(
                    predicted_boxes,
                    gt_boxes_batch,
                    enc_outputs=enc_outputs,
                    enc_mask=enc_mask,
                    gt_lengths=gt_lengths_tensor,
                    spatial_scale=1.0,
                    pooled_size=7,
                    alignment_weight=0.5
                )
                total_roi_loss += loss_roi.item()
            
            n_batches += 1
    
    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    avg_cer = total_cer / total_pred_chars if total_pred_chars > 0 else 0.0
    avg_det_loss = total_det_loss / n_batches if n_batches > 0 else 0.0
    avg_roi_loss = total_roi_loss / n_batches if n_batches > 0 else 0.0
    
    return {
        'loss': avg_loss,
        'cer': avg_cer,
        'det_loss': avg_det_loss,
        'roi_loss': avg_roi_loss,
        'n_batches': n_batches
    }
