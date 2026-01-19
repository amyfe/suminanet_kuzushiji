# Training Logic Analysis: train_two_stage.py

## ✅ WHAT IS IMPLEMENTED

### Stage 1: Detection Training
```
Input Image → UNet → DetectorHead → 3 Tasks:
  1. Heatmap (FocalLoss) - WHERE are characters
  2. BBox Regression (SmoothL1) - Precise box coordinates
  3. Class Logits (CrossEntropy) - Class per pixel
```

**Components Implemented**:
- ✅ FocalLoss for heatmap (alpha=0.25, gamma=2.0)
- ✅ SmoothL1Loss for bbox regression (weight 0.5)
- ✅ CrossEntropyLoss for class prediction (weight 0.5)
- ✅ Mixed precision training (amp.GradScaler)
- ✅ Gradient accumulation (GRADIENT_ACCUMULATION_STEPS)
- ✅ Checkpoint saving and pruning
- ✅ Learning rate scheduling (CosineAnnealingLR)

---

### Stage 2: Sequence Training (Context Understanding)

```
Input Image → UNet (frozen) → EncoderWrapper → SeqDecoderAttention
                                                      ↓
                                            Predicts sequence with context
```

**Components Implemented**:

#### Phase 1: CTC Warmup (2 epochs)
- ✅ Encoder + CTC Head
- ✅ CTC Loss (for alignment)
- ✅ Learns character-sequence alignment
- ✅ Mixed precision + gradient accumulation
- ✅ Frozen detector (spatial guidance)

#### Phase 2: Attention Training (NUM_EPOCHS)
- ✅ Encoder + Decoder with Attention
- ✅ Sequence CrossEntropyLoss
- ✅ Teacher forcing decay (1.0 → 0.1)
- ✅ Mixed precision + gradient accumulation
- ✅ Frozen detector (spatial guidance)

---

## ⚠️ POTENTIAL ISSUES / GAPS

### 1. **Encoder Freeze/Unfreeze Logic**
**Status**: ⚠️ UNCLEAR

```python
# Stage 2 setup:
for p in unet.parameters():
    p.requires_grad = False  # Freeze detector UNet
for p in detector.parameters():
    p.requires_grad = False  # Freeze detector Head

# But then:
encoder = EncoderWrapper(backbone=unet, ...)  # Uses same frozen UNet
```

**Question**: EncoderWrapper wraps the frozen UNet. Does it have its own trainable projection layer?

**Check**: Look at `model.kuronet.encoder_wrapper.py` to see if it adds trainable layers.

---

### 2. **Missing Validation Loop**
**Status**: ❌ NOT IMPLEMENTED

Current code only trains, never validates. Missing:
- Validation dataloader
- Validation loss computation
- CER (Character Error Rate) metric
- Early stopping logic

**Needed Components**:
```python
# From train.py (lines 370-385):
if RUN_VALIDATION and (epoch + 1) % VALIDATION_FREQ == 0:
    val_metrics = validate(...)
    print(f"VAL Loss={val_metrics['loss']:.4f} CER={val_metrics['cer']:.4f}")
```

---

### 3. **Detection Loss Not Used in Stage 2**
**Status**: ⚠️ PARTIAL

```python
if predicted_boxes is not None and any(len(b) > 0 for b in boxes):
    # Can add detection loss here if needed
    # For now, we use boxes for guidance only
    pass
```

**Question**: Should we also train box prediction in Stage 2?

**Options**:
- **Option A**: Don't use detection loss in Stage 2 (current)
  - Pro: Focuses purely on sequence understanding
  - Con: Don't refine box predictions
  
- **Option B**: Add weak supervision from ground truth boxes
  - Pro: Refines detection + sequence jointly
  - Con: Adds complexity, similar to train.py

---

### 4. **Missing TensorBoard / Logging**
**Status**: ❌ NOT IMPLEMENTED

Only has `print()` statements. Missing:
- TensorBoard logging
- Loss plots
- Training curves
- Metrics tracking

**Needed** (from train.py):
```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir=...)
writer.add_scalar('train/loss', loss, global_step)
```

---

### 5. **Encoder Learning Rate**
**Status**: ⚠️ UNCLEAR

```python
encoder = EncoderWrapper(backbone=unet, in_channels=32, enc_dim=256).to(DEVICE)
optimizer = optim.AdamW(
    list(encoder.parameters()) + list(decoder.parameters()) + list(ctc_head.parameters()),
    lr=lr, weight_decay=WEIGHT_DECAY
)
```

**Question**: Should frozen UNet backbone have 0 learning rate?

**Current behavior**: UNet is frozen (no grad), so it won't update anyway. But EncoderWrapper might have additional parameters?

---

### 6. **No Data Augmentation**
**Status**: ⚠️ DEPENDS ON DATASET

Both stages use raw KuzushijiDataset. Check if dataset already includes augmentation:
- Random rotation?
- Elastic deformation?
- Noise?

**If missing**: Could add `torchvision.transforms` to improve robustness.

---

## 🔍 TRAINING LOGIC CHECKLIST

### Stage 1: Detection
```
✅ Load dataset with boxes/labels
✅ Initialize UNet + DetectorHead
✅ Create loss functions
✅ Loop over epochs:
   ✅ Loop over batches:
      ✅ Forward pass
      ✅ Compute 3 losses (heatmap, bbox, class)
      ✅ Backward with gradient accumulation
      ✅ Optimizer step
   ✅ Save checkpoint
   ✅ Prune old checkpoints
✅ Return trained models
```

### Stage 2: Sequence Training
```
✅ Load Stage 1 checkpoint
✅ Freeze detector
✅ Initialize Encoder + Decoder + CTC Head
✅ Create loss functions (CE, CTC)
✅ CTC Warmup (2 epochs):
   ✅ Train encoder with CTC loss
   ✅ Gradient accumulation
   ✅ Learning rate scheduling
❌ Validation during warmup?
✅ Main Attention Training:
   ✅ Loop over epochs:
      ✅ Compute teacher forcing ratio (decay)
      ✅ Loop over batches:
         ✅ Forward pass (encoder + decoder)
         ✅ Sequence loss
         ⚠️ Detection loss? (optional)
         ✅ Backward with gradient accumulation
         ✅ Optimizer step
      ✅ Save checkpoint
      ✅ Prune old checkpoints
❌ Validation during training?
✅ Return trained models
```

---

## RECOMMENDED IMPROVEMENTS

### Priority 1 (Important)
1. **Add validation loop** - Need metrics to know if training is working
   - CER (Character Error Rate)
   - Sequence loss on val set
   - Early stopping

2. **Clarify encoder freeze logic** - Check EncoderWrapper implementation
   - Does it have trainable projection?
   - Should we unfreeze parts of UNet?

3. **Test on actual data** - Run full pipeline end-to-end
   - Stage 1 → visualize detections
   - Stage 2 → check sequence predictions

### Priority 2 (Nice-to-Have)
4. **Add TensorBoard logging** - Better training visualization
5. **Add data augmentation** - If not already in dataset
6. **Clarify detection loss** - Should we refine boxes in Stage 2?

---

## COMPLETE TRAINING FLOW

```
python train_two_stage.py

1. Stage 1: 10 epochs (default NUM_EPOCHS)
   - Trains UNet + DetectorHead
   - Saves checkpoints/stage1_detection/detector_epoch{1-10}.pt

2. Stage 2: 2 epochs warmup + 10 epochs attention
   - Loads Stage 1 checkpoint
   - Trains Encoder + Decoder
   - Saves checkpoints/stage2_sequence/sequence_epoch{1-10}.pt
   
3. Final models ready for inference:
   - Detector: Used for visualization
   - Encoder+Decoder: Used for sequence prediction
```

---

## KEY QUESTIONS TO VERIFY

1. ✅ amp.autocast syntax - Fixed (supports `enabled=` in PyTorch 2.8)
2. ⚠️ Gradient accumulation - Implemented, but is it correct?
3. ⚠️ Encoder trainability - Does EncoderWrapper have frozen UNet issue?
4. ❌ Validation - Should we add during training?
5. ❌ Detection in Stage 2 - Should we use predicted_boxes for loss?

---

## NEXT STEPS

1. **Test Stage 1 training** on sample data
   - Verify detection loss decreases
   - Visualize detected boxes
   - Check checkpoint format

2. **Test Stage 2 training** with Stage 1 checkpoint
   - Verify CTC warmup works
   - Verify attention training converges
   - Check sequence predictions make sense

3. **Add validation** (if results look reasonable)
   - Track CER on validation set
   - Implement early stopping

4. **Compare with train.py** - Verify improvements
