# Two-Stage Training Architecture for Kuzushiji OCR

## Why We Need Both Detection AND Sequence Prediction

### The Problem with Detection-Only Approach:
```
❌ Detect boxes → Classify each independently
   生 → "sei"? "nama"? "iki"? "u(mu)"?
   Cannot disambiguate without context!
```

### Japanese Requires Sequential Context:
- **Same kanji, different readings**: 生 has 10+ readings
- **Context determines meaning**:
  - 生徒 (seito) = student
  - 生きる (ikiru) = to live
  - 生まれる (umareru) = to be born
  - 先生 (sensei) = teacher

### Solution: Hybrid Architecture
```
✓ Stage 1: DetectorHead → Find WHERE characters are (spatial)
✓ Stage 2: Encoder+Decoder → Understand WHAT they say (contextual)
```

---

## Complete Architecture Flow

```
Input Image (512×512)
       ↓
   [UNet] - Feature extraction (32 channels)
       ↓
   ├─────────────────┬──────────────────┐
   ↓                 ↓                  ↓
[DetectorHead]   [Encoder]         [Frozen UNet]
   ↓                 ↓                  ↓
Heatmap+BBox    Sequence Repr.    Spatial Features
   ↓                 ↓                  ↓
Stage 1          [Decoder]         Stage 2
Training         + Attention       Training
   ↓                 ↓
Boxes            Sequence
(WHERE)          (WHAT + Context)
```

---

## Stage 1: Detection Training

**Purpose**: Learn to localize characters spatially

**Components**:
- UNet (feature extraction)
- DetectorHead (heatmap, bbox regression, class logits)

**Training**:
- FocalLoss for heatmap (handles class imbalance)
- SmoothL1 for bbox regression
- CrossEntropy for per-pixel classification

**Output**: Checkpoint with trained detector

**Why First?**:
- Provides spatial grounding for Stage 2
- Can visualize detection quality before sequence training
- Simpler to debug (no sequence complexity)

---

## Stage 2: Sequence Training with Context

**Purpose**: Understand character sequences in context

**Components**:
- EncoderWrapper (converts 2D features → 1D sequence)
- SeqDecoderAttention (predicts sequence with attention)
- CTC Head (auxiliary loss, improves encoder)

**Key Difference from train.py**:
- Detector is **frozen** (already trained in Stage 1)
- Only trains encoder, decoder, CTC head
- Uses detected boxes as **guidance** (not training signal)

**Training Process**:
1. **CTC Warmup** (2 epochs):
   - Trains encoder with CTC loss
   - Learns good sequence representation
   - Helps encoder convergence

2. **Attention Training** (NUM_EPOCHS):
   - Encoder: Image → sequence representation
   - Decoder: Predicts next character with attention
   - Teacher forcing: Gradually reduce (1.0 → 0.1)
   - Uses detected boxes for spatial guidance

**Output**: Checkpoint with encoder + decoder + CTC head

**Why Context Matters**:
```python
# Without context (detection only):
boxes = [生, 徒]
predictions = ["sei?", "to?"]  # Ambiguous!

# With context (sequence model):
sequence = "生徒"
prediction = "seito" (student)  # Correct reading!
```

---

## Key Architectural Decisions

### 1. Why Freeze Detector in Stage 2?
- **Stability**: Detection already learned, don't disturb it
- **Focus**: Stage 2 focuses purely on contextual understanding
- **Debugging**: Can test detection independently

### 2. Why Use Detected Boxes?
- **Spatial Guidance**: Helps attention mechanism focus
- **Reality Check**: Tests if Stage 1 detection is good enough
- **Weak Supervision**: Ground truth boxes also available for fallback

### 3. Why CTC Warmup?
- **Encoder Bootstrap**: Gives encoder good starting point
- **Alignment**: Helps learn character-to-sequence alignment
- **Convergence**: Attention training converges faster

---

## Comparison: train.py vs train_two_stage.py

| Aspect | train.py (End-to-End) | train_two_stage.py (Staged) |
|--------|----------------------|----------------------------|
| **Detection** | Optional (USE_DETECTOR_HEAD) | Required in Stage 1 |
| **Sequence** | Always trained together | Separate Stage 2 |
| **Debugging** | Entangled losses | Modular stages |
| **ROI Attention** | Broken (1D as fake 2D) | Removed (use clean detection) |
| **Training Signal** | Conflated | Staged, clean |
| **Visualization** | Hard (everything mixed) | Easy (stage by stage) |

---

## Training Commands

### Stage 1 Only (Detection):
```bash
python train_two_stage.py  # Runs Stage 1
```

### Both Stages (Detection + Sequence):
```python
# In train_two_stage.py, set:
if __name__ == "__main__":
    train(stage2=True)
```

### Manual Stage Control:
```python
# Stage 1:
unet, detector = train_detector_stage(num_epochs=10)

# Stage 2 (after Stage 1):
encoder, decoder, ctc_head = train_sequence_stage(
    detector_ckpt_path="checkpoints/stage1_detection/detector_epoch10.pt",
    num_epochs=20,
    use_ctc_warmup=True
)
```

---

## Expected Behavior

### Stage 1 Convergence:
- Heatmap loss: 0.3 → 0.05 (FocalLoss)
- BBox loss: 0.1 → 0.02
- Class loss: 1.5 → 0.3
- **Visualization**: Boxes accurately localize characters

### Stage 2 Convergence:
- CTC warmup: 1.5 → 0.8
- Sequence loss: 2.0 → 0.4
- Teacher forcing: 1.0 → 0.1
- **Behavior**: Correctly reads sequences like "生徒" → "seito"

---

## What Gets Saved

### Stage 1 Checkpoint:
```python
{
    'unet_state_dict': ...,
    'detector_state_dict': ...,
    'optimizer_state_dict': ...,
    'loss': ...
}
```

### Stage 2 Checkpoint:
```python
{
    'encoder_state_dict': ...,
    'decoder_state_dict': ...,
    'ctc_head_state_dict': ...,
    'optimizer_state_dict': ...,
    'vocab': ...,
    'loss': ...
}
```

---

## Summary

**You were absolutely right** - we need sequence context for Japanese text!

The architecture now combines:
1. **Spatial detection** (Stage 1) - finds character locations
2. **Contextual understanding** (Stage 2) - reads sequences correctly

This gives the best of both worlds:
- Precise character localization (DetectorHead)
- Contextual disambiguation (Encoder + Decoder)
- Clean, modular training (staged approach)

**train_two_stage.py now fully replaces train.py** with a cleaner, more debuggable implementation.


## Next Steps

### Short Term (Immediate)
- [ ] Train Stage 1 to convergence
- [ ] Visualize detections on validation set
- [ ] Train Stage 2 with GT boxes
- [ ] Measure classification accuracy

### Medium Term
- [ ] Add detection metrics (mAP, precision, recall)
- [ ] Handle furigana grouping
- [ ] Filter picture regions
- [ ] Implement end-to-end CER metric

### Long Term (Future)
- [ ] Add triplet loss for hard negatives
- [ ] Implement picture detection
- [ ] Multi-direction fine-tuning
- [ ] OCR post-processing (language model)

---

## References

- **FocalLoss:** Lin et al. "Focal Loss for Dense Object Detection" (RetinaNet)
- **ROI Align:** He et al. "Mask R-CNN"
- **Heatmap-based detection:** CenterNet, CornerNet papers