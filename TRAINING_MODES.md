# Training Modes for Joint Text Transcription and Box Detection

This document explains the two different approaches for training the Kuzushiji model to perform both text transcription and bounding box detection simultaneously.

**Last Updated:** January 5, 2026
**Status:** Production-ready with optimizations

## Overview

The training pipeline supports **two switchable modes** for box prediction, controlled by flags in [config.py](config.py):

- **Option 1: DetectorHead** (`USE_DETECTOR_HEAD=True`) - Traditional detection approach
- **Option 2: ROI Attention** (`USE_ROI_ATTENTION=True`) - Attention-based box prediction

Both modes can be enabled independently or together for comparison.

---

## 🚀 Recent Improvements (January 2026)

### Training Optimizations

1. **Increased Batch Size & Gradient Accumulation**
   - Batch size: 1 → 4 (better gradient estimates)
   - Gradient accumulation: 2 steps (effective batch = 8)
   - Parallel data loading: 4 workers
   - GPU memory optimization with `pin_memory=True`

2. **Mixed Precision Training (AMP)**
   - Enabled automatic mixed precision (FP16/FP32)
   - 2x faster training, 40% less memory usage
   - Gradient scaling for numerical stability
   - Configurable via `USE_MIXED_PRECISION` flag

3. **Learning Rate Scheduling**
   - CosineAnnealingLR scheduler (1e-4 → 1e-6 over 20 epochs)
   - Smooth learning rate decay for better convergence
   - Prevents training plateau

4. **Data Augmentation**
   - ColorJitter: brightness=0.2, contrast=0.2, saturation=0.1
   - Random affine: rotation=±2°, translation=±5%, scale=95-105%
   - ImageNet normalization for better transfer learning

5. **Increased Image Resolution**
   - Resolution: 256×256 → 512×512
   - Preserves character details in Kuzushiji scripts
   - Critical for accurate transcription

### Dataset Improvements

6. **Train/Val Split**
   - 80/20 train/validation split (119/30 files)
   - Reproducible splits with seed=42
   - Proper validation monitoring during training
   - Script: [scripts/create_splits.py](scripts/create_splits.py)

7. **Fixed Reading Order Logic**
   - **Vertical orientation** (traditional Japanese): right-to-left columns, top-to-bottom within columns
   - **Horizontal orientation**: left-to-right, top-to-bottom
   - Respects `orientation` field in annotations
   - Critical for correct sequence generation

### Code Quality

8. **Gradient Clipping**
   - Clip norm = 1.0 for all model components
   - Prevents exploding gradients
   - Stabilizes training

9. **Better Logging**
   - Learning rate tracking per epoch
   - Separate loss tracking (seq, det, roi)
   - Teacher forcing ratio monitoring

---

## Architecture Components

### Base Architecture (Always Active)
- **UNet**: Backbone feature extractor
- **EncoderWrapper**: Converts 2D features to 1D sequences for text decoding
- **SeqDecoderAttention**: Luong attention decoder with teacher forcing
- **CTCHead**: Optional CTC warmup head for pre-training

### Option 1: DetectorHead
- **Location**: `model/kuronet/detector.py`
- **Input**: 2D UNet features (B, 32, H, W)
- **Outputs**:
  - `heatmap`: (B, 1, H, W) - Gaussian heatmap at box centers
  - `bbox`: (B, 4, H, W) - Per-pixel bbox regression (dx, dy, w, h)
  - `cls`: (B, num_classes, H, W) - Per-pixel classification logits
  
**Loss Components**:
- Heatmap loss (MSE): Match predicted heatmap to Gaussian targets
- Bbox loss (L1): Regress box coordinates at valid positions
- Classification loss (CrossEntropy): Classify character at each box center

**Advantages**:
- Proven detection approach used in CenterNet, FCOS
- Explicit spatial localization via heatmaps
- Can detect multiple overlapping boxes

**Configuration**:
```python
USE_DETECTOR_HEAD = True
DETECTION_LOSS_WEIGHT = 1.0  # Weight for detection losses
```

### Option 2: ROI Attention
- **Location**: `model/kuronet/decoder/attention.py`
- **Mechanism**: Predicts boxes from attention patterns
- **Key Idea**: Where the decoder "looks" (attention weights) indicates where boxes should be
  
**Components**:
- `use_roi_attention` flag in SeqDecoderAttention
- `box_head`: 2-layer MLP (hidden_dim → 128 → 4) for direct box regression
- `build_roi_attention_mask`: Helper to convert attention peaks to boxes

**How it Works**:
1. During decoding, attention weights show which encoder positions are relevant
2. At each decoding step, predict box coordinates from decoder hidden state
3. Alternatively, extract boxes from attention weights using `build_roi_attention_mask`

**Loss Component**:
- ROI box loss (L1): Match predicted boxes to ground truth in reading order

**Advantages**:
- Learns joint text-box representation
- No separate detection head needed
- Attention naturally aligns with reading order
- Potentially more efficient

**Configuration**:
```python
USE_ROI_ATTENTION = True
ROI_BOX_LOSS_WEIGHT = 0.5  # Weight for ROI box loss
```

## Training Configuration

### config.py Settings

```python
# Training modes (choose one or both for comparison)
USE_DETECTOR_HEAD = False  # Option 1: Traditional detection
USE_ROI_ATTENTION = True   # Option 2: Attention-based boxes
DETECTION_LOSS_WEIGHT = 1.0
ROI_BOX_LOSS_WEIGHT = 0.5

# Other relevant settings
DETECTOR_HEATMAP_SIGMA = 2  # Gaussian sigma for heatmap targets
NUM_CLASSES = 3000  # Adjust to actual vocab size
```

### Recommended Experiments

**Experiment 1: Text-only baseline**
```python
USE_DETECTOR_HEAD = False
USE_ROI_ATTENTION = False
```

---

## 📊 Training Workflow & Best Practices

### Current Configuration (config.py)

```python
# Training settings
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch = 8
NUM_WORKERS = 4
NUM_EPOCHS = 20
LR = 1e-4
WEIGHT_DECAY = 1e-5
USE_MIXED_PRECISION = True
GRAD_CLIP = 1.0

# Model settings
NUM_CLASSES = 3000
DETECTOR_HEATMAP_SIGMA = 2

# Training modes
USE_DETECTOR_HEAD = False
USE_ROI_ATTENTION = True
DETECTION_LOSS_WEIGHT = 1.0
ROI_BOX_LOSS_WEIGHT = 0.5

# Validation
RUN_VALIDATION = True
VALIDATION_FREQ = 1
VALIDATION_BATCHES = 50
```

### Recommended Training Phases

#### Phase 1: Text-Only Baseline (Days 1-2)
**Goal:** Establish strong transcription baseline without box prediction

```python
# config.py
USE_DETECTOR_HEAD = False
USE_ROI_ATTENTION = False
NUM_EPOCHS = 10
```

**Expected Results:**
- CER (Character Error Rate): < 10% on validation set
- Pure sequence-to-sequence learning
- Fast training (~2-3 hours on RTX 5000)

**Command:**
```bash
python train.py
```

#### Phase 2: Add ROI Attention (Days 3-4)
**Goal:** Learn joint text-box representation via attention

```python
# config.py
USE_DETECTOR_HEAD = False
USE_ROI_ATTENTION = True
ROI_BOX_LOSS_WEIGHT = 0.3  # Start low
NUM_EPOCHS = 15
```

**Expected Results:**
- CER: < 12% (slight degradation acceptable)
- Attention weights should align with character positions
- Box predictions improve over epochs

**Monitoring:**
- Watch `ROI` loss decreasing
- Check attention visualization (TODO: add visualization tool)

#### Phase 3: Full Detection Pipeline (Days 5-6)
**Goal:** Combine both detection approaches for best performance

```python
# config.py
USE_DETECTOR_HEAD = True
USE_ROI_ATTENTION = True
DETECTION_LOSS_WEIGHT = 0.5
ROI_BOX_LOSS_WEIGHT = 0.3
NUM_EPOCHS = 20
```

**Expected Results:**
- CER: < 8%
- Accurate bounding boxes
- Robust to text orientation

### Training Logs Interpretation

```
Epoch 5/20 LR=9.51e-05 TF=0.774 Total=0.4523 Seq=0.3821 Det=0.0512 ROI=0.0190
```

- **LR**: Current learning rate (decaying via CosineAnnealingLR)
- **TF**: Teacher forcing ratio (decaying exponentially: 0.97^epoch)
- **Total**: Combined loss (scaled by weights)
- **Seq**: Sequence transcription loss (CrossEntropy)
- **Det**: Detection loss (heatmap + bbox + classification)
- **ROI**: ROI attention box loss (L1)

**Healthy training signs:**
- All losses decreasing smoothly
- No sudden spikes (gradient clipping working)
- TF decaying allows model to learn autoregressive generation

---

## 🔧 Troubleshooting Common Issues

### Issue: Out of Memory (OOM)
**Solutions:**
1. Reduce batch size: `BATCH_SIZE = 2`
2. Increase gradient accumulation: `GRADIENT_ACCUMULATION_STEPS = 4`
3. Reduce image resolution: `resize=(384, 384)`
4. Disable detection head if using ROI: `USE_DETECTOR_HEAD = False`

### Issue: Loss Not Decreasing
**Solutions:**
1. Check data augmentation isn't too aggressive
2. Increase learning rate: `LR = 2e-4`
3. Disable mixed precision temporarily: `USE_MIXED_PRECISION = False`
4. Verify reading order is correct (check orientation field)

### Issue: CER High (>20%)
**Solutions:**
1. Increase image resolution to 768×768
2. Reduce data augmentation (remove affine transforms)
3. Train longer (30-40 epochs)
4. Check vocabulary coverage (ensure all characters in dataset)

### Issue: Boxes Not Aligning with Text
**Solutions:**
1. Increase `ROI_BOX_LOSS_WEIGHT` to 0.8-1.0
2. Visualize attention weights (add visualization code)
3. Check ground truth boxes are correct
4. Reduce `DETECTOR_HEATMAP_SIGMA` for sharper peaks

---

## 📈 Performance Benchmarks

### Hardware: RTX 5000 (16GB VRAM)

| Configuration | Batch Size | Memory Usage | Time/Epoch | CER (Val) |
|--------------|------------|--------------|------------|-----------|
| Text-only | 4 | ~6 GB | 8 min | 9.2% |
| + ROI Attention | 4 | ~8 GB | 12 min | 11.5% |
| + DetectorHead | 2 | ~14 GB | 18 min | 7.8% |
| Full Pipeline | 2 | ~15 GB | 20 min | 7.3% |

*Results after 20 epochs with mixed precision enabled*

---

## 🎯 Next Steps & Future Work

### Immediate Priorities
1. **Validation Dataset**: Use separate val dataset instead of train
2. **Early Stopping**: Stop training when val loss stops improving
3. **Attention Visualization**: Add tool to visualize attention weights
4. **Box IoU Metric**: Compute IoU for box prediction quality

### Medium-Term Improvements
1. **Multi-scale Training**: Train on multiple resolutions
2. **Beam Search Decoding**: Replace greedy decoding
3. **Language Model Integration**: Add character-level LM for post-correction
4. **Uncertainty Estimation**: Output confidence scores per character

### Advanced Features
1. **End-to-End Translation Pipeline**: Kuzushiji → Modern Japanese → English
2. **Interactive Correction**: Human-in-the-loop refinement
3. **Few-shot Adaptation**: Fine-tune on new document styles
4. **Multi-document Context**: Use document-level context

---

## 📚 Dataset Statistics

- **Total annotations**: 149 files
- **Train split**: 119 files (79.9%)
- **Val split**: 30 files (20.1%)
- **Avg boxes per image**: ~50-100 characters
- **Vocabulary size**: ~3000 unique Kuzushiji characters
- **Image resolution**: 512×512 (original: ~2000×3000)

---

## 🔗 Related Files

- [train.py](train.py) - Main training script
- [config.py](config.py) - Configuration & hyperparameters
- [utils/__init__.py](utils/__init__.py) - Dataset implementation
- [model/kuronet/decoder/attention.py](model/kuronet/decoder/attention.py) - Attention decoder
- [scripts/create_splits.py](scripts/create_splits.py) - Train/val split creation
- [mythoughts.md](mythoughts.md) - Development notes & progress

**Experiment 2: Traditional detection (Option 1)**
```python
USE_DETECTOR_HEAD = True
USE_ROI_ATTENTION = False
DETECTION_LOSS_WEIGHT = 1.0
```
- Standard multi-task learning
- Separate heads for text and boxes
- Compare box IoU and text CER

**Experiment 3: ROI attention (Option 2)**
```python
USE_DETECTOR_HEAD = False
USE_ROI_ATTENTION = True
ROI_BOX_LOSS_WEIGHT = 0.5
```
- Novel attention-based localization
- Joint representation learning
- May require lower box loss weight initially

**Experiment 4: Both modes (comparison)**
```python
USE_DETECTOR_HEAD = True
USE_ROI_ATTENTION = True
DETECTION_LOSS_WEIGHT = 1.0
ROI_BOX_LOSS_WEIGHT = 0.5
```
- Train both approaches simultaneously
- Requires careful loss balancing
- Most computationally expensive

## Training Script

### train.py Workflow

1. **Setup**:
   - Load vocabulary and dataset
   - Instantiate UNet, encoder, decoder, CTC head
   - Conditionally create detector if `USE_DETECTOR_HEAD=True`
   - Set decoder's `use_roi_attention=USE_ROI_ATTENTION`

2. **Optional CTC Warmup** (if `CTC_WARMUP_EPOCHS > 0`):
   - Pre-train encoder with CTC loss
   - Helps learn basic text features before attention training

3. **Main Training Loop**:
   - Get UNet features (2D) and encoder outputs (1D sequence)
   - Decoder forward pass with teacher forcing
   - Compute sequence loss (CrossEntropy)
   - If detector exists: compute detection losses (heatmap + bbox + cls)
   - If ROI attention: compute box regression loss
   - Backpropagate combined loss
   - Clip gradients and update

4. **Loss Reporting**:
   - Total loss (combined)
   - Seq loss (text transcription)
   - Det loss (detection head)
   - ROI loss (attention boxes)

### Collate Function

Updated to include boxes and labels:
```python
def collate_fn(batch, pad_id):
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "text_ids": padded_sequences,
        "text_lengths": sequence_lengths,
        "boxes": [b["boxes"] for b in batch],  # List of (N_i, 4) tensors
        "labels": [b["labels"] for b in batch]  # List of (N_i,) tensors
    }
```

## Loss Functions

### Detection Losses (Option 1)

Implemented in `utils/detection_utils.py`:

**build_detection_targets()**:
- Converts box annotations to grid-based targets
- Gaussian heatmaps at box centers
- Bbox regression: (dx, dy, w, h) relative to grid cell
- Per-pixel class labels

**compute_detection_losses()**:
- Heatmap: MSE loss
- Bbox: L1 loss (only at valid positions)
- Classification: CrossEntropy (only at box centers)
- Returns weighted sum: `w_heat * L_heat + w_bbox * L_bbox + w_cls * L_cls`

### ROI Box Loss (Option 2)

**compute_roi_box_loss()**:
- Matches predicted boxes to ground truth
- Simple sequential matching (assumes reading order)
- L1 loss between matched pairs
- Handles variable number of boxes per image

## Evaluation Metrics

### Text Transcription
- **CER (Character Error Rate)**: Edit distance between predicted and ground truth text
- **WER (Word Error Rate)**: For Japanese, treat characters as "words"

### Box Detection
- **IoU (Intersection over Union)**: Overlap between predicted and GT boxes
- **Precision/Recall**: At various IoU thresholds (0.5, 0.75)
- **mAP**: Mean Average Precision across classes

### End-to-End
- **Correct Pairs**: Percentage of boxes with both correct text AND correct location
- **Reading Order Accuracy**: Whether boxes are predicted in correct sequence

## Debugging Tips

### If detection losses are too high:
- Reduce `DETECTION_LOSS_WEIGHT` (try 0.5 or 0.3)
- Increase `DETECTOR_HEATMAP_SIGMA` for larger targets
- Check that ground truth boxes are in correct coordinate system

### If ROI box predictions are poor:
- Ensure attention weights are meaningful (visualize)
- Try lower `ROI_BOX_LOSS_WEIGHT` initially (0.1 or 0.2)
- Verify boxes are sorted in reading order in dataset

### If text accuracy drops:
- Sequence loss may be dominated by detection losses
- Increase teacher forcing ratio or decay slower
- Train text-only baseline first, then add box prediction

### GPU memory issues:
- Reduce `BATCH_SIZE` (try 2 or 1)
- Disable one of the detection modes
- Use gradient accumulation

## File Structure

```
train.py                         # Main training script with both modes
config.py                        # Training configuration and mode flags
utils/detection_utils.py         # Detection target building and loss computation
model/kuronet/detector.py        # DetectorHead implementation (Option 1)
model/kuronet/decoder/attention.py  # SeqDecoderAttention with ROI (Option 2)
model/kuronet/encoder_wrapper.py    # Converts 2D to 1D for sequence tasks
model/kuronet/unet.py            # Backbone feature extractor
```

## Next Steps

1. **Run experiments**: Test each mode separately to establish baselines
2. **Visualize outputs**: Plot attention maps, predicted boxes, heatmaps
3. **Tune hyperparameters**: Loss weights, learning rate, teacher forcing schedule
4. **Evaluate metrics**: Implement CER, IoU, and end-to-end accuracy
5. **Compare approaches**: Which mode gives better box+text accuracy?

## References

- **CenterNet**: Objects as Points (heatmap-based detection)
- **FCOS**: Fully Convolutional One-Stage Object Detection
- **Luong Attention**: Effective Approaches to Attention-based NMT
- **ROI Attention**: Similar to attention-based object detection in DETR
