# Training Modes for Joint Text Transcription and Box Detection

This document explains the two different approaches for training the Kuzushiji model to perform both text transcription and bounding box detection simultaneously.

## Overview

The training pipeline now supports **two switchable modes** for box prediction, controlled by flags in `config.py`:

- **Option 1: DetectorHead** (`USE_DETECTOR_HEAD=True`) - Traditional detection approach
- **Option 2: ROI Attention** (`USE_ROI_ATTENTION=True`) - Attention-based box prediction

Both modes can be enabled independently or together for comparison.

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
- Pure transcription training
- Establishes baseline text accuracy (CER metric)

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
