# Architecture Assessment for Kuzushiji OCR
## Evaluation of the Two-Stage Pipeline for Historical Japanese Text

---

## Executive Summary

**Overall Assessment:** ⚠️ **PARTIALLY SUITABLE** - The architecture has good foundations but has **critical gaps** for handling Kuzushiji's unique challenges.

**Confidence Level for Kuzushiji:** ~60-70%

**Key Issues:**
1. ❌ Stage 1 ignores reading direction (hardcoded horizontal assumption)
2. ⚠️ No explicit handling of similar/ambiguous characters
3. ❌ Limited support for vertical text (common in historical Japanese)
4. ✅ Good: Two-stage approach (detection + sequence context)
5. ✅ Good: Pre-existing reading order utilities (but not integrated)

---

## Stage 1: Detection Analysis

### Current Implementation
```python
DetectorHead(in_ch=32, num_classes=vocab_size, predict_classes=False)
- Outputs: heatmap (center detection), bbox (box regression)
- Loss: focal_loss_heatmap + 0.1 * bbox_loss
- Purpose: Spatial localization (WHERE characters are)
```

### ✅ Strengths
1. **Heatmap-based detection** - Good for dense text
2. **Center-based approach** - Works well for varying character sizes
3. **Focal loss** - Handles class imbalance (many background pixels)

### ❌ Critical Issues for Kuzushiji

#### 1. **No Reading Direction Awareness**
```python
# Current: Treats all images as if horizontal
features = unet(images)  # (B, 32, H/8, W/8)
outputs = detector(features)
```

**Problem:** Kuzushiji manuscripts have:
- Vertical text (top-to-bottom, right-to-left columns) - **MOST COMMON**
- Horizontal text (left-to-right, top-to-bottom) - RARE
- Mixed orientations in same document

**Impact:** Detection may work spatially, but sequence order will be **completely wrong** in Stage 2.

#### 2. **No Character Similarity Handling**
With 4246 classes and many visually similar characters:
- 生 (10+ readings: sei, nama, iki, u...)
- 土 vs 士 (earth vs samurai - differ by one pixel)
- 未 vs 末 (not yet vs end - very similar)

**Current:** Stage 1 has NO classification head (disabled due to OOM).
**Result:** ALL disambiguation happens in Stage 2 decoder, which may struggle without spatial class priors.

#### 3. **Sparse Targets (0.7% labeled pixels)**
```
Labeled pixels: 937/131,072 (0.71%)
```
**Problem:** 99% of feature map is unlabeled → model learns mostly on sparse signals.
**Better:** Dense annotation or contrastive learning on unlabeled regions.

---

## Stage 2: Sequence Prediction Analysis

### Current Implementation
```python
EncoderWrapper:
  - orientation: "horizontal" or "vertical"
  - Mean pooling across height (horizontal) or width (vertical)
  - Outputs: (B, T, enc_dim) sequence

SeqDecoderAttention:
  - LSTM decoder with Luong attention
  - Teacher forcing: 1.0 → 0.1 (scheduled decay)
  - Predicts character sequence token-by-token
```

### ✅ Strengths
1. **Contextual understanding** - Critical for ambiguous characters
2. **Attention mechanism** - Can focus on relevant spatial regions
3. **Teacher forcing schedule** - Helps training stability
4. **Support for vertical/horizontal** - Architecture CAN handle it!

### ❌ Critical Gaps

#### 1. **Orientation is Hardcoded in Training**
```python
# train_two_stage.py line ~595
enc_outputs, enc_mask = encoder(images, orientation="horizontal")
```

**Problem:** Training ALWAYS uses horizontal orientation.
**Result:** Model never learns vertical text patterns (the dominant format in Kuzushiji!).

**Fix Needed:**
```python
# Should detect or annotate reading direction per image
orientation = detect_orientation(boxes) or annotations['orientation']
enc_outputs, enc_mask = encoder(images, orientation=orientation)
```

#### 2. **Reading Order Not Applied**
```python
# utils/reading_order.py EXISTS but is NEVER CALLED
def sort_boxes_reading_order(boxes, classes, direction="auto"):
    # Auto-detects vertical vs horizontal
    # Sorts boxes in proper reading order
    # But UNUSED in training pipeline!
```

**Problem:** Even if boxes are detected correctly, they're processed in arbitrary order.
**Result:** Decoder learns wrong character sequence patterns.

**Fix Needed:** Integrate reading order before passing to decoder:
```python
sorted_boxes, sorted_labels, indices = sort_boxes_reading_order(
    detected_boxes, detected_classes, direction="auto"
)
```

#### 3. **No Explicit Similar Character Modeling**
The architecture relies purely on:
- Spatial context (from UNet features)
- Sequence context (from LSTM + attention)

**Missing:** 
- Character confusion matrix
- Hard negative mining for similar pairs
- Triplet loss or contrastive learning for similar glyphs

---

## What's Missing for Kuzushiji

### 1. **Multi-Directional Training** ❌ CRITICAL
```python
# Current: Only horizontal
enc_outputs, enc_mask = encoder(images, orientation="horizontal")

# Needed: Dynamic orientation per image
for batch in dataloader:
    images, text_ids, metadata = batch
    orientation = metadata['orientation']  # Annotate in data
    enc_outputs, enc_mask = encoder(images, orientation=orientation)
```

**Without this:** Model will FAIL on 90% of Kuzushiji documents (vertical text).

### 2. **Reading Order Integration** ❌ CRITICAL
```python
# Needed in Stage 2 training:
# After detection, sort boxes before sequence prediction
detected_boxes, detected_classes = detector(images)
sorted_boxes, sorted_classes, _ = sort_boxes_reading_order(
    detected_boxes, detected_classes, direction="auto"
)
# Then feed sorted sequence to decoder
```

**Without this:** Character sequence will be scrambled → terrible accuracy.

### 3. **Character Confusion Modeling** ⚠️ IMPORTANT
Options:
- **Option A:** Re-enable Stage 1 classification with focal loss focused on confusable pairs
- **Option B:** Add auxiliary loss in Stage 2 for similar character discrimination
- **Option C:** Use character embeddings learned from confusion matrix

### 4. **Layout Analysis** ⚠️ MODERATE
Historical documents have:
- Multiple columns (vertical text)
- Annotations/marginalia
- Mixed text sizes
- Non-text elements (seals, illustrations)

**Current:** Treats entire image as one text block.
**Better:** Segment into text regions first, process each region separately.

---

## Recommendations

### **Immediate Fixes (Required for Basic Functionality)**

1. **Add orientation detection/annotation** 
   ```python
   # In KuzushijiDataset, add:
   sample['orientation'] = detect_orientation_from_boxes(boxes)
   
   # In train_two_stage.py, use:
   enc_outputs, enc_mask = encoder(images, orientation=batch['orientation'])
   ```

2. **Integrate reading_order.py in training**
   ```python
   # After detection in Stage 2:
   from utils.reading_order import sort_boxes_reading_order
   sorted_boxes, sorted_labels, _ = sort_boxes_reading_order(
       detected_boxes, detected_labels, direction=orientation
   )
   ```

3. **Add orientation-aware augmentation**
   ```python
   # Augment with both horizontal AND vertical during training
   if random.random() < 0.5:
       image, boxes = rotate_90(image, boxes)  # Vertical text
       orientation = "vertical"
   else:
       orientation = "horizontal"
   ```

### **High-Priority Improvements**

4. **Re-enable Stage 1 classification with efficient head**
   - Use smaller embedding layer (256-dim) instead of 4246-class softmax
   - Add character similarity loss (triplet or contrastive)
   - This gives spatial class priors to help Stage 2

5. **Add column detection for vertical text**
   - Detect text columns before character detection
   - Process each column separately
   - Essential for multi-column layouts

6. **Character confusion matrix**
   - Analyze common misclassifications
   - Add hard negative mining
   - Weight loss higher for confusable pairs

### **Optional Enhancements**

7. **Layout segmentation** - Separate text from non-text
8. **Multi-scale detection** - Better for varying character sizes
9. **Transformer decoder** - May handle long sequences better than LSTM

---

## Conclusion

**Can this architecture work for Kuzushiji?** 
→ **Yes, BUT only with critical fixes to orientation and reading order.**

**Current State:**
- ✅ Good foundation (two-stage, context-aware)
- ❌ Hardcoded for horizontal text (fatal for Kuzushiji)
- ❌ Reading order logic exists but unused
- ⚠️ No explicit handling of similar characters

**With Fixes:**
- Expected accuracy: 70-85% character-level (with proper training)
- Main challenges: Similar glyphs, degraded manuscripts, missing context

**Without Fixes:**
- Expected accuracy: <30% (sequence order completely wrong for vertical text)
- Model will learn, but produce gibberish for vertical documents

**Priority:** Fix orientation handling FIRST before continuing training.
