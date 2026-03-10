# venv aktivieren
source venv/Scripts/activate

| Phase                         | Status         | Nächster Schritt                        |
| ----------------------------- | -------------- | --------------------------------------- |
| **Feature Extraction (UNet)** | ✅ fertig       | ggf. Fusion-UNet einbinden              |
| **Dataset + Label-Mapping**   | 🟡 vorbereitet | Offene Punkte: 1. Box-Transformierungen / Augmentierungen: z. B. RandomCrop, Resize usw. → optional fürs Training.
|                               2.Fehlerhafte Bilder oder fehlende Annotationen behandeln: aktuell wird nur continue gemacht, könnte man loggen oder filtern.
|                               3. Integration in DataLoader: Batch-Größen, Collate-Funktion (besonders, wenn die Anzahl der Boxes pro Bild variiert).    |
| **Sequenzbildung**            | 🔴 fehlt       | Box-Sortierung + Sequenzgenerator       |
| **Decoder (Translation)**     | 🔴 fehlt       | Seq2Seq oder CTC-Decoder implementieren |
| **Evaluation (CER/WER)**      | 🔴 fehlt       | Metriken hinzufügen                     |
| **End-to-End-Inference**      | 🔴 fehlt       | `translate.py`-Pipeline                 |

1. UNet:
liefert Feature Maps → benötigt für Detection UND Klassifikation
2. DetectorHead:
Heatmaps für Zentren
BBox-Regression
Per-Zelle Klassen-ID
3. GlyphClassifier
klassifiziert das ausgeschnittene Zeichen
4. SeqDecoder
bildet eine Sequenz (japanische Lesereihenfolge)
→ finaler „Textausgabe-Generator“

# Struktur von den Files
Für Transkription braucht man drei Dinge:
- Wo ist das Zeichen → Detector
- Was ist das Zeichen → Classifier
- In welcher Reihenfolge / Abhängigkeit → Decoder
    - Mit Teacher Forcing
Das Ganze basiert auf U-Net als Feature-Extraktor (gemeinsamer Encoder für alle Teilmodule).

## Modelle im Detail
### 🏗️ models/unet.py – Feature-Extraktor / Encoder-Decoder
- Grund:
    OCR braucht kontextuelles Verständnis des Bildes: die Pixel eines Zeichens hängen von Nachbarzeichen ab.
    → U-Net kann lokale und globale Merkmale kombinieren.

Was passiert:
- Encoder 
    - CNN (z. B. ResNet/MobileNet-Blocks) reduziert das Bild zu Featuremaps (z. B. 512 Kanäle, 1/16 der Größe).
- Decoder
    - Up-Convolutions + Skip-Connections, damit auch feine Linien (Strichführung) erhalten bleiben.
- Output
    - Featuremap, die semantische Information über Position und Form jedes Zeichens enthält.

- Ziel: 
    Diese Features werden an detector und classifier weitergegeben.

### 🎯 models/detector.py – Lokalisierung / Bounding Box / Heatmap

Warum:
OCR muss wissen, wo Zeichen stehen, bevor man sie klassifiziert.

Was passiert:

Nimmt U-Net-Featuremap als Input.

Hat 2 Köpfe:

Heatmap Head: pro Pixel, wie wahrscheinlich ein Zeichen-Zentrum dort ist (ähnlich CenterNet oder EAST).

Box Head: gibt pro Zentrum Breite/Höhe/Offset aus.

Diese Heads werden mit Sigmoid/ L1-Loss trainiert (z. B. Focal Loss auf Heatmap, Smooth L1 auf Box).

Ziel: Markiere Regionen im Bild, die ein Zeichen enthalten → → Crops für Classifier.

### 🔠 models/classifier.py – Zeichenerkennung

Warum:
Sobald der Detector weiß, wo Zeichen sind, muss das System bestimmen, welches Zeichen dort steht.

Was passiert:

Nimmt Crops (oder ROI-Pooling von Featuremap).

Kleines CNN (z. B. 3–4 Conv-Blöcke).

Global Average Pool → Linear → Softmax über Zeichenvokabular (z. B. 2200 Klassen).

Loss: CrossEntropyLoss.

Ziel: Jedem erkannten Kasten ein Label zuordnen (z. B. „本“, „花“, „あ“).

### 🔁 models/decoder.py – Sequenzmodell mit Teacher Forcing

Warum:
Japanische Manuskripte (besonders Zeilen) müssen in Reihenfolge gelesen werden, z. B. von oben nach unten, rechts nach links.
Der Decoder lernt also, eine Zeichenfolge zu generieren, nicht nur isolierte Boxen.

Was passiert:

Encoder-Features (z. B. aus U-Net) → Sequence Encoder (z. B. Bidirectional LSTM oder Transformer).

Decoder (z. B. LSTM mit Attention) erzeugt Zeichen eins nach dem anderen.

Teacher Forcing:

Während Training: mit Wahrscheinlichkeit p nutzt er das Ground-Truth-Zeichen als Input für den nächsten Schritt.

Während Inferenz: nutzt er das vom Modell vorhergesagte Zeichen.

Dadurch stabilisiert sich das Training bei langen Sequenzen.

Ziel: Stabilität + Genauigkeit bei langen Sätzen, selbst mit unvollständigen Bounding Boxes.

### Architekturentscheidungen:
① Feature-Extractor: ResNet, FusionNet oder etwas eigenes?
Optionen:
A. ResNet-18 / 34 (empfohlen für Thesis)

    stabil

    gut dokumentiert

    leicht modifizierbar

    Multi-scale möglich

B. FusionNet (wie KuroNet)

    kombiniert 3 Scales → bessere Erkennung kleiner Kuzushiji

    aber deutlich komplexer

    schwieriger zu erklären in Thesis

    ② Normalisierung: BatchNorm oder GroupNorm?

        Clanuwat et al. → GroupNorm
        weil:

        kleine Batchsizes (1–4)

        BatchNorm kollabiert da

        GroupNorm batchunabhängig → stabil


Kontextmodellierung: LSTM oder ConvLSTM?

KuroNet benutzt bi-directional ConvLSTM, weil:

    Textzellen hängen horizontal UND vertikal zusammen

    ConvLSTM behält räumliche Struktur

    besser für historisches/verschobenes Kursive-Kuzushiji

Optionen:

    ConvLSTM (wie KuroNet)

    Transformer Encoder (moderner, aber schwerer zu begründen)

    reine CNNs (zu schwach)

    ④ Decoder: CTC oder Attention-Decoder?
    Kuzushiji OCR: CTC klar besser, weil:

    keine sauber segmentierten Zeichen

    keine exakte bounding box Reihenfolge

    CTC ist de-facto Standard


# Mögliche Probleme

## **No Reading Direction Awareness**
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


### 6. **No Data Augmentation**
**Status**: ⚠️ DEPENDS ON DATASET

Both stages use raw KuzushijiDataset. Check if dataset already includes augmentation:
- Random rotation?
- Elastic deformation?
- Noise?

**If missing**: Could add `torchvision.transforms` to improve robustness.


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

Weitere Infos

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