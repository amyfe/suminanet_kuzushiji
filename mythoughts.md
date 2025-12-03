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

### 🧩 3️⃣ Utils / Infrastruktur
#### 📁 utils/dataset.py

Warum:
Trainingsdaten liegen meist als Bilder + JSON-Annotationen vor (z. B. Bounding Boxes, Label-IDs).

Was passiert:

Lädt Bild (PIL.Image oder cv2.imread)

Liest zugehörige JSON-Datei ({"boxes": [...], "labels": [...]}).

Wandelt alles in Tensoren um.

Optional: erzeugt Heatmaps aus Box-Zentren für den Detector.

Ziel: ein einheitlicher Batch aus (image, heatmap, boxes, labels).

#### 🔧 utils/transforms.py

Warum:
Historische Manuskripte sind oft verblasst, schief, gefaltet → Augmentation ist entscheidend.

Was passiert:

CLAHE (Kontrastverbesserung)

Random Rotation / Perspective Warp

Random Crop / Resize

Normalize

Ziel: robustes Modell gegen Schriftarteinflüsse, Verfärbungen und Scanfehler.

#### 📊 utils/visualization.py

Warum:
Zum Debuggen und für qualitative Tests.
Zeigt Heatmaps, Bounding Boxes, Prediction-Overlays und Trainingsverläufe (Loss, CER).

### 🏋️‍♂️ train.py

Warum:
Hier läuft der gesamte Trainingszyklus zusammen:

Lädt Config

Initialisiert Dataset, Models, Optimizer

Führt Training mit Teacher Forcing durch

Speichert Checkpoints und Logs

Besonderheit:
Training wird modular ausgeführt – du kannst z. B. erst Detector pretrainen, dann Classifier, dann Decoder.

### 📈 evaluate.py

Warum:
Zur Validierung und für quantitative Tests (mAP, CER, Top-k).
Wird nach jedem Epoch-Checkpoint ausgeführt.

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

Wie du schon richtig gesagt hast:

Clanuwat et al. → GroupNorm
weil:

kleine Batchsizes (1–4)

BatchNorm kollabiert da

GroupNorm batchunabhängig → stabil

③ Kontextmodellierung: LSTM oder ConvLSTM?

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