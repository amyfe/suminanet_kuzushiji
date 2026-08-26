# Sumina — Reading the Hand of Edo

Welcome friends of historical Japanese text or Machine Learning methods! 

This repository holds **Sumina**, a system that reads *kuzushiji* — the
cursive, hand-written script used throughout pre-modern Japan — from scans of
Edo-period books, and turns it into transcribed, modern-readable, translated text (available in English and German).

Kuzushiji fell out of common use after Japan standardized its script in the early
20th century. As a result, the vast majority of pre-modern Japanese books,
letters, and records are now unreadable to all but a small number of specially
trained scholars. Millions of pages sit in archives, digitized as images but
effectively locked away as text. This project is an attempt to help open a small
part of that lock: given a page image, automatically find every character,
transcribe it, and translate it into modern Japanese, English, or German.

This is a machine learning thesis project, built end-to-end: from raw archival
scans, to a trained two-stage recognition model, to a web app anyone can upload
a page to.

## The data

The training data comes from the **Kuzushiji Dataset** published by the
[Center for Open Data in the Humanities (CODH)](http://codh.rois.ac.jp/kuzushiji/),
part of Japan's National Institute of Japanese Literature. CODH digitized and
hand-annotated pages from historical Japanese books, with a bounding box and a
Unicode character label for every single character on the page. It's the same
underlying corpus behind the Kuzushiji-MNIST/Kuzushiji-49/Kuzushiji-Kanji
benchmarks and the 2019 Kaggle "Kuzushiji Recognition" competition, though this
project works from the full page-level, multi-character annotations rather than
the pre-cropped single-character benchmark sets. This project uses the data
from the Kuzushiji Dataset, created by CODH and introduced in Clanuwat et al.,
["Deep Learning for Classical Japanese Literature"](https://arxiv.org/abs/1812.01718)
(arXiv:1812.01718, 2018).

Measured directly from the corpus as prepared in this repo (`assets/data/`):

- **60 books** (a mix of numeric CODH IDs and named volumes, e.g. classical
  literature like *Hyakunin Isshu* alongside other Edo-period printed works)
- **5,344 annotated page images**
- **~1.24M character-level bounding box annotations** in total
- **3,113 unique character classes** — kanji, hiragana, katakana, and
  punctuation/iteration marks, with a strongly long-tailed frequency
  distribution (a handful of characters dominate, most appear only a few times)

Note: a known data-quality issue is tracked internally where a subset of
annotation files (roughly 14%) contain every box duplicated — this affects raw
counts above and is accounted for during evaluation, but not yet cleaned at the
source.

Raw CODH downloads are converted into a unified per-page JSON format by
[`scripts/onetime_scripts/prepare_codh_annotations.py`](scripts/onetime_scripts/prepare_codh_annotations.py),
producing `assets/data/annotations/*.json` plus a `label2id.json` /
`id2label.json` vocabulary mapping.

## The pipeline

Sumina turns a page image into translated text in five stages:

**1. Character detection (Stage 1)**
An EfficientNet-B2 backbone with an FPN-style decoder produces a heatmap over
the page; a lightweight detector head turns that heatmap into character
bounding-box proposals. This stage only has to find *where* characters are, not
what they say.

**2. Character recognition (Stage 2 — SuminaNet)**
For every proposal box from Stage 1: a crop encoder extracts visual features
per-character (ROI align), a refinement head cleans up the box geometry, a
reading-order module sorts the boxes the way a human would actually read the
page (columns, right-to-left, top-to-bottom — including separating out
furigana), and a GRU-based context encoder lets neighboring characters inform
each other before an MLP classifier predicts each character over the full
vocabulary. SuminaNet is architecturally inspired by
[*KuroNet: Pre-Modern Japanese Kuzushiji Character Recognition with Deep
Learning*](https://arxiv.org/abs/1910.09433) (Clanuwat et al., ICDAR 2019) but
is an in-house implementation, not a reuse of its code.

**3. Normalization**
Classical character forms and orthography are normalized toward modern
Japanese using MeCab + a UniDic dictionary tuned for Edo-period text, with
graceful fallback tiers (Edo UniDic → UniDic-lite → heuristic-only) if the
richer dictionary isn't available.

**4. Translation**
The normalized modern Japanese is translated into English or German by Claude
(via OpenRouter), which also flags uncertain passages.

**5. Serving**
A FastAPI backend (`app/backend/`) loads the trained model and exposes
`/api/transcribe` and `/api/translate`; a React frontend (`app/frontend/`) lets
someone upload a scan, see the detected character boxes overlaid on the image,
and read the transcription and translation side by side.

```
page image
    │
    ▼
Stage 1: EfficientNet-B2 + FPN  ──►  character box proposals
    │
    ▼
Stage 2: SuminaNet (ROI encode → reading order → context → classify)
    │
    ▼
classical Japanese transcription
    │
    ▼
MeCab + UniDic normalization  ──►  modern Japanese
    │
    ▼
Claude translation  ──►  English / German
    │
    ▼
Sumina web app
```

## Where to go next

- [`INSTALL.md`](INSTALL.md) — environment setup, model weights, quick-start commands
- [`app/backend/README.md`](app/backend/README.md) — API endpoints, model loading, deployment configuration
- `config.py` — every hyperparameter and path in the pipeline, in one place
- `model/suminanet/` — detector and recognizer architecture
- `model/translation/` — normalization and translation pipeline
- `scripts/` — data preparation, evaluation, and visualization tooling
