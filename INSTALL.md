# Installation

## System Requirements

- Python 3.10+
- CUDA-capable GPU: **6 GB VRAM minimum** for SAM2 (`sam2.1_hiera_large`); 8+ GB recommended for comfortable batch sizes

## 1. System Package — MeCab (not pip-installable)

`fugashi` requires the MeCab morphological analyser to be installed at the OS level **before** `pip install`:

```bash
# Ubuntu / Debian
sudo apt install mecab libmecab-dev mecab-ipadic-utf8

# macOS (Homebrew)
brew install mecab mecab-ipadic
```

## 2. Python Packages

```bash
pip install -r requirements.txt
```

For a fully reproducible environment (exact versions used during development):

```bash
pip install -r requirements-lock.txt
```

## 3. SAM2 Model Weights

SAM2 weights are not included in the repository and must be downloaded separately:

```bash
mkdir -p checkpoints/sam2
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt \
     -P checkpoints/sam2/
```

The expected path matches the `SAM2_CHECKPOINT` entry in `config.py`.

## 4. UniDic Edo Dictionary (MeCab normalization)

The MeCab normalizer uses a classical Japanese dictionary optimised for Edo-period text.
It is downloaded automatically on the first run of the pipeline if `unidic` is installed:

```bash
python -c "import unidic; unidic.download()"
```

If this fails or disk space is limited, the system falls back to `unidic-lite` (already in `requirements.txt`), then to a heuristic-only normalization pass.  The active fallback level is reported in the `normalization_method` field of the API response.

## 5. Quick Start

```bash
# Stage 1: train detector
python train_stage1.py

# Stage 2 warmup (ROI pipeline warm-start)
python train_stage2_warmup.py

# Stage 2 SuminaNet (full recognizer)
python train_stage2_suminanet.py

# Run inference on a single page image
python infer.py --image path/to/page.jpg

# Start the web API + frontend
uvicorn backend.app:app --reload
```

SAM2 preprocessing runs automatically at the end of Stage 1 training (illustration masks) and Stage 1 completion (proposal refinement) when `SAM2_PREPROCESSING = True` in `config.py`.
