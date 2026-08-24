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

## 5. Environment Variables

Copy `.env.example` to `.env` and fill in at least one API key:

```bash
cp .env.example .env
```

| Variable | Purpose |
|---|---|
| `OPENROUTER_API_KEY` | OpenRouter gateway (uses `anthropic/claude-sonnet-4-6` under the hood) — required for translation |
| `API_KEY` | Optional. Shared secret gating `/api/transcribe` and `/api/translate` — see the Production checklist below |

## 6. Quick Start

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

## 7. Production Checklist

Before deploying `app/` outside localhost:

- [ ] Set `API_KEY` in the backend's `.env`, and `VITE_API_KEY` (same value) in
      `app/frontend/.env` at build time — gates `/api/transcribe` and
      `/api/translate` behind a shared secret. See
      [`app/backend/README.md`](app/backend/README.md#configuration) for what
      this does and does not protect against.
- [ ] Set `CORS_ORIGINS` in the backend's `.env` to the production frontend's
      origin (defaults to the Vite dev server only).
- [ ] If frontend and backend won't share an origin, set `VITE_API_BASE` in
      `app/frontend/.env` to the backend's URL before `npm run build` — see
      [`app/frontend/README.md`](app/frontend/README.md#talking-to-the-backend).
- [ ] Run Uvicorn with **`--workers 1`** (`uvicorn app.backend.app:app --port 8000
      --workers 1`). Rate limiting and model-ready state live in in-process memory
      — more than one worker or replica silently breaks both (each process gets
      its own copy). Don't raise this without first moving that state to a shared
      store (e.g. Redis).
- [ ] Put a reverse proxy (nginx, Caddy, cloud load balancer) in front for
      HTTPS/TLS termination — the app itself serves plain HTTP.

See [`deploy/README.md`](deploy/README.md) for a concrete runbook satisfying
this checklist end-to-end on an IONOS VPS (systemd + Caddy, CPU-only).
