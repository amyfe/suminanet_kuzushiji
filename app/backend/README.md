# Backend — Sumina API

FastAPI server that serves the Kuzushiji transcription model and proxies translation
requests to Claude. Single Python process, no database, no background workers.

## Stack

| Layer | Choice |
|---|---|
| Framework | [FastAPI](https://fastapi.tiangolo.com/) 0.139+ on [Uvicorn](https://www.uvicorn.org/) 0.50+ (ASGI) |
| Language | Python 3.10+ |
| ML runtime | PyTorch 2.12+ (CUDA if available, falls back to CPU) |
| Transcription model | SuminaNet (in-house, two-stage detector + recognizer, inspired by KuroNet) |
| Translation model | Claude (`claude-sonnet-4-6`), via OpenRouter |
| Japanese NLP | MeCab + UniDic (via `fugashi`) for classical→modern normalization |

No database — the server is stateless per request. Rate limiting and model state
live in an in-process dict/deque, so this design assumes a **single backend process**
(see [Scaling notes](#scaling-notes) if that changes).

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Model load status (`ready`, `device`, `error`, `last_gpu_oom_fallback_at`) — use as the deploy health check |
| `POST` | `/api/transcribe` | Upload an image → character boxes + transcription |
| `POST` | `/api/translate` | Classical Japanese text → normalized modern Japanese → English/German via Claude |

Rate limits (per client IP, in-memory sliding window): `/api/transcribe` 10 req/60s,
`/api/translate` 5 req/600s. Max upload size 20 MB. Transcription inference has a
60s server-side timeout.

`/api/transcribe` and `/api/translate` require an `X-API-Key` header matching the
`API_KEY` environment variable (see Configuration below) whenever `API_KEY` is set.
`/api/health` never requires it, so it stays usable as an unauthenticated infra
health check.

## Model loading

At startup (`lifespan` in [app.py](app.py)) the server loads **one checkpoint**, at
the path `WEBSITE_CHECKPOINT_DIR` resolves to in [`config.py`](../../config.py):

```
checkpoints/E_final_countdown/suminanet_recognizer/suminanet_best.pt
```

This is a manually curated pointer — `WEBSITE_CHECKPOINT_DIR` in `config.py` is the
single source of truth for which checkpoint is "production"; nothing keeps it in
sync automatically as new training runs complete (see `_warn_if_stale_checkpoint()`
in `app.py`, which only warns, it doesn't switch). This single file is self-contained
— it includes both the Stage 1 detector (EfficientNet-B2 + FPN) and the Stage 2
recognizer (SuminaNet) weights, so it's the only checkpoint the production server
needs to load into memory. See the
[top-level checkpoint inventory](../../checkpoints/README.md) for exact sizes and
what else exists in `checkpoints/` (training-only artifacts that do **not** need to
ship to production).

If the model fails to load, `/api/health` reports the error and `/api/transcribe`
returns `503` — the process stays up so orchestration can retry/restart it or route
around it, rather than crashing at boot.

### Shared-GPU robustness

If `config.py`'s `DEVICE` is `"cuda"`, the server keeps **two** copies of the model
loaded: a GPU-resident one (the one actually used per request) and a CPU-resident
one, held purely as a fallback. This matters when the GPU isn't dedicated to this
process — e.g. running on a shared research node where other jobs can saturate the
card at any time:

- **At startup**, loading the GPU copy is retried up to `GPU_LOAD_MAX_RETRIES` times
  (`config.py`, default 3, `GPU_LOAD_RETRY_DELAY_SEC` apart) on a CUDA OOM. If it
  never succeeds, the server starts in CPU-only mode rather than failing to boot —
  the CPU copy always loads first and unconditionally, since building it never
  touches CUDA.
- **Per request**, a CUDA OOM during inference (not any other error) is retried once
  on the CPU copy transparently — slower for that one request, but the caller gets a
  real result instead of an unexplainable 500.
- `/api/health`'s `device` field reports the actual steady-state mode (`"gpu"` or
  `"cpu-only"`), and `last_gpu_oom_fallback_at` is the timestamp of the most recent
  per-request fallback, so contention from other processes on the node is visible
  rather than silent.

## Configuration

All tunables (device selection, image size, rate limits, timeouts, checkpoint paths)
live in [`config.py`](../../config.py) at the project root, not in this directory.

Environment variables (`.env`, see [`.env.example`](../../.env.example) at project root):

| Variable | Required | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | yes | OpenRouter gateway (routes to `anthropic/claude-sonnet-4-6`) — required for `/api/translate` |
| `CORS_ORIGINS` | no (defaults to Vite dev server) | Comma-separated allowed frontend origins — **must be set to the production frontend URL for deployment** |
| `API_KEY` | no (auth disabled if unset) | Shared secret required via `X-API-Key` on `/api/transcribe` and `/api/translate`. **Recommended for any deployment reachable outside localhost** — stops anonymous scraping of GPU time and paid Claude quota. Must match the frontend's `VITE_API_KEY` build-time value. Note: since the frontend is a public SPA, this key ships inside the built JS bundle and is readable by anyone who inspects it — it deters casual/automated abuse, it is not real access control against a targeted actor. Per-IP rate limiting above still applies regardless. |

The translation pipeline is loaded lazily on first `/api/translate` call, not at
startup — a missing API key doesn't block `/api/transcribe` from working.

## Running

```bash
# from project root
uvicorn app.backend.app:app --reload --port 8000       # dev
uvicorn app.backend.app:app --port 8000 --workers 1     # prod (see note below)
```

**Workers must stay at 1** unless the rate-limit and model-state dicts are moved out
of process memory (e.g. Redis) — each Uvicorn worker is a separate process with its
own copy of `_state`, `_translate_request_log`, etc., and would independently reload
the model into memory and enforce rate limits per-process instead of globally.

For a production systemd deployment, see
[`deploy/README.md`](../../deploy/README.md) and
[`deploy/systemd/sumina-backend.service`](../../deploy/systemd/sumina-backend.service).

## Hardware requirements

- CPU-only works (either because no CUDA device is present, or as the automatic
  per-request/startup fallback described above) but is slow for transcription; a
  CUDA GPU is recommended for the detector/recognizer forward pass.
- No GPU is required for the translation step (that's an API call to Claude).
- See [`../../INSTALL.md`](../../INSTALL.md) for full system dependencies (MeCab,
  UniDic download, SAM2 weights — SAM2 is training-only, **not** loaded at inference).

## Not yet production-hardened

Flagging for visibility before going live:
- Rate limiting is per-process, in-memory — resets on restart, and only enforces
  correctly with exactly one Uvicorn worker/replica (see **Running** above). If this
  ever needs to scale horizontally, the rate-limit and model-state dicts need to
  move to a shared store (e.g. Redis) first.
- Auth is a shared API key readable in the frontend bundle (see `API_KEY` above) —
  a deterrent against bots/scraping, not real access control. A targeted actor can
  read the key out of the built JS and call the API directly.
- No HTTPS/TLS termination in this app itself — expected to sit behind a reverse
  proxy (nginx, Caddy, cloud load balancer) that terminates TLS.
- No structured request logging/metrics beyond Python's `logging` module.
