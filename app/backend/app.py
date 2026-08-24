"""
backend/app.py — FastAPI server for Kuzushiji transcription and translation.

Endpoints
---------
  POST /api/transcribe   — upload image, returns transcription + per-char data
  POST /api/translate    — translate classical Japanese text (full pipeline)
  GET  /api/health       — model load status

"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import sys
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from config import (
    DEVICE,
    IMAGE_SIZE,
    SUMINANET_CER_SCORE_THRESH,
    SUMINANET_CHECKPOINT_DIR,
    GPU_LOAD_MAX_RETRIES,
    GPU_LOAD_RETRY_DELAY_SEC,
    MAX_UPLOAD_SIZE_BYTES,
    TRANSCRIBE_INFERENCE_TIMEOUT_SEC,
    TRANSCRIBE_RATE_LIMIT_MAX_REQUESTS,
    TRANSCRIBE_RATE_LIMIT_WINDOW_SECONDS,
    TRANSLATE_RATE_LIMIT_MAX_REQUESTS,
    TRANSLATE_RATE_LIMIT_WINDOW_SECONDS,
    WEBSITE_CHECKPOINT_DIR
)
from model.translation.infer import _TRANSFORM, load_suminanet, run_inference, _unletterbox_boxes
from train_stage2_suminanet import load_vocab


# ---------------------------------------------------------------------------
# Global model state (loaded once at startup)
#
# On a shared GPU node (as opposed to a dedicated one), other processes can
# saturate the card at any time -- so both a GPU-resident and a CPU-resident
# copy of the model are kept loaded, and a per-request CUDA OOM falls back to
# the CPU copy instead of failing the request (see transcribe()).
# ---------------------------------------------------------------------------

_state: dict = {
    "model_gpu": None,
    "model_cpu": None,
    "vocab": None,
    "ready": False,
    "error": None,
    "device_mode": None,       # "gpu" or "cpu-only", fixed once at startup
    "last_gpu_oom_at": None,   # ISO timestamp of the most recent per-request CPU fallback, or None
}

_translation_pipeline = None


def _is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _get_translation_pipeline():
    global _translation_pipeline
    if _translation_pipeline is None:
        if not os.environ.get("OPENROUTER_API_KEY"):
            raise HTTPException(
                503,
                detail=(
                    "Translation pipeline unavailable: no API key configured. "
                    "Set OPENROUTER_API_KEY in the server "
                    "environment and restart."
                ),
            )
        try:
            from model.translation.translation import EdoPeriodTranslationPipeline
            _translation_pipeline = EdoPeriodTranslationPipeline()
        except Exception as exc:
            raise HTTPException(503, detail=f"Translation pipeline unavailable: {exc}")
    return _translation_pipeline


def _require_api_key(x_api_key: Optional[str] = Header(default=None)):
    """Gate /api/transcribe and /api/translate behind a shared secret.
    """
    expected = os.environ.get("API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(401, detail="Missing or invalid API key.")


def _warn_if_stale_checkpoint() -> None:
    """Compare WEBSITE_CHECKPOINT_DIR's best_score against the live
    SUMINANET_CHECKPOINT_DIR retrain's best_score and warn (does not block
    startup) if the live retrain has since surpassed the pinned checkpoint --
    WEBSITE_CHECKPOINT_DIR is a manually curated pointer to whichever
    archived/live checkpoint is currently best-validated, and nothing else
    keeps it in sync as new training runs complete.
    """
    live_best = Path(SUMINANET_CHECKPOINT_DIR) / "suminanet_best.pt"
    if not live_best.exists() or live_best.resolve() == Path(WEBSITE_CHECKPOINT_DIR).resolve():
        return
    try:
        served_score = torch.load(WEBSITE_CHECKPOINT_DIR, map_location="cpu", weights_only=False).get("best_score")
        live_score = torch.load(live_best, map_location="cpu", weights_only=False).get("best_score")
    except Exception as exc:
        print(f"[WARNING] Could not compare checkpoint scores for staleness check: {exc}", flush=True)
        return
    if served_score is not None and live_score is not None and live_score > served_score:
        print(
            f"[WARNING] WEBSITE_CHECKPOINT_DIR ({WEBSITE_CHECKPOINT_DIR}) has "
            f"best_score={served_score:.4f}, but the live retrain at {live_best} "
            f"now scores higher ({live_score:.4f}). If this is a genuinely better "
            f"model, update WEBSITE_CHECKPOINT_DIR in config.py to point at it.",
            flush=True,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.environ.get("OPENROUTER_API_KEY"):
        print(
            "[WARNING] OPENROUTER_API_KEY is not set. "
            "/api/translate will return 503. Copy .env.example to .env and fill in a key.",
            flush=True,
        )
    if not os.environ.get("API_KEY"):
        print(
            "[WARNING] API_KEY not set — /api/transcribe and /api/translate are "
            "open to anyone. Set API_KEY in .env before deploying outside localhost.",
            flush=True,
        )
    _warn_if_stale_checkpoint()
    print("Loading vocab...", flush=True)
    try:
        vocab = load_vocab()
        _state["vocab"] = vocab

        # CPU copy first: always cheap to build and never touches CUDA, so it
        # can't itself fail because of GPU contention from another process.
        print(f"Loading CPU-resident SuminaNet from {WEBSITE_CHECKPOINT_DIR}...", flush=True)
        _state["model_cpu"] = load_suminanet(WEBSITE_CHECKPOINT_DIR, vocab, device="cpu")
        _state["ready"] = True

        if DEVICE == "cuda":
            for attempt in range(1, GPU_LOAD_MAX_RETRIES + 1):
                try:
                    print(f"Loading GPU-resident SuminaNet (attempt {attempt}/{GPU_LOAD_MAX_RETRIES})...", flush=True)
                    _state["model_gpu"] = load_suminanet(WEBSITE_CHECKPOINT_DIR, vocab, device=DEVICE)
                    _state["device_mode"] = "gpu"
                    print("Model ready on GPU (CPU fallback also loaded).", flush=True)
                    break
                except Exception as exc:
                    is_oom = _is_cuda_oom(exc)
                    reason = "CUDA OOM (likely another process on this shared node)" if is_oom else f"unexpected error: {exc}"
                    print(f"[WARNING] GPU load attempt {attempt}/{GPU_LOAD_MAX_RETRIES} failed ({reason}).", flush=True)
                    if not is_oom:
                        break  # not transient contention -- retrying won't help
                    if attempt < GPU_LOAD_MAX_RETRIES:
                        await asyncio.sleep(GPU_LOAD_RETRY_DELAY_SEC)
            if _state["device_mode"] != "gpu":
                _state["device_mode"] = "cpu-only"
                print("[WARNING] Could not load the model onto the GPU -- serving from "
                      "CPU only until the next restart.", flush=True)
        else:
            _state["device_mode"] = "cpu-only"
            print("Model ready on CPU (no CUDA device available).", flush=True)
    except Exception as exc:
        _state["error"] = str(exc)
        print(f"[ERROR] Model failed to load: {exc}", flush=True)
    yield
    # cleanup (nothing to do for CPU/GPU tensors -- process exit reclaims them)


app = FastAPI(title="Kuzushiji Transcription API", lifespan=lifespan)

_DEFAULT_CORS_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
_cors_env = os.environ.get("CORS_ORIGINS", "")
_cors_origins = [o.strip() for o in _cors_env.split(",") if o.strip()] or _DEFAULT_CORS_ORIGINS

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _preprocess_image(data: bytes) -> tuple:
    """Load image bytes → (1,3,H,W) tensor, original (W,H), scale, (pad_x, pad_y).

    Uses the same letterbox preprocessing as load_image() so that box coordinates
    returned by run_inference() can be correctly mapped back with _unletterbox_boxes().
    """
    img = Image.open(io.BytesIO(data)).convert("RGB")
    orig_size = img.size  # PIL: (W, H)
    orig_w, orig_h = orig_size
    size = IMAGE_SIZE if isinstance(IMAGE_SIZE, int) else IMAGE_SIZE[0]
    scale  = size / max(orig_w, orig_h)
    new_w  = round(orig_w * scale)
    new_h  = round(orig_h * scale)
    resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else 1
    img_resized = img.resize((new_w, new_h), resample)
    pad_x  = (size - new_w) // 2
    pad_y  = (size - new_h) // 2
    canvas = Image.new("RGB", (size, size), (128, 128, 128))
    canvas.paste(img_resized, (pad_x, pad_y))
    transformed: Any = _TRANSFORM(canvas)
    tensor = transformed.unsqueeze(0)
    return tensor, orig_size, scale, (pad_x, pad_y)


# ---------------------------------------------------------------------------
# Per-IP sliding-window rate limiting (shared by /api/transcribe and
# /api/translate, each with their own log + limits). In-memory state is fine
# here: single-process demo app (see INSTALL.md).
# ---------------------------------------------------------------------------

def _check_rate_limit(log: deque, client_ip: str, max_requests: int, window: float) -> None:
    now = time.monotonic()
    while log and now - log[0] > window:
        log.popleft()
    if len(log) >= max_requests:
        retry_after = int(window - (now - log[0])) + 1
        raise HTTPException(
            429,
            detail=(
                f"Rate limit exceeded: max {max_requests} requests per {window}s. "
                f"Try again in {retry_after}s."
            ),
            headers={"Retry-After": str(retry_after)},
        )
    log.append(now)


_translate_request_log: dict[str, deque] = defaultdict(deque)
_transcribe_request_log: dict[str, deque] = defaultdict(deque)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {
        "ready": _state["ready"],
        "device": _state["device_mode"] or str(DEVICE),
        "error": _state["error"],
        "last_gpu_oom_fallback_at": _state["last_gpu_oom_at"],
    }


@app.post("/api/transcribe", dependencies=[Depends(_require_api_key)])
async def transcribe(
    http_request: Request,
    image: UploadFile = File(...),
    orientation: str = Form(default="auto"),
    score_thresh: float = Form(default=SUMINANET_CER_SCORE_THRESH),
):
    if not _state["ready"]:
        raise HTTPException(503, detail=_state["error"] or "Model not loaded yet.")

    client_ip = http_request.client.host if http_request.client else "unknown"
    _check_rate_limit(
        _transcribe_request_log[client_ip],
        client_ip,
        TRANSCRIBE_RATE_LIMIT_MAX_REQUESTS,
        TRANSCRIBE_RATE_LIMIT_WINDOW_SECONDS,
    )

    content_type = image.content_type or ""
    if not image or not content_type.startswith("image/"):
        raise HTTPException(400, detail="Uploaded file must be an image.")

    data = await image.read()
    if len(data) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            413,
            detail=f"Image exceeds max upload size of {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB.",
        )
    try:
        image_tensor, orig_size, scale, pad = _preprocess_image(data)
    except Exception as exc:
        raise HTTPException(400, detail=f"Could not decode image: {exc}")

    async def _run_on(model):
        return await asyncio.wait_for(
            asyncio.to_thread(
                run_inference,
                model,
                image_tensor,
                _state["vocab"],
                orientation=orientation,
                score_thresh=score_thresh,
            ),
            timeout=TRANSCRIBE_INFERENCE_TIMEOUT_SEC,
        )

    inference_start = time.perf_counter()
    primary_model = _state["model_gpu"] if _state["model_gpu"] is not None else _state["model_cpu"]
    try:
        result = await _run_on(primary_model)
    except asyncio.TimeoutError:
        raise HTTPException(
            504,
            detail=f"Inference timed out after {TRANSCRIBE_INFERENCE_TIMEOUT_SEC}s.",
        )
    except Exception as exc:
        # On a shared GPU node, another process saturating the card looks
        # identical to a real bug from here -- only retry on CPU when it's
        # actually a CUDA OOM and a CPU copy exists (i.e. we weren't already
        # running on CPU).
        if primary_model is not _state["model_cpu"] and _state["model_cpu"] is not None and _is_cuda_oom(exc):
            _state["last_gpu_oom_at"] = datetime.now(timezone.utc).isoformat()
            logger.warning(
                "CUDA OOM during transcribe (likely GPU contention from another "
                "process on this shared node) -- retrying on CPU."
            )
            try:
                result = await _run_on(_state["model_cpu"])
            except asyncio.TimeoutError:
                raise HTTPException(
                    504,
                    detail=f"Inference timed out after {TRANSCRIBE_INFERENCE_TIMEOUT_SEC}s (CPU fallback).",
                )
            except Exception:
                logger.exception("run_inference failed on CPU fallback")
                raise HTTPException(500, detail="Transcription failed. Please try again.")
        else:
            logger.exception("run_inference failed")
            raise HTTPException(500, detail="Transcription failed. Please try again.")
    logger.info("run_inference took %.2fs (%d chars)", time.perf_counter() - inference_start, result["n_chars"])

    # Map boxes from letterboxed model coords → original image coords
    _unletterbox_boxes(result["chars"], orig_size, scale, pad)

    return {
        "transcription": result["transcription"],
        "orientation":   result["orientation"],
        "n_chars":       result["n_chars"],
        "chars":         result["chars"],
    }


# ---------------------------------------------------------------------------
# Translation route
# ---------------------------------------------------------------------------

class TranslateRequest(BaseModel):
    text: str
    strip_furigana: bool = True
    normalize_historical: bool = True
    chars: Optional[List[Dict[str, Any]]] = None
    lang: str = "en"  # default to English if not specified
    include_notes: bool = True


@app.post("/api/translate", dependencies=[Depends(_require_api_key)])
async def translate(payload: TranslateRequest, http_request: Request):
    """
    Run the full translation pipeline on classical Japanese text:
      MeCab+UniDic normalization → modern Japanese → English (via Claude).

    Requires OPENROUTER_API_KEY in the server environment.
    Returns translation result including per-step token usage.
    """
    if not payload.text.strip():
        raise HTTPException(400, detail="text must not be empty.")

    client_ip = http_request.client.host if http_request.client else "unknown"
    _check_rate_limit(
        _translate_request_log[client_ip],
        client_ip,
        TRANSLATE_RATE_LIMIT_MAX_REQUESTS,
        TRANSLATE_RATE_LIMIT_WINDOW_SECONDS,
    )

    pipeline = _get_translation_pipeline()
    try:
        result = pipeline.translate_text(
            payload.text,
            strip_furigana=payload.strip_furigana,
            normalize_historical=payload.normalize_historical,
            chars=payload.chars,
            lang=payload.lang,
            include_notes=payload.include_notes,
        )
    except ValueError as exc:
        raise HTTPException(400, detail=str(exc))
    except Exception:
        logger.exception("translate_text failed")
        raise HTTPException(500, detail="Translation failed. Please try again.")

    return result
