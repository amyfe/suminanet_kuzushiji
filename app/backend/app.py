"""
backend/app.py — FastAPI server for Kuzushiji transcription and translation.

Endpoints
---------
  POST /api/transcribe   — upload image, returns transcription + per-char data
  POST /api/translate    — translate classical Japanese text (full pipeline)
  GET  /api/health       — model load status

Run from project root:
  uvicorn app.backend.app:app --reload --port 8000
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
from pathlib import Path
from typing import Any, Dict, List, Optional

# INFO-level logs (e.g. the per-step timing breakdown in translation.py,
# logged to diagnose slow /api/translate requests) are silently dropped
# without a configured handler — Python's root logger defaults to WARNING.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Make project root importable regardless of working directory
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from config import (
    CHECKPOINT_DIR,
    DEVICE,
    IMAGE_SIZE,
    KURONET_CER_SCORE_THRESH,
    MAX_UPLOAD_SIZE_BYTES,
    TRANSCRIBE_INFERENCE_TIMEOUT_SEC,
    TRANSCRIBE_RATE_LIMIT_MAX_REQUESTS,
    TRANSCRIBE_RATE_LIMIT_WINDOW_SECONDS,
    TRANSLATE_RATE_LIMIT_MAX_REQUESTS,
    TRANSLATE_RATE_LIMIT_WINDOW_SECONDS,
)
from model.translation.infer import _TRANSFORM, load_kuronet, run_inference, _unletterbox_boxes
from train_stage2_kuronet import load_vocab


# ---------------------------------------------------------------------------
# Global model state (loaded once at startup)
# ---------------------------------------------------------------------------

_state: dict = {"model": None, "vocab": None, "ready": False, "error": None}

# Translation pipeline is loaded lazily on first request (needs ANTHROPIC_API_KEY)
_translation_pipeline = None


def _get_translation_pipeline():
    global _translation_pipeline
    if _translation_pipeline is None:
        # Fail fast without touching MeCab/dict extraction when the failure is
        # deterministic (no key configured) — that won't resolve until the
        # server env changes, so retrying the full construction on every
        # request would just waste I/O for the same error.
        if not os.environ.get("OPENROUTER_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"):
            raise HTTPException(
                503,
                detail=(
                    "Translation pipeline unavailable: no API key configured. "
                    "Set ANTHROPIC_API_KEY or OPENROUTER_API_KEY in the server "
                    "environment and restart."
                ),
            )
        try:
            from model.translation.translation import EdoPeriodTranslationPipeline
            _translation_pipeline = EdoPeriodTranslationPipeline()
        except Exception as exc:
            raise HTTPException(503, detail=f"Translation pipeline unavailable: {exc}")
    return _translation_pipeline


_DEFAULT_CKPT = CHECKPOINT_DIR  / "kuronet_recognizer" / "kuronet_best.pt"


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.environ.get("OPENROUTER_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "[WARNING] Neither OPENROUTER_API_KEY nor ANTHROPIC_API_KEY is set. "
            "/api/translate will return 503. Copy .env.example to .env and fill in a key.",
            flush=True,
        )
    print("Loading vocab...", flush=True)
    try:
        vocab = load_vocab()
        print(f"Loading KuroNet from {_DEFAULT_CKPT}...", flush=True)
        model = load_kuronet(_DEFAULT_CKPT, vocab)
        _state["vocab"] = vocab
        _state["model"] = model
        _state["ready"] = True
        print(f"Model ready on {DEVICE}.", flush=True)
    except Exception as exc:
        _state["error"] = str(exc)
        print(f"[ERROR] Model failed to load: {exc}", flush=True)
    yield
    # cleanup (nothing to do for a CPU model)


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
        "device": str(DEVICE),
        "error": _state["error"],
    }


@app.post("/api/transcribe")
async def transcribe(
    http_request: Request,
    image: UploadFile = File(...),
    orientation: str = Form(default="auto"),
    score_thresh: float = Form(default=KURONET_CER_SCORE_THRESH),
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

    inference_start = time.perf_counter()
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                run_inference,
                _state["model"],
                image_tensor,
                _state["vocab"],
                orientation=orientation,
                score_thresh=score_thresh,
            ),
            timeout=TRANSCRIBE_INFERENCE_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            504,
            detail=f"Inference timed out after {TRANSCRIBE_INFERENCE_TIMEOUT_SEC}s.",
        )
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


@app.post("/api/translate")
async def translate(payload: TranslateRequest, http_request: Request):
    """
    Run the full translation pipeline on classical Japanese text:
      MeCab+UniDic normalization → modern Japanese → English (via Claude).

    Requires ANTHROPIC_API_KEY in the server environment.
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
            lang=payload.lang
        )
    except ValueError as exc:
        raise HTTPException(400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(500, detail=str(exc))

    return result
