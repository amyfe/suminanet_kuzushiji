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

import io
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

# Make project root importable regardless of working directory
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from config import CHECKPOINT_DIR, DEVICE, IMAGE_SIZE, KURONET_CER_SCORE_THRESH
from model.translation.infer import _TRANSFORM, _align_compile_keys, load_kuronet, run_inference, _unletterbox_boxes
from train_stage2_kuronet import build_kuronet_model, load_vocab


# ---------------------------------------------------------------------------
# Global model state (loaded once at startup)
# ---------------------------------------------------------------------------

_state: dict = {"model": None, "vocab": None, "ready": False, "error": None}

# Translation pipeline is loaded lazily on first request (needs ANTHROPIC_API_KEY)
_translation_pipeline = None


def _get_translation_pipeline():
    global _translation_pipeline
    if _translation_pipeline is None:
        try:
            from model.translation.translation import EdoPeriodTranslationPipeline
            _translation_pipeline = EdoPeriodTranslationPipeline()
        except Exception as exc:
            raise HTTPException(503, detail=f"Translation pipeline unavailable: {exc}")
    return _translation_pipeline


_DEFAULT_CKPT = CHECKPOINT_DIR /"A_first_thesis_run" / "kuronet_recognizer" / "kuronet_best.pt"


@asynccontextmanager
async def lifespan(app: FastAPI):
    import os
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
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
    image: UploadFile = File(...),
    orientation: str = Form(default="auto"),
    score_thresh: float = Form(default=KURONET_CER_SCORE_THRESH),
):
    if not _state["ready"]:
        raise HTTPException(503, detail=_state["error"] or "Model not loaded yet.")

    content_type = image.content_type or ""
    if not image or not content_type.startswith("image/"):
        raise HTTPException(400, detail="Uploaded file must be an image.")

    data = await image.read()
    try:
        image_tensor, orig_size, scale, pad = _preprocess_image(data)
    except Exception as exc:
        raise HTTPException(400, detail=f"Could not decode image: {exc}")

    result = run_inference(
        _state["model"],
        image_tensor,
        _state["vocab"],
        orientation=orientation,
        score_thresh=score_thresh,
    )

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


@app.post("/api/translate")
async def translate(request: TranslateRequest):
    """
    Run the full translation pipeline on classical Japanese text:
      MeCab+UniDic normalization → modern Japanese → English (via Claude).

    Requires ANTHROPIC_API_KEY in the server environment.
    Returns translation result including per-step token usage.
    """
    if not request.text.strip():
        raise HTTPException(400, detail="text must not be empty.")

    pipeline = _get_translation_pipeline()
    try:
        result = pipeline.translate_text(
            request.text,
            strip_furigana=request.strip_furigana,
            normalize_historical=request.normalize_historical,
        )
    except Exception as exc:
        raise HTTPException(500, detail=str(exc))

    return result
