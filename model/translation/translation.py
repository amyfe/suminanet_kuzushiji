"""
Edo-period Kuzushiji translation pipeline.

Pipeline:  Image → KuroNet transcription
           → MeCab+UniDic normalization  (mecab_normalizer.py)
           → Modern Japanese             (anthropic.py  ClaudeTranslator)
           → English                     (anthropic.py  ClaudeTranslator)

Usage
-----
  pipeline = EdoPeriodTranslationPipeline()

  # From a pre-transcribed string:
  result = pipeline.translate_text("かくて年月を経て、遂に都へ上りけり")

  # End-to-end from an image (requires KuroNet models loaded):
  result = pipeline.process_image("page.jpg", kuronet_model=model, vocab=vocab)

  # From a result.json produced by infer.py:
  result = pipeline.process_result_json("output/result.json")

Result dict keys
----------------
  classical_japanese, normalized_japanese, normalization_method,
  normalization_notes, modern_japanese, conversion_notes,
  english_translation, translation_notes,
  usage: {input_tokens, output_tokens, total_tokens}
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Optional

from model.translation.anthropic import ClaudeTranslator
from model.translation.mecab_normalizer import HistoricalJapaneseNormalizer


# ---------------------------------------------------------------------------
# Pre-processing helpers
# ---------------------------------------------------------------------------

_HIRAGANA = "ぁ-ゖ"
_KATAKANA = "ァ-ヶ"


def _strip_furigana_heuristic(text: str) -> str:
    """Remove isolated kana sandwiched between kanji (likely OCR furigana noise)."""
    return re.sub(
        r"(?<=[一-鿿㐀-䶿])([" + _HIRAGANA + _KATAKANA + r"])(?=[一-鿿㐀-䶿])",
        "",
        text,
    )


def _preprocess(text: str, strip_furigana: bool = True) -> str:
    if strip_furigana:
        text = _strip_furigana_heuristic(text)
    return text.replace("�", "").replace("\x00", "").strip()


def _sum_usage(*usages: Dict[str, int]) -> Dict[str, int]:
    inp = sum(u["input_tokens"] for u in usages)
    out = sum(u["output_tokens"] for u in usages)
    return {"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class EdoPeriodTranslationPipeline:
    """
    Orchestrates MeCab normalization and two-step Claude translation.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._translator = ClaudeTranslator(api_key=api_key)
        self._normalizer = HistoricalJapaneseNormalizer()

    # ------------------------------------------------------------------
    # Normalization (step 0)
    # ------------------------------------------------------------------

    def normalize(self, text: str) -> Dict:
        """Expose normalization result as a plain dict (useful for inspection)."""
        r = self._normalizer.normalize(text)
        return {
            "normalized": r.normalized,
            "method": r.method,
            "notes": r.notes,
            "tokens": r.tokens,
        }

    # ------------------------------------------------------------------
    # Transcription helpers
    # ------------------------------------------------------------------

    def transcribe_image(
        self,
        image_path: str | Path,
        kuronet_model,
        vocab,
        score_thresh: float = 0.0,
        bg_score_gate: float = 0.5,
    ) -> str:
        from model.translation.infer import load_image, run_inference

        image_tensor, _, _, _ = load_image(image_path)
        result = run_inference(
            kuronet_model,
            image_tensor,
            vocab,
            score_thresh=score_thresh,
            bg_score_gate=bg_score_gate,
        )
        return result["transcription"]

    def transcribe_from_result_json(self, result_json_path: str | Path) -> str:
        with open(result_json_path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("transcription", "")

    # ------------------------------------------------------------------
    # Core translation
    # ------------------------------------------------------------------

    def translate_text(
        self,
        classical_text: str,
        strip_furigana: bool = True,
        normalize_historical: bool = True,
    ) -> Dict:
        cleaned = _preprocess(classical_text, strip_furigana=strip_furigana)

        # Step 0: MeCab + UniDic normalization
        if normalize_historical:
            norm = self._normalizer.normalize(cleaned)
            text_for_llm = norm.normalized
        else:
            norm = None
            text_for_llm = cleaned

        # Step 1: classical → modern Japanese
        modern, modern_usage = self._translator.classical_to_modern(text_for_llm)

        # Step 2: modern → English
        translation, english_usage = self._translator.translate_to_english(
            modern["modern_japanese"], classical_text=cleaned
        )

        return {
            "classical_japanese":   cleaned,
            "normalized_japanese":  norm.normalized if norm else cleaned,
            "normalization_method": norm.method if norm else "none",
            "normalization_notes":  norm.notes if norm else "",
            "modern_japanese":      modern["modern_japanese"],
            "conversion_notes":     modern.get("notes", ""),
            "english_translation":  translation["english_translation"],
            "translation_notes":    translation.get("translation_notes", ""),
            "usage":                _sum_usage(modern_usage, english_usage),
        }

    # ------------------------------------------------------------------
    # End-to-end helpers
    # ------------------------------------------------------------------

    def process_image(
        self,
        image_path: str | Path,
        kuronet_model,
        vocab,
        strip_furigana: bool = True,
        normalize_historical: bool = True,
        score_thresh: float = 0.0,
        bg_score_gate: float = 0.5,
    ) -> Dict:
        classical = self.transcribe_image(
            image_path,
            kuronet_model,
            vocab,
            score_thresh=score_thresh,
            bg_score_gate=bg_score_gate,
        )
        return self.translate_text(
            classical,
            strip_furigana=strip_furigana,
            normalize_historical=normalize_historical,
        )

    def process_result_json(
        self,
        result_json_path: str | Path,
        strip_furigana: bool = True,
        normalize_historical: bool = True,
    ) -> Dict:
        classical = self.transcribe_from_result_json(result_json_path)
        return self.translate_text(
            classical,
            strip_furigana=strip_furigana,
            normalize_historical=normalize_historical,
        )
