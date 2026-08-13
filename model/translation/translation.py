"""
Edo-period Kuzushiji translation pipeline.

Pipeline:  Image → SuminaNet transcription
           → MeCab+UniDic normalization  (mecab_normalizer.py)
           → Modern Japanese             (anthropic.py  ClaudeTranslator)
           → English                     (anthropic.py  ClaudeTranslator)

Usage
-----
  # From a pre-transcribed string:
  result = pipeline.translate_text("かくて年月を経て、遂に都へ上りけり")

  # End-to-end from an image (requires SuminaNet models loaded):
  result = pipeline.process_image("page.jpg", suminanet_model=model, vocab=vocab)

  # From a result.json produced by infer.py:
  result = pipeline.process_result_json("output/result.json")

Result dict keys
----------------
  classical_japanese, normalized_japanese, normalization_method,
  normalization_notes, modern_japanese, conversion_notes, conversion_truncated,
  english_translation, translation_notes, translation_truncated,
  usage: {input_tokens, output_tokens, total_tokens}
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

from config import (
    SUMINANET_CONTEXT_BLOCK_GAP_FACTOR,
    TRANSLATION_MAX_INPUT_CHARS,
    TRANSLATION_UNCERTAIN_SCORE_THRESH,
)
from model.translation.anthropic import ClaudeTranslator
from model.translation.mecab_normalizer import HistoricalJapaneseNormalizer
from utils.training_helpers.helper_translation import _build_llm_text, _preprocess, _sum_usage
logger = logging.getLogger(__name__)

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
        suminanet_model,
        vocab,
        score_thresh: float = 0.0,
        bg_score_gate: float = 0.5,
    ) -> str:
        from model.translation.infer import load_image, run_inference

        image_tensor, _, _, _ = load_image(image_path)
        result = run_inference(
            suminanet_model,
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
        chars: Optional[List[Dict]] = None,
        combined: bool = True,
        lang: str = "en",
        include_notes: bool = True,
    ) -> Dict:
        if chars:
            actual = "".join(c.get("char", "") for c in chars)
            if actual != classical_text:
                logger.warning(
                    "chars misaligned with classical_text (len %d vs %d); "
                    "ignoring chars for this request.",
                    len(actual), len(classical_text),
                )
                chars = None

        cleaned = _preprocess(classical_text, strip_furigana=strip_furigana)

        if len(cleaned) > TRANSLATION_MAX_INPUT_CHARS:
            raise ValueError(
                f"Input text too long ({len(cleaned)} chars, max "
                f"{TRANSLATION_MAX_INPUT_CHARS}). Split into smaller sections "
                "and translate separately."
            )

        # Step 0: MeCab + UniDic normalization
        pipeline_start = time.perf_counter()
        step_start = pipeline_start
        if normalize_historical:
            norm = self._normalizer.normalize(cleaned)
        else:
            norm = None
        logger.info("translate_text: normalize step took %.2fs", time.perf_counter() - step_start)

        if chars:
            annotated = _build_llm_text(
                classical_text, chars, TRANSLATION_UNCERTAIN_SCORE_THRESH, SUMINANET_CONTEXT_BLOCK_GAP_FACTOR,
                mark_furigana=strip_furigana,
            )
            # this _preprocess call now only does stray-char/whitespace cleanup.
            text_for_llm = _preprocess(annotated, strip_furigana=False)
            mecab_reference = norm.normalized if norm else None
        else:
            text_for_llm = norm.normalized if norm else cleaned
            mecab_reference = None
        step_start = time.perf_counter()
        if combined:
            # Step 1+2: classical → modern → English in one call
            translation, usage = self._translator.translate_classical_to_english(
                text_for_llm, mecab_reference=mecab_reference, lang=lang, include_notes=include_notes
            )
            logger.info("translate_text: classical_to_modern_and_english (Claude) took %.2fs", time.perf_counter() - step_start)
            logger.info("translate_text: total pipeline took %.2fs", time.perf_counter() - pipeline_start)
            modern_japanese = translation["modern_japanese"]
            conversion_notes = translation.get("notes", "")
        else:
            # Step 1: classical → modern Japanese
            modern, modern_usage = self._translator.classical_to_modern(
                text_for_llm, mecab_reference=mecab_reference, include_notes=include_notes
            )
            logger.info("translate_text: classical_to_modern (Claude) took %.2fs", time.perf_counter() - step_start)

            # Step 2: modern → English
            step_start = time.perf_counter()
            translation, english_usage = self._translator.translate_to_english(
                modern["modern_japanese"], classical_text=cleaned, lang=lang, include_notes=include_notes
            )
            logger.info("translate_text: translate_to_english (Claude) took %.2fs", time.perf_counter() - step_start)
            logger.info("translate_text: total pipeline took %.2fs", time.perf_counter() - pipeline_start)
            usage = _sum_usage(modern_usage, english_usage)
            modern_japanese = modern["modern_japanese"]
            conversion_notes = modern.get("notes", "")
        return {
            "classical_japanese":   cleaned,
            "normalized_japanese":  norm.normalized if norm else cleaned,
            "normalization_method": norm.method if norm else "none",
            "normalization_notes":  norm.notes if norm else "",
            "modern_japanese":      modern_japanese,
            "conversion_notes":     conversion_notes,
            "english_translation":  translation["english_translation"],
            "translation_notes":    translation.get("translation_notes", ""),
            "translation_truncated": translation.get("truncated", False),
            "usage":                usage,
        }

    # ------------------------------------------------------------------
    # End-to-end helpers
    # ------------------------------------------------------------------

    def process_image(
        self,
        image_path: str | Path,
        suminanet_model,
        vocab,
        strip_furigana: bool = True,
        normalize_historical: bool = True,
        score_thresh: float = 0.0,
        bg_score_gate: float = 0.5,
    ) -> Dict:
        classical = self.transcribe_image(
            image_path,
            suminanet_model,
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
