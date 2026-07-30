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
  normalization_notes, modern_japanese, conversion_notes, conversion_truncated,
  english_translation, translation_notes, translation_truncated,
  usage: {input_tokens, output_tokens, total_tokens}
"""

from __future__ import annotations

import json
import logging
import math
import re
import statistics
from pathlib import Path
from typing import Dict, List, Optional

from config import (
    FURIGANA_MIN_RUN_LENGTH,
    KURONET_CONTEXT_BLOCK_GAP_FACTOR,
    TRANSLATION_MAX_INPUT_CHARS,
    TRANSLATION_UNCERTAIN_SCORE_THRESH,
)
from model.translation.anthropic import ClaudeTranslator
from model.translation.mecab_normalizer import HistoricalJapaneseNormalizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pre-processing helpers
# ---------------------------------------------------------------------------

_HIRAGANA = "ぁ-ゖ"
_KATAKANA = "ァ-ヶ"

_FURIGANA_RUN_RE = re.compile(
    r"(?<=[一-鿿㐀-䶿])([" + _HIRAGANA + _KATAKANA + r"]{" + str(FURIGANA_MIN_RUN_LENGTH) + r",})(?=[一-鿿㐀-䶿])"
)


def _strip_furigana_heuristic(text: str) -> str:
    """Flag runs of FURIGANA_MIN_RUN_LENGTH+ kana sandwiched between kanji as
    possible furigana (likely OCR noise), rather than silently deleting them.

    A single kana between kanji is left untouched — it's more likely a
    grammatical particle/okurigana than a furigana gloss. Every match is
    logged and replaced with a "[furigana?:...]" marker instead of being
    removed, so nothing is silently lost from the text; the Claude prompt
    explains the marker so it can use judgement on flagged spans.
    """
    def _mark(m: "re.Match[str]") -> str:
        run = m.group(1)
        logger.info("Possible furigana run marked: %r at offset %d", run, m.start(1))
        return f"[furigana?:{run}]"

    return _FURIGANA_RUN_RE.sub(_mark, text)


def _detect_block_breaks(chars: List[Dict], gap_factor: float) -> set:
    """Indices into chars/classical_text where a new independent text block
    starts. Reimplements the median-gap heuristic from _compute_block_ids()
    in model/kuronet/kuronet_recognizer.py in plain Python (this module
    stays torch-free) — keep the two in sync if the heuristic changes.
    """
    if len(chars) < 2:
        return set()
    centres = [
        ((c["box"][0] + c["box"][2]) / 2.0, (c["box"][1] + c["box"][3]) / 2.0)
        for c in chars
    ]
    dists = [
        math.hypot(centres[i + 1][0] - centres[i][0], centres[i + 1][1] - centres[i][1])
        for i in range(len(centres) - 1)
    ]
    median = statistics.median(dists)
    if median <= 1e-6:
        return set()
    return {i + 1 for i, d in enumerate(dists) if d > gap_factor * median}


def _build_llm_text(text: str, chars: List[Dict], uncertain_threshold: float, block_gap_factor: float) -> str:
    """Build the LLM-facing text: insert "｜" at detected text-block
    boundaries and wrap OCR-uncertain characters as "[char:NN%]".

    `chars` must be positionally 1:1 with `text` (guaranteed by
    run_inference()'s chars_out). The two transforms are combined into a
    single pass since inserting either kind of token would break the index
    alignment the other one needs.
    """
    breaks = _detect_block_breaks(chars, block_gap_factor)
    out = []
    for i, (ch, c) in enumerate(zip(text, chars)):
        if i in breaks:
            out.append("｜")
        score = c.get("score", 1.0)
        if score < uncertain_threshold:
            out.append(f"[{ch}:{score * 100:.0f}%]")
        else:
            out.append(ch)
    return "".join(out)


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
        chars: Optional[List[Dict]] = None,
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

        # Plain track: always derived from the raw, un-annotated text.
        # Feeds classical_japanese/normalized_japanese (display fields) and
        # the classical_text= reference context for the English-translation
        # call. Unaffected by confidence annotation.
        cleaned = _preprocess(classical_text, strip_furigana=strip_furigana)

        if len(cleaned) > TRANSLATION_MAX_INPUT_CHARS:
            raise ValueError(
                f"Input text too long ({len(cleaned)} chars, max "
                f"{TRANSLATION_MAX_INPUT_CHARS}). Split into smaller sections "
                "and translate separately."
            )

        # Step 0: MeCab + UniDic normalization
        if normalize_historical:
            norm = self._normalizer.normalize(cleaned)
        else:
            norm = None

        # LLM-input track: only for the classical_to_modern call. When chars
        # are available, annotate uncertain characters onto the raw text
        # first (preserves strict 1:1 alignment with chars), then apply
        # furigana-marking on top. MeCab normalization is intentionally
        # skipped for this call — its orthBase substitution can change
        # character identity, and its behavior on injected ASCII/bracket
        # annotations is unvalidated. The plain `norm` above still populates
        # normalized_japanese/normalization_method for display.
        if chars:
            annotated = _build_llm_text(
                classical_text, chars, TRANSLATION_UNCERTAIN_SCORE_THRESH, KURONET_CONTEXT_BLOCK_GAP_FACTOR
            )
            text_for_llm = _preprocess(annotated, strip_furigana=strip_furigana)
        else:
            text_for_llm = norm.normalized if norm else cleaned

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
            "conversion_truncated": modern.get("truncated", False),
            "english_translation":  translation["english_translation"],
            "translation_notes":    translation.get("translation_notes", ""),
            "translation_truncated": translation.get("truncated", False),
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
