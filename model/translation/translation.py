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
import time
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


def _detect_block_breaks_by_distance(chars: List[Dict], gap_factor: float) -> set:
    """Fallback heuristic for chars lists without `col_id` (older cached
    result.json files predating column-id support, or synthetic chars lists
    built without it): flags a break wherever the Euclidean distance between
    consecutive box centres exceeds gap_factor x the page's median inter-char
    distance.

    Kept only as a fallback -- on any real multi-column vertical page this
    fires at EVERY column transition (bottom of one column to top of the
    next is a large jump by construction), not just genuine content breaks.
    Prefer _detect_column_breaks + _detect_block_breaks below whenever
    `col_id` is available, which is now the normal case via run_inference().
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


def _detect_column_breaks(chars: List[Dict]) -> set:
    """Indices into chars/classical_text where a new column (or row, for
    horizontal orientation) starts, using each char's `col_id` (from
    ROIReadingOrder, threaded through by run_inference()). Deterministic --
    no threshold. Returns an empty set if `col_id` isn't present on every
    char, signalling callers to fall back to the distance heuristic instead.
    """
    if len(chars) < 2 or any("col_id" not in c for c in chars):
        return set()
    return {
        i + 1 for i in range(len(chars) - 1)
        if chars[i]["col_id"] != chars[i + 1]["col_id"]
    }


def _column_stats(chars: List[Dict], col_id: int) -> dict:
    """Aggregate geometry for one column: median character area, vertical
    span, and mean x-center -- used to judge whether an adjacent column looks
    like a continuation of the same content or genuinely different content.
    """
    boxes = [c["box"] for c in chars if c.get("col_id") == col_id]
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
    ys = [(b[1] + b[3]) / 2.0 for b in boxes]
    xs = [(b[0] + b[2]) / 2.0 for b in boxes]
    return {
        "median_area": statistics.median(areas),
        "y_span": (max(ys) - min(ys)) if len(ys) > 1 else 0.0,
        "x_center": statistics.mean(xs),
    }


def _detect_block_breaks(
    chars: List[Dict],
    gap_factor: float,
    area_ratio_thresh: float = 2.0,
    gap_pitch_factor: float = 2.0,
) -> set:
    """Indices into chars/classical_text where a new independent text block
    starts (a caption, marginal note, or unrelated inscription) -- as opposed
    to a mere column wrap within one continuous narrative.

    When `col_id` is available, a block boundary is only ever considered AT a
    column-transition point (see _detect_column_breaks), and only when the
    two adjacent columns also look like different content: their median
    character area differs by more than area_ratio_thresh x, or the
    horizontal gap between their x-centers exceeds gap_pitch_factor x the
    page's typical inter-column pitch. A column transition where neither
    holds is treated as a plain column wrap, not a block boundary.

    Falls back to _detect_block_breaks_by_distance when `col_id` is
    unavailable on the chars list (thresholds above then unused). Exact
    thresholds are a first pass -- validate visually against real
    multi-column pages before treating them as final.
    """
    column_breaks = _detect_column_breaks(chars)
    if not column_breaks:
        if len(chars) >= 2 and any("col_id" not in c for c in chars):
            return _detect_block_breaks_by_distance(chars, gap_factor)
        return set()

    col_order: List[int] = []
    seen = set()
    for c in chars:
        cid = c["col_id"]
        if cid not in seen:
            seen.add(cid)
            col_order.append(cid)

    stats_by_col = {cid: _column_stats(chars, cid) for cid in col_order}
    pitches = [
        abs(stats_by_col[col_order[i + 1]]["x_center"] - stats_by_col[col_order[i]]["x_center"])
        for i in range(len(col_order) - 1)
    ]
    median_pitch = statistics.median(pitches) if pitches else 0.0

    blocks = set()
    for i in column_breaks:
        s_prev = stats_by_col[chars[i - 1]["col_id"]]
        s_next = stats_by_col[chars[i]["col_id"]]

        area_ratio = max(s_prev["median_area"], s_next["median_area"]) / max(
            1e-6, min(s_prev["median_area"], s_next["median_area"])
        )
        pitch_gap = abs(s_next["x_center"] - s_prev["x_center"])

        if area_ratio > area_ratio_thresh or (
            median_pitch > 1e-6 and pitch_gap > gap_pitch_factor * median_pitch
        ):
            blocks.add(i)
    return blocks


def _build_llm_text(text: str, chars: List[Dict], uncertain_threshold: float, block_gap_factor: float) -> str:
    """Build the LLM-facing text: insert "｜" at detected text-block
    boundaries, a plain "\\n" at column-only boundaries (no content change,
    just a layout wrap), and wrap OCR-uncertain characters as "[char:NN%]".

    `chars` must be positionally 1:1 with `text` (guaranteed by
    run_inference()'s chars_out). The three transforms are combined into a
    single pass since inserting any of them would break the index alignment
    the others need.
    """
    column_breaks = _detect_column_breaks(chars)
    block_breaks = _detect_block_breaks(chars, block_gap_factor)
    out = []
    for i, (ch, c) in enumerate(zip(text, chars)):
        if i in block_breaks:
            out.append("｜")
        elif i in column_breaks:
            out.append("\n")
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
        combined: bool = True,
        lang: str = "en",
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
                classical_text, chars, TRANSLATION_UNCERTAIN_SCORE_THRESH, KURONET_CONTEXT_BLOCK_GAP_FACTOR
            )
            text_for_llm = _preprocess(annotated, strip_furigana=strip_furigana)
        else:
            text_for_llm = norm.normalized if norm else cleaned

        mecab_reference = norm.normalized if (chars and norm) else None
        step_start = time.perf_counter()
        if combined:
            # Step 1+2: classical → modern → English in one call
            translation, usage = self._translator.translate_classical_to_english(
                text_for_llm, mecab_reference=mecab_reference, lang=lang
            )
            logger.info("translate_text: classical_to_modern_and_english (Claude) took %.2fs", time.perf_counter() - step_start)
            logger.info("translate_text: total pipeline took %.2fs", time.perf_counter() - pipeline_start)
            modern_japanese = translation["modern_japanese"]
            conversion_notes = translation.get("notes", "")
        else:
            # Step 1: classical → modern Japanese
            modern, modern_usage = self._translator.classical_to_modern(
                text_for_llm, mecab_reference=mecab_reference
            )
            logger.info("translate_text: classical_to_modern (Claude) took %.2fs", time.perf_counter() - step_start)

            # Step 2: modern → English
            step_start = time.perf_counter()
            translation, english_usage = self._translator.translate_to_english(
                modern["modern_japanese"], classical_text=cleaned, lang=lang
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
