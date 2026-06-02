"""
Edo-period Kuzushiji translation pipeline.

Pipeline:  Image → KuroNet transcription → MeCab+UniDic normalization
           → Modern Japanese → English

Usage
-----
  # From a pre-transcribed string:
  pipeline = EdoPeriodTranslationPipeline()
  result = pipeline.translate_text("此蝦にいのじはがる")

  # End-to-end from an image (requires KuroNet models loaded):
  result = pipeline.process_image(
      image_path="page.jpg",
      kuronet_model=model,
      vocab=vocab,
  )

  # From a result.json produced by infer.py:
  result = pipeline.process_result_json("output/result.json")
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, Optional

import anthropic
from anthropic.types import TextBlock

from model.translation.mecab_normalizer import HistoricalJapaneseNormalizer


# Characters that are very likely furigana (small kana reading hints beside kanji).
_HIRAGANA  = "ぁ-ゖ"
_KATAKANA  = "ァ-ヶ"
_RE_KANA_ONLY = re.compile(f"^[{_HIRAGANA}{_KATAKANA}]+$")


def _strip_furigana_heuristic(text: str) -> str:
    """
    Lightweight furigana filter: remove isolated kana-only runs that are
    surrounded by kanji (likely reading annotations injected by the OCR).
    """
    text = re.sub(
        r"(?<=[一-鿿㐀-䶿])([" + _HIRAGANA + _KATAKANA + r"])(?=[一-鿿㐀-䶿])",
        "",
        text,
    )
    return text


def _preprocess_for_translation(text: str, strip_furigana: bool = True) -> str:
    """Clean up transcription before sending to the LLM."""
    if strip_furigana:
        text = _strip_furigana_heuristic(text)
    text = text.replace("", "").replace("\x00", "")
    return text.strip()


def _sum_usage(*responses: anthropic.types.Message) -> Dict[str, int]:
    """Accumulate input/output token counts across multiple API responses."""
    inp = sum(r.usage.input_tokens for r in responses)
    out = sum(r.usage.output_tokens for r in responses)
    return {"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out}


class EdoPeriodTranslationPipeline:
    """
    Three-step pipeline:
      0. MeCab + UniDic normalization (ChaMame-style historical kana normalization)
      1. Classical Japanese → Modern Japanese  (Claude)
      2. Modern Japanese   → English           (Claude)
    """

    MODEL = "claude-sonnet-4-6"

    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "Anthropic API key required. Set the ANTHROPIC_API_KEY environment "
                "variable or pass api_key= to the constructor."
            )
        self.client = anthropic.Anthropic(api_key=key)
        self._normalizer = HistoricalJapaneseNormalizer()

    # ------------------------------------------------------------------
    # Step 0: MeCab + UniDic normalization
    # ------------------------------------------------------------------

    def normalize_historical(self, text: str) -> Dict:
        """
        Normalize Edo-period kana conventions to modern kana using MeCab + UniDic.
        Returns a dict with normalized text, method used, and token list.
        """
        result = self._normalizer.normalize(text)
        return {
            "normalized": result.normalized,
            "method": result.method,
            "notes": result.notes,
            "tokens": result.tokens,
        }

    # ------------------------------------------------------------------
    # Step 1: transcription (delegates to infer.py)
    # ------------------------------------------------------------------

    def transcribe_image(
        self,
        image_path: str | Path,
        kuronet_model,
        vocab,
        score_thresh: float = 0.0,
        bg_score_gate: float = 0.5,
    ) -> str:
        from infer import load_image, run_inference

        image_tensor, _ = load_image(image_path)
        result = run_inference(
            kuronet_model, image_tensor, vocab,
            score_thresh=score_thresh,
            bg_score_gate=bg_score_gate,
        )
        return result["transcription"]

    def transcribe_from_result_json(self, result_json_path: str | Path) -> str:
        with open(result_json_path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("transcription", "")

    # ------------------------------------------------------------------
    # Step 2: classical → modern Japanese
    # ------------------------------------------------------------------

    def _build_classical_to_modern_prompt(self, text: str) -> str:
        return f"""You are an expert in classical Japanese literature, specifically Edo period texts (1603-1868).

Convert the following classical Japanese text to modern Japanese (現代日本語).

Requirements:
- Preserve the original meaning and nuance precisely
- Convert classical grammar forms to modern equivalents
- Update archaic vocabulary to contemporary terms
- Maintain the tone and register of the original
- Note ambiguous passages

The text was produced by an OCR model and may contain recognition errors. Use context to infer the intended meaning where possible.

Classical Japanese text:
{text}

Respond in JSON format:
{{
    "modern_japanese": "the converted modern Japanese text",
    "notes": "any ambiguities or important conversion decisions (empty string if none)"
}}"""

    def classical_to_modern(self, classical_text: str) -> Dict[str, str]:
        """Convert Edo-period classical Japanese to modern Japanese via Claude."""
        prompt = self._build_classical_to_modern_prompt(classical_text)
        response = self._call_claude(prompt)
        return self._parse_json_response(self._response_text(response))

    # ------------------------------------------------------------------
    # Step 3: modern Japanese → English
    # ------------------------------------------------------------------

    def _build_translate_to_english_prompt(
        self, modern_japanese: str, classical_text: str = ""
    ) -> str:
        context = (
            f"\n\nOriginal classical text for reference:\n{classical_text}"
            if classical_text
            else ""
        )
        return f"""Translate the following modern Japanese text to natural, fluent English.

The text is from an Edo-period Japanese document that has been converted to modern Japanese. Your translation should:
- Read naturally in English
- Preserve the meaning and nuance
- Maintain appropriate formality
- Note important cultural context where relevant

Modern Japanese text:
{modern_japanese}{context}

Respond in JSON format:
{{
    "english_translation": "the English translation",
    "translation_notes": "cultural context or important decisions (empty string if none)"
}}"""

    def translate_to_english(
        self, modern_japanese: str, classical_text: str = ""
    ) -> Dict[str, str]:
        """Translate modern Japanese to English via Claude."""
        prompt = self._build_translate_to_english_prompt(modern_japanese, classical_text)
        response = self._call_claude(prompt)
        return self._parse_json_response(self._response_text(response))

    # ------------------------------------------------------------------
    # Combined pipelines
    # ------------------------------------------------------------------

    def translate_text(
        self,
        classical_text: str,
        strip_furigana: bool = True,
        normalize_historical: bool = True,
    ) -> Dict:
        """
        Full pipeline from a classical Japanese string.

        Returns dict with: classical_japanese, normalized_japanese,
        normalization_method, normalization_notes, modern_japanese,
        conversion_notes, english_translation, translation_notes, usage.
        """
        cleaned = _preprocess_for_translation(classical_text, strip_furigana=strip_furigana)

        # Step 0: MeCab + UniDic normalization
        if normalize_historical:
            norm = self._normalizer.normalize(cleaned)
            text_for_llm = norm.normalized
        else:
            norm = None
            text_for_llm = cleaned

        # Step 1: classical → modern Japanese
        modern_prompt = self._build_classical_to_modern_prompt(text_for_llm)
        modern_resp = self._call_claude(modern_prompt)
        modern = self._parse_json_response(self._response_text(modern_resp))

        # Step 2: modern → English
        english_prompt = self._build_translate_to_english_prompt(
            modern["modern_japanese"], classical_text=cleaned
        )
        english_resp = self._call_claude(english_prompt)
        translation = self._parse_json_response(self._response_text(english_resp))

        return {
            "classical_japanese":    cleaned,
            "normalized_japanese":   norm.normalized if norm else cleaned,
            "normalization_method":  norm.method if norm else "none",
            "normalization_notes":   norm.notes if norm else "",
            "modern_japanese":       modern["modern_japanese"],
            "conversion_notes":      modern.get("notes", ""),
            "english_translation":   translation["english_translation"],
            "translation_notes":     translation.get("translation_notes", ""),
            "usage":                 _sum_usage(modern_resp, english_resp),
        }

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
        """End-to-end: image file → English translation."""
        classical = self.transcribe_image(
            image_path, kuronet_model, vocab,
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
        """Translate from a pre-computed infer.py result.json file."""
        classical = self.transcribe_from_result_json(result_json_path)
        return self.translate_text(
            classical,
            strip_furigana=strip_furigana,
            normalize_historical=normalize_historical,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _call_claude(self, prompt: str) -> anthropic.types.Message:
        return self.client.messages.create(
            model=self.MODEL,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )

    @staticmethod
    def _response_text(response: anthropic.types.Message) -> str:
        for block in response.content:
            if isinstance(block, TextBlock):
                return block.text
        raise ValueError(f"No TextBlock found in response content: {response.content}")

    @staticmethod
    def _parse_json_response(text: str) -> dict:
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0]
        return json.loads(text.strip())


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample = "かくて年月を経て、遂に都へ上りけり"
    print(f"Input: {sample}\n")

    pipeline = EdoPeriodTranslationPipeline()

    norm = pipeline.normalize_historical(sample)
    print(f"Normalized ({norm['method']}): {norm['normalized']}")
    if norm["notes"]:
        print(f"Norm notes: {norm['notes']}")

    result = pipeline.translate_text(sample)

    print(f"\nModern Japanese : {result['modern_japanese']}")
    if result["conversion_notes"]:
        print(f"Conversion notes: {result['conversion_notes']}")
    print(f"\nEnglish         : {result['english_translation']}")
    if result["translation_notes"]:
        print(f"Translation notes: {result['translation_notes']}")
    print(f"\nTokens used     : {result['usage']}")
