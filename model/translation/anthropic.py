"""
Claude-based translation steps for Edo-period Japanese.

ClaudeTranslator handles only the LLM calls and prompt construction.
Orchestration (normalization → modern → English) lives in translation.py.

Backend selection (auto-detected from env vars):
  OPENROUTER_API_KEY  →  OpenRouter  (openai-compatible, model: anthropic/claude-sonnet-4-6)
  ANTHROPIC_API_KEY   →  Direct Anthropic API
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import anthropic

from config import (
    TRANSLATION_API_MAX_RETRIES,
    TRANSLATION_API_TIMEOUT_SEC,
    TRANSLATION_MAX_TOKENS_CEILING,
    TRANSLATION_MAX_TOKENS_CHARS_MULTIPLIER,
    TRANSLATION_MAX_TOKENS_FLOOR,
)

logger = logging.getLogger(__name__)

_OPENROUTER_BASE = "https://openrouter.ai/api/v1"
_OPENROUTER_MODEL = "anthropic/claude-sonnet-4-6"

_TARGET_LANGUAGES = {"en": "English", "de": "German"}


def _target_language_name(lang: str) -> str:
    try:
        return _TARGET_LANGUAGES[lang]
    except KeyError:
        raise ValueError(
            f"Unsupported target language {lang!r}; expected one of {sorted(_TARGET_LANGUAGES)}"
        ) from None


def _estimate_max_tokens(text: str) -> int:
    """Output token budget scaled from input length, clamped to
    [TRANSLATION_MAX_TOKENS_FLOOR, TRANSLATION_MAX_TOKENS_CEILING]."""
    estimated = int(len(text) * TRANSLATION_MAX_TOKENS_CHARS_MULTIPLIER)
    return max(TRANSLATION_MAX_TOKENS_FLOOR, min(estimated, TRANSLATION_MAX_TOKENS_CEILING))


class ClaudeTranslator:
    """Wraps the two Claude translation steps with prompt building and usage tracking."""

    MODEL = "claude-sonnet-4-6"

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.client: Any
        or_key = os.environ.get("OPENROUTER_API_KEY")
        an_key = os.environ.get("ANTHROPIC_API_KEY")

        if or_key and not api_key:
            import openai
            self.client = openai.OpenAI(
                base_url=_OPENROUTER_BASE,
                api_key=or_key,
                timeout=TRANSLATION_API_TIMEOUT_SEC,
                max_retries=TRANSLATION_API_MAX_RETRIES,
            )
            self._backend = "openrouter"
        else:
            key = api_key or an_key
            if not key:
                raise ValueError(
                    "API key required. Set OPENROUTER_API_KEY or ANTHROPIC_API_KEY, "
                    "or pass api_key=."
                )
            self.client = anthropic.Anthropic(
                api_key=key,
                timeout=TRANSLATION_API_TIMEOUT_SEC,
                max_retries=TRANSLATION_API_MAX_RETRIES,
            )
            self._backend = "anthropic"

    # ------------------------------------------------------------------
    # Step 1: classical → modern Japanese
    # ------------------------------------------------------------------

    def classical_to_modern(
        self, text: str, mecab_reference: Optional[str] = None
    ) -> Tuple[Dict[str, str], Dict[str, int]]:
        """
        Convert classical/Edo-period Japanese to modern Japanese.

        mecab_reference: optional dictionary-normalized text (MeCab + UniDic orthBase
            substitution) for the same source, given to the model as a secondary aid for
            archaic vocabulary/spelling. Not positionally aligned to `text` and contains
            none of its inline annotation markers — see _build_classical_to_modern_prompt.

        Returns (result, usage) where result has keys:
          modern_japanese, notes, truncated
        and usage has keys:
          input_tokens, output_tokens, total_tokens
        """
        prompt = self._build_classical_to_modern_prompt(text, mecab_reference)
        result, response = self._call_structured(
            prompt,
            tool_name="provide_modern_japanese",
            description="Provide the modern Japanese conversion of classical Japanese text.",
            properties={
                "modern_japanese": {
                    "type": "string",
                    "description": "The converted modern Japanese text",
                },
                "notes": {
                    "type": "string",
                    "description": "Any ambiguities or important conversion decisions (empty string if none)",
                },
            },
            required=["modern_japanese", "notes"],
            max_tokens=_estimate_max_tokens(text),
        )
        return result, self._usage(response)

    def _build_classical_to_modern_prompt(
        self, text: str, mecab_reference: Optional[str] = None
    ) -> str:
        reference_block = ""
        if mecab_reference:
            reference_block = f"""

A dictionary-normalized reference version of the same text is provided below, produced by a
separate morphological analyzer (MeCab + UniDic) that substitutes each word's standardized
modern spelling. Character positions in this reference do NOT correspond 1:1 to the annotated
text above (words may have been merged, split, or rewritten during normalization), and it
contains none of the inline markers described above. Use it only as an aid for resolving
archaic vocabulary and spelling — the annotated text above remains the authoritative source for
character-level uncertainty and document structure.

Dictionary-normalized reference (MeCab + UniDic):
{mecab_reference}"""

        return f"""You are an expert in classical Japanese literature, specifically Edo period texts (1603-1868).

Convert the following classical Japanese text to modern Japanese (現代日本語).

Requirements:
- Preserve the original meaning and nuance precisely
- Convert classical grammar forms to modern equivalents
- Update archaic vocabulary to contemporary terms
- Maintain the tone and register of the original
- Note ambiguous passages

The text was produced by an OCR model and may contain recognition errors. Use context to infer the intended meaning where possible.

The text may contain inline annotations added by the OCR/preprocessing pipeline — these are not part of the original document:
- "[char:NN%]" marks a character the OCR model was uncertain about (NN% confidence). Use context to judge whether "char" is correct, or infer a more plausible reading; do not treat the brackets as literal text.
- "[furigana?:XY]" marks a run of kana between two kanji that may be a misdetected furigana gloss rather than main text. Use judgement: if it reads as a phonetic annotation for the adjacent kanji, disregard it; if it reads as legitimate grammatical text, incorporate it normally.
- "｜" marks a detected boundary between independent text blocks on the page (e.g., unrelated inscriptions, captions, or marginal notes). Treat text on either side as potentially unrelated in topic and context — do not assume narrative continuity across this marker. If it appears in your modern_japanese output, preserve it verbatim so the boundary carries through to the English translation step.
- A line break (newline) marks where the original page wraps to a new column. It carries no topic or content boundary — read across it as continuous text, the same as if it were not there.

Classical Japanese text:
{text}{reference_block}"""

    # ------------------------------------------------------------------
    # Step 2: modern Japanese → English
    # ------------------------------------------------------------------

    def translate_to_english(
        self, modern_japanese: str, classical_text: str = "", lang: str = "en"
    ) -> Tuple[Dict[str, str], Dict[str, int]]:
        """
        Translate modern Japanese to the target language (`lang`: "en" or "de").

        Returns (result, usage) where result has keys:
          english_translation, translation_notes, truncated
        (keys keep their "english_translation" name regardless of `lang` for
        pipeline compatibility; the value is in the requested target language)
        """
        language_name = _target_language_name(lang)
        prompt = self._build_translate_to_english_prompt(modern_japanese, classical_text, language_name)
        result, response = self._call_structured(
            prompt,
            tool_name="provide_english_translation",
            description=f"Provide the {language_name} translation of modern Japanese text.",
            properties={
                "english_translation": {
                    "type": "string",
                    "description": f"The {language_name} translation",
                },
                "translation_notes": {
                    "type": "string",
                    "description": "Cultural context or important decisions (empty string if none)",
                },
            },
            required=["english_translation", "translation_notes"],
            max_tokens=_estimate_max_tokens(modern_japanese),
        )
        return result, self._usage(response)

    # ------------------------------------------------------------------
    # Combined step: classical -> modern -> English in one request
    # (experimental, for comparison against the two-call pipeline above)
    # ------------------------------------------------------------------

    def translate_classical_to_english(
        self, text: str, mecab_reference: Optional[str] = None, lang: str = "en"
    ) -> Tuple[Dict[str, str], Dict[str, int]]:
        """
        Single-request variant of classical_to_modern() + translate_to_english():
        asks for both conversions in one tool call, with modern_japanese declared
        before english_translation in the schema so the model still produces the
        modern-Japanese conversion first, before using it to produce the
        target-language translation, within one continuous generation.

        `lang`: target language for the translation step ("en" or "de"); the
        modern_japanese conversion step is unaffected.

        Returns (result, usage) where result has keys:
          modern_japanese, notes, english_translation, translation_notes, truncated
        (keys keep their "english_translation" name regardless of `lang` for
        pipeline compatibility; the value is in the requested target language)
        """
        language_name = _target_language_name(lang)
        prompt = self._build_combined_prompt(text, mecab_reference, language_name)
        result, response = self._call_structured(
            prompt,
            tool_name="provide_modern_and_english",
            description=(
                f"Provide the modern Japanese conversion and the {language_name} translation "
                "of the classical Japanese text, in that order."
            ),
            properties={
                "modern_japanese": {
                    "type": "string",
                    "description": "The classical text converted to modern Japanese",
                },
                "notes": {
                    "type": "string",
                    "description": "Any ambiguities or important conversion decisions in the classical->modern step (empty string if none)",
                },
                "english_translation": {
                    "type": "string",
                    "description": f"The {language_name} translation of the modern Japanese text",
                },
                "translation_notes": {
                    "type": "string",
                    "description": "Cultural context or important decisions in the modern->English step (empty string if none)",
                },
            },
            required=["modern_japanese", "notes", "english_translation", "translation_notes"],
            # Budget must cover both generated texts, not just one -- double the
            # single-step estimate, still bounded by the same ceiling.
            max_tokens=min(TRANSLATION_MAX_TOKENS_CEILING, _estimate_max_tokens(text) * 2),
        )
        return result, self._usage(response)

    def _build_combined_prompt(
        self, text: str, mecab_reference: Optional[str] = None, language_name: str = "English"
    ) -> str:
        reference_block = ""
        if mecab_reference:
            reference_block = f"""

A dictionary-normalized reference version of the same text is provided below, produced by a
separate morphological analyzer (MeCab + UniDic) that substitutes each word's standardized
modern spelling. Character positions in this reference do NOT correspond 1:1 to the annotated
text above (words may have been merged, split, or rewritten during normalization), and it
contains none of the inline markers described above. Use it only as an aid for resolving
archaic vocabulary and spelling — the annotated text above remains the authoritative source for
character-level uncertainty and document structure.

Dictionary-normalized reference (MeCab + UniDic):
{mecab_reference}"""

        return f"""You are an expert in classical Japanese literature, specifically Edo period texts (1603-1868), and in translating historical Japanese into natural, fluent {language_name}.

Perform two sequential conversions on the classical Japanese text below:
1. Convert it to modern Japanese (現代日本語).
2. Translate that modern Japanese into natural, fluent {language_name}.

Requirements for step 1 (classical -> modern Japanese):
- Preserve the original meaning and nuance precisely
- Convert classical grammar forms to modern equivalents
- Update archaic vocabulary to contemporary terms
- Maintain the tone and register of the original
- Note ambiguous passages

Requirements for step 2 (modern Japanese -> {language_name}):
- Read naturally in {language_name}
- Preserve the meaning and nuance
- Maintain appropriate formality
- Note important cultural context where relevant

The text was produced by an OCR model and may contain recognition errors. Use context to infer the intended meaning where possible.

The text may contain inline annotations added by the OCR/preprocessing pipeline — these are not part of the original document:
- "[char:NN%]" marks a character the OCR model was uncertain about (NN% confidence). Use context to judge whether "char" is correct, or infer a more plausible reading; do not treat the brackets as literal text.
- "[furigana?:XY]" marks a run of kana between two kanji that may be a misdetected furigana gloss rather than main text. Use judgement: if it reads as a phonetic annotation for the adjacent kanji, disregard it; if it reads as legitimate grammatical text, incorporate it normally.
- "｜" marks a detected boundary between independent text blocks on the page (e.g., unrelated inscriptions, captions, or marginal notes). Treat text on either side as potentially unrelated in topic and context — do not assume narrative continuity across this marker. Preserve "｜" verbatim in your modern_japanese output (do not convert it to a line break or other formatting there); only in your english_translation output should it become a clear separation (e.g., a paragraph break) rather than being merged into one continuous narrative.
- A line break (newline) marks where the original page wraps to a new column — a different thing from "｜" above. It carries no topic or content boundary — read across it as continuous text, the same as if it were not there.

Classical Japanese text:
{text}{reference_block}"""

    def _build_translate_to_english_prompt(
        self, modern_japanese: str, classical_text: str = "", language_name: str = "English"
    ) -> str:
        context = (
            f"\n\nOriginal classical text for reference:\n{classical_text}"
            if classical_text
            else ""
        )
        return f"""Translate the following modern Japanese text to natural, fluent {language_name}.

The text is from an Edo-period Japanese document that has been converted to modern Japanese. Your translation should:
- Read naturally in {language_name}
- Preserve the meaning and nuance
- Maintain appropriate formality
- Note important cultural context where relevant

The text may contain "｜" markers inherited from the OCR pipeline, indicating a boundary between independent text blocks (unrelated inscriptions, captions, or marginal notes). Reflect this as a clear separation (e.g., a paragraph break) in the {language_name} translation rather than merging the blocks into one continuous narrative.

Modern Japanese text:
{modern_japanese}{context}"""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _call_structured(
        self,
        prompt: str,
        tool_name: str,
        description: str,
        properties: Dict[str, Dict[str, str]],
        required: List[str],
        max_tokens: int,
    ) -> Tuple[dict, Any]:
        """Call Claude with a forced tool call so the response is guaranteed
        schema-conformant JSON — no markdown-fence parsing, no risk of the
        model replying in free text. Also detects max_tokens truncation.
        """
        schema = {"type": "object", "properties": properties, "required": required}

        if self._backend == "openrouter":
            response = self.client.chat.completions.create(
                model=_OPENROUTER_MODEL,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
                tools=[{
                    "type": "function",
                    "function": {"name": tool_name, "description": description, "parameters": schema},
                }],
                tool_choice={"type": "function", "function": {"name": tool_name}},
            )
            call = response.choices[0].message.tool_calls[0]
            result = json.loads(call.function.arguments)
            truncated = response.choices[0].finish_reason == "length"
        else:
            response = self.client.messages.create(
                model=self.MODEL,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
                tools=[{"name": tool_name, "description": description, "input_schema": schema}],
                tool_choice={"type": "tool", "name": tool_name},
            )
            tool_block = next(b for b in response.content if b.type == "tool_use")
            result = tool_block.input
            truncated = response.stop_reason == "max_tokens"

        if truncated:
            logger.warning(
                "Claude response truncated by max_tokens=%d (backend=%s, tool=%s)",
                max_tokens, self._backend, tool_name,
            )
        result = dict(result)
        result["truncated"] = truncated
        return result, response

    def _usage(self, response) -> Dict[str, int]:
        if self._backend == "openrouter":
            inp = response.usage.prompt_tokens
            out = response.usage.completion_tokens
        else:
            inp = response.usage.input_tokens
            out = response.usage.output_tokens
        return {"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out}
