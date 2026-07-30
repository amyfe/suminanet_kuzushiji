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

    def classical_to_modern(self, text: str) -> Tuple[Dict[str, str], Dict[str, int]]:
        """
        Convert classical/Edo-period Japanese to modern Japanese.

        Returns (result, usage) where result has keys:
          modern_japanese, notes, truncated
        and usage has keys:
          input_tokens, output_tokens, total_tokens
        """
        prompt = self._build_classical_to_modern_prompt(text)
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

The text may contain inline annotations added by the OCR/preprocessing pipeline — these are not part of the original document:
- "[char:NN%]" marks a character the OCR model was uncertain about (NN% confidence). Use context to judge whether "char" is correct, or infer a more plausible reading; do not treat the brackets as literal text.
- "[furigana?:XY]" marks a run of kana between two kanji that may be a misdetected furigana gloss rather than main text. Use judgement: if it reads as a phonetic annotation for the adjacent kanji, disregard it; if it reads as legitimate grammatical text, incorporate it normally.
- "｜" marks a detected boundary between independent text blocks on the page (e.g., unrelated inscriptions, captions, or marginal notes). Treat text on either side as potentially unrelated in topic and context — do not assume narrative continuity across this marker. If it appears in your modern_japanese output, preserve it verbatim so the boundary carries through to the English translation step.

Classical Japanese text:
{text}"""

    # ------------------------------------------------------------------
    # Step 2: modern Japanese → English
    # ------------------------------------------------------------------

    def translate_to_english(
        self, modern_japanese: str, classical_text: str = ""
    ) -> Tuple[Dict[str, str], Dict[str, int]]:
        """
        Translate modern Japanese to English.

        Returns (result, usage) where result has keys:
          english_translation, translation_notes, truncated
        """
        prompt = self._build_translate_to_english_prompt(modern_japanese, classical_text)
        result, response = self._call_structured(
            prompt,
            tool_name="provide_english_translation",
            description="Provide the English translation of modern Japanese text.",
            properties={
                "english_translation": {
                    "type": "string",
                    "description": "The English translation",
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

The text may contain "｜" markers inherited from the OCR pipeline, indicating a boundary between independent text blocks (unrelated inscriptions, captions, or marginal notes). Reflect this as a clear separation (e.g., a paragraph break) in the English translation rather than merging the blocks into one continuous narrative.

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
