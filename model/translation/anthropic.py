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
import os
from typing import Dict, Optional, Tuple

from typing import Any

import anthropic
from anthropic.types import TextBlock

_OPENROUTER_BASE = "https://openrouter.ai/api/v1"
_OPENROUTER_MODEL = "anthropic/claude-sonnet-4-6"


class ClaudeTranslator:
    """Wraps the two Claude translation steps with prompt building and usage tracking."""

    MODEL = "claude-sonnet-4-6"

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.client: Any
        or_key = os.environ.get("OPENROUTER_API_KEY")
        an_key = os.environ.get("ANTHROPIC_API_KEY")

        if or_key and not api_key:
            import openai
            self.client = openai.OpenAI(base_url=_OPENROUTER_BASE, api_key=or_key)
            self._backend = "openrouter"
        else:
            key = api_key or an_key
            if not key:
                raise ValueError(
                    "API key required. Set OPENROUTER_API_KEY or ANTHROPIC_API_KEY, "
                    "or pass api_key=."
                )
            self.client = anthropic.Anthropic(api_key=key)
            self._backend = "anthropic"

    # ------------------------------------------------------------------
    # Step 1: classical → modern Japanese
    # ------------------------------------------------------------------

    def classical_to_modern(self, text: str) -> Tuple[Dict[str, str], Dict[str, int]]:
        """
        Convert classical/Edo-period Japanese to modern Japanese.

        Returns (result, usage) where result has keys:
          modern_japanese, notes
        and usage has keys:
          input_tokens, output_tokens, total_tokens
        """
        prompt = self._build_classical_to_modern_prompt(text)
        response = self._call(prompt)
        result = self._parse_json(self._text(response))
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

Classical Japanese text:
{text}

Respond in JSON format:
{{
    "modern_japanese": "the converted modern Japanese text",
    "notes": "any ambiguities or important conversion decisions (empty string if none)"
}}"""

    # ------------------------------------------------------------------
    # Step 2: modern Japanese → English
    # ------------------------------------------------------------------

    def translate_to_english(
        self, modern_japanese: str, classical_text: str = ""
    ) -> Tuple[Dict[str, str], Dict[str, int]]:
        """
        Translate modern Japanese to English.

        Returns (result, usage) where result has keys:
          english_translation, translation_notes
        """
        prompt = self._build_translate_to_english_prompt(modern_japanese, classical_text)
        response = self._call(prompt)
        result = self._parse_json(self._text(response))
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

Modern Japanese text:
{modern_japanese}{context}

Respond in JSON format:
{{
    "english_translation": "the English translation",
    "translation_notes": "cultural context or important decisions (empty string if none)"
}}"""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _call(self, prompt: str):
        if self._backend == "openrouter":
            return self.client.chat.completions.create(
                model=_OPENROUTER_MODEL,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )
        return self.client.messages.create(
            model=self.MODEL,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )

    def _text(self, response) -> str:
        if self._backend == "openrouter":
            return response.choices[0].message.content
        for block in response.content:
            if isinstance(block, TextBlock):
                return block.text
        raise ValueError(f"No TextBlock in response: {response.content}")

    def _usage(self, response) -> Dict[str, int]:
        if self._backend == "openrouter":
            inp = response.usage.prompt_tokens
            out = response.usage.completion_tokens
        else:
            inp = response.usage.input_tokens
            out = response.usage.output_tokens
        return {"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out}

    @staticmethod
    def _parse_json(text: str) -> dict:
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0]
        return json.loads(text.strip())
