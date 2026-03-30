"""Helpers for rendering tokenized historical-character labels consistently."""

from __future__ import annotations

from typing import Iterable


def unicode_token_to_char(token: str) -> str:
    """Convert tokens like 'U+3042' into their Unicode character when valid."""
    if not isinstance(token, str):
        return str(token)

    t = token.strip()
    if len(t) > 2 and (t.startswith("U+") or t.startswith("u+")):
        try:
            codepoint = int(t[2:], 16)
            if 0 <= codepoint <= 0x10FFFF:
                return chr(codepoint)
        except ValueError:
            pass
    return t


def render_tokens(tokens: Iterable[str]) -> str:
    """Render an iterable of label tokens into display text."""
    return "".join(unicode_token_to_char(tok) for tok in tokens)
