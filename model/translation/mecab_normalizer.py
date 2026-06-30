"""
Historical Japanese normalizer following the ChaMame approach.
https://chamame.ninjal.ac.jp/about.html

Uses MeCab + UniDic for morphological analysis and normalized-form extraction.
Falls back to heuristic kana normalization if fugashi/unidic are not installed.

"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# ---------------------------------------------------------------------------
# Historical kana transformations
# ---------------------------------------------------------------------------

# Voiced repeater mark: previous mora + dakuten
_VOICE_MAP = str.maketrans(
    "かきくけこさしすせそたちつてとはひふへほ",
    "がぎぐげござじずぜぞだぢづでどばびぶべぼ",
)

_REPEATER_VOICED_RE = re.compile(r"([^\s])[ゞヾ]")
_REPEATER_PLAIN_RE  = re.compile(r"([^\s])[ゝヽ]")


def _expand_repeaters(text: str) -> str:
    """Expand ゝ/ヽ (plain) and ゞ/ヾ (voiced) iteration marks."""
    text = _REPEATER_VOICED_RE.sub(lambda m: m.group(1) + m.group(1).translate(_VOICE_MAP), text)
    text = _REPEATER_PLAIN_RE.sub(lambda m: m.group(1) * 2, text)
    return text


def apply_kana_normalization(text: str) -> str:
    """
    Mechanical historical-kana → modern-kana conversions.
    These are always correct regardless of context.
    """
    text = _expand_repeaters(text)
    # Obsolete hiragana
    text = text.replace("ゐ", "い").replace("ゑ", "え")
    # Obsolete katakana
    text = text.replace("ヰ", "イ").replace("ヱ", "エ")
    return text


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class NormalizationResult:
    original: str
    normalized: str
    tokens: List[dict] = field(default_factory=list)  # [{surface, pos, normalized}]
    method: str = "heuristic"
    notes: str = ""


# ---------------------------------------------------------------------------
# Normalizer
# ---------------------------------------------------------------------------

class HistoricalJapaneseNormalizer:
    """
    Normalize Edo-period Japanese text using MeCab + UniDic (ChaMame-style).

    MeCab's UniDic `orthBase` field gives the standardized modern spelling
    for each morpheme, which is the same principle ChaMame uses to convert
    pre-modern texts to a searchable/translatable form.

    If fugashi + unidic-lite are not installed, only the heuristic kana
    conversions (ゐ→い, ゑ→え, repeater marks, etc.) are applied.
    """

    def __init__(self) -> None:
        self._tagger = None
        self._method = "heuristic"
        self._try_load_mecab()

    def _try_load_mecab(self) -> None:
        try:
            import fugashi

            # 0. Bundled Edo dict — always try first
            _project_root = Path(__file__).resolve().parent.parent.parent
            _zip = _project_root / "assets" / "translation_dict" / "unidic-kinsei-edo-v202512.zip"
            _edo = _zip.with_suffix("")  # …/unidic-kinsei-edo-v202512/
            if _zip.exists() and not _edo.exists():
                import zipfile
                with zipfile.ZipFile(_zip) as zf:
                    zf.extractall(_edo)
            if _edo.exists():
                self._tagger = fugashi.Tagger(f'-d "{_edo}"')
                self._method = "mecab+unidic-kinsei-edo"
                return

            # 1. UNIDIC_EDO_DIR env var — custom/override Edo dict path
            edo_dir = os.environ.get("UNIDIC_EDO_DIR", "")
            if edo_dir:
                self._tagger = fugashi.Tagger(f'-d "{edo_dir}"')
                self._method = "mecab+unidic-kinsei-edo"
                return

            # 2. Full modern UniDic (pip install unidic && python -m unidic download)
            try:
                import unidic
                self._tagger = fugashi.Tagger(f'-d "{unidic.DICDIR}"')
                self._method = "mecab+unidic"
                return
            except Exception:
                pass

            # 3. unidic-lite (pip install unidic-lite, self-contained)
            import unidic_lite
            self._tagger = fugashi.Tagger(f'-d "{unidic_lite.DICDIR}"')
            self._method = "mecab+unidic-lite"
        except Exception:
            pass  # heuristic fallback

    @property
    def available(self) -> bool:
        return self._tagger is not None

    def normalize(self, text: str) -> NormalizationResult:
        if self._tagger is not None:
            return self._normalize_with_mecab(text)
        return self._normalize_heuristic(text)

    # ------------------------------------------------------------------
    # MeCab path
    # ------------------------------------------------------------------

    def _normalize_with_mecab(self, text: str) -> NormalizationResult:
        tokens: List[dict] = []
        parts: List[str] = []

        for word in self._tagger(text):
            surface = word.surface
            pos = self._extract_pos(word)
            # orthBase returns the dictionary/lemma form for verbs, which breaks conjugations
            # (e.g. 登り → 登る produces 登るけり). Keep surface for verbs to preserve tense.
            if pos == "動詞":
                normalized_form = surface
            else:
                normalized_form = self._extract_orth_base(word) or surface
            tokens.append({"surface": surface, "normalized": normalized_form, "pos": pos})
            parts.append(normalized_form)

        normalized = apply_kana_normalization("".join(parts))
        return NormalizationResult(
            original=text,
            normalized=normalized,
            tokens=tokens,
            method=self._method,
            notes=f"{len(tokens)} morphemes via MeCab+UniDic.",
        )

    @staticmethod
    def _extract_orth_base(word) -> str | None:
        """Return orthBase (standardized spelling) from UniDic feature, or None."""
        try:
            ob = word.feature.orthBase
            return ob if ob and ob != "*" else None
        except AttributeError:
            return None

    @staticmethod
    def _extract_pos(word) -> str:
        try:
            p = word.feature.pos1
            return "" if p == "*" else str(p)
        except AttributeError:
            return ""

    # ------------------------------------------------------------------
    # Heuristic path
    # ------------------------------------------------------------------

    def _normalize_heuristic(self, text: str) -> NormalizationResult:
        normalized = apply_kana_normalization(text)
        return NormalizationResult(
            original=text,
            normalized=normalized,
            tokens=[],
            method="heuristic",
            notes="fugashi/unidic not installed; applied kana-only normalization.",
        )
