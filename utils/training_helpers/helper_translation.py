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
import logging
import math
import re
import statistics
from typing import Dict, List

from config import FURIGANA_MIN_RUN_LENGTH
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pre-processing helpers
# ---------------------------------------------------------------------------

_HIRAGANA = "ぁ-ゖ"
_KATAKANA = "ァ-ヶ"

_FURIGANA_RUN_RE = re.compile(
    r"(?<=[一-鿿㐀-䶿])([" + _HIRAGANA + _KATAKANA + r"]{" + str(FURIGANA_MIN_RUN_LENGTH) + r",})(?=[一-鿿㐀-䶿])"
)



def _mark(m: "re.Match[str]") -> str:
    run = m.group(1)
    logger.info("Possible furigana run marked: %r at offset %d", run, m.start(1))
    return f"[furigana?:{run}]"



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


def _build_llm_text(
    text: str,
    chars: List[Dict],
    uncertain_threshold: float,
    block_gap_factor: float,
    *,
    mark_furigana: bool = True,
) -> str:
    """Build the LLM-facing text: insert "｜" at detected text-block
    boundaries, a plain "\\n" at column-only boundaries (no content change,
    just a layout wrap), wrap OCR-uncertain characters as "[char:NN%]", and
    -- when `mark_furigana` -- wrap runs of kana sandwiched between kanji as
    "[furigana?:...]", using the same heuristic and marker syntax as
    _strip_furigana_heuristic (see its docstring for rationale).

    `chars` must be positionally 1:1 with `text` (guaranteed by
    run_inference()'s chars_out). All four transforms are combined into a
    single pass since inserting any of them would break the index alignment
    the others need. This matters especially for furigana: spans are found
    by running _FURIGANA_RUN_RE against the raw, unmodified `text` BEFORE
    any "[char:NN%]" markers are spliced in, so a low-confidence character in
    the middle of a furigana run no longer breaks the kana-run adjacency the
    regex requires (previously furigana marking ran as a second pass over
    already-annotated text, where this was silently missed).

    A block/column break landing inside a furigana span is not specially
    handled: the break marker is simply emitted inside the
    "[furigana?:...]" wrapping. This is common in practice, not an edge
    case -- on kana-dense classical prose, flagged runs (any 2+ kana between
    kanji, not just genuine furigana glosses) are frequently long enough to
    cross a column wrap (observed ~1 in 4 spans on a real multi-column
    sample). It's an accepted limitation, matching the rigor level of the
    geometry heuristics above: Claude is told to use judgement on each
    "[furigana?:...]" span regardless, so an embedded "\\n" inside one
    doesn't change how it should be interpreted.
    """
    column_breaks = _detect_column_breaks(chars)
    block_breaks = _detect_block_breaks(chars, block_gap_factor)

    furigana_spans: List[tuple] = []
    if mark_furigana:
        for m in _FURIGANA_RUN_RE.finditer(text):
            logger.info("Possible furigana run marked: %r at offset %d", m.group(1), m.start(1))
            furigana_spans.append((m.start(1), m.end(1)))

    out = []
    span_idx = 0
    in_span = False
    for i, (ch, c) in enumerate(zip(text, chars)):
        if i in block_breaks:
            out.append("｜")
        elif i in column_breaks:
            out.append("\n")

        if not in_span and span_idx < len(furigana_spans) and furigana_spans[span_idx][0] == i:
            out.append("[furigana?:")
            in_span = True

        score = c.get("score", 1.0)
        if score < uncertain_threshold:
            out.append(f"[{ch}:{score * 100:.0f}%]")
        else:
            out.append(ch)

        if in_span and i == furigana_spans[span_idx][1] - 1:
            out.append("]")
            in_span = False
            span_idx += 1

    return "".join(out)


def _preprocess(text: str, strip_furigana: bool = True) -> str:
    if strip_furigana:
        text = _FURIGANA_RUN_RE.sub(_mark, text)
    return text.replace("�", "").replace("\x00", "").strip()


def _sum_usage(*usages: Dict[str, int]) -> Dict[str, int]:
    inp = sum(u["input_tokens"] for u in usages)
    out = sum(u["output_tokens"] for u in usages)
    return {"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out}

