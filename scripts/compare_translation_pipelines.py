"""Compare the current two-call translation pipeline (classical->modern, modern->English)
against the single-call combined variant (ClaudeTranslator.translate_classical_to_english),
on the same input text.

Reports, for each version: wall-clock latency, token usage (input/output/total), and the
produced modern_japanese / english_translation, so output quality can be eyeballed side by
side alongside the cost/latency numbers.

Usage:
  python scripts/compare_translation_pipelines.py --dry-run   # build prompts only, no API calls
  python scripts/compare_translation_pipelines.py             # real run, makes billed API calls
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

import torch

from config import SUMINANET_CONTEXT_BLOCK_GAP_FACTOR, TRANSLATION_UNCERTAIN_SCORE_THRESH
from model.suminanet.roi.roi_ordering import ROIReadingOrder
from model.translation.anthropic import ClaudeTranslator
from model.translation.translation import (
    EdoPeriodTranslationPipeline
)
from utils.training_helpers.helper_translation import _build_llm_text, _preprocess, _sum_usage   

from utils.text_normalization import unicode_token_to_char

SAMPLE_ANNOTATION = ROOT / "assets" / "data" / "annotations" / "200021071_00028_1.json"


def load_sample_chars(ann_path: Path) -> list[dict]:
    """Build a `chars` list (char/box/score/col_id) from a ground-truth annotation
    file, standing in for real OCR output for the purposes of this comparison
    (scores are set to 1.0 since ground truth has no notion of OCR confidence).

    col_id is computed via the real ROIReadingOrder column-clustering logic (the
    same code path run_inference() uses), not hand-derived, so this exercises the
    actual column-vs-block detection logic rather than always hitting its
    no-col_id fallback path.
    """
    ann = json.loads(ann_path.read_text(encoding="utf-8"))
    boxes = ann["boxes"]
    labels = ann["labels"]

    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    mask = torch.ones((boxes_tensor.size(0),), dtype=torch.bool)
    _, _, sort_idx, col_ids_sorted = ROIReadingOrder().sort_single(boxes_tensor, mask, "vertical")
    # sort_single reorders into reading order; invert the permutation to get
    # col_id back in the ORIGINAL (already reading-order) annotation order,
    # since the rest of this script assumes ann["boxes"]/ann["labels"] order.
    col_ids_orig = torch.empty_like(col_ids_sorted)
    col_ids_orig[sort_idx] = col_ids_sorted

    chars = []
    for label, box, col_id in zip(labels, boxes, col_ids_orig.tolist()):
        chars.append({
            "char": unicode_token_to_char(label),
            "box": box,
            "score": 1.0,
            "col_id": col_id,
        })
    return chars


def run_two_call(pipeline: EdoPeriodTranslationPipeline, classical_text: str, chars: list[dict]) -> dict:
    t0 = time.perf_counter()
    result = pipeline.translate_text(classical_text, chars=chars)
    elapsed = time.perf_counter() - t0
    return {
        "elapsed_sec": elapsed,
        "classical_text": classical_text,
        "modern_japanese": result["modern_japanese"],
        "english_translation": result["english_translation"],
        "usage": result["usage"],
    }


def run_combined(translator: ClaudeTranslator, classical_text: str, chars: list[dict]) -> dict:
    annotated = _build_llm_text(
        classical_text, chars, TRANSLATION_UNCERTAIN_SCORE_THRESH, SUMINANET_CONTEXT_BLOCK_GAP_FACTOR,
        mark_furigana=True,
    )
    text_for_llm = _preprocess(annotated, strip_furigana=False)

    t0 = time.perf_counter()
    result, usage = translator.translate_classical_to_english(text_for_llm)
    elapsed = time.perf_counter() - t0
    return {
        "elapsed_sec": elapsed,
        "classical_text": classical_text,
        "modern_japanese": result["modern_japanese"],
        "english_translation": result["english_translation"],
        "usage": usage,
    }


def print_dry_run(classical_text: str, chars: list[dict]) -> None:
    translator = object.__new__(ClaudeTranslator)  # skip __init__, no API key needed for prompt building

    annotated = _build_llm_text(
        classical_text, chars, TRANSLATION_UNCERTAIN_SCORE_THRESH, SUMINANET_CONTEXT_BLOCK_GAP_FACTOR,
        mark_furigana=True,
    )
    text_for_llm = _preprocess(annotated, strip_furigana=False)

    two_call_prompt_1 = translator._build_classical_to_modern_prompt(text_for_llm)
    two_call_prompt_2_placeholder = translator._build_translate_to_english_prompt(
        "<modern_japanese output from call 1>", classical_text
    )
    combined_prompt = translator._build_combined_prompt(text_for_llm)

    print("=" * 70)
    print(f"Sample: {len(chars)} chars, classical_text={classical_text!r}")
    print("=" * 70)
    print(f"\n--- TWO-CALL: prompt 1 (classical->modern), {len(two_call_prompt_1)} chars ---")
    print(two_call_prompt_1)
    print(f"\n--- TWO-CALL: prompt 2 (modern->English) shape, {len(two_call_prompt_2_placeholder)} chars ---")
    print(two_call_prompt_2_placeholder)
    print(f"\n--- COMBINED: single prompt, {len(combined_prompt)} chars ---")
    print(combined_prompt)
    print("\n" + "=" * 70)
    dup_chars = len(text_for_llm)  # what gets sent twice today (once as primary, once as context)
    print(f"Text resent as step-2 context today (duplicate transmission): ~{dup_chars} chars")
    print("=" * 70)


def print_run(label: str, run: dict) -> None:
    print("\n" + "=" * 70)
    print(label)
    print("=" * 70)
    print(f"Latency:   {run['elapsed_sec']:.2f}s")
    print(f"Usage:     {run['usage']}")
    print(f"Original Text:\n  {run['classical_text']}")
    print(f"Modern Japanese:\n  {run['modern_japanese']}")
    print(f"English:\n  {run['english_translation']}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="Build prompts only, no API calls")
    p.add_argument("--annotation", type=Path, default=SAMPLE_ANNOTATION)
    p.add_argument(
        "--repeats", type=int, default=1,
        help="Run each variant this many times, to separate architecture effects from "
             "ordinary LLM sampling variance (each repeat is a fresh, independently billed call).",
    )
    args = p.parse_args()

    chars = load_sample_chars(args.annotation)
    classical_text = "".join(c["char"] for c in chars)

    if args.dry_run:
        print_dry_run(classical_text, chars)
        return

    pipeline = EdoPeriodTranslationPipeline()
    translator = ClaudeTranslator()

    two_call_runs, combined_runs = [], []
    for i in range(args.repeats):
        print(f"Running two-call pipeline (rep {i + 1}/{args.repeats})...")
        two_call_runs.append(run_two_call(pipeline, classical_text, chars))
        print(f"Running combined pipeline (rep {i + 1}/{args.repeats})...")
        combined_runs.append(run_combined(translator, classical_text, chars))

    for i, (two_call, combined) in enumerate(zip(two_call_runs, combined_runs)):
        if args.repeats > 1:
            print(f"\n\n########## REP {i + 1}/{args.repeats} ##########")
        print_run("TWO-CALL", two_call)
        print_run("COMBINED", combined)

        print("\n" + "=" * 70)
        print("DELTA")
        print("=" * 70)
        lat_delta = two_call["elapsed_sec"] - combined["elapsed_sec"]
        tok_delta = two_call["usage"]["total_tokens"] - combined["usage"]["total_tokens"]
        print(f"Latency:      combined is {lat_delta:+.2f}s vs two-call ({'faster' if lat_delta > 0 else 'slower'})")
        print(f"Total tokens: combined uses {tok_delta:+d} vs two-call ({'fewer' if tok_delta > 0 else 'more'})")

    if args.repeats > 1:
        print("\n" + "=" * 70)
        print("ACROSS-REPEAT SUMMARY")
        print("=" * 70)
        print(
            "Compare modern_japanese across reps of the SAME variant above to gauge how much "
            "of any two-call-vs-combined difference is sampling noise rather than architecture."
        )


if __name__ == "__main__":
    main()
