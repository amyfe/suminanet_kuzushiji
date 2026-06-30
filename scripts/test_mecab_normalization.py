"""
Inspect MeCab + UniDic normalization output on a sample kuzushiji transcription.

Run from project root:
    python scripts/test_mecab_normalization.py

Add --translate to also run the full Claude pipeline (requires OPENROUTER_API_KEY in .env):
    python scripts/test_mecab_normalization.py --translate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from model.translation.mecab_normalizer import HistoricalJapaneseNormalizer

SAMPLE = (
    "く庖ウ序つは山くんゞ伝醯しを焼庖ふをひつはらのをつみ百りひさし味あけよにさ"
    "孰能もふに血作何ち梅湯之かと第はる集えしりぬん噌おこおべ己るねをれ右しかのつ"
    "同塩きじ中と粉なかこじなるずでのどをら"
)

SAMPLE_SHORT = "かくて年月を経て、遂に都へ上りけり。"
SAMPLE_2 = "春の朝、川辺を歩みければ、桜の花いと美しく咲きたり。行き交ふ人々も、その景色を見て心和みぬ。"
def print_normalization(text: str) -> None:
    normalizer = HistoricalJapaneseNormalizer()
    print(f"Backend: {normalizer._method}\n")

    result = normalizer.normalize(text)

    print(f"Input     : {text}")
    print(f"Normalized: {result.normalized}")
    print(f"Notes     : {result.notes}\n")

    if result.tokens:
        col = 14
        header = f"{'surface':<{col}} {'normalized':<{col}} pos"
        print(header)
        print("-" * len(header))
        changed = [(t["surface"], t["normalized"], t["pos"]) for t in result.tokens]
        for surface, normalized, pos in changed:
            marker = " *" if surface != normalized else ""
            print(f"{surface:<{col}} {normalized:<{col}} {pos}{marker}")
        n_changed = sum(1 for s, n, _ in changed if s != n)
        print(f"\n{n_changed}/{len(result.tokens)} morphemes had their form normalized.")
    else:
        print("(no token-level data — heuristic mode)")


def print_translation(text: str) -> None:
    from model.translation.translation import EdoPeriodTranslationPipeline

    print("\n" + "=" * 60)
    print("Running full translation pipeline…")
    pipeline = EdoPeriodTranslationPipeline()
    result = pipeline.translate_text(text)

    print(f"\nClassical  : {result['classical_japanese']}")
    print(f"Normalized : {result['normalized_japanese']}  ({result['normalization_method']})")
    print(f"Modern JP  : {result['modern_japanese']}")
    if result["conversion_notes"]:
        print(f"  notes    : {result['conversion_notes']}")
    print(f"English    : {result['english_translation']}")
    if result["translation_notes"]:
        print(f"  notes    : {result['translation_notes']}")
    u = result["usage"]
    print(f"\nTokens used: {u['input_tokens']} in + {u['output_tokens']} out = {u['total_tokens']} total")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default=SAMPLE, help="Custom input text")
    parser.add_argument("--translate", action="store_true", help="Also run the Claude pipeline")
    args = parser.parse_args()

    print_normalization(args.text)

    if args.translate:
        print_translation(args.text)
