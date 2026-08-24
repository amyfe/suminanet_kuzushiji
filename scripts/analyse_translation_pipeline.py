"""Translation pipeline analysis for the master's thesis.

Reads the *_translation.json files already produced by
scripts/visualize_qualitative_examples.py and produces:
  1. cost_per_page.png                  — stacked input/output cost per page
  2. cost_vs_input_length.png           — cost vs. source-text length scatter
  3. translation_pipeline_summary.md/.csv — per-page cost/reliability table + 1,000-page extrapolation
  4. normalization_summary.md/.csv      — MeCab method/morpheme-change-rate per page
  5. normalization_change_rate.png      — MeCab change rate per page
  6. normalization_change_vs_cer.png    — MeCab change rate vs. transcription CER
  7. translation_qualitative_review.md  — structured substitute for BLEU/BERT (blocked: no reference translations exist)

Makes ZERO new billed API calls: reads existing translation JSON files, and
recomputes per-morpheme MeCab tokens locally (free, deterministic — not an
LLM call) since those aren't saved in the JSON.

Usage:
    python scripts/analyse_translation_pipeline.py
    python scripts/analyse_translation_pipeline.py --pages "results/qualitative_examples/*_translation.json"
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import TRANSLATION_PRICE_PER_1M_INPUT_USD, TRANSLATION_PRICE_PER_1M_OUTPUT_USD
from visualization.translation import (
    load_translation_records,
    plot_cost_per_page,
    plot_cost_vs_input_length,
    plot_normalization_change_rate,
    plot_normalization_change_vs_cer,
    recompute_normalization_tokens,
    write_normalization_summary_table,
    write_pipeline_cost_table,
    write_qualitative_review,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Translation pipeline analysis — all reports in one command")
    p.add_argument("--pages", default="results/qualitative_examples/*_translation.json",
                   help="Glob of *_translation.json files to analyze")
    p.add_argument("--out-dir", default="results/translation_analysis",
                   help="Directory to save output files")
    p.add_argument("--input-price-per-1m", type=float, default=TRANSLATION_PRICE_PER_1M_INPUT_USD,
                   help="USD per 1M input tokens (default: config.py TRANSLATION_PRICE_PER_1M_INPUT_USD)")
    p.add_argument("--output-price-per-1m", type=float, default=TRANSLATION_PRICE_PER_1M_OUTPUT_USD,
                   help="USD per 1M output tokens (default: config.py TRANSLATION_PRICE_PER_1M_OUTPUT_USD)")
    return p.parse_args()


def _print_summary(records: list[dict], input_price: float, output_price: float) -> None:
    n = len(records)
    costs = []
    for r in records:
        usage = r.get("usage") or {}
        costs.append(usage.get("input_tokens", 0) * input_price / 1e6
                      + usage.get("output_tokens", 0) * output_price / 1e6)
    total_cost = sum(costs)
    mean_cost = total_cost / max(1, n)
    min_cost, max_cost = (min(costs), max(costs)) if costs else (0.0, 0.0)
    n_truncated = sum(1 for r in records if r.get("translation_truncated"))
    method_counts = Counter(r.get("normalization_method", "?") for r in records)

    print("=" * 72)
    print(f"Translation pipeline analysis | n={n} pages")
    print("=" * 72)
    print(f"Total cost:  ${total_cost:.4f}")
    print(f"Mean cost:   ${mean_cost:.4f} / page")
    print(f"Extrapolated / 1,000 pages: ${mean_cost * 1000:.2f}  "
          f"(range ${min_cost * 1000:.2f}–${max_cost * 1000:.2f})")
    print(f"Truncation rate: {n_truncated}/{n} ({100 * n_truncated / max(1, n):.1f}%)")
    print("Normalization method distribution:")
    for method, count in method_counts.most_common():
        print(f"  {method}: {count}/{n}")
    print()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-request price override: the plot/table functions in
    # visualization/translation.py read TRANSLATION_PRICE_PER_1M_*_USD as
    # module globals, so overriding them here (before any of those
    # functions run) affects every downstream cost computation consistently.
    if args.input_price_per_1m != TRANSLATION_PRICE_PER_1M_INPUT_USD or \
            args.output_price_per_1m != TRANSLATION_PRICE_PER_1M_OUTPUT_USD:
        import visualization.translation as _t
        _t.TRANSLATION_PRICE_PER_1M_INPUT_USD = args.input_price_per_1m
        _t.TRANSLATION_PRICE_PER_1M_OUTPUT_USD = args.output_price_per_1m

    print(f"Loading translation records from {args.pages} ...")
    records = load_translation_records(pattern=args.pages)
    if not records:
        raise SystemExit(f"No usable *_translation.json files matched {args.pages}")

    print(f"Recomputing MeCab normalization tokens for {len(records)} pages "
          f"(local, free — no API calls) ...")
    records = recompute_normalization_tokens(records)

    _print_summary(records, args.input_price_per_1m, args.output_price_per_1m)

    written: list[Path] = []
    written.append(plot_cost_per_page(records, out_path=out_dir / "cost_per_page.png"))
    written.append(plot_cost_vs_input_length(records, out_path=out_dir / "cost_vs_input_length.png"))
    written.append(write_pipeline_cost_table(records, out_path=out_dir / "translation_pipeline_summary.md"))
    written.append(write_normalization_summary_table(records, out_path=out_dir / "normalization_summary.md"))
    written.append(plot_normalization_change_rate(records, out_path=out_dir / "normalization_change_rate.png"))
    written.append(plot_normalization_change_vs_cer(records, out_path=out_dir / "normalization_change_vs_cer.png"))
    written.append(write_qualitative_review(records, out_path=out_dir / "translation_qualitative_review.md"))

    print("Wrote:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
