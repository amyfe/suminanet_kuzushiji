"""Translation pipeline analysis: cost/reliability, MeCab normalization
attribution, and qualitative/error review.

Consumes the *_translation.json files produced by
scripts/visualize_qualitative_examples.py (real EdoPeriodTranslationPipeline
output on real pages, including gt_text/pred_text/cer appended by that
script). Nothing in this module makes a billed API call --
recompute_normalization_tokens() re-runs only the local, free MeCab step
(per-morpheme tokens aren't saved in the JSON, only translate_text()'s
summary fields are).
"""

from __future__ import annotations

import csv
import glob
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import TRANSLATION_PRICE_PER_1M_INPUT_USD, TRANSLATION_PRICE_PER_1M_OUTPUT_USD
from visualization.common import BAR_ALPHA, GRID_ALPHA, savefig

DEFAULT_CATEGORIES = ["clean_success", "failure_case", "dense_columns", "illustration_heavy"]

_MORPHEME_COUNT_RE = re.compile(r"(\d+)\s*morphemes")

_UNCERTAINTY_KEYWORDS = ["furigana", "OCR confidence", "truncat", "unclear", "ambiguous", "low confidence"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_translation_records(
    pattern: str = "results/qualitative_examples/*_translation.json",
    known_categories: "list[str] | None" = None,
) -> list[dict]:
    """Load every *_translation.json matching `pattern`, tagging each with
    _category/_stem/_page_id parsed from the filename against
    known_categories (falls back to "uncategorized" rather than raising, so
    a category added later to visualize_qualitative_examples.py doesn't
    break this loader). Records with an "error" key (a failed translation
    call -- see that script's exception-handling branch) are skipped with a
    printed warning, not raised."""
    known_categories = known_categories or DEFAULT_CATEGORIES
    records: list[dict] = []
    for path in sorted(glob.glob(pattern)):
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        if "error" in data:
            print(f"[warn] skipping {p.name}: translation call failed ({data['error']})")
            continue
        stem = p.stem
        if stem.endswith("_translation"):
            stem = stem[: -len("_translation")]
        category = next((c for c in known_categories if stem.startswith(c + "_")), "uncategorized")
        page_id = stem[len(category) + 1:] if category != "uncategorized" else stem
        data["_category"] = category
        data["_stem"] = page_id
        data["_page_id"] = stem
        data["_path"] = p
        records.append(data)
    records.sort(key=lambda r: (r["_category"], r["_stem"]))
    return records


def recompute_normalization_tokens(records: list[dict]) -> list[dict]:
    """Attach per-morpheme {surface, normalized, pos} tokens to each record
    by re-running the local, free MeCab normalization step (one shared
    pipeline instance reused across all records -- dictionary loading is
    expensive to repeat per-record). Uses HistoricalJapaneseNormalizer
    directly (not EdoPeriodTranslationPipeline.normalize(), which would
    construct a ClaudeTranslator and require an API key for no reason --
    the normalizer itself needs none). Warns (does not raise) if the
    recomputed method/morpheme-count disagrees with what translate_text()
    originally stored, which would indicate environment drift since the
    page was translated (e.g. the bundled Edo-period dictionary not
    extracted in whatever environment reruns this later). Records whose
    stored method is "heuristic" get _tokens = [] untouched -- that
    fallback path never produces tokens, so recomputing is meaningless."""
    from model.translation.mecab_normalizer import HistoricalJapaneseNormalizer

    normalizer = HistoricalJapaneseNormalizer()
    for r in records:
        if r.get("normalization_method") == "heuristic":
            r["_tokens"] = []
            continue
        result = normalizer.normalize(r["classical_japanese"])
        r["_tokens"] = result.tokens
        if result.method != r["normalization_method"]:
            print(f"[warn] {r['_page_id']}: normalization method drifted "
                  f"({r['normalization_method']!r} at translation time -> "
                  f"{result.method!r} now)")
        stored_m = _MORPHEME_COUNT_RE.search(r.get("normalization_notes", ""))
        if stored_m and int(stored_m.group(1)) != len(result.tokens):
            print(f"[warn] {r['_page_id']}: morpheme count drifted "
                  f"({stored_m.group(1)} at translation time -> {len(result.tokens)} now)")
    return records


# ---------------------------------------------------------------------------
# Angle 1 -- Cost & reliability
# ---------------------------------------------------------------------------

def _page_cost_usd(record: dict) -> float:
    usage = record.get("usage") or {}
    in_cost = usage.get("input_tokens", 0) * TRANSLATION_PRICE_PER_1M_INPUT_USD / 1e6
    out_cost = usage.get("output_tokens", 0) * TRANSLATION_PRICE_PER_1M_OUTPUT_USD / 1e6
    return in_cost + out_cost


def plot_cost_per_page(records: list[dict], out_path: "str | Path" = "cost_per_page.png") -> Path:
    """Stacked bar (input-cost + output-cost) per page. A named-bar chart
    is the right call at this scale -- every page is individually visible.
    Switch to a histogram (see plot_per_page_cer in visualization/stage2.py)
    once n grows past ~30 pages; this doesn't do that automatically."""
    labels = [f"{r['_category']}\n{r['_stem']}" for r in records]
    usages = [r.get("usage") or {} for r in records]
    in_costs = np.array([u.get("input_tokens", 0) * TRANSLATION_PRICE_PER_1M_INPUT_USD / 1e6 for u in usages])
    out_costs = np.array([u.get("output_tokens", 0) * TRANSLATION_PRICE_PER_1M_OUTPUT_USD / 1e6 for u in usages])

    fig, ax = plt.subplots(figsize=(max(9, 1.7 * len(records)), 5))
    x = np.arange(len(records))
    ax.bar(x, in_costs, label="Input cost", color="steelblue", alpha=BAR_ALPHA)
    ax.bar(x, out_costs, bottom=in_costs, label="Output cost", color="darkorange", alpha=BAR_ALPHA)
    for i, total in enumerate(in_costs + out_costs):
        ax.text(i, total, f"${total:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Cost (USD)")
    ax.set_title(f"Translation cost per page (n={len(records)})")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=GRID_ALPHA)
    fig.tight_layout()
    return savefig(fig, out_path)


def plot_cost_vs_input_length(records: list[dict], out_path: "str | Path" = "cost_vs_input_length.png") -> Path:
    """Scatter: source-text length vs. USD cost, colored by category --
    a descriptive sanity check that cost scales with input length, not a
    fitted model. n is small (stated in the title) so treat as illustrative."""
    categories = sorted(set(r["_category"] for r in records))
    color_map = {c: plt.cm.tab10(i % 10) for i, c in enumerate(categories)}

    fig, ax = plt.subplots(figsize=(7, 5))
    for c in categories:
        pts = [r for r in records if r["_category"] == c]
        xs = [len(r["classical_japanese"]) for r in pts]
        ys = [_page_cost_usd(r) for r in pts]
        ax.scatter(xs, ys, label=c, color=color_map[c], alpha=0.85, s=50)
    ax.set_xlabel("Source text length (chars)")
    ax.set_ylabel("Cost (USD)")
    ax.set_title(f"Cost vs. input length (n={len(records)} -- descriptive only, not a fitted trend)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=GRID_ALPHA)
    fig.tight_layout()
    return savefig(fig, out_path)


def write_pipeline_cost_table(
    records: list[dict], out_path: "str | Path" = "translation_pipeline_summary.md"
) -> Path:
    """Per-page cost/reliability table + TOTAL/MEAN rows + a 1,000-page
    extrapolation range (mean/min/max-page-based -- a range, not a single
    point estimate, since n is too small to support tighter precision)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for r in records:
        usage = r.get("usage") or {}
        rows.append({
            "page": r["_page_id"], "category": r["_category"],
            "input_tok": usage.get("input_tokens", 0), "output_tok": usage.get("output_tokens", 0),
            "cost_usd": _page_cost_usd(r), "truncated": bool(r.get("translation_truncated", False)),
            "cer": r.get("cer"),
        })

    n = len(rows)
    costs = [row["cost_usd"] for row in rows]
    total_cost = sum(costs)
    mean_cost = total_cost / max(1, n)
    min_cost, max_cost = (min(costs), max(costs)) if costs else (0.0, 0.0)
    n_truncated = sum(1 for row in rows if row["truncated"])

    lines = [
        "| page | category | input_tok | output_tok | cost_usd | truncated | cer |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        cer_s = f"{row['cer']:.4f}" if row["cer"] is not None else "—"
        lines.append(
            f"| {row['page']} | {row['category']} | {row['input_tok']} | {row['output_tok']} | "
            f"${row['cost_usd']:.4f} | {row['truncated']} | {cer_s} |"
        )
    lines += [
        f"| **TOTAL** | | | | **${total_cost:.4f}** | {n_truncated}/{n} | |",
        f"| **MEAN** | | | | **${mean_cost:.4f}** | | |",
        "",
        f"Extrapolated / 1,000 pages (mean-based): **${mean_cost * 1000:.2f}**",
        f"Extrapolated / 1,000 pages (min-page-based, lower bound): ${min_cost * 1000:.2f}",
        f"Extrapolated / 1,000 pages (max-page-based, upper bound): ${max_cost * 1000:.2f}",
        "",
        f"Truncation rate: {n_truncated}/{n} ({100 * n_truncated / max(1, n):.1f}%)",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    csv_path = out_path.with_suffix(".csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["page", "category", "input_tok", "output_tok", "cost_usd", "truncated", "cer"])
        for row in rows:
            writer.writerow([row["page"], row["category"], row["input_tok"], row["output_tok"],
                              f"{row['cost_usd']:.6f}", row["truncated"], row["cer"]])

    return out_path


# ---------------------------------------------------------------------------
# Angle 2 -- MeCab normalization attribution
# ---------------------------------------------------------------------------

def _morpheme_change_stats(tokens: list[dict]) -> tuple[int, int]:
    """(total, changed) -- "changed" = surface != normalized, the same
    comparison scripts/test_mecab_normalization.py uses inline."""
    total = len(tokens)
    changed = sum(1 for t in tokens if t.get("surface") != t.get("normalized"))
    return total, changed


def write_normalization_summary_table(
    records: list[dict], out_path: "str | Path" = "normalization_summary.md"
) -> Path:
    """Requires recompute_normalization_tokens() to have populated
    record["_tokens"] first."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for r in records:
        total, changed = _morpheme_change_stats(r.get("_tokens", []))
        rate = changed / total if total else None
        rows.append({
            "page": r["_page_id"], "method": r.get("normalization_method", "?"),
            "morphemes_total": total, "morphemes_changed": changed,
            "change_rate": rate, "cer": r.get("cer"),
        })

    method_counts = Counter(row["method"] for row in rows)
    heuristic_pages = [row["page"] for row in rows if row["method"] == "heuristic"]

    lines = [
        "| page | method | morphemes_total | morphemes_changed | change_rate | cer |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        rate_s = f"{row['change_rate']:.1%}" if row["change_rate"] is not None else "—"
        cer_s = f"{row['cer']:.4f}" if row["cer"] is not None else "—"
        lines.append(
            f"| {row['page']} | {row['method']} | {row['morphemes_total']} | "
            f"{row['morphemes_changed']} | {rate_s} | {cer_s} |"
        )

    lines += ["", "**Method distribution:**", ""]
    for method, count in method_counts.most_common():
        lines.append(f"- {method}: {count}/{len(rows)}")

    lines += ["", "**Heuristic fallback (no morpheme attribution possible):**", ""]
    lines.append(
        ", ".join(heuristic_pages) if heuristic_pages else "None observed in this sample."
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    csv_path = out_path.with_suffix(".csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["page", "method", "morphemes_total", "morphemes_changed", "change_rate", "cer"])
        for row in rows:
            writer.writerow([row["page"], row["method"], row["morphemes_total"],
                              row["morphemes_changed"], row["change_rate"], row["cer"]])

    return out_path


def plot_normalization_change_rate(
    records: list[dict], out_path: "str | Path" = "normalization_change_rate.png"
) -> Path:
    """Bar per page (only pages with recomputed tokens available -- i.e.
    method != heuristic), % of morphemes MeCab actually changed from
    surface form, annotated with the morpheme count."""
    pts = [r for r in records if r.get("_tokens")]
    if not pts:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No pages with recomputed tokens (all heuristic fallback)",
                ha="center", va="center", transform=ax.transAxes)
        return savefig(fig, out_path)

    labels = [f"{r['_category']}\n{r['_stem']}" for r in pts]
    rates, totals = [], []
    for r in pts:
        total, changed = _morpheme_change_stats(r["_tokens"])
        rates.append(100 * changed / total if total else 0.0)
        totals.append(total)

    fig, ax = plt.subplots(figsize=(max(9, 1.7 * len(pts)), 5))
    x = np.arange(len(pts))
    ax.bar(x, rates, color="mediumseagreen", alpha=BAR_ALPHA)
    for i, (rate, total) in enumerate(zip(rates, totals)):
        ax.text(i, rate, f"{rate:.1f}%\n(n={total})", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Morphemes changed (%)")
    ax.set_title(f"MeCab normalization change rate per page (n={len(pts)})")
    ax.grid(True, axis="y", alpha=GRID_ALPHA)
    fig.tight_layout()
    return savefig(fig, out_path)


def plot_normalization_change_vs_cer(
    records: list[dict], out_path: "str | Path" = "normalization_change_vs_cer.png"
) -> Path:
    """Scatter: MeCab change-rate vs. transcription CER (already computed
    upstream, no recomputation needed). Explicitly not a claimed
    correlation -- see the title, which is written to survive even if this
    PNG is later pulled out of context (e.g. into a thesis draft)."""
    pts = [r for r in records if r.get("_tokens") and r.get("cer") is not None]
    fig, ax = plt.subplots(figsize=(7, 5))
    if not pts:
        ax.text(0.5, 0.5, "No pages with both recomputed tokens and cer",
                ha="center", va="center", transform=ax.transAxes)
        return savefig(fig, out_path)

    xs, ys = [], []
    for r in pts:
        total, changed = _morpheme_change_stats(r["_tokens"])
        xs.append(100 * changed / total if total else 0.0)
        ys.append(r["cer"])
    ax.scatter(xs, ys, color="mediumseagreen", alpha=0.85, s=50)
    ax.set_xlabel("Morphemes changed (%)")
    ax.set_ylabel("Transcription CER")
    ax.set_title(
        f"Normalization change rate vs. CER (n={len(pts)} -- descriptive only, "
        "not a statistically meaningful correlation)"
    )
    ax.grid(True, alpha=GRID_ALPHA)
    fig.tight_layout()
    return savefig(fig, out_path)


# ---------------------------------------------------------------------------
# Angle 3 -- Qualitative / error review
# ---------------------------------------------------------------------------

def _keyword_hits(records: list[dict]) -> dict[str, list[str]]:
    hits: dict[str, list[str]] = {kw: [] for kw in _UNCERTAINTY_KEYWORDS}
    for r in records:
        text = (r.get("conversion_notes") or "") + " " + (r.get("translation_notes") or "")
        text_low = text.lower()
        for kw in _UNCERTAINTY_KEYWORDS:
            if kw.lower() in text_low:
                hits[kw].append(r["_page_id"])
    return hits


def write_qualitative_review(
    records: list[dict], out_path: "str | Path" = "translation_qualitative_review.md"
) -> Path:
    """One combined markdown file -- at this n, a single linearly-readable
    file serves a scanning reader better than opening N separate files.
    Revisit as one-file-per-page (or paginated) once n grows large enough
    that linear reading stops being practical; not built now."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(records)

    lines = [
        "# Translation pipeline qualitative review",
        "",
        f"n={n} pages, generated from `results/qualitative_examples/*_translation.json`.",
        "",
        "BLEU/BERT scoring against a reference translation is not possible for this "
        "project -- zero reference English translations exist anywhere in the repo "
        "(see pipeline_analysis.txt 7.5.3). This review is a structured substitute: "
        "it surfaces the LLM's own uncertainty notes and truncation flags instead of "
        "a scored metric.",
        "",
        "## Truncated translations",
        "",
    ]
    truncated = [r for r in records if r.get("translation_truncated")]
    n_truncated = len(truncated)
    lines.append(f"{n_truncated}/{n} pages truncated.")
    lines.append("")
    for r in truncated:
        lines.append(f"### {r['_page_id']}")
        lines.append(f"translation_notes: {r.get('translation_notes', '')}")
        lines.append("")
    lines += [
        "",
        "(Note: `translation.py`'s module docstring mentions a `conversion_truncated` "
        "field; `translate_text()` never actually populates it separately from "
        "`translation_truncated` when running in combined mode -- one flag covers both "
        "steps for these pages, nothing is missing.)",
        "",
        "## Pages sorted by CER (descending)",
        "",
    ]
    by_cer = sorted(records, key=lambda r: r.get("cer") if r.get("cer") is not None else -1, reverse=True)
    for r in by_cer:
        cer_s = f"{r['cer']:.4f}" if r.get("cer") is not None else "—"
        overlay = r["_path"].parent / f"{r['_category']}_{r['_stem']}_overlay.png"
        gt_only = r["_path"].parent / f"{r['_category']}_{r['_stem']}_gt_only.png"
        lines.append(f"### {r['_page_id']}  (cer={cer_s})")
        lines.append(f"- overlay: `{overlay}`")
        lines.append(f"- gt-only reading order: `{gt_only}`")
        lines.append(f"- conversion_notes: {r.get('conversion_notes', '')}")
        lines.append(f"- translation_notes: {r.get('translation_notes', '')}")
        lines.append("")

    lines += ["## Notes density (proxy for how much uncertainty the model flagged)", ""]
    by_density = sorted(
        records,
        key=lambda r: len(r.get("conversion_notes", "")) + len(r.get("translation_notes", "")),
        reverse=True,
    )
    lines.append("| page | conversion_notes chars | translation_notes chars | total |")
    lines.append("|---|---|---|---|")
    for r in by_density:
        cn, tn = len(r.get("conversion_notes", "")), len(r.get("translation_notes", ""))
        lines.append(f"| {r['_page_id']} | {cn} | {tn} | {cn + tn} |")

    lines += ["", "## Uncertainty-keyword tally", ""]
    hits = _keyword_hits(records)
    lines.append("| keyword | pages mentioning it | example page |")
    lines.append("|---|---|---|")
    for kw, pages in hits.items():
        example = pages[0] if pages else "—"
        lines.append(f"| {kw} | {len(pages)}/{n} | {example} |")

    lines += ["", "## Full per-page appendix", ""]
    for r in records:
        cer_s = f"{r['cer']:.4f}" if r.get("cer") is not None else "—"
        lines.append(f"### {r['_page_id']}")
        lines.append(f"- category: {r['_category']}")
        lines.append(f"- cer: {cer_s}")
        lines.append(f"- cost_usd: ${_page_cost_usd(r):.4f}")
        lines.append(f"- normalization_method: {r.get('normalization_method', '?')}")
        lines.append("")
        lines.append(f"**classical_japanese:** {r.get('classical_japanese', '')}")
        lines.append("")
        lines.append(f"**modern_japanese:** {r.get('modern_japanese', '')}")
        lines.append(f"conversion_notes: {r.get('conversion_notes', '')}")
        lines.append("")
        lines.append(f"**english_translation:** {r.get('english_translation', '')}")
        lines.append(f"translation_notes: {r.get('translation_notes', '')}")
        lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
