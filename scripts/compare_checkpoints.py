"""Compare Stage 2 SuminaNet checkpoint variants (e.g. B/C/D) against each
other, and one of them against published external baselines (Clanuwat/
KuroNet, DKNet) for pipeline_analysis.txt items 8.1/8.2.

Prerequisite: each checkpoint variant needs its own validation run first —
this script only reads metrics.json + per_class_errors.csv, it does not
run inference itself:

    python utils/validation/validate_suminanet.py \\
        --ckpt checkpoints/B_gru_efficientnet_no_sam2/suminanet_recognizer/suminanet_best.pt \\
        --out-dir results/checkpoint_compare/B
    python utils/validation/validate_suminanet.py \\
        --ckpt checkpoints/C_gru_efficientnet_sam2/suminanet_recognizer/suminanet_best.pt \\
        --out-dir results/checkpoint_compare/C
    python utils/validation/validate_suminanet.py \\
        --ckpt checkpoints/D_bigru_efficientnet/suminanet_recognizer/suminanet_best.pt \\
        --out-dir results/checkpoint_compare/D

Usage:
    python scripts/compare_checkpoints.py \\
        --runs B=results/checkpoint_compare/B C=results/checkpoint_compare/C D=results/checkpoint_compare/D \\
        --out-dir results/checkpoint_compare/summary \\
        --baseline B
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from visualization.baselines import compare_to_baselines
from visualization.checkpoint_compare import (
    compare_checkpoints_table,
    load_checkpoint_metrics,
    plot_checkpoint_comparison_bars,
    plot_per_class_error_delta,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--runs", nargs="+", required=True, metavar="NAME=DIR",
        help="One or more NAME=DIR pairs, e.g. B=results/checkpoint_compare/B "
             "C=results/checkpoint_compare/C D=results/checkpoint_compare/D",
    )
    p.add_argument("--out-dir", default="results/checkpoint_compare/summary",
                    help="Directory to write comparison table/plots to")
    p.add_argument("--baseline", default=None,
                    help="Run name to use as the reference point for the per-class "
                         "delta plots and the external-baseline comparison "
                         "(defaults to the first --runs entry)")
    p.add_argument("--top-n", type=int, default=20,
                    help="Top-N most-improved / most-regressed characters per pair")
    return p.parse_args()


def _parse_runs(pairs: list[str]) -> dict[str, str]:
    out = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--runs entries must be NAME=DIR, got: {pair!r}")
        name, path = pair.split("=", 1)
        out[name] = path
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = _parse_runs(args.runs)
    baseline_name = args.baseline or next(iter(run_dirs))
    if baseline_name not in run_dirs:
        raise SystemExit(f"--baseline {baseline_name!r} not among --runs names: {list(run_dirs)}")

    print(f"Loading metrics for: {list(run_dirs)}")
    metrics_by_name = load_checkpoint_metrics(run_dirs)

    print("Writing comparison table...")
    table_path = compare_checkpoints_table(metrics_by_name, out_path=out_dir / "checkpoint_comparison.md")
    print(f"  -> {table_path}  (+ .csv)")

    print("Plotting grouped comparison bars (all runs)...")
    bars_path = plot_checkpoint_comparison_bars(metrics_by_name, out_path=out_dir / "checkpoint_comparison_all.png")
    print(f"  -> {bars_path}")

    names = list(run_dirs)
    for a, b in combinations(names, 2):
        pair_dir = out_dir / f"{a}_vs_{b}"
        plot_checkpoint_comparison_bars(
            {a: metrics_by_name[a], b: metrics_by_name[b]},
            out_path=pair_dir / "comparison.png",
        )
        plot_per_class_error_delta(
            metrics_by_name, baseline=a, other=b, top_n=args.top_n,
            out_path=pair_dir / "per_class_error_delta.png",
        )
        print(f"  -> {pair_dir}/ (pairwise {a} vs {b})")

    print(f"Comparing {baseline_name} against published baselines (Clanuwat/KuroNet, DKNet)...")
    baseline_metrics = metrics_by_name[baseline_name]
    your_metrics = {
        f"{baseline_name}: Stage 2 detection F1": baseline_metrics.get("det_f1", 0.0),
        f"{baseline_name}: Stage 2 top-1 accuracy": baseline_metrics.get("top1", 0.0),
    }
    baseline_plot = compare_to_baselines(
        your_metrics, checkpoint_name=baseline_name,
        out_path=out_dir / f"baseline_comparison_{baseline_name}.png",
    )
    print(f"  -> {baseline_plot}")

    print(f"\nAll outputs written under: {out_dir}")


if __name__ == "__main__":
    main()
