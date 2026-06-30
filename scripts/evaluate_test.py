"""
Final held-out test-set evaluation for Stage 1 detector.

IMPORTANT: test.txt == val.txt. Val metrics were used as the Optuna objective,
so these numbers are not fully independent of hyperparameter selection. Going
forward, do NOT use test performance to guide further tuning decisions. Report
these numbers once, at the end of a training cycle, as the final result.

Usage:
    python scripts/evaluate_test.py --checkpoint checkpoints/stage1_detection/detector_best.pt
    python scripts/evaluate_test.py --checkpoint ... --iou-thresh 0.5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1 test-set evaluation")
    parser.add_argument(
        "--checkpoint", type=Path,
        default=ROOT / "checkpoints" / "stage1_detection" / "detector_best.pt",
    )
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--worst-n", type=int, default=30)
    args = parser.parse_args()

    test_split_file = ROOT / "assets" / "data" / "splits" / "test.txt"
    if not test_split_file.exists():
        print(f"[ERROR] test.txt not found at {test_split_file}")
        sys.exit(1)

    print("=" * 70)
    print("STAGE 1 — TEST SET EVALUATION")
    print("=" * 70)
    print(
        "NOTE: test.txt == val.txt. Val was used for Optuna objective selection.\n"
        "Do not use these numbers to tune further hyperparameters.\n"
    )

    from validate_stage1 import validate_stage1
    validate_stage1(
        checkpoint_path=args.checkpoint,
        split="test",
        iou_threshold=args.iou_thresh,
        worst_n=args.worst_n,
    )


if __name__ == "__main__":
    main()
