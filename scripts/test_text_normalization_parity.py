"""Regression checks for old-Japanese token rendering parity in text metrics.

Run:
  python scripts/test_text_normalization_parity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.text_normalization import render_tokens


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def _cer(pred: str, gt: str) -> float:
    return _edit_distance(pred, gt) / max(1, len(gt))


def test_uplus_tokens_render_to_chars() -> None:
    assert render_tokens(["U+3042", "U+3044"]) == "あい"
    assert render_tokens(["u+65E5"]) == "日"
    assert render_tokens(["abc"]) == "abc"


def test_cer_computed_on_rendered_text() -> None:
    # Raw tokenized strings should not be used directly for CER.
    pred_tokens = ["U+3042", "U+3044"]
    gt_tokens = ["U+3042", "U+3048"]

    pred_rendered = render_tokens(pred_tokens)  # あい
    gt_rendered = render_tokens(gt_tokens)      # あえ
    cer_rendered = _cer(pred_rendered, gt_rendered)

    pred_raw = "".join(pred_tokens)
    gt_raw = "".join(gt_tokens)
    cer_raw = _cer(pred_raw, gt_raw)

    assert pred_rendered == "あい"
    assert gt_rendered == "あえ"
    assert cer_rendered == 0.5
    assert cer_raw != cer_rendered


def main() -> None:
    test_uplus_tokens_render_to_chars()
    test_cer_computed_on_rendered_text()
    print("PASS: text normalization parity checks")


if __name__ == "__main__":
    main()
