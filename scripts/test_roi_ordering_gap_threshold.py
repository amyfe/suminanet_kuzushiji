"""Regression checks for ROIReadingOrder._robust_gap_threshold's column/row
gap-clustering, guarding against the brsk004_014 collapse bug (page-skew
compresses the elbow ratio below min_jump_ratio, so 389 boxes across 12 real
columns collapsed into a single column) and the merge regression the fix for
it initially introduced (100249537_00062_2: a page with no jitter tail at
all -- every surviving gap already real -- got two genuine columns merged
because the new weak-elbow tier fired before the "all gaps are already
real" check).

Run:
  python scripts/test_roi_ordering_gap_threshold.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from model.suminanet.roi.roi_ordering import ROIReadingOrder, infer_reading_orientation_from_boxes
from utils.text_normalization import unicode_token_to_char


def _dedupe(gt_boxes: list, gt_chars: list[str]) -> tuple[list, list[str]]:
    seen: set[tuple] = set()
    boxes_out: list = []
    chars_out: list[str] = []
    for box, ch in zip(gt_boxes, gt_chars):
        key = (tuple(box), ch)
        if key in seen:
            continue
        seen.add(key)
        boxes_out.append(box)
        chars_out.append(ch)
    return boxes_out, chars_out


def _num_groups(stem: str) -> int:
    ann = json.loads((ROOT / "assets" / "data" / "annotations" / f"{stem}.json").read_text(encoding="utf-8"))
    raw_chars = [unicode_token_to_char(t) for t in ann["labels"]]
    gt_boxes, gt_chars = _dedupe(ann["boxes"], raw_chars)
    orientation = infer_reading_orientation_from_boxes(gt_boxes)
    boxes_t = torch.tensor(gt_boxes, dtype=torch.float32)
    mask = torch.ones((boxes_t.size(0),), dtype=torch.bool)
    _, _, _, col_ids = ROIReadingOrder().sort_single(boxes_t, mask, orientation)
    return len(Counter(col_ids.detach().cpu().tolist()))


def _orientation(stem: str) -> str:
    ann = json.loads((ROOT / "assets" / "data" / "annotations" / f"{stem}.json").read_text(encoding="utf-8"))
    return infer_reading_orientation_from_boxes(ann["boxes"])


def test_skewed_dense_page_no_longer_collapses() -> None:
    # Directly reproduces the exact gap distribution measured on brsk004_014
    # (389 deduped GT boxes, 12 true columns): the elbow-argmax already finds
    # the right split (35.5 -> 72.5, ratio ~2.04) but that ratio sits just
    # under the strict min_jump_ratio=2.5 gate.
    gaps = torch.tensor([
        9.5, 10.0, 12.0, 12.5, 13.0, 15.0, 18.0, 19.0, 23.5, 23.5, 32.5, 35.5,
        72.5, 97.0, 106.5, 117.5, 120.0, 120.5, 123.0, 130.5, 131.0, 135.0, 141.5,
    ])
    threshold = ROIReadingOrder._robust_gap_threshold(gaps, fallback=150.0)
    assert 48.0 < threshold < 53.0, f"expected ~50.8, got {threshold}"


def test_skewed_dense_page_model_inference_also_recovers() -> None:
    # Same page (brsk004_014), but the gap distribution from the model's own
    # *predicted* boxes (real inference, refine_scores all >= confidence so
    # every box anchors) rather than clean GT boxes: localization noise
    # stacks on top of the page skew and depresses the true elbow ratio
    # further, to ~1.65 -- below the first-pass weak_jump_ratio=1.8, which
    # is why that value wasn't loose enough and had to be retuned to 1.6.
    gaps = torch.tensor([
        7.75, 8.55, 8.95, 9.30, 12.85, 17.5, 17.85, 19.4, 22.1, 26.8, 41.55,
        68.75, 101.35, 118.05, 118.85, 123.1, 123.55, 130.5, 130.9, 132.8,
        134.0, 134.75, 141.35,
    ])
    threshold = ROIReadingOrder._robust_gap_threshold(gaps, fallback=150.30001831054688)
    n_cols = int((gaps > threshold).sum().item()) + 1
    assert n_cols >= 10, f"expected >=10 columns (true count ~12-13), got {n_cols} (threshold={threshold})"


def test_all_real_gaps_page_not_over_merged() -> None:
    # Reproduces 100249537_00062_2 (6 true columns): every surviving gap is
    # already at real-boundary scale (no jitter tail survived noise_floor
    # filtering), so the largest *ratio* jump among them (131.0 -> 242.5,
    # ratio ~1.85) is not a jitter/real split -- it's two real gaps that
    # happen to differ in size. The "all surviving gaps are real" check must
    # win over the elbow logic here, or two genuine columns get merged.
    gaps = torch.tensor([131.0, 242.5, 246.0, 247.5, 251.5])
    threshold = ROIReadingOrder._robust_gap_threshold(gaps, fallback=184.5)
    assert 130.9 < threshold < 131.0, f"expected ~130.999 (all-real), got {threshold}"


def test_single_column_uniform_jitter_stays_one_group() -> None:
    # No real elbow, no gap anywhere near character-width scale -- must
    # still report a single group, not get split by the new weak tier.
    gaps = torch.tensor([5.0, 6.0, 4.5, 7.0, 5.5, 6.5, 5.0, 4.0, 6.0])
    threshold = ROIReadingOrder._robust_gap_threshold(gaps, fallback=150.0)
    assert threshold == 150.0, f"expected fallback (150.0), got {threshold}"


def test_brsk004_014_real_page_recovers_true_columns() -> None:
    assert _num_groups("brsk004_014") == 12


def test_brsk005_005_sibling_page_unregressed() -> None:
    assert _num_groups("brsk005_005") == 12


def test_100249537_00062_2_real_page_not_merged() -> None:
    assert _num_groups("100249537_00062_2") == 6


def test_genuine_single_column_pages_stay_one_group() -> None:
    assert _num_groups("200021063_00022_1") == 1
    assert _num_groups("hnsd007_021") == 1


def test_short_vertical_columns_no_longer_misclassified_horizontal() -> None:
    # infer_reading_orientation_from_boxes's mean-nearest-neighbor-distance
    # heuristic misclassifies short (2-4 char) vertical columns as
    # "horizontal", because with few characters per column the between-
    # column jump is comparable to or larger than the within-column step.
    # A full corpus sweep (5344 annotation files) found 22 pages the
    # heuristic ever calls "horizontal"; 21 of them are this misclassification
    # (spot-checked by hand against raw box coordinates). Sample spanning the
    # affected size range, from the originally-confirmed n=7 caption to a
    # dense n=56 page.
    for stem in [
        "200021925_00012_1",  # n=7, originally-confirmed bug
        "200019865_00047_1",  # n=24
        "200021660_00077_2",  # n=56, dense
    ]:
        assert _orientation(stem) == "vertical", f"{stem} still misclassified horizontal"


def test_genuine_horizontal_stamp_stays_horizontal() -> None:
    # 200021851_00030_2 is a real printed library-archive stamp
    # ("国文学研究資料館" + a date) at the bottom of the page -- genuinely
    # left-to-right, and the one confirmed true positive among the 22 pages
    # the orientation heuristic ever flags "horizontal". The clustering
    # corroboration must not flip this one: unlike the misclassified vertical
    # pages, clustering it as columns does not find more groups than
    # clustering it as rows (both collapse to a single group), so the
    # override does not fire.
    assert _orientation("200021851_00030_2") == "horizontal"


def main() -> None:
    test_skewed_dense_page_no_longer_collapses()
    test_skewed_dense_page_model_inference_also_recovers()
    test_all_real_gaps_page_not_over_merged()
    test_single_column_uniform_jitter_stays_one_group()
    test_brsk004_014_real_page_recovers_true_columns()
    test_brsk005_005_sibling_page_unregressed()
    test_100249537_00062_2_real_page_not_merged()
    test_genuine_single_column_pages_stay_one_group()
    test_short_vertical_columns_no_longer_misclassified_horizontal()
    test_genuine_horizontal_stamp_stays_horizontal()
    print("PASS: ROIReadingOrder gap-threshold regression checks")


if __name__ == "__main__":
    main()
