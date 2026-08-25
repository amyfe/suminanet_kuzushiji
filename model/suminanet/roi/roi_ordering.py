"""Ordnet refined ROIs in Lesereihenfolge.

Input
refined_boxes
roi_mask
orientations pro Sample

Output
reorder indices
sortierte boxes/features/mask
Empfehlung

Zunächst als deterministische Heuristik, nicht lernbasiert.

Für vertikal:

Spalten rechts → links
innerhalb Spalte oben → unten

Für horizontal:

Zeilen oben → unten
innerhalb Zeile links → rechts

Wichtig:

zunächst deterministisch
keine lernbasierte Reihenfolgenvorhersage
arbeitet auf refined boxes
sortiert auch Features, Scores und Masken konsistent mit

ToDos:

1. Das ist bewusst nur eine Heuristik
Sie ist okay als erster Schritt, aber nicht “intelligent”.
Probleme bekommst du, wenn:
Spalten/Zeilen stark unregelmäßig sind
Boxen durch Detector verrauscht sind
Layout nicht sauber einspaltig/mehrspaltig ist

2. line_merge_thresh_ratio ist sehr wichtig
Dieser Wert entscheidet, ob Zeichen derselben Zeile/Spalte zusammengefasst werden oder nicht.
Den musst du später empirisch prüfen.

3. Der Sorter nimmt an, dass mask vorne zusammenhängend valide ist

Das passt zu deiner aktuellen Padding-Logik. Wenn du später irgendwo Löcher in der Sequenz hast, musst du das robuster machen.
"""

# model/suminanet/roi/roi_ordering.py

from __future__ import annotations

from math import isfinite
from typing import List, Optional

import torch

from config import SUMINANET_READING_ORDER_CONFIDENCE, STAGE2_READING_ORDER_LINE_THRESH_RATIO


def _corroborate_horizontal_with_clustering(boxes, debug: bool = False) -> str:
    """
    Second opinion for a tentative "horizontal" call from the nearest-neighbor
    dx/dy heuristic above, which was found to misclassify short vertical
    columns (2-4 chars/column) as horizontal: with few characters per column,
    the between-column jump distance is comparable to or larger than the
    within-column step, making the page's mean nearest-neighbor dx look
    "row-like" even though it's genuinely a right-to-left column layout. A
    full sweep of every page the heuristic ever calls "horizontal" (22/5344)
    found this misclassifies 21/22 of them; the one confirmed genuine case
    (a printed library-archive stamp) is a single tight text block that does
    not cluster into multiple short columns.

    Re-clusters the same boxes both ways -- as columns (the real vertical
    sort path) and as rows (the real horizontal sort path) -- via
    ROIReadingOrder's own column/row detection, and checks which axis finds
    genuine multi-group structure. Short vertical columns cluster cleanly
    into several small column-groups while collapsing into few/degenerate
    row-groups (rows and columns are order-independent of self here -- see
    ROIReadingOrder._vertical_sort_indices/_horizontal_sort_indices, neither
    of which reads any instance state). A real horizontal block does the
    reverse or ties. Only overrides toward "vertical"; never overrides
    toward "horizontal" -- the nearest-neighbor heuristic already handles the
    dominant vertical case correctly, so this stays scoped to the one
    confirmed failure direction.
    """
    boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
    ror = ROIReadingOrder()
    _, col_ids = ror._vertical_sort_indices(boxes_t)
    _, row_ids = ror._horizontal_sort_indices(boxes_t)
    num_cols = int(col_ids.max().item()) + 1 if col_ids.numel() else 0
    num_rows = int(row_ids.max().item()) + 1 if row_ids.numel() else 0

    orientation = "vertical" if num_cols > num_rows else "horizontal"
    if debug:
        print(
            f"[ROI ORI DEBUG] clustering corroboration | num_cols={num_cols} | "
            f"num_rows={num_rows} -> {orientation}"
        )
    return orientation


def infer_reading_orientation_from_boxes(boxes, debug: bool = False) -> str:
    """Infer reading direction from nearest-neighbor center distances."""
    if boxes is None or len(boxes) < 2:
        if debug:
            print("[ROI ORI DEBUG] n=0/1 -> vertical (fallback)")
        return "vertical"

    centers = torch.tensor(
        [[0.5 * (float(b[0]) + float(b[2])), 0.5 * (float(b[1]) + float(b[3]))] for b in boxes],
        dtype=torch.float32,
    )

    dx = torch.cdist(centers[:, :1], centers[:, :1], p=1)
    dy = torch.cdist(centers[:, 1:2], centers[:, 1:2], p=1)

    # Mask self-distances out of nearest-neighbor computations.
    # Using eye * inf produces NaNs because 0 * inf is undefined.
    dx.fill_diagonal_(float("inf"))
    dy.fill_diagonal_(float("inf"))

    mean_min_dx = float(dx.min(dim=1).values.mean().item())
    mean_min_dy = float(dy.min(dim=1).values.mean().item())

    if not (isfinite(mean_min_dx) and isfinite(mean_min_dy)):
        if debug:
            print(
                f"[ROI ORI DEBUG] non-finite distances | mean_min_dx={mean_min_dx} | "
                f"mean_min_dy={mean_min_dy} -> vertical (fallback)"
            )
        return "vertical"

    # Vertical pages tend to form tight x-aligned columns (small nearest-neighbor dx),
    # while horizontal pages tend to form tight y-aligned rows (small nearest-neighbor dy).
    orientation = "vertical" if mean_min_dx <= mean_min_dy else "horizontal"

    if orientation == "horizontal":
        orientation = _corroborate_horizontal_with_clustering(boxes, debug=debug)

    if debug:
        print(
            f"[ROI ORI DEBUG] n={len(boxes)} | mean_min_dx={mean_min_dx:.4f} | "
            f"mean_min_dy={mean_min_dy:.4f} | orientation={orientation}"
        )

    return orientation


class ROIReadingOrder:
    """
    Deterministic reading-order sorter for refined ROI sequences.

    Supported orientations:
        - "horizontal": top-to-bottom lines, left-to-right within line
        - "vertical":   right-to-left columns, top-to-bottom within column

    Important:
    This is a heuristic ordering module, not a learned layout model.
    """

    def __init__(self, line_merge_thresh_ratio: float = STAGE2_READING_ORDER_LINE_THRESH_RATIO):
        """
        Args:
            line_merge_thresh_ratio:
                Controls grouping into approximate rows/columns.
                Threshold is scaled by median box height (horizontal)
                or median box width (vertical).
        """
        self.line_merge_thresh_ratio = float(line_merge_thresh_ratio)

    @staticmethod
    def _valid_count(mask_b: torch.Tensor) -> int:
        return int(mask_b.sum().item())

    def _primary_axis_monotonic_fraction(
        self,
        boxes_sorted_valid: torch.Tensor,
        orientation: str,
    ) -> tuple[float, float]:
        """
        Compute a lightweight ordering-quality diagnostic on sorted valid boxes.

        Returns:
            monotonic_fraction, violation_fraction
        """
        n = int(boxes_sorted_valid.size(0))
        if n <= 1:
            return 1.0, 0.0

        if orientation == "horizontal":
            primary = (boxes_sorted_valid[:, 1] + boxes_sorted_valid[:, 3]) * 0.5
            scale = (boxes_sorted_valid[:, 3] - boxes_sorted_valid[:, 1]).clamp_min(1.0).median()
            tol = float(self.line_merge_thresh_ratio * scale.item())
            ok = primary[1:] >= (primary[:-1] - tol)
        elif orientation == "vertical":
            primary = (boxes_sorted_valid[:, 0] + boxes_sorted_valid[:, 2]) * 0.5
            scale = (boxes_sorted_valid[:, 2] - boxes_sorted_valid[:, 0]).clamp_min(1.0).median()
            tol = float(self.line_merge_thresh_ratio * scale.item())
            ok = primary[1:] <= (primary[:-1] + tol)
        else:
            return 1.0, 0.0

        mono = float(ok.float().mean().item()) if ok.numel() > 0 else 1.0
        return mono, 1.0 - mono

    @staticmethod
    def _percentile_size(sizes: torch.Tensor, p: float = 0.75) -> float:
        """Return the p-th percentile of a 1D size tensor (robust to FP outliers)."""
        n = sizes.numel()
        k = max(1, min(n, int(p * n + 0.5)))
        return float(sizes.kthvalue(k).values.item())

    @staticmethod
    def _robust_gap_threshold(
        gaps: torch.Tensor,
        fallback: float,
        min_jump_ratio: float = 2.5,
        weak_jump_ratio: float = 1.6,
        weak_jump_scale_frac: float = 0.4,
    ) -> float:
        """
        Separate within-column/row jitter gaps from real column/row-boundary
        gaps by finding the largest relative jump in the *sorted gap
        distribution* itself, instead of assuming spacing tracks character
        size (width/height).

        Character-size-based thresholds (the previous `1.5 * p75(size)`
        heuristic) collapse on dense pages where character size approaches
        true column/row pitch: real inter-column gaps end up close to or
        below the size-derived threshold, merging many columns into one and
        degenerating the reading-order sort into a near-global scan by the
        secondary axis. This was the root cause of the ~38% ordering-
        violation rate on dense Kuzushiji pages noted in the sort functions'
        docstrings.

        Column/row gaps are naturally bimodal (many small jitter gaps within
        a column, a handful of large gaps between columns), so the boundary
        between the two populations shows up as a sharp ratio jump in the
        sorted gap array. Falls back to `fallback` when there are too few
        gaps to detect a reliable elbow, or when no jump stands out clearly
        (e.g. a single column with uniform jitter — nothing should be
        treated as a boundary).

        No-elbow-found does NOT always mean "single column, everything's
        jitter", though: on tightly-refined boxes (e.g. SuminaNet's ROI
        refinement snapping same-column boxes to near-identical x-centers),
        every within-column gap can fall below `noise_floor` and get
        filtered out, leaving `positive` containing *only* real,
        evenly-spaced inter-column gaps -- flat in ratio terms (no elbow)
        but sitting at character-width scale rather than jitter scale.
        Falling back to `fallback` (~1.5x character width) in that case can
        exceed this page's actual (tighter) column pitch and wrongly merge
        every column into one, so the *scale* of the surviving gaps, not
        just their ratio pattern, decides which case applies.

        Page skew can also depress the elbow ratio below `min_jump_ratio`
        without moving *where* the elbow is: on assets/data/annotations/
        brsk004_014.json (a visibly rotated scan), the elbow-argmax already
        landed exactly on the true column boundary (ratio ~2.04 at the
        35.5->72.5 gap transition, correctly recovering the page's 12
        columns when checked manually) but the strict `min_jump_ratio=2.5`
        gate rejected it, falling through to `fallback` -- which exceeded
        even the largest real gap on that page, collapsing all 389 boxes
        into a single column. Uniform per-page rotation widens each
        column's own x-center spread (skew smears same-column boxes further
        apart as y increases), inflating the jitter tail enough to blur the
        ratio against the real-boundary floor, even though the boundary's
        *location* is still exactly right. The `weak_jump_ratio` /
        `weak_jump_scale_frac` tier below corroborates a below-threshold
        ratio with an absolute, character-width-anchored sanity check
        instead of loosening `min_jump_ratio` globally (which would also
        relax acceptance for every page whose largest *coincidental* noise
        ratio happens to land in that band, including genuine single-column
        pages) -- it only fires when the candidate boundary gap is itself
        already a substantial fraction of a real character width, which
        ordinary within-column jitter should not be.

        `weak_jump_ratio=1.6` (not e.g. 1.8) because the anchor-scored,
        real-inference path is noisier than the clean GT case above: on
        brsk004_014's own *predicted* boxes (this same page, run through
        the actual detector+SuminaNet pipeline, not hand-labeled GT), model
        localization noise stacks on top of the page skew and depresses the
        true elbow ratio further, to ~1.65 -- still the correct split
        location (would recover ~12-13 columns), just even harder to trust
        by ratio alone. Calibrated via a 300-file random sample sweep over
        assets/data/annotations/ comparing candidate settings against the
        already-shipped 1.8/0.4 tier: 1.6 is the loosest value that also
        fixes the model-inference case, and going looser (down to 1.4)
        recovers no further pages in that sample -- i.e. 1.6 sits on the
        conservative edge of what's needed, not a wide-open threshold.
        """
        noise_floor = max(1e-6, 0.05 * fallback)
        positive = gaps[gaps > noise_floor]
        if positive.numel() < 3:
            return fallback
        sorted_gaps, _ = positive.sort()
        ratios = sorted_gaps[1:] / sorted_gaps[:-1].clamp_min(1e-6)
        elbow_idx = int(ratios.argmax().item())
        elbow_ratio = float(ratios[elbow_idx].item())
        boundary_gap = float(sorted_gaps[elbow_idx + 1].item())
        if float(sorted_gaps[0].item()) >= 0.5 * fallback:
            # All surviving gaps already sit at character-width scale, not
            # jitter scale -- there's no jitter tail left to separate from
            # real gaps (it was fully absorbed by noise_floor), so the
            # ratio-elbow question below doesn't apply: treat every
            # surviving gap as a real boundary, whatever its ratio to its
            # neighbors. Must run before the elbow checks -- e.g. a page
            # with column gaps [131, 242, 246, 247, 251] (all real; no
            # jitter at all) has its largest ratio jump at 131->242
            # (~1.85, clears weak_jump_ratio), which would wrongly discard
            # the genuine 131-gap boundary if checked first.
            return float(sorted_gaps[0].item()) - 1e-3

        strong_elbow = elbow_ratio >= min_jump_ratio
        corroborated_weak_elbow = (
            elbow_ratio >= weak_jump_ratio and boundary_gap >= weak_jump_scale_frac * fallback
        )
        if strong_elbow or corroborated_weak_elbow:
            # Geometric mean of the two gap values straddling the elbow.
            return float((sorted_gaps[elbow_idx] * sorted_gaps[elbow_idx + 1]).sqrt().item())
        return fallback

    @staticmethod
    def _confidence_anchor_mask(
        scores: Optional[torch.Tensor],
        n: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        True = trust this box's own position when establishing column/row
        geometry. Falls back to trusting every box (today's behavior --
        restricting to "all boxes" is a no-op) when no scores are given (e.g.
        the ground-truth box path in KuzushijiDataset) or when fewer than 3
        boxes clear the confidence bar (too little signal to cluster from).
        """
        if scores is None:
            return torch.ones(n, dtype=torch.bool, device=device)
        anchor = torch.sigmoid(scores.detach()) >= SUMINANET_READING_ORDER_CONFIDENCE
        if int(anchor.sum().item()) < 3:
            return torch.ones(n, dtype=torch.bool, device=device)
        return anchor

    @staticmethod
    def _establish_groups(
        center: torch.Tensor,
        anchor: torch.Tensor,
        fallback_gap_thresh: float,
    ) -> tuple[torch.Tensor, int]:
        """
        Cluster `center` values (x-centers for columns, y-centers for rows)
        into contiguous groups using only anchor-flagged entries to detect
        group boundaries -- see _confidence_anchor_mask / _robust_gap_threshold
        for why: unfiltered, noisy low-confidence entries smear the gaps
        between real groups and collapse detection on dense pages.

        Returns (group_ids_full, num_groups). group_ids_full has one entry per
        input position; only anchor positions are meaningful until the caller
        follows up with _snap_small_to_columns(..., is_anchor=anchor, ...) to
        assign non-anchor positions to their nearest anchor-derived group.
        """
        anchor_pos = anchor.nonzero(as_tuple=True)[0]
        anchor_center = center[anchor_pos]
        sorted_c, local_order = anchor_center.sort(descending=False)

        if sorted_c.numel() > 1:
            gaps = sorted_c[1:] - sorted_c[:-1]
            gap_thresh = ROIReadingOrder._robust_gap_threshold(gaps, fallback_gap_thresh)
            is_break = torch.cat([
                torch.ones(1, dtype=torch.bool, device=center.device),
                gaps > gap_thresh,
            ])
        else:
            is_break = torch.ones(1, dtype=torch.bool, device=center.device)

        local_group_ids = is_break.long().cumsum(0) - 1   # (n_anchor,) in local sorted order
        num_groups = int(local_group_ids.max().item()) + 1

        # Translate local (within-anchor-subset) sorted positions back to
        # original indices before writing into the full-length array -- these
        # are NOT the same index space, so a direct scatter with local_order
        # would silently write to the wrong original positions.
        orig_positions_sorted = anchor_pos[local_order]
        group_ids_full = torch.zeros(center.size(0), dtype=torch.long, device=center.device)
        group_ids_full[orig_positions_sorted] = local_group_ids

        return group_ids_full, num_groups

    def _snap_small_to_columns(
        self,
        x_center: torch.Tensor,    # (n,)
        col_ids_orig: torch.Tensor, # (n,) long — column assignment for all chars
        is_anchor: torch.Tensor,   # (n,) bool — True = trust this char's own column id
        num_cols: int,
    ) -> torch.Tensor:
        """
        Reassign non-anchor characters to the nearest anchor-established column.

        Uses the mean x-center of anchor chars per column as the column center.
        Non-anchor chars that landed in a single-char isolated column get snapped to
        the nearest anchor column so they no longer create spurious column boundaries.

        `is_anchor` serves two purposes across this function's two call sites:
        confidence-based (trustworthy detection) in the geometry-establishment pass,
        and width/height-based (main-text vs. furigana) in the furigana-snap pass.

        Returns updated col_ids_orig (same shape, only non-anchor entries changed).
        """
        if not is_anchor.any() or is_anchor.all():
            return col_ids_orig

        # Per-column mean x-center from anchor chars only
        anchor_pos = is_anchor.nonzero(as_tuple=True)[0]
        anchor_col = col_ids_orig[anchor_pos]
        anchor_xc  = x_center[anchor_pos]

        col_sum   = x_center.new_zeros(num_cols)
        col_count = torch.zeros(num_cols, dtype=torch.long, device=x_center.device)
        col_sum.scatter_add_(0, anchor_col, anchor_xc)
        col_count.scatter_add_(0, anchor_col, torch.ones_like(anchor_col))

        valid_col = col_count > 0
        col_mean_xc = col_sum.clone()
        col_mean_xc[valid_col] = (
            col_sum[valid_col] / col_count[valid_col].to(x_center.dtype)
        )
        # Mark columns with no anchor chars as infinitely far
        col_mean_xc[~valid_col] = float("inf")

        # Reassign non-anchor chars to nearest anchor column by x-distance
        small_pos = (~is_anchor).nonzero(as_tuple=True)[0]
        furi_xc   = x_center[small_pos].unsqueeze(1)   # (N_small, 1)
        col_xc    = col_mean_xc.unsqueeze(0)            # (1, num_cols)
        nearest   = (furi_xc - col_xc).abs().argmin(dim=1)  # (N_small,)

        col_ids_out = col_ids_orig.clone()
        col_ids_out[small_pos] = nearest
        return col_ids_out

    def _horizontal_sort_indices(
        self,
        boxes: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Horizontal reading order: rows top-to-bottom, within each row left-to-right.

        Row geometry is established from high-confidence boxes only (see
        _confidence_anchor_mask / _robust_gap_threshold): low-confidence boxes
        in the raw candidate pool would otherwise smear the gaps between real
        rows and collapse row detection on dense pages. Non-confident boxes
        are then snapped to their nearest anchor-established row. Small
        ruby/furigana characters are additionally snapped to the nearest
        main-text row by height after that, preventing them from creating
        spurious extra rows.

        Returns (sort_idx, row_ids_sorted): row_ids_sorted is the row/column
        membership id of each box, reindexed into the same reading order as
        sort_idx -- exposed so callers can tell a mere row wrap apart from a
        genuine change in content (see translation.py's block-break logic).
        """
        if boxes.numel() == 0:
            empty = torch.zeros((0,), dtype=torch.long, device=boxes.device)
            return empty, empty

        y_center = (boxes[:, 1] + boxes[:, 3]) * 0.5
        heights  = (boxes[:, 3] - boxes[:, 1]).clamp_min(1.0)

        # Separate main-text chars from small (ruby/furigana) chars
        h_median = heights.median()
        is_main  = heights >= (0.5 * h_median)

        if is_main.sum() >= 3:
            h75 = self._percentile_size(heights[is_main], 0.75)
        else:
            h75 = self._percentile_size(heights, 0.75)
        fallback_gap_thresh = 1.5 * h75

        anchor = self._confidence_anchor_mask(scores, boxes.size(0), boxes.device)
        row_ids_orig, num_rows = self._establish_groups(y_center, anchor, fallback_gap_thresh)
        row_ids_orig = self._snap_small_to_columns(
            x_center=y_center,
            col_ids_orig=row_ids_orig,
            is_anchor=anchor,
            num_cols=num_rows,
        )

        # Snap small/ruby chars to the nearest main-text row
        row_ids_orig = self._snap_small_to_columns(
            x_center=y_center,
            col_ids_orig=row_ids_orig,
            is_anchor=is_main,
            num_cols=num_rows,
        )

        # Compound key: row_rank * scale + x_left
        scale    = float(boxes[:, 2].max().item()) + 1.0
        sort_key = row_ids_orig.to(boxes.dtype) * scale + boxes[:, 0]

        sort_idx = sort_key.argsort()
        return sort_idx, row_ids_orig[sort_idx]

    def _vertical_sort_indices(
        self,
        boxes: torch.Tensor,
        scores: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Vertical reading order: columns right-to-left (Edo), within each column top-to-bottom.

        Column geometry is established from high-confidence boxes only (see
        _confidence_anchor_mask / _robust_gap_threshold): low-confidence boxes
        in the raw candidate pool (weak/duplicate detections, gap-filled
        synthetic proposals) have x-centers that fall in the gutters between
        real columns, which -- left unfiltered -- smear away the gap
        separation between columns and collapse column detection on dense
        pages (the root cause of the ~38% ordering-violation rate in dense
        Kuzushiji pages). Non-confident boxes are then snapped to their
        nearest anchor-established column. Furigana/small chars are
        additionally snapped to the nearest main-text column by width after
        that, preventing them from creating spurious isolated columns.

        Returns (sort_idx, col_ids_sorted): col_ids_sorted is the column
        membership id of each box, reindexed into the same reading order as
        sort_idx -- exposed so callers can tell a mere column wrap apart from
        a genuine change in content (see translation.py's block-break logic).
        """
        if boxes.numel() == 0:
            empty = torch.zeros((0,), dtype=torch.long, device=boxes.device)
            return empty, empty

        x_center = (boxes[:, 0] + boxes[:, 2]) * 0.5
        widths   = (boxes[:, 2] - boxes[:, 0]).clamp_min(1.0)

        # Separate main-text chars from small (furigana) chars.
        # Chars narrower than 50% of the median width are treated as furigana/ruby.
        w_median = widths.median()
        is_main  = widths >= (0.5 * w_median)

        if is_main.sum() >= 3:
            w75 = self._percentile_size(widths[is_main], 0.75)
        else:
            w75 = self._percentile_size(widths, 0.75)
        fallback_gap_thresh = 1.5 * w75

        anchor = self._confidence_anchor_mask(scores, boxes.size(0), boxes.device)
        col_ids_orig, num_cols = self._establish_groups(x_center, anchor, fallback_gap_thresh)
        col_ids_orig = self._snap_small_to_columns(
            x_center=x_center,
            col_ids_orig=col_ids_orig,
            is_anchor=anchor,
            num_cols=num_cols,
        )

        # Snap furigana/small chars to the nearest main-text column center.
        # Without this, a furigana column between two main columns creates an
        # extra col_id that gets interleaved, producing x-center violations.
        col_ids_orig = self._snap_small_to_columns(
            x_center=x_center,
            col_ids_orig=col_ids_orig,
            is_anchor=is_main,
            num_cols=num_cols,
        )

        # Compound key: col_rank * scale + y_top
        col_rank = (num_cols - 1 - col_ids_orig).to(boxes.dtype)
        scale    = float(boxes[:, 3].max().item()) + 1.0
        sort_key = col_rank * scale + boxes[:, 1]

        sort_idx = sort_key.argsort()
        return sort_idx, col_ids_orig[sort_idx]

    def _compute_isolation_mask(
        self,
        boxes: torch.Tensor,    # (N, 4)  valid sorted boxes
        orientation: str,
        scores: Optional[torch.Tensor] = None,
        min_col_size: int = 3,
        size_ratio: float = 3.0,
        furi_ratio: float = 2.5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns two boolean masks (isolated, furigana) for valid sorted boxes.

        isolated: small group (< min_col_size) OR oversized (area > size_ratio × median).
                  These are likely illustration FPs — get strong BG supervision.
        furigana: sub-columns whose median area is furi_ratio× smaller than an adjacent
                  close group.  These are likely real kana gloss characters — no BG
                  supervision so the classifier can learn them as real characters.

        scores, if given, gate which boxes contribute to each group's "typical size"
        statistic (see _confidence_anchor_mask) -- otherwise a handful of low-
        confidence boxes sharing a group with one genuinely large real character
        (e.g. gap-filled candidates near a large decorative title character) can drag
        the group's typical size down, spuriously flagging the real character as
        oversized/isolated and dropping it from the transcription.
        """
        n = boxes.size(0)
        if n == 0:
            empty = torch.zeros((0,), dtype=torch.bool, device=boxes.device)
            return empty, empty.clone()

        areas = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).clamp_min(1.0)

        if orientation == "vertical":
            center = (boxes[:, 0] + boxes[:, 2]) * 0.5
            sizes  = (boxes[:, 2] - boxes[:, 0]).clamp_min(1.0)
        else:
            center = (boxes[:, 1] + boxes[:, 3]) * 0.5
            sizes  = (boxes[:, 3] - boxes[:, 1]).clamp_min(1.0)

        p75        = self._percentile_size(sizes, 0.75)
        gap_thresh = 1.5 * p75

        sorted_c, sorted_idx = center.sort()

        if n > 1:
            gaps     = sorted_c[1:] - sorted_c[:-1]
            is_break = torch.cat([
                torch.ones(1, dtype=torch.bool, device=boxes.device),
                gaps > gap_thresh,
            ])
        else:
            is_break = torch.ones(1, dtype=torch.bool, device=boxes.device)

        group_ids_sorted = is_break.long().cumsum(0) - 1
        num_groups       = int(group_ids_sorted.max().item()) + 1

        # Map sorted positions back to original reading-order positions
        group_ids = torch.zeros(n, dtype=torch.long, device=boxes.device)
        group_ids.scatter_(0, sorted_idx, group_ids_sorted)

        # Vectorized group stats — groups are contiguous in the sorted-by-center order.
        group_counts = torch.bincount(group_ids_sorted, minlength=num_groups)   # (G,)

        # Typical area per group, from confidence-anchored boxes only where
        # available (falls back to all boxes in the group when none are anchors)
        # -- see this method's docstring for why unfiltered low-confidence boxes
        # would corrupt this statistic. Mean rather than true median so it stays
        # a simple vectorized scatter_add over an arbitrary anchor subset.
        areas_sorted  = areas[sorted_idx]                                         # (n,)
        # Deliberately NOT _confidence_anchor_mask: that helper falls back to
        # "trust every box" when fewer than 3 clear the confidence bar, which
        # is right for gap-based group *detection* (need several points to see
        # a pattern) but wrong here -- a single confident box is still a useful
        # "what does a real character look like" reference, and reverting to
        # "trust everyone" is exactly the dilution this fix exists to avoid.
        # A group with zero confident boxes still falls back via `has_anchor`
        # below, just per-group instead of globally.
        if scores is None:
            anchor = torch.ones(n, dtype=torch.bool, device=boxes.device)
        else:
            anchor = torch.sigmoid(scores.detach()) >= SUMINANET_READING_ORDER_CONFIDENCE
        anchor_sorted = anchor[sorted_idx]                                        # (n,)

        anchor_area_sum = areas_sorted.new_zeros(num_groups)
        anchor_area_sum.scatter_add_(0, group_ids_sorted, areas_sorted * anchor_sorted.to(areas_sorted.dtype))
        anchor_count = torch.zeros(num_groups, dtype=torch.long, device=boxes.device)
        anchor_count.scatter_add_(0, group_ids_sorted, anchor_sorted.long())

        all_area_sum = areas_sorted.new_zeros(num_groups)
        all_area_sum.scatter_add_(0, group_ids_sorted, areas_sorted)

        has_anchor = anchor_count > 0
        group_med_areas = torch.where(
            has_anchor,
            anchor_area_sum / anchor_count.clamp_min(1).to(areas_sorted.dtype),
            all_area_sum / group_counts.clamp_min(1).to(areas_sorted.dtype),
        )                                                                         # (G,)

        # Group centers: mean of sorted_c values within each group
        coord_sum = sorted_c.new_zeros(num_groups)
        coord_sum.scatter_add_(0, group_ids_sorted, sorted_c)
        group_centers_t = coord_sum / group_counts.to(sorted_c.dtype)            # (G,)

        # --- First pass: per-ROI isolation (small group or oversized box) ---
        group_count_per_roi = group_counts[group_ids]     # (n,) in original order
        group_med_per_roi   = group_med_areas[group_ids]  # (n,)
        isolated = (group_count_per_roi < min_col_size) | (areas > size_ratio * group_med_per_roi)

        # --- Second pass: furigana-like sub-column detection ---
        # For adjacent group pairs (g, g+1): if they are close and one group's median
        # area is furi_ratio× smaller, that group is furigana.
        # Returned as a *separate* mask — furigana is likely real kana and should
        # not receive BG supervision.
        furigana = torch.zeros(n, dtype=torch.bool, device=boxes.device)
        if num_groups > 1:
            gaps_g = group_centers_t[1:] - group_centers_t[:-1]           # (G-1,)
            close  = gaps_g < 3.0 * gap_thresh                             # (G-1,)
            ratios = group_med_areas[1:] / (group_med_areas[:-1] + 1e-6)  # (G-1,)

            # ratio = area(g+1)/area(g):
            #   ratio > furi_ratio  → g+1 much bigger → g is the furigana (flag g)
            #   ratio < 1/furi_ratio → g much bigger  → g+1 is the furigana (flag g+1)
            furi_groups = torch.zeros(num_groups, dtype=torch.bool, device=boxes.device)
            furi_groups[:num_groups - 1].logical_or_(close & (ratios > furi_ratio))
            furi_groups[1:].logical_or_(close & (ratios < (1.0 / furi_ratio)))

            furigana = furi_groups[group_ids]

        return isolated, furigana

    def sort_single(
        self,
        boxes: torch.Tensor,        # (T, 4)
        mask: torch.Tensor,         # (T,)
        orientation: str,
        *extra_tensors: torch.Tensor,
        refine_scores: Optional[torch.Tensor] = None,
    ):
        """
        Sort one sample and reorder any extra tensors consistently.

        Args:
            boxes: (T, 4)
            mask: (T,)
            orientation: "horizontal" or "vertical"
            extra_tensors: tensors with leading dimension T
            refine_scores: (T,) optional per-ROI quality logits, used only to
                gate which boxes anchor column/row geometry detection (see
                _confidence_anchor_mask) -- guidance input, not reordered as
                an output in its own right (callers that need it reordered
                should also pass it among extra_tensors, as existing callers
                already do).

        Returns:
            sorted_boxes, sorted_mask, *sorted_extras, sort_idx, col_ids

        col_ids: (T,) long, column/row membership of each box reindexed into
            sort order, -1 at padded/invalid positions (mirroring sort_batch's
            padding convention) -- callers distinguish valid positions via
            sorted_mask, same as for every other output here.
        """
        valid_t = self._valid_count(mask)
        if valid_t <= 1:
            sort_idx = torch.arange(boxes.size(0), device=boxes.device, dtype=torch.long)
            col_ids = torch.full_like(sort_idx, -1)
            if valid_t == 1:
                col_ids[0] = 0
            outputs = [boxes, mask]
            outputs.extend(extra_tensors)
            outputs.append(sort_idx)
            outputs.append(col_ids)
            return tuple(outputs)

        valid_boxes = boxes[:valid_t]
        valid_scores = refine_scores[:valid_t] if refine_scores is not None else None

        if orientation == "horizontal":
            valid_sort_idx, valid_col_ids = self._horizontal_sort_indices(valid_boxes, valid_scores)
        elif orientation == "vertical":
            valid_sort_idx, valid_col_ids = self._vertical_sort_indices(valid_boxes, valid_scores)
        else:
            raise ValueError(f"Unsupported orientation='{orientation}'. Use 'horizontal' or 'vertical'.")

        if valid_t < boxes.size(0):
            tail_idx = torch.arange(valid_t, boxes.size(0), device=boxes.device, dtype=torch.long)
            sort_idx = torch.cat([valid_sort_idx, tail_idx], dim=0)
            col_ids = torch.cat([valid_col_ids, torch.full_like(tail_idx, -1)], dim=0)
        else:
            sort_idx = valid_sort_idx
            col_ids = valid_col_ids

        sorted_boxes = boxes.index_select(0, sort_idx)
        sorted_mask = mask.index_select(0, sort_idx)

        outputs = [sorted_boxes, sorted_mask]
        for tensor in extra_tensors:
            outputs.append(tensor.index_select(0, sort_idx))
        outputs.append(sort_idx)
        outputs.append(col_ids)
        return tuple(outputs)

    def sort_batch(
        self,
        boxes: torch.Tensor,           # (B, T, 4)
        mask: torch.Tensor,            # (B, T)
        orientations: List[str],
        **named_tensors: torch.Tensor, # each (B, T, ...)
    ) -> dict:
        """
        Sort a batch of ROI sequences.

        Args:
            boxes: (B, T, 4)
            mask: (B, T)
            orientations: list of length B
            named_tensors:
                any additional tensors to be reordered along T, e.g.
                refined_feats=(B,T,D), refine_scores=(B,T), aux_logits=(B,T,V)

        Returns:
            dict with sorted tensors:
                boxes, mask, sort_indices, col_ids, plus all named tensors

        col_ids: (B, T) long, column/row membership of each box reindexed into
            sort order, -1 at padded positions -- lets a caller tell a mere
            column/row wrap apart from a genuine content change (see
            translation.py's block-break logic, which consumes this via
            run_inference()'s per-char "col_id" field).
        """
        bsz, t_max, _ = boxes.shape

        if len(orientations) != bsz:
            raise ValueError(
                f"orientations length {len(orientations)} does not match batch size {bsz}"
            )

        sorted_boxes = torch.zeros_like(boxes)
        sorted_mask = torch.zeros_like(mask)
        sort_indices = torch.zeros((bsz, t_max), device=boxes.device, dtype=torch.long)
        col_ids_batch = torch.full((bsz, t_max), -1, dtype=torch.long, device=boxes.device)
        isolation_masks = torch.zeros((bsz, t_max), dtype=torch.bool, device=boxes.device)
        furigana_masks  = torch.zeros((bsz, t_max), dtype=torch.bool, device=boxes.device)
        ordering_primary_mono = torch.ones((bsz,), device=boxes.device, dtype=boxes.dtype)
        ordering_primary_viol = torch.zeros((bsz,), device=boxes.device, dtype=boxes.dtype)
        ordering_valid_counts = mask.to(dtype=torch.long).sum(dim=1)

        sorted_named = {
            name: torch.zeros_like(tensor)
            for name, tensor in named_tensors.items()
        }

        refine_scores_all = named_tensors.get("refine_scores")

        for b in range(bsz):
            extras = [named_tensors[name][b] for name in named_tensors.keys()]
            refine_scores_b = refine_scores_all[b] if refine_scores_all is not None else None
            result = self.sort_single(
                boxes[b],
                mask[b],
                orientations[b],
                *extras,
                refine_scores=refine_scores_b,
            )

            sorted_boxes[b] = result[0]
            sorted_mask[b] = result[1]

            valid_t = self._valid_count(sorted_mask[b])
            if valid_t > 0:
                mono, viol = self._primary_axis_monotonic_fraction(
                    boxes_sorted_valid=sorted_boxes[b, :valid_t],
                    orientation=orientations[b],
                )
                ordering_primary_mono[b] = mono
                ordering_primary_viol[b] = viol

                iso_scores_b = (
                    refine_scores_b[result[-2]][:valid_t] if refine_scores_b is not None else None
                )
                iso, furi = self._compute_isolation_mask(
                    sorted_boxes[b, :valid_t], orientations[b], scores=iso_scores_b,
                )
                isolation_masks[b, :valid_t] = iso
                furigana_masks[b, :valid_t]  = furi

            offset = 2
            for i, name in enumerate(named_tensors.keys()):
                sorted_named[name][b] = result[offset + i]

            # sort_single now returns (..., sort_idx, col_ids) -- col_ids is the
            # new last element, sort_idx the one before it.
            sort_indices[b] = result[-2]
            col_ids_batch[b] = result[-1]

        out = {
            "boxes": sorted_boxes,
            "mask": sorted_mask,
            "sort_indices": sort_indices,
            "col_ids": col_ids_batch,
            "isolation_mask": isolation_masks,
            "furigana_mask":  furigana_masks,
            "ordering_diagnostics": {
                "primary_monotonic_fraction": ordering_primary_mono,
                "primary_violation_fraction": ordering_primary_viol,
                "valid_counts": ordering_valid_counts,
            },
        }
        out.update(sorted_named)
        return out