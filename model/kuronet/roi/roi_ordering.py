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

# model/kuronet/roi/roi_ordering.py

from __future__ import annotations

from math import isfinite
from typing import List

import torch


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

    def __init__(self, line_merge_thresh_ratio: float = 0.6):
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

    def _horizontal_sort_indices(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Horizontal reading order using gap-based row detection.

        1. Sort boxes by y-center (top to bottom).
        2. Detect natural row breaks: gaps > 1.5 × 75th-percentile height.
           Using the 75th percentile rather than the median makes this robust
           to many small FP boxes (illustrations) that would pull the median down.
        3. Sort rows top-to-bottom; sort within each row left-to-right.
        """
        if boxes.numel() == 0:
            return torch.zeros((0,), dtype=torch.long, device=boxes.device)

        y_center = (boxes[:, 1] + boxes[:, 3]) * 0.5
        heights = (boxes[:, 3] - boxes[:, 1]).clamp_min(1.0)

        h75 = self._percentile_size(heights, 0.75)
        gap_thresh = 1.5 * h75

        sorted_yc, sorted_idx = y_center.sort(descending=False)

        if sorted_yc.numel() > 1:
            gaps = sorted_yc[1:] - sorted_yc[:-1]
            is_break = torch.cat([
                torch.ones(1, dtype=torch.bool, device=boxes.device),
                gaps > gap_thresh,
            ])
        else:
            is_break = torch.ones(1, dtype=torch.bool, device=boxes.device)

        row_ids = is_break.long().cumsum(0) - 1
        num_rows = int(row_ids.max().item()) + 1

        sorted_indices: List[int] = []
        for row_id in range(num_rows):
            in_row = (row_ids == row_id)
            orig_indices = sorted_idx[in_row]
            x_left = boxes[orig_indices, 0]
            within_row = orig_indices[x_left.argsort(descending=False)]
            sorted_indices.extend(within_row.tolist())

        return torch.tensor(sorted_indices, device=boxes.device, dtype=torch.long)

    def _vertical_sort_indices(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Vertical reading order using gap-based column detection.

        1. Sort boxes by x-center (left to right).
        2. Detect natural column breaks: gaps > 1.5 × 75th-percentile width.
           The 75th-percentile threshold is more robust to small FP boxes that
           would pull the median width down and merge distinct columns.
        3. Sort columns right-to-left (Edo reading direction).
        4. Sort within each column top-to-bottom.
        """
        if boxes.numel() == 0:
            return torch.zeros((0,), dtype=torch.long, device=boxes.device)

        x_center = (boxes[:, 0] + boxes[:, 2]) * 0.5
        widths = (boxes[:, 2] - boxes[:, 0]).clamp_min(1.0)

        w75 = self._percentile_size(widths, 0.75)
        gap_thresh = 1.5 * w75

        sorted_xc, sorted_idx = x_center.sort(descending=False)  # left → right

        if sorted_xc.numel() > 1:
            gaps = sorted_xc[1:] - sorted_xc[:-1]
            is_break = torch.cat([
                torch.ones(1, dtype=torch.bool, device=boxes.device),
                gaps > gap_thresh,
            ])
        else:
            is_break = torch.ones(1, dtype=torch.bool, device=boxes.device)

        col_ids = is_break.long().cumsum(0) - 1
        num_cols = int(col_ids.max().item()) + 1

        sorted_indices: List[int] = []
        for col_id in range(num_cols - 1, -1, -1):  # right → left
            in_col = (col_ids == col_id)
            orig_indices = sorted_idx[in_col]
            y_top = boxes[orig_indices, 1]
            within_col = orig_indices[y_top.argsort(descending=False)]
            sorted_indices.extend(within_col.tolist())

        return torch.tensor(sorted_indices, device=boxes.device, dtype=torch.long)

    def _compute_isolation_mask(
        self,
        boxes: torch.Tensor,    # (N, 4)  valid sorted boxes
        orientation: str,
        min_col_size: int = 3,
        size_ratio: float = 3.0,
    ) -> torch.Tensor:
        """
        Mark boxes that are likely illustration false-positives.

        A box is isolated (True) when its column/row has fewer than
        min_col_size members OR its area exceeds size_ratio × the column
        median area.  Uses the same gap-based grouping as the sort methods.
        """
        n = boxes.size(0)
        if n == 0:
            return torch.zeros((0,), dtype=torch.bool, device=boxes.device)

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

        isolated = torch.zeros(n, dtype=torch.bool, device=boxes.device)
        for g in range(num_groups):
            in_group = group_ids == g
            count    = int(in_group.sum().item())
            if count < min_col_size:
                isolated[in_group] = True
            else:
                med_area = float(areas[in_group].median().item())
                isolated[in_group] = isolated[in_group] | (areas[in_group] > size_ratio * med_area)

        return isolated

    def sort_single(
        self,
        boxes: torch.Tensor,        # (T, 4)
        mask: torch.Tensor,         # (T,)
        orientation: str,
        *extra_tensors: torch.Tensor,
    ):
        """
        Sort one sample and reorder any extra tensors consistently.

        Args:
            boxes: (T, 4)
            mask: (T,)
            orientation: "horizontal" or "vertical"
            extra_tensors: tensors with leading dimension T

        Returns:
            sorted_boxes, sorted_mask, *sorted_extras, sort_idx
        """
        valid_t = self._valid_count(mask)
        if valid_t <= 1:
            sort_idx = torch.arange(boxes.size(0), device=boxes.device, dtype=torch.long)
            outputs = [boxes, mask]
            outputs.extend(extra_tensors)
            outputs.append(sort_idx)
            return tuple(outputs)

        valid_boxes = boxes[:valid_t]

        if orientation == "horizontal":
            valid_sort_idx = self._horizontal_sort_indices(valid_boxes)
        elif orientation == "vertical":
            valid_sort_idx = self._vertical_sort_indices(valid_boxes)
        else:
            raise ValueError(f"Unsupported orientation='{orientation}'. Use 'horizontal' or 'vertical'.")

        if valid_t < boxes.size(0):
            tail_idx = torch.arange(valid_t, boxes.size(0), device=boxes.device, dtype=torch.long)
            sort_idx = torch.cat([valid_sort_idx, tail_idx], dim=0)
        else:
            sort_idx = valid_sort_idx

        sorted_boxes = boxes.index_select(0, sort_idx)
        sorted_mask = mask.index_select(0, sort_idx)

        outputs = [sorted_boxes, sorted_mask]
        for tensor in extra_tensors:
            outputs.append(tensor.index_select(0, sort_idx))
        outputs.append(sort_idx)
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
                boxes, mask, sort_indices, plus all named tensors
        """
        bsz, t_max, _ = boxes.shape

        if len(orientations) != bsz:
            raise ValueError(
                f"orientations length {len(orientations)} does not match batch size {bsz}"
            )

        sorted_boxes = torch.zeros_like(boxes)
        sorted_mask = torch.zeros_like(mask)
        sort_indices = torch.zeros((bsz, t_max), device=boxes.device, dtype=torch.long)
        isolation_masks = torch.zeros((bsz, t_max), dtype=torch.bool, device=boxes.device)
        ordering_primary_mono = torch.ones((bsz,), device=boxes.device, dtype=boxes.dtype)
        ordering_primary_viol = torch.zeros((bsz,), device=boxes.device, dtype=boxes.dtype)
        ordering_valid_counts = mask.to(dtype=torch.long).sum(dim=1)

        sorted_named = {
            name: torch.zeros_like(tensor)
            for name, tensor in named_tensors.items()
        }

        for b in range(bsz):
            extras = [named_tensors[name][b] for name in named_tensors.keys()]
            result = self.sort_single(
                boxes[b],
                mask[b],
                orientations[b],
                *extras,
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

                iso = self._compute_isolation_mask(sorted_boxes[b, :valid_t], orientations[b])
                isolation_masks[b, :valid_t] = iso

            offset = 2
            for i, name in enumerate(named_tensors.keys()):
                sorted_named[name][b] = result[offset + i]

            sort_indices[b] = result[-1]

        out = {
            "boxes": sorted_boxes,
            "mask": sorted_mask,
            "sort_indices": sort_indices,
            "isolation_mask": isolation_masks,
            "ordering_diagnostics": {
                "primary_monotonic_fraction": ordering_primary_mono,
                "primary_violation_fraction": ordering_primary_viol,
                "valid_counts": ordering_valid_counts,
            },
        }
        out.update(sorted_named)
        return out