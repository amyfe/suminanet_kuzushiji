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

from typing import List

import torch


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

    def _horizontal_sort_indices(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Horizontal reading order:
            1. group into rows by y-center proximity
            2. sort rows top-to-bottom
            3. sort within each row left-to-right
        """
        if boxes.numel() == 0:
            return torch.zeros((0,), dtype=torch.long, device=boxes.device)

        y_center = (boxes[:, 1] + boxes[:, 3]) * 0.5
        x_left = boxes[:, 0]
        heights = (boxes[:, 3] - boxes[:, 1]).clamp_min(1.0)

        median_h = heights.median()
        row_thresh = self.line_merge_thresh_ratio * median_h

        order_y = torch.argsort(y_center, descending=False)
        rows: List[List[int]] = []

        current_row: List[int] = []
        current_row_y = None

        for idx_t in order_y.tolist():
            y = float(y_center[idx_t].item())
            if current_row_y is None:
                current_row = [idx_t]
                current_row_y = y
                continue

            if abs(y - current_row_y) <= float(row_thresh.item()):
                current_row.append(idx_t)
                current_row_y = (current_row_y * (len(current_row) - 1) + y) / len(current_row)
            else:
                rows.append(current_row)
                current_row = [idx_t]
                current_row_y = y

        if current_row:
            rows.append(current_row)

        sorted_indices: List[int] = []
        for row in rows:
            row_tensor = torch.tensor(row, device=boxes.device, dtype=torch.long)
            row_x = x_left[row_tensor]
            row_order = row_tensor[torch.argsort(row_x, descending=False)]
            sorted_indices.extend(row_order.tolist())

        return torch.tensor(sorted_indices, device=boxes.device, dtype=torch.long)

    def _vertical_sort_indices(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Vertical reading order:
            1. group into columns by x-center proximity
            2. sort columns right-to-left
            3. sort within each column top-to-bottom
        """
        if boxes.numel() == 0:
            return torch.zeros((0,), dtype=torch.long, device=boxes.device)

        x_center = (boxes[:, 0] + boxes[:, 2]) * 0.5
        y_top = boxes[:, 1]
        widths = (boxes[:, 2] - boxes[:, 0]).clamp_min(1.0)

        median_w = widths.median()
        col_thresh = self.line_merge_thresh_ratio * median_w

        order_x = torch.argsort(x_center, descending=True)  # right-to-left
        cols: List[List[int]] = []

        current_col: List[int] = []
        current_col_x = None

        for idx_t in order_x.tolist():
            x = float(x_center[idx_t].item())
            if current_col_x is None:
                current_col = [idx_t]
                current_col_x = x
                continue

            if abs(x - current_col_x) <= float(col_thresh.item()):
                current_col.append(idx_t)
                current_col_x = (current_col_x * (len(current_col) - 1) + x) / len(current_col)
            else:
                cols.append(current_col)
                current_col = [idx_t]
                current_col_x = x

        if current_col:
            cols.append(current_col)

        sorted_indices: List[int] = []
        for col in cols:
            col_tensor = torch.tensor(col, device=boxes.device, dtype=torch.long)
            col_y = y_top[col_tensor]
            col_order = col_tensor[torch.argsort(col_y, descending=False)]
            sorted_indices.extend(col_order.tolist())

        return torch.tensor(sorted_indices, device=boxes.device, dtype=torch.long)

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

            offset = 2
            for i, name in enumerate(named_tensors.keys()):
                sorted_named[name][b] = result[offset + i]

            sort_indices[b] = result[-1]

        out = {
            "boxes": sorted_boxes,
            "mask": sorted_mask,
            "sort_indices": sort_indices,
        }
        out.update(sorted_named)
        return out