"""Reading order utilities for Japanese text.

Handles multiple reading directions:
- Horizontal (left to right)
- Vertical (top to bottom, typically right to left columns)
- Auto-detection based on layout analysis
"""
import torch
import numpy as np
from typing import List, Tuple, Optional


def sort_boxes_reading_order(
    boxes: torch.Tensor,
    classes: torch.Tensor,
    direction: str = "auto",
    column_width_threshold: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """Sort detected boxes according to reading order.
    
    Args:
        boxes: (N, 4) boxes [x1, y1, x2, y2] in image coordinates
        classes: (N,) class indices
        direction: Reading direction
                  - "horizontal": Left to right, top to bottom
                  - "vertical": Right to left, top to bottom (typical for Kuzushiji)
                  - "auto": Detect from layout
        column_width_threshold: For vertical, width of columns (relative to image)
    
    Returns:
        sorted_boxes: (N, 4) sorted boxes
        sorted_classes: (N,) sorted classes
        sort_indices: (N,) original indices in sorted order
    """
    if boxes.numel() == 0:
        return boxes, classes, []
    
    N = boxes.shape[0]
    device = boxes.device
    
    # Detect reading direction if auto
    if direction == "auto":
        direction = _detect_direction(boxes)
    
    if direction == "horizontal":
        indices = _sort_horizontal(boxes)
    elif direction == "vertical":
        indices = _sort_vertical(boxes)
    else:
        raise ValueError(f"Unknown direction: {direction}")
    
    sorted_boxes = boxes[indices]
    sorted_classes = classes[indices]
    
    return sorted_boxes, sorted_classes, indices.cpu().tolist()


def _detect_direction(boxes: torch.Tensor, height_threshold: float = 0.5) -> str:
    """Auto-detect reading direction from box layout.
    
    Heuristic: If boxes are arranged in vertical columns (high aspect ratio),
    assume vertical reading. Otherwise, horizontal.
    
    Args:
        boxes: (N, 4) boxes [x1, y1, x2, y2]
        height_threshold: If mean_height / mean_width > threshold, vertical
    
    Returns:
        "vertical" or "horizontal"
    """
    if boxes.numel() == 0:
        return "horizontal"
    
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    
    mean_height = heights.mean().item()
    mean_width = widths.mean().item()
    
    aspect_ratio = mean_height / (mean_width + 1e-6)
    
    # If characters are taller than wide, likely vertical text
    return "vertical" if aspect_ratio > height_threshold else "horizontal"


def _sort_horizontal(boxes: torch.Tensor) -> torch.Tensor:
    """Sort boxes left-to-right, top-to-bottom (reading order for horizontal text).
    
    Algorithm:
    1. Group boxes into rows (based on vertical position)
    2. Sort each row left-to-right
    3. Sort rows top-to-bottom
    
    Args:
        boxes: (N, 4) boxes [x1, y1, x2, y2]
    
    Returns:
        indices: (N,) sorted indices
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    
    # Get box centers and sizes
    y_centers = (boxes[:, 1] + boxes[:, 3]) / 2
    x_centers = (boxes[:, 0] + boxes[:, 2]) / 2
    heights = boxes[:, 3] - boxes[:, 1]
    
    # Group boxes into rows (within one box height)
    avg_height = heights.mean().item()
    device = boxes.device
    
    indices = torch.arange(boxes.shape[0], device=device)
    sorted_idx = []
    
    # Sort by approximate row
    row_assignments = (y_centers / (avg_height + 1e-6)).long()
    
    for row_id in sorted(row_assignments.unique().tolist()):
        row_mask = row_assignments == row_id
        row_indices = indices[row_mask]
        row_x = x_centers[row_mask]
        
        # Sort row left-to-right
        row_indices = row_indices[row_x.argsort()]
        sorted_idx.extend(row_indices.cpu().tolist())
    
    return torch.tensor(sorted_idx, dtype=torch.long, device=device)


def _sort_vertical(boxes: torch.Tensor) -> torch.Tensor:
    """Sort boxes for vertical (right-to-left) Japanese text.
    
    Algorithm:
    1. Group boxes into columns (based on horizontal position)
    2. Sort each column top-to-bottom
    3. Sort columns right-to-left
    
    Args:
        boxes: (N, 4) boxes [x1, y1, x2, y2]
    
    Returns:
        indices: (N,) sorted indices
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    
    # Get box centers and sizes
    x_centers = (boxes[:, 0] + boxes[:, 2]) / 2
    y_centers = (boxes[:, 1] + boxes[:, 3]) / 2
    widths = boxes[:, 2] - boxes[:, 0]
    
    # Group boxes into columns
    avg_width = widths.mean().item()
    device = boxes.device
    
    indices = torch.arange(boxes.shape[0], device=device)
    sorted_idx = []
    
    # Assign to columns
    col_assignments = (x_centers / (avg_width + 1e-6)).long()
    
    # Sort columns right-to-left (descending x)
    for col_id in sorted(col_assignments.unique().tolist(), reverse=True):
        col_mask = col_assignments == col_id
        col_indices = indices[col_mask]
        col_y = y_centers[col_mask]
        
        # Sort column top-to-bottom
        col_indices = col_indices[col_y.argsort()]
        sorted_idx.extend(col_indices.cpu().tolist())
    
    return torch.tensor(sorted_idx, dtype=torch.long, device=device)


def group_boxes_by_direction(
    boxes: torch.Tensor,
    direction: str = "auto"
) -> Tuple[List[List[int]], str]:
    """Group box indices by detected columns/rows.
    
    Useful for understanding layout structure.
    
    Args:
        boxes: (N, 4) boxes [x1, y1, x2, y2]
        direction: "horizontal", "vertical", or "auto"
    
    Returns:
        groups: List of box index lists (one per row/column)
        detected_direction: The direction that was used
    """
    if boxes.numel() == 0:
        return [], "horizontal"
    
    if direction == "auto":
        direction = _detect_direction(boxes)
    
    x_centers = (boxes[:, 0] + boxes[:, 2]) / 2
    y_centers = (boxes[:, 1] + boxes[:, 3]) / 2
    heights = boxes[:, 3] - boxes[:, 1]
    widths = boxes[:, 2] - boxes[:, 0]
    
    indices = torch.arange(boxes.shape[0])
    groups = []
    
    if direction == "horizontal":
        avg_height = heights.mean().item()
        row_assignments = (y_centers / (avg_height + 1e-6)).long()
        
        for row_id in sorted(row_assignments.unique().tolist()):
            row_mask = row_assignments == row_id
            row_indices = indices[row_mask].tolist()
            row_x = x_centers[row_mask]
            row_indices = [i for _, i in sorted(zip(row_x, row_indices))]
            groups.append(row_indices)
    
    else:  # vertical
        avg_width = widths.mean().item()
        col_assignments = (x_centers / (avg_width + 1e-6)).long()
        
        for col_id in sorted(col_assignments.unique().tolist(), reverse=True):
            col_mask = col_assignments == col_id
            col_indices = indices[col_mask].tolist()
            col_y = y_centers[col_mask]
            col_indices = [i for _, i in sorted(zip(col_y, col_indices))]
            groups.append(col_indices)
    
    return groups, direction
