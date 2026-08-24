"""
Shared letterbox geometry helpers (PIL-space, pre-tensor).

Letterboxing preserves aspect ratio by uniformly scaling an image to fit inside
a square canvas, then padding the remainder, instead of stretching width/height
independently. Used by both training data loading (utils/__init__.py) and
inference (model/translation/infer.py) so the two stay pixel-consistent.
"""

from PIL import Image


def letterbox_pil(
    image: Image.Image,
    target_size: int,
    resample: int = Image.LANCZOS,
    fill=(128, 128, 128),
) -> tuple[Image.Image, float, tuple[int, int]]:
    """
    Uniform-scale + pad `image` onto a target_size x target_size canvas.

    Returns
    -------
    canvas : the letterboxed image
    scale  : uniform scale factor applied to the original image
    pad    : (pad_x, pad_y) — pixel offset of the resized image inside the canvas
    """
    orig_w, orig_h = image.size
    scale = target_size / max(orig_w, orig_h)
    new_w, new_h = round(orig_w * scale), round(orig_h * scale)
    resized = image.resize((new_w, new_h), resample)
    pad_x, pad_y = (target_size - new_w) // 2, (target_size - new_h) // 2
    canvas = Image.new(image.mode, (target_size, target_size), fill)
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, scale, (pad_x, pad_y)


def letterbox_boxes_forward(boxes, scale: float, pad: tuple[int, int]):
    """Original-image xyxy boxes -> letterboxed-canvas xyxy boxes."""
    pad_x, pad_y = pad
    return [
        [x1 * scale + pad_x, y1 * scale + pad_y, x2 * scale + pad_x, y2 * scale + pad_y]
        for (x1, y1, x2, y2) in boxes
    ]


def unletterbox_boxes(
    chars: list[dict],
    orig_size: tuple[int, int],
    scale: float,
    pad: tuple[int, int],
) -> None:
    """Convert boxes in-place from letterboxed model coords to original image coords."""
    orig_w, orig_h = orig_size
    pad_x, pad_y = pad
    for c in chars:
        x1, y1, x2, y2 = c["box"]
        x1 = max(0.0, min((x1 - pad_x) / scale, float(orig_w)))
        y1 = max(0.0, min((y1 - pad_y) / scale, float(orig_h)))
        x2 = max(0.0, min((x2 - pad_x) / scale, float(orig_w)))
        y2 = max(0.0, min((y2 - pad_y) / scale, float(orig_h)))
        c["box"] = [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
