"""kuronet subpackage exports.

For attention-based transcription, you need:
- UNet (backbone)
- EncoderWrapper (wraps UNet for sequence tasks)
- decoder.attention.SeqDecoderAttention

DetectorHead and GlyphClassifier are kept for detection tasks if needed.
"""
from .unet import UNet
from .encoder_wrapper import EncoderWrapper
from .detector import DetectorHead
from .roi_sequence import ROISequenceEncoder, ROIContextEncoder

__all__ = ["UNet", "EncoderWrapper", "DetectorHead", "ROISequenceEncoder", "ROIContextEncoder"]
    