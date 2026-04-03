"""kuronet subpackage exports.

For attention-based transcription, you need:
- UNet (backbone)
- EncoderWrapper (wraps UNet for sequence tasks)
- decoder.attention.SeqDecoderAttention

DetectorHead and GlyphClassifier are kept for detection tasks if needed.
"""
from .unet import UNet
from .detector import DetectorHead
from .roi.roi_sequence_deprecated import ROISequenceEncoder
from .context.roi_context import ROIContextEncoder
from .decoder.attention import SeqDecoderAttention
from .roi.roi_tokens import ROITokenProjector
from .roi.roi_sequence_deprecated import ROISequenceEncoder
from .roi.roi_tokens import ROITokenProjector
from .backbone.feature_projector import FeatureProjector

__all__ = ["UNet", "DetectorHead", "ROISequenceEncoder", "ROIContextEncoder", "SeqDecoderAttention", "ROITokenProjector", "FeatureProjector"]