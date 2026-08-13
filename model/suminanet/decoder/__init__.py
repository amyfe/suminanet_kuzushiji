"""Decoder modules for sequence modeling."""
from .attention import SeqDecoderAttention, LuongAttention

__all__ = ["SeqDecoderAttention", "LuongAttention"]
