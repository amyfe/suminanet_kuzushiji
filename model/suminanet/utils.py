"""Small utilities for suminanet models."""
from typing import Callable
import torch.nn as nn


def make_gn(channels: int, groups: int = 8) -> nn.Module:
    """Return a GroupNorm layer with number of groups chosen to divide channels.

    This ensures stability when using small batch sizes (GroupNorm is
    independent of batch statistics). The helper reduces `groups` until it
    exactly divides `channels`.
    """
    g = min(groups, channels)
    while g > 1 and (channels % g) != 0:
        g -= 1
    return nn.GroupNorm(g, channels)
