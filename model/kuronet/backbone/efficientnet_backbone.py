"""EfficientNet-B2 encoder + FPN decoder backbone.

Outputs a feature map at stride=2 (H//2, W//2) with `out_channels` channels.
Pretrained on ImageNet via timm.

Interface matches the existing UNet: backbone(images) -> (B, out_channels, H', W').
All downstream code (DetectorHead, proposal extraction, ROI align) adapts
automatically because they compute spatial scale from feature map shape.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EfficientNetBackbone(nn.Module):
    """
    EfficientNet-B2 encoder with FPN top-down decoder.

    Output is at stride=2 relative to the input (H//2, W//2), giving
    a good balance of spatial resolution and memory use.

    EfficientNet-B2 encoder channels: [16, 24, 48, 120, 352]
    at strides: [2, 4, 8, 16, 32].
    The FPN merges from deepest to shallowest, ending at the stride-2 level.
    """

    _ENC_CHANNELS = [16, 24, 48, 120, 352]
    _FPN_DIM = 128

    def __init__(self, out_channels: int = 64, pretrained: bool = True):
        super().__init__()

        import timm  # local import so the class can be defined without timm installed

        self.encoder = timm.create_model(
            "efficientnet_b2",
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
        )

        fpn_dim = self._FPN_DIM
        enc_channels = self._ENC_CHANNELS

        # Lateral 1×1 convs: project each encoder level to fpn_dim
        self.lat = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, fpn_dim, 1, bias=False),
                nn.GroupNorm(8, fpn_dim),
                nn.ReLU(inplace=True),
            )
            for c in enc_channels
        ])

        # Top-down merge convs (one per transition: P5→P4, P4→P3, P3→P2, P2→P1)
        self.merge = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1, bias=False),
                nn.GroupNorm(8, fpn_dim),
                nn.ReLU(inplace=True),
            )
            for _ in range(4)
        ])

        # Project FPN output (at stride=2) to the requested out_channels
        num_groups = min(8, out_channels)
        self.out_proj = nn.Sequential(
            nn.Conv2d(fpn_dim, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True),
        )

        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)

        Returns:
            (B, out_channels, H//2, W//2)
        """
        feats = self.encoder(x)  # list[5] at strides [2, 4, 8, 16, 32]

        # Apply lateral convs to each encoder level
        lats = [self.lat[i](feats[i]) for i in range(5)]

        # Top-down FPN path: start at P5 (stride=32), propagate to P1 (stride=2)
        p = lats[4]  # P5
        for i in range(3, -1, -1):
            p = F.interpolate(p, size=lats[i].shape[2:], mode="nearest")
            p = p + lats[i]
            merge_idx = 3 - i  # 0 for i=3 (→P4), ..., 3 for i=0 (→P1)
            p = self.merge[merge_idx](p)

        # p is now at stride=2 (256×256 for 512×512 input)
        return self.out_proj(p)
