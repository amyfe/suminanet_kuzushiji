"""Backbone factory for Stage 1 + Stage 2 training.

Supported types:
    'unet'            -- UNet built from scratch (current default, no pretrained weights)
    'efficientnet_b2' -- EfficientNet-B2 encoder + FPN decoder, ImageNet pretrained

Usage:
    from model.kuronet.backbone import build_backbone
    backbone = build_backbone(BACKBONE_TYPE, BACKBONE_BASE_FEATURES)
"""

from .feature_projector import FeatureProjector


def build_backbone(backbone_type: str, base_features: int, pretrained: bool = True):
    """
    Build and return a backbone module.

    Both backbone types share the same output interface:
        images (B, 3, H, W) -> features (B, base_features, H', W')

    For 'unet':           H' = H,   W' = W   (stride 1)
    For 'efficientnet_b2': H' = H//2, W' = W//2 (stride 2 — 4× less memory)

    Downstream code (DetectorHead, ROI align, proposal decoding) adapts
    automatically because they compute spatial scale from feature map shape.

    Args:
        backbone_type: 'unet' or 'efficientnet_b2'
        base_features: number of output channels
        pretrained: load ImageNet weights for EfficientNet (ignored for UNet)
    """
    t = backbone_type.lower().replace("-", "_")

    if t == "unet":
        from model.kuronet.unet import UNet
        return UNet(in_channels=3, base_features=base_features)

    elif t == "efficientnet_b2":
        from .efficientnet_backbone import EfficientNetBackbone
        return EfficientNetBackbone(out_channels=base_features, pretrained=pretrained)

    else:
        raise ValueError(
            f"Unknown backbone_type {backbone_type!r}. "
            "Supported: 'unet', 'efficientnet_b2'."
        )


__all__ = ["build_backbone", "FeatureProjector"]
