# model/kuronet/backbone/roi_crop_encoder.py
"""Pretrained EfficientNet-B0 applied to raw image crops per ROI.

This mirrors Clanuwat's VGG-16 role: the detection backbone (UNet) localises
characters at stride=1, then a separate ImageNet-pretrained network classifies
each character crop.  We use EfficientNet-B0 (smaller than B2, faster, still
ImageNet pretrained) rather than VGG-16.

Input
    images:       (B, 3, H, W)  original images, not feature maps
    roi_boxes:    (B, T, 4)     xyxy boxes in image coordinates (padded)
    roi_mask:     (B, T)        bool, True = valid ROI

Output
    crop_feats:   (B, T, out_dim)  per-ROI features from pretrained network
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align as tv_roi_align


class ROICropEncoder(nn.Module):
    """
    Extracts per-ROI features by applying a pretrained EfficientNet-B0
    to image-space crops (not feature-map crops).

    The crops are extracted with RoIAlign at a fixed spatial size, then
    passed through a frozen (or optionally fine-tuned) EfficientNet encoder,
    global-average-pooled, and projected to out_dim.

    Args:
        out_dim:        output feature dimension (projected to match roi_feat_dim)
        crop_size:      spatial size fed into EfficientNet (height, width)
        freeze_encoder: if True, EfficientNet weights are not updated during training
        pretrained:     if True, load ImageNet weights via timm
    """

    # EfficientNet-B0 last-stage channel count (after MBConv stage 7)
    _EFF_CHANNELS = 1280

    def __init__(
        self,
        out_dim: int = 384,
        crop_size: Tuple[int, int] = (48, 48),
        freeze_encoder: bool = True,
        pretrained: bool = True,
    ):
        super().__init__()

        try:
            import timm
        except ImportError as e:
            raise ImportError(
                "ROICropEncoder requires timm. Install with: pip install timm"
            ) from e

        self.crop_size = crop_size
        self.freeze_encoder = bool(freeze_encoder)

        # EfficientNet-B0 as a feature extractor — strip classifier head
        self.encoder = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,       # remove classification head
            global_pool="avg",   # global avg pool -> (B, 1280)
        )

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

        out_linear = nn.Linear(self._EFF_CHANNELS, out_dim)
        nn.init.normal_(out_linear.weight, std=0.01)
        nn.init.zeros_(out_linear.bias)
        self.out_proj = nn.Sequential(
            out_linear,
            nn.ReLU(inplace=True),
        )

        self.out_dim = out_dim

    def _extract_crops(
        self,
        images: torch.Tensor,   # (B, 3, H, W)
        roi_boxes: torch.Tensor, # (B, T, 4) xyxy image coords
        roi_mask: torch.Tensor,  # (B, T) bool
    ) -> torch.Tensor:
        """
        Use RoIAlign to extract fixed-size crops from the original images.

        Returns:
            crops: (N_valid, 3, Hc, Wc)  where N_valid = roi_mask.sum()
        """
        bsz, t_max, _ = roi_boxes.shape
        Hc, Wc = self.crop_size

        # Gather valid boxes across the batch
        valid_mask_flat = roi_mask.reshape(-1)                           # (B*T,)
        boxes_flat      = roi_boxes.reshape(-1, 4)                       # (B*T, 4)
        # Batch index column: integer-valued, tiled as [0…0, 1…1, …]
        batch_idx_flat  = torch.arange(bsz, device=roi_boxes.device) \
                              .unsqueeze(1).expand(bsz, t_max).reshape(-1)  # (B*T,) int64

        valid_boxes = boxes_flat[valid_mask_flat]        # (N_valid, 4)
        valid_batch = batch_idx_flat[valid_mask_flat]    # (N_valid,) int64

        if valid_boxes.numel() == 0:
            return images.new_zeros((0, 3, Hc, Wc))

        # tv_roi_align requires the rois tensor to be float32 regardless of input dtype.
        # Cast batch index and boxes explicitly — roi_boxes may arrive as float16 under AMP.
        rois = torch.cat(
            [valid_batch.unsqueeze(1).float(), valid_boxes.float()], dim=1
        )  # (N_valid, 5) float32

        h_img, w_img = images.shape[-2:]
        spatial_scale = 1.0  # boxes are in image pixel coords

        # tv_roi_align requires float32; explicit cast so AMP bfloat16 inputs don't silently produce wrong-dtype crops
        crops = tv_roi_align(
            images.to(dtype=torch.float32),
            rois,
            output_size=(Hc, Wc),
            spatial_scale=spatial_scale,
            aligned=True,
        )  # (N_valid, 3, Hc, Wc)

        return crops

    def train(self, mode: bool = True) -> "ROICropEncoder":
        """Keep the encoder in eval mode when frozen so BN running stats don't drift."""
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def forward(
        self,
        images: torch.Tensor,    # (B, 3, H, W)
        roi_boxes: torch.Tensor, # (B, T, 4) xyxy image coords
        roi_mask: torch.Tensor,  # (B, T) bool
    ) -> torch.Tensor:
        """
        Returns:
            crop_feats: (B, T, out_dim)  zero for padding positions
        """
        bsz, t_max, _ = roi_boxes.shape

        # Compute scatter indices from the same mask snapshot used by _extract_crops.
        # Reading roi_mask once here guarantees the (b_valid, t_valid) indices are
        # consistent with the N_valid crops that _extract_crops will extract.
        valid_mask_flat = roi_mask.reshape(-1)   # (B*T,)
        b_idx = torch.arange(bsz, device=images.device).unsqueeze(1).expand(bsz, t_max).reshape(-1)
        t_idx = torch.arange(t_max, device=images.device).unsqueeze(0).expand(bsz, t_max).reshape(-1)
        b_valid = b_idx[valid_mask_flat]   # (N_valid,)
        t_valid = t_idx[valid_mask_flat]   # (N_valid,)

        crops = self._extract_crops(images, roi_boxes, roi_mask)  # (N_valid, 3, Hc, Wc)

        if crops.size(0) == 0:
            return images.new_zeros((bsz, t_max, self.out_dim))

        # Training images are already ImageNet-normalized (T.ToTensor + T.Normalize).
        # RoIAlign on normalized images produces normalized crops — pass directly.
        crops_norm = crops

        if self.freeze_encoder:
            with torch.no_grad():
                enc_feats = self.encoder(crops_norm)  # (N_valid, 1280)
        else:
            enc_feats = self.encoder(crops_norm)

        proj_feats = self.out_proj(enc_feats)  # (N_valid, out_dim)

        out = images.new_zeros((bsz, t_max, self.out_dim))
        out[b_valid, t_valid] = proj_feats.to(out.dtype)
        return out
