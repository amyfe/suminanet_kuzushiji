# model/kuronet/encoder_wrapper.py
import torch
import torch.nn as nn
from typing import Tuple, Optional

class EncoderWrapper(nn.Module):
    def __init__(self, backbone: nn.Module, in_channels: int, enc_dim: int = 256):
        super().__init__()
        self.backbone = backbone
        self.proj = nn.Conv2d(in_channels, enc_dim, kernel_size=1)

    def forward(self, images: torch.Tensor, orientation: str = "horizontal", return_2d: bool = False):
        feats = self.backbone(images)       # (B, in_ch, Hf, Wf)
        feats = self.proj(feats)            # (B, enc_dim, Hf, Wf)
        if return_2d:
            return feats, None

        B, C, Hf, Wf = feats.shape
        if orientation == "horizontal":
            seq = feats.mean(dim=2).permute(0, 2, 1)  # (B, T=Wf, C)
            T = Wf
        elif orientation == "vertical":
            seq = feats.mean(dim=3).permute(0, 2, 1)  # (B, T=Hf, C)
            T = Hf
        else:
            raise ValueError("orientation must be 'horizontal' or 'vertical'")

        enc_mask = torch.ones(B, T, dtype=torch.bool, device=images.device)
        return seq, enc_mask
