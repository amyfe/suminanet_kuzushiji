"""Nimmt die 2D-Featuremap des U-Net und projiziert sie in eine saubere gemeinsame Feature-Dimension für Stage 2.
Kein 1D-Zweig, keine Orientierung oder Sequenzlogik, nur reine 2D-Feature-Projektion.
Neuer EncoderWrapper
Input:

(B, C_in, Hf, Wf)

Output:

(B, C_proj, Hf, Wf)

"""
from torch import nn
import torch
from model.kuronet.utils import make_gn

class FeatureProjector(nn.Module):
    """Project backbone feature maps into a shared Stage-2 feature space.

    Input:
        feat_2d: (B, C_in, Hf, Wf)

    Output:
        proj_2d: (B, C_out, Hf, Wf)

    Notes:
    - This module is intentionally simple.
    - It replaces the old EncoderWrapper logic for Option C.
    - No sequence conversion happens here."""
    def __init__(self, 
        in_channels: int, 
        out_channels: int = 256,
        hidden_channels: int | None = None,
        dropout: float = 0.1):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = out_channels
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            make_gn(hidden_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1, bias=False),
            make_gn(out_channels),
            nn.ReLU(inplace=True),
            
            nn.Dropout(dropout),
        )
        self.out_channels = out_channels

    def forward(self, feat_2d: torch.Tensor) -> torch.Tensor:
        if feat_2d.dim() != 4:
            raise ValueError(
                f"FeatureProjector expected input shape (B, C, H, W), got {tuple(feat_2d.shape)}"
            )
        return self.net(feat_2d)