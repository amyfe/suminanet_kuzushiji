"""U-Net backbone with residual blocks (PyTorch).
A reasonably compact implementation to serve as encoder/decoder.
"""
import torch
import torch.nn as nn
from .utils import make_gn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            make_gn(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            make_gn(out_ch),
        )
        self.relu = nn.ReLU(inplace=True)
        self.residual_conv = None
        if in_ch != out_ch:
            self.residual_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(x)
        res = x if self.residual_conv is None else self.residual_conv(x)
        return self.relu(out + res)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = ConvBlock(in_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)
        return self.block(x)
    
class FusionBlock(nn.Module):
    """
    Fuse two feature maps (high-res skip and lower-res aux skip) deterministically.
    - high: (B, ch_high, H, W)
    - low:  (B, ch_low,  h, w)  -> upsampled to (H,W)
    Output: (B, out_ch, H, W)
    """
    def __init__(self, ch_high: int, ch_low: int, out_ch: int):
        super().__init__()
        in_ch = ch_high + ch_low
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.gn = make_gn(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, high, low):
        low_up = F.interpolate(low, size=high.shape[2:], mode="bilinear", align_corners=False)
        fused = torch.cat([high, low_up], dim=1)
        out = self.conv(fused)
        out = self.gn(out)
        return self.relu(out)


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, skip_main_ch: int, skip_aux_ch: int | None = None):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

        self.use_fusion = skip_aux_ch is not None
        if self.use_fusion:
            self.fusion = FusionBlock(ch_high=skip_main_ch, ch_low=skip_aux_ch, out_ch=out_ch)
            fused_skip_ch = out_ch
        else:
            self.fusion = None
            fused_skip_ch = skip_main_ch
        self.block = ConvBlock(out_ch + fused_skip_ch, out_ch)

    def forward(self, x, skip_main, skip_aux=None):
        x = self.up(x)

        if self.use_fusion:
            if skip_aux is None:
                raise ValueError("Up block expects skip_aux but got None.")
            skip_main = self.fusion(skip_main, skip_aux)

        # Match shapes if needed
        diffY = skip_main.size(2) - x.size(2)
        diffX = skip_main.size(3) - x.size(3)
        if diffY != 0 or diffX != 0:
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])

        x = torch.cat([x, skip_main], dim=1)
        return self.block(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, base_features=32):
        super().__init__()
        f = base_features
        self.inc = ConvBlock(in_channels, f)
        self.down1 = Down(f, f * 2)
        self.down2 = Down(f * 2, f * 4)
        self.down3 = Down(f * 4, f * 8)

        self.mid = ConvBlock(f * 8, f * 16)

        # Up(in_ch, out_ch, skip_main_ch, skip_aux_ch?)
        self.up3 = Up(f * 16, f * 8,  skip_main_ch=f * 8,  skip_aux_ch=None)
        self.up2 = Up(f * 8,  f * 4,  skip_main_ch=f * 4,  skip_aux_ch=f * 2)
        self.up1 = Up(f * 4,  f * 2,  skip_main_ch=f * 2,  skip_aux_ch=f)
        self.up0 = Up(f * 2,  f,      skip_main_ch=f,      skip_aux_ch=None)

    def forward(self, x):
        x1 = self.inc(x)      # f
        x2 = self.down1(x1)   # 2f
        x3 = self.down2(x2)   # 4f
        x4 = self.down3(x3)   # 8f

        xm = self.mid(x4)     # 16f

        u3 = self.up3(xm, x4)
        u2 = self.up2(u3, x3, skip_aux=x2)
        u1 = self.up1(u2, x2, skip_aux=x1)
        u0 = self.up0(u1, x1)
        return u0