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
    def __init__(self, ch):
        super().__init__()
        # We don't know the incoming channel sizes for `high` and `low`
        # at init time (they may differ). Create the 1x1 conv lazily
        # on first forward pass based on the concatenated channel size.
        self.conv: nn.Module | None = None
        self.out_ch = ch
        self.gn: nn.Module | None = None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, high, low):
        # Upsample 'low' to same spatial size as 'high'
        low_up = F.interpolate(low, size=high.shape[2:], mode="bilinear", align_corners=False)
        fused = torch.cat([high, low_up], dim=1)
        # create conv + gn lazily to handle differing input channel sizes
        if self.conv is None:
            in_ch = fused.size(1)
            self.conv = nn.Conv2d(in_ch, self.out_ch, 1, bias=False)
            self.gn = make_gn(self.out_ch)
            # move created modules to the same device as fused
            self.conv.to(fused.device)
            self.gn.to(fused.device)
        out = self.conv(fused)
        assert self.gn is not None
        out = self.gn(out)
        return self.relu(out)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.block = ConvBlock(in_ch, out_ch)
        self.fusion = FusionBlock(out_ch)

    def forward(self, x, skip_main, skip_aux=None):
        x = self.up(x)
        if skip_aux is not None:
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

        self.up3 = Up(f * 16, f * 8)
        self.up2 = Up(f * 8, f * 4)
        self.up1 = Up(f * 4, f * 2)
        self.up0 = Up(f * 2, f)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        xm = self.mid(x4)

        u3 = self.up3(xm, x4)
        u2 = self.up2(u3, x3, skip_aux=x2)   # fuse with shallower encoder feature
        u1 = self.up1(u2, x2, skip_aux=x1)
        u0 = self.up0(u1, x1)
        return u0