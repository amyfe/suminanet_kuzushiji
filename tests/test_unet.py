import torch
from model.kuronet.unet import FusionBlock, UNet


def test_fusionblock_shapes():
    ch = 8
    H, W = 64, 64
    high = torch.randn(1, ch, H, W)
    low = torch.randn(1, ch, H // 2, W // 2)
    fb = FusionBlock(ch)
    out = fb(high, low)
    assert out.shape == (1, ch, H, W)


def test_unet_forward_shape():
    model = UNet(in_channels=3, base_features=8)
    x = torch.randn(1, 3, 128, 128)
    y = model(x)
    # UNet returns tensor with base_features channels and same spatial dims
    assert y.shape == (1, 8, 128, 128)
