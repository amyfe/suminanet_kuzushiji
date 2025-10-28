"""Fine-grained glyph classifier for cropped glyph patches (DKNet-like).
This classifier can be used in stage-2 to refine low-confidence detections.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GlyphClassifier(nn.Module):
    def __init__(self, in_ch=3, n_classes=3000, base=32):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1), nn.BatchNorm2d(base), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base, base*2, 3, padding=1), nn.BatchNorm2d(base*2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(base*2, n_classes)

    def forward(self, x):
        f = self.backbone(x).view(x.size(0), -1)
        logits = self.fc(f)
        return logits