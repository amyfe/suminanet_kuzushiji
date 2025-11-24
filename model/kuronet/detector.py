"""Detector head that turns UNet features into heatmaps and box regressions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


from .utils import make_gn


class DetectorHead(nn.Module):
    def __init__(self, in_ch, num_classes, extra_channels=64):
        super().__init__()
        # shared conv
        self.shared = nn.Sequential(
            nn.Conv2d(in_ch, extra_channels, kernel_size=3, padding=1),
            make_gn(extra_channels),
            nn.ReLU(inplace=True),
        )
        # heatmap: per-class center heatmap (or single foreground map + classifier per-box)
        self.heatmap = nn.Conv2d(extra_channels, 1, kernel_size=1)
        # bbox regression (dx,dy,w,h)
        self.bbox = nn.Conv2d(extra_channels, 4, kernel_size=1)
        # classification logits (per-pixel class logits if desired), here we do a small per-box classifier instead
        self.cls_logits = nn.Conv2d(extra_channels, num_classes, kernel_size=1)

    def forward(self, feat):
        x = self.shared(feat)
        heat = torch.sigmoid(self.heatmap(x))
        bbox = self.bbox(x)  # raw regression
        cls = self.cls_logits(x)  # per-pixel class logits
        return {
            'heatmap': heat,
            'bbox': bbox,
            'cls': cls,
        }


# helper losses (simple versions)

def heatmap_loss(pred, gt):
    # GT is expected to be same shape; use MSE or focal-like loss
    return F.mse_loss(pred, gt)


def bbox_loss(pred, gt):
    return F.l1_loss(pred, gt)


def cls_loss(pred_logits, gt_labels):
    # pred_logits: (B, C, H, W), gt_labels: (B, H, W) with class ids or -1
    # Flatten and compute cross entropy on valid positions
    B, C, H, W = pred_logits.shape
    logits = pred_logits.permute(0,2,3,1).reshape(-1, C)
    labels = gt_labels.reshape(-1)
    valid = labels >= 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=pred_logits.device)
    return F.cross_entropy(logits[valid], labels[valid])