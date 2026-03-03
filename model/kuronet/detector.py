"""Detector head that turns UNet features into heatmaps and box regressions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


from .utils import make_gn


class DetectorHead(nn.Module):
    def __init__(self, in_ch, num_classes, extra_channels=64, predict_boxes=True, dropout_rate=0.3, predict_classes=False):
        super().__init__()
        self.predict_boxes = predict_boxes
        self.predict_classes = predict_classes
        
        # shared conv with dropout to reduce overfitting
        self.shared = nn.Sequential(
            nn.Conv2d(in_ch, extra_channels, kernel_size=3, padding=1),
            make_gn(extra_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
        )
        
        if predict_boxes:
            self.heatmap = nn.Conv2d(extra_channels, 1, kernel_size=1)
            self.bbox = nn.Conv2d(extra_channels, 4, kernel_size=1)
        
        # classification logits (per-pixel class logits) - optional to save memory
        if predict_classes:
            self.cls_logits = nn.Conv2d(extra_channels, num_classes, kernel_size=1)
        else:
            self.cls_logits = None

    def forward(self, feat):
        x = self.shared(feat)
        
        result = {}
        
        if self.cls_logits is not None:
            cls = self.cls_logits(x)
            result['cls'] = cls
        
        if self.predict_boxes:
            heat = torch.sigmoid(self.heatmap(x))
            bbox = self.bbox(x)  # raw regression
            result['heatmap'] = heat
            result['bbox'] = bbox
        
        return result


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