import torch
import torch.nn.functional as F

def heatmap_loss(pred, target):
    return F.mse_loss(pred, target)

def bbox_loss(pred, target):
    return F.l1_loss(pred, target)

def class_loss(pred, target):
    if pred.dim() == 4:
        pred = pred.mean(dim=(2,3))
    return F.cross_entropy(pred, target)

def seq_loss(pred, targets):
    return F.cross_entropy(pred.view(-1, pred.size(-1)), targets.view(-1))
