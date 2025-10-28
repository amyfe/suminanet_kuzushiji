import random
import numpy as np
import torch

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def nms_boxes(boxes, scores, iou_threshold=0.5):
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long)
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2-x1)*(y2-y1)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        xx1 = torch.max(x1[i], x1[rest])
        yy1 = torch.max(y1[i], y1[rest])
        xx2 = torch.min(x2[i], x2[rest])
        yy2 = torch.min(y2[i], y2[rest])
        w = (xx2-xx1).clamp(min=0)
        h = (yy2-yy1).clamp(min=0)
        inter = w*h
        iou = inter / (areas[i] + areas[rest] - inter)
        order = order[1:][iou <= iou_threshold]
    return torch.tensor(keep, dtype=torch.long)
