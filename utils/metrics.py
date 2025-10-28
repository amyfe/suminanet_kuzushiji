import editdistance
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

def cer(pred_seq, gt_seq):
    if len(gt_seq) == 0:
        return 1.0
    # make sure both are strings
    if isinstance(pred_seq, (list, tuple)):
        pred_seq = "".join(map(str, pred_seq))
    if isinstance(gt_seq, (list, tuple)):
        gt_seq = "".join(map(str, gt_seq))
    return editdistance.eval(pred_seq, gt_seq) / len(gt_seq)

def compute_confusion(preds, gts):
    return confusion_matrix(gts, preds)

def compute_accuracy(preds, gts):
    """Returns normalized confusion matrix."""
    preds = np.array(preds)
    targets = np.array(gts)
    if len(targets)==0:
        return 0.0
    return accuracy_score(gts, preds)

def compute_mAP(pred_boxes, gt_boxes, iou_threshold=0.5):
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return 0.0
    ious = []
    for pb in pred_boxes:
        for gb in gt_boxes:
            iou = intersection_over_union(pb, gb)
            ious.append(iou)
    return np.mean([iou > iou_threshold for iou in ious])

def intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return inter / float(areaA + areaB - inter + 1e-6)
