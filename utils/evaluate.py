"""
Evaluation-Skript für Kuronet-Demo:
Berechnet Accuracy, CER, mAP und speichert eine Konfusionsmatrix-Heatmap.
"""

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
import numpy as np

from utils import KuzushijiDataset
from utils.metrics import compute_accuracy, cer, compute_mAP
from model.kuronet import UNet, DetectorHead
from config import DEVICE, CHECKPOINT_DIR

from sklearn.metrics import confusion_matrix

# ---- Hilfsfunktion ----
def plot_confusion_matrix(y_true, y_pred, num_classes, out_path="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-9)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, cmap="Blues", cbar=True) #plt.imshow(cm_norm, cmap="Blues")
    plt.title("Confusion Matrix (normalized)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"✅ Confusion matrix saved to {out_path}")

def visualize_predictions(pil_img_tensor, boxes, true_labels, pred_labels, out_path):
    img = (pil_img_tensor.permute(1,2,0).cpu().numpy() * 255).astype('uint8').copy()
    for i, b in enumerate(boxes):
        x1,y1,x2,y2 = map(int, b)
        true_lbl = true_labels[i]
        pred_lbl = pred_labels[i]
        color = (0,255,0) if true_lbl==pred_lbl else (255,0,0)
        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
        cv2.putText(img, f"T:{true_lbl} P:{pred_lbl}", (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

# ---- Hauptfunktion ----
def evaluate(checkpoint_path, data_root="data", out_dir=CHECKPOINT_DIR, iou_thresh=0.5):
    dataset = KuzushijiDataset(root_dir=data_root)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    label2id = ckpt.get('label2id', dataset.label2id)
    id2label = {v:k for k,v in label2id.items()}
    num_classes = max(1, len(label2id))
    # Modelle laden
    unet = UNet(in_channels=3, base_features=8).to(DEVICE)
    detector = DetectorHead(in_ch=8, num_classes=num_classes).to(DEVICE)

    unet.load_state_dict(ckpt['unet'])
    detector.load_state_dict(ckpt['detector'])
    unet.eval(); detector.eval()

    y_true = []
    y_pred = []
    print("🚀 Running evaluation...")
    
    os.makedirs(os.path.join(out_dir, "visuals"), exist_ok=True)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            imgs = batch["image"].to(DEVICE)
            feats = unet(imgs)
            out = detector(feats)

            # Klassenvorhersage: argmax über Klassenachse
            pred = torch.argmax(out["cls"], dim=1)  # [B,H,W] or out['cls'].squeeze(0)
            pred_label = int(torch.mode(pred.flatten()).values.item())  # dominant class prediction

            # GT
            gt_boxes = batch['boxes'][0].numpy() if isinstance(batch['boxes'], torch.Tensor) else np.array([])
            labels_raw = batch['raw_labels']
            labels = []
            for lbl in labels_raw:
                labels.append(label2id.get(lbl, -1))

            # For confusion: naive matching: if any gt exists, compare dominant pred to first gt
            if len(labels) > 0:
                y_true.append(labels[0])
                y_pred.append(pred_label)
            else:
                # no gt, skip
                continue
            # nimm erstes Label als repräsentativ
            true_label = int(labels[0]) if not isinstance(labels[0], str) else 0

            y_true.append(true_label)
            y_pred.append(pred_label)

            # visualize (draw GT boxes and predicted label)
            vis_path = os.path.join(out_dir, "visuals", f"eval_{i:03d}.png")
            pred_labels_list = [id2label[pred_label]] * max(1, len(labels))
            visualize_predictions(batch["image"][0].cpu(), gt_boxes, labels_raw, pred_labels_list, vis_path)


    #mAP = compute_mAP()

    # Konfusionsmatrix zeichnen
    out_path = os.path.join(CHECKPOINT_DIR, "confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, num_classes, out_path=out_path)


if __name__ == "__main__":
    ckpt_path = CHECKPOINT_DIR / "checkpoint_epoch1.pt"
    evaluate(ckpt_path)
