"""Improved inference script with visualization and scaling."""

import torch
from PIL import Image, ImageDraw
import torchvision.transforms as T
from config import DEVICE
from model.kuronet import UNet, DetectorHead
from utils.common import nms_boxes
import matplotlib.pyplot as plt
import cv2
import numpy as np

def load_checkpoint(path, device=DEVICE):
    ckpt = torch.load(path, map_location=device)
    label2id = ckpt.get('label2id', {})
    num_classes = max(1, len(label2id))
    unet = UNet(in_channels=3, base_features=8).to(DEVICE)
    detector = DetectorHead(in_ch=32, num_classes=num_classes).to(device)
    unet.load_state_dict(ckpt['unet']); detector.load_state_dict(ckpt['detector'])
    unet.eval(); detector.eval()
    return unet, detector


def preprocess_image(path, size=(512, 512)):
    tf = T.Compose([
        T.Resize(size),
        T.ToTensor()
    ])
    img = Image.open(path).convert('RGB')
    tensor = tf(img).unsqueeze(0)
    return img, tensor  # return PIL + tensor for visualization


def run_inference(image_path, ckpt_path, k=30):
    unet, detector = load_checkpoint(ckpt_path)
    pil_img, img_t = preprocess_image(image_path)
    img_t = img_t.to(DEVICE)

    with torch.no_grad():
        feats = unet(img_t)
        out = detector(feats)
        heat = out['heatmap'][0, 0]

        # take top-K peaks
        vals, idxs = torch.topk(heat.flatten(), k)
        H, W = heat.shape
        boxes, scores = [], []

        for v, idx in zip(vals, idxs):
            y = (idx // W).item()
            x = (idx % W).item()

            # hier z.B. 10x10 Box um den Punkt für Visualisierung
            size = 10
            boxes.append([x - size, y - size, x + size, y + size])
            scores.append(v.item())

        boxes = torch.tensor(boxes, dtype=torch.float32)
        scores = torch.tensor(scores)

        # Non-maximum suppression
        keep = nms_boxes(boxes, scores, iou_threshold=0.3)
        boxes, scores = boxes[keep], scores[keep]

        # Skalierung zur Originalbildgröße (falls nötig)
        scale_x = pil_img.width / W
        scale_y = pil_img.height / H
        boxes_scaled = []
        for box in boxes:
            x1, y1, x2, y2 = box
            boxes_scaled.append([
                x1 * scale_x, y1 * scale_y,
                x2 * scale_x, y2 * scale_y
            ])


        draw = ImageDraw.Draw(pil_img)
        for b, s in zip(boxes_scaled, scores):
            draw.rectangle(b, outline="red", width=2)
            draw.text((b[0], b[1]-10), f"{s:.2f}", fill="red")

        plt.figure(figsize=(8, 8))
        plt.imshow(pil_img)
        plt.axis("off")
        plt.show()

        # Return numpy arrays (boxes as list of lists, scores as numpy array)
        return boxes_scaled, scores.numpy()