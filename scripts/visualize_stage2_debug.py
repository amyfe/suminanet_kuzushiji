"""Visual debug utility for Stage 2 hybrid recognition.

Saves a single overlay image with:
- input image
- coarse detector proposals
- refined stage-2 boxes
- ground-truth boxes

Example:
    python scripts/visualize_stage2_debug.py \
        --detector-ckpt checkpoints/stage1_detection/detector_best.pt \
        --stage2-ckpt checkpoints/stage2_hybrid/stage2_hybrid_best.pt \
        --split val --sample-index 0 --out-path checkpoints/stage2_hybrid/debug/sample_0000.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from config import CHECKPOINT_DIR, DATA_DIR, DEVICE, IMAGE_SIZE, STAGE2_CONTEXT_HIDDEN_DIM, STAGE2_DECODER_EMBED_DIM, STAGE2_DECODER_HIDDEN_DIM, STAGE2_DET_NMS_IOU, STAGE2_DET_SCORE_THRESH, STAGE2_DET_TOP_K, STAGE2_DROPOUT_RATE, STAGE2_PROJ_DIM, STAGE2_REFINE_HIDDEN_DIM, STAGE2_ROI_FEAT_DIM, STAGE2_ROI_SIZE, STAGE2_TOKEN_DIM, STAGE2_USE_AUX_HEAD
from model.kuronet import UNet, DetectorHead
from model.kuronet.hybrid_recognizer import HybridKuroNetRecognizer
from utils import KuzushijiDataset
from utils.vocab import VocabManager



def load_vocab() -> VocabManager:
    ann_files = sorted((Path(DATA_DIR) / "annotations").glob("*.json"))
    if not ann_files:
        raise FileNotFoundError(f"No annotation files found in {Path(DATA_DIR) / 'annotations'}")
    return VocabManager.from_annotations(ann_files)


def denormalize_image(image_tensor: torch.Tensor) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = (img * std + mean) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def clamp_boxes(boxes, width: int, height: int):
    out = []
    for box in boxes:
        x1, y1, x2, y2 = [float(v) for v in box]
        x1 = max(0.0, min(x1, width))
        y1 = max(0.0, min(y1, height))
        x2 = max(0.0, min(x2, width))
        y2 = max(0.0, min(y2, height))
        if x2 > x1 and y2 > y1:
            out.append((x1, y1, x2, y2))
    return out


def draw_boxes(img: np.ndarray, boxes, color, label: str, thickness: int):
    if not boxes:
        return
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        if idx == 0:
            cv2.putText(
                img,
                label,
                (max(0, x1), max(15, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )


def build_stage2_model(detector_ckpt_path: Path, vocab: VocabManager) -> HybridKuroNetRecognizer:
    vocab_size = vocab.vocab_size

    backbone = UNet(in_channels=3, base_features=32).to(DEVICE)
    detector = DetectorHead(
        in_ch=32,
        num_classes=vocab_size,
        dropout_rate=STAGE2_DROPOUT_RATE,
        predict_boxes=True,
        predict_classes=False,
    ).to(DEVICE)

    checkpoint = torch.load(detector_ckpt_path, map_location=DEVICE)
    backbone.load_state_dict(checkpoint["unet_state_dict"])
    detector.load_state_dict(checkpoint["detector_state_dict"])

    model = HybridKuroNetRecognizer(
        backbone=backbone,
        detector=detector,
        backbone_out_channels=32,
        vocab_size=vocab_size,
        proj_dim=STAGE2_PROJ_DIM,
        roi_size=STAGE2_ROI_SIZE,
        roi_feat_dim=STAGE2_ROI_FEAT_DIM,
        refine_hidden_dim=STAGE2_REFINE_HIDDEN_DIM,
        token_dim=STAGE2_TOKEN_DIM,
        context_hidden_dim=STAGE2_CONTEXT_HIDDEN_DIM,
        decoder_embed_dim=STAGE2_DECODER_EMBED_DIM,
        decoder_hidden_dim=STAGE2_DECODER_HIDDEN_DIM,
        det_score_thresh=STAGE2_DET_SCORE_THRESH,
        det_top_k=STAGE2_DET_TOP_K,
        det_nms_iou=STAGE2_DET_NMS_IOU,
        use_aux_head=STAGE2_USE_AUX_HEAD,
        dropout=STAGE2_DROPOUT_RATE,
    ).to(DEVICE)

    return model


def load_stage2_weights(model: HybridKuroNetRecognizer, stage2_ckpt_path: Path):
    checkpoint = torch.load(stage2_ckpt_path, map_location=DEVICE)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)


def main():
    parser = argparse.ArgumentParser(description="Save Stage 2 visual debug overlays.")
    parser.add_argument("--detector-ckpt", type=Path, default=CHECKPOINT_DIR / "stage1_detection" / "detector_best.pt")
    parser.add_argument("--stage2-ckpt", type=Path, default=CHECKPOINT_DIR / "stage2_hybrid" / "stage2_hybrid_best.pt")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--out-path", type=Path, default=CHECKPOINT_DIR / "stage2_hybrid" / "debug" / "sample_0000_overlay.png")
    args = parser.parse_args()

    vocab = load_vocab()
    dataset = KuzushijiDataset(
        Path(DATA_DIR),
        vocab=vocab,
        use_sequences=True,
        resize=IMAGE_SIZE,
        split=args.split,
    )

    if args.sample_index < 0 or args.sample_index >= len(dataset):
        raise IndexError(f"sample-index {args.sample_index} out of range for split '{args.split}' with {len(dataset)} samples")

    sample = dataset[args.sample_index]
    image = sample["image"]
    gt_boxes = sample["boxes"].cpu().numpy().tolist() if sample["boxes"] is not None else []
    orientations = [sample["orientation"]]
    text_ids = sample.get("text_ids", None)
    targets = text_ids.unsqueeze(0).to(DEVICE) if text_ids is not None else None

    model = build_stage2_model(args.detector_ckpt, vocab)
    if args.stage2_ckpt.exists():
        load_stage2_weights(model, args.stage2_ckpt)
    else:
        raise FileNotFoundError(f"Stage 2 checkpoint not found: {args.stage2_ckpt}")

    model.eval()

    with torch.no_grad():
        batch_images = image.unsqueeze(0).to(DEVICE)
        outputs = model(
            images=batch_images,
            orientations=orientations,
            targets=targets,
            teacher_forcing_ratio=0.0,
            input_seq=None,
            sos_id=vocab.sos_id,
            eos_id=vocab.eos_id,
            max_len=None,
        )

    coarse_boxes = outputs["roi_boxes"][0][outputs["roi_mask"][0].bool()].detach().cpu().tolist()
    refined_boxes = outputs["ordered_boxes"][0][outputs["ordered_mask"][0].bool()].detach().cpu().tolist()

    img = denormalize_image(image)
    height, width = img.shape[:2]
    gt_boxes = clamp_boxes(gt_boxes, width, height)
    coarse_boxes = clamp_boxes(coarse_boxes, width, height)
    refined_boxes = clamp_boxes(refined_boxes, width, height)

    draw_boxes(img, gt_boxes, (0, 220, 0), "GT", 1)
    draw_boxes(img, coarse_boxes, (255, 170, 0), "Coarse", 1)
    draw_boxes(img, refined_boxes, (0, 0, 255), "Refined", 2)

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out_path), img)
    print(f"Saved overlay to {args.out_path}")


if __name__ == "__main__":
    main()