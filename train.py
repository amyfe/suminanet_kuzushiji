"""
Training script (lightweight CPU version).
Reduziert Speicherverbrauch und stabilisiert Laufzeit für kleine Demos.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import gc
import json
# faulthandler and anomaly detection are useful for debugging but noisy for regular runs

from utils.common import seed_everything
from utils import KuzushijiDataset
from utils.losses import heatmap_loss, bbox_loss, class_loss, seq_loss
#from utils.metrics import cer, compute_accuracy
from utils.logger import TrainLogger, SimpleLogger
from model.kuronet import UNet, DetectorHead, GlyphClassifier, SeqDecoder
from config import DEVICE, NUM_EPOCHS, LR, WEIGHT_DECAY, CHECKPOINT_DIR, DATA_DIR
from pathlib import Path



def train():
    seed_everything()
    logger = SimpleLogger()
    tlogger = TrainLogger()
    # dynamic mapping for label strings -> ids
    label2id: dict = {}

    # Auto-detect dataset root. Prefer prepared `data/` if present, else fall back to `data/demo`.
    def find_dataset_root() -> Path:
        # prefer explicit prepared dataset in config.DATA_DIR, then project-local data/demo
        candidates = [Path(DATA_DIR), Path("data"), Path("data/demo"), Path("data/demo_data")]
        for c in candidates:
            images_dir = c / "images"
            anns_dir = c / "annotations"
            # accept candidate if annotations exist (images may be organized by book subfolders)
            if anns_dir.exists() and any(anns_dir.glob('*.json')):
                return c
            if images_dir.exists() and anns_dir.exists():
                return c
        # if none found, raise informative error
        raise FileNotFoundError("No dataset found. Expected 'images' and 'annotations' under one of: data/, data/demo/")

    data_root = find_dataset_root()
    print(f"Using dataset at: {data_root}")
    print(f"Torch {torch.__version__}, device={DEVICE}")
    # quick sanity counts (search images recursively because images are organized by book)
    imgs = list((data_root).glob("**/*.jpg"))
    anns = list((data_root / "annotations").glob("*.json"))
    print(f"Found {len(imgs)} images and {len(anns)} annotations in {data_root}")

    dataset = KuzushijiDataset(root_dir=str(data_root))
    if len(dataset) == 0:
        raise RuntimeError(f"No images found in dataset root {data_root}. Check your data layout.")
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    # load label mapping if available so model sizes match the dataset
    label2id_path = Path(data_root) / "label2id.json"
    if label2id_path.exists():
        with open(label2id_path, "r", encoding="utf-8") as f:
            label2id = json.load(f)
        print(f"Loaded label2id with {len(label2id)} classes from {label2id_path}")
    else:
        print("No label2id.json found in dataset root; using dataset-built mapping.")

    # set model class sizes according to label2id mapping
    n_classes = len(label2id) if isinstance(label2id, dict) else 128
    unet = UNet(in_channels=3, base_features=16).to(DEVICE)
    detector = DetectorHead(in_ch=16, num_classes=n_classes).to(DEVICE)
    classifier = GlyphClassifier(in_ch=3, n_classes=n_classes, base=16).to(DEVICE)
    decoder = SeqDecoder(embed_dim=64, hidden_dim=128, vocab_size=n_classes).to(DEVICE)

    # --- Optimizer ---
    optimizer = optim.AdamW(
        list(unet.parameters()) +
        list(detector.parameters()) +
        list(classifier.parameters()) +
        list(decoder.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )
    print("Starting training...\n")
    for epoch in range(NUM_EPOCHS):
        logger.start_epoch(epoch)
        unet.train(); detector.train(); classifier.train(); decoder.train()
        total_loss = 0.0
        num_batches = 0
        for batch in loader:
            num_batches += 1
            print(f"Processing batch {num_batches}/{len(loader)} with {len(batch['image'])} images")
            images = batch['image'].to(DEVICE)
            optimizer.zero_grad()

            # boxes/labels collate: DataLoader will produce list per batch for variable-length
            raw_boxes = batch.get('boxes')
            raw_labels = batch.get('labels')
            if isinstance(raw_boxes, list):
                boxes = raw_boxes[0].to(DEVICE)
                labels = raw_labels[0]
            else:
                boxes = raw_boxes.to(DEVICE)
                labels = raw_labels

            # map label strings to ids using loaded mapping
            label_ids = [label2id.get(lab, 0) for lab in labels]

            # --- Forward ---
            feats = unet(images)
            out = detector(feats)

            # build targets at detector output resolution
            B = images.size(0)
            _, _, H_out, W_out = out['heatmap'].shape
            _, _, H_in, W_in = images.shape
            stride_h = float(H_in) / float(H_out)
            stride_w = float(W_in) / float(W_out)

            # initialize targets
            gt_heat = torch.zeros((B, 1, H_out, W_out), device=DEVICE)
            gt_bbox = torch.zeros((B, 4, H_out, W_out), device=DEVICE)
            gt_cls = torch.full((B, H_out, W_out), -1, dtype=torch.long, device=DEVICE)

            # simple gaussian heatmap at box centers + bbox regression at centers
            for i in range(B):
                if boxes is None or boxes.numel() == 0:
                    continue
                # boxes for this image
                bboxes = boxes if boxes.dim() == 2 else boxes[i]
                labs = label_ids
                for (box, lab) in zip(bboxes, labs):
                    x1, y1, x2, y2 = box.tolist()
                    cx = (x1 + x2) / 2.0 / stride_w
                    cy = (y1 + y2) / 2.0 / stride_h
                    # clamp within grid
                    if cx < 0 or cy < 0 or cx >= W_out or cy >= H_out:
                        continue
                    ix = int(cx)
                    iy = int(cy)
                    # gaussian sigma proportional to box size (in output grid units)
                    bw = max((x2 - x1) / stride_w, 1.0)
                    bh = max((y2 - y1) / stride_h, 1.0)
                    sigma = max(1.0, 0.25 * (bw + bh) / 2.0)
                    # draw gaussian
                    yy = torch.arange(0, H_out, device=DEVICE).view(H_out, 1).float()
                    xx = torch.arange(0, W_out, device=DEVICE).view(1, W_out).float()
                    g = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma * sigma))
                    gt_heat[i, 0] = torch.max(gt_heat[i, 0], g)
                    # bbox targets in output-grid units
                    dx = cx - ix
                    dy = cy - iy
                    gw = bw
                    gh = bh
                    gt_bbox[i, :, iy, ix] = torch.tensor([dx, dy, gw, gh], device=DEVICE)
                    gt_cls[i, iy, ix] = lab

            # compute losses
            loss_h = F.mse_loss(out['heatmap'], gt_heat)
            loss_b = F.l1_loss(out['bbox'], gt_bbox)
            # per-pixel class loss: flatten valid positions
            logits = out['cls'].permute(0, 2, 3, 1).reshape(-1, out['cls'].shape[1])
            labels_flat = gt_cls.reshape(-1)
            valid = labels_flat >= 0
            if valid.sum() > 0:
                loss_c = F.cross_entropy(logits[valid], labels_flat[valid])
            else:
                loss_c = torch.tensor(0.0, device=DEVICE)

            loss = loss_h + loss_b + loss_c
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            del loss, images, feats, out, gt_heat, gt_bbox, gt_cls

        # Nur alle paar Epochen speichern (spart Speicher)
        if (epoch + 1) % 5 == 0 or (epoch + 1) == NUM_EPOCHS:
            ckpt_path = CHECKPOINT_DIR / f"checkpoint_epoch{epoch+1}.pt"
            torch.save({
                'unet': unet.state_dict(),
                'detector': detector.state_dict(),
                'classifier': classifier.state_dict(),
                'decoder': decoder.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, ckpt_path)
            print(f"✅ Checkpoint saved: {ckpt_path.name}\n")
        out_path = CHECKPOINT_DIR / f"epoch{epoch+1}_pred.png"
        # Ensure parent directory exists
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tlogger.log(total_loss, 0.0)
        tlogger.save_plot(out_path=str(out_path))
        #Garbage Collection
        gc.collect()
        torch.cuda.empty_cache()

    print("Training completed successfully.")


if __name__ == "__main__":
    train()
