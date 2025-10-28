import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.common import seed_everything
from utils import KuzushijiDataset
from utils.losses import heatmap_loss, bbox_loss, class_loss, seq_loss
from utils.metrics import cer, compute_accuracy
from utils.logger import SimpleLogger
from model.kuronet import UNet, DetectorHead, GlyphClassifier, SeqDecoder
from config import DEVICE, NUM_EPOCHS, LR, WEIGHT_DECAY, CHECKPOINT_DIR

def train():
    seed_everything()
    logger = SimpleLogger()

    dataset = KuzushijiDataset(root_dir="data/demo")
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    unet = UNet(in_channels=3, base_features=16).to(DEVICE)
    detector = DetectorHead(in_ch=16, num_classes=128).to(DEVICE)
    classifier = GlyphClassifier(in_ch=3, n_classes=128, base=16).to(DEVICE)
    decoder = SeqDecoder(embed_dim=64, hidden_dim=128, vocab_size=128).to(DEVICE)

    optimizer = optim.AdamW(
        list(unet.parameters()) + list(detector.parameters()) +
        list(classifier.parameters()) + list(decoder.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )

    for epoch in range(NUM_EPOCHS):
        logger.start_epoch(epoch)
        unet.train(); detector.train(); classifier.train(); decoder.train()
        total_loss = 0.0

        for batch in loader:
            images = batch['image'].to(DEVICE)
            optimizer.zero_grad()

            feats = unet(images)
            out = detector(feats)

            gt_heat = torch.zeros_like(out['heatmap'])
            gt_bbox = torch.zeros_like(out['bbox'])
            gt_cls = torch.zeros(out['cls'].shape[0], dtype=torch.long, device=DEVICE)

            loss = (
                heatmap_loss(out['heatmap'], gt_heat) +
                bbox_loss(out['bbox'], gt_bbox) +
                class_loss(out['cls'].mean(dim=[2,3]), gt_cls)
            )

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            del loss, images, feats, out

        logger.end_epoch(epoch, total_loss)

        torch.save({
            'unet': unet.state_dict(),
            'detector': detector.state_dict(),
            'classifier': classifier.state_dict(),
            'decoder': decoder.state_dict(),
        }, CHECKPOINT_DIR / f"checkpoint_epoch{epoch+1}.pt")

if __name__ == "__main__":
    train()
