#!/usr/bin/env python3
"""Diagnostic script to understand Stage 1 training issues."""
import torch
import torch.nn as nn
from pathlib import Path
import json

from config import DATA_DIR, DEVICE, BATCH_SIZE, IMAGE_SIZE, NUM_WORKERS
from model.kuronet import UNet, DetectorHead
from utils import KuzushijiDataset
from utils.vocab import VocabManager
from train_two_stage import collate_fn
from torch.utils.data import DataLoader

def analyze_dataset():
    """Analyze the dataset to understand data distribution."""
    print("="*60)
    print("DATASET ANALYSIS")
    print("="*60)
    
    # Load vocab
    ann_files = sorted(list((Path(DATA_DIR) / "annotations").glob("*.json")))
    vocab = VocabManager.from_annotations(ann_files)
    
    print(f"Vocabulary size: {vocab.vocab_size}")
    print(f"Pad ID: {vocab.pad_id}, SOS ID: {vocab.sos_id}, EOS ID: {vocab.eos_id}")
    
    # Load dataset
    dataset = KuzushijiDataset(Path(DATA_DIR), vocab=vocab, use_sequences=True, resize=IMAGE_SIZE)
    print(f"Total dataset size: {len(dataset)}")
    
    # Analyze a few samples
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0,
                           collate_fn=lambda b: collate_fn(b, vocab.pad_id))
    
    total_boxes = 0
    total_images = 0
    empty_images = 0
    
    print("\nAnalyzing first 100 batches...")
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 100:
            break
            
        boxes = batch['boxes']
        labels = batch['labels']
        
        for b_idx, box_list in enumerate(boxes):
            total_boxes += len(box_list) if box_list.numel() > 0 else 0
            total_images += 1
            if box_list.numel() == 0:
                empty_images += 1
    
    print(f"\nTotal images analyzed: {total_images}")
    print(f"Empty images (no boxes): {empty_images} ({100*empty_images/total_images:.1f}%)")
    print(f"Total boxes: {total_boxes}")
    print(f"Avg boxes per image: {total_boxes / max(1, total_images - empty_images):.1f}")
    print(f"Avg boxes per all images: {total_boxes / total_images:.1f}")

def analyze_detection_targets():
    """Analyze how many pixels are labeled in detection targets."""
    print("\n" + "="*60)
    print("DETECTION TARGETS ANALYSIS")
    print("="*60)
    
    from utils.detection_utils import build_detection_targets
    
    # Load vocab and dataset
    ann_files = sorted(list((Path(DATA_DIR) / "annotations").glob("*.json")))
    vocab = VocabManager.from_annotations(ann_files)
    dataset = KuzushijiDataset(Path(DATA_DIR), vocab=vocab, use_sequences=True, resize=IMAGE_SIZE)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0,
                           collate_fn=lambda b: collate_fn(b, vocab.pad_id))
    
    labeled_pixels_list = []
    
    print(f"\nAnalyzing detection targets for first 50 batches...")
    # IMAGE_SIZE might be a tuple, so get the height
    img_size = IMAGE_SIZE[0] if isinstance(IMAGE_SIZE, (tuple, list)) else IMAGE_SIZE
    print(f"Feature map size: {img_size // 8} x {img_size // 8}")
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 50:
            break
            
        images = batch['image']
        boxes = batch['boxes']
        labels = batch['labels']
        
        B, _, H_in, W_in = images.shape
        H_feat, W_feat = H_in // 8, W_in // 8
        
        gt_heatmap, gt_bbox, gt_bbox_mask, gt_cls = build_detection_targets(
            boxes, labels, (H_feat, W_feat), (H_in, W_in), DEVICE
        )
        
        B, H, W = gt_cls.shape
        labeled = (gt_cls >= 0).sum().item()
        total = B * H * W
        labeled_pixels_list.append((labeled, total))
        
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}: {labeled}/{total} pixels labeled ({100*labeled/total:.2f}%)")
    
    total_labeled = sum(l for l, _ in labeled_pixels_list)
    total_pixels = sum(t for _, t in labeled_pixels_list)
    
    print(f"\nTotal across batches: {total_labeled}/{total_pixels} pixels labeled ({100*total_labeled/total_pixels:.2f}%)")
    print(f"Average per batch: {total_labeled/len(labeled_pixels_list):.0f} labeled pixels")

def test_model_forward():
    """Test model forward pass to check for issues."""
    print("\n" + "="*60)
    print("MODEL FORWARD PASS TEST")
    print("="*60)
    
    # Load vocab and dataset
    ann_files = sorted(list((Path(DATA_DIR) / "annotations").glob("*.json")))
    vocab = VocabManager.from_annotations(ann_files)
    dataset = KuzushijiDataset(Path(DATA_DIR), vocab=vocab, use_sequences=True, resize=IMAGE_SIZE)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0,
                           collate_fn=lambda b: collate_fn(b, vocab.pad_id))
    
    # Build model
    unet = UNet(in_channels=3, base_features=32).to(DEVICE)
    detector = DetectorHead(in_ch=32, num_classes=vocab.vocab_size, dropout_rate=0.3).to(DEVICE)
    
    print(f"Model on device: {DEVICE}")
    print(f"UNet parameters: {sum(p.numel() for p in unet.parameters()):,}")
    print(f"DetectorHead parameters: {sum(p.numel() for p in detector.parameters()):,}")
    
    # Test forward pass
    batch = next(iter(dataloader))
    images = batch['image'].to(DEVICE)
    
    print(f"\nInput shape: {images.shape}")
    
    with torch.no_grad():
        features = unet(images)
        print(f"UNet output shape: {features.shape}")
        
        outputs = detector(features)
        print(f"Detector outputs: {list(outputs.keys())}")
        for key, val in outputs.items():
            print(f"  {key}: {val.shape}")

if __name__ == "__main__":
    analyze_dataset()
    analyze_detection_targets()
    test_model_forward()
