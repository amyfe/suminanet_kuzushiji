"""Visualization utilities for debugging detection and recognition."""
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path


def visualize_detections(image, boxes, scores=None, classes=None, vocab=None, 
                         save_path=None, show=True, conf_threshold=0.0):
    """Overlay detected boxes on image with labels.
    
    Args:
        image: (C, H, W) tensor or (H, W, C) numpy array
        boxes: (N, 4) boxes [x1, y1, x2, y2]
        scores: (N,) confidence scores (optional)
        classes: (N,) class indices (optional)
        vocab: VocabManager or dict mapping class_id -> character (optional)
        save_path: path to save figure (optional)
        show: whether to display figure
        conf_threshold: only show boxes with score >= threshold
    """
    # Convert tensor to numpy if needed
    if torch.is_tensor(image):
        image = image.cpu().numpy()
        if image.shape[0] == 3:  # (C, H, W) -> (H, W, C)
            image = np.transpose(image, (1, 2, 0))
    
    # Normalize to [0, 1] if needed
    if image.max() > 1.0:
        image = image / 255.0
    
    # Convert boxes to numpy
    if torch.is_tensor(boxes):
        boxes = boxes.cpu().numpy()
    if scores is not None and torch.is_tensor(scores):
        scores = scores.cpu().numpy()
    if classes is not None and torch.is_tensor(classes):
        classes = classes.cpu().numpy()
    
    # Create figure
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(image, cmap='gray' if image.ndim == 2 else None)
    
    # Draw each box
    for i, box in enumerate(boxes):
        # Filter by confidence
        if scores is not None and scores[i] < conf_threshold:
            continue
        
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        
        # Color by score (red = low, green = high)
        if scores is not None:
            color = plt.cm.RdYlGn(scores[i])
        else:
            color = 'lime'
        
        # Draw rectangle
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2, 
                                 edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        # Add label
        label_parts = []
        if classes is not None:
            if vocab is not None:
                if hasattr(vocab, 'id2char'):
                    char = vocab.id2char.get(int(classes[i]), '?')
                elif isinstance(vocab, dict):
                    char = vocab.get(int(classes[i]), '?')
                else:
                    char = str(classes[i])
                label_parts.append(char)
            else:
                label_parts.append(f"cls:{classes[i]}")
        
        if scores is not None:
            label_parts.append(f"{scores[i]:.2f}")
        
        if label_parts:
            label = ' '.join(label_parts)
            ax.text(x1, y1 - 5, label, fontsize=10, color=color,
                   bbox=dict(facecolor='black', alpha=0.5, pad=2))
    
    ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def visualize_heatmap(image, heatmap, save_path=None, show=True, alpha=0.5):
    """Overlay heatmap on image.
    
    Args:
        image: (C, H, W) tensor or (H, W, C) numpy array
        heatmap: (H, W) heatmap tensor or numpy array
        save_path: path to save figure (optional)
        show: whether to display figure
        alpha: transparency of heatmap overlay
    """
    # Convert to numpy
    if torch.is_tensor(image):
        image = image.cpu().numpy()
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
    
    if torch.is_tensor(heatmap):
        heatmap = heatmap.cpu().numpy()
    
    # Normalize
    if image.max() > 1.0:
        image = image / 255.0
    
    # Resize heatmap to match image if needed
    if heatmap.shape != image.shape[:2]:
        from scipy.ndimage import zoom
        zoom_factors = (image.shape[0] / heatmap.shape[0], 
                       image.shape[1] / heatmap.shape[1])
        heatmap = zoom(heatmap, zoom_factors, order=1)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image, cmap='gray' if image.ndim == 2 else None)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Heatmap
    axes[1].imshow(heatmap, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title('Detection Heatmap')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(image, cmap='gray' if image.ndim == 2 else None)
    axes[2].imshow(heatmap, cmap='hot', alpha=alpha, vmin=0, vmax=1)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved heatmap visualization to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def visualize_training_batch(images, gt_boxes, pred_boxes=None, 
                             gt_labels=None, pred_labels=None,
                             save_dir=None, max_samples=4):
    """Visualize a training batch with GT and predictions.
    
    Args:
        images: (B, C, H, W) batch of images
        gt_boxes: list of (N_i, 4) ground truth boxes
        pred_boxes: list of (N_i, 4) predicted boxes (optional)
        gt_labels: list of (N_i,) ground truth labels (optional)
        pred_labels: list of (N_i,) predicted labels (optional)
        save_dir: directory to save figures (optional)
        max_samples: maximum number of samples to visualize
    """
    B = min(images.shape[0], max_samples)
    
    for i in range(B):
        image = images[i]
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        
        # Convert image for visualization
        if torch.is_tensor(image):
            img_np = image.cpu().numpy()
            if img_np.shape[0] == 3:
                img_np = np.transpose(img_np, (1, 2, 0))
            if img_np.max() > 1.0:
                img_np = img_np / 255.0
        
        # Ground truth
        axes[0].imshow(img_np, cmap='gray' if img_np.ndim == 2 else None)
        axes[0].set_title('Ground Truth')
        
        gt_box = gt_boxes[i]
        if torch.is_tensor(gt_box):
            gt_box = gt_box.cpu().numpy()
        
        for j, box in enumerate(gt_box):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            rect = patches.Rectangle((x1, y1), w, h, linewidth=2,
                                    edgecolor='lime', facecolor='none')
            axes[0].add_patch(rect)
            
            if gt_labels is not None and j < len(gt_labels[i]):
                label = str(gt_labels[i][j].item() if torch.is_tensor(gt_labels[i]) else gt_labels[i][j])
                axes[0].text(x1, y1 - 5, label, fontsize=8, color='lime',
                           bbox=dict(facecolor='black', alpha=0.5))
        
        axes[0].axis('off')
        
        # Predictions
        axes[1].imshow(img_np, cmap='gray' if img_np.ndim == 2 else None)
        axes[1].set_title('Predictions')
        
        if pred_boxes is not None and i < len(pred_boxes):
            pred_box = pred_boxes[i]
            if torch.is_tensor(pred_box):
                pred_box = pred_box.cpu().numpy()
            
            for j, box in enumerate(pred_box):
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                rect = patches.Rectangle((x1, y1), w, h, linewidth=2,
                                        edgecolor='red', facecolor='none')
                axes[1].add_patch(rect)
                
                if pred_labels is not None and j < len(pred_labels[i]):
                    label = str(pred_labels[i][j].item() if torch.is_tensor(pred_labels[i]) else pred_labels[i][j])
                    axes[1].text(x1, y1 - 5, label, fontsize=8, color='red',
                               bbox=dict(facecolor='black', alpha=0.5))
        
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_dir / f"batch_sample_{i}.png", dpi=150, bbox_inches='tight')
        
        plt.show()
        plt.close(fig)
