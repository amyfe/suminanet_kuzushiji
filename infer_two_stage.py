"""Inference pipeline for two-stage detection + classification."""
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from config import DEVICE
from model.kuronet import UNet, DetectorHead
from model.kuronet.classifier import build_glyph_classifier
from utils.box_extraction import extract_boxes_from_heatmap
from utils.reading_order import sort_boxes_reading_order
from utils.vocab import VocabManager


class TwoStageInference:
    """Two-stage inference: detect boxes → classify characters."""
    
    def __init__(self, detector_ckpt, classifier_ckpt, vocab_path=None, device=DEVICE):
        """Initialize inference pipeline.
        
        Args:
            detector_ckpt: Path to detector checkpoint
            classifier_ckpt: Path to classifier checkpoint
            vocab_path: Path to vocabulary file
            device: Device to run on
        """
        self.device = device
        
        # Load vocabulary
        self.vocab = VocabManager(vocab_path) if vocab_path else None
        
        # Load detector
        print("Loading detector...")
        self.unet = UNet(in_channels=3, base_features=32).to(device)
        self.detector = DetectorHead(in_ch=32, num_classes=3000).to(device)
        
        detector_ckpt = torch.load(detector_ckpt, map_location=device)
        self.unet.load_state_dict(detector_ckpt['unet_state_dict'])
        self.detector.load_state_dict(detector_ckpt['detector_state_dict'])
        
        self.unet.eval()
        self.detector.eval()
        
        # Load classifier
        print("Loading classifier...")
        self.classifier = build_glyph_classifier(in_ch=32, vocab_size=3000).to(device)
        
        classifier_ckpt = torch.load(classifier_ckpt, map_location=device)
        self.classifier.load_state_dict(classifier_ckpt['classifier_state_dict'])
        
        self.classifier.eval()
        
        print("Models loaded!")
    
    def infer(self, image_path, conf_thresh=0.3, nms_thresh=0.5,
              visualize=False, return_boxes=False):
        """Run full inference pipeline.
        
        Args:
            image_path: Path to image
            conf_thresh: Confidence threshold for detection
            nms_thresh: NMS threshold
            visualize: Whether to visualize results
            return_boxes: Whether to return box information
            
        Returns:
            transcription: String transcription in reading order
            boxes: (Optional) List of detected boxes with predictions
        """
        # Preprocess image
        image, image_tensor = self._preprocess_image(image_path)
        
        with torch.no_grad():
            # Stage 1: Detection
            features = self.unet(image_tensor)  # (1, 32, H/8, W/8)
            heatmap, bbox_reg, cls_logits = self.detector(features)
            
            # Extract boxes
            results = extract_boxes_from_heatmap(
                heatmap, bbox_reg, cls_logits,
                conf_thresh=conf_thresh, nms_thresh=nms_thresh,
                image_size=image.size[::-1]
            )
            
            boxes_data = results[0]
            boxes = boxes_data['boxes']
            scores = boxes_data['scores']
            class_ids = boxes_data['classes']
            
            if len(boxes) == 0:
                return "", [] if return_boxes else ""
            
            # Stage 2: Classification
            char_preds = []
            char_confidences = []
            
            for box in boxes:
                x1, y1, x2, y2 = [int(v) for v in box.tolist()]
                crop = image.crop((x1, y1, x2, y2))
                crop_tensor = self._preprocess_crop(crop)
                
                with torch.no_grad():
                    roi_logits = self.classifier(crop_tensor)
                    char_pred = roi_logits.argmax(dim=1).item()
                    char_conf = F.softmax(roi_logits, dim=1).max(dim=1).values.item()
                
                char_preds.append(char_pred)
                char_confidences.append(char_conf)
            
            # Convert to character labels
            if self.vocab:
                characters = [self.vocab.id2char(p) for p in char_preds]
            else:
                characters = [str(p) for p in char_preds]
            
            # Sort by reading order
            boxes_np = boxes.cpu().numpy()
            sorted_indices = sort_boxes_reading_order(boxes_np, class_ids, direction='auto')
            
            transcription = "".join([characters[i] for i in sorted_indices])
            
            if visualize:
                self._visualize_results(image, boxes, characters, scores, sorted_indices)
            
            if return_boxes:
                sorted_boxes = []
                for idx in sorted_indices:
                    sorted_boxes.append({
                        'box': boxes[idx].tolist(),
                        'char': characters[idx],
                        'conf': float(scores[idx]),
                        'char_conf': char_confidences[idx]
                    })
                return transcription, sorted_boxes
            
            return transcription
    
    def _preprocess_image(self, image_path, target_size=(512, 512)):
        """Load and preprocess image.
        
        Args:
            image_path: Path to image
            target_size: Target size (H, W)
            
        Returns:
            image: PIL Image
            image_tensor: (1, 3, H, W) tensor
        """
        image = Image.open(image_path).convert('RGB')
        
        # Resize maintaining aspect ratio
        image.thumbnail(target_size, Image.Resampling.LANCZOS)
        
        # Pad to target size
        canvas = Image.new('RGB', target_size, (255, 255, 255))
        offset = ((target_size[0] - image.size[0]) // 2,
                  (target_size[1] - image.size[1]) // 2)
        canvas.paste(image, offset)
        
        # Convert to tensor
        image_array = np.array(canvas).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        return canvas, image_tensor
    
    def _preprocess_crop(self, crop, target_size=(32, 32)):
        """Preprocess cropped character image.
        
        Args:
            crop: PIL Image crop
            target_size: Target size
            
        Returns:
            (1, 3, H, W) tensor
        """
        crop = crop.resize(target_size, Image.Resampling.LANCZOS)
        crop_array = np.array(crop).astype(np.float32) / 255.0
        crop_tensor = torch.from_numpy(crop_array).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        return crop_tensor
    
    def _visualize_results(self, image, boxes, characters, scores, sorted_indices):
        """Visualize detection results.
        
        Args:
            image: PIL Image
            boxes: (N, 4) boxes
            characters: List of character strings
            scores: Detection scores
            sorted_indices: Indices in reading order
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.imshow(image)
        
        colors = plt.cm.tab20(np.linspace(0, 1, len(sorted_indices)))
        
        for order, idx in enumerate(sorted_indices):
            x1, y1, x2, y2 = boxes[idx].tolist()
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=colors[order], facecolor='none'
            )
            ax.add_patch(rect)
            
            label = f"{order+1}: {characters[idx]} ({scores[idx]:.2f})"
            ax.text(x1, y1 - 5, label, fontsize=8, color=colors[order],
                   bbox=dict(facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        plt.show()


# if __name__ == "__main__":
#     import argparse
    
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--image', type=str, required=True, help='Image path')
#     parser.add_argument('--detector', type=str, required=True, help='Detector checkpoint')
#     parser.add_argument('--classifier', type=str, required=True, help='Classifier checkpoint')
#     parser.add_argument('--vocab', type=str, default=None, help='Vocabulary file')
#     parser.add_argument('--visualize', action='store_true', help='Visualize results')
#     parser.add_argument('--conf-thresh', type=float, default=0.3)
#     parser.add_argument('--nms-thresh', type=float, default=0.5)
    
#     args = parser.parse_args()
    
#     # Run inference
#     pipeline = TwoStageInference(args.detector, args.classifier, args.vocab)
#     transcription, boxes = pipeline.infer(
#         args.image,
#         conf_thresh=args.conf_thresh,
#         nms_thresh=args.nms_thresh,
#         visualize=args.visualize,
#         return_boxes=True
#     )
    
#     print("\nTranscription (reading order):")
#     print(transcription)
#     print(f"\nDetected {len(boxes)} characters")
