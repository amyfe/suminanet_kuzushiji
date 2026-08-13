"""Fine-grained glyph classifier for cropped glyph patches (DKNet-like).

This classifier supports two modes:
1. in_ch=3: Classify raw cropped image patches (RGB)
2. in_ch=32: Classify ROI-aligned features from UNet (two-stage architecture)

Can be used in stage-2 to refine low-confidence detections.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


from .utils import make_gn


class GlyphClassifier(nn.Module):
    """Classify character crops from images or UNet features.
    
    Args:
        in_ch: Input channels
               - 3 for raw RGB image patches
               - 32 for ROI-aligned UNet features (two-stage architecture)
        n_classes: Number of character classes
        base: Base number of features (scales with input channels)
    """
    def __init__(self, in_ch=3, n_classes=3000, base=32):
        super().__init__()
        # Scale base features if input is high-dimensional (e.g., UNet features)
        if in_ch > 16:
            base = max(base, 64)  # Use larger base for UNet features
        
        self.backbone = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1), make_gn(base), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base, base*2, 3, padding=1), make_gn(base*2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(base*2, n_classes)

    def forward(self, x):
        f = self.backbone(x).view(x.size(0), -1)
        logits = self.fc(f)
        return logits
    
class GlyphClassifierWithEmbedding(nn.Module):
    """Character classifier that also outputs feature embeddings.
    
    Useful for:
    - Triplet loss training (future work)
    - Feature visualization
    - Similarity-based retrieval
    
    Args:
        in_ch: Input channels (3 for images, 32 for UNet features)
        n_classes: Number of character classes
        base_features: Base number of features
        embedding_dim: Dimension of embedding space
    """
    
    def __init__(self, in_ch=32, n_classes=3000, 
                 base_features=64, embedding_dim=256):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, base_features, kernel_size=3, padding=1),
            make_gn(base_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(base_features, base_features * 2, kernel_size=3, padding=1),
            make_gn(base_features * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(base_features * 2, base_features * 4, kernel_size=3, padding=1),
            make_gn(base_features * 4),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Embedding layer
        self.embedding = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(base_features * 4, embedding_dim),
            nn.ReLU(inplace=True)
        )
        
        # Classification head
        self.classifier = nn.Linear(embedding_dim, n_classes)
    
    def forward(self, roi_features, return_embedding=False):
        """
        Args:
            roi_features: (N, in_ch, H_roi, W_roi)
            return_embedding: if True, also return feature embeddings
        
        Returns:
            logits: (N, n_classes)
            embeddings: (N, embedding_dim) if return_embedding=True
        """
        x = self.features(roi_features)
        x = x.view(x.size(0), -1)
        
        embeddings = self.embedding(x)
        logits = self.classifier(embeddings)
        
        if return_embedding:
            return logits, embeddings
        return logits
    
    def get_embeddings(self, roi_features):
        """Extract embeddings only (for retrieval/visualization)."""
        x = self.features(roi_features)
        x = x.view(x.size(0), -1)
        embeddings = self.embedding(x)
        return embeddings


def build_glyph_classifier(vocab_size=2000, in_ch=32, use_embedding=False, base=None):
    """Factory function to build appropriate classifier.
    
    Args:
        vocab_size: Number of character classes
        in_ch: Number of input channels
               - 3 for raw RGB image patches
               - 32 for ROI-aligned UNet features (recommended for two-stage)
        use_embedding: Whether to use embedding version (for triplet loss)
        base: Base features (auto-selected if None based on in_ch)
    
    Returns:
        GlyphClassifier or GlyphClassifierWithEmbedding
    """
    if use_embedding:
        return GlyphClassifierWithEmbedding(
            in_ch=in_ch,
            n_classes=vocab_size,
            embedding_dim=256
        )
    else:
        # Auto-select base if not specified
        if base is None:
            base = 64 if in_ch >= 16 else 32
        return GlyphClassifier(
            in_ch=in_ch,
            n_classes=vocab_size,
            base=base
        )
