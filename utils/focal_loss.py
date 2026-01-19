"""Focal Loss for addressing class imbalance in detection tasks.

FocalLoss(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

Useful for heatmap training where negative pixels >> positive pixels.
Downweights easy negatives, focuses on hard positives/negatives.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for binary or multi-class classification.
    
    Args:
        alpha: Weighting factor in range [0, 1] to balance positive vs negative samples
               or a sequence of weights for each class. Default: 0.25
        gamma: Exponent of the modulating factor (1 - p_t)^gamma. Default: 2.0
        reduction: Specifies reduction to apply to output. Default: 'mean'
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) logits or (N, 1) for binary
            targets: (N,) class indices for multi-class or (N,) for binary
            
        Returns:
            Focal loss value
        """
        # Get probabilities
        if inputs.dim() == 1 or (inputs.dim() == 2 and inputs.size(1) == 1):
            # Binary case
            p = torch.sigmoid(inputs.squeeze(-1))
            p_t = torch.where(targets == 1, p, 1 - p)
            alpha_t = self.alpha if targets.float().mean() < 0.5 else (1 - self.alpha)
        else:
            # Multi-class case
            p = F.softmax(inputs, dim=1)
            p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
            alpha_t = self.alpha
        
        # Focal loss
        ce = F.cross_entropy(inputs, targets, reduction='none') if inputs.dim() > 1 and inputs.size(1) > 1 \
             else F.binary_cross_entropy_with_logits(inputs.squeeze(-1), targets.float(), reduction='none')
        focal_weight = (1 - p_t) ** self.gamma
        loss = alpha_t * focal_weight * ce
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def focal_loss_heatmap(pred, target, alpha=0.25, gamma=2.0, pos_weight=1.0):
    """Focal loss specifically for heatmap predictions.
    
    Args:
        pred: (B, 1, H, W) predicted heatmap logits (raw, before sigmoid)
        target: (B, 1, H, W) target heatmap [0, 1]
        alpha: Weighting for positive class
        gamma: Focusing parameter
        pos_weight: Additional weight for positive pixels
        
    Returns:
        Focal loss value
    """
    # Flatten
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    # Probabilities
    p = torch.sigmoid(pred_flat)
    p_t = torch.where(target_flat > 0.5, p, 1 - p)
    
    # Focal weight
    focal_weight = (1 - p_t) ** gamma
    
    # CE loss with pos_weight
    ce_loss = F.binary_cross_entropy_with_logits(
        pred_flat, target_flat, reduction='none', pos_weight=torch.tensor(pos_weight, device=pred.device)
    )
    
    # Apply alpha
    alpha_t = torch.where(target_flat > 0.5, alpha, 1 - alpha)
    
    # Focal loss
    loss = alpha_t * focal_weight * ce_loss
    
    return loss.mean()
