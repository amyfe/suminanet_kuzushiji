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


def focal_loss_heatmap(
    pred,
    target,
    alpha=0.25,
    gamma=2.0,
    pos_weight=1.0,
    pos_threshold=0.3,
    illus_mask=None,
    illus_neg_weight=4.0,
):
    """Focal loss for soft Gaussian heatmap predictions.

    Uses pos_threshold (default 0.3) instead of 0.5 to decide positive vs negative.
    With sigma≈2 pixels, a pixel 1.5σ from center has target≈0.32, which should be
    treated as a positive — the old 0.5 threshold incorrectly penalized it as background.

    Args:
        pred: (B, 1, H, W) raw logits (before sigmoid)
        target: (B, 1, H, W) soft Gaussian heatmap targets in [0, 1]
        alpha: Weight for positive pixels
        gamma: Focal exponent
        pos_weight: Additional scalar weight on positive BCE terms
        pos_threshold: Gaussian value above which a pixel is treated as positive
        illus_mask: (B, 1, H, W) bool tensor marking illustration regions, or None.
                    When provided, negative cells inside illustrations receive
                    illus_neg_weight × extra loss to suppress false positives there.
        illus_neg_weight: Multiplier for illustration false positives (default 4.0).
    """
    pred_flat   = pred.view(-1)
    target_flat = target.view(-1)

    p = torch.sigmoid(pred_flat)
    pos_mask = target_flat >= pos_threshold
    p_t = torch.where(pos_mask, p, 1 - p)

    focal_weight = (1 - p_t) ** gamma

    ce_loss = F.binary_cross_entropy_with_logits(
        pred_flat, target_flat, reduction='none',
        pos_weight=torch.tensor(pos_weight, device=pred.device),
    )

    alpha_t = torch.where(pos_mask,
                          torch.full_like(target_flat, alpha),
                          torch.full_like(target_flat, 1.0 - alpha))

    per_element = alpha_t * focal_weight * ce_loss

    if illus_mask is not None:
        # Upweight negative cells inside illustration regions only.
        # Positive cells (annotated characters) keep weight 1 — they shouldn't
        # be inside illustrations after the GT filter, but we never punish correct
        # predictions.
        illus_flat = illus_mask.view(-1)
        hard_neg   = illus_flat & (~pos_mask)
        weight     = torch.where(hard_neg,
                                 torch.full_like(per_element, illus_neg_weight),
                                 torch.ones_like(per_element))
        per_element = per_element * weight

    return per_element.mean()
