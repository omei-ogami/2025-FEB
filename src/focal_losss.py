import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        """
        gamma: Focusing parameter (higher = more focus on difficult samples)
        alpha: Class balance factor (use tensor of shape [num_classes] if needed)
        reduction: "mean" | "sum" | "none"
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: (B, C, H, W) - Model raw outputs (before softmax)
        targets: (B, H, W) - Ground truth labels
        """
        # Convert logits to probabilities using softmax
        probs = F.softmax(logits, dim=1)
        probs = probs.clamp(min=1e-7, max=1.0)  # Avoid log(0)

        # Gather the probabilities of the target class
        targets_one_hot = F.one_hot(targets, num_classes=logits.shape[1])  # (B, H, W) -> (B, H, W, C)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        target_probs = (probs * targets_one_hot).sum(dim=1)  # (B, C, H, W) -> (B, H, W)

        # Compute the focal loss
        focal_weight = (1 - target_probs) ** self.gamma
        ce_loss = F.cross_entropy(logits, targets, reduction="none")  # Standard CE loss
        loss = focal_weight * ce_loss

        # Apply alpha weighting (optional)
        if self.alpha is not None:
            alpha_factor = self.alpha[targets]  # Select alpha for each target class
            loss *= alpha_factor

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss  # No reduction (useful for per-pixel loss analysis)
