"""Loss functions for segmentation."""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import segmentation_models_pytorch as smp


def get_loss_fn(
    loss_type: str = "bce",
    pos_weight: Optional[float] = None,
    smooth: float = 1.0,
) -> nn.Module:
    """Get loss function by name.

    Args:
        loss_type: Loss type ('bce', 'dice', 'bce_dice', 'focal').
        pos_weight: Positive class weight for BCE loss.
        smooth: Smoothing factor for Dice loss.

    Returns:
        Loss function module.
    """
    loss_type = loss_type.lower()

    if loss_type == "bce":
        return BCEWithLogitsLoss(pos_weight=pos_weight)
    elif loss_type == "dice":
        return smp.losses.DiceLoss(mode="binary", from_logits=True, smooth=smooth)
    elif loss_type == "bce_dice":
        return BCEDiceLoss(pos_weight=pos_weight, smooth=smooth)
    elif loss_type == "focal":
        return smp.losses.FocalLoss(mode="binary")
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


class BCEWithLogitsLoss(nn.Module):
    """Binary Cross Entropy with optional positive class weighting."""

    def __init__(self, pos_weight: Optional[float] = None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.pos_weight is not None:
            pos_weight = torch.tensor([self.pos_weight], device=pred.device)
        else:
            pos_weight = None

        return F.binary_cross_entropy_with_logits(pred, target, pos_weight=pos_weight)


class BCEDiceLoss(nn.Module):
    """Combined BCE and Dice loss."""

    def __init__(
        self,
        pos_weight: Optional[float] = None,
        smooth: float = 1.0,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
    ):
        super().__init__()
        self.bce = BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = smp.losses.DiceLoss(mode="binary", from_logits=True, smooth=smooth)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss
