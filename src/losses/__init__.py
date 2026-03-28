"""
Loss functions for training.
"""

from .combined_loss import CombinedLoss, L1Loss, L2Loss
from .perceptual_loss import VGGPerceptualLoss, CombinedPerceptualLoss

__all__ = ["CombinedLoss", "L1Loss", "L2Loss", "VGGPerceptualLoss", "CombinedPerceptualLoss"]

