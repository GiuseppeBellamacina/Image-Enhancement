"""
Loss functions for training.
"""

# from .l1_loss import L1Loss
from .combined_loss import CombinedLoss, L2Loss
from .perceptual_loss import VGGPerceptualLoss, CombinedPerceptualLoss

__all__ = ["CombinedLoss", "L2Loss", "VGGPerceptualLoss", "CombinedPerceptualLoss"]
