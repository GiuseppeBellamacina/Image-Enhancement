"""
Loss functions for training.
"""

from .l1_loss import L1Loss
from .l2_loss import L2Loss
from .perceptual_loss import PerceptualLoss
from .ssim_loss import SSIMLoss

__all__ = ['L1Loss', 'L2Loss', 'PerceptualLoss', 'SSIMLoss']
