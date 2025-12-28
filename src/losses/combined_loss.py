"""
Loss functions for Image Enhancement
Combines L1 and SSIM losses
"""

import torch
import torch.nn as nn
from pytorch_msssim import SSIM


class CombinedLoss(nn.Module):
    """
    Combined L1 + SSIM Loss for image enhancement.

    Args:
        alpha: Weight for L1 loss (default: 0.84)
        beta: Weight for SSIM loss (default: 0.16)

    The weights are based on empirical results from image restoration papers.
    L1 ensures pixel-wise accuracy, SSIM ensures structural similarity.
    """

    def __init__(self, alpha=0.84, beta=0.16):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIM(data_range=1.0, size_average=True, channel=3)

    def forward(self, pred, target):
        """
        Args:
            pred: Predicted image (B, 3, H, W) in range [-1, 1] (after normalization)
            target: Target image (B, 3, H, W) in range [-1, 1]

        Returns:
            Combined loss value
        """
        # L1 loss
        l1 = self.l1_loss(pred, target)

        # SSIM loss (convert to 0-1 range for SSIM calculation)
        pred_01 = (pred + 1) / 2  # [-1, 1] -> [0, 1]
        target_01 = (target + 1) / 2
        ssim_val = self.ssim_loss(pred_01, target_01)
        ssim_loss = 1 - ssim_val  # Convert to loss (lower is better)

        # Combined loss
        total_loss = self.alpha * l1 + self.beta * ssim_loss

        return total_loss, {
            "l1": l1.item(),
            "ssim": ssim_val.item(),
            "total": total_loss.item(),
        }


class L1Loss(nn.Module):
    """Simple L1 Loss wrapper"""

    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, pred, target):
        loss = self.loss(pred, target)
        return loss, {"l1": loss.item(), "total": loss.item()}


class L2Loss(nn.Module):
    """Simple L2/MSE Loss wrapper"""

    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, pred, target):
        loss = self.loss(pred, target)
        return loss, {"mse": loss.item(), "total": loss.item()}


def test_losses():
    """Test loss functions"""
    batch_size = 4
    pred = torch.randn(batch_size, 3, 128, 128) * 0.5  # Range approx [-1, 1]
    target = torch.randn(batch_size, 3, 128, 128) * 0.5

    # Test Combined Loss
    combined_loss = CombinedLoss()
    loss, metrics = combined_loss(pred, target)

    print("Combined Loss Test:")
    print(f"  Total loss: {metrics['total']:.4f}")
    print(f"  L1 loss: {metrics['l1']:.4f}")
    print(f"  SSIM: {metrics['ssim']:.4f}")

    # Test L1
    l1_loss = L1Loss()
    loss, metrics = l1_loss(pred, target)
    print(f"\nL1 Loss: {metrics['total']:.4f}")

    # Test L2
    l2_loss = L2Loss()
    loss, metrics = l2_loss(pred, target)
    print(f"L2 Loss: {metrics['total']:.4f}")

    print("\n✅ Loss functions test passed!")


if __name__ == "__main__":
    test_losses()
