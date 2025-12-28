"""
Evaluation metrics for image enhancement
Includes PSNR, SSIM, MAE, MSE calculations
"""

import torch
import torch.nn.functional as F
from typing import Dict


def calculate_psnr(
    img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0
) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio between two images.

    Args:
        img1: First image tensor (B, C, H, W) or (C, H, W)
        img2: Second image tensor (same shape as img1)
        max_val: Maximum possible pixel value (default: 1.0 for normalized images)

    Returns:
        PSNR value in dB
    """
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float("inf")

    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()


def calculate_ssim(
    img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, max_val: float = 1.0
) -> float:
    """
    Calculate Structural Similarity Index between two images.
    Simplified version for RGB images.

    Args:
        img1: First image tensor (B, C, H, W) or (C, H, W)
        img2: Second image tensor (same shape as img1)
        window_size: Size of gaussian window
        max_val: Maximum possible pixel value

    Returns:
        SSIM value (0 to 1, higher is better)
    """
    # Add batch dimension if needed
    if img1.ndim == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)

    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    # Calculate means
    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size // 2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size // 2)

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    # Calculate variances and covariance
    sigma1_sq = (
        F.avg_pool2d(img1 * img1, window_size, stride=1, padding=window_size // 2)
        - mu1_sq
    )
    sigma2_sq = (
        F.avg_pool2d(img2 * img2, window_size, stride=1, padding=window_size // 2)
        - mu2_sq
    )
    sigma12 = (
        F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size // 2)
        - mu1_mu2
    )

    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    return ssim_map.mean().item()


def calculate_mae(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    Calculate Mean Absolute Error between two images.

    Args:
        img1: First image tensor
        img2: Second image tensor

    Returns:
        MAE value
    """
    return F.l1_loss(img1, img2).item()


def calculate_mse(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    Calculate Mean Squared Error between two images.

    Args:
        img1: First image tensor
        img2: Second image tensor

    Returns:
        MSE value
    """
    return F.mse_loss(img1, img2).item()


def calculate_all_metrics(
    restored: torch.Tensor, target: torch.Tensor, max_val: float = 1.0
) -> Dict[str, float]:
    """
    Calculate all evaluation metrics between restored and target images.

    Args:
        restored: Restored image tensor (B, C, H, W) or (C, H, W)
        target: Target clean image tensor (same shape as restored)
        max_val: Maximum possible pixel value

    Returns:
        Dictionary with all metrics (PSNR, SSIM, MAE, MSE)
    """
    with torch.no_grad():
        metrics = {
            "psnr": calculate_psnr(restored, target, max_val),
            "ssim": calculate_ssim(restored, target, max_val=max_val),
            "mae": calculate_mae(restored, target),
            "mse": calculate_mse(restored, target),
        }

    return metrics


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """
    Pretty print metrics.

    Args:
        metrics: Dictionary of metrics
        prefix: Optional prefix for the output
    """
    if prefix:
        print(f"\n{prefix}")

    print(f"   PSNR: {metrics['psnr']:.2f} dB")
    print(f"   SSIM: {metrics['ssim']:.4f}")
    print(f"   MAE:  {metrics['mae']:.6f}")
    print(f"   MSE:  {metrics['mse']:.6f}")


if __name__ == "__main__":
    # Test metrics
    print("Testing metrics...")

    # Create test images
    img1 = torch.rand(1, 3, 256, 256)
    img2 = img1 + torch.randn_like(img1) * 0.1  # Add noise

    metrics = calculate_all_metrics(img1, img2)
    print_metrics(metrics, "Metrics between original and noisy image:")
