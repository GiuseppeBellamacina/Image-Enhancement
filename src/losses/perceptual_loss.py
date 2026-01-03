"""
Perceptual Loss using VGG16 features

Computes loss in feature space rather than pixel space, leading to more
perceptually pleasing results for image restoration tasks.

Reference:
    Johnson et al. "Perceptual Losses for Real-Time Style Transfer and Super-Resolution"
    https://arxiv.org/abs/1603.08155
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import List, Optional

from .combined_loss import CombinedLoss


class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG16 features.

    Extracts features from multiple layers of a pre-trained VGG16 network
    and computes the L1 distance between predicted and target features.

    Args:
        layers: Which VGG layers to use for feature extraction.
                Options: 'relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'
                Default uses early-to-mid layers for efficiency
        weights: Weights for each layer's contribution to the loss

    Example:
        >>> loss_fn = VGGPerceptualLoss(layers=['relu2_2', 'relu3_3'])
        >>> pred = torch.randn(1, 3, 128, 128)
        >>> target = torch.randn(1, 3, 128, 128)
        >>> loss = loss_fn(pred, target)
    """

    def __init__(
        self,
        layers: List[str] = ["relu2_2", "relu3_3"],
        weights: Optional[List[float]] = None,
    ):
        super().__init__()

        # Map layer names to VGG feature indices
        self.layer_name_mapping = {
            "relu1_2": 3,  # Early features (edges, colors)
            "relu2_2": 8,  # Mid-level features (textures)
            "relu3_3": 15,  # Higher-level features (patterns)
            "relu4_3": 22,  # Very high-level features
            "relu5_3": 29,  # Abstract features
        }

        # Validate layer names
        for layer in layers:
            if layer not in self.layer_name_mapping:
                raise ValueError(
                    f"Invalid layer name: {layer}. "
                    f"Choose from {list(self.layer_name_mapping.keys())}"
                )

        self.layers = layers
        self.weights = weights if weights is not None else [1.0] * len(layers)

        if len(self.weights) != len(self.layers):
            raise ValueError(
                f"Number of weights ({len(self.weights)}) must match "
                f"number of layers ({len(self.layers)})"
            )

        # Load pre-trained VGG16
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        vgg_features = vgg.features

        # Freeze VGG parameters (we only use it for feature extraction)
        for param in vgg_features.parameters():
            param.requires_grad = False

        # Create feature extractors for each layer
        self.slice_indices = [self.layer_name_mapping[layer] for layer in layers]

        # Build sequential extractors up to each layer
        self.extractors = nn.ModuleList()
        for idx in self.slice_indices:
            self.extractors.append(
                nn.Sequential(*list(vgg_features.children())[: idx + 1])
            )

        # VGG normalization (ImageNet stats)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        # Move to eval mode
        self.eval()

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize images to ImageNet stats.

        Args:
            x: Input tensor in range [0, 1]

        Returns:
            Normalized tensor
        """
        mean = (
            self.mean
            if isinstance(self.mean, torch.Tensor)
            else torch.tensor(self.mean)
        )
        std = self.std if isinstance(self.std, torch.Tensor) else torch.tensor(self.std)
        return (x - mean) / std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss between predicted and target images.

        Args:
            pred: Predicted image (B, 3, H, W) in range [0, 1]
            target: Target image (B, 3, H, W) in range [0, 1]

        Returns:
            Perceptual loss value
        """
        # Normalize inputs
        pred_norm = self.normalize(pred)
        target_norm = self.normalize(target)

        # Extract features and compute loss
        loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        with torch.no_grad():
            # Extract target features (no grad needed)
            target_features = [extractor(target_norm) for extractor in self.extractors]

        # Extract predicted features (need grad for backprop)
        pred_features = [extractor(pred_norm) for extractor in self.extractors]

        # Compute weighted L1 loss for each layer
        for i, (pred_feat, target_feat, weight) in enumerate(
            zip(pred_features, target_features, self.weights)
        ):
            loss += weight * torch.nn.functional.l1_loss(pred_feat, target_feat)

        # Normalize by number of layers
        loss = loss / len(self.layers)

        return loss

    def __repr__(self):
        layer_info = ", ".join(
            [f"{layer}({w:.2f})" for layer, w in zip(self.layers, self.weights)]
        )
        return f"VGGPerceptualLoss(layers=[{layer_info}])"


class CombinedPerceptualLoss(nn.Module):
    """
    Combined L1 + SSIM + Perceptual Loss for image restoration.

    Args:
        alpha: Weight for L1 loss (default: 0.7)
        beta: Weight for SSIM loss (default: 0.2)
        gamma: Weight for perceptual loss (default: 0.1)
        vgg_layers: Which VGG layers to use for perceptual loss

    Example:
        >>> criterion = CombinedPerceptualLoss(alpha=0.7, beta=0.2, gamma=0.1)
        >>> pred = torch.randn(1, 3, 128, 128)
        >>> target = torch.randn(1, 3, 128, 128)
        >>> loss, metrics = criterion(pred, target)
    """

    def __init__(
        self,
        alpha: float = 0.7,  # L1 weight
        beta: float = 0.2,  # SSIM weight
        gamma: float = 0.0,  # Perceptual weight (0 = use CombinedLoss only)
        vgg_layers: List[str] = ["relu2_2", "relu3_3"],
    ):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Type hints for conditional attributes
        self.combined_loss: Optional[CombinedLoss] = None
        self.perceptual_loss: Optional[VGGPerceptualLoss] = None
        self.l1_loss: Optional[nn.L1Loss] = None
        self.ssim_loss: Optional[nn.Module] = None

        # If gamma is 0, use CombinedLoss directly (no VGG)
        if gamma == 0:
            # Normalize alpha and beta to sum to 1.0
            total = alpha + beta
            if total > 0:
                norm_alpha = alpha / total
                norm_beta = beta / total
            else:
                norm_alpha = 0.84
                norm_beta = 0.16

            self.combined_loss = CombinedLoss(alpha=norm_alpha, beta=norm_beta)
        else:
            # Create individual loss components including perceptual
            from pytorch_msssim import SSIM

            self.l1_loss = nn.L1Loss()
            self.ssim_loss = SSIM(data_range=1.0, size_average=True, channel=3)
            self.perceptual_loss = VGGPerceptualLoss(layers=vgg_layers)

            # Validate weights sum (should be close to 1.0)
            total = alpha + beta + gamma
            if abs(total - 1.0) > 0.01:
                import warnings

                warnings.warn(
                    f"Loss weights sum to {total:.3f}, not 1.0. "
                    f"Consider normalizing: α={alpha}, β={beta}, γ={gamma}"
                )

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Compute combined loss.

        Args:
            pred: Predicted image (B, 3, H, W) in range [-1, 1] (normalized)
            target: Target image (B, 3, H, W) in range [-1, 1] (normalized)

        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        # If gamma=0, use CombinedLoss directly (no VGG)
        if self.gamma == 0:
            assert (
                self.combined_loss is not None
            ), "combined_loss should be initialized when gamma=0"
            return self.combined_loss(pred, target)

        # Otherwise compute with perceptual loss
        assert (
            self.l1_loss is not None
            and self.ssim_loss is not None
            and self.perceptual_loss is not None
        ), "Individual losses should be initialized when gamma>0"

        # Convert from [-1, 1] to [0, 1] for all losses
        pred_01 = (pred + 1) / 2
        target_01 = (target + 1) / 2

        # Compute individual losses
        l1 = self.l1_loss(pred_01, target_01)

        ssim_val = self.ssim_loss(pred_01, target_01)
        ssim_loss = 1 - ssim_val

        # Compute perceptual loss
        perceptual = self.perceptual_loss(pred_01, target_01)
        total_loss = self.alpha * l1 + self.beta * ssim_loss + self.gamma * perceptual
        metrics = {
            "l1": l1.item(),
            "ssim": ssim_val.item(),
            "perceptual": perceptual.item(),
            "total": total_loss.item(),
        }

        return total_loss, metrics

    def __repr__(self):
        if self.gamma == 0:
            return f"CombinedPerceptualLoss(γ=0 → using {self.combined_loss})"
        return (
            f"CombinedPerceptualLoss(α={self.alpha}, β={self.beta}, "
            f"γ={self.gamma}, vgg={self.perceptual_loss})"
        )


if __name__ == "__main__":
    # Test the perceptual loss
    print("Testing VGGPerceptualLoss...")

    loss_fn = VGGPerceptualLoss(layers=["relu2_2", "relu3_3"])

    # Create dummy inputs
    pred = torch.randn(2, 3, 128, 128).clamp(0, 1)
    target = torch.randn(2, 3, 128, 128).clamp(0, 1)

    loss = loss_fn(pred, target)
    print(f"Perceptual loss: {loss.item():.6f}")
    print(f"Loss function: {loss_fn}")

    # Test combined loss
    print("\nTesting CombinedPerceptualLoss...")
    combined = CombinedPerceptualLoss(alpha=0.7, beta=0.2, gamma=0.1)

    # Convert to [-1, 1] range (as used in training)
    pred_norm = pred * 2 - 1
    target_norm = target * 2 - 1

    total_loss, metrics = combined(pred_norm, target_norm)
    print(f"Combined loss: {total_loss.item():.6f}")
    print(f"Metrics: {metrics}")
    print(f"Loss function: {combined}")
