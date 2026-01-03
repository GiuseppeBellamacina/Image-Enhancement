"""
Attention U-Net for Image Restoration

Implements U-Net with attention gates that help the model focus on relevant
spatial regions while suppressing irrelevant activations. This is particularly
effective for image denoising and restoration tasks.

Reference:
    Oktay et al. "Attention U-Net: Learning Where to Look for the Pancreas"
    https://arxiv.org/abs/1804.03999
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate(nn.Module):
    """
    Attention Gate module.

    Learns to focus on specific regions by generating attention coefficients
    that weight the feature maps from skip connections.

    Args:
        F_g: Number of channels in gating signal (from decoder)
        F_l: Number of channels in skip connection (from encoder)
        F_int: Number of intermediate channels
    """

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()

        # Transform gating signal
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        # Transform skip connection
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
        )

        # Generate attention coefficients
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            g: Gating signal from decoder path (B, F_g, H_g, W_g)
            x: Skip connection from encoder path (B, F_l, H_x, W_x)

        Returns:
            Attention-weighted skip connection (B, F_l, H_x, W_x)
        """
        # Resize gating signal to match skip connection size
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # Upsample gating signal if needed
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(
                g1, size=x1.shape[2:], mode="bilinear", align_corners=False
            )

        # Compute attention coefficients
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        # Apply attention (element-wise multiplication)
        return x * psi


class DoubleConv(nn.Module):
    """Double convolution block: (Conv2d -> BatchNorm -> ReLU) x 2"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downsampling block: MaxPool2d -> DoubleConv"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upsampling block with attention gate.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        bilinear: If True, use bilinear upsampling; otherwise use transposed convolution
    """

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()

        # Upsampling layer
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

        # Attention gate
        self.attention = AttentionGate(
            F_g=in_channels // 2,  # Gating signal channels (after upsampling)
            F_l=in_channels // 2,  # Skip connection channels
            F_int=out_channels,  # Intermediate channels
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x1: Feature maps from decoder path (to be upsampled)
            x2: Skip connection from encoder path

        Returns:
            Upsampled and concatenated features
        """
        # Upsample decoder features
        x1 = self.up(x1)

        # Resize if needed (for odd-sized inputs)
        if x1.shape[2:] != x2.shape[2:]:
            x1 = F.interpolate(
                x1, size=x2.shape[2:], mode="bilinear", align_corners=False
            )

        # Apply attention to skip connection
        x2_att = self.attention(g=x1, x=x2)

        # Concatenate
        x = torch.cat([x2_att, x1], dim=1)

        # Apply convolutions
        return self.conv(x)


class AttentionUNet(nn.Module):
    """
    Attention U-Net for image restoration.

    Enhanced U-Net with attention gates that allow the model to focus on
    salient features while suppressing irrelevant regions.

    Args:
        in_channels: Number of input channels (3 for RGB)
        out_channels: Number of output channels (3 for RGB)
        features: Base number of feature channels (default: 64)
        bilinear: Use bilinear upsampling instead of transposed convolutions

    Example:
        >>> model = AttentionUNet(in_channels=3, out_channels=3, features=64)
        >>> x = torch.randn(1, 3, 256, 256)
        >>> output = model(x)
        >>> assert output.shape == (1, 3, 256, 256)
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        features: int = 64,
        bilinear: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        self.bilinear = bilinear

        # Encoder path
        self.inc = DoubleConv(in_channels, features)
        self.down1 = Down(features, features * 2)
        self.down2 = Down(features * 2, features * 4)
        self.down3 = Down(features * 4, features * 8)

        # Bottleneck
        factor = 2 if bilinear else 1
        self.down4 = Down(features * 8, features * 16 // factor)

        # Decoder path (with attention gates)
        self.up1 = Up(features * 16, features * 8 // factor, bilinear)
        self.up2 = Up(features * 8, features * 4 // factor, bilinear)
        self.up3 = Up(features * 4, features * 2 // factor, bilinear)
        self.up4 = Up(features * 2, features, bilinear)

        # Output layer
        self.outc = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Attention U-Net.

        Args:
            x: Input image tensor (B, C, H, W)

        Returns:
            Restored image tensor (B, C, H, W)
        """
        # Encoder path (with skip connections)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder path (attention-gated skip connections)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Output
        logits = self.outc(x)

        return logits

    def get_num_params(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    model = AttentionUNet(in_channels=3, out_channels=3, features=64, bilinear=True)
    x = torch.randn(2, 3, 256, 256)

    print(f"Input shape: {x.shape}")
    output = model(x)
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {model.get_num_params():,}")

    # Test with bilinear=False
    model_conv = AttentionUNet(
        in_channels=3, out_channels=3, features=64, bilinear=False
    )
    output_conv = model_conv(x)
    print("\nWith transposed conv:")
    print(f"Output shape: {output_conv.shape}")
    print(f"Total parameters: {model_conv.get_num_params():,}")
