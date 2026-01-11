"""
Residual UNet Architecture for Image Denoising
Predicts noise residual instead of clean image - more effective for denoising tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Double convolution block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # Use bilinear upsampling or transposed conv
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            # Use kernel_size=4 with padding=1 to avoid checkerboard artifacts
            # Standard kernel_size=2 creates uneven pixel overlaps causing grid patterns
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=4, stride=2, padding=1
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Handle size mismatch (if input size not divisible by 16)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final output convolution"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNetResidual(nn.Module):
    """
    Residual UNet for Image Denoising

    Instead of predicting the clean image directly, this network predicts
    the noise/residual and subtracts it from the input. This approach is
    more effective for denoising tasks as the network learns to identify
    and remove noise patterns.

    Forward pass: output = input - predicted_noise

    Args:
        in_channels: Number of input channels (3 for RGB)
        out_channels: Number of output channels (3 for RGB, must match in_channels)
        features: Base number of features (default: 64)
        bilinear: Use bilinear upsampling instead of transposed conv
    """

    def __init__(self, in_channels=3, out_channels=3, features=64, bilinear=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        assert in_channels == out_channels, (
            "UNetResidual requires in_channels == out_channels "
            f"(got {in_channels} and {out_channels})"
        )

        # Encoder
        self.inc = DoubleConv(in_channels, features)
        self.down1 = Down(features, features * 2)
        self.down2 = Down(features * 2, features * 4)
        self.down3 = Down(features * 4, features * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(features * 8, features * 16 // factor)

        # Decoder
        self.up1 = Up(features * 16, features * 8 // factor, bilinear)
        self.up2 = Up(features * 8, features * 4 // factor, bilinear)
        self.up3 = Up(features * 4, features * 2 // factor, bilinear)
        self.up4 = Up(features * 2, features, bilinear)

        # Output - predicts noise residual
        self.outc = OutConv(features, out_channels)

    def forward(self, x):
        """
        Forward pass with residual learning

        Args:
            x: Input noisy image [B, C, H, W]

        Returns:
            Denoised image [B, C, H, W] = input - predicted_noise
        """
        # Store input for residual connection
        input_image = x

        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Predict noise residual
        noise = self.outc(x)

        # Subtract noise from input to get clean image
        # This is the key difference from standard UNet!
        denoised = input_image - noise

        return denoised

    def get_num_params(self):
        """Return number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def test_unet_residual():
    """Test UNetResidual forward pass"""
    model = UNetResidual(in_channels=3, out_channels=3, features=64)
    x = torch.randn(2, 3, 128, 128)

    print(f"Input shape: {x.shape}")

    with torch.no_grad():
        output = model(x)

    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {model.get_num_params():,}")

    assert output.shape == x.shape, "Output shape mismatch!"
    print("✅ UNetResidual test passed!")

    # Verify residual learning
    print("\n📊 Residual Learning Test:")
    noise = torch.randn_like(x) * 0.1
    noisy = x + noise
    with torch.no_grad():
        denoised = model(noisy)
    print(f"   Input range: [{x.min():.3f}, {x.max():.3f}]")
    print(f"   Noisy range: [{noisy.min():.3f}, {noisy.max():.3f}]")
    print(f"   Output range: [{denoised.min():.3f}, {denoised.max():.3f}]")


if __name__ == "__main__":
    test_unet_residual()
