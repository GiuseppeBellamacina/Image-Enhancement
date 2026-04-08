import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation style Channel Attention.

    Computes a per-channel weight in [0, 1] by:
      1. Global Average Pooling  → [B, C, 1, 1]
      2. Two 1×1 convolutions (C → C//reduction → C)
      3. Sigmoid activation
      4. Element-wise multiplication with the input feature map
    """

    def __init__(self, channels: int, reduction: int = 16):
        """
        Args:
            channels:  Number of input (and output) channels.
            reduction: Bottleneck ratio for the FC-like path (default: 16).
        """
        super(ChannelAttention, self).__init__()

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        weights = self.gap(x)       # [B, C, 1, 1]
        weights = self.fc(weights)  # [B, C, 1, 1]  values in (0, 1)
        return x * weights          # broadcast over H, W


class EAB(nn.Module):
    """
    Enhancement Attention Block (EAB).

    Three sequential dilated 3×3 convolutions (dilation = 1, 2, 3) capture
    multi-scale context without increasing the spatial footprint.  A Channel
    Attention module re-weights the channels, and a local residual connection
    adds the block input back to the attended features.

    BatchNorm after each conv (paper-style). Prefer validation forward in full
    precision and loss in fp32 (see training loop) to reduce val NaN with AMP;
    fewer EABs also lowers depth and numerical load.

    Spatial dimensions are preserved throughout (padding = dilation).
    """

    def __init__(
        self,
        channels: int = 64,
        reduction: int = 16,
        residual_scale: float = 1.0,
    ):
        """
        Args:
            channels:  Number of feature channels (input = output).
            reduction: Reduction ratio passed to ChannelAttention.
            residual_scale: Multiplier on the non-identity branch before the local skip
                (values < 1 stabilize very deep stacks; 1.0 matches the original formulation).
        """
        super(EAB, self).__init__()

        self.residual_scale = residual_scale

        self.branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.ca = ChannelAttention(channels, reduction)

    def forward(self, x):
        delta = self.branch(x)
        delta = self.ca(delta)
        return x + self.residual_scale * delta


class RIDNet(nn.Module):
    """
    RIDNet: Real Image Denoising Network with Feature Attention.

    Architecture (Anwar & Barnes, 2019):
      - Feature Extraction : Conv(C_in → features, 3×3) + ReLU
      - Enhancement        : num_eab × EAB
      - Reconstruction     : Conv(features → C_in, 3×3)
      - Global skip        : output = noisy_input - predicted_noise

    The global skip implements residual (noise) learning identically to DnCNN,
    but the EABs provide richer multi-scale, attention-guided feature extraction.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        features: int = 64,
        num_eab: int = 3,
        reduction: int = 16,
        eab_residual_scale: float = 1.0,
    ):
        """
        Args:
            in_channels:  Number of input channels (3 for RGB, 1 for grayscale).
            out_channels: Number of output channels (must equal in_channels).
            features:     Number of feature maps throughout the network (default: 64).
            num_eab:      Number of Enhancement Attention Blocks (default: 3; paper uses 4).
            reduction:    Channel attention reduction ratio (default: 16).
            eab_residual_scale: Passed to each EAB (e.g. 0.1 for deep RIDNet).
        """
        super(RIDNet, self).__init__()

        assert in_channels == out_channels, (
            f"RIDNet requires in_channels == out_channels "
            f"(got {in_channels} and {out_channels})"
        )

        # Feature extraction
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

        # Enhancement blocks
        self.enhancement = nn.Sequential(
            *[
                EAB(
                    channels=features,
                    reduction=reduction,
                    residual_scale=eab_residual_scale,
                )
                for _ in range(num_eab)
            ]
        )

        # Reconstruction (predicts the noise residual)
        self.reconstruction = nn.Conv2d(
            features, out_channels, kernel_size=3, padding=1, bias=True
        )

        self._initialize_weights()

    def forward(self, x):
        """
        Forward pass with global residual learning.

        Args:
            x: Noisy image [B, C, H, W]

        Returns:
            Denoised image [B, C, H, W]
        """
        features = self.feature_extraction(x)
        features = self.enhancement(features)
        noise = self.reconstruction(features)

        output = x - noise

        if not self.training:
            output = torch.clamp(output, min=-1.0, max=1.0)

        return output

    def get_num_params(self):
        """Conta i parametri addestrabili."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    model = RIDNet(in_channels=3, out_channels=3, features=64, num_eab=3)
    print(f"RIDNet parameters: {model.get_num_params():,}")

    x = torch.randn(4, 3, 128, 128)
    y = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    assert x.shape == y.shape, "Input and output must have the same shape!"
    print("RIDNet forward pass OK.")
