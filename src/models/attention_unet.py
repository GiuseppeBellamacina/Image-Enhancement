import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# -------------------------
# Normalization factory
# -------------------------
def NormLayer(norm: str, num_channels: int):
    if norm == "batch":
        return nn.BatchNorm2d(num_channels)
    elif norm == "group":
        # GroupNorm requires num_channels to be divisible by num_groups
        # For single channel, use BatchNorm instead
        if num_channels == 1:
            return nn.BatchNorm2d(num_channels)
        # Use min(8, num_channels) to handle cases where num_channels < 8
        num_groups = min(8, num_channels) if num_channels >= 8 else num_channels
        return nn.GroupNorm(num_groups, num_channels)
    else:
        raise ValueError(f"Unknown norm type: {norm}")


# -------------------------
# Attention Gate
# -------------------------
class AttentionGate(nn.Module):
    def __init__(self, F_g: int, F_l: int, F_int: int, norm: str = "batch"):
        super().__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            NormLayer(norm, F_int),
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            NormLayer(norm, F_int),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            NormLayer(norm, 1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(
                g1, size=x1.shape[2:], mode="bilinear", align_corners=False
            )

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


# -------------------------
# Double Conv
# -------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm: str):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            NormLayer(norm, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            NormLayer(norm, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm: str):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, norm),
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: str,
        bilinear: bool,
        use_attention: bool,
    ):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)

        self.use_attention = use_attention
        if use_attention:
            self.attention = AttentionGate(
                F_g=in_channels // 2,
                F_l=in_channels // 2,
                F_int=out_channels,
                norm=norm,
            )

        self.conv = DoubleConv(in_channels, out_channels, norm)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        if x1.shape[2:] != x2.shape[2:]:
            x1 = F.interpolate(
                x1, size=x2.shape[2:], mode="bilinear", align_corners=False
            )

        if self.use_attention:
            x2 = self.attention(x1, x2)

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# -------------------------
# Attention U-Net (parametrico)
# -------------------------
class AttentionUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        features: int = 64,
        bilinear: bool = True,
        use_attention: bool = True,
        attention_levels: Tuple[str] = ("bottleneck",),
        norm: str = "batch",
        trained_noise_sigma: float = 100.0,
    ):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            features: Base number of features
            bilinear: Use bilinear upsampling instead of transposed conv
            use_attention: Enable attention gates
            attention_levels: Where to apply attention
                ("bottleneck",) -> solo up1
                ("all",)        -> tutti gli up
                ()              -> nessuna attention
            norm: Normalization type ('batch' or 'group')
            trained_noise_sigma: Noise level the model was trained on (for adaptive blending)
        """
        super().__init__()
        
        self.trained_noise_sigma = trained_noise_sigma

        self.use_attention = use_attention
        self.attention_levels = attention_levels

        # Encoder
        self.inc = DoubleConv(in_channels, features, norm)
        self.down1 = Down(features, features * 2, norm)
        self.down2 = Down(features * 2, features * 4, norm)
        self.down3 = Down(features * 4, features * 8, norm)

        factor = 2 if bilinear else 1
        self.down4 = Down(features * 8, features * 16 // factor, norm)

        # Decoder
        self.up1 = Up(
            features * 16,
            features * 8 // factor,
            norm,
            bilinear,
            use_attention
            and ("bottleneck" in attention_levels or "all" in attention_levels),
        )
        self.up2 = Up(
            features * 8,
            features * 4 // factor,
            norm,
            bilinear,
            use_attention and "all" in attention_levels,
        )
        self.up3 = Up(
            features * 4,
            features * 2 // factor,
            norm,
            bilinear,
            use_attention and "all" in attention_levels,
        )
        self.up4 = Up(
            features * 2,
            features,
            norm,
            bilinear,
            use_attention and "all" in attention_levels,
        )

        self.outc = nn.Conv2d(features, out_channels, 1)

    def forward(self, x, noise_sigma: Optional[float] = None):
        """
        Forward pass with optional adaptive denoising strength.
        
        Args:
            x: Input tensor (B, C, H, W)
            noise_sigma: Actual noise level in the input (optional)
                If None, returns full denoising output
                If provided, automatically applies adaptive blending based on noise ratio
        
        Returns:
            Denoised output tensor
        """
        # Standard U-Net forward pass
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        out = self.up1(x5, x4)
        out = self.up2(out, x3)
        out = self.up3(out, x2)
        out = self.up4(out, x1)
        out = self.outc(out)
        
        # Adaptive blending if noise level is provided
        if noise_sigma is not None:
            # Calculate blending weight: alpha = actual_sigma / trained_sigma
            alpha = min(noise_sigma / self.trained_noise_sigma, 1.0)
            # Blend: lower noise = more denoising, higher noise = keep more input
            out = alpha * x + (1.0 - alpha) * out
        
        return out

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
