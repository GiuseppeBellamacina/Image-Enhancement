"""
NAFNet architecture for image restoration.

Reference: "Simple Baselines for Image Restoration" (ECCV 2022).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    """Stochastic Depth (Drop Path) regularization.

    During training, randomly drops entire residual branches with probability
    ``drop_prob``, scaling surviving paths by ``1 / (1 - drop_prob)`` to keep
    the expected value unchanged. Disabled at eval time.

    This is the standard technique recommended by the NAFNet / ConvNeXt papers
    to prevent overfitting, especially on smaller datasets like DIV2K.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        # one random value per sample, broadcast over all spatial/channel dims
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, dtype=x.dtype, device=x.device).floor_().add_(keep_prob).clamp_(max=1.0)
        return x / keep_prob * mask

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:.3f}"


class LayerNorm2d(nn.Module):
    """Channel-wise layer normalization for 2D feature maps."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class SimpleGate(nn.Module):
    """Split channels in half and multiply them element-wise."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """Core NAFNet block without nonlinear activations."""

    def __init__(
        self,
        channels: int,
        dw_expand: int = 2,
        ffn_expand: int = 2,
        drop_out_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()

        dw_channels = channels * dw_expand
        self.conv1 = nn.Conv2d(channels, dw_channels, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(
            dw_channels,
            dw_channels,
            kernel_size=3,
            padding=1,
            groups=dw_channels,
            bias=True,
        )
        self.conv3 = nn.Conv2d(dw_channels // 2, channels, kernel_size=1, bias=True)

        self.sg = SimpleGate()

        # Simplified channel attention branch
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channels // 2, dw_channels // 2, kernel_size=1, bias=True),
        )

        ffn_channels = channels * ffn_expand
        self.conv4 = nn.Conv2d(channels, ffn_channels, kernel_size=1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channels // 2, channels, kernel_size=1, bias=True)

        self.norm1 = LayerNorm2d(channels)
        self.norm2 = LayerNorm2d(channels)

        self.dropout1 = (
            nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        )
        self.dropout2 = (
            nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        )

        # Stochastic Depth: drops the entire residual branch during training
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, channels, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, channels, 1, 1)), requires_grad=True)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.norm1(inp)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)
        y = inp + self.drop_path(x * self.beta)

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)
        return y + self.drop_path(x * self.gamma)


class NAFNet(nn.Module):
    """
    NAFNet for single-image restoration tasks.

    Args:
        img_channel: Number of image channels.
        width: Base feature width.
        middle_blk_num: Number of NAF blocks in the bottleneck.
        enc_blk_nums: Number of blocks for each encoder stage.
        dec_blk_nums: Number of blocks for each decoder stage.
        dw_expand: Expansion factor for depthwise branch.
        ffn_expand: Expansion factor for FFN branch.
        drop_out_rate: Dropout probability inside each block.
        drop_path_rate: Maximum stochastic depth rate (linearly increases
            from 0 at the shallowest block to this value at the deepest).
    """

    def __init__(
        self,
        img_channel: int = 3,
        width: int = 32,
        middle_blk_num: int = 1,
        enc_blk_nums: tuple[int, ...] = (1, 1, 1, 28),
        dec_blk_nums: tuple[int, ...] = (1, 1, 1, 1),
        dw_expand: int = 2,
        ffn_expand: int = 2,
        drop_out_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()

        if len(enc_blk_nums) != len(dec_blk_nums):
            raise ValueError("enc_blk_nums and dec_blk_nums must have the same length")

        self.intro = nn.Conv2d(img_channel, width, kernel_size=3, padding=1, bias=True)
        self.ending = nn.Conv2d(width, img_channel, kernel_size=3, padding=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        # Compute total number of blocks to distribute drop path rates linearly
        total_blocks = sum(enc_blk_nums) + middle_blk_num + sum(dec_blk_nums)
        block_idx = 0  # running counter across the whole network

        chan = width
        for n_blocks in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[
                        NAFBlock(
                            chan,
                            dw_expand=dw_expand,
                            ffn_expand=ffn_expand,
                            drop_out_rate=drop_out_rate,
                            drop_path_rate=drop_path_rate * (block_idx + i) / max(total_blocks - 1, 1),
                        )
                        for i in range(n_blocks)
                    ]
                )
            )
            block_idx += n_blocks
            self.downs.append(nn.Conv2d(chan, 2 * chan, kernel_size=2, stride=2))
            chan *= 2

        self.middle_blks = nn.Sequential(
            *[
                NAFBlock(
                    chan,
                    dw_expand=dw_expand,
                    ffn_expand=ffn_expand,
                    drop_out_rate=drop_out_rate,
                    drop_path_rate=drop_path_rate * (block_idx + i) / max(total_blocks - 1, 1),
                )
                for i in range(middle_blk_num)
            ]
        )
        block_idx += middle_blk_num

        for n_blocks in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, kernel_size=1, bias=False),
                    nn.PixelShuffle(2),
                )
            )
            chan //= 2
            self.decoders.append(
                nn.Sequential(
                    *[
                        NAFBlock(
                            chan,
                            dw_expand=dw_expand,
                            ffn_expand=ffn_expand,
                            drop_out_rate=drop_out_rate,
                            drop_path_rate=drop_path_rate * (block_idx + i) / max(total_blocks - 1, 1),
                        )
                        for i in range(n_blocks)
                    ]
                )
            )
            block_idx += n_blocks

        self.padder_size = 2 ** len(self.encoders)

    def check_image_size(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        if pad_h == 0 and pad_w == 0:
            return x
        return F.pad(x, (0, pad_w, 0, pad_h))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        _, _, h, w = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            skips.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, skip in zip(self.decoders, self.ups, skips[::-1]):
            x = up(x)
            x = x + skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp
        return x[:, :, :h, :w]

    def get_num_params(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def test_nafnet() -> None:
    """Quick sanity test for NAFNet forward pass."""
    # Test without stochastic depth
    model = NAFNet(img_channel=3, width=32)
    x = torch.randn(2, 3, 128, 128)

    with torch.no_grad():
        y = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Parameters: {model.get_num_params():,}")
    assert y.shape == x.shape, "Output shape mismatch"

    # Test with stochastic depth
    model_sd = NAFNet(img_channel=3, width=32, drop_path_rate=0.1)
    model_sd.train()
    y_train = model_sd(x)
    model_sd.eval()
    with torch.no_grad():
        y_eval = model_sd(x)

    print(f"\nWith drop_path_rate=0.1:")
    print(f"  Train output shape: {y_train.shape}")
    print(f"  Eval output shape:  {y_eval.shape}")
    assert y_train.shape == x.shape, "Train output shape mismatch"
    assert y_eval.shape == x.shape, "Eval output shape mismatch"
    print("✅ Stochastic Depth test passed!")


if __name__ == "__main__":
    test_nafnet()
