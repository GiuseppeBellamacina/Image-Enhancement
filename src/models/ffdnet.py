"""
FFDNet implementation (PyTorch).

Reference idea:
- Downsample input via pixel-unshuffle (scale factor 2)
- Build a noise level map (sigma) and downsample it as well
- Concatenate [x_down, sigma_down] along channels
- Predict noise in the downsampled space (output channels = in_channels * r^2)
- Upsample predicted noise via pixel-shuffle and subtract from input

This model expects a noise level input (sigma), typically expressed in [0, 255] (uint8 scale)
and internally normalized to [0, 1] for the noise map (sigma/255).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


TensorOrScalar = Union[torch.Tensor, float, int]


def _pixel_unshuffle(x: torch.Tensor, downscale_factor: int = 2) -> torch.Tensor:
    """
    Space-to-depth. (B, C, H, W) -> (B, C*r^2, H/r, W/r)
    """
    if x.dim() != 4:
        raise ValueError(f"Expected 4D tensor (B,C,H,W), got {x.shape}")
    b, c, h, w = x.shape
    r = downscale_factor
    if h % r != 0 or w % r != 0:
        raise ValueError(f"H and W must be divisible by {r}. Got H={h}, W={w}")
    x = x.view(b, c, h // r, r, w // r, r)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    return x.view(b, c * r * r, h // r, w // r)


def _pixel_shuffle(x: torch.Tensor, upscale_factor: int = 2) -> torch.Tensor:
    """
    Depth-to-space. (B, C*r^2, H, W) -> (B, C, H*r, W*r)
    """
    if x.dim() != 4:
        raise ValueError(f"Expected 4D tensor (B,C,H,W), got {x.shape}")
    b, cr2, h, w = x.shape
    r = upscale_factor
    if cr2 % (r * r) != 0:
        raise ValueError(f"Channels must be divisible by r^2={r*r}. Got C={cr2}")
    c = cr2 // (r * r)
    x = x.view(b, c, r, r, h, w)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    return x.view(b, c, h * r, w * r)


@dataclass
class FFDNetConfig:
    in_channels: int = 3
    num_features: int = 96
    num_layers: int = 12  # 12 for color images as per paper
    scale_factor: int = 2
    # Noise map normalization: sigma is assumed in [0, 255] unless changed
    sigma_divisor: float = 127.5 # because input images are in [-1, 1]
    # Clamp output in eval mode (matches your DnCNN behavior)
    output_range: Tuple[float, float] = (-1.0, 1.0)
    clamp_in_eval: bool = True
    residual_learning: bool = False  # FFDNet predicts clean image directly


class FFDNet(nn.Module):
    """
    FFDNet for blind Gaussian denoising with known noise level (sigma).

    Forward:
        denoised = model(x_noisy, sigma)

    sigma can be:
    - float/int scalar (applied to whole batch)
    - Tensor of shape [B], [B,1], [B,1,1,1], or [B,1,H,W]
      Values are assumed in [0, 255] unless you set sigma_divisor accordingly.
    """

    def __init__(self, config: Optional[FFDNetConfig] = None, **kwargs):
        super().__init__()
        if config is None:
            config = FFDNetConfig(**kwargs)
        self.cfg = config

        r = self.cfg.scale_factor
        c = self.cfg.in_channels

        # After pixel-unshuffle: image channels become c*r^2
        image_down_ch = c * r * r
        # Noise map is concatenated as 1 channel (not unshuffled): 4C+1 as per paper
        sigma_down_ch = 1
        in_ch = image_down_ch + sigma_down_ch
        out_ch = image_down_ch  # predict clean image in downsampled space

        layers = []
        # First layer: Conv + ReLU (no BN)
        layers.append(nn.Conv2d(in_ch, self.cfg.num_features, kernel_size=3, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))

        # Intermediate layers: Conv + BN + ReLU (as per paper)
        for _ in range(self.cfg.num_layers - 2):
            layers.append(nn.Conv2d(self.cfg.num_features, self.cfg.num_features, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(self.cfg.num_features))
            layers.append(nn.ReLU(inplace=True))

        # Last layer: Conv only (no BN, no ReLU)
        layers.append(nn.Conv2d(self.cfg.num_features, out_ch, kernel_size=3, padding=1, bias=False))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        """Orthogonal initialization as recommended in the FFDNet paper."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @staticmethod
    def get_num_params(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())

    def _sigma_to_map(self, x: torch.Tensor, sigma: TensorOrScalar = 100.0) -> torch.Tensor:
        """
        Returns sigma map of shape [B, 1, H, W] on x.device and x.dtype.
        """
        b, _, h, w = x.shape
        device = x.device
        dtype = x.dtype

        if isinstance(sigma, (float, int)):
            s = torch.tensor(float(sigma), device=device, dtype=dtype).view(1, 1, 1, 1).expand(b, 1, 1, 1)
        elif isinstance(sigma, torch.Tensor):
            s = sigma.to(device=device, dtype=dtype)
            if s.dim() == 1:          # [B]
                s = s.view(b, 1, 1, 1)
            elif s.dim() == 2:        # [B,1] or [B,C]
                s = s[:, :1].view(b, 1, 1, 1)
            elif s.dim() == 4:        # [B,1,H,W]
                if s.shape[1] != 1:
                    s = s[:, :1, :, :]
                if s.shape[2] != h or s.shape[3] != w:
                    # allow broadcasting if [B,1,1,1] like shapes
                    if s.shape[2] == 1 and s.shape[3] == 1:
                        pass
                    else:
                        raise ValueError(f"sigma map spatial size must match input: got {s.shape}, input {x.shape}")
            else:
                raise ValueError(f"Unsupported sigma tensor shape: {s.shape}")
        else:
            raise TypeError(f"Unsupported sigma type: {type(sigma)}")

        # Normalize sigma to [0,1] typical for FFDNet conditioning
        s = s / float(self.cfg.sigma_divisor)
        return s.expand(b, 1, h, w)

    def forward(self, x: torch.Tensor, sigma: TensorOrScalar = 100.0) -> torch.Tensor:
        """
        Args:
            x: noisy image tensor [B,C,H,W] (typically normalized, e.g. [-1,1])
            sigma: noise std (typically in [0,255] scale)

        Returns:
            denoised image [B,C,H,W] in same scale as x
        """
        if x.dim() != 4:
            raise ValueError(f"Expected x with shape [B,C,H,W], got {x.shape}")
        if x.shape[1] != self.cfg.in_channels:
            raise ValueError(f"Expected in_channels={self.cfg.in_channels}, got {x.shape[1]}")

        r = self.cfg.scale_factor

        sigma_map = self._sigma_to_map(x, sigma)               # [B,1,H,W]
        x_down = _pixel_unshuffle(x, r)                        # [B,C*r^2,H/r,W/r]
        
        # Downsample sigma_map spatially (to match x_down size) but keep as 1 channel
        # Paper: input is W/2 x H/2 x (4C+1)
        h_down, w_down = x_down.shape[2], x_down.shape[3]
        sigma_down = nn.functional.interpolate(
            sigma_map, size=(h_down, w_down), mode='nearest'
        )  # [B,1,H/r,W/r]

        inp = torch.cat([x_down, sigma_down], dim=1)           # [B,C*r^2+1,H/r,W/r]
        pred_down = self.net(inp)                              # [B,C*r^2,H/r,W/r]
        pred = _pixel_shuffle(pred_down, r)                    # [B,C,H,W]

        # FFDNet predicts the clean image directly (no residual learning)
        if self.cfg.residual_learning:
            out = x - pred  # residual: predict noise and subtract
        else:
            out = pred      # direct: predict clean image

        if (not self.training) and self.cfg.clamp_in_eval and self.cfg.output_range is not None:
            lo, hi = self.cfg.output_range
            out = torch.clamp(out, min=float(lo), max=float(hi))

        return out