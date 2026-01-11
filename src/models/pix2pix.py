"""
Pix2Pix GAN for Image-to-Image Translation

Architecture:
- Generator: UNet-based with skip connections
- Discriminator: PatchGAN (70x70 receptive field)

Reference:
    Image-to-Image Translation with Conditional Adversarial Networks
    Isola et al., CVPR 2017
"""

import torch
import torch.nn as nn


class Pix2PixGenerator(nn.Module):
    """
    UNet-based Generator for Pix2Pix.
    
    Encoder-decoder architecture with skip connections that preserve
    spatial information from encoder to decoder.
    
    Args:
        in_channels: Number of input channels (default: 3 for RGB)
        out_channels: Number of output channels (default: 3 for RGB)
        features: Base number of features (default: 64)
        use_tanh: Whether to use tanh activation on output (default: True)
                  Set to False for residual learning where output is noise
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 3, features: int = 64, use_tanh: bool = True):
        super().__init__()
        
        self.use_tanh = use_tanh
        
        # Encoder (downsampling) - 6 layers for 256x256 input
        self.enc1 = self._conv_block(in_channels, features, normalize=False)  # No BN in first layer
        self.enc2 = self._conv_block(features, features * 2)
        self.enc3 = self._conv_block(features * 2, features * 4)
        self.enc4 = self._conv_block(features * 4, features * 8)
        self.enc5 = self._conv_block(features * 8, features * 8)
        self.enc6 = self._conv_block(features * 8, features * 8)
        
        # Bottleneck
        self.bottleneck = self._conv_block(features * 8, features * 8, normalize=False)
        
        # Decoder (upsampling) - 6 layers to match encoder
        self.dec1 = self._upconv_block(features * 8, features * 8, dropout=True)
        self.dec2 = self._upconv_block(features * 16, features * 8, dropout=True)  # *16 due to concatenation
        self.dec3 = self._upconv_block(features * 16, features * 8, dropout=True)
        self.dec4 = self._upconv_block(features * 16, features * 4)
        self.dec5 = self._upconv_block(features * 8, features * 2)
        self.dec6 = self._upconv_block(features * 4, features)
        
        # Final layer (configurable activation)
        if use_tanh:
            self.final = nn.Sequential(
                nn.ConvTranspose2d(features * 2, out_channels, kernel_size=4, stride=2, padding=1),
                nn.Tanh()  # Output in [-1, 1] for direct image prediction
            )
        else:
            # No activation for residual learning (predicting noise)
            self.final = nn.ConvTranspose2d(features * 2, out_channels, kernel_size=4, stride=2, padding=1)
        
        # No need for separate pooling - downsampling is done in conv blocks
        
    def _conv_block(self, in_channels: int, out_channels: int, normalize: bool = True):
        """Encoder block: Conv (stride=2 for downsampling) -> BatchNorm -> LeakyReLU"""
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)
    
    def _upconv_block(self, in_channels: int, out_channels: int, dropout: bool = False):
        """Decoder block: ConvTranspose -> BatchNorm -> Dropout -> ReLU"""
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass with skip connections.
        
        Args:
            x: Input tensor (B, 3, H, W) in range [-1, 1]
            
        Returns:
            If use_tanh=True: Output tensor (B, 3, H, W) in range [-1, 1] (restored image)
            If use_tanh=False: Output tensor (B, 3, H, W) unbounded (predicted noise/residual)
        """
        # Encoder with skip connections (downsampling via stride=2)
        e1 = self.enc1(x)           # (B, 64, H/2, W/2)
        e2 = self.enc2(e1)          # (B, 128, H/4, W/4)
        e3 = self.enc3(e2)          # (B, 256, H/8, W/8)
        e4 = self.enc4(e3)          # (B, 512, H/16, W/16)
        e5 = self.enc5(e4)          # (B, 512, H/32, W/32)
        e6 = self.enc6(e5)          # (B, 512, H/64, W/64)
        
        # Bottleneck
        b = self.bottleneck(e6)     # (B, 512, H/128, W/128)
        
        # Decoder with skip connections (upsampling via stride=2)
        d1 = self.dec1(b)                   # (B, 512, H/64, W/64)
        d1 = torch.cat([d1, e6], dim=1)     # (B, 1024, H/64, W/64)
        
        d2 = self.dec2(d1)                  # (B, 512, H/32, W/32)
        d2 = torch.cat([d2, e5], dim=1)     # (B, 1024, H/32, W/32)
        
        d3 = self.dec3(d2)                  # (B, 512, H/16, W/16)
        d3 = torch.cat([d3, e4], dim=1)     # (B, 1024, H/16, W/16)
        
        d4 = self.dec4(d3)                  # (B, 256, H/8, W/8)
        d4 = torch.cat([d4, e3], dim=1)     # (B, 512, H/8, W/8)
        
        d5 = self.dec5(d4)                  # (B, 128, H/4, W/4)
        d5 = torch.cat([d5, e2], dim=1)     # (B, 256, H/4, W/4)
        
        d6 = self.dec6(d5)                  # (B, 64, H/2, W/2)
        d6 = torch.cat([d6, e1], dim=1)     # (B, 128, H/2, W/2)
        
        # Final output (upsample to original size)
        out = self.final(d6)                # (B, 3, H, W)
        
        return out
    
    def get_num_params(self):
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN Discriminator for Pix2Pix.
    
    Classifies whether 70x70 patches are real or fake.
    Input: Concatenation of condition image and target/generated image.
    
    Args:
        in_channels: Number of input channels (6 = 3 for condition + 3 for target)
        features: Base number of features (default: 64)
    """
    
    def __init__(self, in_channels: int = 6, features: int = 64):
        super().__init__()
        
        # PatchGAN architecture
        # Each layer has receptive field that grows to 70x70
        self.model = nn.Sequential(
            # Input: (B, 6, H, W)
            self._conv_block(in_channels, features, normalize=False),  # (B, 64, H/2, W/2)
            self._conv_block(features, features * 2),                  # (B, 128, H/4, W/4)
            self._conv_block(features * 2, features * 4),              # (B, 256, H/8, W/8)
            self._conv_block(features * 4, features * 8, stride=1),    # (B, 512, H/8, W/8)
            # Final layer: outputs LOGITS (no sigmoid - use BCEWithLogitsLoss)
            nn.Conv2d(features * 8, 1, kernel_size=4, stride=1, padding=1),  # (B, 1, H/8, W/8)
        )
        
    def _conv_block(self, in_channels: int, out_channels: int, stride: int = 2, normalize: bool = True):
        """Discriminator block: Conv -> BatchNorm -> LeakyReLU"""
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)
    
    def forward(self, condition, target):
        """
        Forward pass.
        
        Args:
            condition: Condition image (B, 3, H, W) - e.g., degraded image
            target: Target image (B, 3, H, W) - e.g., clean or generated image
            
        Returns:
            Logit map (B, 1, H/8, W/8) where each pixel represents
            logit score for corresponding 70x70 patch (use with BCEWithLogitsLoss)
        """
        # Concatenate condition and target
        x = torch.cat([condition, target], dim=1)  # (B, 6, H, W)
        
        # Pass through discriminator (output logits, not probabilities)
        out = self.model(x)  # (B, 1, H/8, W/8)
        
        return out
    
    def get_num_params(self):
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Test code
if __name__ == "__main__":
    # Test generator
    print("Testing Pix2Pix Generator:")
    gen = Pix2PixGenerator(in_channels=3, out_channels=3, features=64)
    x = torch.randn(2, 3, 256, 256)  # Batch of 2 images
    y = gen(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    print(f"  Output range: [{y.min():.3f}, {y.max():.3f}]")
    print(f"  Parameters: {gen.get_num_params():,}")
    
    print("\nTesting PatchGAN Discriminator:")
    disc = PatchGANDiscriminator(in_channels=6, features=64)
    condition = torch.randn(2, 3, 256, 256)
    target = torch.randn(2, 3, 256, 256)
    out = disc(condition, target)
    print(f"  Condition shape: {condition.shape}")
    print(f"  Target shape: {target.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Output range: [{out.min():.3f}, {out.max():.3f}]")
    print(f"  Parameters: {disc.get_num_params():,}")
