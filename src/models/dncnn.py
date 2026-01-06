import torch
import torch.nn as nn


class DnCNN(nn.Module):
    """
    DnCNN: Denoising Convolutional Neural Network
    
    Architecture:
    - Layer 1: Conv(3→64) + ReLU (no BN)
    - Layers 2-16: Conv(64→64) + BN + ReLU (x15)
    - Layer 17: Conv(64→3) (output noise residual)
    """
    
    def __init__(
        self, 
        in_channels: int = 3, 
        out_channels: int = 3, 
        num_layers: int = 17, 
        features: int = 64
    ):
        """
        Args:
            in_channels: Number of input channels (3 for RGB)
            out_channels: Number of output channels (3 for RGB)
            num_layers: Total number of convolutional layers (default: 17)
            features: Number of feature maps in hidden layers (default: 64)
        """
        super(DnCNN, self).__init__()
        
        self.num_layers = num_layers
        
        layers = []
        
        # Layer 1: Conv + ReLU (NO Batch Normalization)
        layers.append(
            nn.Conv2d(
                in_channels, 
                features, 
                kernel_size=3, 
                padding=1, 
                bias=True  # bias=True perché non c'è BN
            )
        )
        layers.append(nn.ReLU(inplace=True))
        
        # Layers 2 to (num_layers - 1): Conv + BN + ReLU
        for _ in range(num_layers - 2):
            layers.append(
                nn.Conv2d(
                    features, 
                    features, 
                    kernel_size=3, 
                    padding=1, 
                    bias=False  # bias=False perché c'è BN dopo
                )
            )
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        
        # Layer num_layers: Conv finale (predice il rumore)
        layers.append(
            nn.Conv2d(
                features, 
                out_channels, 
                kernel_size=3, 
                padding=1, 
                bias=True
            )
        )
        
        self.dncnn = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass con Residual Learning
        
        Args:
            x: Immagine rumorosa [B, C, H, W]
        
        Returns:
            Immagine denoised [B, C, H, W]
        """
        # Predici il rumore
        noise = self.dncnn(x)

        output = x - noise

        if not self.training:
            output = torch.clamp(output, min=-1.0, max=1.0)
        
        # Residual learning: sottrai il rumore dall'input
        # clean = noisy - noise
        return output
    
    def get_num_params(self):
        """Conta i parametri addestrabili"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = DnCNN(in_channels=3, out_channels=3, num_layers=17, features=64)
    print(f"DnCNN-17 parameters: {model.get_num_params():,}")
    
    # Test forward pass
    x = torch.randn(4, 3, 128, 128)  # Batch di 4 immagini 128×128
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    assert x.shape == y.shape, "Input and output must have same shape!"