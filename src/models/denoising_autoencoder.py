"""
Denoising Autoencoder Model
"""
import torch

class DenoisingAutoencoder(torch.nn.Module):
    def __init__(self, features=32):
        super(DenoisingAutoencoder, self).__init__()

        # Encoder
        self.encoder = torch.nn.Sequential(
            #128x128 -> 64x64
            torch.nn.Conv2d(3, features, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(features),
            torch.nn.LeakyReLU(0.2, inplace=True),

            #64x64 -> 32x32
            torch.nn.Conv2d(features, features*2, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(features*2),
            torch.nn.LeakyReLU(0.2, inplace=True),

            #32x32 -> 16x16
            torch.nn.Conv2d(features*2, features*4, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(features*4),
            torch.nn.LeakyReLU(0.2, inplace=True),

            # Bottleneck
            #16x16 -> 8x8 
            torch.nn.Conv2d(features*4, features*8, kernel_size=3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(features*8),
            torch.nn.LeakyReLU(0.2, inplace=True),
        )

        # Decoder
        self.decoder = torch.nn.Sequential(
            #8x8 -> 16x16
            torch.nn.ConvTranspose2d(features*8, features * 4, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            torch.nn.BatchNorm2d(features * 4),
            torch.nn.ReLU(inplace=True),

            #16x16 -> 32x32
            torch.nn.ConvTranspose2d(features * 4, features * 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            torch.nn.BatchNorm2d(features * 2),
            torch.nn.ReLU(inplace=True),
            
            #32x32 -> 64x64
            torch.nn.ConvTranspose2d(features * 2, features, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            torch.nn.BatchNorm2d(features),
            torch.nn.ReLU(inplace=True),

            #64x64 -> 128x128 
            torch.nn.ConvTranspose2d(features, 3, kernel_size=3, stride=2, padding=1, output_padding=1),   
            torch.nn.Tanh(),  # Output between -1 and 1
        )

    def forward(self, x):   
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
if __name__ == "__main__":
    # Test the Denoising Autoencoder
    model = DenoisingAutoencoder()
    sample_image = torch.randn(1, 3, 128, 128)  # Batch size of 1, 3 channels, 128x128 image
    

    with torch.inference_mode():
        output = model(sample_image)
        
    print("Input shape:", sample_image.shape)
    print("Shape at Bottleneck:", model.encoder(sample_image).shape)
    print("Output shape:", output.shape)

    if sample_image.shape == output.shape:
        print("✅ Output shape matches input shape.")
    else:
        raise RuntimeError("Output shape does not match input shape.")