"""
Denoising Autoencoder Model
"""
import torch

class DenoisingAutoencoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
        )

        # Decoder
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.Sigmoid(),  # To ensure output is between 0 and 1
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
    print("Output shape:", output.shape)

    if sample_image.shape == output.shape:
        print("✅ Output shape matches input shape.")
    else:
        raise RuntimeError("Output shape does not match input shape.")