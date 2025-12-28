"""
Gaussian Noise Degradation

Adds Gaussian noise to images with configurable standard deviation.
"""

import numpy as np
from typing import Union, Optional


def add_gaussian_noise(
    image: np.ndarray, sigma: Union[int, float] = 25, seed: Optional[int] = None
) -> np.ndarray:
    """
    Add Gaussian noise to an image.

    Args:
        image: Input image as numpy array (H, W, C) in range [0, 255]
        sigma: Standard deviation of Gaussian noise (default: 25)
        seed: Random seed for reproducibility

    Returns:
        Noisy image as numpy array in range [0, 255]

    Example:
        >>> img = cv2.imread('image.jpg')
        >>> noisy = add_gaussian_noise(img, sigma=25)
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate Gaussian noise
    noise = np.random.randn(*image.shape) * sigma

    # Add noise to image
    noisy_image = image + noise

    # Clip to valid range
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    return noisy_image


if __name__ == "__main__":
    # Test the function

    # Create a test image
    test_img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

    # Add noise with different sigma values
    for sigma in [5, 15, 25, 50]:
        noisy = add_gaussian_noise(test_img, sigma=sigma, seed=42)
        print(
            f"Sigma {sigma}: min={noisy.min()}, max={noisy.max()}, mean={noisy.mean():.2f}"
        )
