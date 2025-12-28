"""
Color Quantization and Dithering Degradation

Reduces color depth and applies various dithering algorithms to simulate
low bit-depth images, vintage effects, or palette-limited formats.
"""

import numpy as np
from typing import Optional, Literal


def quantize_image(image: np.ndarray, bits_per_channel: int = 4) -> np.ndarray:
    """
    Quantize image to specified bit depth per channel.

    Args:
        image: Input image as numpy array (H, W, C) in range [0, 255]
        bits_per_channel: Number of bits per channel (1-8)

    Returns:
        Quantized image in range [0, 255]

    Example:
        >>> img = cv2.imread('image.jpg')
        >>> quantized = quantize_image(img, bits_per_channel=4)  # 16 colors per channel
    """
    # Calculate number of quantization levels
    levels = 2**bits_per_channel

    # Quantize
    quantized = np.floor(image / 256 * levels) * (256 / levels)

    return np.clip(quantized, 0, 255).astype(np.uint8)


def apply_random_dithering(
    image: np.ndarray,
    bits_per_channel: int = 4,
    noise_strength: float = 1.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Apply random dithering (additive noise before quantization).

    Args:
        image: Input image as numpy array (H, W, C) in range [0, 255]
        bits_per_channel: Target bit depth
        noise_strength: Strength of random noise (0-2, default 1.0)
        seed: Random seed for reproducibility

    Returns:
        Dithered and quantized image
    """
    if seed is not None:
        np.random.seed(seed)

    levels = 2**bits_per_channel
    step = 256 / levels

    # Add random noise scaled to quantization step
    noise = np.random.uniform(-step / 2, step / 2, image.shape) * noise_strength
    noisy_image = image + noise

    # Quantize
    quantized = np.floor(noisy_image / 256 * levels) * (256 / levels)

    return np.clip(quantized, 0, 255).astype(np.uint8)


def apply_ordered_dithering(
    image: np.ndarray,
    bits_per_channel: int = 4,
    pattern: Literal["bayer2", "bayer4", "bayer8"] = "bayer4",
) -> np.ndarray:
    """
    Apply ordered (Bayer) dithering.

    Args:
        image: Input image as numpy array (H, W, C) in range [0, 255]
        bits_per_channel: Target bit depth
        pattern: Bayer matrix size ('bayer2', 'bayer4', 'bayer8')

    Returns:
        Dithered and quantized image
    """
    # Bayer matrices (normalized to [0, 1])
    bayer_matrices = {
        "bayer2": np.array([[0, 2], [3, 1]]) / 4.0,
        "bayer4": np.array(
            [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]
        )
        / 16.0,
        "bayer8": np.array(
            [
                [0, 32, 8, 40, 2, 34, 10, 42],
                [48, 16, 56, 24, 50, 18, 58, 26],
                [12, 44, 4, 36, 14, 46, 6, 38],
                [60, 28, 52, 20, 62, 30, 54, 22],
                [3, 35, 11, 43, 1, 33, 9, 41],
                [51, 19, 59, 27, 49, 17, 57, 25],
                [15, 47, 7, 39, 13, 45, 5, 37],
                [63, 31, 55, 23, 61, 29, 53, 21],
            ]
        )
        / 64.0,
    }

    bayer = bayer_matrices[pattern]
    h, w, c = image.shape
    bayer_h, bayer_w = bayer.shape

    levels = 2**bits_per_channel
    step = 256 / levels

    # Tile Bayer matrix to match image size
    bayer_tiled = np.tile(bayer, (h // bayer_h + 1, w // bayer_w + 1))[:h, :w]

    # Apply to each channel
    result = np.zeros_like(image, dtype=np.float32)
    for ch in range(c):
        # Add Bayer threshold
        threshold = (bayer_tiled - 0.5) * step
        dithered = image[:, :, ch].astype(np.float32) + threshold

        # Quantize
        result[:, :, ch] = np.floor(dithered / 256 * levels) * (256 / levels)

    return np.clip(result, 0, 255).astype(np.uint8)


def apply_floyd_steinberg_dithering(
    image: np.ndarray, bits_per_channel: int = 4
) -> np.ndarray:
    """
    Apply Floyd-Steinberg error diffusion dithering.
    This is a more sophisticated dithering method that produces better quality.

    Args:
        image: Input image as numpy array (H, W, C) in range [0, 255]
        bits_per_channel: Target bit depth

    Returns:
        Dithered and quantized image
    """
    levels = 2**bits_per_channel
    h, w, c = image.shape

    # Work with float for error propagation
    result = image.astype(np.float32).copy()

    # Error diffusion kernel (Floyd-Steinberg)
    # Distributes error to: right (7/16), bottom-left (3/16),
    # bottom (5/16), bottom-right (1/16)

    for ch in range(c):
        for y in range(h):
            for x in range(w):
                old_pixel = result[y, x, ch]

                # Quantize
                new_pixel = np.round(old_pixel / 256 * (levels - 1)) * (
                    256 / (levels - 1)
                )
                result[y, x, ch] = new_pixel

                # Calculate error
                error = old_pixel - new_pixel

                # Distribute error to neighboring pixels
                if x + 1 < w:
                    result[y, x + 1, ch] += error * 7 / 16
                if y + 1 < h:
                    if x > 0:
                        result[y + 1, x - 1, ch] += error * 3 / 16
                    result[y + 1, x, ch] += error * 5 / 16
                    if x + 1 < w:
                        result[y + 1, x + 1, ch] += error * 1 / 16

    return np.clip(result, 0, 255).astype(np.uint8)


def apply_quantization_dithering(
    image: np.ndarray,
    bits_per_channel: int = 4,
    dithering_type: Literal[
        "none", "random", "bayer2", "bayer4", "bayer8", "floyd_steinberg"
    ] = "random",
    noise_strength: float = 1.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Universal function to apply color quantization with various dithering methods.

    Args:
        image: Input image as numpy array (H, W, C) in range [0, 255]
        bits_per_channel: Number of bits per channel (1-8)
        dithering_type: Type of dithering to apply
        noise_strength: For random dithering only
        seed: Random seed for reproducibility

    Returns:
        Quantized and dithered image

    Example:
        >>> # Low quality (corrupted)
        >>> corrupted = apply_quantization_dithering(img, bits_per_channel=3, dithering_type='random')

        >>> # High quality (target for training)
        >>> target = apply_quantization_dithering(img, bits_per_channel=6, dithering_type='floyd_steinberg')
    """
    if dithering_type == "none":
        return quantize_image(image, bits_per_channel)
    elif dithering_type == "random":
        return apply_random_dithering(image, bits_per_channel, noise_strength, seed)
    elif dithering_type in ["bayer2", "bayer4", "bayer8"]:
        # Type narrowing: at this point dithering_type can only be one of the bayer patterns
        pattern: Literal["bayer2", "bayer4", "bayer8"] = dithering_type  # type: ignore
        return apply_ordered_dithering(image, bits_per_channel, pattern)
    elif dithering_type == "floyd_steinberg":
        return apply_floyd_steinberg_dithering(image, bits_per_channel)
    else:
        raise ValueError(f"Unknown dithering type: {dithering_type}")


if __name__ == "__main__":
    # Test the functions
    print("Testing Quantization and Dithering...")

    # Create a gradient test image
    test_img = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(256):
        test_img[:, i, :] = i  # Horizontal gradient

    # Test different methods
    methods: list[
        tuple[
            Literal["none", "random", "bayer2", "bayer4", "bayer8", "floyd_steinberg"],
            int,
            float,
        ]
    ] = [
        ("none", 4, 1.0),
        ("random", 4, 1.0),
        ("bayer4", 4, 1.0),
        ("floyd_steinberg", 4, 1.0),
        ("random", 2, 1.0),  # Very low bit depth
    ]

    for method, bits, strength in methods:
        result = apply_quantization_dithering(
            test_img,
            bits_per_channel=bits,
            dithering_type=method,
            noise_strength=strength,
            seed=42,
        )
        unique_values = len(np.unique(result[:, :, 0]))
        print(
            f"{method} @ {bits}bit: {unique_values} unique values (expected ~{2**bits})"
        )

    print("\nTraining pair examples:")
    print("Corrupted: random dithering @ 3-bit")
    print("Target 1: Original image (8-bit)")
    print("Target 2: Floyd-Steinberg @ 6-bit (high-quality dithering)")
