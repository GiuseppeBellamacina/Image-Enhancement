"""
Inference utilities for full-resolution image restoration
Uses sliding window approach with overlapping patches for efficiency
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from pathlib import Path
import cv2


def normalize_image(img: np.ndarray) -> torch.Tensor:
    """
    Normalize image from [0, 255] uint8 to [-1, 1] float tensor.

    Args:
        img: Input image as numpy array (H, W, C) in range [0, 255]

    Returns:
        Normalized tensor (C, H, W) in range [-1, 1]
    """
    # Convert to float and normalize to [0, 1]
    img = img.astype(np.float32) / 255.0

    # Convert to tensor and transpose to (C, H, W)
    img_tensor = torch.from_numpy(img).permute(2, 0, 1)

    # Normalize to [-1, 1]
    img_tensor = (img_tensor - 0.5) / 0.5

    return img_tensor


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Denormalize tensor from [-1, 1] to [0, 255] uint8 image.

    Args:
        tensor: Input tensor (C, H, W) in range [-1, 1]

    Returns:
        Image as numpy array (H, W, C) in range [0, 255]
    """
    # Denormalize from [-1, 1] to [0, 1]
    img = (tensor * 0.5) + 0.5

    # Clamp to valid range
    img = torch.clamp(img, 0, 1)

    # Convert to numpy and transpose to (H, W, C)
    img = img.cpu().numpy().transpose(1, 2, 0)

    # Scale to [0, 255] and convert to uint8
    img = (img * 255).astype(np.uint8)

    return img


def sliding_window_inference(
    model: nn.Module,
    image: torch.Tensor,
    patch_size: int = 256,
    overlap: int = 32,
    device: str = "cuda",
    noise_sigma: Optional[float] = None,
) -> torch.Tensor:
    """
    Perform inference on full-resolution image using sliding window approach.

    Args:
        model: Trained model for restoration
        image: Input image tensor (C, H, W) in range [-1, 1]
        patch_size: Size of patches to process
        overlap: Overlap between adjacent patches (for smooth blending)
        device: Device to run inference on

    Returns:
        Restored image tensor (C, H, W) in range [-1, 1]
    """
    C, H, W = image.shape

    # Move image to device
    image = image.to(device)

    # Calculate stride (patch_size - overlap)
    stride = patch_size - overlap

    # Initialize output tensor and weight map for blending
    output = torch.zeros_like(image)
    weights = torch.zeros((H, W), device=device)

    # Create weight window for smooth blending (higher weight at center)
    window = create_blend_window(patch_size, device)

    # Calculate number of patches needed
    n_patches_h = (H - overlap + stride - 1) // stride
    n_patches_w = (W - overlap + stride - 1) // stride

    model.eval()
    with torch.no_grad():
        for i in range(n_patches_h):
            for j in range(n_patches_w):
                # Calculate patch coordinates
                y = min(i * stride, H - patch_size)
                x = min(j * stride, W - patch_size)

                # Extract patch
                patch = image[:, y : y + patch_size, x : x + patch_size]

                # Add batch dimension and process
                patch = patch.unsqueeze(0)
                if noise_sigma is not None:
                    restored_patch = model(patch, noise_sigma).squeeze(0)
                else:
                    restored_patch = model(patch).squeeze(0)

                # Accumulate with weights
                output[:, y : y + patch_size, x : x + patch_size] += (
                    restored_patch * window
                )
                weights[y : y + patch_size, x : x + patch_size] += window

    # Normalize by weight sum (add epsilon to avoid division by zero)
    eps = 1e-8
    output = output / (weights.unsqueeze(0) + eps)

    return output


def create_blend_window(size: int, device: str) -> torch.Tensor:
    """
    Create a 2D blending window with higher weights at center.
    Uses cosine window for smooth blending.

    Args:
        size: Size of the square window
        device: Device to create tensor on

    Returns:
        2D weight tensor (size, size)
    """
    # Create 1D cosine window
    window_1d = torch.hann_window(size, device=device)

    # Create 2D window by outer product
    window_2d = window_1d.unsqueeze(0) * window_1d.unsqueeze(1)

    return window_2d


def restore_image(
    model: nn.Module,
    image_path: Path,
    output_path: Optional[Path] = None,
    patch_size: int = 256,
    overlap: int = 32,
    device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Restore a full-resolution image from file.

    Args:
        model: Trained restoration model
        image_path: Path to degraded image
        output_path: Optional path to save restored image
        patch_size: Size of patches for sliding window
        overlap: Overlap between patches
        device: Device to run inference on

    Returns:
        Tuple of (degraded_image, restored_image) as numpy arrays (H, W, C)
    """
    # Load image
    degraded_bgr = cv2.imread(str(image_path))
    if degraded_bgr is None:
        raise ValueError(f"Could not load image from {image_path}")

    # Convert BGR to RGB
    degraded_rgb = cv2.cvtColor(degraded_bgr, cv2.COLOR_BGR2RGB)

    # Normalize to tensor
    degraded_tensor = normalize_image(degraded_rgb)

    # Run sliding window inference
    restored_tensor = sliding_window_inference(
        model=model,
        image=degraded_tensor,
        patch_size=patch_size,
        overlap=overlap,
        device=device,
    )

    # Denormalize to image
    restored_rgb = denormalize_image(restored_tensor)

    # Save if output path provided
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        restored_bgr = cv2.cvtColor(restored_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), restored_bgr)

    return degraded_rgb, restored_rgb


if __name__ == "__main__":
    # Test the functions
    print("Testing inference utilities...")

    # Create dummy image
    test_img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)

    # Test normalization
    tensor = normalize_image(test_img)
    print(f"Normalized tensor shape: {tensor.shape}")
    print(f"Normalized tensor range: [{tensor.min():.3f}, {tensor.max():.3f}]")

    # Test denormalization
    img_back = denormalize_image(tensor)
    print(f"Denormalized image shape: {img_back.shape}")
    print(f"Denormalized image range: [{img_back.min()}, {img_back.max()}]")

    # Test blend window
    window = create_blend_window(128, "cpu")
    print(f"Blend window shape: {window.shape}")
    print(f"Blend window range: [{window.min():.3f}, {window.max():.3f}]")
