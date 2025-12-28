"""
Visualization utilities for image enhancement experiments
Functions for plotting, denormalization, and displaying results
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional


def denormalize_tensor(
    tensor: torch.Tensor,
    mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    std: Tuple[float, float, float] = (0.5, 0.5, 0.5)
) -> np.ndarray:
    """
    Denormalize a tensor and convert to numpy array for visualization.
    
    Args:
        tensor: Input tensor in (C, H, W) format, range [-1, 1]
        mean: Normalization mean used during preprocessing
        std: Normalization std used during preprocessing
    
    Returns:
        Numpy array in (H, W, C) format, range [0, 1]
    """
    # Convert to numpy and transpose
    img = tensor.cpu().numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
    
    # Denormalize
    img = img * np.array(std) + np.array(mean)
    
    # Clip to valid range
    img = np.clip(img, 0, 1)
    
    return img


def plot_image_comparison(
    degraded_batch: torch.Tensor,
    clean_batch: torch.Tensor,
    n_samples: int = 4,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (16, 8)
) -> None:
    """
    Plot comparison between degraded and clean images.
    
    Args:
        degraded_batch: Batch of degraded images (B, C, H, W)
        clean_batch: Batch of clean images (B, C, H, W)
        n_samples: Number of samples to display
        save_path: Path to save the figure (optional)
        figsize: Figure size (width, height)
    """
    n_samples = min(n_samples, len(degraded_batch))
    fig, axes = plt.subplots(2, n_samples, figsize=figsize)
    
    # Handle single sample case
    if n_samples == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(n_samples):
        # Denormalize
        deg_img = denormalize_tensor(degraded_batch[i])
        clean_img = denormalize_tensor(clean_batch[i])
        
        # Degraded
        axes[0, i].imshow(deg_img)
        axes[0, i].set_title('Degraded', fontweight='bold')
        axes[0, i].axis('off')
        
        # Clean
        axes[1, i].imshow(clean_img)
        axes[1, i].set_title('Clean Target', fontweight='bold')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_inference_results(
    degraded_batch: torch.Tensor,
    restored_batch: torch.Tensor,
    clean_batch: torch.Tensor,
    n_samples: int = 4,
    save_path: Optional[Path] = None,
    figsize: Optional[Tuple[int, int]] = None
) -> None:
    """
    Plot inference results: degraded input, restored output, clean target.
    
    Args:
        degraded_batch: Batch of degraded images (B, C, H, W)
        restored_batch: Batch of restored images (B, C, H, W)
        clean_batch: Batch of clean target images (B, C, H, W)
        n_samples: Number of samples to display
        save_path: Path to save the figure (optional)
        figsize: Figure size (width, height), auto-calculated if None
    """
    n_samples = min(n_samples, len(degraded_batch))
    
    if figsize is None:
        figsize = (4 * n_samples, 12)
    
    fig, axes = plt.subplots(3, n_samples, figsize=figsize)
    
    # Handle single sample case
    if n_samples == 1:
        axes = axes.reshape(3, 1)
    
    for i in range(n_samples):
        # Denormalize
        deg = denormalize_tensor(degraded_batch[i])
        restored = denormalize_tensor(restored_batch[i])
        clean = denormalize_tensor(clean_batch[i])
        
        # Degraded input
        axes[0, i].imshow(deg)
        axes[0, i].set_title('Degraded Input', fontweight='bold', fontsize=11)
        axes[0, i].axis('off')
        
        # Restored output
        axes[1, i].imshow(restored)
        axes[1, i].set_title('Restored Output', fontweight='bold', fontsize=11)
        axes[1, i].axis('off')
        
        # Clean target
        axes[2, i].imshow(clean)
        axes[2, i].set_title('Clean Target', fontweight='bold', fontsize=11)
        axes[2, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_training_curves(
    history: dict,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (16, 12)
) -> None:
    """
    Plot training curves from history dictionary.
    
    Args:
        history: Dictionary with training history containing:
                 'train_loss', 'val_loss', 'train_l1', 'val_l1',
                 'train_ssim', 'val_ssim', 'lr'
        save_path: Path to save the figure (optional)
        figsize: Figure size (width, height)
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Total Loss
    axes[0, 0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Val', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Total Loss', fontsize=12)
    axes[0, 0].set_title('Combined Loss (L1 + SSIM)', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(alpha=0.3)
    
    # L1 Loss
    axes[0, 1].plot(history['train_l1'], label='Train', linewidth=2)
    axes[0, 1].plot(history['val_l1'], label='Val', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('L1 Loss', fontsize=12)
    axes[0, 1].set_title('L1 Loss', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(alpha=0.3)
    
    # SSIM
    axes[1, 0].plot(history['train_ssim'], label='Train', linewidth=2)
    axes[1, 0].plot(history['val_ssim'], label='Val', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('SSIM', fontsize=12)
    axes[1, 0].set_title('SSIM (higher is better)', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(alpha=0.3)
    
    # Learning Rate
    axes[1, 1].plot(history['lr'], linewidth=2, color='green')
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Learning Rate', fontsize=12)
    axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
