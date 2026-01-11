# -*- coding: utf-8 -*-
"""
Shared training utilities for all training loops.
"""

import torch
from tqdm.auto import tqdm


def cleanup_cuda_memory(device: str, *tensors) -> None:
    """
    Clean up CUDA memory by clearing cache and deleting tensors.
    
    Args:
        device: Device string ('cuda' or 'cpu')
        *tensors: Variable number of tensors to delete
    """
    # Delete tensors
    for tensor in tensors:
        if tensor is not None:
            del tensor
    
    # Clear CUDA cache if on GPU
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def handle_oom_error(
    batch_idx: int,
    total_batches: int,
    device: str,
    *tensors,
    is_training: bool = True
) -> None:
    """
    Handle CUDA Out of Memory errors with cleanup and user guidance.
    
    Args:
        batch_idx: Current batch index (0-indexed)
        total_batches: Total number of batches
        device: Device string ('cuda' or 'cpu')
        *tensors: Tensors to clean up
        is_training: Whether error occurred during training (vs validation)
        
    Raises:
        torch.cuda.OutOfMemoryError: Always re-raises after cleanup
    """
    phase = "training" if is_training else "validation"
    
    print(f"\n❌ CUDA Out of Memory during {phase} at batch {batch_idx + 1}/{total_batches}!")
    print("   Attempting recovery...")
    
    # Clean up memory
    cleanup_cuda_memory(device, *tensors)
    print("   ✅ CUDA cache cleared")
    
    print(f"\n⚠️  OOM ERROR DETECTED - {phase.upper()} HALTED")
    print("   Suggestions to prevent OOM:")
    print("   1. Reduce batch_size in config")
    print("   2. Reduce patch_size in config")
    print("   3. Enable mixed precision (use_amp=True)")
    print("   4. Reduce model_features/channels in config")
    
    if is_training:
        print("\n   The training state has been saved.")
        print("   You can resume from the last checkpoint after adjusting config.")
    
    # Re-raise with informative message
    raise torch.cuda.OutOfMemoryError(
        f"CUDA OOM during {phase} at batch {batch_idx + 1}. "
        f"Reduce batch_size or patch_size and {'resume training' if is_training else 'retry'}."
    )


def create_progress_bar(
    dataloader,
    epoch: int,
    phase: str = "Train",
    leave: bool = False,
    position: int = 1
) -> tqdm:
    """
    Create a standardized progress bar for training/validation.
    
    Args:
        dataloader: DataLoader to iterate over
        epoch: Current epoch number
        phase: Phase name ('Train', 'Val', etc.)
        leave: Whether to leave the progress bar after completion
        position: Position for nested progress bars
        
    Returns:
        Configured tqdm progress bar
    """
    return tqdm(
        dataloader,
        desc=f"Epoch {epoch} [{phase}]",
        leave=leave,
        position=position,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
    )


def apply_gradient_clipping(
    model: torch.nn.Module,
    max_norm: float,
    scaler: torch.amp.GradScaler | None = None
) -> None:
    """
    Apply gradient clipping to model parameters.
    
    Args:
        model: PyTorch model
        max_norm: Maximum gradient norm
        scaler: GradScaler for mixed precision (will unscale first if provided)
    """
    if scaler is not None:
        # Unscale gradients before clipping when using AMP
        scaler.unscale_(model.parameters().__iter__().__next__().grad.device)
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)


def apply_gradient_clipping_optimizer(
    optimizer: torch.optim.Optimizer,
    parameters,
    max_norm: float,
    scaler: torch.amp.GradScaler | None = None
) -> None:
    """
    Apply gradient clipping with optimizer-aware unscaling.
    
    Args:
        optimizer: PyTorch optimizer
        parameters: Model parameters to clip
        max_norm: Maximum gradient norm
        scaler: GradScaler for mixed precision (will unscale first if provided)
    """
    if scaler is not None:
        # Unscale gradients before clipping when using AMP
        scaler.unscale_(optimizer)
    
    torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)
