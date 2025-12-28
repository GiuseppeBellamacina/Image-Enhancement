"""
Checkpoint utilities for saving and loading model states
"""

import torch
from pathlib import Path
from typing import Optional


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    epoch: int,
    metrics: dict,
    filepath: Path
) -> None:
    """
    Save a training checkpoint.
    
    Args:
        model: PyTorch model to save
        optimizer: Optimizer state to save
        scheduler: Learning rate scheduler state (optional)
        epoch: Current epoch number
        metrics: Dictionary of metrics to save
        filepath: Path where to save the checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics
    }
    
    # Create parent directory if needed
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    device: str = 'cpu'
) -> dict:
    """
    Load a training checkpoint.
    
    Args:
        filepath: Path to the checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        device: Device to map tensors to
    
    Returns:
        Dictionary containing epoch and metrics from checkpoint
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state if provided
    if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return {
        'epoch': checkpoint['epoch'],
        'metrics': checkpoint.get('metrics', {})
    }
