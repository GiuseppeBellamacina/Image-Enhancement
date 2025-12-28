"""
Training and validation loop utilities
"""

import torch
from tqdm.auto import tqdm
from typing import Tuple


def train_epoch(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    gradient_clip: float = 1.0
) -> dict:
    """
    Train the model for one epoch.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on ('cuda' or 'cpu')
        epoch: Current epoch number (for progress display)
        gradient_clip: Maximum gradient norm for clipping
    
    Returns:
        Dictionary with average metrics: {'loss', 'l1', 'ssim'}
    """
    model.train()
    
    running_loss = 0.0
    running_l1 = 0.0
    running_ssim = 0.0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]', leave=False)
    
    for degraded, clean in pbar:
        degraded = degraded.to(device)
        clean = clean.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(degraded)
        
        # Compute loss
        loss, metrics = criterion(output, clean)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
        
        optimizer.step()
        
        # Update metrics
        running_loss += metrics['total']
        running_l1 += metrics['l1']
        running_ssim += metrics['ssim']
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{metrics['total']:.4f}",
            'l1': f"{metrics['l1']:.4f}",
            'ssim': f"{metrics['ssim']:.3f}"
        })
    
    # Average metrics
    n_batches = len(train_loader)
    avg_metrics = {
        'loss': running_loss / n_batches,
        'l1': running_l1 / n_batches,
        'ssim': running_ssim / n_batches
    }
    
    return avg_metrics


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: str,
    epoch: int
) -> dict:
    """
    Validate the model.
    
    Args:
        model: PyTorch model to validate
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to run validation on ('cuda' or 'cpu')
        epoch: Current epoch number (for progress display)
    
    Returns:
        Dictionary with average metrics: {'loss', 'l1', 'ssim'}
    """
    model.eval()
    
    running_loss = 0.0
    running_l1 = 0.0
    running_ssim = 0.0
    
    pbar = tqdm(val_loader, desc=f'Epoch {epoch+1} [Val]', leave=False)
    
    for degraded, clean in pbar:
        degraded = degraded.to(device)
        clean = clean.to(device)
        
        # Forward pass
        output = model(degraded)
        
        # Compute loss
        loss, metrics = criterion(output, clean)
        
        # Update metrics
        running_loss += metrics['total']
        running_l1 += metrics['l1']
        running_ssim += metrics['ssim']
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{metrics['total']:.4f}",
            'ssim': f"{metrics['ssim']:.3f}"
        })
    
    # Average metrics
    n_batches = len(val_loader)
    avg_metrics = {
        'loss': running_loss / n_batches,
        'l1': running_l1 / n_batches,
        'ssim': running_ssim / n_batches
    }
    
    return avg_metrics
