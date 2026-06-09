# -*- coding: utf-8 -*-
"""
Training and validation loop utilities
"""

import torch
from torch.amp.autocast_mode import autocast

from .training_utils import (
    handle_oom_error,
    create_progress_bar,
    apply_gradient_clipping_optimizer,
)


def _forward_with_sigma(
    model: torch.nn.Module, x: torch.Tensor, noise_sigma: float | None
):
    if noise_sigma is None:
        return model(x)
    return model(x, sigma=noise_sigma)


def train_epoch(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    gradient_clip: float = 1.0,
    scaler: torch.amp.grad_scaler.GradScaler | None = None,
    use_amp: bool = False,
    noise_sigma: float = 100.0,
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
        scaler: GradScaler for mixed precision training (optional)
        use_amp: Whether to use automatic mixed precision (loss is still computed in float32 when AMP is on).

    Returns:
        Dictionary with average metrics: {'loss', 'l1', 'ssim'}
    """
    model.train()

    running_loss = 0.0
    running_l1 = 0.0
    running_ssim = 0.0
    running_perceptual = 0.0

    pbar = create_progress_bar(
        train_loader, epoch, phase="Train", leave=False, position=1
    )

    for batch_idx, (degraded, clean) in enumerate(pbar):
        output = None
        loss = None

        try:
            degraded = degraded.to(device)
            clean = clean.to(device)

            optimizer.zero_grad()

            if use_amp and scaler is not None:
                # Forward in mixed precision; loss (L1 + SSIM) in float32 — SSIM in fp16 often yields NaN.
                with autocast(device_type=device):
                    output = _forward_with_sigma(model, degraded, noise_sigma)
                loss, metrics = criterion(output, clean)

                scaler.scale(loss).backward()

                # Gradient clipping (unscale first for proper clipping)
                apply_gradient_clipping_optimizer(
                    optimizer, model.parameters(), max_norm=gradient_clip, scaler=scaler
                )

                # Optimizer step with scaler
                scaler.step(optimizer)
                scaler.update()
            else:
                output = _forward_with_sigma(model, degraded, noise_sigma)
                loss, metrics = criterion(output, clean)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=gradient_clip
                )
                optimizer.step()

            # Update metrics
            running_loss += metrics["total"]
            running_l1 += metrics["l1"]
            running_ssim += metrics["ssim"]
            if "perceptual" in metrics:
                running_perceptual += metrics["perceptual"]

            # Update progress bar (include perceptual if available)
            postfix = {
                "loss": f"{metrics['total']:.4f}",
                "l1": f"{metrics['l1']:.4f}",
                "ssim": f"{metrics['ssim']:.3f}",
            }
            if "perceptual" in metrics and metrics["perceptual"] > 0:
                postfix["perceptual"] = f"{metrics['perceptual']:.4f}"
            pbar.set_postfix(postfix)

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            # Check if it's an OOM error
            if "out of memory" in str(e).lower():
                pbar.close()
                handle_oom_error(
                    batch_idx,
                    len(train_loader),
                    device,
                    degraded,
                    clean,
                    output,
                    loss,
                    is_training=True,
                )
            else:
                # Re-raise non-OOM RuntimeErrors
                raise

    # Average metrics
    n_batches = len(train_loader)
    avg_metrics = {
        "loss": running_loss / n_batches,
        "l1": running_l1 / n_batches,
        "ssim": running_ssim / n_batches,
    }
    if running_perceptual > 0:
        avg_metrics["perceptual"] = running_perceptual / n_batches

    return avg_metrics


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: str,
    epoch: int,
    use_amp: bool = False,
    noise_sigma: float = 100.0,
) -> dict:
    """
    Validate the model.

    Args:
        model: PyTorch model to validate
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to run validation on ('cuda' or 'cpu')
        epoch: Current epoch number (for progress display)
        use_amp: Ignored for validation. The forward pass always runs in full precision so that
            BatchNorm in eval() and loss metrics stay numerically stable while training may still use AMP.

    Returns:
        Dictionary with average metrics: {'loss', 'l1', 'ssim'}
    """
    model.eval()

    running_loss = 0.0
    running_l1 = 0.0
    running_ssim = 0.0
    running_perceptual = 0.0

    pbar = create_progress_bar(val_loader, epoch, phase="Val", leave=False, position=1)

    for batch_idx, (degraded, clean) in enumerate(pbar):
        output = None
        loss = None

        try:
            degraded = degraded.to(device)
            clean = clean.to(device)

            # Always full-precision forward on val: with AMP, train often stays stable while
            # eval()+BatchNorm+fp16 produces NaN losses; val is cheap vs train.
            output = _forward_with_sigma(model, degraded, noise_sigma)
            loss, metrics = criterion(output, clean)

            # Update metrics
            running_loss += metrics["total"]
            running_l1 += metrics["l1"]
            running_ssim += metrics["ssim"]
            if "perceptual" in metrics:
                running_perceptual += metrics["perceptual"]

            # Update progress bar (include perceptual if available)
            postfix = {
                "loss": f"{metrics['total']:.4f}",
                "ssim": f"{metrics['ssim']:.3f}",
            }
            if "perceptual" in metrics and metrics["perceptual"] > 0:
                postfix["perceptual"] = f"{metrics['perceptual']:.4f}"
            pbar.set_postfix(postfix)

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            # Check if it's an OOM error
            if "out of memory" in str(e).lower():
                pbar.close()
                handle_oom_error(
                    batch_idx,
                    len(val_loader),
                    device,
                    degraded,
                    clean,
                    output,
                    loss,
                    is_training=False,
                )
            else:
                # Re-raise non-OOM RuntimeErrors
                raise

    # Average metrics
    n_batches = len(val_loader)
    avg_metrics = {
        "loss": running_loss / n_batches,
        "l1": running_l1 / n_batches,
        "ssim": running_ssim / n_batches,
    }
    if running_perceptual > 0:
        avg_metrics["perceptual"] = running_perceptual / n_batches

    return avg_metrics
