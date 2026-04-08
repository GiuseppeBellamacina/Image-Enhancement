# -*- coding: utf-8 -*-
"""
Training and validation loop utilities
"""

import torch
import math
from torch.amp.autocast_mode import autocast

from .training_utils import (
    handle_oom_error,
    create_progress_bar,
    apply_gradient_clipping_optimizer,
)


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
        use_amp: Whether to use automatic mixed precision

    Returns:
        Dictionary with average metrics: {'loss', 'l1', 'ssim'}
    """
    model.train()

    running_loss = 0.0
    running_l1 = 0.0
    running_ssim = 0.0
    running_perceptual = 0.0
    valid_batches = 0
    skipped_non_finite = 0

    pbar = create_progress_bar(
        train_loader, epoch, phase="Train", leave=False, position=1
    )

    def _metrics_are_finite(loss: torch.Tensor, metrics: dict) -> bool:
        if not torch.isfinite(loss):
            return False

        for value in metrics.values():
            if isinstance(value, torch.Tensor):
                value = value.item()
            if not math.isfinite(float(value)):
                return False

        return True

    for batch_idx, (degraded, clean) in enumerate(pbar):
        output = None
        loss = None

        try:
            degraded = degraded.to(device)
            clean = clean.to(device)

            optimizer.zero_grad()

            # Forward pass with mixed precision if enabled
            if use_amp and scaler is not None:
                with autocast(device_type=device):
                    output = model(degraded)
                    loss, metrics = criterion(output, clean)

                # If AMP produces non-finite values, retry once in float32 before skipping.
                if not _metrics_are_finite(loss, metrics):
                    with autocast(device_type=device, enabled=False):
                        output = model(degraded.float())
                        loss, metrics = criterion(output, clean.float())

                if not _metrics_are_finite(loss, metrics):
                    skipped_non_finite += 1
                    optimizer.zero_grad(set_to_none=True)
                    pbar.set_postfix({"loss": "nan", "skip": skipped_non_finite})
                    continue

                # Backward pass with gradient scaling
                scaler.scale(loss).backward()

                # Gradient clipping (unscale first for proper clipping)
                apply_gradient_clipping_optimizer(
                    optimizer, model.parameters(), max_norm=gradient_clip, scaler=scaler
                )

                # Optimizer step with scaler
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard training without mixed precision
                output = model(degraded)
                loss, metrics = criterion(output, clean)

                # Skip unstable batches to avoid corrupting optimizer/model state.
                if not _metrics_are_finite(loss, metrics):
                    with autocast(device_type=device, enabled=False):
                        output = model(degraded.float())
                        loss, metrics = criterion(output, clean.float())

                if not _metrics_are_finite(loss, metrics):
                    skipped_non_finite += 1
                    optimizer.zero_grad(set_to_none=True)
                    pbar.set_postfix({"loss": "nan", "skip": skipped_non_finite})
                    continue

                # Backward pass
                loss.backward()

                # Gradient clipping
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
            valid_batches += 1

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
    n_batches = valid_batches
    if n_batches == 0:
        raise RuntimeError(
            "All training batches in this epoch were non-finite (NaN/Inf). "
            "Lower learning rate or restart from a clean checkpoint."
        )
    if skipped_non_finite > 0:
        print(
            f"⚠️  Epoch {epoch}: skipped {skipped_non_finite} non-finite training batches"
        )
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
) -> dict:
    """
    Validate the model.

    Args:
        model: PyTorch model to validate
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to run validation on ('cuda' or 'cpu')
        epoch: Current epoch number (for progress display)
        use_amp: Whether to use automatic mixed precision

    Returns:
        Dictionary with average metrics: {'loss', 'l1', 'ssim'}
    """
    model.eval()

    running_loss = 0.0
    running_l1 = 0.0
    running_ssim = 0.0
    running_perceptual = 0.0
    valid_batches = 0
    skipped_non_finite = 0

    def _metrics_are_finite(loss: torch.Tensor, metrics: dict) -> bool:
        if not torch.isfinite(loss):
            return False

        for value in metrics.values():
            if isinstance(value, torch.Tensor):
                value = value.item()
            if not math.isfinite(float(value)):
                return False

        return True

    pbar = create_progress_bar(val_loader, epoch, phase="Val", leave=False, position=1)

    for batch_idx, (degraded, clean) in enumerate(pbar):
        output = None
        loss = None

        try:
            degraded = degraded.to(device)
            clean = clean.to(device)

            # Forward pass with mixed precision if enabled
            if use_amp:
                with autocast(device_type=device):
                    output = model(degraded)
                    loss, metrics = criterion(output, clean)
            else:
                output = model(degraded)
                loss, metrics = criterion(output, clean)

            if not _metrics_are_finite(loss, metrics):
                with autocast(device_type=device, enabled=False):
                    output = model(degraded.float())
                    loss, metrics = criterion(output, clean.float())

            if not _metrics_are_finite(loss, metrics):
                skipped_non_finite += 1
                pbar.set_postfix({"loss": "nan", "skip": skipped_non_finite})
                continue

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
            valid_batches += 1

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
    n_batches = valid_batches
    if n_batches == 0:
        raise RuntimeError(
            "All validation batches in this epoch were non-finite (NaN/Inf)."
        )
    if skipped_non_finite > 0:
        print(
            f"⚠️  Epoch {epoch}: skipped {skipped_non_finite} non-finite validation batches"
        )
    avg_metrics = {
        "loss": running_loss / n_batches,
        "l1": running_l1 / n_batches,
        "ssim": running_ssim / n_batches,
    }
    if running_perceptual > 0:
        avg_metrics["perceptual"] = running_perceptual / n_batches

    return avg_metrics
