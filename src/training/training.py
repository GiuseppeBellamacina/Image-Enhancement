# -*- coding: utf-8 -*-
"""
Training and validation loop utilities
"""

import torch
from torch.amp.autocast_mode import autocast
from tqdm.auto import tqdm


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
        use_amp: Whether to use automatic mixed precision

    Returns:
        Dictionary with average metrics: {'loss', 'l1', 'ssim'}
    """
    model.train()

    running_loss = 0.0
    running_l1 = 0.0
    running_ssim = 0.0

    pbar = tqdm(
        train_loader,
        desc=f"Epoch {epoch} [Train]",
        leave=False,
        position=1,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
    )

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
                    if noise_sigma is not None:
                        output = model(degraded, noise_sigma)
                    else:
                        output = model(degraded)
                    loss, metrics = criterion(output, clean)

                # Backward pass with gradient scaling
                scaler.scale(loss).backward()

                # Gradient clipping (unscale first for proper clipping)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=gradient_clip
                )

                # Optimizer step with scaler
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard training without mixed precision
                output = model(degraded)
                loss, metrics = criterion(output, clean)

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

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{metrics['total']:.4f}",
                    "l1": f"{metrics['l1']:.4f}",
                    "ssim": f"{metrics['ssim']:.3f}",
                }
            )

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            # Check if it's an OOM error
            if "out of memory" in str(e).lower():
                pbar.close()
                print(
                    f"\n❌ CUDA Out of Memory at batch {batch_idx + 1}/{len(train_loader)}!"
                )
                print("   Attempting recovery...")

                # Clear CUDA cache
                if device == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    print("   ✅ CUDA cache cleared")

                # Free batch memory
                del degraded, clean
                if output is not None:
                    del output
                if loss is not None:
                    del loss

                print("\n⚠️  OOM ERROR DETECTED - TRAINING HALTED")
                print("   Suggestions to prevent OOM:")
                print("   1. Reduce batch_size (currently using batch in this epoch)")
                print("   2. Reduce patch_size in config")
                print("   3. Enable mixed precision if not already active")
                print("   4. Reduce model_features in config")
                print("\n   The training state has been saved.")
                print(
                    "   You can resume from the last checkpoint after adjusting config."
                )

                # Re-raise to be caught by run_training
                raise torch.cuda.OutOfMemoryError(
                    f"CUDA OOM during training at batch {batch_idx + 1}. "
                    "Reduce batch_size or patch_size and resume training."
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
        use_amp: Whether to use automatic mixed precision

    Returns:
        Dictionary with average metrics: {'loss', 'l1', 'ssim'}
    """
    model.eval()

    running_loss = 0.0
    running_l1 = 0.0
    running_ssim = 0.0

    pbar = tqdm(
        val_loader,
        desc=f"Epoch {epoch} [Val]",
        leave=False,
        position=1,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
    )

    for batch_idx, (degraded, clean) in enumerate(pbar):
        output = None
        loss = None

        try:
            degraded = degraded.to(device)
            clean = clean.to(device)

            # Forward pass with mixed precision if enabled
            if use_amp:
                with autocast(device_type=device):
                    if noise_sigma is not None:
                        output = model(degraded, noise_sigma)
                    else:
                        output = model(degraded)
                    loss, metrics = criterion(output, clean)
            else:
                if noise_sigma is not None:
                    output = model(degraded, noise_sigma)
                else:
                    output = model(degraded)
                loss, metrics = criterion(output, clean)

            # Update metrics
            running_loss += metrics["total"]
            running_l1 += metrics["l1"]
            running_ssim += metrics["ssim"]

            # Update progress bar
            pbar.set_postfix(
                {"loss": f"{metrics['total']:.4f}", "ssim": f"{metrics['ssim']:.3f}"}
            )

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            # Check if it's an OOM error
            if "out of memory" in str(e).lower():
                pbar.close()
                print(
                    f"\n❌ CUDA Out of Memory during validation at batch {batch_idx + 1}/{len(val_loader)}!"
                )
                print("   Attempting recovery...")

                # Clear CUDA cache
                if device == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    print("   ✅ CUDA cache cleared")

                # Free batch memory
                del degraded, clean
                if output is not None:
                    del output
                if loss is not None:
                    del loss

                # Re-raise to be caught by run_training
                raise torch.cuda.OutOfMemoryError(
                    f"CUDA OOM during validation at batch {batch_idx + 1}. Reduce batch_size."
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

    return avg_metrics
