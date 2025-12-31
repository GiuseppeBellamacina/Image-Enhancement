# -*- coding: utf-8 -*-
"""
Training orchestrator for running complete training experiments
"""

import time
import torch
from pathlib import Path
from typing import Dict, Tuple
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LRScheduler

from .training import train_epoch, validate
from ..utils.checkpoints import save_checkpoint
from ..utils.experiment import save_training_history, save_experiment_stats, load_experiment_stats


def run_training(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LRScheduler | None,
    device: str,
    num_epochs: int,
    checkpoints_dir: Path,
    writer: SummaryWriter,
    warmup_epochs: int = 0,
    learning_rate: float = 1e-4,
    patience: int = 5,
    save_every: int = 5,
    val_every: int = 1,
    gradient_clip: float = 1.0,
    start_epoch: int = 0,
    initial_best_loss: float = float("inf"),
    initial_best_epoch: int = 0,
    initial_history: Dict | None = None,
    use_amp: bool = False,
) -> Tuple[Dict, Dict]:
    """
    Run complete training loop with validation, checkpointing, and early stopping.

    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        device: Device to train on ('cuda' or 'cpu')
        num_epochs: Total number of epochs to train
        checkpoints_dir: Directory to save checkpoints
        writer: TensorBoard SummaryWriter
        warmup_epochs: Number of warmup epochs for learning rate
        learning_rate: Initial learning rate (used for warmup)
        patience: Patience for early stopping
        save_every: Save checkpoint every N epochs
        val_every: Validate every N epochs
        gradient_clip: Maximum gradient norm for clipping
        start_epoch: Starting epoch (0 for new training, >0 for resumed training)
        initial_best_loss: Initial best validation loss (inf for new training)
        initial_history: Previous training history to continue from (optional)
        use_amp: Whether to use automatic mixed precision (reduces VRAM by ~40-50%)

    Returns:
        Tuple of (history, best_info) where:
        - history: Dict with training metrics history
        - best_info: Dict with best epoch and validation loss
    """
    # Training history - validate and merge with previous if exists
    if initial_history is not None and isinstance(initial_history, dict):
        # Validate that history has the expected structure
        required_keys = [
            "train_loss",
            "train_l1",
            "train_ssim",
            "val_loss",
            "val_l1",
            "val_ssim",
            "lr",
        ]
        if all(key in initial_history for key in required_keys):
            history = initial_history
            # Add timing/memory fields if missing (for backwards compatibility)
            if "epoch_time" not in history:
                history["epoch_time"] = []
            if "inference_time" not in history:
                history["inference_time"] = []
            if "memory_allocated_mb" not in history:
                history["memory_allocated_mb"] = []
            if "memory_reserved_mb" not in history:
                history["memory_reserved_mb"] = []
            print(
                f"📊 Continuing from previous history ({len(history['train_loss'])} epochs recorded)"
            )
        else:
            print("⚠️  Previous history has invalid structure, starting fresh")
            history = {
                "train_loss": [],
                "train_l1": [],
                "train_ssim": [],
                "val_loss": [],
                "val_l1": [],
                "val_ssim": [],
                "lr": [],
                "epoch_time": [],
                "inference_time": [],
                "memory_allocated_mb": [],
                "memory_reserved_mb": [],
            }
    else:
        # Start fresh history
        history = {
            "train_loss": [],
            "train_l1": [],
            "train_ssim": [],
            "val_loss": [],
            "val_l1": [],
            "val_ssim": [],
            "lr": [],
            "epoch_time": [],
            "inference_time": [],
            "memory_allocated_mb": [],
            "memory_reserved_mb": [],
        }

    # Initialize GradScaler for mixed precision training
    scaler = None
    if use_amp and device == "cuda":
        scaler = torch.amp.grad_scaler.GradScaler()
        print("⚡ Mixed Precision Training: ENABLED (fp16)")
    elif use_amp and device == "cpu":
        use_amp = False

    # Best model tracking
    best_val_loss = initial_best_loss
    best_epoch = initial_best_epoch if initial_best_epoch > 0 else 0
    patience_counter = 0

    print("\n" + "=" * 80)
    if start_epoch > 1:
        print(f"🔄 Resuming Training from Epoch {start_epoch}")
        print(f"   Will train epochs {start_epoch} to {num_epochs}")
        print(f"   Previous best loss: {initial_best_loss:.4f}")
    else:
        print("🚀 Starting Training")
        print(f"   Epochs: 1 to {num_epochs}")
    print("=" * 80 + "\n")

    # Track if training completed successfully
    training_interrupted = False

    try:
        for epoch in range(start_epoch if start_epoch > 0 else 1, num_epochs + 1):
            epoch_start_time = time.time()

            # Learning rate warmup (only if not already completed in previous training)
            if epoch <= warmup_epochs and start_epoch <= warmup_epochs:
                lr = learning_rate * epoch / warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

            # Train
            train_metrics = train_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                epoch,
                gradient_clip,
                scaler=scaler,
                use_amp=use_amp,
            )

            # Validate
            inference_time = 0.0
            if epoch % val_every == 0:
                val_start_time = time.time()
                val_metrics = validate(
                    model, val_loader, criterion, device, epoch, use_amp=use_amp
                )
                inference_time = time.time() - val_start_time
            else:
                val_metrics = None

            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time

            # Track memory usage (if CUDA available)
            memory_allocated = 0.0
            memory_reserved = 0.0
            if device == "cuda" and torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / (1024**2)  # MB
                memory_reserved = torch.cuda.memory_reserved() / (1024**2)  # MB

            # Update learning rate
            if epoch > warmup_epochs and scheduler:
                scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]

            # Log to TensorBoard (always log, even if not saving to history)
            writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
            writer.add_scalar("L1/train", train_metrics["l1"], epoch)
            writer.add_scalar("SSIM/train", train_metrics["ssim"], epoch)
            writer.add_scalar("Learning_Rate", current_lr, epoch)
            writer.add_scalar("Time/epoch_seconds", epoch_time, epoch)
            writer.add_scalar("Memory/allocated_mb", memory_allocated, epoch)
            writer.add_scalar("Memory/reserved_mb", memory_reserved, epoch)

            if val_metrics:
                writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
                writer.add_scalar("L1/val", val_metrics["l1"], epoch)
                writer.add_scalar("SSIM/val", val_metrics["ssim"], epoch)
                writer.add_scalar("Time/inference_seconds", inference_time, epoch)

            # Print epoch summary
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(
                f"  Train - Loss: {train_metrics['loss']:.4f} | L1: {train_metrics['l1']:.4f} | SSIM: {train_metrics['ssim']:.3f}"
            )
            if val_metrics:
                print(
                    f"  Val   - Loss: {val_metrics['loss']:.4f} | L1: {val_metrics['l1']:.4f} | SSIM: {val_metrics['ssim']:.3f}"
                )
            print(f"  LR: {current_lr:.6f} | Time: {epoch_time:.1f}s", end="")
            if val_metrics:
                print(f" (inference: {inference_time:.1f}s)", end="")
            if device == "cuda":
                print(f" | VRAM: {memory_allocated:.0f}MB")
            else:
                print()

            # Update history only when we validate (to keep history synchronized with validation)
            if val_metrics:
                history["train_loss"].append(train_metrics["loss"])
                history["train_l1"].append(train_metrics["l1"])
                history["train_ssim"].append(train_metrics["ssim"])
                history["lr"].append(current_lr)
                history["val_loss"].append(val_metrics["loss"])
                history["val_l1"].append(val_metrics["l1"])
                history["val_ssim"].append(val_metrics["ssim"])
                history["epoch_time"].append(epoch_time)
                history["inference_time"].append(inference_time)
                history["memory_allocated_mb"].append(memory_allocated)
                history["memory_reserved_mb"].append(memory_reserved)

                # Save history after every validation (will be truncated at resume based on checkpoint)
                save_training_history(history, checkpoints_dir.parent)

            # Save checkpoint when validation improves
            if val_metrics and val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_epoch = epoch
                patience_counter = 0

                # Save best model checkpoint
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    {"train": train_metrics, "val": val_metrics},
                    checkpoints_dir / "best_model.pth",
                    metadata={
                        "total_epochs_completed": len(history["train_loss"]),
                        "best_epoch": epoch,
                    },
                )

                print(f"  ✅ Best model saved! (val_loss: {best_val_loss:.4f})")
            elif val_metrics:
                # Only increment patience if we actually validated and didn't improve
                patience_counter += 1

            # Periodic checkpoint
            if epoch % save_every == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    (
                        {"train": train_metrics, "val": val_metrics}
                        if val_metrics
                        else {"train": train_metrics}
                    ),
                    checkpoints_dir / f"checkpoint_epoch_{epoch:03d}.pth",
                    metadata={
                        "total_epochs_completed": len(history["train_loss"]),
                    },
                )
                print(f"  💾 Checkpoint saved (epoch {epoch})")

            # Early stopping
            if patience_counter >= patience:
                print(
                    f"\n⚠️  Early stopping triggered! No improvement for {patience} epochs."
                )
                break

            print("-" * 80)

    except KeyboardInterrupt:
        training_interrupted = True
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        # Check if it's an OOM error
        if "out of memory" in str(e).lower():
            print("\n" + "=" * 80)
            print("💥 CUDA OUT OF MEMORY ERROR")
            print("=" * 80)
            print(f"\nError: {e}\n")

            # Save emergency checkpoint
            print("💾 Saving emergency checkpoint...")
            try:
                # Get current epoch from history length
                current_epoch = len(history["train_loss"])

                # Create emergency metrics
                emergency_metrics = {
                    "train": {
                        "loss": (
                            history["train_loss"][-1]
                            if history["train_loss"]
                            else float("inf")
                        )
                    }
                }
                if history["val_loss"]:
                    emergency_metrics["val"] = {"loss": history["val_loss"][-1]}

                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    current_epoch,
                    emergency_metrics,
                    checkpoints_dir
                    / f"emergency_checkpoint_oom_epoch_{current_epoch}.pth",
                )
                print(
                    f"✅ Emergency checkpoint saved: emergency_checkpoint_oom_epoch_{current_epoch}.pth"
                )
            except Exception as save_error:
                print(f"⚠️  Failed to save emergency checkpoint: {save_error}")

            # Clear CUDA cache
            if device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            training_interrupted = True
            # Don't re-raise, allow graceful shutdown
        else:
            # Non-OOM RuntimeError
            print(f"\n❌ Runtime error during training: {e}")
            training_interrupted = True
            raise
    except Exception as e:
        print(f"\n❌ Unexpected error during training: {e}")
        training_interrupted = True
        raise
    finally:
        # Always close writer, even if interrupted
        try:
            writer.close()
        except Exception as e:
            print(f"⚠️  Failed to close writer: {e}")

        # Calculate and save cumulative statistics
        if history.get("epoch_time") and len(history["epoch_time"]) > 0:
            # Load previous stats if resuming
            prev_stats = load_experiment_stats(checkpoints_dir.parent)

            # Calculate current session stats
            total_time_current = sum(history["epoch_time"])
            avg_epoch_time = sum(history["epoch_time"]) / len(history["epoch_time"])
            
            # Calculate cumulative stats
            total_training_time = prev_stats.get("total_training_time_seconds", 0.0) + total_time_current
            total_epochs_trained = len(history["train_loss"])
            
            # Memory stats
            peak_memory = max(history["memory_allocated_mb"]) if history.get("memory_allocated_mb") else 0.0
            peak_memory = max(peak_memory, prev_stats.get("peak_memory_allocated_mb", 0.0))
            
            # Inference stats
            if history.get("inference_time") and len(history["inference_time"]) > 0:
                avg_inference_time = sum(history["inference_time"]) / len(history["inference_time"])
            else:
                avg_inference_time = prev_stats.get("avg_inference_time_seconds", 0.0)

            # Save cumulative stats
            save_experiment_stats(
                checkpoints_dir.parent,
                total_training_time,
                total_epochs_trained,
                peak_memory,
                avg_epoch_time,
                avg_inference_time,
            )

    if not training_interrupted:
        print("\n" + "=" * 80)
        print("✅ Training Completed!")
        print(f"   Best model: epoch {best_epoch} with val_loss {best_val_loss:.4f}")
        print(f"   Validation points saved: {len(history['train_loss'])}")
        print("=" * 80 + "\n")
    else:
        completed_epochs = len(history["train_loss"])
        print("\n" + "=" * 80)
        print("⚠️  Training Interrupted")
        if best_epoch > 0:
            print(
                f"   Best model: epoch {best_epoch} (val_loss: {best_val_loss:.4f})"
            )
        print(
            f"   History contains {completed_epochs} validation point{'s' if completed_epochs != 1 else ''}"
        )
        print("=" * 80 + "\n")

    # Return history and best model info
    best_info = {"best_epoch": best_epoch, "best_val_loss": best_val_loss}

    return history, best_info
