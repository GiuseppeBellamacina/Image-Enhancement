"""
Training orchestrator for running complete training experiments
"""

import torch
from pathlib import Path
from typing import Dict, Tuple
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LRScheduler

from .training import train_epoch, validate
from ..utils.checkpoints import save_checkpoint
from ..utils.experiment import save_training_history


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
    initial_history: Dict | None = None,
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

    Returns:
        Tuple of (history, best_info) where:
        - history: Dict with training metrics history
        - best_info: Dict with best epoch and validation loss
    """
    # Training history - validate and merge with previous if exists
    if initial_history is not None and isinstance(initial_history, dict):
        # Validate that history has the expected structure
        required_keys = ["train_loss", "train_l1", "train_ssim", "val_loss", "val_l1", "val_ssim", "lr"]
        if all(key in initial_history for key in required_keys):
            history = initial_history
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
        }

    # Best model tracking
    best_val_loss = initial_best_loss
    best_epoch = start_epoch if start_epoch > 0 else 1
    patience_counter = 0

    print("\n" + "=" * 80)
    if start_epoch > 1:
        print(f"🔄 Resuming Training from Epoch {start_epoch}")
        print(f"   Will train epochs {start_epoch} to {num_epochs}")
        print(f"   Previous best loss: {initial_best_loss:.4f}")
    else:
        print("🚀 Starting Training")
        print(f"   Will train epochs 1 to {num_epochs}")
    print("=" * 80 + "\n")

    # Track if training completed successfully
    training_interrupted = False
    
    try:
        for epoch in range(start_epoch if start_epoch > 0 else 1, num_epochs + 1):

            # Learning rate warmup (only if not already completed in previous training)
            if epoch <= warmup_epochs and start_epoch <= warmup_epochs:
                lr = learning_rate * epoch / warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

            # Train
            train_metrics = train_epoch(
                model, train_loader, criterion, optimizer, device, epoch, gradient_clip
            )

            # Validate
            if epoch % val_every == 0:
                val_metrics = validate(model, val_loader, criterion, device, epoch)
            else:
                val_metrics = None

            # Update learning rate
            if epoch > warmup_epochs and scheduler:
                scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]

            # Save metrics
            history["train_loss"].append(train_metrics["loss"])
            history["train_l1"].append(train_metrics["l1"])
            history["train_ssim"].append(train_metrics["ssim"])
            history["lr"].append(current_lr)

            if val_metrics:
                history["val_loss"].append(val_metrics["loss"])
                history["val_l1"].append(val_metrics["l1"])
                history["val_ssim"].append(val_metrics["ssim"])

            # Log to TensorBoard
            writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
            writer.add_scalar("L1/train", train_metrics["l1"], epoch)
            writer.add_scalar("SSIM/train", train_metrics["ssim"], epoch)
            writer.add_scalar("Learning_Rate", current_lr, epoch)

            if val_metrics:
                writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
                writer.add_scalar("L1/val", val_metrics["l1"], epoch)
                writer.add_scalar("SSIM/val", val_metrics["ssim"], epoch)

            # Print epoch summary
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(
                f"  Train - Loss: {train_metrics['loss']:.4f} | L1: {train_metrics['l1']:.4f} | SSIM: {train_metrics['ssim']:.3f}"
            )
            if val_metrics:
                print(
                    f"  Val   - Loss: {val_metrics['loss']:.4f} | L1: {val_metrics['l1']:.4f} | SSIM: {val_metrics['ssim']:.3f}"
                )
            print(f"  LR: {current_lr:.6f}")

            # Save checkpoint
            if val_metrics and val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_epoch = epoch
                patience_counter = 0

                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    {"train": train_metrics, "val": val_metrics},
                    checkpoints_dir / "best_model.pth",
                )
                print(f"  ✅ Best model saved! (val_loss: {best_val_loss:.4f})")
            else:
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
                )
                print(f"  💾 Checkpoint saved (epoch {epoch})")

            # Early stopping
            if patience_counter >= patience:
                print(
                    f"\n⚠️  Early stopping triggered! No improvement for {patience} epochs."
                )
                break

            # Save history after each epoch
            save_training_history(history, checkpoints_dir.parent)
            
            print("-" * 80)

    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user!")
        training_interrupted = True
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        training_interrupted = True
        raise
    finally:
        # Always save history and close writer, even if interrupted
        print("\n💾 Saving final state...")
        try:
            save_training_history(history, checkpoints_dir.parent)
            print("✅ Training history saved")
        except Exception as e:
            print(f"⚠️  Failed to save history: {e}")
        
        try:
            writer.close()
            print("✅ TensorBoard writer closed")
        except Exception as e:
            print(f"⚠️  Failed to close writer: {e}")

    if not training_interrupted:
        print("\n" + "=" * 80)
        print("✅ Training Completed!")
        print(f"   Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
        print("=" * 80 + "\n")
    else:
        print("\n" + "=" * 80)
        print("⚠️  Training Interrupted")
        print(f"   Best validation loss so far: {best_val_loss:.4f} at epoch {best_epoch}")
        print(f"   Completed epochs: {len(history['train_loss'])}")
        print("=" * 80 + "\n")

    # Return history and best model info
    best_info = {"best_epoch": best_epoch, "best_val_loss": best_val_loss}

    return history, best_info
