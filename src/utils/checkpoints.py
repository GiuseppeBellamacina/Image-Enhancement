"""
Checkpoint utilities for saving and loading model states
"""

import torch
import shutil
from pathlib import Path
from typing import Optional, Union


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    epoch: int,
    metrics: dict,
    filepath: Path,
    metadata: Optional[dict] = None,
) -> None:
    """
    Save a training checkpoint.

    Args:
        model: PyTorch model to save
        optimizer: Optimizer state to save
        scheduler: Learning rate scheduler state (optional)
        epoch: Current epoch number (the epoch that was just completed)
        metrics: Dictionary of metrics to save
        filepath: Path where to save the checkpoint
        metadata: Optional dictionary with additional metadata (e.g., total_epochs_trained)
    """
    checkpoint = {
        "epoch": epoch,  # The epoch that was just completed when this model was saved
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "metrics": metrics,
        "metadata": metadata or {},
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
    device: str = "cpu",
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
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load optimizer state if provided
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scheduler state if provided
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return {
        "epoch": checkpoint["epoch"],
        "metrics": checkpoint.get("metrics", {}),
        "metadata": checkpoint.get("metadata", {}),
    }


def ensure_best_model_exists(
    checkpoints_dir: Path,
    checkpoint_name: str = "best_model.pth",
) -> bool:
    """
    Ensure that best_model.pth exists in checkpoints directory.
    If it doesn't exist, create it from the most recent checkpoint.

    This is useful when training was interrupted (e.g., OOM crash) and only
    emergency/periodic checkpoints exist but best_model.pth is missing.

    Args:
        checkpoints_dir: Directory containing checkpoints
        checkpoint_name: Name of the best model checkpoint (default: 'best_model.pth')

    Returns:
        True if best_model.pth exists (or was successfully created), False otherwise
    """
    best_model_path = checkpoints_dir / checkpoint_name

    # If best_model.pth already exists, we're good
    if best_model_path.exists():
        return True

    print(f"\n⚠️  {checkpoint_name} not found in {checkpoints_dir}")

    # Look for alternative checkpoints to use
    checkpoint_candidates = []

    # 1. Look for emergency checkpoints (from OOM crashes)
    emergency_checkpoints = sorted(
        checkpoints_dir.glob("emergency_checkpoint_oom_epoch_*.pth"), reverse=True
    )
    checkpoint_candidates.extend(emergency_checkpoints)

    # 2. Look for periodic checkpoints (checkpoint_epoch_XXX.pth)
    periodic_checkpoints = sorted(
        checkpoints_dir.glob("checkpoint_epoch_*.pth"), reverse=True
    )
    checkpoint_candidates.extend(periodic_checkpoints)

    if not checkpoint_candidates:
        print(f"   ❌ No alternative checkpoints found in {checkpoints_dir}")
        return False

    # Use the most recent checkpoint
    source_checkpoint = checkpoint_candidates[0]
    print(f"   📋 Found alternative checkpoint: {source_checkpoint.name}")
    print(f"   🔄 Copying to {checkpoint_name}...")

    try:
        shutil.copy2(source_checkpoint, best_model_path)
        print(f"   ✅ Successfully created {checkpoint_name}")
        print(
            "   ⚠️  Note: This may not be the actual best model, just the most recent checkpoint"
        )
        return True
    except Exception as e:
        print(f"   ❌ Failed to copy checkpoint: {e}")
        return False


def load_pretrained_model(
    model: torch.nn.Module,
    experiment_path: Union[str, Path],
    model_name: Optional[str] = None,
    degradation: Optional[str] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    device: str = "cpu",
    checkpoint_name: str = "best_model.pth",
) -> dict:
    """
    Load a pretrained model from a previous experiment.

    This function supports two modes:
    1. Relative path mode: Pass model_name and degradation to automatically find
       the most recent experiment for that combination.
    2. Absolute path mode: Pass the full path to a specific experiment directory.

    Args:
        model: Model to load state into
        experiment_path: Either:
            - Relative path from experiments/[model_name]/[degradation]/ (e.g., "20251229_224726")
            - Full path to experiment directory
            - "latest" to load the most recent experiment
        model_name: Model name (e.g., 'unet', 'dncnn') - required if using relative path
        degradation: Degradation type (e.g., 'gaussian', 'dithering') - required if using relative path
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        device: Device to map tensors to
        checkpoint_name: Name of checkpoint file (default: 'best_model.pth')

    Returns:
        Dictionary containing epoch and metrics from checkpoint

    Examples:
        # Load most recent experiment for unet/gaussian
        >>> load_pretrained_model(model, "latest", model_name="unet", degradation="gaussian")

        # Load specific experiment by timestamp
        >>> load_pretrained_model(model, "20251229_224726", model_name="unet", degradation="gaussian")

        # Load from full path
        >>> load_pretrained_model(model, "experiments/unet/gaussian/20251229_224726")

    Raises:
        FileNotFoundError: If the experiment or checkpoint file is not found
        ValueError: If invalid arguments are provided
    """
    from .paths import get_model_experiments_dir, find_project_root

    experiment_path = str(experiment_path)

    # Determine the full path to the experiment directory
    if model_name and degradation:
        # Relative path mode
        base_dir = get_model_experiments_dir(model_name, degradation)

        if experiment_path == "latest":
            # Find the most recent experiment
            if not base_dir.exists():
                raise FileNotFoundError(
                    f"No experiments found for {model_name}/{degradation} at {base_dir}"
                )

            experiment_dirs = sorted(
                [d for d in base_dir.iterdir() if d.is_dir()],
                key=lambda x: x.name,
                reverse=True,
            )

            if not experiment_dirs:
                raise FileNotFoundError(
                    f"No experiments found for {model_name}/{degradation} at {base_dir}"
                )

            exp_dir = experiment_dirs[0]  # Use most recent experiment
            print(f"📂 Loading most recent experiment: {exp_dir.name}")
        else:
            # Use provided experiment name/timestamp
            exp_dir = base_dir / experiment_path

    else:
        # Absolute path mode
        exp_dir = Path(experiment_path)

        # If path doesn't start with experiments/, assume it's relative from experiments root
        if not exp_dir.is_absolute() and not str(exp_dir).startswith("experiments"):
            # Path like "unet/gaussian/20251229_224726"
            root = find_project_root()
            exp_dir = root / "experiments" / exp_dir

    # Verify experiment directory exists
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    # Load the checkpoint
    checkpoint_path = exp_dir / "checkpoints" / checkpoint_name

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint file not found: {checkpoint_path}\n"
            f"Available files: {list((exp_dir / 'checkpoints').glob('*.pth')) if (exp_dir / 'checkpoints').exists() else []}"
        )

    print(f"📥 Loading checkpoint from: {checkpoint_path}")
    result = load_checkpoint(checkpoint_path, model, optimizer, scheduler, device)

    print(
        f"✅ Loaded model from epoch {result['epoch']} "
        f"(loss: {result['metrics'].get('val', {}).get('loss', 'N/A')})"
    )

    return result


def resume_training(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    experiment_path: Union[str, Path] = "latest",
    model_name: Optional[str] = None,
    degradation: Optional[str] = None,
    device: str = "cpu",
    checkpoint_name: str = "best_model.pth",
) -> tuple[dict, int, dict, Path]:
    """
    Resume training from a previous checkpoint.

    This is a convenience wrapper around load_pretrained_model specifically for
    resuming training. It automatically loads model, optimizer, and scheduler states
    and returns the starting epoch for continuing training.

    Args:
        model: Model to load state into (required)
        optimizer: Optimizer to load state into (required)
        scheduler: Scheduler to load state into (optional but recommended)
        experiment_path: Either:
            - "latest" to load the most recent experiment (default)
            - Relative path from experiments/[model_name]/[degradation]/ (e.g., "20251229_224726")
            - Full path to experiment directory
        model_name: Model name (e.g., 'unet', 'dncnn') - required if using relative path
        degradation: Degradation type (e.g., 'gaussian', 'dithering') - required if using relative path
        device: Device to map tensors to
        checkpoint_name: Name of checkpoint file (default: 'best_model.pth')

    Returns:
        Tuple of (checkpoint_info, start_epoch, history, exp_dir) where:
        - checkpoint_info: Dictionary containing epoch and metrics from checkpoint
        - start_epoch: Epoch number to start training from (checkpoint epoch + 1)
        - history: Previous training history (empty dict if not found)
        - exp_dir: Path to the experiment directory

    Examples:
        # Resume from most recent unet/gaussian experiment
        >>> info, start_epoch, history, exp_dir = resume_training(
        ...     model=model,
        ...     optimizer=optimizer,
        ...     scheduler=scheduler,
        ...     experiment_path="latest",
        ...     model_name="unet",
        ...     degradation="gaussian"
        ... )

    Raises:
        FileNotFoundError: If the experiment or checkpoint file is not found
        ValueError: If invalid arguments are provided
    """
    from .paths import get_model_experiments_dir, find_project_root
    from .experiment import load_training_history

    print("\n" + "=" * 80)
    print("🔄 Resuming Training from Checkpoint")
    print("=" * 80 + "\n")

    # Determine experiment directory path first
    if model_name and degradation:
        base_dir = get_model_experiments_dir(model_name, degradation)
        if experiment_path == "latest":
            experiment_dirs = sorted(
                [d for d in base_dir.iterdir() if d.is_dir()],
                key=lambda x: x.name,
                reverse=True,
            )
            if not experiment_dirs:
                raise FileNotFoundError(f"No experiments found in {base_dir}")
            exp_dir = experiment_dirs[0]
        else:
            exp_dir = base_dir / experiment_path
    else:
        exp_dir = Path(experiment_path)
        if not exp_dir.is_absolute() and not str(exp_dir).startswith("experiments"):
            root = find_project_root()
            exp_dir = root / "experiments" / exp_dir

    # Ensure checkpoints directory exists
    checkpoints_dir = exp_dir / "checkpoints"
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")

    # Ensure best_model.pth exists (create from latest checkpoint if missing)
    ensure_best_model_exists(checkpoints_dir, checkpoint_name)

    # Load checkpoint with optimizer and scheduler
    checkpoint_info = load_pretrained_model(
        model=model,
        experiment_path=experiment_path,
        model_name=model_name,
        degradation=degradation,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_name=checkpoint_name,
    )

    # Try to load previous training history
    history = {}
    try:
        history = load_training_history(exp_dir)

        # Truncate history to match the checkpoint
        # Use metadata.total_epochs_completed if available, fallback to checkpoint epoch
        checkpoint_epoch = checkpoint_info["epoch"]
        metadata = checkpoint_info.get("metadata", {})
        total_epochs_in_checkpoint = metadata.get(
            "total_epochs_completed", checkpoint_epoch
        )

        if history and "train_loss" in history and len(history["train_loss"]) > 0:
            # Truncate to the number of epochs present when checkpoint was saved
            # This correctly handles val_every > 1 scenarios
            if len(history["train_loss"]) > total_epochs_in_checkpoint:
                for key in history:
                    history[key] = history[key][:total_epochs_in_checkpoint]
                print(
                    f"📊 Loaded history and truncated to {total_epochs_in_checkpoint} validation points"
                )
            else:
                print(
                    f"📊 Loaded history ({len(history['train_loss'])} validation points)"
                )
    except FileNotFoundError:
        history = {}

    # Calculate starting epoch for resuming
    checkpoint_epoch = checkpoint_info["epoch"]
    start_epoch = checkpoint_epoch + 1
    best_val_loss = checkpoint_info["metrics"].get("val", {}).get("loss", float("inf"))

    print(
        f"📦 Resuming from epoch {start_epoch} (checkpoint at epoch {checkpoint_epoch})"
    )
    if checkpoint_name == "best_model.pth":
        print(f"   Best model val_loss: {best_val_loss:.4f}")
    print("=" * 80 + "\n")

    return checkpoint_info, start_epoch, history, exp_dir
