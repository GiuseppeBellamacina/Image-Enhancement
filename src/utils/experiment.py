# -*- coding: utf-8 -*-
"""
Experiment setup utilities for managing output directories and configurations
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional, Set

from .paths import get_model_experiments_dir


def setup_experiment(
    model_name: str,
    degradation: str,
    config: dict,
    custom_name: Optional[str] = None,
    create_subdirs: bool = True,
) -> Tuple[Path, Dict[str, Path]]:
    """
    Setup experiment directory structure with timestamp.

    Args:
        model_name: Name of the model (e.g., 'unet', 'dncnn')
        degradation: Type of degradation (e.g., 'gaussian', 'dithering')
        config: Configuration dictionary to save
        custom_name: Optional custom name for the experiment (will be appended to timestamp)
        create_subdirs: Whether to create checkpoints/samples/logs subdirectories

    Returns:
        Tuple of (exp_dir, subdirs) where:
        - exp_dir: Path to the experiment directory
        - subdirs: Dict with 'checkpoints', 'samples', 'logs' paths
    """
    # Create experiment directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{timestamp}_{custom_name}" if custom_name else timestamp

    # Get base directory for this model/degradation combination
    base_dir = get_model_experiments_dir(model_name, degradation)
    exp_dir = base_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    subdirs = {}
    if create_subdirs:
        subdirs["checkpoints"] = exp_dir / "checkpoints"
        subdirs["samples"] = exp_dir / "samples"
        subdirs["logs"] = exp_dir / "logs"

        for subdir in subdirs.values():
            subdir.mkdir(exist_ok=True)

    # Save configuration
    with open(exp_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # Print summary
    print(f"\n📁 Experiment: {exp_name}")

    return exp_dir, subdirs


def load_existing_experiment(
    model_name: str,
    degradation: str,
    resume_experiment: str = "latest",
) -> Tuple[Path, Path, Path, Path]:
    """
    Load an existing experiment directory and its subdirectories.

    Args:
        model_name: Name of the model (e.g., 'unet', 'dncnn')
        degradation: Type of degradation (e.g., 'gaussian', 'dithering')
        resume_experiment: Which experiment to load ('latest' or timestamp)

    Returns:
        Tuple of (exp_dir, checkpoints_dir, samples_dir, logs_dir)

    Raises:
        FileNotFoundError: If no experiments found or specified experiment doesn't exist
    """
    print("\n" + "=" * 80)
    print("🔄 RESUME MODE: Loading existing experiment")
    print("=" * 80 + "\n")

    base_dir = get_model_experiments_dir(model_name, degradation)

    # Create base directory if it doesn't exist
    if not base_dir.exists():
        base_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created experiment directory: {base_dir}")

    if resume_experiment == "latest":
        # Find most recent experiment
        experiment_dirs = sorted(
            [d for d in base_dir.iterdir() if d.is_dir()],
            key=lambda x: x.name,
            reverse=True,
        )
        if not experiment_dirs:
            raise FileNotFoundError(f"No experiments found in {base_dir}")
        exp_dir = experiment_dirs[0]
    else:
        # Use specified experiment
        exp_dir = base_dir / resume_experiment
        if not exp_dir.exists():
            raise FileNotFoundError(f"Experiment not found: {exp_dir}")

    print(f"📁 Experiment: {exp_dir.name}")

    # Set subdirectory paths
    checkpoints_dir = exp_dir / "checkpoints"
    samples_dir = exp_dir / "samples"
    logs_dir = exp_dir / "logs"

    # Verify directories exist, create if missing
    for dir_path in [checkpoints_dir, samples_dir, logs_dir]:
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80 + "\n")

    return exp_dir, checkpoints_dir, samples_dir, logs_dir


def setup_or_resume_experiment(
    model_name: str,
    degradation: str,
    config: dict,
    resume_from_checkpoint: bool = False,
    resume_experiment: str = "latest",
    custom_name: Optional[str] = None,
    auto_fork_on_config_change: bool = True,
) -> Tuple[Path, Path, Path, Path]:
    """
    Setup a new experiment or resume an existing one based on configuration.

    This function intelligently handles config changes during resume:
    - If config matches the saved one, resumes normally
    - If config changed and auto_fork_on_config_change=True, creates a forked experiment
    - If config changed and auto_fork_on_config_change=False, raises an error

    Args:
        model_name: Name of the model (e.g., 'unet', 'dncnn')
        degradation: Type of degradation (e.g., 'gaussian', 'dithering/random')
        config: Configuration dictionary to save (only used for new experiments)
        resume_from_checkpoint: Whether to resume from existing experiment
        resume_experiment: Which experiment to resume ('latest' or timestamp)
        custom_name: Custom name suffix for new experiments
        auto_fork_on_config_change: If True, automatically fork on config changes

    Returns:
        Tuple of (exp_dir, checkpoints_dir, samples_dir, logs_dir)

    Raises:
        ValueError: If config changed and auto_fork_on_config_change=False

    Examples:
        # Fresh training
        >>> exp_dir, ckpt_dir, samples_dir, logs_dir = setup_or_resume_experiment(
        ...     model_name="unet",
        ...     degradation="gaussian",
        ...     config=config,
        ...     resume_from_checkpoint=False,
        ...     custom_name="v1"
        ... )

        # Resume training (auto-fork if config changed)
        >>> exp_dir, ckpt_dir, samples_dir, logs_dir = setup_or_resume_experiment(
        ...     model_name="unet",
        ...     degradation="gaussian",
        ...     config=config,
        ...     resume_from_checkpoint=True,
        ...     resume_experiment="latest"
        ... )
    """
    if resume_from_checkpoint:
        # Load existing experiment directory
        parent_exp_dir, checkpoints_dir, samples_dir, logs_dir = (
            load_existing_experiment(
                model_name=model_name,
                degradation=degradation,
                resume_experiment=resume_experiment,
            )
        )

        # Load saved config and compare with current config
        try:
            saved_config = load_experiment_config(parent_exp_dir)
            has_changes, config_changes = compare_configs(saved_config, config)

            if has_changes:
                if auto_fork_on_config_change:
                    # Fork experiment with new config
                    exp_dir, subdirs = fork_experiment(
                        parent_exp_dir=parent_exp_dir,
                        model_name=model_name,
                        degradation=degradation,
                        new_config=config,
                        config_changes=config_changes,
                        custom_name=custom_name,
                        copy_checkpoint="best_model.pth",
                    )

                    # Extract subdirectory paths
                    checkpoints_dir = subdirs["checkpoints"]
                    samples_dir = subdirs["samples"]
                    logs_dir = subdirs["logs"]
                else:
                    # Raise error if auto-fork is disabled
                    changes_str = "\n".join(
                        f"  • {key}: {old} → {new}"
                        for key, (old, new) in config_changes.items()
                    )
                    raise ValueError(
                        f"Configuration has changed but auto_fork_on_config_change=False:\n"
                        f"{changes_str}\n\n"
                        f"Either:\n"
                        f"1. Set auto_fork_on_config_change=True to fork automatically\n"
                        f"2. Use the original config to resume without changes\n"
                        f"3. Set resume_from_checkpoint=False to create a new experiment"
                    )
            else:
                # No changes, use existing experiment
                exp_dir = parent_exp_dir

        except FileNotFoundError:
            # No config.json found in parent (old experiment), use as-is
            exp_dir = parent_exp_dir
    else:
        # New training: Create new experiment directory
        exp_dir, subdirs = setup_experiment(
            model_name=model_name,
            degradation=degradation,
            config=config,
            custom_name=custom_name,
        )

        # Extract subdirectory paths
        checkpoints_dir = subdirs["checkpoints"]
        samples_dir = subdirs["samples"]
        logs_dir = subdirs["logs"]

    return exp_dir, checkpoints_dir, samples_dir, logs_dir


def load_experiment_config(exp_dir: Path) -> dict:
    """
    Load configuration from an experiment directory.

    Args:
        exp_dir: Path to the experiment directory

    Returns:
        Configuration dictionary
    """
    config_path = exp_dir / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    return config


def compare_configs(
    old_config: dict,
    new_config: dict,
    ignore_keys: Optional[Set[str]] = None,
) -> Tuple[bool, Dict[str, Tuple]]:
    """
    Compare two configuration dictionaries and find significant differences.

    Args:
        old_config: Original configuration
        new_config: New configuration to compare
        ignore_keys: Set of keys to ignore in comparison (e.g., resume flags)

    Returns:
        Tuple of (has_changes, changes_dict) where:
        - has_changes: Boolean indicating if there are differences
        - changes_dict: Dict mapping changed keys to (old_value, new_value) tuples
    """
    if ignore_keys is None:
        # Default keys to ignore (related to resume logic, not training)
        ignore_keys = {
            "resume_from_checkpoint",
            "resume_experiment",
            "num_epochs",
            "save_every",
            "val_every",
            "device",  # Device can change between runs
        }

    changes = {}

    def normalize_value(value):
        """Normalize tuple/list for comparison (JSON doesn't preserve tuples)"""
        if isinstance(value, (tuple, list)):
            return list(value)
        return value

    # Check all keys in new config
    for key, new_value in new_config.items():
        if key in ignore_keys:
            continue

        old_value = old_config.get(key)

        # Normalize values for comparison (handle tuple/list equivalence)
        norm_old = normalize_value(old_value)
        norm_new = normalize_value(new_value)

        # Check if value changed
        if norm_old != norm_new:
            changes[key] = (old_value, new_value)

    # Check for removed keys (existed in old but not in new)
    for key in old_config:
        if key in ignore_keys:
            continue
        if key not in new_config:
            changes[key] = (old_config[key], None)

    return len(changes) > 0, changes


def fork_experiment(
    parent_exp_dir: Path,
    model_name: str,
    degradation: str,
    new_config: dict,
    config_changes: Dict[str, Tuple],
    custom_name: Optional[str] = None,
    copy_checkpoint: str = "best_model.pth",
) -> Tuple[Path, Dict[str, Path]]:
    """
    Create a new forked experiment from an existing one with modified config.

    This creates a new experiment directory, copies the checkpoint from the parent,
    and saves metadata about the fork (parent experiment, config changes).

    Args:
        parent_exp_dir: Path to the parent experiment directory
        model_name: Name of the model
        degradation: Type of degradation
        new_config: New configuration for the forked experiment
        config_changes: Dictionary of config changes from compare_configs
        custom_name: Optional custom name suffix
        copy_checkpoint: Name of checkpoint to copy from parent

    Returns:
        Tuple of (exp_dir, subdirs) for the new forked experiment

    Raises:
        FileNotFoundError: If parent checkpoint doesn't exist
    """
    print("\n" + "=" * 80)
    print("🍴 CONFIG CHANGED: Forking experiment")
    print("=" * 80 + "\n")

    print(f"Parent: {parent_exp_dir.name}")
    print(f"Changes: {len(config_changes)} parameter(s) modified\n")

    # Create new experiment with fork suffix
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fork_suffix = f"fork_{custom_name}" if custom_name else "fork"
    exp_name = f"{timestamp}_{fork_suffix}"

    base_dir = get_model_experiments_dir(model_name, degradation)
    exp_dir = base_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    subdirs = {
        "checkpoints": exp_dir / "checkpoints",
        "samples": exp_dir / "samples",
        "logs": exp_dir / "logs",
    }
    for subdir in subdirs.values():
        subdir.mkdir(exist_ok=True)

    # Copy checkpoint from parent
    parent_checkpoint = parent_exp_dir / "checkpoints" / copy_checkpoint
    if not parent_checkpoint.exists():
        raise FileNotFoundError(
            f"Parent checkpoint not found: {parent_checkpoint}\n"
            f"Cannot fork experiment without a checkpoint to copy."
        )

    new_checkpoint = subdirs["checkpoints"] / copy_checkpoint
    shutil.copy2(parent_checkpoint, new_checkpoint)
    print(f"\n✅ Copied checkpoint: {copy_checkpoint}")

    # Copy training history from parent if it exists
    parent_history = parent_exp_dir / "history.json"
    if parent_history.exists():
        new_history = exp_dir / "history.json"
        shutil.copy2(parent_history, new_history)
        print(
            f"✅ Copied training history from parent ({parent_history.stat().st_size} bytes)"
        )
    else:
        print("⚠️  No history.json found in parent (starting with empty history)")

    # Copy experiment stats from parent if they exist (cumulative stats)
    parent_stats = parent_exp_dir / "experiment_stats.json"
    if parent_stats.exists():
        new_stats = exp_dir / "experiment_stats.json"
        shutil.copy2(parent_stats, new_stats)
        print("✅ Copied experiment stats from parent (cumulative)")
    else:
        print("⚠️  No experiment_stats.json found in parent (starting fresh)")

    # Save new config
    with open(exp_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(new_config, f, indent=2)

    # Save fork metadata
    fork_metadata = {
        "parent_experiment": str(parent_exp_dir.relative_to(base_dir.parent)),
        "parent_timestamp": parent_exp_dir.name,
        "fork_timestamp": timestamp,
        "copied_checkpoint": copy_checkpoint,
        "config_changes": {
            key: {"old": old_val, "new": new_val}
            for key, (old_val, new_val) in config_changes.items()
        },
    }

    with open(exp_dir / "fork_metadata.json", "w", encoding="utf-8") as f:
        json.dump(fork_metadata, f, indent=2)

    print(f"\n\u2705 Forked to: {exp_name}")
    print("=" * 80 + "\n")

    return exp_dir, subdirs


def save_training_history(history: dict, exp_dir: Path) -> Path:
    """
    Save training history to experiment directory.

    Args:
        history: Dictionary with training metrics
        exp_dir: Path to the experiment directory

    Returns:
        Path to the saved history file
    """
    history_path = exp_dir / "history.json"

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return history_path


def load_training_history(exp_dir: Path) -> dict:
    """
    Load training history from an experiment directory.

    Args:
        exp_dir: Path to the experiment directory

    Returns:
        History dictionary with training metrics
    """
    history_path = exp_dir / "history.json"

    if not history_path.exists():
        raise FileNotFoundError(f"History file not found: {history_path}")

    with open(history_path, "r") as f:
        history = json.load(f)

    return history


def save_experiment_stats(
    exp_dir: Path,
    total_training_time: float,
    total_epochs_trained: int,
    peak_memory_mb: float,
    avg_epoch_time: float,
    avg_inference_time: float,
) -> Path:
    """
    Save cumulative experiment statistics to experiment directory.
    These stats are preserved and accumulated when forking experiments.

    Args:
        exp_dir: Path to the experiment directory
        total_training_time: Total training time in seconds (cumulative)
        total_epochs_trained: Total number of epochs trained (cumulative)
        peak_memory_mb: Peak memory allocated in MB
        avg_epoch_time: Average time per epoch in seconds
        avg_inference_time: Average inference time in seconds

    Returns:
        Path to the saved stats file
    """
    stats_path = exp_dir / "experiment_stats.json"

    stats = {
        "total_training_time_seconds": round(total_training_time, 2),
        "total_training_time_hours": round(total_training_time / 3600, 2),
        "total_epochs_trained": total_epochs_trained,
        "peak_memory_allocated_mb": round(peak_memory_mb, 2),
        "avg_epoch_time_seconds": round(avg_epoch_time, 2),
        "avg_inference_time_seconds": round(avg_inference_time, 2),
    }

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return stats_path


def load_experiment_stats(exp_dir: Path) -> dict:
    """
    Load experiment statistics from an experiment directory.

    Args:
        exp_dir: Path to the experiment directory

    Returns:
        Stats dictionary with cumulative metrics (empty dict if not found)
    """
    stats_path = exp_dir / "experiment_stats.json"

    if not stats_path.exists():
        return {
            "total_training_time_seconds": 0.0,
            "total_training_time_hours": 0.0,
            "total_epochs_trained": 0,
            "peak_memory_allocated_mb": 0.0,
            "avg_epoch_time_seconds": 0.0,
            "avg_inference_time_seconds": 0.0,
        }

    with open(stats_path, "r") as f:
        stats = json.load(f)

    return stats


def print_training_summary(
    history: dict,
    best_epoch: int,
    best_val_loss: float,
    exp_dir: Path,
    checkpoints_dir: Path,
    samples_dir: Path,
    logs_dir: Path,
) -> None:
    """
    Print a comprehensive training summary.

    Args:
        history: Dictionary with training metrics history
        best_epoch: Epoch number (1-based) of the best model
        best_val_loss: Best validation loss achieved
        exp_dir: Path to the experiment directory
        checkpoints_dir: Path to checkpoints directory
        samples_dir: Path to samples directory
        logs_dir: Path to logs directory
    """
    print("\n" + "=" * 80)
    print("📊 TRAINING SUMMARY")
    print("=" * 80)
    print(f"\nExperiment directory: {exp_dir}")

    # Handle empty history
    if not history.get("train_loss"):
        print("\n⚠️  No training history available")
        print("=" * 80)
        return

    print(f"\nValidation points saved: {len(history['train_loss'])}")
    print(f"Best epoch: {best_epoch}")

    # Find the index of best validation loss in history
    # Cannot use best_epoch - 1 because with val_every > 1, history indices don't match epoch numbers
    # Filter out None values before finding minimum
    if history.get("val_loss") and len(history["val_loss"]) > 0:
        val_losses = [v for v in history["val_loss"] if v is not None]
        if val_losses:
            best_val_loss_value = min(val_losses)
            best_idx = history["val_loss"].index(best_val_loss_value)
        else:
            best_idx = 0
    else:
        best_idx = 0

    # Check if we have validation metrics
    if history.get("val_loss") and len(history["val_loss"]) > best_idx:
        print("\nBest Validation Metrics:")
        print(f"  Loss: {best_val_loss:.4f}")
        print(f"  L1: {history['val_l1'][best_idx]:.4f}")
        print(f"  SSIM: {history['val_ssim'][best_idx]:.4f}")

    # Check if we have training metrics
    if history.get("train_loss") and len(history["train_loss"]) > 0:
        print("\nFinal Training Metrics:")
        print(f"  Loss: {history['train_loss'][-1]:.4f}")
        print(f"  L1: {history['train_l1'][-1]:.4f}")
        print(f"  SSIM: {history['train_ssim'][-1]:.4f}")

    # Print timing and memory statistics
    stats = load_experiment_stats(exp_dir)
    if stats.get("total_training_time_seconds", 0) > 0:
        print("\nPerformance Statistics:")
        print(
            f"  Total training time: {stats['total_training_time_hours']:.2f}h ({stats['total_training_time_seconds']:.0f}s)"
        )
        print(f"  Total epochs trained: {stats['total_epochs_trained']}")
        if stats.get("avg_epoch_time_seconds", 0) > 0:
            print(f"  Avg epoch time: {stats['avg_epoch_time_seconds']:.1f}s")
        if stats.get("avg_inference_time_seconds", 0) > 0:
            print(f"  Avg inference time: {stats['avg_inference_time_seconds']:.1f}s")
        if stats.get("peak_memory_allocated_mb", 0) > 0:
            print(f"  Peak memory: {stats['peak_memory_allocated_mb']:.0f}MB")

    print("\nSaved files:")
    print(f"  Best model: {checkpoints_dir / 'best_model.pth'}")
    print(f"  Training history: {exp_dir / 'history.json'}")
    print(f"  Training curves: {exp_dir / 'training_curves.png'}")
    print(f"  Inference samples: {samples_dir / 'inference_results.png'}")
    print(f"  TensorBoard logs: {logs_dir}")
    print("\n" + "=" * 80)
    print("🎉 All done!")
    print("=" * 80)
