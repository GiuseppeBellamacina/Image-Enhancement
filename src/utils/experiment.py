"""
Experiment setup utilities for managing output directories and configurations
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional

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
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Print summary
    print(f"\n📁 Experiment directory: {exp_dir}")
    if create_subdirs:
        print(f"   Checkpoints: {subdirs['checkpoints']}")
        print(f"   Samples: {subdirs['samples']}")
        print(f"   Logs: {subdirs['logs']}")

    return exp_dir, subdirs


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

    with open(history_path, "w") as f:
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
        best_epoch: Index of the best epoch (0-based)
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
    print(f"\nTraining completed: {len(history['train_loss'])} epochs")
    print(f"Best epoch: {best_epoch + 1}")
    print("\nBest Validation Metrics:")
    print(f"  Loss: {best_val_loss:.4f}")
    print(f"  L1: {history['val_l1'][best_epoch]:.4f}")
    print(f"  SSIM: {history['val_ssim'][best_epoch]:.4f}")
    print("\nFinal Training Metrics:")
    print(f"  Loss: {history['train_loss'][-1]:.4f}")
    print(f"  L1: {history['train_l1'][-1]:.4f}")
    print(f"  SSIM: {history['train_ssim'][-1]:.4f}")
    print("\nSaved files:")
    print(f"  ✓ Best model: {checkpoints_dir / 'best_model.pth'}")
    print(f"  ✓ Training history: {exp_dir / 'history.json'}")
    print(f"  ✓ Training curves: {exp_dir / 'training_curves.png'}")
    print(f"  ✓ Inference samples: {samples_dir / 'inference_results.png'}")
    print(f"  ✓ TensorBoard logs: {logs_dir}")
    print("\n" + "=" * 80)
    print("🎉 All done!")
    print("=" * 80)
