"""
Script to generate experiment summary files for all experiments
Combines config.json, evaluation_metrics.json, and history.json into a single summary
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


def load_json(file_path: Path) -> Optional[Dict[str, Any]]:
    """Load JSON file if it exists"""
    if file_path.exists():
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def extract_timestamp(experiment_id: str) -> str:
    """Extract timestamp from experiment ID"""
    try:
        # Format: YYYYMMDD_HHMMSS or YYYYMMDD_HHMMSS_description
        date_time = experiment_id.split('_')[:2]
        if len(date_time) == 2 and date_time[0].isdigit() and date_time[1].isdigit():
            date_str = date_time[0]
            time_str = date_time[1]
            dt = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
            return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        pass
    return "Unknown"


def count_epochs_from_history(history: Dict[str, Any]) -> int:
    """Count number of epochs from history"""
    if history and 'train_loss_G' in history:
        return len(history['train_loss_G'])
    return 0


def create_experiment_summary(experiment_dir: Path) -> Dict[str, Any]:
    """Create a comprehensive summary for an experiment"""
    
    experiment_id = experiment_dir.name
    
    # Load all JSON files
    config = load_json(experiment_dir / "config.json")
    evaluation = load_json(experiment_dir / "evaluation_metrics.json")
    history = load_json(experiment_dir / "history.json")
    
    if not config:
        return None
    
    # Extract timestamp
    timestamp = extract_timestamp(experiment_id)
    
    # Count epochs
    epochs_completed = count_epochs_from_history(history)
    
    # Build summary structure
    summary = {
        "experiment_id": experiment_id,
        "model": "pix2pix",
        "degradation": "gaussian",
        "timestamp": timestamp,
        
        "configuration": {
            "dataset": {
                "train_degraded_dir": config.get("train_degraded_dir"),
                "train_clean_dir": config.get("train_clean_dir"),
                "val_degraded_dir": config.get("val_degraded_dir"),
                "val_clean_dir": config.get("val_clean_dir"),
                "patch_size": config.get("patch_size"),
                "patches_per_image": config.get("patches_per_image"),
                "batch_size": config.get("batch_size"),
                "num_workers": config.get("num_workers")
            },
            "model_architecture": {
                "generator_features": config.get("generator_features"),
                "discriminator_features": config.get("discriminator_features")
            },
            "training_hyperparameters": {
                "num_epochs": config.get("num_epochs"),
                "learning_rate_G": config.get("learning_rate_G"),
                "learning_rate_D": config.get("learning_rate_D"),
                "beta1": config.get("beta1"),
                "beta2": config.get("beta2"),
                "lambda_L1": config.get("lambda_L1"),
                "gradient_clip": config.get("gradient_clip"),
                "scheduler": config.get("scheduler"),
                "warmup_epochs": config.get("warmup_epochs"),
                "min_lr": config.get("min_lr"),
                "patience": config.get("patience")
            },
            "degradation_parameters": {
                "noise_sigma": config.get("noise_sigma")
            },
            "training_setup": {
                "save_every": config.get("save_every"),
                "val_every": config.get("val_every"),
                "use_amp": config.get("use_amp"),
                "device": config.get("device"),
                "seed": config.get("seed"),
                "resume_from_checkpoint": config.get("resume_from_checkpoint"),
                "resume_experiment": config.get("resume_experiment")
            }
        }
    }
    
    # Add results if available
    if evaluation:
        summary["results"] = {
            "final_metrics": {
                "mean": evaluation.get("mean", {}),
                "std": evaluation.get("std", {}),
                "n_images": evaluation.get("n_images", 0)
            },
            "training_epochs_completed": epochs_completed
        }
    else:
        summary["results"] = {
            "final_metrics": "No evaluation metrics available",
            "training_epochs_completed": epochs_completed
        }
    
    # Add notes section
    notes = {
        "description": f"Pix2Pix training on Gaussian noise (sigma={config.get('noise_sigma', 'N/A')})",
        "status": "Completed" if evaluation else "Incomplete (no evaluation)",
        "key_observations": []
    }
    
    # Add key observations based on config
    if config.get("lambda_L1", 100) != 100:
        notes["key_observations"].append(f"Non-standard lambda_L1: {config.get('lambda_L1')}")
    
    if config.get("patch_size") == 128:
        notes["key_observations"].append("Using smaller patches (128x128)")
    elif config.get("patch_size") == 256:
        notes["key_observations"].append("Using larger patches (256x256)")
    
    if config.get("scheduler"):
        notes["key_observations"].append(f"Learning rate scheduler: {config.get('scheduler')}")
    
    if epochs_completed > 0:
        notes["key_observations"].append(f"Training stopped at epoch {epochs_completed}")
    
    summary["notes"] = notes
    
    return summary


def main():
    """Generate summaries for all experiments"""
    
    experiments_base = Path(r"c:\Development\Deep_Learning\Progetto1\Image-Enhancement\experiments\pix2pix\gaussian")
    
    if not experiments_base.exists():
        print(f"❌ Experiments directory not found: {experiments_base}")
        return
    
    # Get all experiment directories
    experiment_dirs = [d for d in experiments_base.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    print(f"📊 Found {len(experiment_dirs)} experiment directories")
    print()
    
    summaries_created = 0
    summaries_skipped = 0
    
    for exp_dir in sorted(experiment_dirs):
        print(f"Processing: {exp_dir.name}")
        
        # Check if summary already exists
        summary_file = exp_dir / "experiment_summary.json"
        if summary_file.exists():
            print(f"  ⏭️  Summary already exists, skipping...")
            summaries_skipped += 1
            continue
        
        # Check if config exists
        if not (exp_dir / "config.json").exists():
            print(f"  ⚠️  No config.json found, skipping...")
            summaries_skipped += 1
            continue
        
        # Create summary
        summary = create_experiment_summary(exp_dir)
        
        if summary:
            # Save summary
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            print(f"  ✅ Summary created")
            summaries_created += 1
        else:
            print(f"  ❌ Failed to create summary")
            summaries_skipped += 1
        
        print()
    
    print(f"\n{'='*60}")
    print(f"📊 Summary:")
    print(f"   Created: {summaries_created}")
    print(f"   Skipped: {summaries_skipped}")
    print(f"   Total:   {len(experiment_dirs)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
