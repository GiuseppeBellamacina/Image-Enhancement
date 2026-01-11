"""
Script to generate a concise summary with hyperparameters and training metrics
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional


def load_json(file_path: Path) -> Optional[Dict[str, Any]]:
    """Load JSON file if it exists"""
    if file_path.exists():
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def calculate_final_metrics(history: Dict[str, Any]) -> Dict[str, float]:
    """Calculate final (last epoch) metrics from history"""
    metrics = {}
    
    if history:
        # Training metrics - take last values
        if 'train_loss_G' in history and history['train_loss_G']:
            metrics['train_loss_G'] = history['train_loss_G'][-1]
        if 'train_loss_D' in history and history['train_loss_D']:
            metrics['train_loss_D'] = history['train_loss_D'][-1]
        if 'train_loss_G_GAN' in history and history['train_loss_G_GAN']:
            metrics['train_loss_G_GAN'] = history['train_loss_G_GAN'][-1]
        if 'train_loss_G_L1' in history and history['train_loss_G_L1']:
            metrics['train_loss_G_L1'] = history['train_loss_G_L1'][-1]
        
        # Validation metrics - take last values
        if 'val_loss' in history and history['val_loss']:
            metrics['val_loss'] = history['val_loss'][-1]
        if 'val_psnr' in history and history['val_psnr']:
            metrics['val_psnr'] = history['val_psnr'][-1]
        if 'val_ssim' in history and history['val_ssim']:
            metrics['val_ssim'] = history['val_ssim'][-1]
        
        # Count epochs
        if 'train_loss_G' in history:
            metrics['epochs_completed'] = len(history['train_loss_G'])
    
    return metrics


def main():
    experiments_base = Path(r"c:\Development\Deep_Learning\Progetto1\Image-Enhancement\experiments\pix2pix\gaussian")
    
    # Collect all experiments
    experiments_data = []
    
    for exp_dir in sorted(experiments_base.iterdir()):
        if exp_dir.is_dir() and not exp_dir.name.startswith('.'):
            config = load_json(exp_dir / "config.json")
            history = load_json(exp_dir / "history.json")
            
            if not config:
                continue
            
            # Extract hyperparameters
            hyperparams = {
                "experiment_id": exp_dir.name,
                "batch_size": config.get("batch_size"),
                "num_epochs": config.get("num_epochs"),
                "learning_rate_G": config.get("learning_rate_G"),
                "learning_rate_D": config.get("learning_rate_D"),
                "beta1": config.get("beta1"),
                "beta2": config.get("beta2"),
                "lambda_L1": config.get("lambda_L1"),
                "patch_size": config.get("patch_size"),
                "patches_per_image": config.get("patches_per_image"),
                "generator_features": config.get("generator_features"),
                "discriminator_features": config.get("discriminator_features"),
                "noise_sigma": config.get("noise_sigma"),
                "gradient_clip": config.get("gradient_clip"),
                "scheduler": config.get("scheduler"),
                "warmup_epochs": config.get("warmup_epochs"),
                "patience": config.get("patience"),
                "use_amp": config.get("use_amp"),
                "seed": config.get("seed")
            }
            
            # Calculate metrics
            metrics = calculate_final_metrics(history)
            
            # Combine
            experiments_data.append({
                "hyperparameters": hyperparams,
                "final_metrics": metrics
            })
    
    # Generate output file
    output_file = experiments_base / "hyperparameters_and_metrics.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(experiments_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Generated summary for {len(experiments_data)} experiments")
    print(f"📄 Saved to: {output_file}")
    
    # Also generate a readable markdown version
    md_lines = []
    md_lines.append("# Experiments: Hyperparameters and Training Metrics\n")
    
    for exp_data in experiments_data:
        hyper = exp_data["hyperparameters"]
        metrics = exp_data["final_metrics"]
        
        md_lines.append(f"## {hyper['experiment_id']}\n")
        
        md_lines.append("### Hyperparameters")
        md_lines.append(f"- **Batch Size**: {hyper['batch_size']}")
        md_lines.append(f"- **Num Epochs**: {hyper['num_epochs']}")
        md_lines.append(f"- **Learning Rate G**: {hyper['learning_rate_G']}")
        md_lines.append(f"- **Learning Rate D**: {hyper['learning_rate_D']}")
        md_lines.append(f"- **Beta1**: {hyper['beta1']}")
        md_lines.append(f"- **Beta2**: {hyper['beta2']}")
        md_lines.append(f"- **Lambda L1**: {hyper['lambda_L1']}")
        md_lines.append(f"- **Patch Size**: {hyper['patch_size']}")
        md_lines.append(f"- **Patches per Image**: {hyper['patches_per_image']}")
        md_lines.append(f"- **Generator Features**: {hyper['generator_features']}")
        md_lines.append(f"- **Discriminator Features**: {hyper['discriminator_features']}")
        md_lines.append(f"- **Noise Sigma**: {hyper['noise_sigma']}")
        md_lines.append(f"- **Gradient Clip**: {hyper['gradient_clip']}")
        md_lines.append(f"- **Scheduler**: {hyper['scheduler']}")
        md_lines.append(f"- **Warmup Epochs**: {hyper['warmup_epochs']}")
        md_lines.append(f"- **Patience**: {hyper['patience']}")
        md_lines.append(f"- **Use AMP**: {hyper['use_amp']}")
        md_lines.append(f"- **Seed**: {hyper['seed']}\n")
        
        md_lines.append("### Final Training Metrics")
        if metrics:
            if 'epochs_completed' in metrics:
                md_lines.append(f"- **Epochs Completed**: {metrics['epochs_completed']}")
            if 'train_loss_G' in metrics:
                md_lines.append(f"- **Train Loss G**: {metrics['train_loss_G']:.6f}")
            if 'train_loss_G_GAN' in metrics:
                md_lines.append(f"- **Train Loss G GAN** (adversarial): {metrics['train_loss_G_GAN']:.6f}")
            if 'train_loss_G_L1' in metrics:
                md_lines.append(f"- **Train Loss G L1** (reconstruction): {metrics['train_loss_G_L1']:.6f}")
            if 'train_loss_D' in metrics:
                md_lines.append(f"- **Train Loss D**: {metrics['train_loss_D']:.6f}")
            if 'val_loss' in metrics:
                md_lines.append(f"- **Val Loss**: {metrics['val_loss']:.6f}")
            if 'val_psnr' in metrics:
                md_lines.append(f"- **Val PSNR**: {metrics['val_psnr']:.2f} dB")
            if 'val_ssim' in metrics:
                md_lines.append(f"- **Val SSIM**: {metrics['val_ssim']:.4f}")
        else:
            md_lines.append("- No training metrics available")
        
        md_lines.append("\n---\n")
    
    md_output_file = experiments_base / "hyperparameters_and_metrics.md"
    with open(md_output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    
    print(f"📄 Markdown version: {md_output_file}")


if __name__ == "__main__":
    main()
