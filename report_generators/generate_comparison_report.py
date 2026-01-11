"""
Script to generate a comprehensive comparison report across all experiments
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import statistics


def load_experiment_summary(summary_file: Path) -> Dict[str, Any]:
    """Load experiment summary"""
    with open(summary_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    experiments_base = Path(r"c:\Development\Deep_Learning\Progetto1\Image-Enhancement\experiments\pix2pix\gaussian")
    
    # Collect all summaries
    experiments = []
    for exp_dir in sorted(experiments_base.iterdir()):
        if exp_dir.is_dir():
            summary_file = exp_dir / "experiment_summary.json"
            if summary_file.exists():
                try:
                    summary = load_experiment_summary(summary_file)
                    experiments.append(summary)
                except:
                    pass
    
    print(f"✅ Loaded {len(experiments)} experiment summaries")
    
    # Generate comparison report
    report_lines = []
    report_lines.append("# Complete Experiments Comparison Report")
    report_lines.append("# Pix2Pix Model - Gaussian Noise Degradation\n")
    report_lines.append(f"Total Experiments: {len(experiments)}\n")
    
    # Summary table header
    report_lines.append("## Summary Table\n")
    report_lines.append("| Experiment ID | Timestamp | LR_G | LR_D | λ_L1 | Patch | Epochs | PSNR | SSIM | MAE |")
    report_lines.append("|---------------|-----------|------|------|------|-------|--------|------|------|-----|")
    
    # Collect metrics for ranking
    experiments_with_metrics = []
    
    for exp in experiments:
        exp_id = exp.get("experiment_id", "N/A")
        timestamp = exp.get("timestamp", "N/A")[:16]  # YYYY-MM-DD HH:MM
        
        config = exp.get("configuration", {})
        hyper = config.get("training_hyperparameters", {})
        dataset = config.get("dataset", {})
        
        lr_g = hyper.get("learning_rate_G", 0)
        lr_d = hyper.get("learning_rate_D", 0)
        lambda_l1 = hyper.get("lambda_L1", 0)
        patch_size = dataset.get("patch_size", 0)
        
        results = exp.get("results", {})
        epochs = results.get("training_epochs_completed", 0)
        
        metrics = results.get("final_metrics", {})
        if isinstance(metrics, dict) and "mean" in metrics:
            mean_metrics = metrics.get("mean", {})
            psnr = mean_metrics.get("psnr", 0)
            ssim = mean_metrics.get("ssim", 0)
            mae = mean_metrics.get("mae", 0)
            
            psnr_str = f"{psnr:.2f}" if psnr > 0 else "N/A"
            ssim_str = f"{ssim:.3f}" if ssim > 0 else "N/A"
            mae_str = f"{mae:.4f}" if mae > 0 else "N/A"
            
            if psnr > 0:
                experiments_with_metrics.append({
                    "id": exp_id,
                    "psnr": psnr,
                    "ssim": ssim,
                    "mae": mae,
                    "lr_g": lr_g,
                    "lr_d": lr_d,
                    "lambda_l1": lambda_l1
                })
        else:
            psnr_str = "N/A"
            ssim_str = "N/A"
            mae_str = "N/A"
        
        # Format learning rates
        lr_g_str = f"{lr_g:.0e}" if lr_g > 0 else "N/A"
        lr_d_str = f"{lr_d:.0e}" if lr_d > 0 else "N/A"
        
        report_lines.append(f"| {exp_id[:25]}{'...' if len(exp_id) > 25 else ''} | {timestamp} | {lr_g_str} | {lr_d_str} | {lambda_l1} | {patch_size} | {epochs} | {psnr_str} | {ssim_str} | {mae_str} |")
    
    # Rankings
    if experiments_with_metrics:
        report_lines.append("\n## 🏆 Rankings\n")
        
        # Best PSNR
        best_psnr = max(experiments_with_metrics, key=lambda x: x["psnr"])
        report_lines.append("### Best PSNR")
        report_lines.append(f"- **{best_psnr['id']}**: {best_psnr['psnr']:.2f} dB")
        report_lines.append(f"  - SSIM: {best_psnr['ssim']:.3f}")
        report_lines.append(f"  - MAE: {best_psnr['mae']:.4f}")
        report_lines.append(f"  - Learning Rates: G={best_psnr['lr_g']:.0e}, D={best_psnr['lr_d']:.0e}")
        report_lines.append(f"  - Lambda L1: {best_psnr['lambda_l1']}\n")
        
        # Best SSIM
        best_ssim = max(experiments_with_metrics, key=lambda x: x["ssim"])
        report_lines.append("### Best SSIM")
        report_lines.append(f"- **{best_ssim['id']}**: {best_ssim['ssim']:.3f}")
        report_lines.append(f"  - PSNR: {best_ssim['psnr']:.2f} dB")
        report_lines.append(f"  - MAE: {best_ssim['mae']:.4f}")
        report_lines.append(f"  - Learning Rates: G={best_ssim['lr_g']:.0e}, D={best_ssim['lr_d']:.0e}")
        report_lines.append(f"  - Lambda L1: {best_ssim['lambda_l1']}\n")
        
        # Best MAE (lowest is best)
        best_mae = min(experiments_with_metrics, key=lambda x: x["mae"])
        report_lines.append("### Best MAE (Lowest Error)")
        report_lines.append(f"- **{best_mae['id']}**: {best_mae['mae']:.4f}")
        report_lines.append(f"  - PSNR: {best_mae['psnr']:.2f} dB")
        report_lines.append(f"  - SSIM: {best_mae['ssim']:.3f}")
        report_lines.append(f"  - Learning Rates: G={best_mae['lr_g']:.0e}, D={best_mae['lr_d']:.0e}")
        report_lines.append(f"  - Lambda L1: {best_mae['lambda_l1']}\n")
        
        # Statistics
        report_lines.append("## 📊 Overall Statistics\n")
        psnr_values = [e["psnr"] for e in experiments_with_metrics]
        ssim_values = [e["ssim"] for e in experiments_with_metrics]
        mae_values = [e["mae"] for e in experiments_with_metrics]
        
        report_lines.append(f"- **PSNR**: Mean={statistics.mean(psnr_values):.2f} dB, Std={statistics.stdev(psnr_values) if len(psnr_values) > 1 else 0:.2f} dB, Range=[{min(psnr_values):.2f}, {max(psnr_values):.2f}]")
        report_lines.append(f"- **SSIM**: Mean={statistics.mean(ssim_values):.3f}, Std={statistics.stdev(ssim_values) if len(ssim_values) > 1 else 0:.3f}, Range=[{min(ssim_values):.3f}, {max(ssim_values):.3f}]")
        report_lines.append(f"- **MAE**: Mean={statistics.mean(mae_values):.4f}, Std={statistics.stdev(mae_values) if len(mae_values) > 1 else 0:.4f}, Range=[{min(mae_values):.4f}, {max(mae_values):.4f}]\n")
    
    # Hyperparameter analysis
    report_lines.append("## 🔬 Hyperparameter Analysis\n")
    
    if experiments_with_metrics:
        # Learning rate analysis
        report_lines.append("### Learning Rate Impact")
        lr_g_unique = sorted(set(e["lr_g"] for e in experiments_with_metrics))
        report_lines.append(f"- Tested Generator LRs: {', '.join([f'{lr:.0e}' for lr in lr_g_unique])}")
        
        # Group by LR_G and show average PSNR
        for lr_g in lr_g_unique:
            exps_with_lr = [e for e in experiments_with_metrics if e["lr_g"] == lr_g]
            avg_psnr = statistics.mean([e["psnr"] for e in exps_with_lr])
            avg_ssim = statistics.mean([e["ssim"] for e in exps_with_lr])
            report_lines.append(f"  - LR_G={lr_g:.0e}: Avg PSNR={avg_psnr:.2f} dB, Avg SSIM={avg_ssim:.3f} ({len(exps_with_lr)} experiments)")
        
        report_lines.append("")
        
        # Lambda L1 analysis
        lambda_unique = sorted(set(e["lambda_l1"] for e in experiments_with_metrics))
        report_lines.append("### Lambda L1 Impact")
        report_lines.append(f"- Tested Lambda L1 values: {', '.join([str(int(l)) for l in lambda_unique])}")
        
        for lambda_val in lambda_unique:
            exps_with_lambda = [e for e in experiments_with_metrics if e["lambda_l1"] == lambda_val]
            avg_psnr = statistics.mean([e["psnr"] for e in exps_with_lambda])
            avg_ssim = statistics.mean([e["ssim"] for e in exps_with_lambda])
            report_lines.append(f"  - λ_L1={int(lambda_val)}: Avg PSNR={avg_psnr:.2f} dB, Avg SSIM={avg_ssim:.3f} ({len(exps_with_lambda)} experiments)")
        
        report_lines.append("")
    
    # Recommendations
    report_lines.append("## 💡 Recommendations\n")
    
    if experiments_with_metrics:
        best_overall = max(experiments_with_metrics, key=lambda x: x["psnr"] + 10*x["ssim"])  # Weighted score
        report_lines.append(f"### Best Overall Configuration")
        report_lines.append(f"Based on weighted metrics (PSNR + 10×SSIM):")
        report_lines.append(f"- **Experiment**: {best_overall['id']}")
        report_lines.append(f"- **Results**: PSNR={best_overall['psnr']:.2f} dB, SSIM={best_overall['ssim']:.3f}, MAE={best_overall['mae']:.4f}")
        report_lines.append(f"- **Config**: LR_G={best_overall['lr_g']:.0e}, LR_D={best_overall['lr_d']:.0e}, λ_L1={int(best_overall['lambda_l1'])}")
        report_lines.append("")
    
    report_lines.append("### General Observations")
    report_lines.append("1. Monitor training stability across different learning rates")
    report_lines.append("2. Lambda L1 parameter significantly affects reconstruction quality")
    report_lines.append("3. Extended training (100+ epochs) generally improves results")
    report_lines.append("4. Patch size affects training speed and memory usage")
    report_lines.append("5. Consider ensemble methods or model averaging for best results")
    
    # Save report
    output_file = experiments_base / "complete_experiments_comparison.md"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\n✅ Comparison report saved to: {output_file}")
    print(f"📊 Analyzed {len(experiments)} experiments, {len(experiments_with_metrics)} with complete metrics")
    
    if experiments_with_metrics:
        print(f"\n🏆 Best Results:")
        print(f"   PSNR: {max(e['psnr'] for e in experiments_with_metrics):.2f} dB")
        print(f"   SSIM: {max(e['ssim'] for e in experiments_with_metrics):.3f}")
        print(f"   MAE:  {min(e['mae'] for e in experiments_with_metrics):.4f}")


if __name__ == "__main__":
    main()
