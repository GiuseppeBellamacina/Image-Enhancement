"""
Script to concatenate all evaluation_metrics.json files from experiments
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional


def load_evaluation_metrics(metrics_file: Path) -> Optional[Dict[str, Any]]:
    """Load evaluation metrics file if it exists"""
    if metrics_file.exists():
        with open(metrics_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def main():
    experiments_base = Path(r"c:\Development\Deep_Learning\Progetto1\Image-Enhancement\experiments\pix2pix\gaussian")
    
    # Collect all evaluation metrics
    all_evaluations = []
    
    for exp_dir in sorted(experiments_base.iterdir()):
        if exp_dir.is_dir() and not exp_dir.name.startswith('.'):
            metrics_file = exp_dir / "evaluation_metrics.json"
            
            if metrics_file.exists():
                metrics = load_evaluation_metrics(metrics_file)
                if metrics:
                    all_evaluations.append({
                        "experiment_id": exp_dir.name,
                        "evaluation_metrics": metrics
                    })
    
    # Save concatenated JSON
    output_json = experiments_base / "all_evaluation_metrics.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(all_evaluations, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Concatenated {len(all_evaluations)} evaluation metrics files")
    print(f"📄 Saved to: {output_json}")
    
    # Generate readable markdown version
    md_lines = []
    md_lines.append("# All Evaluation Metrics\n")
    md_lines.append(f"Total Experiments with Evaluation: {len(all_evaluations)}\n")
    
    # Summary table
    md_lines.append("## Summary Table\n")
    md_lines.append("| Experiment ID | N Images | PSNR (mean±std) | SSIM (mean±std) | MAE (mean±std) | MSE (mean±std) |")
    md_lines.append("|---------------|----------|-----------------|-----------------|----------------|----------------|")
    
    for eval_data in all_evaluations:
        exp_id = eval_data["experiment_id"]
        metrics = eval_data["evaluation_metrics"]
        
        if "mean" in metrics:
            mean = metrics["mean"]
            std = metrics["std"]
            n_images = metrics.get("n_images", 0)
            
            psnr_str = f"{mean['psnr']:.2f}±{std['psnr']:.2f}"
            ssim_str = f"{mean['ssim']:.3f}±{std['ssim']:.3f}"
            mae_str = f"{mean['mae']:.4f}±{std['mae']:.4f}"
            mse_str = f"{mean['mse']:.5f}±{std['mse']:.5f}"
            
            md_lines.append(f"| {exp_id[:30]}{'...' if len(exp_id) > 30 else ''} | {n_images} | {psnr_str} | {ssim_str} | {mae_str} | {mse_str} |")
    
    md_lines.append("\n---\n")
    
    # Detailed per-experiment results
    md_lines.append("## Detailed Results\n")
    
    for eval_data in all_evaluations:
        exp_id = eval_data["experiment_id"]
        metrics = eval_data["evaluation_metrics"]
        
        md_lines.append(f"### {exp_id}\n")
        
        if "mean" in metrics:
            mean = metrics["mean"]
            std = metrics["std"]
            n_images = metrics.get("n_images", 0)
            
            md_lines.append(f"**Summary Statistics** ({n_images} images):\n")
            md_lines.append(f"- **PSNR**: {mean['psnr']:.2f} dB ± {std['psnr']:.2f}")
            md_lines.append(f"- **SSIM**: {mean['ssim']:.4f} ± {std['ssim']:.4f}")
            md_lines.append(f"- **MAE**: {mean['mae']:.6f} ± {std['mae']:.6f}")
            md_lines.append(f"- **MSE**: {mean['mse']:.6f} ± {std['mse']:.6f}\n")
            
            # Per-image details
            if "per_image" in metrics:
                md_lines.append("**Per-Image Results**:\n")
                md_lines.append("| Image | PSNR (dB) | SSIM | MAE | MSE |")
                md_lines.append("|-------|-----------|------|-----|-----|")
                
                for img_metrics in metrics["per_image"]:
                    filename = img_metrics.get("filename", "N/A")
                    psnr = img_metrics.get("psnr", 0)
                    ssim = img_metrics.get("ssim", 0)
                    mae = img_metrics.get("mae", 0)
                    mse = img_metrics.get("mse", 0)
                    
                    md_lines.append(f"| {filename} | {psnr:.2f} | {ssim:.4f} | {mae:.6f} | {mse:.6f} |")
                
                md_lines.append("")
        else:
            md_lines.append("No evaluation metrics available\n")
        
        md_lines.append("---\n")
    
    # Best performers
    md_lines.append("## 🏆 Best Performers\n")
    
    valid_evals = [e for e in all_evaluations if "mean" in e["evaluation_metrics"]]
    
    if valid_evals:
        # Best PSNR
        best_psnr = max(valid_evals, key=lambda x: x["evaluation_metrics"]["mean"]["psnr"])
        md_lines.append("### Best PSNR")
        md_lines.append(f"- **Experiment**: {best_psnr['experiment_id']}")
        md_lines.append(f"- **PSNR**: {best_psnr['evaluation_metrics']['mean']['psnr']:.2f} dB")
        md_lines.append(f"- **SSIM**: {best_psnr['evaluation_metrics']['mean']['ssim']:.4f}")
        md_lines.append(f"- **MAE**: {best_psnr['evaluation_metrics']['mean']['mae']:.6f}\n")
        
        # Best SSIM
        best_ssim = max(valid_evals, key=lambda x: x["evaluation_metrics"]["mean"]["ssim"])
        md_lines.append("### Best SSIM")
        md_lines.append(f"- **Experiment**: {best_ssim['experiment_id']}")
        md_lines.append(f"- **SSIM**: {best_ssim['evaluation_metrics']['mean']['ssim']:.4f}")
        md_lines.append(f"- **PSNR**: {best_ssim['evaluation_metrics']['mean']['psnr']:.2f} dB")
        md_lines.append(f"- **MAE**: {best_ssim['evaluation_metrics']['mean']['mae']:.6f}\n")
        
        # Best MAE (lowest)
        best_mae = min(valid_evals, key=lambda x: x["evaluation_metrics"]["mean"]["mae"])
        md_lines.append("### Best MAE (Lowest Error)")
        md_lines.append(f"- **Experiment**: {best_mae['experiment_id']}")
        md_lines.append(f"- **MAE**: {best_mae['evaluation_metrics']['mean']['mae']:.6f}")
        md_lines.append(f"- **PSNR**: {best_mae['evaluation_metrics']['mean']['psnr']:.2f} dB")
        md_lines.append(f"- **SSIM**: {best_mae['evaluation_metrics']['mean']['ssim']:.4f}\n")
    
    # Save markdown
    output_md = experiments_base / "all_evaluation_metrics.md"
    with open(output_md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    
    print(f"📄 Markdown version: {output_md}")
    
    if valid_evals:
        print(f"\n🏆 Best Results:")
        print(f"   PSNR: {max(e['evaluation_metrics']['mean']['psnr'] for e in valid_evals):.2f} dB")
        print(f"   SSIM: {max(e['evaluation_metrics']['mean']['ssim'] for e in valid_evals):.4f}")
        print(f"   MAE:  {min(e['evaluation_metrics']['mean']['mae'] for e in valid_evals):.6f}")


if __name__ == "__main__":
    main()
