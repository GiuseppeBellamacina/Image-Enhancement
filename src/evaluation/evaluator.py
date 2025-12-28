"""
Full evaluation pipeline for image restoration models
Processes entire dataset and calculates comprehensive metrics
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from tqdm.auto import tqdm
import numpy as np
import cv2
import json

from .inference import (
    normalize_image,
    denormalize_image,
    sliding_window_inference,
)
from .metrics import calculate_all_metrics


class ImageRestorationEvaluator:
    """
    Evaluator for image restoration models.
    Processes full-resolution images and calculates metrics.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        patch_size: int = 256,
        overlap: int = 32,
    ):
        """
        Initialize evaluator.

        Args:
            model: Trained restoration model
            device: Device to run inference on
            patch_size: Size of patches for sliding window inference
            overlap: Overlap between patches
        """
        self.model = model
        self.device = device
        self.patch_size = patch_size
        self.overlap = overlap

        self.model.eval()

    def evaluate_image(
        self,
        degraded_path: Path,
        clean_path: Path,
        save_output: bool = False,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, float]:
        """
        Evaluate model on a single image pair.

        Args:
            degraded_path: Path to degraded image
            clean_path: Path to clean target image
            save_output: Whether to save restored image
            output_dir: Directory to save output (required if save_output=True)

        Returns:
            Dictionary with metrics (PSNR, SSIM, MAE, MSE)
        """
        # Load images
        degraded_bgr = cv2.imread(str(degraded_path))
        clean_bgr = cv2.imread(str(clean_path))

        if degraded_bgr is None or clean_bgr is None:
            raise ValueError(f"Could not load images: {degraded_path}, {clean_path}")

        # Convert to RGB
        degraded_rgb = cv2.cvtColor(degraded_bgr, cv2.COLOR_BGR2RGB)
        clean_rgb = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB)

        # Normalize to tensors
        degraded_tensor = normalize_image(degraded_rgb)
        clean_tensor = normalize_image(clean_rgb)

        # Run inference
        with torch.no_grad():
            restored_tensor = sliding_window_inference(
                model=self.model,
                image=degraded_tensor,
                patch_size=self.patch_size,
                overlap=self.overlap,
                device=self.device,
            )

        # Move clean tensor to same device as restored for metrics calculation
        clean_tensor = clean_tensor.to(self.device)

        # Denormalize tensors from [-1, 1] to [0, 1] for metrics calculation
        restored_denorm = (restored_tensor * 0.5) + 0.5
        clean_denorm = (clean_tensor * 0.5) + 0.5

        # Calculate metrics (on denormalized tensors in [0, 1] range)
        metrics = calculate_all_metrics(
            restored=restored_denorm.unsqueeze(0),
            target=clean_denorm.unsqueeze(0),
            max_val=1.0,
        )

        # Save output if requested
        if save_output and output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)

            # Denormalize and save
            restored_rgb = denormalize_image(restored_tensor)
            restored_bgr = cv2.cvtColor(restored_rgb, cv2.COLOR_RGB2BGR)

            output_path = output_dir / degraded_path.name
            cv2.imwrite(str(output_path), restored_bgr)

        return metrics

    def evaluate_dataset(
        self,
        degraded_dir: Path,
        clean_dir: Path,
        output_dir: Optional[Path] = None,
        save_outputs: bool = False,
        max_images: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate model on entire dataset.

        Args:
            degraded_dir: Directory with degraded images
            clean_dir: Directory with clean target images
            output_dir: Directory to save restored images (optional)
            save_outputs: Whether to save restored images
            max_images: Maximum number of images to evaluate (None for all)

        Returns:
            Dictionary with:
                - 'per_image': List of per-image metrics
                - 'mean': Mean metrics across dataset
                - 'std': Standard deviation of metrics
        """
        degraded_dir = Path(degraded_dir)
        clean_dir = Path(clean_dir)

        # Get image files
        degraded_files = sorted(
            list(degraded_dir.glob("*.png")) + list(degraded_dir.glob("*.jpg"))
        )

        if max_images is not None:
            degraded_files = degraded_files[:max_images]

        # Process all images
        all_metrics: List[Dict[str, Union[str, float]]] = []

        print(f"\n📊 Evaluating {len(degraded_files)} images...")

        for degraded_path in tqdm(degraded_files, desc="Processing images"):
            # Find corresponding clean image
            clean_path = clean_dir / degraded_path.name

            if not clean_path.exists():
                print(f"⚠️  Clean image not found: {clean_path.name}, skipping")
                continue

            # Evaluate image
            try:
                metrics = self.evaluate_image(
                    degraded_path=degraded_path,
                    clean_path=clean_path,
                    save_output=save_outputs,
                    output_dir=output_dir,
                )

                # Add filename to metrics
                metrics_with_name: Dict[str, Union[str, float]] = {
                    **metrics,
                    "filename": degraded_path.name,
                }
                all_metrics.append(metrics_with_name)

            except Exception as e:
                print(f"⚠️  Error processing {degraded_path.name}: {e}")
                continue

        # Calculate statistics
        metric_names = ["psnr", "ssim", "mae", "mse"]

        mean_metrics = {
            name: float(np.mean([float(m[name]) for m in all_metrics]))
            for name in metric_names
        }

        std_metrics = {
            name: float(np.std([float(m[name]) for m in all_metrics]))
            for name in metric_names
        }

        results = {
            "per_image": all_metrics,
            "mean": mean_metrics,
            "std": std_metrics,
            "n_images": len(all_metrics),
        }

        return results

    def save_results(self, results: Dict[str, Any], output_path: Path):
        """
        Save evaluation results to JSON file.

        Args:
            results: Results dictionary from evaluate_dataset()
            output_path: Path to save results
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Results saved to {output_path}")

    def print_summary(self, results: Dict[str, Any]):
        """
        Print evaluation summary.

        Args:
            results: Results dictionary from evaluate_dataset()
        """
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)

        print(f"\n📊 Evaluated {results['n_images']} images")

        print("\n📈 Mean Metrics:")
        for name, value in results["mean"].items():
            std_value = results["std"][name]
            if name == "psnr":
                print(f"   {name.upper()}: {value:.2f} ± {std_value:.2f} dB")
            elif name == "ssim":
                print(f"   {name.upper()}: {value:.4f} ± {std_value:.4f}")
            else:
                print(f"   {name.upper()}: {value:.6f} ± {std_value:.6f}")

        # Show best and worst images
        if results["per_image"]:
            print("\n🏆 Best Image (highest PSNR):")
            best_img = max(results["per_image"], key=lambda x: x["psnr"])
            print(f"   {best_img['filename']}")
            print(f"   PSNR: {best_img['psnr']:.2f} dB, SSIM: {best_img['ssim']:.4f}")

            print("\n⚠️  Worst Image (lowest PSNR):")
            worst_img = min(results["per_image"], key=lambda x: x["psnr"])
            print(f"   {worst_img['filename']}")
            print(f"   PSNR: {worst_img['psnr']:.2f} dB, SSIM: {worst_img['ssim']:.4f}")

        print("\n" + "=" * 60)


if __name__ == "__main__":
    print("Evaluator module - use in notebooks or scripts for full evaluation")
