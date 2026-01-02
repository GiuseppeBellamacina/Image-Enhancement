"""
Generate degraded datasets with various degradation types
Creates corrupted versions of DIV2K train and validation sets
"""

import cv2
from pathlib import Path
from tqdm.auto import tqdm
import argparse
from typing import Literal
from .quantization_dithering import apply_quantization_dithering
from .gaussian_noise import add_gaussian_noise


def generate_degraded_dataset(
    input_dir: Path,
    output_dir: Path,
    degradation_type: Literal["quantization", "gaussian_noise"] = "quantization",
    # Quantization parameters
    bits_per_channel: int = 2,
    dithering_type: Literal[
        "none", "random", "bayer2", "bayer4", "bayer8", "floyd_steinberg"
    ] = "random",
    # Gaussian noise parameters
    noise_sigma: float = 25.0,
    # General parameters
    seed: int = 42,
):
    """
    Generate degraded dataset using various degradation methods.

    Args:
        input_dir: Path to original images
        output_dir: Path to save degraded images
        degradation_type: Type of degradation ('quantization' or 'gaussian_noise')
        bits_per_channel: Bit depth for quantization (used if degradation_type='quantization')
        dithering_type: Type of dithering (used if degradation_type='quantization')
        noise_sigma: Noise standard deviation (used if degradation_type='gaussian_noise')
        seed: Random seed for reproducibility
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all image files
    image_files = sorted(
        list(input_dir.glob("**/*.png")) + list(input_dir.glob("**/*.jpg"))
    )

    # Print degradation info
    print(f"🔍 Found {len(image_files)} images in {input_dir}")
    if degradation_type == "quantization":
        print(
            f"📦 Degradation: Quantization ({dithering_type} dithering @ {bits_per_channel}-bit)"
        )
    elif degradation_type == "gaussian_noise":
        print(f"📦 Degradation: Gaussian Noise (σ={noise_sigma})")
    print(f"💾 Saving to: {output_dir}")

    for img_path in tqdm(image_files, desc="Processing images"):
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠️  Could not load {img_path.name}, skipping")
            continue

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply degradation based on type
        if degradation_type == "quantization":
            degraded = apply_quantization_dithering(
                img_rgb,
                bits_per_channel=bits_per_channel,
                dithering_type=dithering_type,
                seed=seed,
            )
        elif degradation_type == "gaussian_noise":
            degraded = add_gaussian_noise(img_rgb, sigma=noise_sigma, seed=seed)
        else:
            raise ValueError(f"Unknown degradation type: {degradation_type}")

        # Convert back to BGR for saving
        degraded_bgr = cv2.cvtColor(degraded, cv2.COLOR_RGB2BGR)

        # Save
        output_path = output_dir / img_path.name
        cv2.imwrite(str(output_path), degraded_bgr)

    print(f"✅ Done! Processed {len(image_files)} images")


def main():
    parser = argparse.ArgumentParser(
        description="Generate degraded dataset with various degradation types",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quantization with random dithering
  python generate_degraded_dataset.py --input data/raw --output data/degraded --type quantization --bits 2 --dithering random
  
  # Gaussian noise
  python generate_degraded_dataset.py --input data/raw --output data/degraded --type gaussian_noise --sigma 25
        """,
    )

    # Required arguments
    parser.add_argument(
        "--input", type=str, required=True, help="Input directory with original images"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output directory for degraded images"
    )

    # Degradation type
    parser.add_argument(
        "--type",
        type=str,
        default="quantization",
        choices=["quantization", "gaussian_noise"],
        help="Type of degradation (default: quantization)",
    )

    # Quantization parameters
    parser.add_argument(
        "--bits",
        type=int,
        default=2,
        help="Bits per channel for quantization (default: 2)",
    )
    parser.add_argument(
        "--dithering",
        type=str,
        default="random",
        choices=["none", "random", "bayer2", "bayer4", "bayer8", "floyd_steinberg"],
        help="Dithering type for quantization (default: random)",
    )

    # Gaussian noise parameters
    parser.add_argument(
        "--sigma",
        type=float,
        default=25.0,
        help="Noise standard deviation for gaussian_noise (default: 25.0)",
    )

    # General parameters
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    generate_degraded_dataset(
        input_dir=args.input,
        output_dir=args.output,
        degradation_type=args.type,
        bits_per_channel=args.bits,
        dithering_type=args.dithering,
        noise_sigma=args.sigma,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
