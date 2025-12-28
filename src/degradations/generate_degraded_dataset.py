"""
Generate degraded dataset with random dithering @ 2-bit
Creates corrupted versions of DIV2K train and validation sets
"""

import cv2
from pathlib import Path
from tqdm import tqdm
import argparse
from typing import Literal
from .quantization_dithering import apply_quantization_dithering


def generate_degraded_dataset(
    input_dir: Path,
    output_dir: Path,
    bits_per_channel: int = 2,
    dithering_type: Literal['none', 'random', 'bayer2', 'bayer4', 'bayer8', 'floyd_steinberg'] = 'random',
    seed: int = 42
):
    """
    Generate degraded dataset using quantization + dithering.
    
    Args:
        input_dir: Path to original images
        output_dir: Path to save degraded images
        bits_per_channel: Bit depth for quantization
        dithering_type: Type of dithering
        seed: Random seed
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = sorted(list(input_dir.glob('*.png')) + list(input_dir.glob('*.jpg')))
    
    print(f"🔍 Found {len(image_files)} images in {input_dir}")
    print(f"📦 Degrading with: {dithering_type} @ {bits_per_channel}-bit")
    print(f"💾 Saving to: {output_dir}")
    
    for img_path in tqdm(image_files, desc="Processing images"):
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠️  Could not load {img_path.name}, skipping")
            continue
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply degradation
        degraded = apply_quantization_dithering(
            img_rgb,
            bits_per_channel=bits_per_channel,
            dithering_type=dithering_type,
            seed=seed
        )
        
        # Convert back to BGR for saving
        degraded_bgr = cv2.cvtColor(degraded, cv2.COLOR_RGB2BGR)
        
        # Save
        output_path = output_dir / img_path.name
        cv2.imwrite(str(output_path), degraded_bgr)
    
    print(f"✅ Done! Processed {len(image_files)} images")


def main():
    parser = argparse.ArgumentParser(description='Generate degraded dataset')
    parser.add_argument('--input', type=str, required=True, help='Input directory with original images')
    parser.add_argument('--output', type=str, required=True, help='Output directory for degraded images')
    parser.add_argument('--bits', type=int, default=2, help='Bits per channel (default: 2)')
    parser.add_argument('--dithering', type=str, default='random', 
                       choices=['none', 'random', 'bayer2', 'bayer4', 'bayer8', 'floyd_steinberg'],
                       help='Dithering type (default: random)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    generate_degraded_dataset(
        input_dir=args.input,
        output_dir=args.output,
        bits_per_channel=args.bits,
        dithering_type=args.dithering,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
