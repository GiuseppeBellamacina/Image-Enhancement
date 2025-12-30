"""
Download DIV2K dataset for image enhancement training.
"""

import zipfile
from pathlib import Path
import httpx
from tqdm import tqdm


# Dataset URLs
DATASETS = {
    "DIV2K_train_HR": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
    "DIV2K_valid_HR": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
}


def download_file(url: str, destination: Path) -> None:
    """
    Download a file with progress bar.

    Args:
        url: URL to download from
        destination: Path where to save the file
    """
    print(f"\n📥 Downloading: {url}")
    print(f"   Saving to: {destination}")

    with httpx.stream("GET", url, follow_redirects=True, timeout=None) as response:
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with open(destination, "wb") as file:
            with tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=destination.name,
            ) as pbar:
                for chunk in response.iter_bytes(chunk_size=8192):
                    file.write(chunk)
                    pbar.update(len(chunk))

    print(f"✅ Downloaded: {destination.name}")


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """
    Extract a zip file and ensure images are in the root directory.

    Args:
        zip_path: Path to the zip file
        extract_to: Directory where to extract files
    """
    print(f"\n📦 Extracting: {zip_path.name}")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        # Get list of files in zip
        file_list = zip_ref.namelist()

        # Check if there's a top-level directory
        top_dirs = set()
        for filename in file_list:
            parts = Path(filename).parts
            if len(parts) > 1:
                top_dirs.add(parts[0])

        # Extract all files
        zip_ref.extractall(extract_to)

        # If there's only one top-level directory and it matches the expected name,
        # move files up one level
        if len(top_dirs) == 1:
            top_dir = list(top_dirs)[0]
            nested_dir = extract_to / top_dir

            if nested_dir.exists() and nested_dir.is_dir():
                # Move all files from nested directory to parent
                print(f"   Moving files from {top_dir}/ to root...")

                for item in nested_dir.iterdir():
                    target = extract_to / item.name
                    if target.exists():
                        target.unlink()
                    item.rename(target)

                # Remove empty nested directory
                nested_dir.rmdir()

    print(f"✅ Extracted to: {extract_to}")


def download_div2k_dataset(raw_data_dir: Path | None = None) -> dict[str, Path]:
    """
    Download DIV2K training and validation datasets.

    Args:
        raw_data_dir: Directory where to save raw datasets.
                     If None, uses project_root/data/raw

    Returns:
        Dictionary with paths to train and validation directories
    """
    # Setup directories
    if raw_data_dir is None:
        from .paths import get_raw_data_dir

        raw_data_dir = get_raw_data_dir()

    downloads_dir = raw_data_dir / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("📊 Downloading DIV2K Dataset")
    print("=" * 80)

    paths = {}

    # Download and extract each dataset
    for dataset_name, url in DATASETS.items():
        zip_file = downloads_dir / f"{dataset_name}.zip"
        extract_dir = raw_data_dir / dataset_name

        # Check if already exists
        if extract_dir.exists() and any(extract_dir.glob("*.png")):
            n_images = len(list(extract_dir.glob("*.png")))
            print(f"\n✅ {dataset_name} already exists ({n_images} images)")
            print(f"   Location: {extract_dir}")
            paths[dataset_name] = extract_dir
            continue

        # Download
        if not zip_file.exists():
            download_file(url, zip_file)
        else:
            print(f"\n✅ Already downloaded: {zip_file.name}")

        # Extract
        extract_dir.mkdir(parents=True, exist_ok=True)
        extract_zip(zip_file, extract_dir)

        # Verify extraction
        n_images = len(list(extract_dir.glob("*.png")))
        print(f"   📸 Images extracted: {n_images}")

        paths[dataset_name] = extract_dir

    print("\n" + "=" * 80)
    print("✅ Dataset download complete!")
    print("=" * 80)
    print(f"\n📁 Data location: {raw_data_dir.absolute()}")

    return paths
