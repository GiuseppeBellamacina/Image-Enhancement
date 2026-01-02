"""
PyTorch Dataset for Image Enhancement
Handles patch extraction, augmentation, and loading
"""

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple


class ImageEnhancementDataset(Dataset):
    """
    Dataset for image enhancement task.
    Loads degraded images and clean targets, extracts random patches.

    Args:
        degraded_dir: Directory with degraded images
        clean_dir: Directory with clean target images
        patch_size: Size of patches to extract (default: 128)
        patches_per_image: Number of patches to extract per image (default: 10)
        augment: Whether to apply data augmentation (default: True)
        mode: 'train' or 'val' for different augmentation strategies
    """

    def __init__(
        self,
        degraded_dir: str,
        clean_dir: str,
        patch_size: int = 128,
        patches_per_image: int = 10,
        augment: bool = True,
        mode: str = "train",
    ):
        self.degraded_dir = Path(degraded_dir)
        self.clean_dir = Path(clean_dir)
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.mode = mode

        # Get all image files
        self.degraded_files = sorted(
            list(self.degraded_dir.glob("*.png"))
            + list(self.degraded_dir.glob("*.jpg"))
        )
        self.clean_files = sorted(
            list(self.clean_dir.glob("*.png")) + list(self.clean_dir.glob("*.jpg"))
        )

        assert len(self.degraded_files) == len(
            self.clean_files
        ), f"Mismatch in number of files: {len(self.degraded_files)} degraded vs {len(self.clean_files)} clean"

        # Verify file correspondence
        for deg, clean in zip(self.degraded_files, self.clean_files):
            assert deg.name == clean.name, f"File mismatch: {deg.name} != {clean.name}"

        # Setup augmentations
        if augment and mode == "train":
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                    ToTensorV2(),
                ],
                additional_targets={"image0": "image"},
            )
        else:
            # Only normalize for validation
            self.transform = A.Compose(
                [A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), ToTensorV2()]
            )

    def __len__(self):
        return len(self.degraded_files) * self.patches_per_image

    def extract_random_patch(
        self, img1: np.ndarray, img2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract random patch from two aligned images"""
        h, w = img1.shape[:2]

        # Random crop coordinates
        top = np.random.randint(0, h - self.patch_size + 1)
        left = np.random.randint(0, w - self.patch_size + 1)

        # Extract patches
        patch1 = img1[top : top + self.patch_size, left : left + self.patch_size]
        patch2 = img2[top : top + self.patch_size, left : left + self.patch_size]

        return patch1, patch2

    def __getitem__(self, idx):
        # Get image index (multiple patches per image)
        img_idx = idx // self.patches_per_image

        # Load images
        degraded = cv2.imread(str(self.degraded_files[img_idx]))
        clean = cv2.imread(str(self.clean_files[img_idx]))

        # Convert BGR to RGB
        degraded = cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB)
        clean = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)

        # Extract random patches
        degraded_patch, clean_patch = self.extract_random_patch(degraded, clean)

        # Apply transformations (same transform to both)
        # Use additional_targets to apply same transform to both images
        if self.mode == "train":
            # For training: apply same random augmentation to both
            transformed = self.transform(image=degraded_patch, image0=clean_patch)
            degraded_patch = transformed["image"]
            clean_patch = transformed["image0"]
        else:
            # For validation: only normalize
            degraded_patch = self.transform(image=degraded_patch)["image"]
            clean_patch = self.transform(image=clean_patch)["image"]

        return degraded_patch, clean_patch


def get_dataloaders(
    train_degraded_dir: str,
    train_clean_dir: str,
    val_degraded_dir: str,
    val_clean_dir: str,
    batch_size: int = 16,
    patch_size: int = 128,
    patches_per_image: int = 10,
    num_workers: int = 4,
):
    """
    Create train and validation dataloaders.

    Returns:
        train_loader, val_loader
    """
    # Create datasets
    train_dataset = ImageEnhancementDataset(
        degraded_dir=train_degraded_dir,
        clean_dir=train_clean_dir,
        patch_size=patch_size,
        patches_per_image=patches_per_image,
        augment=True,
        mode="train",
    )

    val_dataset = ImageEnhancementDataset(
        degraded_dir=val_degraded_dir,
        clean_dir=val_clean_dir,
        patch_size=patch_size,
        patches_per_image=patches_per_image,
        augment=False,
        mode="val",
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False,
        drop_last=True,
        persistent_workers=(
            True if num_workers > 0 else False
        ),  # Avoid worker restart overhead
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False,
        persistent_workers=(
            True if num_workers > 0 else False
        ),  # Avoid worker restart overhead
    )

    print("📊 Dataset Summary:")
    print(
        f"   Train: {len(train_dataset)} patches from {len(train_dataset.degraded_files)} images"
    )
    print(
        f"   Val:   {len(val_dataset)} patches from {len(val_dataset.degraded_files)} images"
    )
    print(f"   Batch size: {batch_size}")
    print(f"   Patch size: {patch_size}x{patch_size}")

    return train_loader, val_loader


if __name__ == "__main__":
    from ..utils.paths import get_degraded_data_dir, get_raw_data_dir

    dataset = ImageEnhancementDataset(
        degraded_dir=str(get_degraded_data_dir() / "DIV2K_train_HR"),
        clean_dir=str(get_raw_data_dir() / "DIV2K_train_HR"),
        patch_size=128,
        patches_per_image=2,
    )

    print(f"Dataset size: {len(dataset)}")

    degraded, clean = dataset[0]
    print(f"Degraded shape: {degraded.shape}")
    print(f"Clean shape: {clean.shape}")
    print(f"Degraded range: [{degraded.min():.3f}, {degraded.max():.3f}]")
    print(f"Clean range: [{clean.min():.3f}, {clean.max():.3f}]")

    print("✅ Dataset test passed!")
