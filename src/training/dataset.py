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
from functools import lru_cache  # <-- NEW


class ImageEnhancementDataset(Dataset):
    """
    Dataset for image enhancement task.
    Loads degraded images and clean targets, extracts random patches.
    """

    @staticmethod
    @lru_cache(maxsize=64)  # <-- NEW: cache per-process (quindi per worker)
    def _read_rgb(path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

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

        valid_files = []
        for deg, clean in zip(self.degraded_files, self.clean_files):
            assert deg.name == clean.name, f"File mismatch: {deg.name} != {clean.name}"

            img = cv2.imread(str(deg), cv2.IMREAD_COLOR)
            if img is None:
                print(f"Warning: Could not read {deg.name}, skipping")
                continue

            h, w = img.shape[:2]
            if h < patch_size or w < patch_size:
                print(
                    f"Warning: {deg.name} ({h}x{w}) is smaller than patch size ({patch_size}x{patch_size}), skipping"
                )
                continue

            valid_files.append((deg, clean))

        self.degraded_files = [f[0] for f in valid_files]
        self.clean_files = [f[1] for f in valid_files]

        assert (
            len(self.degraded_files) > 0
        ), f"No valid images found! All images are smaller than {patch_size}x{patch_size}"

        print(f"Loaded {len(self.degraded_files)} valid images for {mode} set")

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
            self.transform = A.Compose(
                [A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), ToTensorV2()]
            )

    def __len__(self):
        return len(self.degraded_files) * self.patches_per_image

    def extract_random_patch(
        self, img1: np.ndarray, img2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        h, w = img1.shape[:2]
        assert h >= self.patch_size and w >= self.patch_size

        top = np.random.randint(0, h - self.patch_size + 1)
        left = np.random.randint(0, w - self.patch_size + 1)

        # .copy(): evita view su immagine cached e rende l'array contiguo (albumentations spesso è più felice)
        patch1 = img1[top : top + self.patch_size, left : left + self.patch_size].copy()
        patch2 = img2[top : top + self.patch_size, left : left + self.patch_size].copy()

        return patch1, patch2

    def __getitem__(self, idx):
        img_idx = idx // self.patches_per_image

        degraded = self._read_rgb(str(self.degraded_files[img_idx]))  # <-- CHANGED
        clean = self._read_rgb(str(self.clean_files[img_idx]))  # <-- CHANGED

        degraded_patch, clean_patch = self.extract_random_patch(degraded, clean)

        if self.mode == "train":
            transformed = self.transform(image=degraded_patch, image0=clean_patch)
            degraded_patch = transformed["image"]
            clean_patch = transformed["image0"]
        else:
            degraded_patch = self.transform(image=degraded_patch)["image"]
            clean_patch = self.transform(image=clean_patch)["image"]

        return degraded_patch, clean_patch


def _worker_init_fn(worker_id: int):  # <-- NEW
    # Evita che ogni worker usi più thread OpenCV (spesso rallenta tantissimo)
    try:
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass

    # seed numpy per worker (import locale per non aggiungere dipendenze globali)
    try:
        import torch

        seed = torch.initial_seed() % (2**32)
        np.random.seed(seed + worker_id)
    except Exception:
        pass


def get_dataloaders(
    train_degraded_dir: str,
    train_clean_dir: str,
    val_degraded_dir: str,
    val_clean_dir: str,
    batch_size: int = 16,
    patch_size: int = 128,
    patches_per_image: int = 10,
    patches_per_image_val: int | None = None,  # <-- NEW
    num_workers: int = 4,
    prefetch_factor: int = None,
):
    """
    Create train and validation dataloaders.
    """
    if patches_per_image_val is None:
        patches_per_image_val = min(
            8, patches_per_image
        )  # <-- NEW default (val più veloce)

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
        patches_per_image=patches_per_image_val,  # <-- CHANGED
        augment=False,
        mode="val",
    )

    common_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False,
        persistent_workers=True if num_workers > 0 else False,
    )
    if num_workers > 0:
        common_kwargs["worker_init_fn"] = _worker_init_fn
        if prefetch_factor is not None:
            common_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        **common_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        **common_kwargs,
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
