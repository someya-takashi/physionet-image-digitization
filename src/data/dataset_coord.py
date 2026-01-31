"""Coord Model Dataset - processes full images with offset trimming."""
from pathlib import Path
from typing import Dict, Optional, Union

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .mask_utils import load_coo_mask
from .transforms import apply_transform, get_train_transforms, get_val_transforms


class CoordDataset(Dataset):
    """Dataset for Coord Model training.

    Loads rectified ECG images and optionally trims from the top.
    Processes all 4 series as separate channels.

    Args:
        df: DataFrame with image paths and metadata.
        mask_dir: Directory containing mask files.
        offset: Number of pixels to trim from top (default: 416).
        transform: Albumentations transform for augmentation.
        is_train: Whether this is training data (enables augmentation).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        mask_dir: Union[str, Path],
        offset: int = 416,
        transform=None,
        is_train: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.mask_dir = Path(mask_dir)
        self.offset = offset
        self.transform = transform
        self.is_train = is_train

        if transform is None:
            if is_train:
                self.transform = get_train_transforms()
            else:
                self.transform = get_val_transforms()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        # Load image
        image_path = row["image_path"]
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        H, W, C = image.shape

        # Load mask
        sample_id = row["id"]
        type_id = row["type_id"]
        mask_path = self.mask_dir / f"{sample_id}_{type_id}.npz"

        if mask_path.exists():
            mask = load_coo_mask(mask_path)  # (4, H, W)
        else:
            # Create empty mask if not found
            mask = np.zeros((4, H, W), dtype=np.float32)

        # Apply offset (trim from top)
        if self.offset > 0 and self.offset < H:
            image = image[self.offset:, :, :]
            mask = mask[:, self.offset:, :]

        # Apply transform
        if self.transform is not None:
            # For Coord model, mask has shape (4, H, W)
            # Transpose for albumentations: (4, H, W) -> (H, W, 4)
            mask_t = mask.transpose(1, 2, 0)
            image, mask_t = apply_transform(image, mask_t, self.transform)
            # Transpose back: (H, W, 4) -> (4, H, W)
            mask = mask_t.transpose(2, 0, 1)

        # Convert to tensors
        # Image: (H, W, C) -> (C, H, W)
        image = image.transpose(2, 0, 1)

        return {
            "image": torch.from_numpy(image).byte(),
            "pixel": torch.from_numpy(mask).float(),
            "id": sample_id,
            "type_id": type_id,
        }


def create_coord_dataloader(
    df: pd.DataFrame,
    mask_dir: Union[str, Path],
    batch_size: int = 4,
    offset: int = 416,
    is_train: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> torch.utils.data.DataLoader:
    """Create DataLoader for Coord Model.

    Args:
        df: DataFrame with image paths.
        mask_dir: Directory containing mask files.
        batch_size: Batch size.
        offset: Number of pixels to trim from top.
        is_train: Whether this is training data.
        num_workers: Number of data loading workers.
        pin_memory: Whether to pin memory for faster GPU transfer.

    Returns:
        DataLoader instance.
    """
    dataset = CoordDataset(
        df=df,
        mask_dir=mask_dir,
        offset=offset,
        is_train=is_train,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=is_train,
    )

    return loader
