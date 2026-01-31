"""Lead Model Dataset - processes 4 separate lead images."""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .mask_utils import load_coo_mask
from .transforms import apply_transform, get_train_transforms, get_val_transforms


class LeadDataset(Dataset):
    """Dataset for Lead Model training.

    Loads rectified ECG images and crops around each lead's zero_mv baseline.
    Returns 4 separate lead images per sample.

    Args:
        df: DataFrame with image paths and metadata.
        mask_dir: Directory containing mask files.
        window_size: Half-height of crop window (total height = 2 * window_size).
        zero_mv_positions: Y positions of zero_mv baselines for each lead.
        transform: Albumentations transform for augmentation.
        is_train: Whether this is training data (enables augmentation).
    """

    # Default zero_mv positions for 4 leads (typical ECG layout)
    DEFAULT_ZERO_MV = [150, 450, 750, 1050]

    def __init__(
        self,
        df: pd.DataFrame,
        mask_dir: Union[str, Path],
        window_size: int = 240,
        zero_mv_positions: Optional[List[int]] = None,
        transform=None,
        is_train: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.mask_dir = Path(mask_dir)
        self.window_size = window_size
        self.zero_mv = zero_mv_positions or self.DEFAULT_ZERO_MV
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

        # Crop around each lead
        lead_images = []
        lead_masks = []

        for lead_idx, y_center in enumerate(self.zero_mv):
            y_start = max(0, y_center - self.window_size)
            y_end = min(H, y_center + self.window_size)

            # Crop image
            lead_img = image[y_start:y_end, :, :]

            # Crop mask for this lead
            lead_msk = mask[lead_idx, y_start:y_end, :]

            # Apply transform
            if self.transform is not None:
                lead_img, lead_msk = apply_transform(lead_img, lead_msk, self.transform)

            lead_images.append(lead_img)
            lead_masks.append(lead_msk)

        # Stack leads: (4, H_crop, W, C) -> (4, C, H_crop, W)
        lead_images = np.stack(lead_images, axis=0)  # (4, H, W, C)
        lead_images = lead_images.transpose(0, 3, 1, 2)  # (4, C, H, W)

        # Stack masks: (4, H_crop, W)
        lead_masks = np.stack(lead_masks, axis=0)  # (4, H, W)
        lead_masks = lead_masks[:, np.newaxis, :, :]  # (4, 1, H, W)

        return {
            "image": torch.from_numpy(lead_images).byte(),
            "pixel": torch.from_numpy(lead_masks).float(),
            "id": sample_id,
            "type_id": type_id,
        }


def create_lead_dataloader(
    df: pd.DataFrame,
    mask_dir: Union[str, Path],
    batch_size: int = 4,
    window_size: int = 240,
    is_train: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> torch.utils.data.DataLoader:
    """Create DataLoader for Lead Model.

    Args:
        df: DataFrame with image paths.
        mask_dir: Directory containing mask files.
        batch_size: Batch size.
        window_size: Half-height of crop window.
        is_train: Whether this is training data.
        num_workers: Number of data loading workers.
        pin_memory: Whether to pin memory for faster GPU transfer.

    Returns:
        DataLoader instance.
    """
    dataset = LeadDataset(
        df=df,
        mask_dir=mask_dir,
        window_size=window_size,
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
