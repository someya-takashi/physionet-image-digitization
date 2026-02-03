"""Series Model Dataset - processes 4 separate series images."""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .mask_utils import load_sparse_mask
from .transforms import apply_transform, get_train_transforms, get_val_transforms


class SeriesDataset(Dataset):
    """Dataset for Series Model training.

    Loads rectified ECG images and crops around each series's zero_mv baseline.
    Returns 4 separate series images per sample.

    Args:
        df: DataFrame with image paths and metadata.
        mask_dir: Directory containing mask files.
        window_size: Half-height of crop window (total height = 2 * window_size).
        zero_mv_positions: Y positions of zero_mv baselines for each series.
        transform: Albumentations transform for augmentation.
        is_train: Whether this is training data (enables augmentation).
    """
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
        self.zero_mv = [703.5, 987.5, 1271.5, 1531.5]
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
        image = cv2.resize(image, (5600, 1700),interpolation=cv2.INTER_LINEAR)
        image = image[:1696, :5600]

        # Load mask
        sample_id = str(row["id"])
        mask_path = self.mask_dir / f"{sample_id}.mask-coo.npz"

        mask = load_sparse_mask(mask_path)  # (4, H, W)

        # Data augmentation
        if self.transform is not None:
            # Transpose for albumentations: (4, H, W) -> (H, W, 4)
            mask_t = mask.transpose(1, 2, 0)
            image, mask_t = apply_transform(image, mask_t, self.transform)
            # Transpose back: (H, W, 4) -> (4, H, W)
            mask = mask_t.transpose(2, 0, 1)

        images = []
        masks = []
        H, W, _ = image.shape
        for i, zmv in enumerate(self.zero_mv):
            # zero mv +- self.window_sizeの範囲をcrop（はみ出た部分は黒埋め）
            h0, h1 = int(zmv) - self.window_size, int(zmv) + self.window_size
            src_h0 = max(0, h0)
            src_h1 = min(H, h1)
            dst_h0 = src_h0 - h0
            dst_h1 = dst_h0 + (src_h1 - src_h0)

            series_img = np.zeros((self.window_size*2, W, 3))
            series_img[dst_h0:dst_h1, :, :] = image[src_h0:src_h1, :, :]
            images.append(series_img)

            # Initialize cropped mask
            series_mask = np.zeros((self.window_size*2, W))

            # Copy labels within crop range
            series_mask[dst_h0:dst_h1, :] = mask[i][src_h0:src_h1, :]

            # Relocate labels above crop range to top edge
            if src_h0 > 0:
                above_range = mask[i][:src_h0, :]  # Labels above crop
                has_label_above = (above_range > 0).any(axis=0)  # Check each column
                series_mask[dst_h0, has_label_above] = 1.0  # Move to top edge

            # Relocate labels below crop range to bottom edge
            if src_h1 < H:
                below_range = mask[i][src_h1:, :]  # Labels below crop
                has_label_below = (below_range > 0).any(axis=0)  # Check each column
                series_mask[dst_h1 - 1, has_label_below] = 1.0  # Move to bottom edge

            masks.append(series_mask)

        images = np.stack(images) # (4, H, W, 3)
        masks = np.stack(masks) # (4, 1, H, W)

        # Convert to torch tensors
        images = torch.from_numpy(images.transpose(0, 3, 1, 2)).contiguous() # (4, 3, H, W)
        masks = torch.from_numpy(masks).contiguous()

        return {
            "image": images,
            "pixel": masks,
            "id": sample_id,
        }


def create_series_dataloader(
    df: pd.DataFrame,
    mask_dir: Union[str, Path],
    batch_size: int = 4,
    window_size: int = 240,
    is_train: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> torch.utils.data.DataLoader:
    """Create DataLoader for Series Model.

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
    dataset = SeriesDataset(
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
