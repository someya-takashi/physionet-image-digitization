"""Mask loading utilities for COO sparse format."""
from pathlib import Path
from typing import Tuple, Union

import numpy as np
from scipy import sparse


def load_coo_mask(
    mask_path: Union[str, Path],
    shape: Tuple[int, int] = None,
) -> np.ndarray:
    """Load mask from COO sparse format (.npz file).

    Args:
        mask_path: Path to .npz file containing sparse mask data.
        shape: Expected shape (H, W). If None, inferred from file.

    Returns:
        Dense mask array of shape (4, H, W) with float32 dtype.

    File format:
        The .npz file should contain:
        - 'data': Non-zero values
        - 'row': Row indices
        - 'col': Column indices
        - 'shape': Original shape
    """
    mask_path = Path(mask_path)

    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    # Load sparse data
    data = np.load(mask_path, allow_pickle=True)

    # Handle different sparse formats
    if "data" in data and "row" in data and "col" in data:
        # COO format
        stored_shape = tuple(data["shape"]) if "shape" in data else shape
        if stored_shape is None:
            raise ValueError("Shape must be provided or stored in file")

        coo = sparse.coo_matrix(
            (data["data"], (data["row"], data["col"])),
            shape=stored_shape,
        )
        mask = coo.toarray()
    elif "arr_0" in data:
        # Compressed dense format (fallback)
        mask = data["arr_0"]
    else:
        # Try to load as compressed sparse matrix
        try:
            mask = sparse.load_npz(mask_path).toarray()
        except Exception:
            raise ValueError(f"Unknown mask format in {mask_path}")

    # Ensure correct dtype and shape
    mask = mask.astype(np.float32)

    # Reshape if needed (should be 4 x H x W for 4 series)
    if mask.ndim == 2:
        # Single channel, expand
        mask = mask[np.newaxis, ...]

    return mask


def save_coo_mask(
    mask: np.ndarray,
    save_path: Union[str, Path],
    compress: bool = True,
) -> None:
    """Save mask in COO sparse format.

    Args:
        mask: Dense mask array of shape (C, H, W) or (H, W).
        save_path: Path to save .npz file.
        compress: Whether to use compression.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Flatten if needed for sparse storage
    if mask.ndim == 3:
        C, H, W = mask.shape
        flat_mask = mask.reshape(C * H, W)
    else:
        flat_mask = mask

    # Convert to COO
    coo = sparse.coo_matrix(flat_mask)

    # Save
    if compress:
        np.savez_compressed(
            save_path,
            data=coo.data,
            row=coo.row,
            col=coo.col,
            shape=mask.shape,
        )
    else:
        np.savez(
            save_path,
            data=coo.data,
            row=coo.row,
            col=coo.col,
            shape=mask.shape,
        )
