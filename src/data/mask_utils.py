"""Mask loading utilities for COO sparse format."""
from pathlib import Path
from typing import Tuple, Union

import numpy as np
from scipy import sparse

def load_sparse_mask(filepath):
    """Load sparse mask from COO format.

    Args:
        filepath: Input .npz file path

    Returns:
        mask: (4, H, W) float32 mask array
    """
    filepath = Path(filepath)

    # Load compressed data
    data = np.load(filepath)
    shape = tuple(data['shape'])

    # Reconstruct mask
    mask = np.zeros(shape, dtype=np.float32)

    for i in range(shape[0]):
        y = data[f'ch{i}_y']
        x = data[f'ch{i}_x']
        v = data[f'ch{i}_v']

        mask[i, y, x] = v

    return mask