"""Convert pixel masks to time series signals."""
import numpy as np
from typing import Tuple, Union


def pixel_to_series(
    mask: np.ndarray,
    threshold: float = 0.5,
    method: str = "weighted_mean",
) -> np.ndarray:
    """Convert pixel mask to time series.

    Args:
        mask: Mask array of shape (H, W) or (C, H, W).
        threshold: Threshold for binary conversion.
        method: Conversion method:
            - 'argmax': Use argmax along height dimension.
            - 'weighted_mean': Weighted average of y positions.
            - 'threshold_mean': Mean of positions above threshold.

    Returns:
        Time series array of shape (W,) or (C, W).
    """
    mask = np.asarray(mask, dtype=np.float32)

    if mask.ndim == 2:
        return _pixel_to_series_single(mask, threshold, method)
    elif mask.ndim == 3:
        C, H, W = mask.shape
        series = np.zeros((C, W), dtype=np.float32)
        for c in range(C):
            series[c] = _pixel_to_series_single(mask[c], threshold, method)
        return series
    else:
        raise ValueError(f"Expected 2D or 3D mask, got shape {mask.shape}")


def _pixel_to_series_single(
    mask: np.ndarray,
    threshold: float,
    method: str,
) -> np.ndarray:
    """Convert single-channel mask to series."""
    H, W = mask.shape
    series = np.zeros(W, dtype=np.float32)

    y_positions = np.arange(H, dtype=np.float32)

    for x in range(W):
        col = mask[:, x]

        if method == "argmax":
            series[x] = np.argmax(col)

        elif method == "weighted_mean":
            col_sum = col.sum()
            if col_sum > 1e-8:
                series[x] = np.sum(col * y_positions) / col_sum
            else:
                series[x] = H / 2

        elif method == "threshold_mean":
            above_thresh = col > threshold
            if above_thresh.any():
                series[x] = y_positions[above_thresh].mean()
            else:
                # Fallback to weighted mean
                col_sum = col.sum()
                if col_sum > 1e-8:
                    series[x] = np.sum(col * y_positions) / col_sum
                else:
                    series[x] = H / 2

        else:
            raise ValueError(f"Unknown method: {method}")

    return series


def series_to_values(
    series: np.ndarray,
    y_offset: float = 0.0,
    y_scale: float = 1.0,
    invert: bool = True,
) -> np.ndarray:
    """Convert pixel positions to actual signal values.

    Args:
        series: Series of y positions (pixel coordinates).
        y_offset: Offset to subtract from positions.
        y_scale: Scale factor (pixels per mV).
        invert: Whether to invert y-axis (image y increases downward).

    Returns:
        Signal values in physical units (e.g., mV).
    """
    series = np.asarray(series, dtype=np.float32)

    # Apply offset
    values = series - y_offset

    # Invert y-axis if needed
    if invert:
        values = -values

    # Apply scale
    values = values / y_scale

    return values


def resample_series(
    series: np.ndarray,
    source_length: int,
    target_length: int,
) -> np.ndarray:
    """Resample series to target length using linear interpolation.

    Args:
        series: Input series of shape (L,) or (C, L).
        source_length: Original signal length.
        target_length: Target signal length.

    Returns:
        Resampled series.
    """
    series = np.asarray(series)

    if series.ndim == 1:
        x_source = np.linspace(0, source_length - 1, len(series))
        x_target = np.linspace(0, source_length - 1, target_length)
        return np.interp(x_target, x_source, series)
    elif series.ndim == 2:
        C, L = series.shape
        result = np.zeros((C, target_length), dtype=series.dtype)
        for c in range(C):
            x_source = np.linspace(0, source_length - 1, L)
            x_target = np.linspace(0, source_length - 1, target_length)
            result[c] = np.interp(x_target, x_source, series[c])
        return result
    else:
        raise ValueError(f"Expected 1D or 2D series, got shape {series.shape}")
