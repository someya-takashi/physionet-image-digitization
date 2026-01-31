"""Signal-to-Noise Ratio (SNR) calculation for PhysioNet competition."""
import numpy as np
from typing import Union


def calculate_snr(
    signal: np.ndarray,
    reconstruction: np.ndarray,
    epsilon: float = 1e-8,
) -> float:
    """Calculate Signal-to-Noise Ratio.

    SNR = 10 * log10(sum(signal^2) / sum((signal - reconstruction)^2))

    Args:
        signal: Ground truth signal array.
        reconstruction: Reconstructed signal array.
        epsilon: Small value to avoid log(0).

    Returns:
        SNR value in dB.
    """
    signal = np.asarray(signal, dtype=np.float64)
    reconstruction = np.asarray(reconstruction, dtype=np.float64)

    signal_power = np.sum(signal ** 2)
    noise_power = np.sum((signal - reconstruction) ** 2)

    if noise_power < epsilon:
        return np.inf

    snr = 10 * np.log10((signal_power + epsilon) / (noise_power + epsilon))

    return float(snr)


def snr_score(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    sample_weights: Union[np.ndarray, list, None] = None,
) -> float:
    """Calculate average SNR score across multiple samples.

    Args:
        y_true: List/array of ground truth signals.
        y_pred: List/array of predicted signals.
        sample_weights: Optional weights for each sample.

    Returns:
        Weighted average SNR in dB.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim == 1:
        return calculate_snr(y_true, y_pred)

    n_samples = len(y_true)

    if sample_weights is None:
        sample_weights = np.ones(n_samples)
    else:
        sample_weights = np.asarray(sample_weights)

    snrs = []
    for i in range(n_samples):
        snr = calculate_snr(y_true[i], y_pred[i])
        if not np.isinf(snr):
            snrs.append(snr * sample_weights[i])

    if len(snrs) == 0:
        return 0.0

    return float(np.sum(snrs) / np.sum(sample_weights[:len(snrs)]))
