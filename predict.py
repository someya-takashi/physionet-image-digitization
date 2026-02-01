#!/usr/bin/env python
"""Prediction script for PhysioNet Stage2 Segmentation.

Usage:
    python predict.py --weight outputs/whole_fold0/best.pth --fold 0
    python predict.py --weight outputs/series_fold0/best.pth --fold 0
"""
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from scipy import signal as scipy_signal
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.factory import create_model
from src.data.dataset_whole import create_whole_dataloader
from src.data.dataset_series import create_series_dataloader
from src.utils.checkpoint import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Predict with PhysioNet Stage2 Model")
    parser.add_argument(
        "--weight",
        type=str,
        required=True,
        help="Path to model weight file",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Fold to use for prediction",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for prediction",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size (overrides config)",
    )
    return parser.parse_args()


def read_truth_series(sample_id: str, ecg_dir: str) -> Optional[np.ndarray]:
    """Read ground truth series from CSV file."""
    csv_path = Path(ecg_dir) / f"{sample_id}.csv"
    if not csv_path.exists():
        return None

    try:
        truth_df = pd.read_csv(csv_path)

        # Create II-rhythm column (copy of II)
        truth_df["II-rhythm"] = truth_df["II"]

        # Set II to NaN where I is NaN
        truth_df.loc[truth_df["I"].isna(), "II"] = np.nan

        # Fill NaN with 0
        truth_df.fillna(0, inplace=True)

        # Combine leads into 4 series
        series0 = (
            truth_df["I"] + truth_df["aVR"] + truth_df["V1"] + truth_df["V4"]
        ).values
        series1 = (
            truth_df["II"] + truth_df["aVL"] + truth_df["V2"] + truth_df["V5"]
        ).values
        series2 = (
            truth_df["III"] + truth_df["aVF"] + truth_df["V3"] + truth_df["V6"]
        ).values
        series3 = truth_df["II-rhythm"].values

        series = np.stack([series0, series1, series2, series3])
        return series
    except Exception:
        return None


def pixel_to_series(
    pixel: np.ndarray, zero_mv: List[float], length: int
) -> np.ndarray:
    """Convert pixel probability map to signal series."""
    _, H, W = pixel.shape

    series = []
    for j in range(4):
        p = pixel[j]

        # Find max (upper envelope)
        amax = p.argmax(0)
        amin = H - 1 - p[::-1].argmax(0)
        mask = amax >= zero_mv[j]

        s = mask * amax + (1 - mask) * amin

        # Find missing
        miss = (p > 0.1).sum(0) == 0
        s[miss] = zero_mv[j]
        series.append(s)

    series = np.stack(series).astype(np.float32)

    if length != W:
        resampled_series = []
        for s in series:
            rs = scipy_signal.resample(s, length).astype(np.float32)
            resampled_series.append(rs)
        series = np.stack(resampled_series)

    return series


def calculate_snr(predict: np.ndarray, truth: np.ndarray) -> float:
    """Calculate SNR ratio (not in dB)."""
    eps = 1e-7
    signal_power = (truth**2).sum()
    noise_power = ((predict - truth) ** 2).sum()
    snr_ratio = signal_power / (noise_power + eps)
    return snr_ratio


def create_dataloader(config, df, model_type: str, batch_size: int):
    """Create dataloader based on model type."""
    num_workers = config.get("num_workers", 4)
    mask_dir = config.data.get("mask_dir")

    if model_type == "series":
        window_size = config.get("series", {}).get("window_size", 240)
        loader = create_series_dataloader(
            df=df,
            mask_dir=mask_dir,
            batch_size=batch_size,
            window_size=window_size,
            is_train=False,
            num_workers=num_workers,
        )
    else:  # whole
        offset = config.get("whole", {}).get("offset", 416)
        loader = create_whole_dataloader(
            df=df,
            mask_dir=mask_dir,
            batch_size=batch_size,
            offset=offset,
            is_train=False,
            num_workers=num_workers,
        )

    return loader


@torch.no_grad()
def predict_and_evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    config,
    sample_to_sig_len: Dict[str, int],
    device: str = "cuda",
) -> Dict[str, float]:
    """Run prediction and evaluate SNR.

    Args:
        model: Model to use for prediction.
        dataloader: DataLoader for prediction.
        config: Configuration object.
        sample_to_sig_len: Mapping from sample_id to signal length.
        device: Device to use.

    Returns:
        Dictionary with evaluation metrics.
    """
    model.eval()

    ecg_dir = config.data.get("ecg_dir")
    t0 = config.get("t0")
    t1 = config.get("t1")
    mv_to_pixel = config.get("mv_to_pixel")
    zero_mv = config.get("zero_mv")

    total_loss = 0.0
    num_batches = 0
    all_snr_ratios = []

    pbar = tqdm(dataloader, desc="Predicting")

    for batch in pbar:
        with autocast(device_type=device):
            output = model(batch)
            loss = output["pixel_loss"]

        total_loss += loss.item()
        num_batches += 1

        # Calculate SNR for each sample in batch
        pred_masks = output["pixel"].float().cpu().numpy()
        if len(pred_masks.shape) == 5:  # for series model output
            pred_masks = pred_masks.squeeze(2)
        batch_size = pred_masks.shape[0]

        for i in range(batch_size):
            sample_id = batch["id"][i]

            # Get sig_len for this sample
            sig_len = sample_to_sig_len.get(sample_id, 5000)

            # Load ground truth signal
            truth_series = read_truth_series(sample_id, ecg_dir)
            if truth_series is None:
                continue

            # Extract pixel map and crop to signal region
            pixel = pred_masks[i]  # (4, H, W)

            # Crop width to signal region [t0:t1]
            pixel_cropped = pixel[:, :, t0:t1]

            # Convert pixel probabilities to pixel coordinates
            series_in_pixel = pixel_to_series(pixel_cropped, zero_mv, sig_len)

            # Convert pixel coordinates to mV values
            series = (
                np.array(zero_mv).reshape(4, 1) - series_in_pixel
            ) / mv_to_pixel

            # Calculate SNR for each series
            sample_snr_ratios = []
            for j in range(4):
                snr_ratio = calculate_snr(series[j], truth_series[j])
                sample_snr_ratios.append(snr_ratio)

            mean_snr_ratio = np.mean(sample_snr_ratios)
            all_snr_ratios.append(mean_snr_ratio)

        # Update progress bar
        if all_snr_ratios:
            avg_snr_db = 10 * np.log10(np.mean(all_snr_ratios))
            postfix = {
                "loss": f"{total_loss / num_batches:.4f}",
                "snr": f"{avg_snr_db:.2f}dB",
            }
            pbar.set_postfix(postfix)

    metrics = {
        "loss": total_loss / num_batches,
    }

    if all_snr_ratios:
        avg_snr_db = 10 * np.log10(np.mean(all_snr_ratios))
        metrics["snr"] = avg_snr_db
        metrics["num_samples"] = len(all_snr_ratios)

    return metrics


def main():
    args = parse_args()

    # Check device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    # Load checkpoint
    print(f"Loading checkpoint from {args.weight}")
    checkpoint = torch.load(args.weight, map_location=device, weights_only=False)

    if "config" not in checkpoint:
        raise ValueError("Checkpoint does not contain config. Cannot proceed.")

    config = OmegaConf.create(checkpoint["config"])

    print("=" * 60)
    print("Configuration:")
    print(OmegaConf.to_yaml(config))
    print("=" * 60)

    # Create model and load weights
    model = create_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Best score: {checkpoint.get('best_score', 'unknown')}")

    # Load data
    csv_path = config.data.get("csv_path", "train_fold_with_synthesis_ver2_no_dup.csv")
    df = pd.read_csv(csv_path)

    # Filter by fold
    fold = args.fold
    df_pred = df[df["fold"] == fold].reset_index(drop=True)
    print(f"Prediction samples (fold={fold}): {len(df_pred)}")

    if len(df_pred) == 0:
        print(f"No samples found for fold {fold}")
        return

    # Build sample_to_sig_len mapping
    sample_to_sig_len = {}
    for _, row in df.iterrows():
        sample_to_sig_len[str(row["id"])] = row["sig_len"]

    # Create dataloader
    model_type = config.model.get("type", "whole")
    batch_size = args.batch_size or config.training.get("batch_size", 4)
    dataloader = create_dataloader(config, df_pred, model_type, batch_size)

    # Run prediction and evaluation
    metrics = predict_and_evaluate(
        model=model,
        dataloader=dataloader,
        config=config,
        sample_to_sig_len=sample_to_sig_len,
        device=device,
    )

    # Print results
    print("=" * 60)
    print("Results:")
    print(f"  Loss: {metrics['loss']:.4f}")
    if "snr" in metrics:
        print(f"  SNR: {metrics['snr']:.2f} dB")
        print(f"  Samples evaluated: {metrics['num_samples']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
