"""Trainer class for model training."""
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import signal as scipy_signal
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.checkpoint import save_checkpoint
from ..utils.wandb_utils import log_metrics


class Trainer:
    """Training loop manager.

    Args:
        model: Model to train.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler (optional).
        device: Device to train on.
        amp: Whether to use automatic mixed precision.
        gradient_accumulation_steps: Number of steps to accumulate gradients.
        max_grad_norm: Maximum gradient norm for clipping (optional).
        snr_config: Configuration for SNR evaluation (optional).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        device: str = "cuda",
        amp: bool = True,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: Optional[float] = None,
        config: Optional[Dict] = None,
        sample_to_sig_len: Optional[Dict] = None,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.amp = amp
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm

        self.scaler = GradScaler() if amp else None

        self.best_score = float("-inf")
        self.current_epoch = 0

        # SNR evaluation config
        self.snr_config = config
        self.ecg_dir = self.snr_config.data.get("ecg_dir")
        self.sample_to_sig_len = sample_to_sig_len

        # Image parameters for SNR calculation
        self.t0 = self.snr_config.get("t0")
        self.t1 = self.snr_config.get("t1")
        self.mv_to_pixel = self.snr_config.get("mv_to_pixel")

        self.zero_mv = self.snr_config.get("zero_mv")

    def _read_truth_series(self, sample_id: str) -> Optional[np.ndarray]:
        """Read ground truth series from CSV file."""
        if self.ecg_dir is None:
            return None

        csv_path = Path(self.ecg_dir) / f"{sample_id}.csv"
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

    def _pixel_to_series(
        self, pixel: np.ndarray, zero_mv: List[float], length: int
    ) -> np.ndarray:
        """Convert pixel probability map to signal series."""
        _, H, W = pixel.shape
        eps=1e-8
        y_idx = np.arange(H, dtype=np.float32)[:, None]  # (H, 1) for broadcasting

        series = []
        for j in [0, 1, 2, 3]:
            p = pixel[j]
            denom = p.sum(axis=0)  # (W,)
            y_exp = (p * y_idx).sum(axis=0) / (denom + eps)  # (W,)
            
            series.append(y_exp)
        series = np.stack(series).astype(np.float32)

        if length != W:
            resampled_series = []
            for s in series:
                rs = scipy_signal.resample(s, length).astype(np.float32)
                resampled_series.append(rs)
            series = np.stack(resampled_series)

        return series

    def _calculate_snr(self, predict: np.ndarray, truth: np.ndarray) -> float:
        """Calculate SNR ratio (not in dB)."""
        eps = 1e-7
        signal_power = (truth**2).sum()
        noise_power = ((predict - truth) ** 2).sum()
        snr_ratio = signal_power / (noise_power + eps)
        return snr_ratio

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            train_loader: Training data loader.
            epoch: Current epoch number.

        Returns:
            Dictionary of training metrics.
        """
        self.model.train()
        self.current_epoch = epoch

        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")

        for batch_idx, batch in enumerate(pbar):
            # Forward pass
            with autocast(device_type=self.device):
                output = self.model(batch)
                loss = output["pixel_loss"]

                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps

            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation step
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.scaler is not None:
                    if self.max_grad_norm is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.max_grad_norm
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    if self.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.max_grad_norm
                        )
                    self.optimizer.step()

                self.optimizer.zero_grad()

            # Accumulate metrics
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({"loss": total_loss / num_batches})
            
            # if batch_idx > 500:
            #     break

        # Step scheduler (if epoch-based)
        if self.scheduler is not None and not isinstance(
            self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
        ):
            self.scheduler.step()

        metrics = {
            "loss": total_loss / num_batches,
            "lr": self.optimizer.param_groups[0]["lr"],
        }

        log_metrics(metrics, step=epoch, prefix="train/")

        return metrics

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        epoch: int,
        compute_snr: bool = True,
    ) -> Dict[str, float]:
        """Validate model.

        Args:
            val_loader: Validation data loader.
            epoch: Current epoch number.
            compute_snr: Whether to compute SNR metrics.

        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        # SNR tracking
        all_snr_ratios = []

        pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")

        for batch in pbar:
            with autocast(device_type=self.device):
                output = self.model(batch)
                loss = output["pixel_loss"]

            total_loss += loss.item()
            num_batches += 1

            # Calculate SNR for each sample in batch
            pred_masks = output["pixel"].float().cpu().numpy()
            if len(pred_masks.shape) == 5: # for series model output
                pred_masks = pred_masks.squeeze(2)
            batch_size = pred_masks.shape[0]

            for i in range(batch_size):
                sample_id = batch["id"][i]

                # Get sig_len for this sample
                sig_len = self.sample_to_sig_len[sample_id]

                # Load ground truth signal
                truth_series = self._read_truth_series(sample_id)
                if truth_series is None:
                    continue

                # Extract pixel map and crop to signal region
                pixel = pred_masks[i]  # (4, H, W)

                # Crop width to signal region [t0:t1]
                pixel_cropped = pixel[:, :, self.t0:self.t1]

                # Convert pixel probabilities to pixel coordinates
                series_in_pixel = self._pixel_to_series(
                    pixel_cropped, self.zero_mv, sig_len
                )

                # Convert pixel coordinates to mV values
                series = (
                    np.array(self.zero_mv).reshape(4, 1) - series_in_pixel
                ) / self.mv_to_pixel

                # Calculate SNR for each series
                sample_snr_ratios = []
                for j in range(4):
                    snr_ratio = self._calculate_snr(series[j], truth_series[j])
                    sample_snr_ratios.append(snr_ratio)

                mean_snr_ratio = np.mean(sample_snr_ratios)
                all_snr_ratios.append(mean_snr_ratio)

            # Update progress bar
            avg_snr_db = 10 * np.log10(np.mean(all_snr_ratios))
            postfix = {
                "loss": f"{total_loss / num_batches:.4f}",
                "snr": f"{avg_snr_db:.2f}dB",
                }
            pbar.set_postfix(postfix)

        metrics = {
            "loss": total_loss / num_batches,
        }

        # Add SNR metric
        if all_snr_ratios:
            avg_snr_db = 10 * np.log10(np.mean(all_snr_ratios))
            metrics["snr"] = avg_snr_db

        # Step ReduceLROnPlateau scheduler
        if self.scheduler is not None and isinstance(
            self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
        ):
            self.scheduler.step(metrics["loss"])

        log_metrics(metrics, step=epoch, prefix="val/")

        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        save_dir: Union[str, Path],
        save_best: bool = True,
        save_last: bool = True,
        metric_for_best: str = "loss",
        higher_is_better: Optional[bool] = None,
        config: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """Full training loop.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            num_epochs: Number of epochs to train.
            save_dir: Directory to save checkpoints.
            save_best: Whether to save best model.
            save_last: Whether to save last model.
            metric_for_best: Metric to use for best model selection ('loss' or 'snr').
            higher_is_better: Whether higher metric is better.
                              If None, automatically set based on metric_for_best.
            config: Configuration to save with checkpoint.

        Returns:
            Dictionary with best metrics.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Auto-determine higher_is_better based on metric
        if higher_is_better is None:
            if metric_for_best == "snr":
                higher_is_better = True
            else:
                higher_is_better = False

        best_metric = float("-inf") if higher_is_better else float("inf")

        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)

            # Validate
            compute_snr = metric_for_best == "snr" or self.ecg_dir is not None
            val_metrics = self.validate(val_loader, epoch, compute_snr=compute_snr)

            # Check if best
            current_metric = val_metrics.get(metric_for_best)
            if current_metric is None:
                print(
                    f"Warning: metric '{metric_for_best}' not found in val_metrics. "
                    f"Available: {list(val_metrics.keys())}"
                )
                current_metric = val_metrics["loss"]

            is_best = (
                current_metric > best_metric
                if higher_is_better
                else current_metric < best_metric
            )

            if is_best:
                best_metric = current_metric
                if save_best:
                    save_checkpoint(
                        path=save_dir / "best.pth",
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        epoch=epoch,
                        best_score=best_metric,
                        config=config,
                    )

            # Save last
            if save_last:
                save_checkpoint(
                    path=save_dir / "last.pth",
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    best_score=best_metric,
                    config=config,
                )

            # Print epoch summary
            summary = (
                f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f}, "
                f"val_loss={val_metrics['loss']:.4f}"
            )
            if "snr" in val_metrics:
                summary += f", val_snr={val_metrics['snr']:.2f}dB"
            summary += f", best_{metric_for_best}={best_metric:.4f}"
            print(summary)

        return {"best_" + metric_for_best: best_metric}
