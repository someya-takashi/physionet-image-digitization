"""Trainer class for model training."""
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
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
            with autocast(enabled=self.amp):
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
    ) -> Dict[str, float]:
        """Validate model.

        Args:
            val_loader: Validation data loader.
            epoch: Current epoch number.

        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")

        for batch in pbar:
            with autocast(enabled=self.amp):
                output = self.model(batch)
                loss = output["pixel_loss"]

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({"loss": total_loss / num_batches})

        metrics = {
            "loss": total_loss / num_batches,
        }

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
        higher_is_better: bool = False,
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
            metric_for_best: Metric to use for best model selection.
            higher_is_better: Whether higher metric is better.
            config: Configuration to save with checkpoint.

        Returns:
            Dictionary with best metrics.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        best_metric = float("-inf") if higher_is_better else float("inf")

        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)

            # Validate
            val_metrics = self.validate(val_loader, epoch)

            # Check if best
            current_metric = val_metrics[metric_for_best]
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

            print(
                f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f}, "
                f"val_loss={val_metrics['loss']:.4f}, best={best_metric:.4f}"
            )

        return {"best_" + metric_for_best: best_metric}
