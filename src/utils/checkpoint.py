"""Checkpoint save/load utilities."""
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    best_score: float = 0.0,
    config: Optional[Dict] = None,
) -> None:
    """Save training checkpoint.

    Args:
        path: Path to save checkpoint.
        model: Model to save.
        optimizer: Optimizer state (optional).
        scheduler: Scheduler state (optional).
        epoch: Current epoch.
        best_score: Best validation score.
        config: Configuration dict (optional).
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "best_score": best_score,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if config is not None:
        checkpoint["config"] = config

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Load training checkpoint.

    Args:
        path: Path to checkpoint file.
        model: Model to load weights into.
        optimizer: Optimizer to load state into (optional).
        scheduler: Scheduler to load state into (optional).
        device: Device to load checkpoint to.

    Returns:
        Dict containing epoch, best_score, and optionally config.
    """
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return {
        "epoch": checkpoint.get("epoch", 0),
        "best_score": checkpoint.get("best_score", 0.0),
        "config": checkpoint.get("config", None),
    }
