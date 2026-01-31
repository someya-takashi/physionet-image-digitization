"""Weights & Biases integration utilities."""
from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf


def init_wandb(
    config: DictConfig,
    project: Optional[str] = None,
    name: Optional[str] = None,
    tags: Optional[list] = None,
) -> Optional[Any]:
    """Initialize Weights & Biases logging.

    Args:
        config: Configuration object.
        project: W&B project name (overrides config).
        name: Run name (overrides config).
        tags: Tags for the run.

    Returns:
        W&B run object or None if disabled.
    """
    wandb_cfg = config.get("wandb", {})

    if not wandb_cfg.get("enabled", False):
        return None

    try:
        import wandb
    except ImportError:
        print("wandb not installed, skipping W&B logging")
        return None

    project = project or wandb_cfg.get("project", "physionet-stage2")
    name = name or wandb_cfg.get("name", None)

    run = wandb.init(
        project=project,
        name=name,
        config=OmegaConf.to_container(config, resolve=True),
        tags=tags,
    )

    return run


def log_metrics(
    metrics: Dict[str, float],
    step: Optional[int] = None,
    prefix: str = "",
) -> None:
    """Log metrics to W&B if active.

    Args:
        metrics: Dictionary of metric names to values.
        step: Training step (optional).
        prefix: Prefix for metric names (e.g., "train/", "val/").
    """
    try:
        import wandb

        if wandb.run is None:
            return

        logged = {f"{prefix}{k}": v for k, v in metrics.items()}
        wandb.log(logged, step=step)
    except ImportError:
        pass


def finish_wandb() -> None:
    """Finish W&B run if active."""
    try:
        import wandb

        if wandb.run is not None:
            wandb.finish()
    except ImportError:
        pass
