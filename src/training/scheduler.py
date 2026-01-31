"""Learning rate schedulers."""
from typing import Any, Dict, Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    ReduceLROnPlateau,
    StepLR,
)


def get_scheduler(
    optimizer: Optimizer,
    scheduler_name: str,
    num_epochs: int = 50,
    steps_per_epoch: Optional[int] = None,
    **kwargs,
) -> Any:
    """Get learning rate scheduler by name.

    Args:
        optimizer: PyTorch optimizer.
        scheduler_name: Scheduler name.
        num_epochs: Total number of epochs.
        steps_per_epoch: Steps per epoch (required for OneCycleLR).
        **kwargs: Additional scheduler arguments.

    Returns:
        Learning rate scheduler.
    """
    scheduler_name = scheduler_name.lower()

    if scheduler_name == "cosineannealinglr":
        T_max = kwargs.get("T_max", num_epochs)
        eta_min = kwargs.get("eta_min", 0)
        return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

    elif scheduler_name == "cosineannealingwarmrestarts":
        T_0 = kwargs.get("T_0", 10)
        T_mult = kwargs.get("T_mult", 2)
        eta_min = kwargs.get("eta_min", 0)
        return CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
        )

    elif scheduler_name == "onecyclelr":
        if steps_per_epoch is None:
            raise ValueError("steps_per_epoch is required for OneCycleLR")
        max_lr = kwargs.get("max_lr", optimizer.defaults["lr"])
        pct_start = kwargs.get("pct_start", 0.3)
        return OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=pct_start,
        )

    elif scheduler_name == "reducelronplateau":
        mode = kwargs.get("mode", "min")
        factor = kwargs.get("factor", 0.1)
        patience = kwargs.get("patience", 10)
        return ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience)

    elif scheduler_name == "steplr":
        step_size = kwargs.get("step_size", 30)
        gamma = kwargs.get("gamma", 0.1)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif scheduler_name == "none" or scheduler_name is None:
        return None

    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
