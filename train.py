#!/usr/bin/env python
"""Main training script for PhysioNet Stage2 Segmentation.

Usage:
    # Series Model
    python train.py --config configs/series_model.yaml

    # Whole Model
    python train.py --config configs/whole_model.yaml

    # With parameter overrides
    python train.py --config configs/series_model.yaml \
        training.batch_size=8 \
        cv.val_fold=1
"""
import argparse
from pathlib import Path

import pandas as pd
import torch
from omegaconf import OmegaConf

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.wandb_utils import init_wandb, finish_wandb
from src.models.factory import create_model
from src.data.dataset_series import create_series_dataloader
from src.data.dataset_whole import create_whole_dataloader
from src.training.trainer import Trainer
from src.training.scheduler import get_scheduler


def parse_args():
    parser = argparse.ArgumentParser(description="Train PhysioNet Stage2 Model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for checkpoints and logs (overrides config)",
    )
    # Allow arbitrary overrides
    args, unknown = parser.parse_known_args()

    # Parse overrides (key=value format)
    overrides = []
    for arg in unknown:
        if "=" in arg:
            overrides.append(arg)
        else:
            print(f"Warning: Ignoring unknown argument: {arg}")

    return args, overrides


def create_dataloaders(config, df_train, df_val, df_synthesis):
    """Create train and validation dataloaders based on model type."""
    model_type = config.model.get("type", "whole")
    batch_size = config.training.get("batch_size", 4)
    num_workers = config.get("num_workers", 4)
    mask_dir = config.data.get("mask_dir")

    if model_type == "series":
        window_size = config.get("series", {}).get("window_size", 240)
        train_loader = create_series_dataloader(
            df=df_train,
            df_synthesis=df_synthesis,
            mask_dir=mask_dir,
            batch_size=batch_size,
            window_size=window_size,
            is_train=True,
            num_workers=num_workers,
        )
        val_loader = create_series_dataloader(
            df=df_val,
            df_synthesis=df_synthesis,
            mask_dir=mask_dir,
            batch_size=batch_size,
            window_size=window_size,
            is_train=False,
            num_workers=num_workers,
        )
    else:  # whole
        offset = config.get("whole", {}).get("offset", 416)
        train_loader = create_whole_dataloader(
            df=df_train,
            df_synthesis=df_synthesis,
            mask_dir=mask_dir,
            batch_size=batch_size,
            offset=offset,
            is_train=True,
            num_workers=num_workers,
        )
        val_loader = create_whole_dataloader(
            df=df_val,
            df_synthesis=df_synthesis,
            mask_dir=mask_dir,
            batch_size=batch_size,
            offset=offset,
            is_train=False,
            num_workers=num_workers,
        )

    return train_loader, val_loader


def main():
    args, overrides = parse_args()

    # Load config
    config = load_config(args.config, overrides)
    print("=" * 60)
    print("Configuration:")
    print(OmegaConf.to_yaml(config))
    print("=" * 60)

    # Set seed
    set_seed(config.get("seed", 42))

    # Set device
    device = config.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    # Load data
    csv_path = config.data.get("csv_path")
    df = pd.read_csv(csv_path, dtype={"id": str, "type_id": str})

    # Split by fold
    val_fold = config.cv.get("val_fold", 0)
    competition_df = df[~df["is_synthesis"]]
    df_synthesis = df[df["is_synthesis"]]
    df_train = competition_df[competition_df["fold"] != val_fold].reset_index(drop=True)
    df_val = competition_df[competition_df["fold"] == val_fold].reset_index(drop=True)

    # Handle synthesis data
    if len(df_synthesis) and config.data.num_synthesis_data:
        df_synthesis_sample = df_synthesis.sample(config.data.num_synthesis_data)
        df_train = pd.concat([df_train, df_synthesis_sample], ignore_index=True)

    print(f"Train samples: {len(df_train)}, Val samples: {len(df_val)}")

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config, df_train, df_val, df_synthesis)

    # Create model
    model = create_model(config)
    model = model.to(device)
    if config.model.enable_checkpointing:
        model.encoder.model.set_grad_checkpointing()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create optimizer
    optimizer_cfg = config.get("optimizer", {})
    optimizer_name = optimizer_cfg.get("name", "AdamW")
    lr = optimizer_cfg.get("lr", 1e-3)
    weight_decay = optimizer_cfg.get("weight_decay", 1e-2)

    if optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    elif optimizer_name == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    elif optimizer_name == "SGD":
        momentum = optimizer_cfg.get("momentum", 0.9)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Create scheduler
    scheduler_cfg = config.get("scheduler", {})
    scheduler_name = scheduler_cfg.get("name", "CosineAnnealingLR")
    num_epochs = config.training.get("num_epochs", 50)

    scheduler = get_scheduler(
        optimizer=optimizer,
        scheduler_name=scheduler_name,
        num_epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        **{k: v for k, v in scheduler_cfg.items() if k != "name"},
    )

    # Initialize W&B
    run = init_wandb(config)

    # Build sample_to_sig_len mapping for SNR calculation
    sample_to_sig_len = {}
    for _, row in df.iterrows():
        sample_to_sig_len[str(row["id"])] = row["sig_len"]

    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        amp=config.training.get("amp", True),
        gradient_accumulation_steps=config.training.get("gradient_accumulation_steps", 1),
        config=config,
        sample_to_sig_len=sample_to_sig_len,
    )

    # Setup output directory
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(config.get("output_dir", "outputs/default"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    OmegaConf.save(config, output_dir / "config.yaml")

    # Train
    metric_for_best = config.training.get("metric_for_best", "loss")
    try:
        results = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            save_dir=output_dir,
            metric_for_best=metric_for_best,
            config=OmegaConf.to_container(config, resolve=True),
        )
        best_key = f"best_{metric_for_best}"
        print(f"Training completed. {best_key}: {results[best_key]:.4f}")
    finally:
        finish_wandb()


if __name__ == "__main__":
    main()
