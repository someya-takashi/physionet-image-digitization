"""Configuration loading utilities using OmegaConf."""
from pathlib import Path
from typing import List, Optional

from omegaconf import OmegaConf, DictConfig


def load_config(
    config_path: str,
    overrides: Optional[List[str]] = None,
) -> DictConfig:
    """Load configuration from YAML file with optional CLI overrides.

    Args:
        config_path: Path to the YAML config file.
        overrides: List of CLI overrides in "key=value" format.
                  Supports nested keys like "training.batch_size=8".

    Returns:
        Merged configuration as DictConfig.

    Example:
        >>> cfg = load_config("configs/lead_model_b6.yaml", ["training.batch_size=8"])
        >>> print(cfg.training.batch_size)
        8
    """
    config_path = Path(config_path)

    # Load main config
    cfg = OmegaConf.load(config_path)

    # Handle defaults (base config inheritance)
    if "defaults" in cfg:
        base_configs = cfg.pop("defaults")
        merged = OmegaConf.create()

        for base_name in base_configs:
            base_path = config_path.parent / f"{base_name}.yaml"
            if base_path.exists():
                base_cfg = OmegaConf.load(base_path)
                merged = OmegaConf.merge(merged, base_cfg)

        # Merge main config on top of base
        cfg = OmegaConf.merge(merged, cfg)

    # Apply CLI overrides
    if overrides:
        cli_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, cli_cfg)

    return cfg
