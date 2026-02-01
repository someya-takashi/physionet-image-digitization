"""Model factory for creating models from config."""
from typing import Union

import torch.nn as nn
from omegaconf import DictConfig

from .series_model import Net as SeriesNet
from .whole_model import Net as WholeNet


def create_model(config: Union[DictConfig, dict]) -> nn.Module:
    """Create model from configuration.

    Args:
        config: Configuration dict or DictConfig with model settings.
                Must contain 'model.type' ('series' or 'whole').

    Returns:
        Instantiated model.

    Example:
        >>> cfg = load_config("configs/series_model.yaml")
        >>> model = create_model(cfg)
    """
    model_cfg = config.get("model", config)
    model_type = model_cfg.get("type", "whole")

    if model_type == "series":
        model = SeriesNet(
            encoder_name=model_cfg.get("encoder_name", "resnet34"),
            encoder_weights=model_cfg.get("encoder_weights", "imagenet"),
            fusion_type=model_cfg.get("fusion_type", "conv2d"),
            fusion_levels=model_cfg.get("fusion_levels", [1, 2, 3, 4]),
            num_series=model_cfg.get("num_series", 4),
            loss_weight=model_cfg.get("loss_weight", 10),
            conv3d_depth=model_cfg.get("conv3d_depth", 2),
        )
    elif model_type == "whole":
        model = WholeNet(
            encoder_name=model_cfg.get("encoder_name", "resnet34"),
            encoder_weights=model_cfg.get("encoder_weights", "imagenet"),
            whole_decoder_depth=model_cfg.get("whole_decoder_depth", 4),
            loss_weight=model_cfg.get("loss_weight", 10),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'series' or 'whole'.")

    return model
