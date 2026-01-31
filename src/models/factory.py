"""Model factory for creating models from config."""
from typing import Union

import torch.nn as nn
from omegaconf import DictConfig

from .lead_model import Net as LeadNet
from .coord_model import Net as CoordNet


def create_model(config: Union[DictConfig, dict]) -> nn.Module:
    """Create model from configuration.

    Args:
        config: Configuration dict or DictConfig with model settings.
                Must contain 'model.type' ('lead' or 'coord').

    Returns:
        Instantiated model.

    Example:
        >>> cfg = load_config("configs/lead_model_b6.yaml")
        >>> model = create_model(cfg)
    """
    model_cfg = config.get("model", config)
    model_type = model_cfg.get("type", "coord")

    if model_type == "lead":
        model = LeadNet(
            encoder_name=model_cfg.get("encoder_name", "resnet34"),
            encoder_weights=model_cfg.get("encoder_weights", "imagenet"),
            fusion_type=model_cfg.get("fusion_type", "conv2d"),
            fusion_levels=model_cfg.get("fusion_levels", [1, 2, 3, 4]),
            num_leads=model_cfg.get("num_leads", 4),
            loss_weight=model_cfg.get("loss_weight", 10),
            conv3d_depth=model_cfg.get("conv3d_depth", 2),
        )
    elif model_type == "coord":
        model = CoordNet(
            encoder_name=model_cfg.get("encoder_name", "resnet34"),
            encoder_weights=model_cfg.get("encoder_weights", "imagenet"),
            decoder_name=model_cfg.get("decoder_name", "unet"),
            use_coord_conv=model_cfg.get("use_coord_conv", False),
            coord_decoder_depth=model_cfg.get("coord_decoder_depth", 4),
            loss_weight=model_cfg.get("loss_weight", 10),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'lead' or 'coord'.")

    return model
