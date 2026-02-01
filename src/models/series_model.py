"""Series Model with Cross-Series Feature Fusion.

This module provides a segmentation model that processes 4 separate series images
(cropped around zero_mv baselines) with cross-series feature fusion to learn
inter-series relationships.

Key features:
- Input: (B, 4, 3, H, W) - 4 series images
- Output: (B, 4, 1, H, W) - Single-channel mask per series
- Shared encoder/decoder across all series
- Multiple fusion strategies: none, conv2d, shared_conv2d, conv3d
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import segmentation_models_pytorch as smp


class Conv3dBlock(nn.Module):
    """3D Convolution block for cross-series feature fusion."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3, 3, 3),
        padding=(0, 1, 1),
        stride=(1, 1, 1),
        padding_mode="replicate",
    ):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
            bias=False,
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.activation = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class CrossSeriesFusion(nn.Module):
    """Fuses features across 4 series with residual connection.

    Fusion strategies:
    - conv2d: Per-series channel reduction + concat + mix
    - shared_conv2d: Shared channel reduction (parameter efficient)
    - conv3d: 3D convolution along series dimension
    """

    def __init__(
        self,
        channels,
        num_series=4,
        fusion_type="conv2d",
        reduction_ratio=4,
        conv3d_depth=2,
    ):
        super().__init__()
        self.channels = channels
        self.num_series = num_series
        self.fusion_type = fusion_type

        valid_types = ["conv2d", "shared_conv2d", "conv3d"]
        if fusion_type not in valid_types:
            raise ValueError(f"fusion_type must be one of {valid_types}")

        if fusion_type in ["conv2d", "shared_conv2d"]:
            self._build_conv2d_fusion(
                channels, num_series, reduction_ratio, shared=(fusion_type == "shared_conv2d")
            )
        elif fusion_type == "conv3d":
            self._build_conv3d_fusion(channels, num_series, conv3d_depth)

    def _build_conv2d_fusion(self, channels, num_series, reduction_ratio, shared):
        self.reduced_channels = channels // reduction_ratio

        if shared:
            self.reduce_conv = nn.Sequential(
                nn.Conv2d(channels, self.reduced_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.reduced_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.reduce_convs = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(channels, self.reduced_channels, kernel_size=1, bias=False),
                        nn.BatchNorm2d(self.reduced_channels),
                        nn.ReLU(inplace=True),
                    )
                    for _ in range(num_series)
                ]
            )

        self.mix_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def _build_conv3d_fusion(self, channels, num_series, depth):
        self.conv3d_blocks = nn.ModuleList()
        num_reduction_stages = int(np.log2(num_series))

        if depth < num_reduction_stages:
            raise ValueError(
                f"conv3d_depth={depth} is too small for {num_series} series."
            )

        for i in range(depth):
            if i < num_reduction_stages:
                stride = (2, 1, 1)
                padding = (0, 1, 1)
            else:
                stride = (1, 1, 1)
                padding = (1, 1, 1)

            self.conv3d_blocks.append(
                Conv3dBlock(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=(3, 3, 3),
                    padding=padding,
                    stride=stride,
                    padding_mode="replicate",
                )
            )

    def forward(self, x, batch_size):
        if self.fusion_type in ["conv2d", "shared_conv2d"]:
            return self._forward_conv2d(x, batch_size)
        elif self.fusion_type == "conv3d":
            return self._forward_conv3d(x, batch_size)

    def _forward_conv2d(self, x, batch_size):
        B = batch_size
        C, H, W = x.shape[1:]

        x_series = x.view(B, self.num_series, C, H, W)

        if self.fusion_type == "shared_conv2d":
            reduced = [self.reduce_conv(x_series[:, i]) for i in range(self.num_series)]
        else:
            reduced = [self.reduce_convs[i](x_series[:, i]) for i in range(self.num_series)]

        concat = torch.cat(reduced, dim=1)
        mixed = self.mix_conv(concat)
        mixed_broadcast = mixed.unsqueeze(1).expand(B, self.num_series, C, H, W)
        fused = x_series + mixed_broadcast

        return fused.view(B * self.num_series, C, H, W)

    def _forward_conv3d(self, x, batch_size):
        B = batch_size
        C, H, W = x.shape[1:]

        x_series = x.view(B, self.num_series, C, H, W)
        x_3d = x_series.transpose(1, 2)

        fused = x_3d
        for block in self.conv3d_blocks:
            series_dim = fused.shape[2]
            if series_dim == 1:
                break
            fused = block(fused)

        if fused.shape[2] != 1:
            raise RuntimeError(f"Expected series dimension to be 1, got {fused.shape[2]}")

        fused = fused.squeeze(2)
        fused_broadcast = fused.unsqueeze(1).expand(B, self.num_series, C, H, W)
        output = x_series + fused_broadcast

        return output.view(B * self.num_series, C, H, W)


class Net(nn.Module):
    """Series Model with Cross-Series Feature Fusion.

    Args:
        encoder_name: Encoder backbone name.
        encoder_weights: Pretrained weights.
        fusion_type: Fusion strategy ('none', 'conv2d', 'shared_conv2d', 'conv3d').
        fusion_levels: Which encoder levels to fuse [1, 2, 3, 4].
        num_series: Number of series (default: 4).
        loss_weight: Positive class weight for BCE loss.
        conv3d_depth: Number of Conv3dBlocks for conv3d fusion.
    """

    def __init__(
        self,
        encoder_name="resnet34",
        encoder_weights="imagenet",
        fusion_type="conv2d",
        fusion_levels=None,
        num_series=4,
        loss_weight=10,
        conv3d_depth=2,
    ):
        super().__init__()

        if fusion_levels is None:
            fusion_levels = [1, 2, 3, 4]

        if fusion_type is None:
            fusion_type = "none"
        fusion_type = fusion_type.lower()

        valid_types = ["none", "conv2d", "shared_conv2d", "conv3d"]
        if fusion_type not in valid_types:
            raise ValueError(f"fusion_type must be one of {valid_types}")

        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.fusion_type = fusion_type
        self.fusion_levels = fusion_levels
        self.num_series = num_series
        self.loss_weight = loss_weight
        self.conv3d_depth = conv3d_depth

        self.output_type = ["infer", "loss"]

        self.register_buffer("D", torch.tensor(0))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))

        self._build_model(encoder_name, encoder_weights)

        self.dice_loss_fn = smp.losses.DiceLoss(mode="binary", from_logits=True)

    def _build_model(self, encoder_name, encoder_weights):
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
            decoder_channels=[256, 128, 64, 32, 16],
        )

        self.encoder = model.encoder
        self.decoder = model.decoder

        self.fusion_modules = nn.ModuleDict()

        if self.fusion_type != "none":
            encoder_channels = self.encoder.out_channels

            for level in self.fusion_levels:
                channels = encoder_channels[level + 1]

                self.fusion_modules[f"fusion_{level}"] = CrossSeriesFusion(
                    channels=channels,
                    num_series=self.num_series,
                    fusion_type=self.fusion_type,
                    conv3d_depth=self.conv3d_depth,
                )

        self.pixel_head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, batch, L=None):
        device = self.D.device

        image = batch["image"].to(device)
        B, num_series, C, H, W = image.shape

        x = image.float() / 255
        x = (x - self.mean) / self.std
        x = x.view(B * num_series, C, H, W)

        features = self.encoder(x)

        if self.fusion_type != "none":
            fused_features = list(features)
            for level in self.fusion_levels:
                fusion_key = f"fusion_{level}"
                if fusion_key in self.fusion_modules:
                    fused_features[level + 1] = self.fusion_modules[fusion_key](
                        features[level + 1], batch_size=B
                    )
            features = fused_features

        decoder_output = self.decoder(features)
        pixel = self.pixel_head(decoder_output)
        pixel = pixel.view(B, num_series, 1, H, W)

        output = {}

        if "loss" in self.output_type or "dice_loss" in self.output_type:
            pixel_flat = pixel.view(B * num_series, 1, H, W)
            target = batch["pixel"].to(device)
            target_flat = target.view(B * num_series, 1, H, W)

        if "loss" in self.output_type:
            output["pixel_loss"] = F.binary_cross_entropy_with_logits(
                pixel_flat,
                target_flat,
                pos_weight=torch.tensor([self.loss_weight]).to(device),
            )

        if "dice_loss" in self.output_type:
            output["pixel_dice_loss"] = self.dice_loss_fn(pixel_flat, target_flat)

        if "infer" in self.output_type:
            output["pixel"] = torch.sigmoid(pixel)

        return output
