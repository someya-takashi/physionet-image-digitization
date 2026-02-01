"""Whole Model with CoordConv decoder.

This module provides a Net class that supports:
- Multiple encoders from timm/smp
- CoordConv integration for position-aware decoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import segmentation_models_pytorch as smp


class MyCoordDecoderBlock(nn.Module):
    """CoordConv decoder block.

    Adds coordinate information (x, y) to each decoder block for position-aware decoding.
    """

    def __init__(self, in_channel, skip_channel, out_channel, scale=2):
        super().__init__()
        self.scale = scale
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel + skip_channel + 2, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.attention1 = nn.Identity()
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.attention2 = nn.Identity()

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=self.scale, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)

        b, c, h, w = x.shape
        coordx, coordy = torch.meshgrid(
            torch.linspace(-2, 2, w, dtype=x.dtype, device=x.device),
            torch.linspace(-2, 2, h, dtype=x.dtype, device=x.device),
            indexing="xy",
        )
        coordxy = torch.stack([coordx, coordy], dim=1).reshape(1, 2, h, w).repeat(b, 1, 1, 1)
        x = torch.cat([x, coordxy], dim=1)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class MyCoordUnetDecoder(nn.Module):
    """CoordConv U-Net decoder.

    Args:
        in_channel: Input channel dimension from encoder bottleneck.
        skip_channel: List of skip connection channels.
        out_channel: List of output channels for each decoder block.
        scale: List of upsampling scales for each block.
        depth: Number of decoder blocks (4 or 5).
    """

    def __init__(self, in_channel, skip_channel, out_channel=None, scale=None, depth=4):
        super().__init__()
        self.center = nn.Identity()
        self.depth = depth

        if out_channel is None:
            if depth == 4:
                out_channel = [256, 128, 64, 32]
            elif depth == 5:
                out_channel = [256, 128, 64, 32, 16]
            else:
                raise ValueError(f"depth must be 4 or 5, got {depth}")

        if scale is None:
            scale = [2] * depth

        if len(out_channel) != depth:
            raise ValueError(f"out_channel length ({len(out_channel)}) must match depth ({depth})")
        if len(scale) != depth:
            raise ValueError(f"scale length ({len(scale)}) must match depth ({depth})")
        if len(skip_channel) != depth:
            raise ValueError(f"skip_channel length ({len(skip_channel)}) must match depth ({depth})")

        i_channel = [in_channel] + out_channel[:-1]
        s_channel = skip_channel
        o_channel = out_channel
        block = [
            MyCoordDecoderBlock(i, s, o, sc)
            for i, s, o, sc in zip(i_channel, s_channel, o_channel, scale)
        ]
        self.block = nn.ModuleList(block)

    def forward(self, feature, skip):
        d = self.center(feature)
        decode = []
        for i, block in enumerate(self.block):
            s = skip[i]
            d = block(d, s)
            decode.append(d)
        last = d
        return last, decode


class Net(nn.Module):
    """Whole Model with CoordConv decoder.

    Args:
        encoder_name: Encoder backbone name.
        encoder_weights: Pretrained weights.
        whole_decoder_depth: Depth for CoordConv decoder (4 or 5).
        loss_weight: Positive class weight for BCE loss.
    """

    def __init__(
        self,
        encoder_name="resnet34",
        encoder_weights="imagenet",
        whole_decoder_depth=4,
        loss_weight=10,
        pretrained=True,
    ):
        super().__init__()

        if encoder_weights is None and pretrained:
            encoder_weights = "imagenet"

        if whole_decoder_depth not in [4, 5]:
            raise ValueError(f"whole_decoder_depth must be 4 or 5")

        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.whole_decoder_depth = whole_decoder_depth
        self.loss_weight = loss_weight

        self.decoder_channels_whole_4 = [256, 128, 64, 32]
        self.decoder_channels_whole_5 = [256, 128, 64, 32, 16]

        self.output_type = ["infer", "loss"]
        self.register_buffer("D", torch.tensor(0))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))

        self._build_whole_model(encoder_name, encoder_weights)

        self.pixel = nn.Conv2d(self.final_decoder_channels + 1, 4, kernel_size=1)

    def _build_whole_model(self, encoder_name, encoder_weights):
        self.encoder = smp.encoders.get_encoder(
            name=encoder_name, in_channels=3, depth=5, weights=encoder_weights
        )

        encoder_channels = self.encoder.out_channels

        if self.whole_decoder_depth == 4:
            encoder_dim = encoder_channels[2:]
            decoder_channels = self.decoder_channels_whole_4
            skip_channel = encoder_dim[:-1][::-1] + [0]
        elif self.whole_decoder_depth == 5:
            encoder_dim = encoder_channels[1:]
            decoder_channels = self.decoder_channels_whole_5
            skip_channel = encoder_dim[:-1][::-1] + [0]
        else:
            raise ValueError(f"whole_decoder_depth must be 4 or 5")

        self.decoder_channels = decoder_channels

        self.decoder = MyCoordUnetDecoder(
            in_channel=encoder_dim[-1],
            skip_channel=skip_channel,
            out_channel=None,
            scale=None,
            depth=self.whole_decoder_depth,
        )

        self.final_decoder_channels = decoder_channels[-1]

    def forward(self, batch, L=None):
        device = self.D.device

        image = batch["image"].to(device)
        B, _3_, H, W = image.shape
        x = image.float() / 255
        x = (x - self.mean) / self.std

        features = self.encoder(x)

        if self.whole_decoder_depth == 4:
            features_for_decoder = features[2:]
            skip_features = features_for_decoder[:-1][::-1] + [None]
        elif self.whole_decoder_depth == 5:
            features_for_decoder = features[1:]
            skip_features = features_for_decoder[:-1][::-1] + [None]
        else:
            raise ValueError(f"whole_decoder_depth must be 4 or 5")

        last, decode = self.decoder(feature=features_for_decoder[-1], skip=skip_features)
        decoder_output = last

        _, _, H_out, W_out = decoder_output.shape
        coordy = torch.arange(H_out, device=device).reshape(1, 1, H_out, 1).repeat(B, 1, 1, W_out)
        coordy = coordy / (H_out - 1) * 2 - 1

        last = torch.cat([decoder_output, coordy], dim=1)
        pixel = self.pixel(last)

        if pixel.shape[2:] != (H, W):
            pixel = F.interpolate(pixel, size=(H, W), mode="bilinear", align_corners=False)

        output = {}

        if "loss" in self.output_type:
            output["pixel_loss"] = F.binary_cross_entropy_with_logits(
                pixel,
                batch["pixel"].to(device),
                pos_weight=torch.tensor([self.loss_weight]).to(device),
            )

        if "infer" in self.output_type:
            output["pixel"] = torch.sigmoid(pixel)

        return output
