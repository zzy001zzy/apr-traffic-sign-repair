# SPDX-FileCopyrightText: 2026 Chih-Yuan Cheng and Bor-Jiunn Hwang
# SPDX-License-Identifier: MIT
"""Small high-resolution GIVEN_SIGN_ROI locator candidates for V2 remediation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


ROI_SIZE = 128


@dataclass(frozen=True, slots=True)
class ROILocatorOutput:
    mask_logits: Tensor
    presence_logit: Tensor


@dataclass(frozen=True, slots=True)
class ROIArchitecture:
    candidate_id: str
    parameters: int
    trainable_parameters: int
    input_shape: tuple[int, int, int]
    mask_output_shape: tuple[int, int, int]
    explicit_presence_head: bool
    template_conditioned: bool


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        groups = 8 if out_channels % 8 == 0 else 4 if out_channels % 4 == 0 else 1
        self.norm = nn.GroupNorm(groups, out_channels)
        self.activation = nn.SiLU(inplace=False)

    def forward(self, value: Tensor) -> Tensor:
        return self.activation(self.norm(self.conv(value)))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.first = ConvNormAct(channels, channels)
        self.second = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8 if channels % 8 == 0 else 4, channels),
        )
        self.activation = nn.SiLU(inplace=False)

    def forward(self, value: Tensor) -> Tensor:
        return self.activation(value + self.second(self.first(value)))


class DeterministicGlobalMaxPool2d(nn.Module):
    """Exact 1x1 global max without adaptive-max CUDA backward dispatch."""

    def forward(self, value: Tensor) -> Tensor:
        return value.flatten(2).max(dim=2).values[:, :, None, None]


class ROIEncoder(nn.Module):
    def __init__(self, base: int = 16) -> None:
        super().__init__()
        self.stem = nn.Sequential(ConvNormAct(3, base), ResidualBlock(base))
        self.down1 = nn.Sequential(
            ConvNormAct(base, base * 2, stride=2), ResidualBlock(base * 2)
        )
        self.down2 = nn.Sequential(
            ConvNormAct(base * 2, base * 4, stride=2), ResidualBlock(base * 4)
        )

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        high = self.stem(image)
        mid = self.down1(high)
        deep = self.down2(mid)
        return high, mid, deep


class ROIDecoder(nn.Module):
    def __init__(self, *, deep_channels: int, mid_channels: int, high_channels: int) -> None:
        super().__init__()
        self.deep_to_mid = ConvNormAct(deep_channels + mid_channels, mid_channels)
        self.mid_refine = ResidualBlock(mid_channels)
        self.mid_to_high = ConvNormAct(mid_channels + high_channels, high_channels)
        self.high_refine = ResidualBlock(high_channels)
        self.mask_head = nn.Conv2d(high_channels, 1, 1)

    def forward(self, high: Tensor, mid: Tensor, deep: Tensor) -> Tensor:
        up_mid = F.interpolate(deep, size=mid.shape[-2:], mode="bilinear", align_corners=False)
        mid_value = self.mid_refine(self.deep_to_mid(torch.cat((mid, up_mid), dim=1)))
        up_high = F.interpolate(
            mid_value, size=high.shape[-2:], mode="bilinear", align_corners=False
        )
        high_value = self.high_refine(
            self.mid_to_high(torch.cat((high, up_high), dim=1))
        )
        return self.mask_head(high_value)


class ROIUNetSmall(nn.Module):
    candidate_id = "V2_C1_ROI_UNET"
    template_conditioned = False

    def __init__(self) -> None:
        super().__init__()
        self.encoder = ROIEncoder(base=16)
        self.decoder = ROIDecoder(deep_channels=64, mid_channels=32, high_channels=16)
        self.presence_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 1),
        )

    def forward(self, image: Tensor) -> ROILocatorOutput:
        _validate_image(image)
        high, mid, deep = self.encoder(image)
        mask = self.decoder(high, mid, deep)
        presence = self.presence_head(deep)[:, 0]
        _validate_output(mask, presence, image.shape[0])
        return ROILocatorOutput(mask_logits=mask, presence_logit=presence)


class ROITemplateCorrelation(nn.Module):
    candidate_id = "V2_C2_TEMPLATE_CORRELATION"
    template_conditioned = True

    def __init__(self, template_rgb_normalized: Tensor) -> None:
        super().__init__()
        _validate_template(template_rgb_normalized)
        self.register_buffer(
            "template_rgb_normalized",
            template_rgb_normalized.detach().clone().contiguous(),
            persistent=True,
        )
        self.search_stem = nn.Sequential(
            ConvNormAct(3, 20),
            ConvNormAct(20, 28, stride=2),
            ResidualBlock(28),
        )
        self.template_stem = nn.Sequential(
            ConvNormAct(3, 20),
            ConvNormAct(20, 28, stride=2),
            ResidualBlock(28),
        )
        self.context = nn.Sequential(
            ConvNormAct(28, 48, stride=2),
            ResidualBlock(48),
            ResidualBlock(48),
        )
        self.refine = nn.Sequential(
            ConvNormAct(28 + 1 + 48, 48),
            ResidualBlock(48),
            ConvNormAct(48, 28),
        )
        self.mask_head = nn.Sequential(
            ConvNormAct(28, 20),
            nn.Conv2d(20, 1, 1),
        )
        self.presence_head = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Linear(48, 1),
        )

    def forward(self, image: Tensor) -> ROILocatorOutput:
        _validate_image(image)
        search = self.search_stem(image)
        template = self.template_stem(self.template_rgb_normalized)
        prototype = F.normalize(template.flatten(2).mean(2), dim=1, eps=1e-6)
        search_normalized = F.normalize(search, dim=1, eps=1e-6)
        similarity = torch.einsum("bchw,bc->bhw", search_normalized, prototype.expand(image.shape[0], -1))[:, None]
        context = self.context(search)
        context_up = F.interpolate(
            context, size=search.shape[-2:], mode="bilinear", align_corners=False
        )
        refined = self.refine(torch.cat((search, similarity, context_up), dim=1))
        mask = F.interpolate(
            self.mask_head(refined),
            size=(ROI_SIZE, ROI_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        presence = self.presence_head(context)[:, 0]
        _validate_output(mask, presence, image.shape[0])
        return ROILocatorOutput(mask_logits=mask, presence_logit=presence)


class ROIPresenceMaskCascade(nn.Module):
    candidate_id = "V2_C3_PRESENCE_MASK_CASCADE"
    template_conditioned = False

    def __init__(self) -> None:
        super().__init__()
        self.encoder = ROIEncoder(base=16)
        self.rough_head = nn.Conv2d(64, 1, 1)
        self.presence_head = nn.Sequential(
            DeterministicGlobalMaxPool2d(),
            nn.Flatten(),
            nn.Linear(64, 1),
        )
        self.refine_mid = nn.Sequential(
            ConvNormAct(32 + 1, 32),
            ResidualBlock(32),
        )
        self.refine_high = nn.Sequential(
            ConvNormAct(16 + 32 + 1, 32),
            ResidualBlock(32),
            ConvNormAct(32, 16),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, image: Tensor) -> ROILocatorOutput:
        _validate_image(image)
        high, mid, deep = self.encoder(image)
        rough = self.rough_head(deep)
        rough_mid = F.interpolate(
            rough, size=mid.shape[-2:], mode="bilinear", align_corners=False
        )
        mid_value = self.refine_mid(torch.cat((mid, rough_mid), dim=1))
        rough_high = F.interpolate(
            rough, size=high.shape[-2:], mode="bilinear", align_corners=False
        )
        mid_high = F.interpolate(
            mid_value, size=high.shape[-2:], mode="bilinear", align_corners=False
        )
        mask = self.refine_high(torch.cat((high, mid_high, rough_high), dim=1))
        presence = self.presence_head(deep)[:, 0]
        _validate_output(mask, presence, image.shape[0])
        return ROILocatorOutput(mask_logits=mask, presence_logit=presence)


def build_candidate(
    candidate_id: str, *, template_rgb_normalized: Tensor
) -> nn.Module:
    if candidate_id == ROIUNetSmall.candidate_id:
        return ROIUNetSmall()
    if candidate_id == ROITemplateCorrelation.candidate_id:
        return ROITemplateCorrelation(template_rgb_normalized)
    if candidate_id == ROIPresenceMaskCascade.candidate_id:
        return ROIPresenceMaskCascade()
    raise KeyError(candidate_id)


def architecture_receipt(model: nn.Module) -> ROIArchitecture:
    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return ROIArchitecture(
        candidate_id=str(model.candidate_id),
        parameters=parameters,
        trainable_parameters=trainable,
        input_shape=(3, ROI_SIZE, ROI_SIZE),
        mask_output_shape=(1, ROI_SIZE, ROI_SIZE),
        explicit_presence_head=True,
        template_conditioned=bool(model.template_conditioned),
    )


def _validate_image(image: Tensor) -> None:
    if image.dtype != torch.float32 or tuple(image.shape[1:]) != (3, ROI_SIZE, ROI_SIZE):
        raise ValueError("ROI image must be float32 Bx3x128x128")


def _validate_output(mask: Tensor, presence: Tensor, batch: int) -> None:
    if tuple(mask.shape) != (batch, 1, ROI_SIZE, ROI_SIZE):
        raise RuntimeError("ROI mask output geometry differs")
    if tuple(presence.shape) != (batch,):
        raise RuntimeError("ROI presence output geometry differs")


def _validate_template(template: Tensor) -> None:
    if template.dtype != torch.float32 or tuple(template.shape) != (1, 3, ROI_SIZE, ROI_SIZE):
        raise ValueError("template must be float32 1x3x128x128")
    if not bool(torch.isfinite(template).all()):
        raise ValueError("template must be finite")


__all__ = [
    "ROI_SIZE",
    "ROIArchitecture",
    "DeterministicGlobalMaxPool2d",
    "ROILocatorOutput",
    "ROIUNetSmall",
    "ROITemplateCorrelation",
    "ROIPresenceMaskCascade",
    "architecture_receipt",
    "build_candidate",
]
