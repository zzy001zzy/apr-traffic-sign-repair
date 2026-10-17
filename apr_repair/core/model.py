# SPDX-FileCopyrightText: 2026 Chih-Yuan Cheng and Bor-Jiunn Hwang
# SPDX-License-Identifier: MIT
"""Frozen KCR-R-96 repair network.

This module deliberately contains one graph only.  The network consumes an
already hard-erased RGB crop plus its known binary mask; it has no template,
projection, gate, search, or fallback path.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


CANONICAL_SIZE = 96
EXPECTED_PARAMETER_COUNT = 90_123
EXPECTED_CONV_MACS = 132_378_624


class DSRes(nn.Module):
    """Depthwise-separable residual block fixed by the KCR-R V1 contract."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=channels,
            bias=True,
        )
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, value: Tensor) -> Tensor:
        residual = self.pointwise(F.silu(self.depthwise(value)))
        return F.silu(value + residual)


class KCRR96(nn.Module):
    """The single-path 4-channel to RGB KCR-R-96 repair model."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(4, 24, kernel_size=3, stride=1, padding=1, bias=True)
        self.encoder_24 = DSRes(24)
        self.down_48 = nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1, bias=True)
        self.encoder_48 = DSRes(48)
        self.down_96 = nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1, bias=True)
        self.bottleneck_a = DSRes(96)
        self.bottleneck_b = DSRes(96)
        self.fuse_48 = nn.Conv2d(144, 48, kernel_size=1, bias=True)
        self.decoder_48 = DSRes(48)
        self.fuse_24 = nn.Conv2d(72, 24, kernel_size=1, bias=True)
        self.decoder_24 = DSRes(24)
        self.head = nn.Conv2d(24, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.assert_frozen_architecture()

    def assert_frozen_architecture(self) -> None:
        actual = sum(parameter.numel() for parameter in self.parameters())
        if actual != EXPECTED_PARAMETER_COUNT:
            raise RuntimeError(
                f"KCR-R-96 parameter count differs: {actual} != {EXPECTED_PARAMETER_COUNT}"
            )

    def forward(self, value: Tensor) -> Tensor:
        if not isinstance(value, Tensor):
            raise TypeError("KCR-R input must be a torch.Tensor")
        if value.dtype != torch.float32:
            raise TypeError("KCR-R input dtype must be torch.float32")
        if value.ndim != 4 or tuple(value.shape[1:]) != (4, 96, 96):
            raise ValueError("KCR-R input shape must be Bx4x96x96")
        if value.shape[0] < 1:
            raise ValueError("KCR-R batch must be nonempty")
        if not torch.isfinite(value).all():
            raise ValueError("KCR-R input must be finite")

        s0 = self.encoder_24(F.silu(self.stem(value)))
        s1 = self.encoder_48(F.silu(self.down_48(s0)))
        bottleneck = self.bottleneck_b(self.bottleneck_a(F.silu(self.down_96(s1))))

        up_48 = F.interpolate(bottleneck, size=(48, 48), mode="bilinear", align_corners=False)
        up_48 = self.decoder_48(F.silu(self.fuse_48(torch.cat((up_48, s1), dim=1))))
        up_24 = F.interpolate(up_48, size=(96, 96), mode="bilinear", align_corners=False)
        up_24 = self.decoder_24(F.silu(self.fuse_24(torch.cat((up_24, s0), dim=1))))
        logits = self.head(up_24)
        return 0.5 + 0.5 * torch.tanh(logits)


def parameter_count(model: nn.Module) -> int:
    """Return the number of trainable and non-trainable parameter scalars."""

    return sum(parameter.numel() for parameter in model.parameters())


__all__ = [
    "CANONICAL_SIZE",
    "DSRes",
    "EXPECTED_CONV_MACS",
    "EXPECTED_PARAMETER_COUNT",
    "KCRR96",
    "parameter_count",
]
