# SPDX-FileCopyrightText: 2026 Chih-Yuan Cheng and Bor-Jiunn Hwang
# SPDX-License-Identifier: MIT
"""Successor V2 context reconstruction with attack-evidence soft gating."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .model import CANONICAL_SIZE, KCRR96


EXPECTED_PARAMETER_COUNT_V2 = 103_756
EXPECTED_CONV_MACS_V2 = 197_554_176


@dataclass(frozen=True, slots=True)
class ContextGateOutput:
    """Training-visible tensors from the single V2 repair path."""

    candidate: Tensor
    gate: Tensor
    repaired: Tensor


class AttackEvidenceGate(nn.Module):
    """Tiny pixel gate driven only by raw evidence and a detached candidate."""

    def __init__(self) -> None:
        super().__init__()
        self.conv_0 = nn.Conv2d(10, 16, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(16, 24, kernel_size=3, stride=2, padding=1)
        self.conv_2 = nn.Conv2d(24, 24, kernel_size=3, padding=1)
        self.conv_3 = nn.Conv2d(24, 16, kernel_size=3, padding=1)
        self.head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, evidence: Tensor) -> Tensor:
        value = F.silu(self.conv_0(evidence))
        value = F.silu(self.conv_1(value))
        value = F.silu(self.conv_2(value))
        value = F.interpolate(
            value,
            size=(CANONICAL_SIZE, CANONICAL_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        value = F.silu(self.conv_3(value))
        return self.head(value)


class ContextGateRepair96(nn.Module):
    """One fixed context-candidate plus soft evidence-gate forward graph."""

    def __init__(self) -> None:
        super().__init__()
        self.reconstruction = KCRR96()
        self.evidence = AttackEvidenceGate()
        actual = sum(parameter.numel() for parameter in self.parameters())
        if actual != EXPECTED_PARAMETER_COUNT_V2:
            raise RuntimeError(
                f"V2 parameter count differs: {actual} != {EXPECTED_PARAMETER_COUNT_V2}"
            )

    def forward(
        self,
        context_input: Tensor,
        raw_rgb: Tensor,
        mask: Tensor,
    ) -> ContextGateOutput:
        if context_input.dtype != torch.float32 or raw_rgb.dtype != torch.float32:
            raise TypeError("V2 RGB inputs must be torch.float32")
        if mask.dtype != torch.bool:
            raise TypeError("V2 mask must be strict torch.bool")
        if (
            context_input.ndim != 4
            or tuple(context_input.shape[1:]) != (4, CANONICAL_SIZE, CANONICAL_SIZE)
        ):
            raise ValueError("context_input must have shape Bx4x96x96")
        expected_rgb = (
            context_input.shape[0],
            3,
            CANONICAL_SIZE,
            CANONICAL_SIZE,
        )
        if tuple(raw_rgb.shape) != expected_rgb:
            raise ValueError("raw_rgb must have shape Bx3x96x96")
        if tuple(mask.shape) != (
            context_input.shape[0],
            1,
            CANONICAL_SIZE,
            CANONICAL_SIZE,
        ):
            raise ValueError("mask must have shape Bx1x96x96")
        if context_input.shape[0] < 1:
            raise ValueError("V2 batch must be nonempty")
        if (
            context_input.device != raw_rgb.device
            or context_input.device != mask.device
        ):
            raise ValueError("V2 inputs must share one device")
        if not torch.isfinite(context_input).all() or not torch.isfinite(raw_rgb).all():
            raise ValueError("V2 inputs must be finite")

        candidate = self.reconstruction(context_input)
        mask_float = mask.to(dtype=torch.float32)
        raw_masked = raw_rgb * mask_float
        candidate_masked = candidate.detach() * mask_float
        evidence = torch.cat(
            (
                raw_masked,
                candidate_masked,
                torch.abs(raw_masked - candidate_masked),
                mask_float,
            ),
            dim=1,
        )
        gate = torch.sigmoid(self.evidence(evidence)) * mask_float
        repaired_inside = raw_rgb * (1.0 - gate) + candidate * gate
        repaired = torch.where(mask, repaired_inside, raw_rgb)
        return ContextGateOutput(candidate=candidate, gate=gate, repaired=repaired)


__all__ = [
    "AttackEvidenceGate",
    "ContextGateOutput",
    "ContextGateRepair96",
    "EXPECTED_CONV_MACS_V2",
    "EXPECTED_PARAMETER_COUNT_V2",
]
