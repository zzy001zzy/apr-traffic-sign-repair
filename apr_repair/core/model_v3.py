# SPDX-FileCopyrightText: 2026 Chih-Yuan Cheng and Bor-Jiunn Hwang
# SPDX-License-Identifier: MIT
"""APR-Gate V3: analytic Telea prior, bounded residual, and soft evidence gate."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .model import CANONICAL_SIZE


RESIDUAL_BOUND = 48.0 / 255.0
EXPECTED_REFINER_PARAMETER_COUNT_V3 = 13_123
EXPECTED_GATE_PARAMETER_COUNT_V3 = 14_065
EXPECTED_PARAMETER_COUNT_V3 = 27_188
EXPECTED_CONV_MACS_V3 = 185_647_104


@dataclass(frozen=True, slots=True)
class APRGateOutput:
    """Training-visible tensors from the single APR-Gate forward path."""

    residual: Tensor
    candidate: Tensor
    gate: Tensor
    repaired: Tensor


class DilatedDepthwiseResidual(nn.Module):
    """One deterministic full-resolution residual-refinement block."""

    def __init__(self, dilation: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            32,
            32,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=32,
        )
        self.pointwise = nn.Conv2d(32, 32, kernel_size=1)

    def forward(self, value: Tensor) -> Tensor:
        residual = F.silu(self.depthwise(value))
        residual = self.pointwise(residual)
        return F.silu(value + residual)


class AnalyticPriorResidualRefiner(nn.Module):
    """Tiny bounded refiner that cannot observe raw attacked mask pixels."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(9, 32, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            DilatedDepthwiseResidual(dilation)
            for dilation in (1, 2, 4, 8, 4, 2, 1)
        )
        self.head = nn.Conv2d(32, 3, kernel_size=3, padding=1)

    def forward(self, refiner_input: Tensor) -> Tensor:
        value = F.silu(self.stem(refiner_input))
        for block in self.blocks:
            value = block(value)
        return torch.tanh(self.head(value)) * RESIDUAL_BOUND


class APREvidenceGate(nn.Module):
    """V2-shaped gate over raw/proposal/candidate evidence only."""

    def __init__(self) -> None:
        super().__init__()
        self.conv_0 = nn.Conv2d(13, 16, kernel_size=3, padding=1)
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


class APRGateRepair96(nn.Module):
    """Single analytic-prior residual repair with a clean-safety soft gate."""

    def __init__(self) -> None:
        super().__init__()
        self.refiner = AnalyticPriorResidualRefiner()
        self.evidence = APREvidenceGate()
        actual_refiner = sum(value.numel() for value in self.refiner.parameters())
        actual_gate = sum(value.numel() for value in self.evidence.parameters())
        actual = sum(value.numel() for value in self.parameters())
        expected = (
            EXPECTED_REFINER_PARAMETER_COUNT_V3,
            EXPECTED_GATE_PARAMETER_COUNT_V3,
            EXPECTED_PARAMETER_COUNT_V3,
        )
        if (actual_refiner, actual_gate, actual) != expected:
            raise RuntimeError(
                "V3 parameter count differs: "
                f"{(actual_refiner, actual_gate, actual)} != {expected}"
            )

    def forward(
        self,
        refiner_input: Tensor,
        raw_rgb: Tensor,
        proposal_rgb: Tensor,
        mask: Tensor,
    ) -> APRGateOutput:
        for label, value in (
            ("refiner_input", refiner_input),
            ("raw_rgb", raw_rgb),
            ("proposal_rgb", proposal_rgb),
        ):
            if value.dtype != torch.float32:
                raise TypeError(f"{label} must be torch.float32")
            if not torch.isfinite(value).all():
                raise ValueError(f"{label} must be finite")
        if mask.dtype != torch.bool:
            raise TypeError("V3 mask must be strict torch.bool")
        batch = refiner_input.shape[0]
        expected_hw = (CANONICAL_SIZE, CANONICAL_SIZE)
        if tuple(refiner_input.shape) != (batch, 9, *expected_hw) or batch < 1:
            raise ValueError("refiner_input must have shape nonempty Bx9x96x96")
        if tuple(raw_rgb.shape) != (batch, 3, *expected_hw):
            raise ValueError("raw_rgb must have shape Bx3x96x96")
        if tuple(proposal_rgb.shape) != (batch, 3, *expected_hw):
            raise ValueError("proposal_rgb must have shape Bx3x96x96")
        if tuple(mask.shape) != (batch, 1, *expected_hw):
            raise ValueError("mask must have shape Bx1x96x96")
        if len({refiner_input.device, raw_rgb.device, proposal_rgb.device, mask.device}) != 1:
            raise ValueError("V3 inputs must share one device")

        residual = self.refiner(refiner_input)
        candidate = torch.clamp(proposal_rgb + residual, 0.0, 1.0)
        mask_float = mask.to(dtype=torch.float32)
        raw_masked = raw_rgb * mask_float
        proposal_masked = proposal_rgb * mask_float
        candidate_masked = candidate.detach() * mask_float
        evidence = torch.cat(
            (
                raw_masked,
                proposal_masked,
                candidate_masked,
                torch.abs(raw_masked - candidate_masked),
                mask_float,
            ),
            dim=1,
        )
        gate = torch.sigmoid(self.evidence(evidence)) * mask_float
        repaired_inside = raw_rgb * (1.0 - gate) + candidate * gate
        repaired = torch.where(mask, repaired_inside, raw_rgb)
        return APRGateOutput(
            residual=residual,
            candidate=candidate,
            gate=gate,
            repaired=repaired,
        )


__all__ = [
    "APREvidenceGate",
    "APRGateOutput",
    "APRGateRepair96",
    "AnalyticPriorResidualRefiner",
    "EXPECTED_CONV_MACS_V3",
    "EXPECTED_GATE_PARAMETER_COUNT_V3",
    "EXPECTED_PARAMETER_COUNT_V3",
    "EXPECTED_REFINER_PARAMETER_COUNT_V3",
    "RESIDUAL_BOUND",
]
