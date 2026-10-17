# SPDX-FileCopyrightText: 2026 Chih-Yuan Cheng and Bor-Jiunn Hwang
# SPDX-License-Identifier: MIT
"""APR-HardSelect V4.1: bounded Telea refiner with component hard selection.

V4.1 is the smallest causal successor to APR-Gate V3.3.  It retains the
successful analytic proposal and residual candidate, but removes the soft
pixel-wise attenuation that rejected recoverable REAP candidates.  The gate
is trained as an input-only attacked/clean classifier and makes one frozen
0.5 decision per connected mask component.  No detector output is available
to this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .model_v3 import (
    APREvidenceGate,
    AnalyticPriorResidualRefiner,
    EXPECTED_PARAMETER_COUNT_V3,
)


HARD_SELECT_THRESHOLD = 0.5
EXPECTED_PARAMETER_COUNT_V4_1 = EXPECTED_PARAMETER_COUNT_V3


@dataclass(frozen=True, slots=True)
class APRHardSelectOutput:
    residual: Tensor
    candidate: Tensor
    gate: Tensor
    component_probability: Tensor
    component_decision: Tensor
    repaired: Tensor


class APRHardSelectRepair96(nn.Module):
    """V3-compatible parameters with a non-attenuating component selector."""

    def __init__(self) -> None:
        super().__init__()
        self.refiner = AnalyticPriorResidualRefiner()
        self.evidence = APREvidenceGate()
        actual = sum(parameter.numel() for parameter in self.parameters())
        if actual != EXPECTED_PARAMETER_COUNT_V4_1:
            raise RuntimeError(
                f"V4.1 parameter count differs: {actual} != "
                f"{EXPECTED_PARAMETER_COUNT_V4_1}"
            )

    def forward(
        self,
        refiner_input: Tensor,
        raw_rgb: Tensor,
        proposal_rgb: Tensor,
        mask: Tensor,
    ) -> APRHardSelectOutput:
        for label, value in (
            ("refiner_input", refiner_input),
            ("raw_rgb", raw_rgb),
            ("proposal_rgb", proposal_rgb),
        ):
            if value.dtype != torch.float32:
                raise TypeError(f"{label} must be torch.float32")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{label} must be finite")
        if mask.dtype != torch.bool:
            raise TypeError("V4.1 mask must be strict torch.bool")
        batch = int(refiner_input.shape[0])
        if batch < 1 or tuple(refiner_input.shape) != (batch, 9, 96, 96):
            raise ValueError("refiner_input must have shape nonempty Bx9x96x96")
        if tuple(raw_rgb.shape) != (batch, 3, 96, 96):
            raise ValueError("raw_rgb must have shape Bx3x96x96")
        if tuple(proposal_rgb.shape) != (batch, 3, 96, 96):
            raise ValueError("proposal_rgb must have shape Bx3x96x96")
        if tuple(mask.shape) != (batch, 1, 96, 96):
            raise ValueError("mask must have shape Bx1x96x96")
        if len({refiner_input.device, raw_rgb.device, proposal_rgb.device, mask.device}) != 1:
            raise ValueError("V4.1 inputs must share one device")
        counts = mask.sum(dim=(1, 2, 3), keepdim=True)
        if bool(torch.any(counts == 0)):
            raise ValueError("every V4.1 component mask must be nonempty")

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
        component_probability = gate.sum(
            dim=(1, 2, 3), keepdim=True
        ) / counts.to(dtype=torch.float32)
        component_decision = component_probability >= HARD_SELECT_THRESHOLD
        selected = component_decision.expand_as(mask)
        repaired = torch.where(mask & selected, candidate, raw_rgb)
        return APRHardSelectOutput(
            residual=residual,
            candidate=candidate,
            gate=gate,
            component_probability=component_probability,
            component_decision=component_decision,
            repaired=repaired,
        )


__all__ = [
    "APRHardSelectOutput",
    "APRHardSelectRepair96",
    "EXPECTED_PARAMETER_COUNT_V4_1",
    "HARD_SELECT_THRESHOLD",
]
