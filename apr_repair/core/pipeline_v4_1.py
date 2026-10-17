# SPDX-FileCopyrightText: 2026 Chih-Yuan Cheng and Bor-Jiunn Hwang
# SPDX-License-Identifier: MIT
"""Canonical inference and exact composition for APR-HardSelect V4.1."""

from __future__ import annotations

import torch
from torch import Tensor

from .model_v4_1 import APRHardSelectRepair96
from .pipeline import inverse_resize_predictions
from .pipeline_v3 import prepare_apr_gate_batch


def repair_frame_v4_1(
    model: APRHardSelectRepair96,
    source_rgb: Tensor,
    mask: Tensor,
    *,
    device: torch.device | str,
) -> tuple[Tensor, tuple[float, ...], tuple[bool, ...]]:
    """Repair known-mask components and preserve every byte outside the mask."""

    if not bool(mask.any()):
        return source_rgb.clone().contiguous(), (), ()
    canonical = prepare_apr_gate_batch(source_rgb, mask)
    result = model(
        canonical.refiner_inputs.to(device=device, dtype=torch.float32),
        canonical.raw_inputs.to(device=device, dtype=torch.float32),
        canonical.proposal_inputs.to(device=device, dtype=torch.float32),
        canonical.masks.to(device=device, dtype=torch.bool),
    )
    predictions = inverse_resize_predictions(
        canonical.as_v1_geometry(),
        result.repaired.detach().to(device="cpu", dtype=torch.float32).contiguous(),
    )
    output = source_rgb.clone()
    for component, roi, candidate in zip(
        canonical.components,
        canonical.rois,
        predictions,
        strict=True,
    ):
        frame_left, frame_top, frame_right, frame_bottom = roi.frame_xyxy
        canvas_left, canvas_top, canvas_right, canvas_bottom = roi.canvas_xyxy
        component_slice = component.mask[frame_top:frame_bottom, frame_left:frame_right]
        output_slice = output[frame_top:frame_bottom, frame_left:frame_right]
        candidate_slice = candidate[canvas_top:canvas_bottom, canvas_left:canvas_right]
        output_slice[component_slice] = candidate_slice[component_slice]
    if not torch.equal(output[~mask], source_rgb[~mask]):
        raise RuntimeError("V4.1 outside-mask byte identity violation")
    probabilities = tuple(
        float(value)
        for value in result.component_probability.detach().cpu().flatten().tolist()
    )
    decisions = tuple(
        bool(value)
        for value in result.component_decision.detach().cpu().flatten().tolist()
    )
    return output.contiguous(), probabilities, decisions


__all__ = ["repair_frame_v4_1"]
