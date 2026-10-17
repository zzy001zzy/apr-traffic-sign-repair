# SPDX-FileCopyrightText: 2026 Chih-Yuan Cheng and Bor-Jiunn Hwang
# SPDX-License-Identifier: MIT
"""Canonical raw/context preparation and exact-composite inference for V2."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as F

from .model import CANONICAL_SIZE
from .model_v2 import ContextGateRepair96
from .pipeline import (
    CanonicalBatch,
    MaskComponent,
    RepairPlan,
    RepairROI,
    _copy_to_canvas,
    _validate_frame,
    canonicalize_repair_plan,
    inverse_resize_predictions,
    plan_repair_geometry,
    validate_mask,
)


@dataclass(frozen=True, slots=True)
class ContextGateCanonicalBatch:
    context_inputs: Tensor
    raw_inputs: Tensor
    masks: Tensor
    targets: Tensor | None
    components: tuple[MaskComponent, ...]
    rois: tuple[RepairROI, ...]

    def as_v1_geometry(self) -> CanonicalBatch:
        return CanonicalBatch(
            inputs=self.context_inputs,
            masks=self.masks,
            targets=self.targets,
            components=self.components,
            rois=self.rois,
        )


def _canonicalize_prevalidated_plan(
    source_rgb: Tensor,
    mask: Tensor,
    plan: RepairPlan,
    *,
    clean_target_rgb: Tensor | None,
) -> ContextGateCanonicalBatch:
    """Canonicalize a plan returned by ``plan_repair_geometry`` without rescanning."""

    source_chw = source_rgb.permute(2, 0, 1).to(dtype=torch.float32).div(255.0)
    mask_chw = mask.unsqueeze(0)
    erased_chw = torch.where(mask_chw, torch.full_like(source_chw, 0.5), source_chw)
    target_chw = (
        clean_target_rgb.permute(2, 0, 1).to(dtype=torch.float32).div(255.0)
        if clean_target_rgb is not None
        else None
    )
    context_rows: list[Tensor] = []
    raw_rows: list[Tensor] = []
    mask_rows: list[Tensor] = []
    target_rows: list[Tensor] = []
    for roi in plan.rois:
        erased_canvas = _copy_to_canvas(erased_chw, roi, fill=0.5)
        raw_canvas = _copy_to_canvas(source_chw, roi, fill=0.5)
        mask_canvas = _copy_to_canvas(mask_chw.to(torch.float32), roi, fill=0.0)
        resized_mask = (
            F.interpolate(
                mask_canvas.unsqueeze(0),
                size=(CANONICAL_SIZE, CANONICAL_SIZE),
                mode="bilinear",
                align_corners=False,
            )[0]
            > 0.0
        )
        resized_erased = F.interpolate(
            erased_canvas.unsqueeze(0),
            size=(CANONICAL_SIZE, CANONICAL_SIZE),
            mode="bilinear",
            align_corners=False,
        )[0]
        restricted = torch.where(
            resized_mask,
            torch.full_like(resized_erased, 0.5),
            resized_erased,
        )
        context_rows.append(
            torch.cat((restricted, resized_mask.to(torch.float32)), dim=0)
        )
        raw_rows.append(
            F.interpolate(
                raw_canvas.unsqueeze(0),
                size=(CANONICAL_SIZE, CANONICAL_SIZE),
                mode="bilinear",
                align_corners=False,
            )[0]
        )
        mask_rows.append(resized_mask)
        if target_chw is not None:
            target_canvas = _copy_to_canvas(target_chw, roi, fill=0.5)
            target_rows.append(
                F.interpolate(
                    target_canvas.unsqueeze(0),
                    size=(CANONICAL_SIZE, CANONICAL_SIZE),
                    mode="bilinear",
                    align_corners=False,
                )[0]
            )
    return ContextGateCanonicalBatch(
        context_inputs=torch.stack(context_rows).contiguous(),
        raw_inputs=torch.stack(raw_rows).contiguous(),
        masks=torch.stack(mask_rows).contiguous(),
        targets=torch.stack(target_rows).contiguous() if target_rows else None,
        components=plan.components,
        rois=plan.rois,
    )


def canonicalize_context_gate_plan(
    attacked_rgb: Tensor,
    mask: Tensor,
    plan: RepairPlan,
    *,
    clean_target_rgb: Tensor | None = None,
) -> ContextGateCanonicalBatch:
    """Prepare identical-geometry context, raw, mask, and optional target crops."""

    canonical = canonicalize_repair_plan(
        attacked_rgb,
        mask,
        plan,
        clean_target_rgb=clean_target_rgb,
    )
    attacked_chw = attacked_rgb.permute(2, 0, 1).to(dtype=torch.float32).div(255.0)
    raw_rows: list[Tensor] = []
    for roi in plan.rois:
        raw_canvas = _copy_to_canvas(attacked_chw, roi, fill=0.5)
        raw_rows.append(
            F.interpolate(
                raw_canvas.unsqueeze(0),
                size=(CANONICAL_SIZE, CANONICAL_SIZE),
                mode="bilinear",
                align_corners=False,
            )[0]
        )
    return ContextGateCanonicalBatch(
        context_inputs=canonical.inputs,
        raw_inputs=torch.stack(raw_rows).contiguous(),
        masks=canonical.masks,
        targets=canonical.targets,
        components=canonical.components,
        rois=canonical.rois,
    )


def prepare_context_gate_batch(
    attacked_rgb: Tensor,
    mask: Tensor,
    *,
    clean_target_rgb: Tensor | None = None,
) -> ContextGateCanonicalBatch:
    plan = plan_repair_geometry(
        attacked_rgb,
        mask,
        clean_target_rgb=clean_target_rgb,
    )
    return _canonicalize_prevalidated_plan(
        attacked_rgb,
        mask,
        plan,
        clean_target_rgb=clean_target_rgb,
    )


def prepare_paired_context_gate_batches(
    attacked_rgb: Tensor,
    clean_rgb: Tensor,
    mask: Tensor,
) -> tuple[ContextGateCanonicalBatch, ContextGateCanonicalBatch]:
    """Prepare attacked and paired-clean tensors with one geometry scan."""

    plan = plan_repair_geometry(
        attacked_rgb,
        mask,
        clean_target_rgb=clean_rgb,
    )
    attacked = _canonicalize_prevalidated_plan(
        attacked_rgb,
        mask,
        plan,
        clean_target_rgb=clean_rgb,
    )
    clean = _canonicalize_prevalidated_plan(
        clean_rgb,
        mask,
        plan,
        clean_target_rgb=clean_rgb,
    )
    return attacked, clean


def prepare_clean_raw_batch(
    clean_rgb: Tensor,
    mask: Tensor,
    plan: RepairPlan,
) -> Tensor:
    """Prepare clean raw crops with the exact same immutable geometry."""

    _validate_frame(clean_rgb, label="clean_rgb")
    validate_mask(mask, tuple(clean_rgb.shape))
    clean_chw = clean_rgb.permute(2, 0, 1).to(dtype=torch.float32).div(255.0)
    rows = []
    for roi in plan.rois:
        canvas = _copy_to_canvas(clean_chw, roi, fill=0.5)
        rows.append(
            F.interpolate(
                canvas.unsqueeze(0),
                size=(CANONICAL_SIZE, CANONICAL_SIZE),
                mode="bilinear",
                align_corners=False,
            )[0]
        )
    return torch.stack(rows).contiguous()


def repair_frame_v2(
    model: ContextGateRepair96,
    attacked_rgb: Tensor,
    mask: Tensor,
    *,
    device: torch.device | str,
) -> Tensor:
    """Run V2 and preserve every source byte outside the strict known mask."""

    canonical = prepare_context_gate_batch(attacked_rgb, mask)
    output = model(
        canonical.context_inputs.to(device=device, dtype=torch.float32),
        canonical.raw_inputs.to(device=device, dtype=torch.float32),
        canonical.masks.to(device=device, dtype=torch.bool),
    )
    prediction_cpu = (
        output.repaired.detach().to(device="cpu", dtype=torch.float32).contiguous()
    )
    candidates = inverse_resize_predictions(
        canonical.as_v1_geometry(),
        prediction_cpu,
    )
    output = attacked_rgb.clone()
    for component, roi, candidate in zip(
        canonical.components,
        canonical.rois,
        candidates,
        strict=True,
    ):
        frame_left, frame_top, frame_right, frame_bottom = roi.frame_xyxy
        canvas_left, canvas_top, canvas_right, canvas_bottom = roi.canvas_xyxy
        component_slice = component.mask[frame_top:frame_bottom, frame_left:frame_right]
        output_slice = output[frame_top:frame_bottom, frame_left:frame_right]
        candidate_slice = candidate[canvas_top:canvas_bottom, canvas_left:canvas_right]
        output_slice[component_slice] = candidate_slice[component_slice]
    if not torch.equal(output[~mask], attacked_rgb[~mask]):
        raise RuntimeError("V2 outside-mask byte identity violation")
    return output.contiguous()


__all__ = [
    "ContextGateCanonicalBatch",
    "canonicalize_context_gate_plan",
    "prepare_clean_raw_batch",
    "prepare_context_gate_batch",
    "prepare_paired_context_gate_batches",
    "repair_frame_v2",
]
