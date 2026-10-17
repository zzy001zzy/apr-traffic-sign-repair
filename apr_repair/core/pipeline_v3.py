# SPDX-FileCopyrightText: 2026 Chih-Yuan Cheng and Bor-Jiunn Hwang
# SPDX-License-Identifier: MIT
"""Telea-prior canonicalization and exact-composite inference for APR-Gate V3."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from .model import CANONICAL_SIZE
from .model_v3 import APRGateRepair96
from .pipeline import (
    CanonicalBatch,
    MaskComponent,
    RepairPlan,
    RepairROI,
    _copy_to_canvas,
    connected_components,
    inverse_resize_predictions,
    plan_repair_geometry,
)
from .pipeline_v2 import _canonicalize_prevalidated_plan


TELEA_RADIUS = 3.0
TELEA_ROI_PADDING = 4
TELEA_CONTEXT_FILL = 0
_COORDINATE_AXIS = torch.linspace(
    -1.0,
    1.0,
    CANONICAL_SIZE,
    dtype=torch.float32,
)
_COORDINATE_YY, _COORDINATE_XX = torch.meshgrid(
    _COORDINATE_AXIS,
    _COORDINATE_AXIS,
    indexing="ij",
)
_CACHED_COORDINATE_PLANES = torch.stack(
    (_COORDINATE_XX, _COORDINATE_YY),
).unsqueeze(0).contiguous()


@dataclass(frozen=True, slots=True)
class APRCanonicalBatch:
    refiner_inputs: Tensor
    raw_inputs: Tensor
    proposal_inputs: Tensor
    masks: Tensor
    targets: Tensor | None
    components: tuple[MaskComponent, ...]
    rois: tuple[RepairROI, ...]

    def as_v1_geometry(self) -> CanonicalBatch:
        return CanonicalBatch(
            inputs=self.refiner_inputs[:, :4],
            masks=self.masks,
            targets=self.targets,
            components=self.components,
            rois=self.rois,
        )


def _telea_proposal_for_components(
    source_rgb: Tensor,
    mask: Tensor,
    components: tuple[MaskComponent, ...],
) -> Tensor:
    image = np.ascontiguousarray(source_rgb.cpu().numpy(), dtype=np.uint8)
    mask_u8 = np.ascontiguousarray(mask.cpu().numpy(), dtype=np.uint8) * 255
    if not components:
        return source_rgb.clone()
    left = max(
        0,
        min(component.bbox_xyxy[0] for component in components)
        - TELEA_ROI_PADDING,
    )
    top = max(
        0,
        min(component.bbox_xyxy[1] for component in components)
        - TELEA_ROI_PADDING,
    )
    right = min(
        image.shape[1],
        max(component.bbox_xyxy[2] for component in components)
        + TELEA_ROI_PADDING,
    )
    bottom = min(
        image.shape[0],
        max(component.bbox_xyxy[3] for component in components)
        + TELEA_ROI_PADDING,
    )
    proposal = image.copy()
    proposal[top:bottom, left:right] = cv2.inpaint(
        image[top:bottom, left:right],
        mask_u8[top:bottom, left:right],
        TELEA_RADIUS,
        cv2.INPAINT_TELEA,
    )
    return torch.from_numpy(np.ascontiguousarray(proposal, dtype=np.uint8))


def telea_proposal(source_rgb: Tensor, mask: Tensor) -> Tensor:
    """Apply the byte-exact frozen Telea-r3 output through component ROIs."""

    if source_rgb.dtype != torch.uint8 or source_rgb.ndim != 3:
        raise TypeError("source_rgb must be contiguous uint8 HWC")
    if mask.dtype != torch.bool or tuple(mask.shape) != tuple(source_rgb.shape[:2]):
        raise TypeError("mask must be strict bool HW matching source_rgb")
    components = connected_components(mask) if bool(mask.any()) else ()
    return _telea_proposal_for_components(source_rgb, mask, components)


def _masked_context_source(source_rgb: Tensor, mask: Tensor) -> Tensor:
    """Erase bytes that are semantically hidden by the known mask."""

    normalized = source_rgb.clone()
    normalized[mask] = TELEA_CONTEXT_FILL
    return normalized.contiguous()


def _context_only_telea_proposal_for_components(
    source_rgb: Tensor,
    mask: Tensor,
    components: tuple[MaskComponent, ...],
) -> Tensor:
    """Build the Telea prior from visible context only.

    OpenCV Telea can depend on source bytes inside the inpaint mask for some
    degenerate image/ROI geometries. Those bytes are not observable evidence
    under the known-mask protocol, so normalize them before invoking the same
    frozen Telea-r3 implementation.
    """

    return _telea_proposal_for_components(
        _masked_context_source(source_rgb, mask),
        mask,
        components,
    )


def context_only_telea_proposal(source_rgb: Tensor, mask: Tensor) -> Tensor:
    """Apply frozen Telea-r3 after deterministic known-mask pre-erasure."""

    if source_rgb.dtype != torch.uint8 or source_rgb.ndim != 3:
        raise TypeError("source_rgb must be contiguous uint8 HWC")
    if mask.dtype != torch.bool or tuple(mask.shape) != tuple(source_rgb.shape[:2]):
        raise TypeError("mask must be strict bool HW matching source_rgb")
    components = connected_components(mask) if bool(mask.any()) else ()
    return _context_only_telea_proposal_for_components(
        source_rgb,
        mask,
        components,
    )


def _coordinate_planes(batch_size: int) -> Tensor:
    return _CACHED_COORDINATE_PLANES.expand(batch_size, -1, -1, -1)


def _roi_rgb_canvas(source_rgb: Tensor, roi: RepairROI) -> Tensor:
    """Crop uint8 first, then convert only the intersecting ROI to FP32."""

    frame_left, frame_top, frame_right, frame_bottom = roi.frame_xyxy
    canvas_left, canvas_top, canvas_right, canvas_bottom = roi.canvas_xyxy
    canvas = torch.full((3, roi.size, roi.size), 0.5, dtype=torch.float32)
    crop = (
        source_rgb[frame_top:frame_bottom, frame_left:frame_right]
        .permute(2, 0, 1)
        .to(torch.float32)
        .div(255.0)
    )
    canvas[:, canvas_top:canvas_bottom, canvas_left:canvas_right] = crop
    return canvas


def _roi_mask_canvas(mask: Tensor, roi: RepairROI) -> Tensor:
    frame_left, frame_top, frame_right, frame_bottom = roi.frame_xyxy
    canvas_left, canvas_top, canvas_right, canvas_bottom = roi.canvas_xyxy
    canvas = torch.zeros((1, roi.size, roi.size), dtype=torch.float32)
    canvas[:, canvas_top:canvas_bottom, canvas_left:canvas_right] = mask[
        frame_top:frame_bottom,
        frame_left:frame_right,
    ].to(torch.float32)
    return canvas


def _canonicalize_proposal(proposal_rgb: Tensor, plan: RepairPlan) -> Tensor:
    proposal_chw = proposal_rgb.permute(2, 0, 1).to(torch.float32).div(255.0)
    rows = []
    for roi in plan.rois:
        canvas = _copy_to_canvas(proposal_chw, roi, fill=0.5)
        rows.append(
            F.interpolate(
                canvas.unsqueeze(0),
                size=(CANONICAL_SIZE, CANONICAL_SIZE),
                mode="bilinear",
                align_corners=False,
            )[0]
        )
    return torch.stack(rows).contiguous()


def prepare_apr_gate_batch(
    source_rgb: Tensor,
    mask: Tensor,
    *,
    clean_target_rgb: Tensor | None = None,
    proposal_rgb: Tensor | None = None,
) -> APRCanonicalBatch:
    """Prepare one plan and one packed resize per ROI for identical V3 tensors."""

    plan = plan_repair_geometry(
        source_rgb,
        mask,
        clean_target_rgb=clean_target_rgb,
    )
    proposal = (
        _context_only_telea_proposal_for_components(
            source_rgb,
            mask,
            plan.components,
        )
        if proposal_rgb is None
        else proposal_rgb
    )
    context_rows = []
    raw_rows = []
    proposal_rows = []
    mask_rows = []
    target_rows = []
    for roi in plan.rois:
        raw_canvas = _roi_rgb_canvas(source_rgb, roi)
        mask_canvas = _roi_mask_canvas(mask, roi)
        erased_canvas = torch.where(
            mask_canvas.to(torch.bool),
            torch.full_like(raw_canvas, 0.5),
            raw_canvas,
        )
        channels = [
            erased_canvas,
            raw_canvas,
            mask_canvas,
            _roi_rgb_canvas(proposal, roi),
        ]
        if clean_target_rgb is not None:
            channels.append(_roi_rgb_canvas(clean_target_rgb, roi))
        resized = F.interpolate(
            torch.cat(channels, dim=0).unsqueeze(0),
            size=(CANONICAL_SIZE, CANONICAL_SIZE),
            mode="bilinear",
            align_corners=False,
        )[0]
        resized_mask = resized[6:7] > 0.0
        context_rows.append(
            torch.where(
                resized_mask,
                torch.full_like(resized[:3], 0.5),
                resized[:3],
            )
        )
        raw_rows.append(resized[3:6])
        mask_rows.append(resized_mask)
        proposal_rows.append(resized[7:10])
        if clean_target_rgb is not None:
            target_rows.append(resized[10:13])
    context_inputs = torch.stack(context_rows).contiguous()
    raw_inputs = torch.stack(raw_rows).contiguous()
    masks = torch.stack(mask_rows).contiguous()
    proposal_inputs = torch.stack(proposal_rows).contiguous()
    coordinates = _coordinate_planes(int(proposal_inputs.shape[0]))
    refiner_inputs = torch.cat(
        (
            context_inputs,
            proposal_inputs,
            masks.to(torch.float32),
            coordinates,
        ),
        dim=1,
    ).contiguous()
    return APRCanonicalBatch(
        refiner_inputs=refiner_inputs,
        raw_inputs=raw_inputs,
        proposal_inputs=proposal_inputs,
        masks=masks,
        targets=torch.stack(target_rows).contiguous() if target_rows else None,
        components=plan.components,
        rois=plan.rois,
    )


def prepare_paired_apr_gate_batches(
    attacked_rgb: Tensor,
    clean_rgb: Tensor,
    mask: Tensor,
) -> tuple[APRCanonicalBatch, Tensor]:
    """Build one attacked candidate input and the paired-clean gate raw crops."""

    if not torch.equal(attacked_rgb[~mask], clean_rgb[~mask]):
        raise RuntimeError("paired attacked/clean visible contexts differ")
    proposal = context_only_telea_proposal(attacked_rgb, mask)
    clean_proposal = context_only_telea_proposal(clean_rgb, mask)
    if not torch.equal(proposal, clean_proposal):
        raise RuntimeError(
            "paired context-only Telea proposals differ despite identical visible context"
        )
    attacked = prepare_apr_gate_batch(
        attacked_rgb,
        mask,
        clean_target_rgb=clean_rgb,
        proposal_rgb=proposal,
    )
    clean_base = _canonicalize_prevalidated_plan(
        clean_rgb,
        mask,
        RepairPlan(tuple(clean_rgb.shape), attacked.components, attacked.rois),
        clean_target_rgb=clean_rgb,
    )
    if not torch.equal(
        attacked.refiner_inputs[:, :3],
        clean_base.context_inputs[:, :3],
    ):
        raise RuntimeError("paired attacked/clean mask-excluded contexts differ")
    return attacked, clean_base.raw_inputs


def repair_frame_v3(
    model: APRGateRepair96,
    source_rgb: Tensor,
    mask: Tensor,
    *,
    device: torch.device | str,
) -> Tensor:
    """Run APR-Gate and preserve every source byte outside the known mask."""

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
        raise RuntimeError("V3 outside-mask byte identity violation")
    return output.contiguous()


__all__ = [
    "APRCanonicalBatch",
    "TELEA_CONTEXT_FILL",
    "TELEA_RADIUS",
    "TELEA_ROI_PADDING",
    "context_only_telea_proposal",
    "prepare_apr_gate_batch",
    "prepare_paired_apr_gate_batches",
    "repair_frame_v3",
    "telea_proposal",
]
