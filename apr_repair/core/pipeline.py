# SPDX-FileCopyrightText: 2026 Chih-Yuan Cheng and Bor-Jiunn Hwang
# SPDX-License-Identifier: MIT
"""Exact known-mask KCR-R crop, inference, and uint8 compositing path."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Protocol

import cv2
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from .model import CANONICAL_SIZE


@dataclass(frozen=True, slots=True)
class MaskComponent:
    """One 8-connected source-mask component in deterministic order."""

    index: int
    bbox_xyxy: tuple[int, int, int, int]
    mask: Tensor


@dataclass(frozen=True, slots=True)
class RepairROI:
    """Raw square ROI and its intersection/canvas mapping."""

    component_index: int
    left: int
    top: int
    size: int
    frame_xyxy: tuple[int, int, int, int]
    canvas_xyxy: tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class CanonicalBatch:
    """Canonical model inputs and immutable inverse-warp metadata."""

    inputs: Tensor
    masks: Tensor
    targets: Tensor | None
    components: tuple[MaskComponent, ...]
    rois: tuple[RepairROI, ...]


@dataclass(frozen=True, slots=True)
class RepairPlan:
    """Validated connected components and ROIs for one immutable source mask."""

    frame_shape: tuple[int, int, int]
    components: tuple[MaskComponent, ...]
    rois: tuple[RepairROI, ...]


class RepairModel(Protocol):
    def __call__(self, value: Tensor) -> Tensor: ...


def _validate_frame(frame: Tensor, *, label: str) -> None:
    if not isinstance(frame, Tensor):
        raise TypeError(f"{label} must be a torch.Tensor")
    if frame.device.type != "cpu":
        raise ValueError(f"{label} must be host-resident")
    if frame.dtype != torch.uint8:
        raise TypeError(f"{label} must be HWC RGB torch.uint8")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"{label} must have shape HxWx3")
    if frame.shape[0] < 1 or frame.shape[1] < 1:
        raise ValueError(f"{label} must be nonempty")
    if not frame.is_contiguous():
        raise ValueError(f"{label} must be contiguous")


def validate_mask(mask: Tensor, frame_shape: tuple[int, int, int]) -> None:
    """Validate the inference mask without coercing soft or integer inputs."""

    if not isinstance(mask, Tensor):
        raise TypeError("mask must be a torch.Tensor")
    if mask.device.type != "cpu":
        raise ValueError("mask must be host-resident")
    if mask.dtype != torch.bool:
        raise TypeError("mask must be strict torch.bool")
    if mask.ndim != 2 or tuple(mask.shape) != tuple(frame_shape[:2]):
        raise ValueError("mask must be HxW and aligned with the frame")
    if not mask.is_contiguous():
        raise ValueError("mask must be contiguous")
    if not bool(mask.any()):
        raise ValueError("mask must be nonempty")


def connected_components(mask: Tensor) -> tuple[MaskComponent, ...]:
    """Return sorted 8-connected components for a validated binary mask."""

    if mask.device.type != "cpu" or mask.dtype != torch.bool or mask.ndim != 2:
        raise ValueError("connected_components requires a CPU HxW bool mask")
    if not mask.is_contiguous():
        raise ValueError("connected_components requires a contiguous mask")
    if not bool(mask.any()):
        raise ValueError("mask must be nonempty")
    array = mask.numpy().astype(np.uint8, copy=False)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(array, connectivity=8)
    records: list[tuple[tuple[int, int, int, int], Tensor]] = []
    for label in range(1, count):
        left = int(stats[label, cv2.CC_STAT_LEFT])
        top = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        bbox = (left, top, left + width, top + height)
        component_mask = torch.from_numpy((labels == label).copy())
        records.append((bbox, component_mask))
    records.sort(key=lambda record: (record[0][1], record[0][0], record[0][3], record[0][2]))
    return tuple(
        MaskComponent(index=index, bbox_xyxy=bbox, mask=component_mask)
        for index, (bbox, component_mask) in enumerate(records)
    )


def roi_for_component(
    component: MaskComponent,
    *,
    height: int,
    width: int,
) -> RepairROI:
    """Create the exact metadata-only 2.5x square ROI for one component."""

    x0, y0, x1, y1 = component.bbox_xyxy
    diameter = max(x1 - x0, y1 - y0)
    size = max(32, 2 * math.ceil(1.25 * diameter))
    center_x = (x0 + x1) / 2.0
    center_y = (y0 + y1) / 2.0
    left = math.floor(center_x - size / 2.0)
    top = math.floor(center_y - size / 2.0)
    right = left + size
    bottom = top + size
    frame_left = max(left, 0)
    frame_top = max(top, 0)
    frame_right = min(right, width)
    frame_bottom = min(bottom, height)
    if frame_right <= frame_left or frame_bottom <= frame_top:
        raise ValueError("component ROI does not intersect the frame")
    canvas_left = frame_left - left
    canvas_top = frame_top - top
    canvas_right = canvas_left + (frame_right - frame_left)
    canvas_bottom = canvas_top + (frame_bottom - frame_top)
    return RepairROI(
        component_index=component.index,
        left=left,
        top=top,
        size=size,
        frame_xyxy=(frame_left, frame_top, frame_right, frame_bottom),
        canvas_xyxy=(canvas_left, canvas_top, canvas_right, canvas_bottom),
    )


def _copy_to_canvas(
    source_chw: Tensor,
    roi: RepairROI,
    *,
    fill: float,
) -> Tensor:
    canvas = torch.full(
        (source_chw.shape[0], roi.size, roi.size),
        fill,
        dtype=torch.float32,
        device="cpu",
    )
    frame_left, frame_top, frame_right, frame_bottom = roi.frame_xyxy
    canvas_left, canvas_top, canvas_right, canvas_bottom = roi.canvas_xyxy
    canvas[:, canvas_top:canvas_bottom, canvas_left:canvas_right] = source_chw[
        :, frame_top:frame_bottom, frame_left:frame_right
    ]
    return canvas


def prepare_canonical_batch(
    attacked_rgb: Tensor,
    mask: Tensor,
    *,
    clean_target_rgb: Tensor | None = None,
    _timing_ns: dict[str, int] | None = None,
) -> CanonicalBatch:
    """Hard-erase in source space, then build all canonical component crops."""

    roi_started = time.perf_counter_ns()
    plan = plan_repair_geometry(
        attacked_rgb,
        mask,
        clean_target_rgb=clean_target_rgb,
    )
    roi_finished = time.perf_counter_ns()
    canonical = canonicalize_repair_plan(
        attacked_rgb,
        mask,
        plan,
        clean_target_rgb=clean_target_rgb,
    )
    mask_finished = time.perf_counter_ns()
    if _timing_ns is not None:
        _timing_ns["roi_crop_ns"] = roi_finished - roi_started
        _timing_ns["mask_prepare_ns"] = mask_finished - roi_finished
    return canonical


def plan_repair_geometry(
    attacked_rgb: Tensor,
    mask: Tensor,
    *,
    clean_target_rgb: Tensor | None = None,
) -> RepairPlan:
    """Validate one frame and deterministically plan all component ROIs."""

    _validate_frame(attacked_rgb, label="attacked_rgb")
    validate_mask(mask, tuple(attacked_rgb.shape))
    if clean_target_rgb is not None:
        _validate_frame(clean_target_rgb, label="clean_target_rgb")
        if clean_target_rgb.shape != attacked_rgb.shape:
            raise ValueError("clean target shape must match attacked frame")

    height, width = int(attacked_rgb.shape[0]), int(attacked_rgb.shape[1])
    components = connected_components(mask)
    rois = tuple(
        roi_for_component(component, height=height, width=width) for component in components
    )
    return RepairPlan(
        frame_shape=tuple(int(value) for value in attacked_rgb.shape),
        components=components,
        rois=rois,
    )


def _validate_geometry_against_mask(
    *,
    frame_shape: tuple[int, int, int],
    mask: Tensor,
    components: tuple[MaskComponent, ...],
    rois: tuple[RepairROI, ...],
) -> None:
    if len(components) != len(rois) or not components:
        raise ValueError("repair geometry component/ROI count differs")
    expected_components = connected_components(mask)
    expected_rois = tuple(
        roi_for_component(
            component,
            height=frame_shape[0],
            width=frame_shape[1],
        )
        for component in expected_components
    )
    if len(components) != len(expected_components):
        raise ValueError("repair geometry connected-component count differs")
    for observed, expected in zip(components, expected_components, strict=True):
        if (
            observed.index != expected.index
            or observed.bbox_xyxy != expected.bbox_xyxy
            or observed.mask.device.type != "cpu"
            or observed.mask.dtype != torch.bool
            or not observed.mask.is_contiguous()
            or not torch.equal(observed.mask, expected.mask)
        ):
            raise ValueError("repair geometry component differs from source mask")
    if rois != expected_rois:
        raise ValueError("repair geometry ROI differs from source mask")


def canonicalize_repair_plan(
    attacked_rgb: Tensor,
    mask: Tensor,
    plan: RepairPlan,
    *,
    clean_target_rgb: Tensor | None = None,
) -> CanonicalBatch:
    """Apply source erasure and canonical resampling for a validated ROI plan."""

    _validate_frame(attacked_rgb, label="attacked_rgb")
    validate_mask(mask, tuple(attacked_rgb.shape))
    if clean_target_rgb is not None:
        _validate_frame(clean_target_rgb, label="clean_target_rgb")
        if clean_target_rgb.shape != attacked_rgb.shape:
            raise ValueError("clean target shape must match attacked frame")
    if plan.frame_shape != tuple(attacked_rgb.shape):
        raise ValueError("repair plan frame shape differs")
    _validate_geometry_against_mask(
        frame_shape=plan.frame_shape,
        mask=mask,
        components=plan.components,
        rois=plan.rois,
    )

    attacked_chw = attacked_rgb.permute(2, 0, 1).to(dtype=torch.float32).div(255.0)
    full_mask_chw = mask.unsqueeze(0)
    erased_chw = torch.where(full_mask_chw, torch.full_like(attacked_chw, 0.5), attacked_chw)
    target_chw = (
        clean_target_rgb.permute(2, 0, 1).to(dtype=torch.float32).div(255.0)
        if clean_target_rgb is not None
        else None
    )

    input_rows: list[Tensor] = []
    mask_rows: list[Tensor] = []
    target_rows: list[Tensor] = []
    for roi in plan.rois:
        erased_canvas = _copy_to_canvas(erased_chw, roi, fill=0.5)
        mask_canvas = _copy_to_canvas(full_mask_chw.to(torch.float32), roi, fill=0.0)
        resized_rgb = F.interpolate(
            erased_canvas.unsqueeze(0),
            size=(CANONICAL_SIZE, CANONICAL_SIZE),
            mode="bilinear",
            align_corners=False,
        )[0]
        resized_mask = (
            F.interpolate(
                mask_canvas.unsqueeze(0),
                size=(CANONICAL_SIZE, CANONICAL_SIZE),
                mode="bilinear",
                align_corners=False,
            )[0]
            > 0.0
        )
        restricted_rgb = torch.where(resized_mask, torch.full_like(resized_rgb, 0.5), resized_rgb)
        input_rows.append(torch.cat((restricted_rgb, resized_mask.to(torch.float32)), dim=0))
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

    inputs = torch.stack(input_rows).contiguous()
    masks = torch.stack(mask_rows).contiguous()
    targets = torch.stack(target_rows).contiguous() if target_rows else None
    return CanonicalBatch(
        inputs=inputs,
        masks=masks,
        targets=targets,
        components=plan.components,
        rois=plan.rois,
    )


def quantize_prediction(prediction: Tensor) -> Tensor:
    """Apply the frozen CPU FP32 ties-to-even RGB quantization."""

    if prediction.device.type != "cpu" or prediction.dtype != torch.float32:
        raise TypeError("prediction must be contiguous CPU torch.float32")
    if prediction.ndim != 3 or prediction.shape[0] != 3:
        raise ValueError("prediction must have shape 3xHxW")
    if not prediction.is_contiguous():
        raise ValueError("prediction must be contiguous")
    if not torch.isfinite(prediction).all():
        raise ValueError("prediction must be finite")
    return (
        torch.round(torch.clamp(prediction, 0.0, 1.0) * 255.0)
        .to(torch.uint8)
        .permute(1, 2, 0)
        .contiguous()
    )


def compose_predictions(
    attacked_rgb: Tensor,
    source_mask: Tensor,
    canonical: CanonicalBatch,
    predictions: Tensor,
    *,
    _timing_ns: dict[str, int] | None = None,
) -> Tensor:
    """Inverse-resize and write each prediction only to its source component."""

    compose_started = time.perf_counter_ns()
    candidates = inverse_resize_predictions(canonical, predictions)
    inverse_finished = time.perf_counter_ns()
    output = exact_compose_candidates(
        attacked_rgb,
        source_mask,
        canonical,
        candidates,
    )
    compose_finished = time.perf_counter_ns()
    if _timing_ns is not None:
        _timing_ns["inverse_resize_ns"] = inverse_finished - compose_started
        _timing_ns["exact_compose_ns"] = compose_finished - inverse_finished
    return output


def inverse_resize_predictions(
    canonical: CanonicalBatch,
    predictions: Tensor,
) -> tuple[Tensor, ...]:
    """Inverse-resize and quantize predictions without writing source pixels."""

    if predictions.device.type != "cpu" or predictions.dtype != torch.float32:
        raise TypeError("predictions must be CPU torch.float32")
    expected_shape = (len(canonical.rois), 3, CANONICAL_SIZE, CANONICAL_SIZE)
    if tuple(predictions.shape) != expected_shape:
        raise ValueError(f"prediction shape must be {expected_shape}")
    if not predictions.is_contiguous():
        raise ValueError("predictions must be contiguous")
    if not torch.isfinite(predictions).all():
        raise ValueError("predictions must be finite")

    candidates: list[Tensor] = []
    for roi, prediction in zip(canonical.rois, predictions, strict=True):
        candidate = F.interpolate(
            prediction.unsqueeze(0),
            size=(roi.size, roi.size),
            mode="bilinear",
            align_corners=False,
        )[0].contiguous()
        candidates.append(quantize_prediction(candidate))
    return tuple(candidates)


def exact_compose_candidates(
    attacked_rgb: Tensor,
    source_mask: Tensor,
    canonical: CanonicalBatch,
    candidates: tuple[Tensor, ...],
) -> Tensor:
    """Write pre-resized candidates only to their exact source components."""

    _validate_frame(attacked_rgb, label="attacked_rgb")
    validate_mask(source_mask, tuple(attacked_rgb.shape))
    _validate_geometry_against_mask(
        frame_shape=tuple(attacked_rgb.shape),
        mask=source_mask,
        components=canonical.components,
        rois=canonical.rois,
    )
    if len(candidates) != len(canonical.rois):
        raise ValueError("candidate count must match canonical ROIs")
    output = attacked_rgb.clone()
    for component, roi, candidate_u8 in zip(
        canonical.components, canonical.rois, candidates, strict=True
    ):
        if (
            candidate_u8.device.type != "cpu"
            or candidate_u8.dtype != torch.uint8
            or tuple(candidate_u8.shape) != (roi.size, roi.size, 3)
            or not candidate_u8.is_contiguous()
        ):
            raise ValueError("inverse-resized candidate contract differs")
        frame_left, frame_top, frame_right, frame_bottom = roi.frame_xyxy
        canvas_left, canvas_top, canvas_right, canvas_bottom = roi.canvas_xyxy
        component_slice = component.mask[frame_top:frame_bottom, frame_left:frame_right]
        output_slice = output[frame_top:frame_bottom, frame_left:frame_right]
        candidate_slice = candidate_u8[canvas_top:canvas_bottom, canvas_left:canvas_right]
        output_slice[component_slice] = candidate_slice[component_slice]

    if not torch.equal(output[~source_mask], attacked_rgb[~source_mask]):
        raise RuntimeError("outside-mask byte identity violation")
    return output.contiguous()


def repair_frame(
    model: RepairModel,
    attacked_rgb: Tensor,
    mask: Tensor,
    *,
    device: torch.device | str,
) -> Tensor:
    """Run the full host-input to host-output repair-only application path."""

    canonical = prepare_canonical_batch(attacked_rgb, mask)
    model_input = canonical.inputs.to(device=device, dtype=torch.float32)
    prediction = model(model_input)
    if prediction.dtype != torch.float32:
        raise TypeError("model output must be torch.float32")
    prediction_cpu = prediction.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return compose_predictions(attacked_rgb, mask, canonical, prediction_cpu)


__all__ = [
    "CanonicalBatch",
    "MaskComponent",
    "RepairPlan",
    "RepairROI",
    "canonicalize_repair_plan",
    "compose_predictions",
    "connected_components",
    "exact_compose_candidates",
    "inverse_resize_predictions",
    "plan_repair_geometry",
    "prepare_canonical_batch",
    "quantize_prediction",
    "repair_frame",
    "roi_for_component",
    "validate_mask",
]
