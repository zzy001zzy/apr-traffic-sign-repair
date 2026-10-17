# SPDX-FileCopyrightText: 2026 Chih-Yuan Cheng and Bor-Jiunn Hwang
# SPDX-License-Identifier: MIT
"""Tensor-only weight loading and known-support-mask RGB repair."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors.torch import load_file
import torch

from .core.model_v4_1 import APRHardSelectRepair96
from .core.pipeline_v4_1 import repair_frame_v4_1
from .locator import ROIPresenceMaskCascade


def _load_verified_state(path: str | Path) -> dict[str, torch.Tensor]:
    path = Path(path)
    manifest = json.loads((path.parent / "manifest.json").read_text())
    matches = [item for item in manifest if item["file"] == path.name]
    if len(matches) != 1:
        raise ValueError("Weight file must have one entry in manifest.json")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != matches[0]["sha256"]:
        raise ValueError("Weight SHA-256 differs from the release manifest")
    return load_file(str(path), device="cpu")


def load_repair_model(path: str | Path) -> APRHardSelectRepair96:
    """Load the 27,188-parameter repair model on CPU in evaluation mode."""
    model = APRHardSelectRepair96()
    model.load_state_dict(_load_verified_state(path), strict=True)
    return model.eval()


def load_locator_model(path: str | Path) -> ROIPresenceMaskCascade:
    """Load the selected architecture for normalized 128x128 sign-ROI tensors."""
    model = ROIPresenceMaskCascade()
    model.load_state_dict(_load_verified_state(path), strict=True)
    return model.eval()


def repair_rgb(
    model: APRHardSelectRepair96,
    image: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    """Repair an RGB uint8 image using its aligned boolean support mask.

    The mask is provided by the caller. Gate means and component branch
    decisions describe the frozen repair selector.
    """
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise TypeError("image must be an RGB uint8 numpy array")
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
        raise ValueError("image must have nonempty HxWx3 shape")
    if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_:
        raise TypeError("mask must be a boolean numpy array")
    if mask.shape != image.shape[:2]:
        raise ValueError("mask must have the same HxW coordinates as image")
    source = torch.from_numpy(np.array(image, copy=True, order="C"))
    support = torch.from_numpy(np.array(mask, copy=True, order="C"))
    device = next(model.parameters()).device
    with torch.inference_mode():
        output, gate_means, selected = repair_frame_v4_1(
            model, source, support, device=device
        )
    return output.numpy().copy(), {
        "mask_provenance": "CALLER_PROVIDED_SUPPORT",
        "component_gate_means": gate_means,
        "component_refined_branch": selected,
        "outside_mask_byte_identity": bool(torch.equal(output[~support], source[~support])),
    }
