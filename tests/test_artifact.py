# SPDX-FileCopyrightText: 2026 Chih-Yuan Cheng and Bor-Jiunn Hwang
# SPDX-License-Identifier: MIT
"""CPU engineering checks using synthetic inputs only."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from apr_repair import load_locator_model, load_repair_model, repair_rgb

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model = load_repair_model(WEIGHTS / "apr-hardselect96.safetensors")

    def test_frozen_repair_parameter_count(self):
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), 27188)

    def test_empty_support_preserves_every_byte(self):
        image = np.arange(96 * 112 * 3, dtype=np.uint8).reshape(96, 112, 3)
        output, receipt = repair_rgb(self.model, image, np.zeros(image.shape[:2], bool))
        np.testing.assert_array_equal(output, image)
        self.assertEqual(receipt["component_gate_means"], ())

    def test_disconnected_border_support_preserves_outside_pixels(self):
        image = np.arange(96 * 112 * 3, dtype=np.uint8).reshape(96, 112, 3)
        support = np.zeros(image.shape[:2], dtype=bool)
        support[:13, :11] = True
        support[61:76, 73:88] = True
        output, receipt = repair_rgb(self.model, image, support)
        self.assertEqual(output.dtype, np.uint8)
        self.assertEqual(output.shape, image.shape)
        np.testing.assert_array_equal(output[~support], image[~support])
        self.assertTrue(receipt["outside_mask_byte_identity"])
        self.assertEqual(len(receipt["component_gate_means"]), 2)

    def test_mask_dtype_and_coordinate_alignment_are_checked(self):
        image = np.zeros((32, 48, 3), dtype=np.uint8)
        with self.assertRaises(TypeError):
            repair_rgb(self.model, image, np.zeros((32, 48), dtype=np.uint8))
        with self.assertRaises(ValueError):
            repair_rgb(self.model, image, np.zeros((48, 32), dtype=bool))

    def test_weight_tampering_is_detected_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "manifest.json").write_text((WEIGHTS / "manifest.json").read_text())
            (path / "apr-hardselect96.safetensors").write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_repair_model(path / "apr-hardselect96.safetensors")

    def test_selected_locator_loads_and_returns_expected_tensor_shapes(self):
        locator = load_locator_model(WEIGHTS / "roi-presence-mask-cascade.safetensors")
        with torch.inference_mode():
            output = locator(torch.zeros((1, 3, 128, 128), dtype=torch.float32))
        self.assertEqual(tuple(output.mask_logits.shape), (1, 1, 128, 128))
        self.assertEqual(output.presence_logit.numel(), 1)
        self.assertTrue(bool(torch.isfinite(output.mask_logits).all()))


if __name__ == "__main__":
    unittest.main()
