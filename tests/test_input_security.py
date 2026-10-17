# SPDX-FileCopyrightText: 2026 Chih-Yuan Cheng and Bor-Jiunn Hwang
# SPDX-License-Identifier: MIT
"""Regression checks of public input and model-loading trust boundaries."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from safetensors import SafetensorError
import torch

from apr_repair import load_repair_model, repair_rgb

WEIGHTS = Path(__file__).resolve().parents[1] / "weights"


class InputSecurityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.model = load_repair_model(WEIGHTS / "apr-hardselect96.safetensors")

    def test_image_types_are_rejected_before_processing(self):
        mask = np.zeros((16, 24), dtype=bool)
        for image in [None, [], np.zeros((16, 24, 3), dtype=np.float32)]:
            with self.subTest(image_type=type(image).__name__):
                with self.assertRaises(TypeError):
                    repair_rgb(self.model, image, mask)

    def test_empty_and_non_rgb_dimensions_are_rejected(self):
        for shape in [(0, 24, 3), (16, 0, 3), (16, 24), (16, 24, 1), (16, 24, 4), (1, 16, 24, 3)]:
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError):
                    repair_rgb(self.model, np.zeros(shape, dtype=np.uint8), np.zeros((16, 24), bool))

    def test_non_boolean_mask_types_are_rejected(self):
        image = np.zeros((16, 24, 3), dtype=np.uint8)
        for mask in [None, [], np.zeros((16, 24), dtype=np.float32), np.zeros((16, 24), dtype=np.uint8)]:
            with self.subTest(mask_type=type(mask).__name__):
                with self.assertRaises(TypeError):
                    repair_rgb(self.model, image, mask)

    def test_missing_and_duplicate_manifest_entries_are_rejected(self):
        filename = "apr-hardselect96.safetensors"
        content = b"untrusted synthetic model bytes"
        entry = {"file": filename, "sha256": hashlib.sha256(content).hexdigest()}
        for entries in [[], [entry, entry]]:
            with self.subTest(entry_count=len(entries)):
                with tempfile.TemporaryDirectory() as folder:
                    root = Path(folder)
                    (root / filename).write_bytes(content)
                    (root / "manifest.json").write_text(json.dumps(entries))
                    with self.assertRaisesRegex(ValueError, "one entry"):
                        load_repair_model(root / filename)

    def test_matching_hash_does_not_make_invalid_safetensors_loadable(self):
        filename = "apr-hardselect96.safetensors"
        content = b"invalid tensor-only file"
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / filename).write_bytes(content)
            (root / "manifest.json").write_text(json.dumps([
                {"file": filename, "sha256": hashlib.sha256(content).hexdigest()}
            ]))
            with self.assertRaises(SafetensorError):
                load_repair_model(root / filename)

    def test_readonly_strided_inputs_and_return_do_not_alias(self):
        image = np.arange(96 * 112 * 3, dtype=np.uint8).reshape(96, 112, 3)[:, ::-1]
        mask = np.zeros(image.shape[:2], dtype=bool)
        mask[20:31, 22:37] = True
        image_before, mask_before = image.copy(), mask.copy()
        image.setflags(write=False)
        mask.setflags(write=False)
        result, receipt = repair_rgb(self.model, image, mask)
        np.testing.assert_array_equal(image, image_before)
        np.testing.assert_array_equal(mask, mask_before)
        np.testing.assert_array_equal(result[~mask], image_before[~mask])
        self.assertTrue(receipt["outside_mask_byte_identity"])
        result[:] = 0
        np.testing.assert_array_equal(image, image_before)


if __name__ == "__main__":
    unittest.main()
