# SPDX-FileCopyrightText: 2026 Chih-Yuan Cheng and Bor-Jiunn Hwang
# SPDX-License-Identifier: MIT
"""Synthetic CPU contracts of the released model and geometry components.

These checks test software invariants, not traffic-sign detection performance.
"""

from dataclasses import replace
import unittest

import torch

from apr_repair.core.model import DSRes, KCRR96, parameter_count
from apr_repair.core.model_v2 import ContextGateRepair96
from apr_repair.core.model_v3 import APRGateRepair96, RESIDUAL_BOUND
from apr_repair.core.model_v4_1 import APRHardSelectRepair96
from apr_repair.core.pipeline import (
    canonicalize_repair_plan,
    compose_predictions,
    connected_components,
    exact_compose_candidates,
    inverse_resize_predictions,
    plan_repair_geometry,
    prepare_canonical_batch,
    quantize_prediction,
    repair_frame,
    validate_mask,
)
from apr_repair.core.pipeline_v2 import (
    canonicalize_context_gate_plan,
    prepare_clean_raw_batch,
    prepare_context_gate_batch,
    prepare_paired_context_gate_batches,
    repair_frame_v2,
)
from apr_repair.core.pipeline_v3 import (
    context_only_telea_proposal,
    prepare_apr_gate_batch,
    prepare_paired_apr_gate_batches,
    repair_frame_v3,
    telea_proposal,
)
from apr_repair.locator import (
    ROIUNetSmall,
    ROITemplateCorrelation,
    ROIPresenceMaskCascade,
    architecture_receipt,
    build_candidate,
)


class ReleasedComponentContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.image = torch.arange(64 * 80 * 3).remainder(256).to(torch.uint8).reshape(64, 80, 3)
        self.mask = torch.zeros((64, 80), dtype=torch.bool)
        self.mask[:7, :9] = True
        self.mask[40:47, 58:67] = True
        self.clean = self.image.clone()
        self.clean[self.mask] = 17

    def assert_preserves_context(self, output):
        self.assertEqual(output.dtype, torch.uint8)
        self.assertEqual(output.shape, self.image.shape)
        self.assertTrue(output.is_contiguous())
        self.assertTrue(torch.equal(output[~self.mask], self.image[~self.mask]))

    def test_historical_models_keep_frozen_sizes_and_output_ranges(self):
        v1 = KCRR96().eval()
        v2 = ContextGateRepair96().eval()
        v3 = APRGateRepair96().eval()
        self.assertEqual(parameter_count(v1), 90123)
        self.assertEqual(parameter_count(v2), 103756)
        self.assertEqual(parameter_count(v3), 27188)
        context = torch.zeros((1, 4, 96, 96))
        raw = torch.full((1, 3, 96, 96), 0.4)
        proposal = torch.full_like(raw, 0.6)
        mask = torch.zeros((1, 1, 96, 96), dtype=torch.bool)
        mask[:, :, 20:40, 30:50] = True
        with torch.inference_mode():
            output1 = v1(context)
            output2 = v2(context, raw, mask)
            output3 = v3(torch.zeros((1, 9, 96, 96)), raw, proposal, mask)
        self.assertEqual(output1.shape, raw.shape)
        self.assertTrue(bool(((output1 >= 0) & (output1 <= 1)).all()))
        for output in (output2, output3):
            self.assertTrue(torch.equal(output.repaired[~mask.expand_as(raw)], raw[~mask.expand_as(raw)]))
            self.assertTrue(bool((output.gate[~mask] == 0).all()))
            self.assertTrue(bool(((output.candidate >= 0) & (output.candidate <= 1)).all()))
        self.assertLessEqual(float(output3.residual.abs().max()), RESIDUAL_BOUND)

    def test_v1_input_validation_rejects_unsafe_shapes_and_nonfinite_values(self):
        model = KCRR96()
        cases = [(None, TypeError), (torch.zeros((1, 4, 96, 96), dtype=torch.float64), TypeError),
                 (torch.zeros((1, 3, 96, 96)), ValueError), (torch.zeros((0, 4, 96, 96)), ValueError),
                 (torch.full((1, 4, 96, 96), float("nan")), ValueError)]
        for value, error in cases:
            with self.subTest(error=error, value_type=type(value).__name__):
                with self.assertRaises(error):
                    model(value)
        with self.assertRaises(ValueError):
            DSRes(0)

    def test_v3_and_v4_models_reject_invalid_tensor_contracts(self):
        base = [torch.zeros((1, 9, 96, 96)), torch.zeros((1, 3, 96, 96)),
                torch.zeros((1, 3, 96, 96)), torch.ones((1, 1, 96, 96), dtype=torch.bool)]
        mutations = [(0, base[0].double(), TypeError), (0, torch.full_like(base[0], float("inf")), ValueError),
                     (3, base[3].float(), TypeError), (0, torch.zeros((1, 8, 96, 96)), ValueError),
                     (1, torch.zeros((1, 2, 96, 96)), ValueError),
                     (2, torch.zeros((1, 2, 96, 96)), ValueError),
                     (3, torch.ones((1, 2, 96, 96), dtype=torch.bool), ValueError)]
        for model in (APRGateRepair96(), APRHardSelectRepair96()):
            for index, invalid, error in mutations:
                args = base.copy()
                args[index] = invalid
                with self.subTest(model=type(model).__name__, argument=index, error=error):
                    with self.assertRaises(error):
                        model(*args)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            APRHardSelectRepair96()(*base[:3], torch.zeros_like(base[3]))

    def test_component_order_and_canonical_erasure_are_deterministic(self):
        plan = plan_repair_geometry(self.image, self.mask, clean_target_rgb=self.clean)
        self.assertEqual([item.bbox_xyxy for item in plan.components], [(0, 0, 9, 7), (58, 40, 67, 47)])
        self.assertLess(plan.rois[0].left, 0)
        timings = {}
        batch = prepare_canonical_batch(self.image, self.mask, clean_target_rgb=self.clean, _timing_ns=timings)
        self.assertEqual(batch.inputs.shape, (2, 4, 96, 96))
        self.assertEqual(batch.targets.shape, (2, 3, 96, 96))
        self.assertTrue(bool((batch.inputs[:, :3][batch.masks.expand(-1, 3, -1, -1)] == 0.5).all()))
        self.assertEqual(set(timings), {"roi_crop_ns", "mask_prepare_ns"})
        self.assertTrue(all(value >= 0 for value in timings.values()))

    def test_stale_geometry_and_misaligned_clean_targets_are_rejected(self):
        plan = plan_repair_geometry(self.image, self.mask)
        variants = [replace(plan, frame_shape=(80, 64, 3)), replace(plan, rois=()),
                    replace(plan, components=plan.components[:1], rois=plan.rois[:1]),
                    replace(plan, components=(replace(plan.components[0], index=99), plan.components[1])),
                    replace(plan, rois=(replace(plan.rois[0], left=99), plan.rois[1]))]
        for stale in variants:
            with self.subTest(plan=stale.frame_shape, component_count=len(stale.components)):
                with self.assertRaises(ValueError):
                    canonicalize_repair_plan(self.image, self.mask, stale)
        with self.assertRaises(ValueError):
            plan_repair_geometry(self.image, self.mask, clean_target_rgb=self.clean[:32].contiguous())
        with self.assertRaises(ValueError):
            canonicalize_repair_plan(self.image, self.mask, plan, clean_target_rgb=self.clean[:32].contiguous())

    def test_frame_and_mask_contracts_fail_before_geometry_processing(self):
        bad_frames = [(None, TypeError), (self.image.float(), TypeError),
                      (self.image[:, :, :1].contiguous(), ValueError),
                      (self.image[:0].contiguous(), ValueError), (self.image[:, ::2], ValueError)]
        for frame, error in bad_frames:
            with self.subTest(frame_type=type(frame).__name__, error=error):
                with self.assertRaises(error):
                    plan_repair_geometry(frame, self.mask)
        bad_masks = [(None, TypeError), (self.mask.float(), TypeError),
                     (self.mask[:32].contiguous(), ValueError), (self.mask.t(), ValueError),
                     (torch.zeros_like(self.mask), ValueError)]
        for mask, error in bad_masks:
            with self.subTest(mask_type=type(mask).__name__, error=error):
                with self.assertRaises(error):
                    validate_mask(mask, tuple(self.image.shape))
        for mask in (self.mask.float(), self.mask.t(), torch.zeros_like(self.mask)):
            with self.assertRaises(ValueError):
                connected_components(mask)

    def test_quantization_clamps_and_uses_ties_to_even(self):
        prediction = torch.tensor([-1.0, 0.5 / 255, 1.5 / 255, 2.5 / 255, 2.0]).reshape(1, 1, 5).expand(3, -1, -1).contiguous()
        output = quantize_prediction(prediction)
        self.assertEqual(output[:, :, 0].tolist(), [[0, 0, 2, 2, 255]])
        for invalid, error in [(prediction.double(), TypeError), (prediction[:2], ValueError),
                               (prediction[:, :, ::2], ValueError),
                               (torch.full_like(prediction, float("nan")), ValueError)]:
            with self.subTest(error=error):
                with self.assertRaises(error):
                    quantize_prediction(invalid)

    def test_exact_composition_changes_only_each_component(self):
        batch = prepare_canonical_batch(self.image, self.mask)
        predictions = torch.stack((torch.zeros((3, 96, 96)), torch.ones((3, 96, 96))))
        timing = {}
        output = compose_predictions(self.image, self.mask, batch, predictions, _timing_ns=timing)
        self.assert_preserves_context(output)
        self.assertTrue(bool((output[batch.components[0].mask] == 0).all()))
        self.assertTrue(bool((output[batch.components[1].mask] == 255).all()))
        self.assertEqual(set(timing), {"inverse_resize_ns", "exact_compose_ns"})
        candidates = inverse_resize_predictions(batch, predictions)
        with self.assertRaises(ValueError):
            exact_compose_candidates(self.image, self.mask, batch, candidates[:1])
        with self.assertRaises(ValueError):
            exact_compose_candidates(self.image, self.mask, batch, (candidates[0].float(), candidates[1]))
        for invalid, error in [(predictions.double(), TypeError), (predictions[:1], ValueError),
                               (predictions.transpose(2, 3), ValueError),
                               (torch.full_like(predictions, float("inf")), ValueError)]:
            with self.subTest(error=error):
                with self.assertRaises(error):
                    inverse_resize_predictions(batch, invalid)

    def test_historical_end_to_end_paths_preserve_unmasked_bytes(self):
        with torch.inference_mode():
            outputs = [repair_frame(KCRR96().eval(), self.image, self.mask, device="cpu"),
                       repair_frame_v2(ContextGateRepair96().eval(), self.image, self.mask, device="cpu"),
                       repair_frame_v3(APRGateRepair96().eval(), self.image, self.mask, device="cpu")]
        for output in outputs:
            self.assert_preserves_context(output)

    def test_paired_v2_preparation_retains_identical_crop_geometry(self):
        plan = plan_repair_geometry(self.image, self.mask)
        batch = prepare_context_gate_batch(self.image, self.mask, clean_target_rgb=self.clean)
        canonical = canonicalize_context_gate_plan(self.image, self.mask, plan, clean_target_rgb=self.clean)
        self.assertTrue(torch.equal(batch.context_inputs, canonical.context_inputs))
        self.assertTrue(torch.equal(batch.raw_inputs, canonical.raw_inputs))
        attacked, clean = prepare_paired_context_gate_batches(self.image, self.clean, self.mask)
        self.assertEqual(attacked.rois, clean.rois)
        self.assertTrue(torch.equal(attacked.context_inputs, clean.context_inputs))
        clean_raw = prepare_clean_raw_batch(self.clean, self.mask, plan)
        self.assertTrue(torch.equal(clean_raw, clean.raw_inputs))

    def test_telea_and_apr_prior_are_invariant_to_hidden_mask_bytes(self):
        prior = context_only_telea_proposal(self.image, self.mask)
        self.assertTrue(torch.equal(prior, context_only_telea_proposal(self.clean, self.mask)))
        self.assert_preserves_context(prior)
        self.assert_preserves_context(telea_proposal(self.image, self.mask))
        for operation in (telea_proposal, context_only_telea_proposal):
            self.assertTrue(torch.equal(operation(self.image, torch.zeros_like(self.mask)), self.image))
            with self.assertRaises(TypeError):
                operation(self.image.float(), self.mask)
            with self.assertRaises(TypeError):
                operation(self.image, self.mask.float())
        attacked, clean_raw = prepare_paired_apr_gate_batches(self.image, self.clean, self.mask)
        standalone = prepare_apr_gate_batch(self.image, self.mask, clean_target_rgb=self.clean)
        self.assertTrue(torch.equal(attacked.refiner_inputs, standalone.refiner_inputs))
        self.assertEqual(clean_raw.shape, attacked.raw_inputs.shape)
        self.assertEqual(attacked.targets.shape, attacked.proposal_inputs.shape)
        changed_context = self.clean.clone()
        changed_context[20, 20, 0] ^= 1
        with self.assertRaisesRegex(RuntimeError, "visible contexts"):
            prepare_paired_apr_gate_batches(self.image, changed_context, self.mask)

    def test_all_released_locator_candidates_obey_roi_geometry(self):
        template = torch.zeros((1, 3, 128, 128))
        for candidate in (ROIUNetSmall, ROITemplateCorrelation, ROIPresenceMaskCascade):
            model = build_candidate(candidate.candidate_id, template_rgb_normalized=template).eval()
            with torch.inference_mode():
                output = model(template)
            receipt = architecture_receipt(model)
            self.assertEqual(receipt.parameters, parameter_count(model))
            self.assertEqual(receipt.input_shape, (3, 128, 128))
            self.assertEqual(output.mask_logits.shape, (1, 1, 128, 128))
            self.assertEqual(output.presence_logit.shape, (1,))
            self.assertTrue(bool(torch.isfinite(output.mask_logits).all()))
            with self.assertRaises(ValueError):
                model(template[:, :, :64])
        for invalid in (template.double(), template[:, :, :64], torch.full_like(template, float("nan"))):
            with self.assertRaises(ValueError):
                ROITemplateCorrelation(invalid)
        with self.assertRaises(KeyError):
            build_candidate("UNKNOWN", template_rgb_normalized=template)


if __name__ == "__main__":
    unittest.main()
