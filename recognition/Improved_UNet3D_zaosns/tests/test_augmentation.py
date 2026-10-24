"""Check opt-in training transforms on asymmetric synthetic prepared volumes."""

from dataclasses import replace
import json
from pathlib import Path
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch

from dataset import (CaseRecord, PreprocessedCase, iter_case_batches,
                     training_augmentation_policy)


class TrainingAugmentationTests(unittest.TestCase):
    def setUp(self):
        target = torch.arange(36, dtype=torch.int64).reshape(2, 3, 6) % 6
        affine = torch.tensor([[2., .1, 0., 4.], [0., 3., .2, 5.],
                               [0., 0., 4., 6.], [0., 0., 0., 1.]], dtype=torch.float64)
        self.prepared = PreprocessedCase(
            record=CaseRecord("case-a", "patient-a", Path("image.nii"), Path("label.nii")),
            image=target.to(torch.float32).unsqueeze(0), target=target,
            affine_dhw=affine, source_shape_dhw=(4, 6, 12),
            source_affine_dhw=affine.clone(), spatial_unit="mm", label_unit_assumed=False,
            normalization_mean=12., normalization_std=3., normalization_scale=3.,
            nonzero_count=30, downsample_factor=2,
        )

    def batches(self, prepared=None, **options):
        cases = [self.prepared] if prepared is None else list(prepared)
        by_id = {case.record.case_id: case for case in cases}
        # Loading and preprocessing are tested separately with actual NIfTI files.
        # Keep these fixtures asymmetric and exactly measurable after preprocessing.
        with patch("dataset.load_nifti_case", side_effect=lambda record, **_: record) as reader, \
                patch("dataset.preprocess_case", side_effect=lambda record, **_: by_id[record.case_id]):
            result = list(iter_case_batches([case.record for case in cases], **options))
        return {call.args[0].case_id: batch for call, batch in zip(reader.call_args_list, result)}

    def test_paired_flip_preserves_alignment_vocabulary_and_zero_values(self):
        directions = set()
        for seed in range(12):
            with self.subTest(seed=seed):
                image, target = self.batches(training=True, augment=True, seed=seed)["case-a"]
                original = self.prepared.target.unsqueeze(0)
                if torch.equal(target, original):
                    directions.add("unchanged")
                else:
                    self.assertTrue(torch.equal(target, original.flip(-1)))
                    directions.add("flipped")
                self.assertEqual(target.unique().tolist(), [0, 1, 2, 3, 4, 5])
                self.assertEqual(target.dtype, torch.int64)
                self.assertEqual(image.dtype, torch.float32)
                self.assertEqual(image.shape, (1, 1, 2, 3, 6))
                self.assertEqual(target.shape, (1, 2, 3, 6))
                foreground = target != 0
                factors = image[:, 0][foreground] / target[foreground]
                scale = factors[0].item()
                self.assertGreaterEqual(scale, 0.9)
                self.assertLessEqual(scale, 1.1)
                torch.testing.assert_close(factors, torch.full_like(factors, scale))
                self.assertTrue(torch.equal(image[:, 0][~foreground], torch.zeros_like(factors[:6])))
        self.assertEqual(directions, {"unchanged", "flipped"})

    def test_positive_scale_preserves_intensity_signs(self):
        prepared = replace(self.prepared, image=self.prepared.image - 2.)
        for seed in range(5):
            image, target = self.batches([prepared], training=True, augment=True, seed=seed)["case-a"]
            expected_sign = torch.sign(target.to(torch.float32) - 2.)
            self.assertTrue(torch.equal(torch.sign(image[:, 0]), expected_sign))

    def test_same_seed_and_case_are_repeatable_independent_of_order(self):
        second = replace(self.prepared, record=replace(self.prepared.record, case_id="case-b"))
        initial = self.batches([self.prepared, second], training=True, augment=True, seed=3710)
        repeated = self.batches([self.prepared, second], training=True, augment=True, seed=3710)
        reordered = self.batches([second, self.prepared], training=True, augment=True, seed=3710)
        standalone = self.batches(training=True, augment=True, seed=3710)
        for case_id in initial:
            for expected, actual, swapped in zip(initial[case_id], repeated[case_id], reordered[case_id]):
                self.assertTrue(torch.equal(expected, actual))
                self.assertTrue(torch.equal(expected, swapped))
        for expected, actual in zip(initial["case-a"], standalone["case-a"]):
            self.assertTrue(torch.equal(expected, actual))
        self.assertFalse(torch.equal(initial["case-a"][0], initial["case-b"][0]))
        another_epoch = self.batches(training=True, augment=True, seed=3711)
        self.assertFalse(torch.equal(standalone["case-a"][0], another_epoch["case-a"][0]))

    def test_does_not_change_python_numpy_or_torch_global_rng(self):
        python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        self.batches(training=True, augment=True, seed=3710)
        self.assertEqual(random.getstate(), python_state)
        current_numpy = np.random.get_state()
        self.assertEqual(current_numpy[0], numpy_state[0])
        np.testing.assert_array_equal(current_numpy[1], numpy_state[1])
        self.assertEqual(current_numpy[2:], numpy_state[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))

    def test_prepared_tensors_and_metadata_are_not_modified_or_shared(self):
        names = ("image", "target", "affine_dhw", "source_affine_dhw")
        originals = {name: getattr(self.prepared, name).clone() for name in names}
        for seed in range(5):
            image, target = self.batches(training=True, augment=True, seed=seed)["case-a"]
            image.fill_(99.)
            target.fill_(99)
            for name in names:
                self.assertTrue(torch.equal(getattr(self.prepared, name), originals[name]))
        self.assertEqual(self.prepared.source_shape_dhw, (4, 6, 12))
        self.assertEqual(self.prepared.normalization_mean, 12.)
        self.assertEqual(self.prepared.normalization_std, 3.)
        self.assertEqual(self.prepared.spatial_unit, "mm")

    def test_defaults_and_disabled_training_augmentation_are_exactly_unchanged(self):
        for options in ({}, {"seed": 3710}, {"training": False, "augment": False},
                        {"training": True, "augment": False, "seed": 3710}):
            with self.subTest(options=options):
                image, target = self.batches(**options)["case-a"]
                self.assertTrue(torch.equal(image, self.prepared.image.unsqueeze(0)))
                self.assertTrue(torch.equal(target, self.prepared.target.unsqueeze(0)))

    def test_validation_augmentation_and_missing_seed_rejected_before_loading(self):
        for options in ({"augment": True, "seed": 3710},
                        {"training": False, "augment": True, "seed": 3710},
                        {"training": True, "augment": True},
                        {"training": 1}, {"augment": "yes"},
                        {"training": True, "augment": True, "seed": True}):
            with self.subTest(options=options), patch("dataset.load_nifti_case") as reader:
                with self.assertRaises(ValueError):
                    next(iter_case_batches([self.prepared.record], **options))
                reader.assert_not_called()

    def test_policy_is_versioned_json_and_returns_independent_values(self):
        first = training_augmentation_policy()
        self.assertEqual(first["version"], 1)
        self.assertEqual(json.loads(json.dumps(first)), first)
        first["spatial_flip"]["probability"] = 0.
        self.assertEqual(training_augmentation_policy()["spatial_flip"]["probability"], 0.5)


if __name__ == "__main__":
    unittest.main()
