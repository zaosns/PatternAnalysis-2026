"""Check normalization and resized voxel geometry without real MRI data."""

from dataclasses import replace
from itertools import product
import math
from pathlib import Path
import unittest

import torch

from dataset import CaseRecord, LoadedNiftiCase, PreprocessedCase, preprocess_case


class PreprocessingTests(unittest.TestCase):
    def make_case(self, image=None, target=None):
        if image is None:
            image = torch.arange(1., 61., dtype=torch.float32).reshape(1, 3, 4, 5)
        if target is None:
            target = torch.zeros(image.shape[1:], dtype=torch.int64)
        affine_dhw = torch.tensor([
            [2., .2, 0., 10.],
            [0., 3., .4, -7.],
            [0., 0., -4., 8.],
            [0., 0., 0., 1.],
        ], dtype=torch.float64)
        return LoadedNiftiCase(
            record=CaseRecord("synthetic_case", "synthetic_patient", Path("image.nii"), Path("label.nii")),
            image=image, target=target,
            affine_ijk=affine_dhw[:, [2, 1, 0, 3]].clone(), affine_dhw=affine_dhw,
            source_shape=tuple(image.shape[1:][::-1]), image_spatial_unit="mm",
            label_spatial_unit="unknown", spatial_unit="mm", label_unit_assumed=True,
            image_qform_code=0, image_sform_code=2, label_qform_code=0, label_sform_code=2,
        )

    def test_nonzero_population_zscore_matches_hand_calculation(self):
        sample = self.make_case(torch.tensor([[[[0., 2., 4., 6.]]]]))
        result = preprocess_case(sample, downsample_factor=1)
        self.assertIsInstance(result, PreprocessedCase)
        self.assertEqual(result.normalization_mean, 4.)
        self.assertAlmostEqual(result.normalization_std, math.sqrt(8. / 3.))
        self.assertEqual(result.normalization_scale, result.normalization_std)
        self.assertEqual(result.nonzero_count, 3)
        expected = torch.tensor([[[[0., -math.sqrt(1.5), 0., math.sqrt(1.5)]]]])
        torch.testing.assert_close(result.image, expected)
        self.assertEqual(result.image.dtype, torch.float32)
        self.assertEqual(result.target.dtype, torch.int64)
        self.assertEqual(result.image.device.type, "cpu")
        self.assertEqual(result.record, sample.record)
        self.assertEqual(result.spatial_unit, "mm")
        self.assertTrue(result.label_unit_assumed)

    def test_target_values_do_not_affect_intensity_processing(self):
        sample = self.make_case(torch.tensor([[[[0., -3., 0., 1., 5.]]]]))
        changed = replace(sample, target=torch.tensor([[[77, 0, 91, 2, 0]]], dtype=torch.int64))
        before = preprocess_case(sample)
        after = preprocess_case(changed)
        torch.testing.assert_close(before.image, after.image, rtol=0, atol=0)
        self.assertEqual(before.normalization_mean, after.normalization_mean)
        self.assertEqual(before.normalization_std, after.normalization_std)
        self.assertEqual(before.nonzero_count, 3)

    def test_constant_empty_and_small_variance_regions(self):
        sample = self.make_case(torch.tensor([[[[0., -2., -2., 0.]]]]))
        result = preprocess_case(sample)
        self.assertEqual(result.normalization_mean, -2.)
        self.assertEqual(result.normalization_std, 0.)
        self.assertEqual(result.normalization_scale, 1e-8)
        self.assertEqual(result.nonzero_count, 2)
        self.assertTrue(torch.equal(result.image, torch.zeros_like(result.image)))
        singleton = self.make_case(torch.tensor([[[[0., 7.]]]]))
        self.assertEqual(preprocess_case(singleton).normalization_std, 0.)
        with self.assertRaisesRegex(ValueError, "all-zero"):
            preprocess_case(replace(sample, image=torch.zeros_like(sample.image)))
        small = self.make_case(torch.tensor([[[[1e-10, 3e-10]]]]))
        small_result = preprocess_case(small, downsample_factor=1)
        self.assertEqual(small_result.normalization_scale, 1e-8)
        torch.testing.assert_close(small_result.image, torch.tensor([[[[-.01, .01]]]]))

    def test_even_and_odd_grids_preserve_world_coordinates_and_outer_edges(self):
        for source_shape, factor in (((4, 6, 8), 2), ((3, 5, 7), 2), ((5, 8, 11), 3)):
            with self.subTest(source_shape=source_shape, factor=factor):
                sample = self.make_case(torch.ones((1, *source_shape), dtype=torch.float32))
                result = preprocess_case(sample, downsample_factor=factor)
                output_shape = tuple(math.ceil(size / factor) for size in source_shape)
                self.assertEqual(tuple(result.image.shape), (1, *output_shape))
                self.assertEqual(tuple(result.target.shape), output_shape)
                self.assertEqual(result.source_shape_dhw, source_shape)
                torch.testing.assert_close(result.source_affine_dhw, sample.affine_dhw, rtol=0, atol=0)
                # The corners of the outer voxel faces delimit the same physical
                # volume before and after resizing, even with shear/reflection.
                for sides in product((0, 1), repeat=3):
                    source_corner = torch.tensor([
                        source_shape[axis] - .5 if side else -.5 for axis, side in enumerate(sides)
                    ] + [1.], dtype=torch.float64)
                    output_corner = torch.tensor([
                        output_shape[axis] - .5 if side else -.5 for axis, side in enumerate(sides)
                    ] + [1.], dtype=torch.float64)
                    torch.testing.assert_close(result.affine_dhw @ output_corner, sample.affine_dhw @ source_corner)
                for output_index in ((0, 0, 0), tuple(size - 1 for size in output_shape)):
                    source_index = torch.tensor([
                        (index + .5) * source_size / output_size - .5
                        for index, source_size, output_size in zip(output_index, source_shape, output_shape)
                    ] + [1.], dtype=torch.float64)
                    output_point = torch.tensor([*output_index, 1.], dtype=torch.float64)
                    torch.testing.assert_close(result.affine_dhw @ output_point, sample.affine_dhw @ source_index)

    def test_trilinear_samples_half_pixel_centres_and_can_mix_background(self):
        d, h, w = torch.meshgrid(torch.arange(3.), torch.arange(5.), torch.arange(7.), indexing="ij")
        ramp = (1. + 2. * d + 3. * h + 5. * w).unsqueeze(0)
        result = preprocess_case(self.make_case(ramp))
        for indices in product(range(2), range(3), range(4)):
            point = tuple((index + .5) * n / m - .5 for index, n, m in zip(indices, (3, 5, 7), (2, 3, 4)))
            intensity = 1. + 2. * point[0] + 3. * point[1] + 5. * point[2]
            expected = (intensity - result.normalization_mean) / result.normalization_scale
            self.assertAlmostEqual(result.image[(0, *indices)].item(), expected, places=6)
        boundary = preprocess_case(self.make_case(torch.tensor([[[[0., 2., 0., 6.]]]])))
        torch.testing.assert_close(boundary.image, torch.tensor([[[[-.5, .5]]]]))

    def test_nearest_labels_preserve_large_integers_and_choose_upper_at_ties(self):
        labels = (torch.arange(4 * 5 * 6, dtype=torch.int64) + 2**54 + 1).reshape(4, 5, 6)
        sample = self.make_case(torch.ones((1, 4, 5, 6)), labels)
        result = preprocess_case(sample)
        expected = labels[[1, 3]][:, [0, 2, 4]][:, :, [1, 3, 5]]
        torch.testing.assert_close(result.target, expected, rtol=0, atol=0)
        self.assertEqual(result.target.dtype, torch.int64)
        tie = self.make_case(torch.tensor([[[[1., 2.]]]]), torch.tensor([[[2**63 - 2, 2**63 - 1]]]))
        self.assertEqual(preprocess_case(tie).target.item(), 2**63 - 1)
        # A factor exceeding every dimension still yields one valid voxel.
        self.assertEqual(tuple(preprocess_case(sample, downsample_factor=100).target.shape), (1, 1, 1))

    def test_factor_one_and_two_leave_inputs_unchanged_without_shared_storage(self):
        for factor in (1, 2):
            with self.subTest(factor=factor):
                sample = self.make_case()
                sample.image.requires_grad_(True)
                sample.affine_dhw.requires_grad_(True)
                originals = {name: getattr(sample, name).detach().clone() for name in ("image", "target", "affine_dhw", "affine_ijk")}
                result = preprocess_case(sample, downsample_factor=factor)
                self.assertEqual(result.downsample_factor, factor)
                for name in ("image", "target", "affine_dhw", "source_affine_dhw"):
                    tensor = getattr(result, name)
                    source = sample.affine_dhw if name == "source_affine_dhw" else getattr(sample, name)
                    self.assertNotEqual(tensor.data_ptr(), source.data_ptr())
                    self.assertFalse(tensor.requires_grad)
                    self.assertIsNone(tensor.grad_fn)
                    tensor.fill_(99)
                for name, original in originals.items():
                    torch.testing.assert_close(getattr(sample, name), original, rtol=0, atol=0)

    def test_invalid_tensor_values_shapes_and_factor_are_rejected(self):
        sample = self.make_case()
        for factor in (True, False, 0, -1, 1.5, "2", None):
            with self.subTest(factor=factor), self.assertRaises(ValueError):
                preprocess_case(sample, downsample_factor=factor)
        invalid = [
            None,
            replace(sample, image=sample.image.double()),
            replace(sample, image=sample.image[0]),
            replace(sample, image=sample.image.expand(2, -1, -1, -1)),
            replace(sample, image=torch.full_like(sample.image, float("nan"))),
            replace(sample, image=torch.full_like(sample.image, float("inf"))),
            replace(sample, target=sample.target.float()),
            replace(sample, target=sample.target.unsqueeze(0)),
            replace(sample, target=sample.target[:, :, :-1]),
            replace(sample, target=torch.full_like(sample.target, -1)),
            replace(sample, image=sample.image[:, :0], target=sample.target[:0]),
            replace(sample, image=torch.empty(sample.image.shape, device="meta")),
            replace(sample, target=torch.empty(sample.target.shape, device="meta", dtype=torch.int64)),
        ]
        for index, invalid_sample in enumerate(invalid):
            with self.subTest(index=index), self.assertRaises(ValueError):
                preprocess_case(invalid_sample)

    def test_inconsistent_geometry_and_unit_metadata_are_rejected(self):
        sample = self.make_case()
        singular = sample.affine_dhw.clone()
        singular[0, :3] = 0.
        not_homogeneous = sample.affine_dhw.clone()
        not_homogeneous[3, 0] = 1.
        invalid = [
            replace(sample, record=None),
            replace(sample, source_shape=(5, 4, 4)),
            replace(sample, source_shape=(5, 4, True)),
            replace(sample, source_shape=[5, 4, 3]),
            replace(sample, affine_dhw=sample.affine_dhw.float()),
            replace(sample, affine_ijk=torch.eye(4, dtype=torch.float64)),
            replace(sample, affine_dhw=singular),
            replace(sample, affine_dhw=not_homogeneous),
            replace(sample, affine_dhw=torch.full((4, 4), float("nan"), dtype=torch.float64)),
            replace(sample, spatial_unit="unknown"),
            replace(sample, image_spatial_unit="meter"),
            replace(sample, label_unit_assumed=1),
            replace(sample, label_unit_assumed=False),
            replace(sample, label_spatial_unit="mm"),
            replace(sample, image_sform_code=0),
            replace(sample, label_qform_code=-1),
            replace(sample, label_sform_code=True),
        ]
        for index, invalid_sample in enumerate(invalid):
            with self.subTest(index=index), self.assertRaises(ValueError):
                preprocess_case(invalid_sample)
        explicit_units = replace(sample, label_spatial_unit="mm", label_unit_assumed=False)
        self.assertFalse(preprocess_case(explicit_units).label_unit_assumed)


if __name__ == "__main__":
    unittest.main()
