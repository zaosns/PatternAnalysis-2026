"""Read small synthetic NIfTI files without relying on private MRI data."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch

from dataset import CaseRecord, load_nifti_case


class NiftiLoadingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.affine = np.array([
            [-2.0, 0.0, 0.0, 40.0],
            [0.0, -3.0, 0.0, 50.0],
            [0.0, 0.0, 4.0, 60.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        i, j, k = np.indices((2, 3, 4))
        self.image = (100 * i + 10 * j + k).astype(np.float32)
        self.label = ((i + j + k) % 6).astype(np.int16)
        self.record = CaseRecord("synthetic_case", "synthetic_patient",
                                 self.root / "image.nii", self.root / "label.nii")
        self.save("image", self.image)
        self.save("label", self.label)

    def save(self, role, values, *, unit="mm", affine=None, scaling=None):
        volume = nib.Nifti1Image(values, self.affine if affine is None else affine,
                                dtype=values.dtype)
        volume.header.set_xyzt_units(unit)
        if scaling is not None:
            volume.header.set_slope_inter(*scaling)
        nib.save(volume, getattr(self.record, f"{role}_path"))

    def patch_header(self, role, **updates):
        # Modify on-disk metadata directly; saving a Nifti1Image again can reset
        # its header to agree with the image object's cached affine.
        with getattr(self.record, f"{role}_path").open("r+b") as stream:
            header = nib.Nifti1Header.from_fileobj(stream)
            for name, value in updates.items():
                header[name] = value
            stream.seek(0)
            header.write_to(stream)

    def load(self, **kwargs):
        return load_nifti_case(self.record, allowed_labels=range(6), **kwargs)

    def test_native_values_axes_and_both_world_coordinate_transforms(self):
        loaded = self.load()
        self.assertEqual(loaded.source_shape, (2, 3, 4))
        self.assertEqual(tuple(loaded.image.shape), (1, 4, 3, 2))
        self.assertEqual(tuple(loaded.target.shape), (4, 3, 2))
        self.assertEqual(loaded.image.dtype, torch.float32)
        self.assertEqual(loaded.target.dtype, torch.int64)
        self.assertEqual(loaded.affine_ijk.dtype, torch.float64)
        self.assertEqual(loaded.affine_dhw.dtype, torch.float64)
        self.assertEqual(loaded.image.device.type, "cpu")
        self.assertTrue(loaded.image.is_contiguous() and loaded.target.is_contiguous())
        self.assertEqual(loaded.spatial_unit, "mm")
        self.assertFalse(loaded.label_unit_assumed)
        self.assertGreater(loaded.image_sform_code, 0)
        for i, j, k in np.ndindex(self.image.shape):
            self.assertEqual(loaded.image[0, k, j, i].item(), self.image[i, j, k])
            self.assertEqual(loaded.target[k, j, i].item(), self.label[i, j, k])
            source_point = torch.tensor([i, j, k, 1], dtype=torch.float64)
            tensor_point = torch.tensor([k, j, i, 1], dtype=torch.float64)
            expected = torch.from_numpy(self.affine @ np.array([i, j, k, 1]))
            torch.testing.assert_close(loaded.affine_ijk @ source_point, expected, rtol=0, atol=0)
            torch.testing.assert_close(loaded.affine_dhw @ tensor_point, expected, rtol=0, atol=0)

    def test_scaled_intensities_and_labels_are_used_before_conversion(self):
        self.save("image", self.image.astype(np.int16), scaling=(2.0, -7.0))
        self.save("label", self.label, scaling=(2.0, 1.0))
        loaded = load_nifti_case(self.record, allowed_labels=(1, 3, 5, 7, 9, 11))
        expected_image = torch.from_numpy((2 * self.image - 7).transpose(2, 1, 0).copy())
        expected_label = torch.from_numpy((2 * self.label + 1).transpose(2, 1, 0).astype(np.int64))
        torch.testing.assert_close(loaded.image[0], expected_image, rtol=0, atol=0)
        torch.testing.assert_close(loaded.target, expected_label, rtol=0, atol=0)
        self.save("label", self.label, scaling=(0.5, 0.0))
        with self.assertRaisesRegex(ValueError, "integers"):
            self.load()

    def test_unit_assumption_is_explicit_limited_and_recorded(self):
        self.save("label", self.label, unit="unknown")
        with self.assertRaisesRegex(ValueError, "units"):
            self.load()
        loaded = self.load(assume_label_units_from_image=True)
        self.assertEqual(loaded.image_spatial_unit, "mm")
        self.assertEqual(loaded.label_spatial_unit, "unknown")
        self.assertEqual(loaded.spatial_unit, "mm")
        self.assertTrue(loaded.label_unit_assumed)
        self.save("label", self.label, unit="meter")
        with self.assertRaisesRegex(ValueError, "units"):
            self.load(assume_label_units_from_image=True)
        self.save("image", self.image, unit="unknown")
        self.save("label", self.label, unit="unknown")
        with self.assertRaisesRegex(ValueError, "Image spatial units"):
            self.load(assume_label_units_from_image=True)
        self.save("image", self.image, unit="meter")
        self.save("label", self.label, unit="meter")
        loaded = self.load()
        self.assertEqual(loaded.spatial_unit, "meter")
        torch.testing.assert_close(loaded.affine_ijk, torch.from_numpy(self.affine), rtol=0, atol=0)

    def test_native_grid_mismatch_and_non_3d_volumes_are_rejected(self):
        self.save("label", self.label[:, :, :3])
        with self.assertRaisesRegex(ValueError, "shapes"):
            self.load()
        shifted = self.affine.copy()
        shifted[0, 3] = 100_000.0
        self.save("image", self.image, affine=shifted)
        shifted[0, 3] += 0.125
        self.save("label", self.label, affine=shifted)
        with self.assertRaisesRegex(ValueError, "affines"):
            self.load()
        self.save("image", self.image)
        self.save("label", self.label[..., None])
        with self.assertRaisesRegex(ValueError, "3D"):
            self.load()

    def test_missing_and_invalid_geometry_are_rejected(self):
        modifications = (
            ({"qform_code": 0, "sform_code": 0}, "coded"),
            ({"srow_x": [np.nan, 0, 0, 40]}, "finite"),
            ({"srow_x": [0, 0, 0, 40]}, "nonsingular"),
        )
        for changes, message in modifications:
            with self.subTest(changes=changes):
                self.save("label", self.label)
                self.patch_header("label", **changes)
                with self.assertRaisesRegex(ValueError, message):
                    self.load()
        # A coded qform alone is sufficient; do not require a coded sform.
        self.save("label", self.label)
        label = nib.load(self.record.label_path, mmap=False)
        label.set_qform(self.affine, code=1)
        label.set_sform(self.affine, code=0)
        nib.save(label, self.record.label_path)
        np.testing.assert_array_equal(
            np.asarray(nib.load(self.record.label_path, mmap=False).dataobj), self.label,
        )
        self.assertEqual(self.load().label_qform_code, 1)

    def test_label_vocabulary_fractional_and_nonfinite_values(self):
        self.save("label", np.zeros_like(self.label))
        self.assertEqual(self.load().target.unique().tolist(), [0])
        for value, message in ((6.0, "allowed_labels"), (-1.0, "allowed_labels"),
                               (0.25, "integers"), (np.nan, "finite"), (np.inf, "finite")):
            with self.subTest(value=value):
                label = self.label.astype(np.float64)
                label[0, 0, 0] = value
                self.save("label", label)
                with self.assertRaisesRegex(ValueError, message):
                    self.load()

    def test_label_integer_bounds_are_checked_without_float32_rounding(self):
        # This value is rounded if labels are prematurely read as float32.
        precise = 2**24 + 1
        self.save("label", np.full(self.label.shape, precise, dtype=np.int64))
        loaded = load_nifti_case(self.record, allowed_labels=(precise,))
        self.assertEqual(loaded.target.unique().tolist(), [precise])
        for values in (np.full(self.label.shape, 2**63, dtype=np.uint64),
                       np.full(self.label.shape, float(2**63), dtype=np.float64)):
            with self.subTest(dtype=values.dtype):
                self.save("label", values)
                with self.assertRaisesRegex(ValueError, "allowed_labels"):
                    self.load()

    def test_complex_and_nonfinite_or_float32_overflow_images_are_rejected(self):
        for values, message in (
            (self.image.astype(np.complex64) + 1j, "real"),
            (np.full(self.image.shape, np.nan), "finite"),
            (np.full(self.image.shape, np.inf), "finite"),
            (np.full(self.image.shape, 1e100), "finite"),
        ):
            with self.subTest(dtype=values.dtype, value=values.flat[0]):
                self.save("image", values)
                with self.assertRaisesRegex(ValueError, message):
                    self.load()
        self.save("image", self.image)
        self.save("label", self.label.astype(np.complex64))
        with self.assertRaisesRegex(ValueError, "real"):
            self.load()

    def test_voxel_limit_precedes_voxel_reads_and_options_are_validated(self):
        with patch.object(nib.arrayproxy.ArrayProxy, "__array__",
                          side_effect=AssertionError("Voxel data must not be read")):
            with self.assertRaisesRegex(ValueError, "max_voxels"):
                self.load(max_voxels=23)
        for limit in (0, -1, True, 2.5):
            with self.subTest(limit=limit), self.assertRaisesRegex(ValueError, "max_voxels"):
                self.load(max_voxels=limit)
        for labels in ((), (True,), (-1,), (2**63,), (0.5,), None):
            with self.subTest(labels=labels), self.assertRaisesRegex(ValueError, "allowed_labels"):
                load_nifti_case(self.record, allowed_labels=labels)
        with self.assertRaisesRegex(ValueError, "boolean"):
            self.load(assume_label_units_from_image=1)


if __name__ == "__main__":
    unittest.main()
