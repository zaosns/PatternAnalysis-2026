"""Check frozen assignments and lazy training batches with tiny NIfTI volumes."""

import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import weakref

import nibabel as nib
import numpy as np
import torch

from dataset import (iter_case_batches, load_manifest, load_nifti_case,
                     load_patient_split, preprocess_case)


class TrainingDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        entries = []
        specifications = (("a0", "a", (6, 8, 10)), ("a1", "a", (7, 9, 11)),
                          ("b0", "b", (6, 8, 10)), ("c0", "c", (6, 8, 10)))
        for index, (case_id, patient_id, shape) in enumerate(specifications):
            entry = dict(case_id=case_id, patient_id=patient_id,
                         image_path=f"{case_id}_image.nii", label_path=f"{case_id}_label.nii")
            for role, values in (
                ("image", np.arange(1, np.prod(shape) + 1, dtype=np.float32).reshape(shape)),
                ("label", np.full(shape, index, dtype=np.int16)),
            ):
                image = nib.Nifti1Image(values, np.diag([2., 3., 4., 1.]))
                image.header.set_xyzt_units("mm")
                nib.save(image, self.root / entry[f"{role}_path"])
            entries.append(entry)
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(json.dumps(entries), encoding="utf-8")
        self.digest = hashlib.sha256(self.manifest.read_bytes()).hexdigest()
        self.records = load_manifest(self.manifest)
        self.path = self.root / "split.json"
        self.payload = dict(format_version=1, seed=3710, validation_patients=1,
                            test_patients=1, patient_key_rule="Explicit synthetic patient IDs",
                            manifest_sha256=self.digest, train=["a1", "a0"],
                            validation=["b0"], test=["c0"])

    def load_split(self, payload=None, digest=None):
        self.path.write_text(json.dumps(self.payload if payload is None else payload), encoding="utf-8")
        return load_patient_split(self.path, self.records,
                                  manifest_sha256=self.digest if digest is None else digest)

    def test_frozen_order_is_preserved_without_opening_images(self):
        state = random.getstate()
        with patch("nibabel.load", side_effect=AssertionError("No voxel or header access allowed")):
            split = self.load_split()
        self.assertEqual([case.case_id for case in split.train], ["a1", "a0"])
        self.assertEqual([case.case_id for case in split.validation], ["b0"])
        self.assertEqual([case.case_id for case in split.test], ["c0"])
        self.assertEqual(random.getstate(), state)

    def test_missing_duplicate_and_unknown_case_assignments_are_rejected(self):
        changes = (
            (dict(train=["a0"]), "missing case"),
            (dict(train=["a1", "a0", "a0"]), "Duplicate case"),
            (dict(validation=["b0", "a0"]), "Duplicate case"),
            (dict(train=["a1", "unknown"]), "Unknown case"),
        )
        for change, message in changes:
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, message):
                self.load_split(dict(self.payload, **change))

    def test_patient_overlap_and_wrong_holdout_counts_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Patient 'a'.*both"):
            self.load_split(dict(self.payload, train=["a0"], validation=["b0", "a1"]))
        for key in ("validation_patients", "test_patients"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "does not match"):
                self.load_split(dict(self.payload, **{key: 2}))

    def test_manifest_drift_and_malformed_digests_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.load_split(digest="0" * 64)
        for value in (None, True, "", "x" * 64, "A" * 64):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "SHA256"):
                self.load_split(dict(self.payload, manifest_sha256=value))
        self.path.write_text(json.dumps(self.payload), encoding="utf-8")
        for value in (None, True, "invalid"):
            with self.subTest(supplied=value), self.assertRaisesRegex(ValueError, "SHA256"):
                load_patient_split(self.path, self.records, manifest_sha256=value)

    def test_schema_types_and_required_partitions_are_checked(self):
        for key, values in (
            ("format_version", (True, 1.0, 2)), ("seed", (True, None, 2.5)),
            ("validation_patients", (True, -1, 1.0)), ("test_patients", (False, -1, None)),
            ("patient_key_rule", (None, "", " rule ")), ("train", ([], "a0", [4], [" a0"])),
            ("validation", ([], None)), ("test", (None, {"c0": 1})),
        ):
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.load_split(dict(self.payload, **{key: value}))
        for payload in ([], dict(self.payload, typo=1),
                        {key: value for key, value in self.payload.items() if key != "seed"}):
            with self.subTest(payload=payload), self.assertRaisesRegex(ValueError, "exactly"):
                self.load_split(payload)

    def test_test_partition_may_be_empty_when_count_is_zero(self):
        payload = dict(self.payload, train=["c0", "a0", "a1"], test=[], test_patients=0)
        split = self.load_split(payload)
        self.assertEqual([case.case_id for case in split.train], ["c0", "a0", "a1"])
        self.assertEqual(split.test, ())

    def test_batches_are_lazy_variable_sized_and_do_not_read_test_voxels(self):
        # A corrupt test file still passes metadata validation. Any attempt to
        # load its header/voxels would fail, even without the reader spy below.
        self.records[-1].image_path.write_bytes(b"Must remain untouched during training")
        split = self.load_split()
        with patch("dataset.load_nifti_case", wraps=load_nifti_case) as reader:
            batches = iter_case_batches(split.train)
            reader.assert_not_called()
            first_image, first_target = next(batches)
            self.assertEqual(reader.call_count, 1)
            self.assertEqual(tuple(first_image.shape), (1, 1, 6, 5, 4))
            self.assertEqual(tuple(first_target.shape), (1, 6, 5, 4))
            self.assertEqual(first_target.unique().tolist(), [1])
            second_image, second_target = next(batches)
            self.assertEqual(reader.call_count, 2)
            self.assertEqual(tuple(second_image.shape), (1, 1, 5, 4, 3))
            self.assertEqual(tuple(second_target.shape), (1, 5, 4, 3))
            with self.assertRaises(StopIteration):
                next(batches)
            self.assertEqual([call.args[0].case_id for call in reader.call_args_list], ["a1", "a0"])
        for image, target in ((first_image, first_target), (second_image, second_target)):
            self.assertEqual(image.device.type, "cpu")
            self.assertEqual(target.device.type, "cpu")
            self.assertEqual(image.dtype, torch.float32)
            self.assertEqual(target.dtype, torch.int64)
            self.assertTrue(torch.isfinite(image).all())

    def test_shuffling_is_repeatable_and_does_not_change_global_rng(self):
        original_order = tuple(self.records)
        def order(seed):
            return [target.flatten()[0].item() for _, target in iter_case_batches(self.records, seed=seed)]
        state = random.getstate()
        shuffled = order(3710)
        self.assertEqual(order(3710), shuffled)
        self.assertCountEqual(shuffled, [0, 1, 2, 3])
        self.assertNotEqual(order(72), shuffled)
        self.assertEqual(order(None), [0, 1, 2, 3])
        self.assertEqual(random.getstate(), state)
        self.assertEqual(self.records, original_order)

    def test_original_and_previous_preprocessed_volumes_are_released(self):
        original_references, prepared_references = [], []
        def read(record, **kwargs):
            # The consumer drops its previous batch before requesting another.
            self.assertTrue(all(reference() is None for reference in original_references))
            self.assertTrue(all(reference() is None for reference in prepared_references))
            loaded = load_nifti_case(record, **kwargs)
            original_references.extend((weakref.ref(loaded.image), weakref.ref(loaded.target)))
            return loaded
        def preprocess(loaded, **kwargs):
            prepared = preprocess_case(loaded, **kwargs)
            prepared_references.extend((weakref.ref(prepared.image), weakref.ref(prepared.target)))
            return prepared
        with patch("dataset.load_nifti_case", new=read), \
                patch("dataset.preprocess_case", new=preprocess):
            batches = iter_case_batches(self.records[:2])
            batch = next(batches)
            self.assertTrue(all(reference() is None for reference in original_references))
            del batch
            batch = next(batches)
            self.assertTrue(all(reference() is None for reference in original_references))
            del batch
            with self.assertRaises(StopIteration):
                next(batches)
        self.assertTrue(all(reference() is None for reference in prepared_references))

    def test_bad_options_fail_before_loading_any_case(self):
        for keyword, values in (("downsample_factor", (0, -1, True, 1.5)),
                                ("num_classes", (0, 1, True, 6.0)),
                                ("seed", (True, 1.5, "3710")),
                                ("assume_label_units_from_image", (0, "yes"))):
            for value in values:
                with self.subTest(keyword=keyword, value=value), \
                        patch("dataset.load_nifti_case") as reader, self.assertRaises(ValueError):
                    next(iter_case_batches(self.records, **{keyword: value}))
                reader.assert_not_called()
        for records in ([], ["case"]):
            with self.subTest(records=records), self.assertRaisesRegex(ValueError, "CaseRecord"):
                next(iter_case_batches(records))

    def test_reading_and_preprocessing_errors_include_case_id(self):
        broken = self.records[0]
        broken.label_path.write_bytes(b"Invalid NIfTI file")
        with self.assertRaisesRegex(RuntimeError, "Case 'a0'.*loaded or preprocessed") as captured:
            next(iter_case_batches((broken,)))
        self.assertIsNotNone(captured.exception.__cause__)
        with patch("dataset.preprocess_case", side_effect=ValueError("all-zero image")), \
                self.assertRaisesRegex(RuntimeError, "Case 'a1'.*all-zero image"):
            next(iter_case_batches((self.records[1],)))

    def test_explicit_unit_assumption_and_label_vocabulary_are_forwarded(self):
        case = self.records[-1]  # Constant class 3.
        # A header rewrite to the same file must not read through a mapping
        # that can be invalidated when the destination is truncated on Linux.
        label = nib.load(case.label_path, mmap=False)
        label.header.set_xyzt_units("unknown")
        nib.save(label, case.label_path)
        np.testing.assert_array_equal(
            np.asarray(nib.load(case.label_path, mmap=False).dataobj),
            np.full((6, 8, 10), 3, dtype=np.int16),
        )
        with self.assertRaisesRegex(RuntimeError, "Case 'c0'.*units"):
            next(iter_case_batches((case,)))
        _, target = next(iter_case_batches((case,), assume_label_units_from_image=True))
        self.assertEqual(target.unique().tolist(), [3])
        with self.assertRaisesRegex(RuntimeError, "Case 'c0'.*allowed_labels"):
            next(iter_case_batches((case,), assume_label_units_from_image=True, num_classes=3))


if __name__ == "__main__":
    unittest.main()
