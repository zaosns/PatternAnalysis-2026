"""Check manifests and patient separation using temporary synthetic file pairs."""

from dataclasses import replace
import json
import os
from pathlib import Path
import random
import tempfile
import unittest

from dataset import load_manifest, split_cases_by_patient


class DatasetPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.entries = []
        # Unequal scan counts ensure splitting by scan would not be equivalent.
        for patient, scan_count in (("p1", 3), ("p2", 1), ("p3", 2), ("p4", 1)):
            for scan in range(scan_count):
                case = f"{patient}_scan{scan}"
                for role in ("image", "label"):
                    (self.root / f"{case}_{role}.nii.gz").write_bytes(b"synthetic placeholder")
                self.entries.append({
                    "case_id": case, "patient_id": patient,
                    "image_path": f"{case}_image.nii.gz", "label_path": f"{case}_label.nii.gz",
                })
        self.manifest = self.root / "manifest.json"

    def write_manifest(self, entries):
        self.manifest.write_text(json.dumps(entries), encoding="utf-8")
        return self.manifest

    def test_paths_resolve_relative_to_manifest(self):
        entries = [dict(entry) for entry in self.entries]
        entries[0]["image_path"] = str(self.root / entries[0]["image_path"])
        records = load_manifest(self.write_manifest(entries))
        self.assertEqual(len(records), 7)
        self.assertEqual(records[0].image_path, (self.root / entries[0]["image_path"]).resolve())
        self.assertTrue(all(case.image_path.is_absolute() and case.label_path.is_file() for case in records))

    def test_patient_groups_are_disjoint_complete_and_reproducible(self):
        cases = load_manifest(self.write_manifest(self.entries))
        state = random.getstate()
        split = split_cases_by_patient(cases, validation_patients=1, test_patients=1, seed=27)
        self.assertEqual(random.getstate(), state)
        reversed_split = split_cases_by_patient(tuple(reversed(cases)), validation_patients=1,
                                                test_patients=1, seed=27)
        self.assertEqual(split, reversed_split)
        groups = (split.train, split.validation, split.test)
        patient_sets = [{case.patient_id for case in group} for group in groups]
        self.assertEqual([len(ids) for ids in patient_sets], [2, 1, 1])
        for i in range(3):
            for j in range(i):
                self.assertFalse(patient_sets[i] & patient_sets[j])
        assigned_ids = [case.case_id for group in groups for case in group]
        self.assertCountEqual(assigned_ids, [case.case_id for case in cases])
        self.assertEqual(len(set(assigned_ids)), len(cases))
        for group in groups:
            self.assertEqual(list(group), sorted(group, key=lambda case: case.case_id))

    def test_zero_holdout_and_invalid_counts(self):
        cases = load_manifest(self.write_manifest(self.entries))
        split = split_cases_by_patient(cases, validation_patients=0, test_patients=0)
        self.assertEqual(len(split.train), len(cases))
        self.assertEqual(split.validation, ())
        self.assertEqual(split.test, ())
        for validation, test, seed in ((-1, 0, 0), (True, 0, 0), (0, 1.5, 0),
                                        (2, 2, 0), (0, 0, True), (0, 0, 2.5)):
            with self.subTest(validation=validation, test=test, seed=seed), self.assertRaises(ValueError):
                split_cases_by_patient(cases, validation_patients=validation, test_patients=test, seed=seed)

    def test_bad_manifest_records_are_rejected(self):
        first = self.entries[0]
        cases = [
            {}, [], [dict(first, unexpected="typo")],
            [{key: value for key, value in first.items() if key != "patient_id"}],
            [dict(first, patient_id=" ")], [dict(first, patient_id=" p1")],
            [dict(first, case_id=1)], [dict(first, image_path="")],
            [dict(first, label_path=first["image_path"])],
            [first, dict(self.entries[1], case_id=first["case_id"])],
            [first, dict(self.entries[1], image_path=first["image_path"])],
            [dict(first, image_path=".")],
        ]
        for entries in cases:
            with self.subTest(entries=entries), self.assertRaises(ValueError):
                load_manifest(self.write_manifest(entries))
        with self.assertRaises(FileNotFoundError):
            load_manifest(self.write_manifest([dict(first, image_path="missing.nii.gz")]))

    def test_file_aliases_and_direct_records_cannot_bypass_validation(self):
        first, second = self.entries[:2]
        original = self.root / first["image_path"]
        for kind in ("symlink", "hardlink"):
            alias = self.root / kind
            if kind == "symlink":
                alias.symlink_to(original)
            else:
                os.link(original, alias)
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "reused"):
                load_manifest(self.write_manifest([first, dict(second, image_path=str(alias))]))
        records = load_manifest(self.write_manifest(self.entries))
        duplicate = replace(records[1], label_path=records[0].label_path)
        with self.assertRaisesRegex(ValueError, "reused"):
            split_cases_by_patient((records[0], duplicate), validation_patients=0, test_patients=0)
        with self.assertRaisesRegex(ValueError, "patient_id"):
            split_cases_by_patient((replace(records[0], patient_id=""),),
                                   validation_patients=0, test_patients=0)


if __name__ == "__main__":
    unittest.main()
