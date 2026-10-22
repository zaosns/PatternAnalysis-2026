"""Synthetic checks for frozen paired evaluation and held-out test gating."""

from contextlib import redirect_stdout
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch

import evaluate
import train
import train_baseline2d


class FrozenEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.use_deterministic_algorithms,
                        torch.are_deterministic_algorithms_enabled(),
                        warn_only=torch.is_deterministic_algorithms_warn_only_enabled())
        self.addCleanup(setattr, torch.backends.cudnn, "deterministic",
                        torch.backends.cudnn.deterministic)
        self.addCleanup(setattr, torch.backends.cudnn, "benchmark",
                        torch.backends.cudnn.benchmark)
        self.addCleanup(torch.set_float32_matmul_precision, torch.get_float32_matmul_precision())
        self.addCleanup(setattr, torch.backends.cuda.matmul, "allow_tf32",
                        torch.backends.cuda.matmul.allow_tf32)
        self.addCleanup(setattr, torch.backends.cudnn, "allow_tf32",
                        torch.backends.cudnn.allow_tf32)
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        self.addCleanup(random.setstate, random.getstate())
        self.addCleanup(np.random.set_state, np.random.get_state())
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
        rng = torch.random.fork_rng(devices=devices)
        rng.__enter__()
        self.addCleanup(rng.__exit__, None, None, None)
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manifest = self.root / "manifest.json"
        self.split = self.root / "split.json"
        self.validation_ids = ["validation_a", "validation_b", "validation_c"]
        entries = []
        patients = ["train_patient", "repeated_patient", "repeated_patient",
                    "single_patient", "test_patient"]
        for index, (case_id, patient_id) in enumerate(zip(
                ["train", *self.validation_ids, "test"], patients)):
            coordinates = np.indices((9 + index, 8, 7 + index))
            image = (coordinates[0] * 3 + coordinates[1] * 2
                     + coordinates[2] + 1).astype(np.float32)
            labels = ((coordinates[0] // 2 + coordinates[1] // 2
                       + coordinates[2] // 2) % 6).astype(np.int16)
            image_path, label_path = (self.root / f"{case_id}_{kind}.nii"
                                      for kind in ("image", "label"))
            for values, path in ((image, image_path), (labels, label_path)):
                volume = nib.Nifti1Image(values, np.diag([2., 3., 4., 1.]))
                volume.header.set_xyzt_units("mm")
                nib.save(volume, path)
            entries.append(dict(case_id=case_id, patient_id=patient_id,
                                image_path=image_path.name, label_path=label_path.name))
        self.manifest.write_text(json.dumps(entries), encoding="utf-8")
        self.split.write_text(json.dumps(dict(
            format_version=1, seed=3710, validation_patients=2, test_patients=1,
            patient_key_rule="Synthetic patients; two validation scans share one patient",
            manifest_sha256=self.sha256(self.manifest), train=["train"],
            validation=self.validation_ids, test=["test"],
        )), encoding="utf-8")
        self.run_dirs = {kind: self.root / f"training_{kind}" for kind in ("2d", "3d")}
        self.states = {}

    @staticmethod
    def sha256(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def train_fixture(self):
        for kind, trainer in (("2d", train_baseline2d), ("3d", train)):
            args = trainer.build_parser().parse_args([
                "--manifest", str(self.manifest), "--split", str(self.split),
                "--output-dir", str(self.run_dirs[kind]), "--epochs", "1",
                "--device", "cpu", "--num-classes", "6", "--channels", "2", "3",
                "--learning-rate", "0.002", "--downsample-factor", "2",
                "--threads", "1", "--seed", "3710", "--deterministic",
            ])
            with redirect_stdout(io.StringIO()):
                self.states[kind] = trainer.run_training(args)

    def write_plan(self, plan, name="plan.json"):
        path = self.root / name
        path.write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return path, self.sha256(path)

    def freeze(self, *, partition="validation", validation_report=None, name="plan.json"):
        plan = evaluate.build_plan(self.run_dirs, self.manifest, self.split,
                                   partition=partition, device="cpu", threads=1,
                                   validation_report=validation_report)
        path, digest = self.write_plan(plan, name)
        return path, digest, plan

    @staticmethod
    def arguments(path, digest, output):
        return SimpleNamespace(plan=path, plan_sha256=digest, output_dir=output)

    def run_plan(self, path, digest, output):
        with redirect_stdout(io.StringIO()):
            return evaluate.run_evaluation(self.arguments(path, digest, output))

    def assert_rejected_before_voxels(self, path, digest, output):
        with patch.object(evaluate, "load_nifti_case") as loader, \
                patch.object(nib, "load") as nifti_loader, self.assertRaises(ValueError):
            self.run_plan(path, digest, output)
        loader.assert_not_called()
        nifti_loader.assert_not_called()
        self.assertFalse(output.exists())

    def test_freezing_binds_the_full_cohort_without_reading_any_nifti_bytes(self):
        self.train_fixture()
        read_bytes = Path.read_bytes

        def metadata_only(path):
            self.assertNotEqual(path.suffix, ".nii", "Freezing must not read any voxel file")
            return read_bytes(path)

        with patch.object(evaluate, "load_nifti_case") as loader, \
                patch.object(nib, "load") as nifti_loader, \
                patch.object(Path, "read_bytes", new=metadata_only):
            path, digest, plan = self.freeze()
        loader.assert_not_called()
        nifti_loader.assert_not_called()
        self.assertEqual(digest, self.sha256(path))
        self.assertEqual(plan["partition"], "validation")
        self.assertEqual([row["case_id"] for row in plan["cohort"]], self.validation_ids)
        self.assertEqual(set(plan["models"]), {"2d", "3d"})
        for kind, spec in plan["models"].items():
            self.assertEqual(spec["checkpoint_name"], self.states[kind]["best_checkpoint"])
            self.assertEqual(spec["checkpoint_sha256"], self.sha256(
                self.run_dirs[kind] / self.states[kind]["best_checkpoint"]))

    def test_wrong_plan_digest_is_rejected_before_data_or_output(self):
        self.train_fixture()
        path, _, _ = self.freeze()
        self.assert_rejected_before_voxels(path, "0" * 64, self.root / "bad_digest")

    def test_frozen_source_run_checkpoint_and_data_bindings_reject_changes(self):
        self.train_fixture()
        path, digest, plan = self.freeze()
        bound_paths = {
            "manifest": self.manifest,
            "split": self.split,
            **{f"{kind}_run": directory / "run.json"
               for kind, directory in self.run_dirs.items()},
            **{f"{kind}_checkpoint": directory / self.states[kind]["best_checkpoint"]
               for kind, directory in self.run_dirs.items()},
        }
        for name, bound_path in bound_paths.items():
            original = bound_path.read_bytes()
            with self.subTest(binding=name):
                try:
                    # Whitespace leaves JSON semantically equivalent; byte binding
                    # must still reject it. A trailing byte similarly changes .pt.
                    bound_path.write_bytes(original + b"\n")
                    self.assert_rejected_before_voxels(path, digest, self.root / name)
                finally:
                    bound_path.write_bytes(original)
        read_bytes = Path.read_bytes
        source = Path(evaluate.__file__).resolve()

        def simulated_source_edit(candidate):
            original = read_bytes(candidate)
            return original + b"\n# simulated edit\n" if candidate == source else original

        with patch.object(Path, "read_bytes", new=simulated_source_edit):
            self.assert_rejected_before_voxels(path, digest, self.root / "changed_source")
        for name in ("python", "torch", "numpy", "nibabel", "cuda_runtime", "cudnn", "machine"):
            changed_runtime = {**plan["runtime_identity"], name: "different_runtime"}
            with self.subTest(runtime=name), \
                    patch.object(evaluate, "_runtime_identity", return_value=changed_runtime):
                self.assert_rejected_before_voxels(path, digest, self.root / f"runtime_{name}")

    def test_subsets_duplicates_and_reordered_cohorts_cannot_be_executed(self):
        self.train_fixture()
        _, _, original = self.freeze()
        cohorts = [original["cohort"][:1], original["cohort"][::-1],
                   [*original["cohort"], original["cohort"][0]]]
        for index, cohort in enumerate(cohorts):
            with self.subTest(cohort=cohort):
                plan = copy.deepcopy(original)
                plan["cohort"] = cohort
                path, digest = self.write_plan(plan, f"bad-cohort-{index}.json")
                self.assert_rejected_before_voxels(path, digest, self.root / f"cohort-{index}")

    def test_full_validation_then_full_test_read_only_their_own_partition(self):
        self.train_fixture()
        saved_runs = {kind: {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}
                      for kind, directory in self.run_dirs.items()}
        path, digest, _ = self.freeze()
        validation_output = self.root / "validation"
        with patch.object(evaluate, "load_nifti_case", wraps=evaluate.load_nifti_case) as loader:
            report = self.run_plan(path, digest, validation_output)
        self.assertEqual([call.args[0].case_id for call in loader.call_args_list],
                         self.validation_ids * 2)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["accuracy_role"], "cpu_diagnostic")
        self.assertFalse(report["test_voxels_read"])
        self.assertTrue(report["eligible_for_test_plan"])
        self.assertTrue(report["paired_inputs_and_targets_match"])
        self.assertEqual(report["training_updates"], 0)
        for kind, result in report["models"].items():
            self.assertEqual(result["status"], "completed")
            self.assertTrue(result["all_saved_metrics_match"])
            self.assertEqual(result["summary"]["num_cases"], 3)
            self.assertEqual(result["summary"]["num_patients"], 2)
            self.assertEqual(result["summary"]["valid_case_count"], [3] * 5)
            reference = self.states[kind]["history"][0]["validation"]
            for index, case in enumerate(result["cases"]):
                for metric in ("dice", "iou", "valid"):
                    self.assertEqual(case["metrics"][metric], reference[metric][index])
        self.assertEqual(json.loads((validation_output / "evaluation.json").read_text()), report)
        with patch.object(evaluate, "load_nifti_case") as loader, \
                patch.object(nib, "load") as nifti_loader:
            path, digest, _ = self.freeze(partition="test", name="test-plan.json",
                                         validation_report=validation_output / "evaluation.json")
        loader.assert_not_called()
        nifti_loader.assert_not_called()
        with patch.object(evaluate, "load_nifti_case", wraps=evaluate.load_nifti_case) as loader:
            report = self.run_plan(path, digest, self.root / "test")
        self.assertEqual([call.args[0].case_id for call in loader.call_args_list], ["test", "test"])
        self.assertEqual(report["partition"], "test")
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["accuracy_role"], "cpu_diagnostic")
        self.assertTrue(report["test_voxels_read"])
        for result in report["models"].values():
            self.assertEqual(len(result["cases"]), 1)
            self.assertNotIn("saved_metric_comparison", result["cases"][0])
            self.assertIsNone(result["prostate_case_mean_target_met"])
        self.assertEqual(saved_runs, {
            kind: {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}
            for kind, directory in self.run_dirs.items()})

    def test_test_stage_requires_matching_complete_validation_evidence_and_bounded_metrics(self):
        self.train_fixture()
        with patch.object(evaluate, "load_nifti_case") as loader, self.assertRaises(ValueError):
            self.freeze(partition="test")
        loader.assert_not_called()
        path, digest, _ = self.freeze()
        output = self.root / "validation"
        report = self.run_plan(path, digest, output)
        # A valid prior report cannot authorize a fresh plan under a changed
        # numerical runtime, even though the current runtime matches that plan.
        changed_runtime = {**report["plan"]["runtime_identity"], "torch": "different_runtime"}
        with patch.object(evaluate, "_runtime_identity", return_value=changed_runtime), \
                patch.object(evaluate, "load_nifti_case") as loader, self.assertRaises(ValueError):
            self.freeze(partition="test", validation_report=output / "evaluation.json")
        loader.assert_not_called()
        proof = self.root / "proof.json"
        problems = ("failed", "started", "wrong_partition", "wrong_setup", "partial_cohort",
                    "missing_model", "failed_model", "metric_difference", "validity_difference")
        for problem in problems:
            candidate = copy.deepcopy(report)
            if problem in ("failed", "started"):
                candidate["status"] = problem
            elif problem == "wrong_partition":
                candidate["partition"] = "test"
            elif problem == "wrong_setup":
                candidate["setup_fingerprint"] = "0" * 64
            elif problem == "partial_cohort":
                candidate["models"]["2d"]["cases"].pop()
            elif problem == "missing_model":
                del candidate["models"]["3d"]
            elif problem == "failed_model":
                candidate["models"]["3d"]["status"] = "failed"
            else:
                metrics = candidate["models"]["2d"]["cases"][0]["metrics"]
                if problem == "metric_difference":
                    score = metrics["dice"][0]
                    metrics["dice"][0] = score + (0.006 if score < 0.5 else -0.006)
                else:
                    metrics["valid"][0] = False
                    metrics["dice"][0] = metrics["iou"][0] = None
            proof.write_text(json.dumps(candidate), encoding="utf-8")
            with self.subTest(problem=problem), \
                    patch.object(evaluate, "load_nifti_case") as loader, \
                    self.assertRaises(ValueError):
                self.freeze(partition="test", validation_report=proof)
            loader.assert_not_called()
        # The diagnostic backend bound accepts a small difference while
        # preserving the stricter saved-metric comparison in the report.
        candidate = copy.deepcopy(report)
        metrics = candidate["models"]["2d"]["cases"][0]["metrics"]
        score = metrics["dice"][0]
        metrics["dice"][0] = score + (0.004 if score < 0.5 else -0.004)
        proof.write_text(json.dumps(candidate), encoding="utf-8")
        path, digest, _ = self.freeze(partition="test", validation_report=proof, name="bounded.json")
        # Proof bytes are independently bound, including after freezing.
        proof.write_text(json.dumps(candidate) + "\n", encoding="utf-8")
        self.assert_rejected_before_voxels(path, digest, self.root / "changed_proof")

    def test_summary_weights_patients_equally_and_reports_empty_class_support(self):
        def row(case_id, patient_id, scores):
            return dict(case_id=case_id, patient_id=patient_id, metrics=dict(
                class_ids=[1, 2, 3, 4, 5], dice=scores,
                iou=[value / (2 - value) if value is not None else None for value in scores],
                valid=[value is not None for value in scores]))

        cases = [row("a1", "a", [1., 0.8, None, 0., None]),
                 row("a2", "a", [1., None, None, None, None]),
                 row("b1", "b", [0., 0.2, None, 0., None])]
        summary = evaluate.summarize_cases(cases)
        self.assertAlmostEqual(summary["case_mean"]["dice"][0], 2 / 3)
        self.assertAlmostEqual(summary["patient_macro"]["dice"][0], 0.5)
        self.assertEqual(summary["case_mean"]["dice"][1:], [0.5, None, 0., None])
        self.assertEqual(summary["valid_case_count"], [3, 2, 0, 2, 0])
        self.assertEqual(summary["valid_patient_count"], [2, 2, 0, 2, 0])
        self.assertEqual(summary["per_patient"]["a"]["num_cases"], 2)
        self.assertEqual(summary["per_patient"]["a"]["valid_count"], [2, 1, 0, 1, 0])
        self.assertAlmostEqual(summary["case_mean"]["iou"][1], 7 / 18)
        self.assertEqual(json.loads(json.dumps(summary, allow_nan=False)), summary)
        with self.assertRaises(ValueError):
            evaluate.summarize_cases([])
        with self.assertRaises(ValueError):
            evaluate.summarize_cases([cases[0], cases[0]])
        empty = evaluate.summarize_cases([row("empty", "empty_patient", [None] * 5)])
        self.assertEqual(empty["case_mean"]["dice"], [None] * 5)
        self.assertEqual(empty["patient_macro"]["dice"], [None] * 5)
        self.assertEqual(empty["valid_case_count"], [0] * 5)
        self.assertEqual(empty["valid_patient_count"], [0] * 5)

    def test_second_model_payload_is_validated_before_first_voxel_load(self):
        self.train_fixture()
        checkpoint = self.run_dirs["3d"] / self.states["3d"]["best_checkpoint"]
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        weights = payload["model_state_dict"]
        first = next(iter(weights))
        weights[first] = torch.full_like(weights[first], float("nan"))
        torch.save(payload, checkpoint)
        path, digest, _ = self.freeze()
        with patch.object(evaluate, "load_best_model", wraps=evaluate.load_best_model) as setup:
            self.assert_rejected_before_voxels(path, digest, self.root / "bad_second_model")
        self.assertEqual([call.args[0] for call in setup.call_args_list],
                         [self.run_dirs["2d"].resolve(), self.run_dirs["3d"].resolve()])

    def test_mid_run_failure_keeps_started_report_and_completed_case_rows(self):
        self.train_fixture()
        path, digest, _ = self.freeze()
        output = self.root / "interrupted"
        actual_loader = evaluate.load_nifti_case
        calls = []

        def fail_second_case(case, **kwargs):
            persisted = json.loads((output / "evaluation.json").read_text())
            self.assertEqual(persisted["status"], "started")
            self.assertEqual(persisted["models"]["2d"]["status"], "started")
            calls.append(case.case_id)
            if len(calls) == 2:
                self.assertEqual(len(persisted["models"]["2d"]["cases"]), 1)
                raise RuntimeError("synthetic interrupted NIfTI read")
            return actual_loader(case, **kwargs)

        with patch.object(evaluate, "load_nifti_case", side_effect=fail_second_case), \
                self.assertRaisesRegex(RuntimeError, "synthetic interrupted"):
            self.run_plan(path, digest, output)
        report = json.loads((output / "evaluation.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertIn("synthetic interrupted", report["error"])
        self.assertEqual([r["case_id"] for r in report["models"]["2d"]["cases"]],
                         self.validation_ids[:1])
        self.assertNotIn("3d", report["models"])
        self.assertFalse(report["test_voxels_read"])
        self.assertIn("ended_at", report)

    def test_existing_output_is_preserved_without_loading_voxels(self):
        self.train_fixture()
        path, digest, _ = self.freeze()
        output = self.root / "existing"
        output.mkdir()
        marker = output / "keep.txt"
        marker.write_bytes(b"previous evaluation\n")
        with patch.object(evaluate, "load_nifti_case") as loader, \
                self.assertRaises(FileExistsError):
            self.run_plan(path, digest, output)
        loader.assert_not_called()
        self.assertEqual(list(output.iterdir()), [marker])
        self.assertEqual(marker.read_bytes(), b"previous evaluation\n")


if __name__ == "__main__":
    unittest.main()
