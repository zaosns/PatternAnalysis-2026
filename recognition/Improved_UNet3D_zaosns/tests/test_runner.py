"""Synthetic end-to-end checks for training, selection and safe continuation."""

from contextlib import redirect_stdout
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch

import train


class TrainingRunnerTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.use_deterministic_algorithms,
                        torch.are_deterministic_algorithms_enabled(),
                        warn_only=torch.is_deterministic_algorithms_warn_only_enabled())
        self.addCleanup(setattr, torch.backends.cudnn, "deterministic", torch.backends.cudnn.deterministic)
        self.addCleanup(setattr, torch.backends.cudnn, "benchmark", torch.backends.cudnn.benchmark)
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        self.addCleanup(random.setstate, random.getstate())
        rng = torch.random.fork_rng(devices=[])
        rng.__enter__()
        self.addCleanup(rng.__exit__, None, None, None)
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manifest = self.root / "manifest.json"
        self.split = self.root / "split.json"
        entries = []
        for index, case_id in enumerate(("train_a", "train_b", "validation", "test")):
            image_path = self.root / f"{case_id}_image.nii"
            label_path = self.root / f"{case_id}_label.nii"
            if case_id == "test":
                # Existence checks are permitted; opening these as NIfTI fails.
                image_path.write_text("Held-out image must not be loaded.\n")
                label_path.write_text("Held-out label must not be loaded.\n")
            else:
                coordinates = np.indices((7 + index, 9, 5 + index))
                image = (coordinates[0] * 3 + coordinates[1] * 2
                         + coordinates[2] + 1 + index).astype(np.float32)
                label = (coordinates.sum(axis=0) % 3).astype(np.int16)
                for values, path in ((image, image_path), (label, label_path)):
                    volume = nib.Nifti1Image(values, np.diag([2., 3., 4., 1.]))
                    volume.header.set_xyzt_units("mm")
                    nib.save(volume, path)
            entries.append(dict(case_id=case_id, patient_id=f"patient_{index}",
                                image_path=image_path.name, label_path=label_path.name))
        self.manifest.write_text(json.dumps(entries), encoding="utf-8")
        self.split.write_text(json.dumps(dict(
            format_version=1, seed=3710, validation_patients=1, test_patients=1,
            patient_key_rule="Synthetic independent patients",
            manifest_sha256=hashlib.sha256(self.manifest.read_bytes()).hexdigest(),
            train=["train_b", "train_a"], validation=["validation"], test=["test"],
        )), encoding="utf-8")

    def arguments(self, output, *, epochs=2, resume=False, extra=()):
        arguments = ["--manifest", str(self.manifest), "--split", str(self.split),
                     "--output-dir", str(output), "--epochs", str(epochs),
                     "--device", "cpu", "--num-classes", "3", "--channels", "2", "3",
                     "--dropout", "0.25", "--learning-rate", "0.002",
                     "--downsample-factor", "2", "--threads", "1", "--seed", "3710"]
        if resume:
            arguments.append("--resume")
        return arguments + list(extra)

    def run_training(self, output, **options):
        args = train.build_parser().parse_args(self.arguments(output, **options))
        with redirect_stdout(io.StringIO()):
            result = train.run_training(args)
        self.assertEqual(result, self.read_state(output))
        return result

    def read_state(self, output):
        return json.loads((output / "run.json").read_text(encoding="utf-8"),
                          parse_constant=lambda value: self.fail(f"Non-JSON number: {value}"))

    def payload(self, output, state):
        return torch.load(output / state["latest_checkpoint"],
                          map_location="cpu", weights_only=True)

    def assert_nested_equal(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assert_nested_equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(type(actual), type(expected))
            self.assertEqual(len(actual), len(expected))
            for value, reference in zip(actual, expected):
                self.assert_nested_equal(value, reference)
        else:
            self.assertEqual(actual, expected)

    def stable_history(self, state):
        return [{key: value for key, value in row.items()
                 if key not in {"elapsed_seconds", "peak_cuda_allocated_mib"}}
                for row in state["history"]]

    def snapshot(self, output):
        return {path.relative_to(output): path.read_bytes()
                for path in output.rglob("*") if path.is_file()}

    def test_main_trains_real_model_on_variable_shapes_without_reading_test(self):
        output = self.root / "main_run"
        with redirect_stdout(io.StringIO()):
            self.assertEqual(train.main(self.arguments(output)), 0)
        state = self.read_state(output)
        self.assertEqual(state["format_version"], 1)
        self.assertEqual(state["completed_epochs"], 2)
        self.assertEqual([row["epoch"] for row in state["history"]], [1, 2])
        for row in state["history"]:
            self.assertEqual(row["train_samples"], 2)
            self.assertEqual(row["validation_samples"], 1)
            self.assertEqual(row["validation"]["case_ids"], ["validation"])
            self.assertEqual(row["validation"]["class_ids"], [1, 2])
            self.assertEqual(len(row["validation"]["dice"]), 1)
            self.assertTrue(np.isfinite(row["train_loss"]))
            self.assertTrue(np.isfinite(row["validation_loss"]))
            self.assertIsNone(row["peak_cuda_allocated_mib"])
        payload = self.payload(output, state)
        self.assertEqual(payload["completed_epochs"], 2)
        self.assertTrue(payload["optimizer_state_dict"]["state"])
        self.assertTrue(any(item["exp_avg"].abs().sum() > 0
                            for item in payload["optimizer_state_dict"]["state"].values()))
        with (output / "history.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 2)
        self.assertEqual(float(rows[-1]["validation_loss"]),
                         state["history"][-1]["validation_loss"])
        help_result = subprocess.run(
            [sys.executable, "-s", str(Path(train.__file__).resolve()), "--help"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("--resume", help_result.stdout)
        self.assertIn("--manifest", help_result.stdout)
        self.assertIn("--deterministic", help_result.stdout)
        self.assertFalse(state["config"]["reproducibility"]["deterministic_requested"])

    def test_resume_reproduces_uninterrupted_weights_adam_rng_and_metrics(self):
        continuous = self.root / "continuous"
        resumed = self.root / "resumed"
        expected = self.run_training(continuous)
        self.run_training(resumed, epochs=1)
        actual = self.run_training(resumed, epochs=2, resume=True)
        self.assert_nested_equal(self.payload(resumed, actual), self.payload(continuous, expected))
        self.assertEqual(self.stable_history(actual), self.stable_history(expected))
        self.assertEqual(actual["best_epoch"], expected["best_epoch"])
        self.assertEqual(actual["best_checkpoint"], expected["best_checkpoint"])
        original_files = self.snapshot(resumed)
        with patch.object(train, "train_one_epoch") as training, \
                patch.object(train, "validate_one_epoch") as validation:
            unchanged = self.run_training(resumed, epochs=2, resume=True)
        training.assert_not_called()
        validation.assert_not_called()
        self.assertEqual(unchanged, actual)
        self.assertEqual(self.snapshot(resumed), original_files)

    def test_deterministic_resume_matches_and_binds_actual_workspace_policy(self):
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        continuous = self.root / "deterministic_continuous"
        resumed = self.root / "deterministic_resumed"
        options = {"extra": ("--deterministic",)}
        expected = self.run_training(continuous, **options)
        self.assertEqual(expected["config"]["reproducibility"], {
            "deterministic_requested": True,
            "deterministic_algorithms": True,
            "deterministic_warn_only": False,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "cublas_workspace_config": ":4096:8",
        })
        self.run_training(resumed, epochs=1, **options)
        actual = self.run_training(resumed, epochs=2, resume=True, **options)
        self.assert_nested_equal(self.payload(resumed, actual), self.payload(continuous, expected))
        self.assertEqual(self.stable_history(actual), self.stable_history(expected))
        original = self.snapshot(resumed)
        with self.assertRaisesRegex(ValueError, "configuration"):
            self.run_training(resumed, epochs=3, resume=True)
        self.assertEqual(self.snapshot(resumed), original)
        # Respect the user's different valid workspace choice, but never resume
        # a run whose recorded policy used a different value.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        with self.assertRaisesRegex(ValueError, "configuration"):
            self.run_training(resumed, epochs=3, resume=True, **options)
        self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], ":16:8")
        self.assertEqual(self.snapshot(resumed), original)

    def test_first_minimum_selects_best_and_undefined_metrics_are_json_null(self):
        output = self.root / "selection"
        original_validate = train.validate_one_epoch
        losses = iter((0.3, 0.4, 0.3))

        def controlled_validation(*args, **kwargs):
            metrics = original_validate(*args, **kwargs)
            metrics["loss"] = next(losses)
            absent = metrics["class_ids"].tolist().index(2)
            metrics["valid"][:, absent] = False
            metrics["valid_count"][absent] = 0
            for name in ("dice", "iou"):
                metrics[name][:, absent] = float("nan")
                metrics[f"mean_{name}"][absent] = float("nan")
            return metrics

        with patch.object(train, "validate_one_epoch", side_effect=controlled_validation):
            state = self.run_training(output, epochs=3)
        self.assertEqual(state["best_epoch"], 1)
        self.assertEqual(state["best_checkpoint"], "checkpoint-000001.pt")
        self.assertEqual(state["latest_checkpoint"], "checkpoint-000003.pt")
        self.assertEqual({path.name for path in output.glob("checkpoint-*.pt")},
                         {"checkpoint-000001.pt", "checkpoint-000003.pt"})
        self.assertEqual(self.payload(output, state)["best_validation_loss"], 0.3)
        for row in state["history"]:
            metrics = row["validation"]
            self.assertEqual(metrics["valid_count"][1], 0)
            self.assertFalse(metrics["valid"][0][1])
            self.assertIsNone(metrics["dice"][0][1])
            self.assertIsNone(metrics["iou"][0][1])
            self.assertIsNone(metrics["mean_dice"][1])
            self.assertIsNone(metrics["mean_iou"][1])

    def test_incompatible_resume_and_fresh_overwrite_leave_output_unchanged(self):
        output = self.root / "protected"
        self.run_training(output)
        original = self.snapshot(output)
        for options in (
            dict(epochs=3, resume=True, extra=("--dropout", "0.1")),
            dict(epochs=1, resume=True),
            dict(epochs=3, resume=False),
        ):
            with self.subTest(options=options):
                with self.assertRaises((ValueError, FileExistsError)):
                    self.run_training(output, **options)
                self.assertEqual(self.snapshot(output), original)
        relocated = self.root / "relocated"
        relocated.mkdir()
        for source in (self.manifest, self.split, *self.root.glob("*.nii")):
            shutil.copy2(source, relocated / source.name)
        self.assertEqual((relocated / self.manifest.name).read_bytes(), self.manifest.read_bytes())
        self.assertEqual((relocated / self.split.name).read_bytes(), self.split.read_bytes())
        # Byte-identical manifests now resolve their relative paths elsewhere.
        with self.assertRaises(ValueError):
            self.run_training(output, epochs=3, resume=True, extra=(
                "--manifest", str(relocated / self.manifest.name),
                "--split", str(relocated / self.split.name),
            ))
        self.assertEqual(self.snapshot(output), original)

    def test_missing_committed_checkpoint_is_rejected_without_changing_run(self):
        output = self.root / "missing_checkpoint"
        state = self.run_training(output)
        (output / state["latest_checkpoint"]).unlink()
        original = self.snapshot(output)
        with patch.object(train, "train_one_epoch") as training:
            with self.assertRaisesRegex(ValueError, "[Cc]heckpoint"):
                self.run_training(output, epochs=3, resume=True)
        training.assert_not_called()
        self.assertEqual(self.snapshot(output), original)

    def test_failed_epoch_commit_can_resume_from_previous_committed_state(self):
        continuous = self.root / "reference"
        interrupted = self.root / "interrupted"
        expected = self.run_training(continuous)
        original_write = train._atomic_json

        def interrupt_second_commit(path, state):
            if Path(path).name == "run.json" and state["completed_epochs"] == 2:
                raise OSError("Simulated interruption before epoch 2 commit")
            return original_write(path, state)

        with patch.object(train, "_atomic_json", side_effect=interrupt_second_commit):
            with self.assertRaisesRegex(OSError, "epoch 2"):
                self.run_training(interrupted)
        committed = self.read_state(interrupted)
        self.assertEqual(committed["completed_epochs"], 1)
        self.assertTrue((interrupted / committed["latest_checkpoint"]).is_file())
        actual = self.run_training(interrupted, resume=True)
        self.assert_nested_equal(self.payload(interrupted, actual), self.payload(continuous, expected))
        self.assertEqual(self.stable_history(actual), self.stable_history(expected))
        self.assertEqual({path.name for path in interrupted.glob("checkpoint-*.pt")},
                         {actual["best_checkpoint"], actual["latest_checkpoint"]})

    def test_failure_before_first_epoch_resumes_from_recorded_configuration(self):
        continuous = self.root / "one_epoch_reference"
        interrupted = self.root / "before_first_epoch"
        expected = self.run_training(continuous, epochs=1)
        with patch.object(train, "train_one_epoch", side_effect=OSError("First read failed")):
            with self.assertRaisesRegex(OSError, "First read"):
                self.run_training(interrupted, epochs=1)
        initial = self.read_state(interrupted)
        self.assertEqual(initial["completed_epochs"], 0)
        self.assertEqual(initial["history"], [])
        self.assertIsNone(initial["latest_checkpoint"])
        self.assertIsNone(initial["best_checkpoint"])
        self.assertIsNone(initial["best_epoch"])
        actual = self.run_training(interrupted, epochs=1, resume=True)
        self.assert_nested_equal(self.payload(interrupted, actual), self.payload(continuous, expected))
        self.assertEqual(self.stable_history(actual), self.stable_history(expected))


if __name__ == "__main__":
    unittest.main()
