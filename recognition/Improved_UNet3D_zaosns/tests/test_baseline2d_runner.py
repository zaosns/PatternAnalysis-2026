"""Small end-to-end checks for the case-weighted 2D baseline runner."""

from contextlib import redirect_stdout
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from baseline2d import SliceWiseUNet2D
import train_baseline2d as runner


class Baseline2DRunnerTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.use_deterministic_algorithms,
                        torch.are_deterministic_algorithms_enabled(),
                        warn_only=torch.is_deterministic_algorithms_warn_only_enabled())
        self.addCleanup(setattr, torch.backends.cudnn, "deterministic",
                        torch.backends.cudnn.deterministic)
        self.addCleanup(setattr, torch.backends.cudnn, "benchmark",
                        torch.backends.cudnn.benchmark)
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        self.addCleanup(random.setstate, random.getstate())
        self.addCleanup(np.random.set_state, np.random.get_state())
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
                # Existence checks are allowed; a voxel/header read must fail.
                image_path.write_text("Held-out image must not be loaded.\n")
                label_path.write_text("Held-out label must not be loaded.\n")
            else:
                # Downsampling leaves variable D >= 3 and H/W with enough
                # bottleneck pixels for slice-wise instance normalization.
                coordinates = np.indices((9 + 2 * index, 11, 7 + 2 * index))
                image = (coordinates[0] * 3 + coordinates[1] * 2
                         + coordinates[2] + 1 + index).astype(np.float32)
                label = (coordinates.sum(axis=0) % 6).astype(np.int16)
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
                     "--device", "cpu", "--num-classes", "6", "--channels", "2", "3",
                     "--learning-rate", "0.002", "--downsample-factor", "2",
                     "--threads", "1", "--seed", "3710", "--deterministic"]
        if resume:
            arguments.append("--resume")
        return arguments + list(extra)

    def run_training(self, output, **options):
        args = runner.build_parser().parse_args(self.arguments(output, **options))
        with redirect_stdout(io.StringIO()):
            state = runner.run_training(args)
        self.assertEqual(state, self.read_state(output))
        return state

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

    def test_real_training_updates_once_per_case_without_loading_test_volumes(self):
        output = self.root / "complete"
        original_step = torch.optim.Adam.step
        updated_parameter_counts = []

        def counted_step(optimizer, *args, **kwargs):
            updated_parameter_counts.append(sum(parameter.grad is not None
                for group in optimizer.param_groups for parameter in group["params"]))
            return original_step(optimizer, *args, **kwargs)

        with patch.object(torch.optim.Adam, "step", new=counted_step):
            state = self.run_training(output)
        self.assertEqual(len(updated_parameter_counts), 2 * 2)
        self.assertTrue(all(count > 0 for count in updated_parameter_counts))
        self.assertEqual(state["completed_epochs"], 2)
        self.assertEqual(state["config"]["model_kind"], "slice_wise_unet2d")
        self.assertEqual(state["config"]["objective_unit"], "complete reconstructed case volume")
        self.assertTrue(state["config"]["reproducibility"]["deterministic_algorithms"])
        self.assertFalse(state["config"]["reproducibility"]["deterministic_warn_only"])
        for row in state["history"]:
            self.assertEqual(row["train_samples"], 2)
            self.assertEqual(row["validation_samples"], 1)
            self.assertEqual(row["validation"]["case_ids"], ["validation"])
            self.assertEqual(row["validation"]["class_ids"], [1, 2, 3, 4, 5])
            self.assertEqual(len(row["validation"]["dice"]), 1)
            self.assertTrue(np.isfinite(row["train_loss"]))
            self.assertTrue(np.isfinite(row["validation_loss"]))
            self.assertIsNone(row["peak_cuda_allocated_mib"])
        payload = self.payload(output, state)
        self.assertEqual(payload["model_kind"], "slice_wise_unet2d")
        self.assertEqual(payload["completed_epochs"], 2)
        self.assertTrue(payload["optimizer_state_dict"]["state"])
        for item in payload["optimizer_state_dict"]["state"].values():
            self.assertEqual(item["step"].item(), 4)
        self.assertTrue(any(item["exp_avg"].abs().sum() > 0
                            for item in payload["optimizer_state_dict"]["state"].values()))
        best = min(state["history"], key=lambda row: row["validation_loss"])
        self.assertEqual(state["best_epoch"], best["epoch"])
        self.assertEqual(payload["best_validation_loss"], best["validation_loss"])
        self.assertEqual({path.name for path in output.glob("checkpoint-*.pt")},
                         {state["best_checkpoint"], state["latest_checkpoint"]})
        with (output / "history.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 2)
        self.assertEqual(float(rows[-1]["validation_loss"]),
                         state["history"][-1]["validation_loss"])

    def test_resumed_training_exactly_matches_model_adam_rng_and_case_metrics(self):
        continuous, resumed = self.root / "continuous", self.root / "resumed"
        expected = self.run_training(continuous)
        self.run_training(resumed, epochs=1)
        actual = self.run_training(resumed, resume=True)
        self.assert_nested_equal(self.payload(resumed, actual), self.payload(continuous, expected))
        self.assertEqual(self.stable_history(actual), self.stable_history(expected))
        self.assertEqual(actual["best_epoch"], expected["best_epoch"])
        self.assertEqual(actual["best_checkpoint"], expected["best_checkpoint"])
        original = self.snapshot(resumed)
        with patch.object(runner, "train_one_epoch") as training, \
                patch.object(runner, "validate_one_epoch") as validation:
            self.assertEqual(self.run_training(resumed, resume=True), actual)
        training.assert_not_called()
        validation.assert_not_called()
        self.assertEqual(self.snapshot(resumed), original)

    def test_volume_overlap_differs_from_averaging_slice_scores(self):
        class EncodedSlicePredictor(nn.Module):
            num_classes = 6

            def forward(self, images):
                labels = images[:, 0].long()
                return F.one_hot(labels, self.num_classes).movedim(-1, 1).float() * 8

        model = SliceWiseUNet2D(num_classes=6, channels=(2, 3))
        model.model = EncodedSlicePredictor()
        image = torch.zeros((1, 1, 3, 4, 5))
        image[:, :, 0] = 5
        target = image[:, 0].long()
        target[:, 1, 0, 0] = 5
        criterion = runner.CrossEntropyDiceLoss()
        metrics = runner.validate_one_epoch(model, [(image, target)], criterion,
                                             num_classes=6, include_background=False)
        # The large, perfect plane has 20 voxels; the missed plane has one.
        # A mean of defined per-plane Dice/IoU would give (1 + 0) / 2 = .5.
        self.assertEqual(metrics["num_samples"], 1)
        self.assertEqual(tuple(metrics["dice"].shape), (1, 5))
        self.assertAlmostEqual(metrics["mean_dice"][4].item(), 40 / 41, places=6)
        self.assertAlmostEqual(metrics["mean_iou"][4].item(), 20 / 21, places=6)
        self.assertGreater(abs(metrics["mean_dice"][4].item() - .5), .4)
        self.assertEqual(metrics["valid_count"].tolist(), [0, 0, 0, 0, 1])
        expected_loss = criterion(model(image), target).item()
        slice_loss = sum(criterion(model(image[:, :, z:z + 1]),
                                   target[:, z:z + 1]).item() for z in range(3)) / 3
        self.assertEqual(metrics["loss"], expected_loss)
        self.assertGreater(abs(expected_loss - slice_loss), .01)

    def test_changed_training_configuration_and_existing_output_are_rejected(self):
        output = self.root / "protected"
        self.run_training(output, epochs=1)
        original = self.snapshot(output)
        for options in (
            dict(epochs=2, resume=True, extra=("--learning-rate", "0.003")),
            dict(epochs=2, resume=True, extra=("--channels", "2", "4")),
            dict(epochs=2, resume=False),
        ):
            with self.subTest(options=options):
                with self.assertRaises((ValueError, FileExistsError)):
                    self.run_training(output, **options)
                self.assertEqual(self.snapshot(output), original)

    def test_changed_source_hash_rejects_resume_without_editing_source(self):
        output = self.root / "source_bound"
        state = self.run_training(output, epochs=1)
        self.assertIn("baseline2d.py", state["config"]["source_sha256"])
        original = self.snapshot(output)
        read_bytes = Path.read_bytes
        baseline_source = Path(runner.__file__).resolve().with_name("baseline2d.py")

        def simulated_source_edit(path):
            content = read_bytes(path)
            return content + b"\n# simulated source change\n" if path == baseline_source else content

        with patch.object(Path, "read_bytes", new=simulated_source_edit), \
                patch.object(runner, "train_one_epoch") as training:
            with self.assertRaisesRegex(ValueError, "configuration|source code"):
                self.run_training(output, resume=True)
        training.assert_not_called()
        self.assertEqual(self.snapshot(output), original)

    def test_run_and_checkpoint_architecture_tags_reject_other_models(self):
        output = self.root / "architecture"
        state = self.run_training(output, epochs=1)
        state_path = output / "run.json"
        original_state = state_path.read_bytes()
        state["config"]["model_kind"] = "improved_unet3d"
        state_path.write_text(json.dumps(state), encoding="utf-8")
        altered = self.snapshot(output)
        with self.assertRaisesRegex(ValueError, "different architecture"):
            self.run_training(output, resume=True)
        self.assertEqual(self.snapshot(output), altered)
        state_path.write_bytes(original_state)
        state = self.read_state(output)
        payload = self.payload(output, state)
        payload["model_kind"] = "improved_unet3d"
        torch.save(payload, output / state["latest_checkpoint"])
        altered = self.snapshot(output)
        with self.assertRaisesRegex(ValueError, "2D baseline checkpoint|other architectures"):
            self.run_training(output, resume=True)
        self.assertEqual(self.snapshot(output), altered)

    def test_failed_epoch_commit_resumes_previous_committed_checkpoint(self):
        continuous, interrupted = self.root / "reference", self.root / "interrupted"
        expected = self.run_training(continuous)
        original_write = runner._atomic_json

        def interrupt_second_commit(path, state):
            if Path(path).name == "run.json" and state["completed_epochs"] == 2:
                raise OSError("Simulated failure before epoch 2 commit")
            return original_write(path, state)

        with patch.object(runner, "_atomic_json", side_effect=interrupt_second_commit):
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


if __name__ == "__main__":
    unittest.main()
