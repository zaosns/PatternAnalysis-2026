"""Synthetic checks for checkpoint-bound validation inspection without test access."""

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

import predict
import prediction_panels
import train


class PredictionTests(unittest.TestCase):
    def setUp(self):
        # Both CLIs configure process-wide state; leave subsequent tests unchanged.
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
        entries = []
        for index, case_id in enumerate(("train", "validation_a", "validation_b", "test")):
            image_path = self.root / f"{case_id}_image.nii"
            label_path = self.root / f"{case_id}_label.nii"
            if case_id == "test":
                image_path.write_text("Held-out image must never be opened as NIfTI.\n")
                label_path.write_text("Held-out label must never be opened as NIfTI.\n")
            else:
                coordinates = np.indices((9 + index, 8, 7 + index))
                image = (coordinates[0] * 3 + coordinates[1] * 2
                         + coordinates[2] + 1).astype(np.float32)
                # Factor two still retains all six classes in this synthetic grid.
                label = ((coordinates[0] // 2 + coordinates[1] // 2
                          + coordinates[2] // 2) % 6).astype(np.int16)
                for values, path in ((image, image_path), (label, label_path)):
                    volume = nib.Nifti1Image(values, np.diag([2., 3., 4., 1.]))
                    volume.header.set_xyzt_units("mm")
                    nib.save(volume, path)
            entries.append(dict(case_id=case_id, patient_id=f"patient_{index}",
                                image_path=image_path.name, label_path=label_path.name))
        self.manifest.write_text(json.dumps(entries), encoding="utf-8")
        self.split.write_text(json.dumps(dict(
            format_version=1, seed=3710, validation_patients=2, test_patients=1,
            patient_key_rule="Independent synthetic patients",
            manifest_sha256=hashlib.sha256(self.manifest.read_bytes()).hexdigest(),
            train=["train"], validation=["validation_a", "validation_b"], test=["test"],
        )), encoding="utf-8")
        self.run_dir = self.root / "training"
        self.state = None

        # Rendering is tested separately. Keep the real prediction/data/model path.
        renderer_patch = patch.object(prediction_panels, "render_case_panels",
                                      side_effect=self.fake_render)
        self.render = renderer_patch.start()
        self.addCleanup(renderer_patch.stop)

    @staticmethod
    def fake_render(prepared, prediction, output, *, case_label, metrics):
        Path(output).write_bytes(b"synthetic renderer output\n")
        return dict(case_label=case_label, shape=list(prediction.shape))

    def train_fixture(self):
        args = train.build_parser().parse_args([
            "--manifest", str(self.manifest), "--split", str(self.split),
            "--output-dir", str(self.run_dir), "--epochs", "2", "--device", "cpu",
            "--num-classes", "6", "--channels", "2", "3", "--dropout", "0.25",
            "--learning-rate", "0.002", "--downsample-factor", "2",
            "--threads", "1", "--seed", "3710", "--deterministic",
        ])
        with redirect_stdout(io.StringIO()):
            self.state = train.run_training(args)
        return self.state

    def arguments(self, output, *, case_ids=("validation_b",), extra=()):
        return predict.build_parser().parse_args([
            "--run-dir", str(self.run_dir), "--manifest", str(self.manifest),
            "--split", str(self.split), "--case-ids", *case_ids,
            "--output-dir", str(output), "--device", "cpu", "--threads", "1",
            "--repeats", "2", *extra,
        ])

    def reject_before_voxels(self, output, *, error=ValueError, message=None, **options):
        with patch.object(predict, "load_nifti_case") as loader:
            if message is None:
                with self.assertRaises(error):
                    predict.run_prediction(self.arguments(output, **options))
            else:
                with self.assertRaisesRegex(error, message):
                    predict.run_prediction(self.arguments(output, **options))
            loader.assert_not_called()
        self.assertFalse(output.exists())
        self.render.assert_not_called()

    def test_real_checkpoint_prediction_matches_selected_validation_only(self):
        state = self.train_fixture()
        original_run = {path.name: path.read_bytes() for path in self.run_dir.iterdir()
                        if path.is_file()}
        # A fresh training pass or reading unselected validation voxels now fails.
        for name in ("train", "validation_a"):
            for suffix in ("image", "label"):
                (self.root / f"{name}_{suffix}.nii").write_text("Must not load this case.\n")
        output = self.root / "inspection"
        checkpoint = self.run_dir / state["best_checkpoint"]
        expected_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        with patch.object(predict, "load_nifti_case", wraps=predict.load_nifti_case) as loader, \
                redirect_stdout(io.StringIO()):
            result = predict.run_prediction(self.arguments(output, extra=(
                "--checkpoint-sha256", expected_sha,
            )))
        self.assertEqual([call.args[0].case_id for call in loader.call_args_list],
                         ["validation_b"])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["partition"], "validation")
        self.assertTrue(result["all_saved_metrics_match"])
        self.assertEqual(result["selected_case_ids"], ["validation_b"])
        self.assertEqual(result["checkpoint_sha256"], expected_sha)
        self.assertEqual(result["best_epoch"], state["best_epoch"])
        row = result["cases"][0]
        reference = state["history"][state["best_epoch"] - 1]["validation"]
        index = reference["case_ids"].index("validation_b")
        self.assertEqual(row["metrics"]["class_ids"], [1, 2, 3, 4, 5])
        for name in ("dice", "iou", "valid"):
            self.assertEqual(row["metrics"][name], reference[name][index])
        self.assertEqual(row["saved_metric_comparison"]["maximum_absolute_difference"], 0.)
        self.assertEqual(row["timing"]["measured_forwards"], 2)
        self.assertEqual(len(row["timing"]["forward_seconds"]), 2)
        self.assertIsNone(row["timing"]["peak_cuda_allocated_mib"])
        self.assertEqual(row["figure_sha256"], hashlib.sha256(
            (output / row["figure"]).read_bytes()).hexdigest())
        self.assertEqual(json.loads((output / "prediction.json").read_text()), result)
        self.render.assert_called_once()
        self.assertEqual({path.name: path.read_bytes() for path in self.run_dir.iterdir()
                          if path.is_file()}, original_run)

    def test_invalid_selection_and_checkpoint_digest_rejected_before_output_or_voxels(self):
        self.train_fixture()
        for index, cases in enumerate((("train",), ("test",), ("missing",),
                                       ("validation_b", "test"),
                                       ("validation_b", "validation_b"))):
            with self.subTest(cases=cases):
                self.reject_before_voxels(self.root / f"invalid-selection-{index}",
                                         case_ids=cases)
        self.reject_before_voxels(self.root / "wrong-digest", message="SHA-256",
                                 extra=("--checkpoint-sha256", "0" * 64))

    def test_checkpoint_tensor_corruption_and_progress_are_rejected(self):
        state = self.train_fixture()
        path = self.run_dir / state["best_checkpoint"]
        original = path.read_bytes()
        for index, problem in enumerate(("nan", "shape", "dtype", "missing", "progress")):
            with self.subTest(problem=problem):
                payload = torch.load(io.BytesIO(original), map_location="cpu", weights_only=True)
                weights = payload["model_state_dict"]
                name = next(iter(weights))
                if problem == "nan":
                    weights[name] = torch.full_like(weights[name], float("nan"))
                elif problem == "shape":
                    weights[name] = weights[name].flatten()[:1]
                elif problem == "dtype":
                    weights[name] = weights[name].double()
                elif problem == "missing":
                    del weights[name]
                else:
                    payload["completed_epochs"] += 1
                torch.save(payload, path)
                message = ("configuration/progress" if problem == "progress" else
                           "model keys" if problem == "missing" else "Invalid checkpoint tensor")
                self.reject_before_voxels(self.root / f"bad-checkpoint-{index}",
                                         message=message)
        path.write_bytes(original)

    def test_existing_output_is_preserved_without_voxel_loading(self):
        self.train_fixture()
        output = self.root / "existing"
        output.mkdir()
        marker = output / "keep.txt"
        marker.write_bytes(b"existing inspection\n")
        with patch.object(predict, "load_nifti_case") as loader, \
                self.assertRaises(FileExistsError):
            predict.run_prediction(self.arguments(output))
        loader.assert_not_called()
        self.render.assert_not_called()
        self.assertEqual(list(output.iterdir()), [marker])
        self.assertEqual(marker.read_bytes(), b"existing inspection\n")

    def test_untrained_run_is_rejected_before_loading_data(self):
        self.run_dir.mkdir()
        (self.run_dir / ".run.lock").touch()
        (self.run_dir / "run.json").write_text(json.dumps(dict(
            format_version=1, config={}, completed_epochs=0, best_epoch=None,
            latest_checkpoint=None, best_checkpoint=None, history=[],
        )))
        self.reject_before_voxels(self.root / "untrained-inspection",
                                 message="completed training epoch")

    def test_metric_comparison_rejects_malformed_rows_and_respects_undefined_scores(self):
        actual = dict(class_ids=[1, 2, 3, 4, 5], valid=[True] * 4 + [False],
                      dice=[0.1, 0.2, 0.3, 0.4, None],
                      iou=[0.05, 0.1, 0.15, 0.2, None])
        reference = dict(class_ids=actual["class_ids"],
                         **{name: [actual[name]] for name in ("dice", "iou", "valid")})
        self.assertTrue(predict.metric_comparison(actual, reference, 0)["matches"])
        for side in ("actual", "reference"):
            for problem in ("empty", "class_order", "null_valid", "inf", "out_of_range"):
                with self.subTest(side=side, problem=problem):
                    candidate = copy.deepcopy(actual)
                    saved = copy.deepcopy(reference)
                    target = candidate if side == "actual" else saved
                    scores = candidate["dice"] if side == "actual" else saved["dice"][0]
                    if problem == "empty":
                        scores.clear()
                    elif problem == "class_order":
                        target["class_ids"].reverse()
                    elif problem == "null_valid":
                        scores[0] = None
                    elif problem == "inf":
                        scores[0] = float("inf")
                    else:
                        scores[0] = 1.1
                    with self.assertRaises(ValueError):
                        predict.metric_comparison(candidate, saved, 0)
        changed = copy.deepcopy(actual)
        changed["valid"][4] = True
        changed["dice"][4] = changed["iou"][4] = 0.
        self.assertFalse(predict.metric_comparison(changed, reference, 0)["matches"])

    def test_predict_prepared_uses_eval_and_no_grad_for_warmup_and_timed_forwards(self):
        observed = []

        class RecordingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Conv3d(1, 6, kernel_size=1)

            def forward(self, image):
                observed.append((self.training, torch.is_grad_enabled()))
                return self.layer(image)

        model = RecordingModel().train()
        prepared = SimpleNamespace(image=torch.ones(1, 3, 4, 5, requires_grad=True),
                                   target=torch.arange(60).reshape(3, 4, 5) % 6)
        prediction, metrics, timing = predict.predict_prepared(
            model, prepared, device="cpu", repeats=2)
        self.assertEqual(observed, [(False, False)] * 3)
        self.assertFalse(model.training)
        self.assertFalse(prediction.requires_grad)
        self.assertIsNone(prepared.image.grad)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertEqual(prediction.shape, prepared.target.shape)
        self.assertEqual(prediction.dtype, torch.long)
        self.assertEqual(metrics["class_ids"], [1, 2, 3, 4, 5])
        self.assertEqual(timing["warmup_forwards"], 1)
        self.assertEqual(timing["measured_forwards"], 2)
        self.assertTrue(all(np.isfinite(value) and value >= 0
                            for value in timing["forward_seconds"]))


if __name__ == "__main__":
    unittest.main()
