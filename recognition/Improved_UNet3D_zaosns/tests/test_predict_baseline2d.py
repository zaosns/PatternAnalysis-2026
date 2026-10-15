"""Checkpoint-bound 2D validation prediction and voxel-count regression checks."""

from contextlib import redirect_stdout
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

import predict
import prediction_panels
import train_baseline2d


class Baseline2DPredictionTests(unittest.TestCase):
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
        for index, case_id in enumerate(("train", "validation_a", "validation_b",
                                        "validation_c", "test")):
            image_path = self.root / f"{case_id}_image.nii"
            label_path = self.root / f"{case_id}_label.nii"
            if case_id == "test":
                image_path.write_text("Held-out image must never be opened as NIfTI.\n")
                label_path.write_text("Held-out label must never be opened as NIfTI.\n")
            else:
                coordinates = np.indices((9 + 2 * index, 11, 7 + 2 * index))
                image = (coordinates[0] * 3 + coordinates[1] * 2
                         + coordinates[2] + 1).astype(np.float32)
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
            format_version=1, seed=3710, validation_patients=3, test_patients=1,
            patient_key_rule="Independent synthetic patients",
            manifest_sha256=hashlib.sha256(self.manifest.read_bytes()).hexdigest(),
            train=["train"], validation=["validation_a", "validation_b", "validation_c"],
            test=["test"],
        )), encoding="utf-8")
        self.run_dir = self.root / "training"
        renderer_patch = patch.object(prediction_panels, "render_case_panels",
                                      side_effect=self.fake_render)
        self.render = renderer_patch.start()
        self.addCleanup(renderer_patch.stop)

    @staticmethod
    def fake_render(prepared, prediction, output, *, case_label, metrics, **options):
        Path(output).write_bytes(b"synthetic renderer output\n")
        return dict(case_label=case_label, shape=list(prediction.shape), **options)

    def train_fixture(self):
        args = train_baseline2d.build_parser().parse_args([
            "--manifest", str(self.manifest), "--split", str(self.split),
            "--output-dir", str(self.run_dir), "--epochs", "2", "--device", "cpu",
            "--num-classes", "6", "--channels", "2", "3", "--learning-rate", "0.002",
            "--downsample-factor", "2", "--threads", "1", "--seed", "3710",
            "--deterministic",
        ])
        with redirect_stdout(io.StringIO()):
            return train_baseline2d.run_training(args)

    def arguments(self, output, *, case_ids=("validation_b", "validation_a"), extra=()):
        return predict.build_parser().parse_args([
            "--run-dir", str(self.run_dir), "--manifest", str(self.manifest),
            "--split", str(self.split), "--case-ids", *case_ids,
            "--output-dir", str(output), "--device", "cpu", "--threads", "1",
            "--repeats", "1", *extra,
        ])

    def reject_before_voxels(self, output, **options):
        with patch.object(predict, "load_nifti_case") as loader, \
                self.assertRaises(ValueError):
            predict.run_prediction(self.arguments(output, **options))
        loader.assert_not_called()
        self.render.assert_not_called()
        self.assertFalse(output.exists())

    def test_real_2d_checkpoint_metrics_and_counts_preserve_validation_isolation(self):
        state = self.train_fixture()
        original_run = {path.name: path.read_bytes() for path in self.run_dir.iterdir()
                        if path.is_file()}
        for case_id in ("train", "validation_c"):
            for suffix in ("image", "label"):
                (self.root / f"{case_id}_{suffix}.nii").write_text("Unselected voxels must not load.\n")
        output = self.root / "prediction"
        expected_sha = hashlib.sha256((self.run_dir / state["best_checkpoint"]).read_bytes()).hexdigest()
        with patch.object(predict, "load_nifti_case", wraps=predict.load_nifti_case) as loader, \
                redirect_stdout(io.StringIO()):
            result = predict.run_prediction(self.arguments(output, extra=(
                "--checkpoint-sha256", expected_sha,
            )))
        self.assertEqual([call.args[0].case_id for call in loader.call_args_list],
                         ["validation_b", "validation_a"])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["model_kind"], "slice_wise_unet2d")
        self.assertEqual(result["partition"], "validation")
        self.assertEqual(result["selected_case_ids"], ["validation_b", "validation_a"])
        self.assertEqual(result["panel_case_ids"], result["selected_case_ids"])
        self.assertEqual(result["checkpoint_sha256"], expected_sha)
        self.assertEqual(result["best_epoch"], state["best_epoch"])
        self.assertTrue(result["all_saved_metrics_match"])
        reference = state["history"][state["best_epoch"] - 1]["validation"]
        for row in result["cases"]:
            index = reference["case_ids"].index(row["case_id"])
            for key in ("dice", "iou", "valid"):
                self.assertEqual(row["metrics"][key], reference[key][index])
            self.assertEqual(row["saved_metric_comparison"]["maximum_absolute_difference"], 0.)
            counts = row["metrics"]["voxel_counts"]
            matrix = np.asarray(counts["confusion_matrix"])
            self.assertEqual(counts["class_ids"], list(range(6)))
            self.assertEqual(matrix.shape, (6, 6))
            self.assertEqual(int(matrix.sum()), int(np.prod(row["shape_dhw"])))
            self.assertEqual(matrix.sum(axis=1).tolist(), counts["target_voxels"])
            self.assertEqual(matrix.sum(axis=0).tolist(), counts["predicted_voxels"])
            self.assertEqual(matrix.diagonal().tolist(), counts["true_positive"])
            for class_id in range(1, 6):
                tp = counts["true_positive"][class_id]
                fp = counts["false_positive"][class_id]
                fn = counts["false_negative"][class_id]
                self.assertEqual(tp + fn, counts["target_voxels"][class_id])
                self.assertEqual(tp + fp, counts["predicted_voxels"][class_id])
                if 2 * tp + fp + fn:
                    self.assertAlmostEqual(row["metrics"]["dice"][class_id - 1],
                                           2 * tp / (2 * tp + fp + fn), places=6)
                    self.assertAlmostEqual(row["metrics"]["iou"][class_id - 1],
                                           tp / (tp + fp + fn), places=6)
                else:
                    self.assertIsNone(row["metrics"]["dice"][class_id - 1])
            self.assertEqual(row["figure_sha256"], hashlib.sha256(
                (output / row["figure"]).read_bytes()).hexdigest())
        self.assertEqual(self.render.call_count, 2)
        self.assertEqual(json.loads((output / "prediction.json").read_text()), result)
        self.assertEqual({path.name: path.read_bytes() for path in self.run_dir.iterdir()
                          if path.is_file()}, original_run)

    def test_only_requested_panels_render_but_every_selected_case_keeps_counts(self):
        self.train_fixture()
        output = self.root / "selected_panels"
        with redirect_stdout(io.StringIO()):
            result = predict.run_prediction(self.arguments(output, extra=(
                "--panel-case-ids", "validation_a", "--include-rectum-panels",
            )))
        self.assertEqual(result["panel_case_ids"], ["validation_a"])
        self.assertTrue(result["include_rectum_panels"])
        self.assertEqual(len(result["cases"]), 2)
        self.render.assert_called_once()
        self.assertTrue(self.render.call_args.kwargs["include_rectum"])
        skipped, rendered = result["cases"]
        self.assertEqual(skipped["case_id"], "validation_b")
        for key in ("panel", "figure", "figure_sha256"):
            self.assertIsNone(skipped[key])
        self.assertEqual(rendered["case_id"], "validation_a")
        self.assertEqual(rendered["figure"], "validation-02.png")
        self.assertEqual([path.name for path in output.glob("*.png")], ["validation-02.png"])
        for row in result["cases"]:
            self.assertIn("voxel_counts", row["metrics"])
            self.assertTrue(row["saved_metric_comparison"]["matches"])

    def test_invalid_or_duplicate_panel_requests_fail_before_any_voxel_read(self):
        self.train_fixture()
        for index, panel_ids in enumerate((("validation_c",), ("train",), ("test",),
                                          ("missing",), ("validation_a", "validation_a"))):
            with self.subTest(panel_ids=panel_ids):
                self.reject_before_voxels(self.root / f"bad-panel-{index}",
                                         extra=("--panel-case-ids", *panel_ids))

    def test_hand_counted_confusion_separates_absent_and_disjoint_predictions(self):
        target = torch.tensor([0, 0, 1, 1, 2, 2, 4, 4, 5, 5]).reshape(1, 2, 5)
        prediction = torch.tensor([0, 5, 1, 3, 2, 5, 0, 0, 0, 0]).reshape(1, 2, 5)
        result = predict.voxel_confusion(prediction, target, num_classes=6)
        expected = dict(class_ids=list(range(6)), confusion_matrix=[
            [1, 0, 0, 0, 0, 1], [0, 1, 0, 1, 0, 0], [0, 0, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0]],
            target_voxels=[2, 2, 2, 0, 2, 2], predicted_voxels=[5, 1, 1, 1, 0, 2],
            true_positive=[1, 1, 1, 0, 0, 0], false_positive=[4, 0, 0, 1, 0, 2],
            false_negative=[1, 1, 1, 0, 2, 2])
        self.assertEqual(result, expected)
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), expected)
        # Classes 4 and 5 both have zero Dice, with distinct failure mechanisms.
        self.assertEqual(result["predicted_voxels"][4], 0)
        self.assertEqual(result["predicted_voxels"][5], 2)
        self.assertEqual(result["true_positive"][4:], [0, 0])

    def test_voxel_counts_reject_invalid_target_prediction_and_class_contracts(self):
        valid = torch.zeros((2, 3, 4), dtype=torch.long)
        invalid_masks = [valid.float(), valid.bool(), torch.full_like(valid, -1),
                         torch.full_like(valid, 6), torch.full_like(valid.float(), float("nan")),
                         torch.zeros((2, 3, 5), dtype=torch.long), valid[0], valid.unsqueeze(0),
                         valid[:0], valid.tolist()]
        for index, bad in enumerate(invalid_masks):
            for side in ("target", "prediction"):
                with self.subTest(index=index, side=side):
                    with self.assertRaises((ValueError, TypeError)):
                        predict.voxel_confusion(bad if side == "prediction" else valid,
                                                bad if side == "target" else valid, num_classes=6)
        for classes in (True, 0, -1, 6.0):
            with self.subTest(num_classes=classes):
                with self.assertRaises((ValueError, TypeError)):
                    predict.voxel_confusion(valid, valid, num_classes=classes)

    def test_mixed_run_and_payload_architecture_labels_are_rejected_before_voxels(self):
        state = self.train_fixture()
        run_path = self.run_dir / "run.json"
        checkpoint = self.run_dir / state["best_checkpoint"]
        original_run, original_checkpoint = run_path.read_bytes(), checkpoint.read_bytes()
        for index, problem in enumerate(("run_missing", "run_unknown", "run_3d",
                                         "payload_missing", "payload_3d")):
            with self.subTest(problem=problem):
                candidate = json.loads(original_run)
                payload = torch.load(io.BytesIO(original_checkpoint), map_location="cpu", weights_only=True)
                if problem == "run_missing":
                    del candidate["config"]["model_kind"]
                elif problem.startswith("run_"):
                    candidate["config"]["model_kind"] = ("legacy_unet3d" if problem == "run_3d"
                                                            else "unknown_architecture")
                elif problem == "payload_missing":
                    del payload["model_kind"]
                else:
                    payload["model_kind"] = "legacy_unet3d"
                run_path.write_text(json.dumps(candidate), encoding="utf-8")
                torch.save(payload, checkpoint)
                self.reject_before_voxels(self.root / f"architecture-{index}")
        run_path.write_bytes(original_run)
        checkpoint.write_bytes(original_checkpoint)

    def test_baseline_source_changes_are_bound_before_voxel_reads(self):
        self.train_fixture()
        read_bytes = Path.read_bytes
        source_dir = Path(predict.__file__).resolve().parent
        for index, name in enumerate(("baseline2d.py", "train_baseline2d.py")):
            changed_source = source_dir / name

            def simulated_edit(path):
                original = read_bytes(path)
                return original + b"\n# simulated source edit\n" if path == changed_source else original

            with self.subTest(source=name), patch.object(Path, "read_bytes", new=simulated_edit):
                self.reject_before_voxels(self.root / f"source-{index}")

    def test_full_checkpoint_configuration_and_metadata_hash_mismatch_reject_early(self):
        state = self.train_fixture()
        run_path = self.run_dir / "run.json"
        checkpoint = self.run_dir / state["best_checkpoint"]
        original_run, original_checkpoint = run_path.read_bytes(), checkpoint.read_bytes()
        for index, problem in enumerate(("checkpoint_config", "run_config", "manifest_hash",
                                         "split_hash", "resolved_hash", "checkpoint_hash")):
            with self.subTest(problem=problem):
                candidate = json.loads(original_run)
                payload = torch.load(io.BytesIO(original_checkpoint), map_location="cpu", weights_only=True)
                extra = ()
                if problem == "checkpoint_config":
                    payload["config"]["seed"] += 1
                elif problem == "run_config":
                    candidate["config"]["learning_rate"] *= 2
                elif problem == "resolved_hash":
                    candidate["config"]["resolved_records_sha256"] = "0" * 64
                elif problem == "checkpoint_hash":
                    extra = ("--checkpoint-sha256", "0" * 64)
                else:
                    candidate["config"][problem.replace("_hash", "_sha256")] = "0" * 64
                run_path.write_text(json.dumps(candidate), encoding="utf-8")
                torch.save(payload, checkpoint)
                self.reject_before_voxels(self.root / f"configuration-{index}", extra=extra)
        run_path.write_bytes(original_run)
        checkpoint.write_bytes(original_checkpoint)


if __name__ == "__main__":
    unittest.main()
