"""Synthetic CPU checks for epoch-boundary checkpoint continuation."""

from copy import deepcopy
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import torch

from checkpoint import load_checkpoint, save_checkpoint
from modules import SegmentationModel3D
from objectives import SoftDiceLoss
from train import train_one_epoch


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(random.setstate, random.getstate())
        rng = torch.random.fork_rng(devices=[])
        rng.__enter__()
        self.addCleanup(rng.__exit__, None, None, None)
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(2)
        random.seed(3710)
        torch.manual_seed(3710)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "latest.pt"
        self.config = dict(num_classes=3, channels=(3, 5), dropout=0.3,
                           deep_supervision=True)

    def make_model_optimizer(self, **overrides):
        model = SegmentationModel3D(**(self.config | overrides))
        return model, torch.optim.Adam(model.parameters(), lr=0.002)

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

    def test_adam_and_rng_resume_matches_uninterrupted_next_epoch(self):
        model, optimizer = self.make_model_optimizer()
        criterion = SoftDiceLoss()
        batches = [(torch.randn(2, 1, 3, 5, 7), torch.randint(3, (2, 3, 5, 7)))
                   for _ in range(2)]
        train_one_epoch(model, batches[:1], optimizer, criterion)
        saved_optimizer = deepcopy(optimizer.state_dict())
        self.assertTrue(saved_optimizer["state"])
        self.assertTrue(any(state["exp_avg"].abs().sum() > 0
                            for state in saved_optimizer["state"].values()))
        save_checkpoint(self.path, model, optimizer, completed_epochs=1,
                        best_validation_loss=0.6)

        expected_python = random.random()
        expected_torch = torch.rand(4)
        expected_result = train_one_epoch(model, batches[1:], optimizer, criterion)
        expected_rng = torch.get_rng_state().clone()
        expected_python_state = random.getstate()

        fresh, restored_optimizer = self.make_model_optimizer()
        restored_optimizer.param_groups[0]["lr"] = 0.9
        metadata = load_checkpoint(self.path, fresh, restored_optimizer)
        self.assertEqual(metadata, dict(completed_epochs=1, best_validation_loss=0.6,
                                        model_config=model.get_config()))
        self.assert_nested_equal(restored_optimizer.state_dict(), saved_optimizer)
        self.assertEqual(random.random(), expected_python)
        torch.testing.assert_close(torch.rand(4), expected_torch, rtol=0, atol=0)
        actual_result = train_one_epoch(fresh, batches[1:], restored_optimizer, criterion)
        self.assertEqual(actual_result, expected_result)
        self.assert_nested_equal(fresh.state_dict(), model.state_dict())
        self.assert_nested_equal(restored_optimizer.state_dict(), optimizer.state_dict())
        self.assertEqual(random.getstate(), expected_python_state)
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))

    def test_constructor_settings_are_checked_before_loading_weights(self):
        source, optimizer = self.make_model_optimizer()
        original_config = source.get_config()
        source.get_config()["dropout"] = 0.9
        self.assertEqual(source.get_config(), original_config)
        self.assertEqual(SegmentationModel3D(**original_config).get_config(), original_config)
        save_checkpoint(self.path, source, optimizer, completed_epochs=0)
        for settings in ({"dropout": 0.1}, {"deep_supervision": False}):
            with self.subTest(settings=settings):
                target, target_optimizer = self.make_model_optimizer(**settings)
                with patch.object(target, "load_state_dict", wraps=target.load_state_dict) as load:
                    with self.assertRaises(ValueError):
                        load_checkpoint(self.path, target, target_optimizer)
                    load.assert_not_called()

    def test_optimizer_class_and_parameter_group_order_must_match(self):
        source, _ = self.make_model_optimizer()
        parameters = list(source.parameters())
        middle = len(parameters) // 2
        optimizer = torch.optim.Adam([{"params": parameters[:middle]},
                                      {"params": parameters[middle:]}], lr=0.002)
        save_checkpoint(self.path, source, optimizer, completed_epochs=0)
        for mismatch in ("class", "group_order", "parameter_order"):
            with self.subTest(mismatch=mismatch):
                target, _ = self.make_model_optimizer()
                parameters = list(target.parameters())
                groups = [parameters[:middle], parameters[middle:]]
                if mismatch == "group_order":
                    groups.reverse()
                elif mismatch == "parameter_order":
                    groups[0].reverse()
                cls = torch.optim.SGD if mismatch == "class" else torch.optim.Adam
                target_optimizer = cls([{"params": group} for group in groups], lr=0.002)
                with patch.object(target, "load_state_dict", wraps=target.load_state_dict) as load:
                    with self.assertRaises(ValueError):
                        load_checkpoint(self.path, target, target_optimizer)
                    load.assert_not_called()

    def test_model_dtype_mismatch_is_rejected_before_loading_weights(self):
        source, optimizer = self.make_model_optimizer()
        save_checkpoint(self.path, source, optimizer, completed_epochs=0)
        target = SegmentationModel3D(**source.get_config()).double()
        target_optimizer = torch.optim.Adam(target.parameters(), lr=0.002)
        with patch.object(target, "load_state_dict", wraps=target.load_state_dict) as load:
            with self.assertRaises(ValueError):
                load_checkpoint(self.path, target, target_optimizer)
            load.assert_not_called()

    def test_optional_rng_restore_preserves_current_random_states(self):
        source, optimizer = self.make_model_optimizer()
        save_checkpoint(self.path, source, optimizer, completed_epochs=3)
        target, target_optimizer = self.make_model_optimizer()
        random.random()
        torch.rand(3)
        python_state, torch_state = random.getstate(), torch.get_rng_state().clone()
        metadata = load_checkpoint(self.path, target, target_optimizer, restore_rng=False)
        self.assertEqual(metadata["completed_epochs"], 3)
        self.assertIsNone(metadata["best_validation_loss"])
        self.assertEqual(random.getstate(), python_state)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
        self.assert_nested_equal(target.state_dict(), source.state_dict())

    def test_failed_write_preserves_previous_file_and_removes_temporary_file(self):
        model, optimizer = self.make_model_optimizer()
        save_checkpoint(self.path, model, optimizer, completed_epochs=1)
        original = self.path.read_bytes()

        def interrupted_save(payload, output):
            output.write(b"incomplete checkpoint")
            raise OSError("simulated write failure")

        with patch("checkpoint.torch.save", side_effect=interrupted_save):
            with self.assertRaises(OSError):
                save_checkpoint(self.path, model, optimizer, completed_epochs=2)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])
        self.assertEqual(load_checkpoint(self.path, model, optimizer)["completed_epochs"], 1)

    def test_invalid_schema_and_progress_are_rejected(self):
        model, optimizer = self.make_model_optimizer()
        save_checkpoint(self.path, model, optimizer, completed_epochs=1)
        payload = torch.load(self.path, map_location="cpu", weights_only=True)
        corrupt_path = self.path.with_name("invalid.pt")
        invalid_payloads = [[], {**payload, "unexpected": True},
                            {key: value for key, value in payload.items() if key != "model_config"}]
        invalid_payloads += [payload | changes for changes in (
            {"format_version": 2}, {"format_version": True}, {"completed_epochs": -1},
            {"completed_epochs": True}, {"completed_epochs": 1.5},
            {"best_validation_loss": float("nan")}, {"best_validation_loss": float("inf")},
        )]
        for index, invalid in enumerate(invalid_payloads):
            with self.subTest(payload=index):
                torch.save(invalid, corrupt_path)
                with patch.object(model, "load_state_dict", wraps=model.load_state_dict) as load:
                    with self.assertRaises(ValueError):
                        load_checkpoint(corrupt_path, model, optimizer)
                    load.assert_not_called()
        original = self.path.read_bytes()
        for progress in ({"completed_epochs": -1}, {"completed_epochs": True},
                         {"completed_epochs": 1, "best_validation_loss": float("nan")},
                         {"completed_epochs": 1, "best_validation_loss": float("inf")}):
            with self.subTest(progress=progress), self.assertRaises(ValueError):
                save_checkpoint(self.path, model, optimizer, **progress)
        self.assertEqual(self.path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
