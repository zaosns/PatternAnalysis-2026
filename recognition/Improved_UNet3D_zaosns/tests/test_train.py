"""Synthetic CPU checks for training and complete-case validation epochs."""

import math
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from modules import SegmentationModel3D
from objectives import SoftDiceLoss
from train import CrossEntropyDiceLoss, train_one_epoch, validate_one_epoch


class EpochTests(unittest.TestCase):
    def setUp(self):
        rng = torch.random.fork_rng(devices=[])
        rng.__enter__()
        self.addCleanup(rng.__exit__, None, None, None)
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(2)
        torch.manual_seed(3710)

    def test_explicit_voxel_mean_matches_original_loss_and_gradients(self):
        criterion = CrossEntropyDiceLoss()
        for shape in ((2, 3, 3, 5, 7), (3, 6, 2, 7, 4)):
            with self.subTest(shape=shape):
                logits = torch.randn(shape, dtype=torch.float64, requires_grad=True)
                reference_logits = logits.detach().clone().requires_grad_()
                target = torch.randint(shape[1], (shape[0], *shape[2:]))
                actual = criterion(logits, target)
                expected = (F.cross_entropy(reference_logits, target)
                            + SoftDiceLoss(include_background=False)(reference_logits, target))
                torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)
                actual_gradient, = torch.autograd.grad(actual, logits)
                expected_gradient, = torch.autograd.grad(expected, reference_logits)
                torch.testing.assert_close(actual_gradient, expected_gradient,
                                           rtol=1e-14, atol=1e-14)

    def test_real_segmentation_model_updates(self):
        model = SegmentationModel3D(num_classes=3, channels=(2, 3), dropout=0).eval()
        before = [parameter.detach().clone() for parameter in model.parameters()]
        images = torch.randn(2, 1, 3, 5, 7)
        target = torch.randint(3, (2, 3, 5, 7))
        result = train_one_epoch(model, [(images, target)],
                                 torch.optim.SGD(model.parameters(), lr=0.1), SoftDiceLoss())
        self.assertTrue(model.training)
        self.assertEqual(result["num_samples"], 2)
        self.assertTrue(math.isfinite(result["loss"]))
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, model.parameters())))

    def test_training_resets_gradients_and_weights_loss_by_samples(self):
        model = nn.Conv3d(1, 2, 1)
        nn.init.zeros_(model.weight)
        nn.init.zeros_(model.bias)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        batches = [(torch.ones(n, 1, 1, 1, 1), torch.zeros(n, 1, 1, 1, dtype=torch.long))
                   for n in (2, 1)]
        with patch.object(optimizer, "step", wraps=optimizer.step) as step:
            result = train_one_epoch(model, iter(batches), optimizer, lambda logits, _: logits[:, 0].mean())
        self.assertEqual(step.call_count, 2)
        self.assertEqual(result["num_samples"], 3)
        # Each step subtracts 0.1 from the first weight and bias. Batch losses
        # are therefore 0 and -0.2, with sample counts 2 and 1 respectively.
        self.assertAlmostEqual(result["loss"], -0.2 / 3, places=6)
        torch.testing.assert_close(model.weight.detach().flatten(), torch.tensor([-0.2, 0.0]))
        torch.testing.assert_close(model.bias.detach(), torch.tensor([-0.2, 0.0]))

    def test_validation_case_scores_empty_classes_and_batch_regrouping(self):
        prediction = torch.tensor([[0, 1, 0, 0], [0, 0, 0, 0],
                                   [0, 2, 1, 0], [0, 1, 1, 0]]).reshape(4, 1, 1, 4)
        target = torch.tensor([[0, 1, 1, 0], [0, 0, 0, 0],
                               [0, 0, 0, 0], [0, 1, 1, 0]]).reshape(4, 1, 1, 4)
        logits = F.one_hot(prediction, num_classes=4).movedim(-1, 1).float()
        logits *= torch.tensor([2., 4., 8., 16.]).reshape(4, 1, 1, 1, 1)
        criterion = lambda values, _: values[:, 0, 0, 0, 0].mean()
        results = []
        for cut in (1, 2, 3):
            batches = [(logits[:cut], target[:cut]), (logits[cut:], target[cut:])]
            results.append(validate_one_epoch(nn.Identity(), iter(batches), criterion,
                                               num_classes=4, include_background=False))
        nan = float("nan")
        expected = {
            "dice": torch.tensor([[2 / 3, nan, nan], [nan, nan, nan], [0., 0., nan], [1., nan, nan]]),
            "iou": torch.tensor([[0.5, nan, nan], [nan, nan, nan], [0., 0., nan], [1., nan, nan]]),
            "valid": torch.tensor([[True, False, False], [False, False, False],
                                    [True, True, False], [True, False, False]]),
            "valid_count": torch.tensor([3, 1, 0]),
            "class_ids": torch.tensor([1, 2, 3]),
            # Class 1 has three valid cases across unequal batch sizes. Its
            # case mean is (2/3 + 0 + 1)/3, not the mean of batch means.
            "mean_dice": torch.tensor([5 / 9, 0., nan], dtype=torch.float64),
            "mean_iou": torch.tensor([0.5, 0., nan], dtype=torch.float64),
        }
        for cut, result in zip((1, 2, 3), results):
            with self.subTest(cut=cut):
                self.assertAlmostEqual(result["loss"], 7.5, places=6)
                self.assertEqual(result["num_samples"], 4)
            for key, value in expected.items():
                with self.subTest(cut=cut, key=key):
                    self.assertEqual(result[key].device.type, "cpu")
                    self.assertFalse(result[key].requires_grad)
                    torch.testing.assert_close(result[key], value, equal_nan=True)
                    torch.testing.assert_close(result[key], results[0][key], equal_nan=True)

    def test_validation_uses_eval_and_no_grad_without_changing_state(self):
        model = nn.Sequential(nn.BatchNorm3d(1), nn.Dropout3d(0.5), nn.Conv3d(1, 2, 1)).train()
        before = {key: value.clone() for key, value in model.state_dict().items()}
        observations = []
        handle = model.register_forward_hook(
            lambda module, inputs, output: observations.append((module.training, torch.is_grad_enabled())))
        self.addCleanup(handle.remove)

        def criterion(logits, target):
            self.assertFalse(torch.is_grad_enabled())
            return F.cross_entropy(logits, target)

        result = validate_one_epoch(model, [(torch.randn(2, 1, 2, 3, 4),
                                           torch.zeros(2, 2, 3, 4, dtype=torch.long))],
                                    criterion, num_classes=2)
        self.assertEqual(observations, [(False, False)])
        self.assertFalse(model.training)
        self.assertEqual(result["dice"].shape, (2, 2))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        for key, old in before.items():
            torch.testing.assert_close(model.state_dict()[key], old, rtol=0, atol=0)

    def test_empty_or_invalid_batches_and_class_options_are_rejected(self):
        images = torch.ones(1, 2, 1, 1, 2)
        target = torch.zeros(1, 1, 1, 2, dtype=torch.long)
        bad_batches = [[], [(images[:0], target[:0])], [(images[0], target)],
                       [(images.long(), target)], [(images * float("nan"), target)],
                       [(images, target.float())], [(images, target[..., :1])],
                       [(images, target - 1)], [(images, target + 2)]]
        for batches in bad_batches:
            with self.subTest(batches=batches), self.assertRaises(ValueError):
                validate_one_epoch(nn.Identity(), batches, SoftDiceLoss(), num_classes=2)
        for options in ({"num_classes": True}, {"num_classes": 1}, {"num_classes": 2.0},
                        {"num_classes": 3}, {"num_classes": 2, "include_background": 1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                validate_one_epoch(nn.Identity(), [(images, target)], SoftDiceLoss(), **options)
        model = nn.Conv3d(2, 2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        for batches in bad_batches:
            with self.subTest(training=True, batches=batches), self.assertRaises(ValueError):
                train_one_epoch(model, batches, optimizer, SoftDiceLoss())

    def test_invalid_losses_never_step_optimizer(self):
        images = torch.ones(1, 1, 1, 1, 2)
        target = torch.zeros(1, 1, 1, 2, dtype=torch.long)
        criteria = [lambda logits, _: logits.mean().repeat(2),
                    lambda logits, _: logits.mean() * float("nan"),
                    lambda logits, _: logits.mean() * float("inf"),
                    lambda logits, _: torch.tensor(0, dtype=torch.long)]
        for criterion in criteria:
            model = nn.Conv3d(1, 2, 1)
            before = [parameter.detach().clone() for parameter in model.parameters()]
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            with self.subTest(criterion=criterion), patch.object(optimizer, "step") as step:
                with self.assertRaises(ValueError):
                    train_one_epoch(model, [(images, target)], optimizer, criterion)
                step.assert_not_called()
                with self.assertRaises(ValueError):
                    validate_one_epoch(model, [(images, target)], criterion, num_classes=2)
                for old, new in zip(before, model.parameters()):
                    torch.testing.assert_close(old, new.detach(), rtol=0, atol=0)

    def test_nonfinite_gradient_rejects_an_otherwise_finite_loss(self):
        class InfiniteGradient(torch.autograd.Function):
            @staticmethod
            def forward(ctx, value):
                ctx.save_for_backward(value)
                return value.mean()

            @staticmethod
            def backward(ctx, gradient):
                (value,) = ctx.saved_tensors
                return torch.full_like(value, float("inf"))

        model = nn.Conv3d(1, 2, 1)
        before = [parameter.detach().clone() for parameter in model.parameters()]
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        batches = [(torch.ones(1, 1, 1, 1, 2), torch.zeros(1, 1, 1, 2, dtype=torch.long))]
        with patch.object(optimizer, "step") as step, self.assertRaises(ValueError):
            train_one_epoch(model, batches, optimizer, lambda logits, _: InfiniteGradient.apply(logits))
        step.assert_not_called()
        for old, new in zip(before, model.parameters()):
            torch.testing.assert_close(old, new.detach(), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
