"""Synthetic CPU checks for objectives.py.

Run from the project directory: python -s -m unittest discover -s tests -v
"""

import unittest

import torch

from objectives import SoftDiceLoss, dice_iou_per_class


class TestSegmentationObjectives(unittest.TestCase):
    def setUp(self) -> None:
        # Every case has its own seed and restores the caller's CPU RNG/thread state.
        rng = torch.random.fork_rng(devices=[])
        rng.__enter__()
        self.addCleanup(rng.__exit__, None, None, None)
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(2)
        torch.manual_seed(3710)

    def test_known_scores(self) -> None:
        """Use exact masks/probabilities with independently calculated answers."""
        target = torch.tensor([
            [1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0],
            [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0],
        ]).reshape(6, 1, 1, 4)
        prediction = torch.tensor([
            [1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 0, 0],
            [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
        ]).reshape_as(target)
        result = dice_iou_per_class(prediction, target, 3, include_background=False)
        expected_dice = torch.tensor([1., 0., 2/3, 0., 0., float("nan")])
        expected_iou = torch.tensor([1., 0., 0.5, 0., 0., float("nan")])
        torch.testing.assert_close(result["dice"][:, 0], expected_dice, equal_nan=True)
        torch.testing.assert_close(result["iou"][:, 0], expected_iou, equal_nan=True)
        self.assertTrue(torch.isnan(result["dice"][:, 1]).all() and torch.isnan(result["iou"][:, 1]).all())
        self.assertTrue(torch.equal(result["valid"], torch.isfinite(result["dice"])))
        self.assertTrue(result["valid_count"].tolist() == [5, 0])
        self.assertTrue(result["class_ids"].tolist() == [1, 2])
        included = dice_iou_per_class(prediction, target, 3)
        torch.testing.assert_close(included["dice"][:, 1:], result["dice"], equal_nan=True)
        self.assertTrue(included["dice"][-1, 0] == 1 and included["class_ids"].tolist() == [0, 1, 2])

        probabilities = torch.tensor([
            [[0.2, 0.6], [0.8, 0.4]], [[0.8, 0.9], [0.2, 0.1]],
        ], dtype=torch.float64).reshape(2, 2, 1, 1, 2)
        target = torch.tensor([[1, 0], [0, 0]]).reshape(2, 1, 1, 2)
        logits = probabilities.log().requires_grad_()
        smooth = 1e-5
        loss = SoftDiceLoss(include_background=False, smooth=smooth)
        # Case 0: intersection .8, denominator 2.2. Case 1: empty GT, predicted mass .3.
        expected = 1 - ((1.6 + smooth)/(2.2 + smooth) + smooth/(0.3 + smooth))/2
        actual = loss(logits, target)
        torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.float64))
        all_classes = SoftDiceLoss(smooth=smooth)(logits[:1], target[:1])
        expected_all = 1 - ((1.2 + smooth)/(1.8 + smooth) + (1.6 + smooth)/(2.2 + smooth))/2
        torch.testing.assert_close(all_classes, torch.tensor(expected_all, dtype=torch.float64))
        actual.backward()
        self.assertTrue(logits.grad is not None and torch.isfinite(logits.grad).all())
        self.assertTrue(logits.grad[1].abs().sum() > 0, "Empty target must still penalize foreground mass.")
        self.assertTrue(torch.autograd.gradcheck(loss, (logits.detach().requires_grad_(), target), fast_mode=True))

    def test_numerics_and_inputs(self) -> None:
        """Check low-precision accumulation and reject ambiguous or invalid labels."""
        for dtype in (torch.float16, torch.bfloat16):
            logits = torch.zeros(1, 2, 1, 1, 131072, dtype=dtype, requires_grad=True)
            target = (torch.arange(131072) % 2).reshape(1, 1, 1, -1)
            value = SoftDiceLoss()(logits, target)
            self.assertTrue(value.dtype == torch.float32 and torch.isfinite(value))
            torch.testing.assert_close(value, torch.tensor(0.5))
            value.backward()
            self.assertTrue(torch.isfinite(logits.grad).all() and logits.grad.float().abs().sum() > 0)
        for sign in (-1, 1):
            logits = (sign * torch.tensor([1000., -1000.])).reshape(1, 2, 1, 1, 1).requires_grad_()
            value = SoftDiceLoss(include_background=False)(logits, torch.zeros(1, 1, 1, 1, dtype=torch.long))
            self.assertTrue(torch.isfinite(value) and 0 <= value <= 1)
            value.backward()
            self.assertTrue(torch.isfinite(logits.grad).all())

        logits = torch.zeros(1, 2, 1, 2, 2)
        target = torch.zeros(1, 1, 2, 2, dtype=torch.long)
        checks = (
            lambda: SoftDiceLoss(smooth=0),
            lambda: SoftDiceLoss(smooth=float("nan")),
            lambda: SoftDiceLoss(smooth=float("inf")),
            lambda: SoftDiceLoss(smooth=1e-40)(logits, target),
            lambda: SoftDiceLoss(smooth=1e-100)(logits, target),
            lambda: SoftDiceLoss(smooth=1e100)(logits, target),
            lambda: SoftDiceLoss(include_background="False"),
            lambda: SoftDiceLoss()(logits.long(), target),
            lambda: SoftDiceLoss()(logits + float("nan"), target),
            lambda: SoftDiceLoss()(logits[:, :1], target),
            lambda: SoftDiceLoss()(logits, target.float()),
            lambda: SoftDiceLoss()(logits, target - 1),
            lambda: SoftDiceLoss()(logits, target + 2),
            lambda: SoftDiceLoss()(logits, target[..., :1]),
            lambda: SoftDiceLoss()(logits[:0], target[:0]),
            lambda: dice_iou_per_class(target.float(), target, 2),
            lambda: dice_iou_per_class(target, target, True),
            lambda: dice_iou_per_class(target, target[..., :1], 2),
        )
        for check in checks:
            try:
                check()
            except ValueError:
                pass
            else:
                raise AssertionError("An invalid loss/metric input was accepted.")

    def test_model_integration(self) -> None:
        """Check Dice gradients reach all instantiated model heads and the backbone."""
        from modules import SegmentationModel3D

        for supervision in (False, True):
            model = SegmentationModel3D(3, channels=(3, 5, 7, 9), dropout=0.0,
                                        deep_supervision=supervision)
            image = torch.randn(1, 1, 9, 11, 13, requires_grad=True)
            target = torch.randint(3, (1, 9, 11, 13))
            logits = model(image)
            value = SoftDiceLoss(include_background=False)(logits, target)
            self.assertTrue(torch.isfinite(value) and 0 <= value <= 1)
            value.backward()
            self.assertTrue(image.grad is not None and torch.isfinite(image.grad).all())
            self.assertTrue(image.grad.abs().sum() > 0)
            for name, parameter in model.named_parameters():
                self.assertTrue(parameter.grad is not None and torch.isfinite(parameter.grad).all(), name)
                if parameter.ndim == 5:
                    self.assertTrue(parameter.grad.abs().sum() > 0, f"Zero convolution gradient: {name}")
            metrics = dice_iou_per_class(logits.argmax(dim=1), target, 3, include_background=False)
            self.assertTrue(metrics["dice"].shape == (1, 2) and metrics["valid"].all())
            for key in ("dice", "iou"):
                self.assertTrue(((metrics[key] >= 0) & (metrics[key] <= 1)).all())


if __name__ == "__main__":
    unittest.main()
