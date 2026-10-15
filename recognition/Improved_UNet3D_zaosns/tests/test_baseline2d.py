"""Synthetic CPU checks for the conventional slice-wise 2D U-Net baseline."""

import io
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from baseline2d import SliceWiseUNet2D, UNet2D


class TestBaseline2D(unittest.TestCase):
    def setUp(self) -> None:
        rng = torch.random.fork_rng(devices=[])
        rng.__enter__()
        self.addCleanup(rng.__exit__, None, None, None)
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(2)
        torch.manual_seed(3710)

    def test_shapes_for_even_odd_and_thin_inputs(self) -> None:
        model = UNet2D(num_classes=3, in_channels=2, channels=(2, 4, 6)).eval()
        with torch.no_grad():
            for height, width in ((12, 16), (13, 19), (4, 8), (9, 5)):
                with self.subTest(height=height, width=width):
                    logits = model(torch.randn(2, 2, height, width))
                    self.assertEqual(logits.shape, (2, 3, height, width))
                    self.assertTrue(torch.isfinite(logits).all())

    def test_raw_logits_are_not_probabilities(self) -> None:
        model = SliceWiseUNet2D(num_classes=3, channels=(2, 4)).eval()
        with torch.no_grad():
            model.model.head.weight.zero_()
            model.model.head.bias.copy_(torch.tensor([-2.0, 3.0, 0.5]))
            logits = model(torch.randn(2, 1, 3, 9, 11))
        expected = torch.tensor([-2.0, 3.0, 0.5]).view(1, 3, 1, 1, 1).expand_as(logits)
        torch.testing.assert_close(logits, expected, rtol=0, atol=0)

    def test_whole_volume_loss_backpropagates_through_all_blocks(self) -> None:
        model = SliceWiseUNet2D(num_classes=3, channels=(2, 4))
        images = torch.randn(2, 1, 3, 9, 11, requires_grad=True)
        original = images.detach().clone()
        labels = torch.randint(3, (2, 3, 9, 11))
        logits = model(images)
        # One whole-volume objective, rather than separate slice-mean losses.
        loss = F.cross_entropy(logits, labels, reduction="none").mean()
        loss.backward()
        self.assertIsNotNone(images.grad)
        self.assertTrue(torch.isfinite(images.grad).all())
        self.assertGreater(images.grad.abs().sum().item(), 0)
        for name, parameter in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                if parameter.ndim == 4:
                    self.assertGreater(parameter.grad.abs().sum().item(), 0)
        torch.testing.assert_close(images.detach(), original, rtol=0, atol=0)

    def test_wrapper_matches_separate_slices_and_keeps_case_depth_order(self) -> None:
        model = SliceWiseUNet2D(num_classes=3, in_channels=2, channels=(2, 4)).eval()
        # Transposition makes the volume noncontiguous, exercising reshape order.
        images = torch.randn(2, 2, 3, 13, 9).transpose(-1, -2)
        self.assertFalse(images.is_contiguous())
        with torch.no_grad():
            actual = model(images)
            expected = torch.stack(
                [torch.stack([model.model(images[n:n + 1, :, d])[0]
                              for d in range(images.shape[2])], dim=1)
                 for n in range(images.shape[0])]
            )
        self.assertEqual(actual.shape, (2, 3, 3, 9, 13))
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)

    def test_slices_and_cases_cannot_influence_each_other(self) -> None:
        model = SliceWiseUNet2D(num_classes=3, channels=(2, 4))
        images = torch.randn(2, 1, 4, 9, 11)
        changed = images.clone()
        changed[1, :, 2] = torch.randn_like(changed[1, :, 2]) * 3
        for training in (False, True):
            with self.subTest(training=training), torch.no_grad():
                model.train(training)
                before = model(images)
                after = model(changed)
                torch.testing.assert_close(before[0], after[0], rtol=0, atol=0)
                for depth in (0, 1, 3):
                    torch.testing.assert_close(before[1, :, depth], after[1, :, depth], rtol=0, atol=0)
                self.assertGreater((before[1, :, 2] - after[1, :, 2]).abs().max().item(), 0)

    def test_config_and_state_round_trip(self) -> None:
        for constructor in (UNet2D, SliceWiseUNet2D):
            with self.subTest(constructor=constructor.__name__):
                original = constructor(num_classes=3, in_channels=2, channels=(2, 4)).eval()
                config = original.get_config()
                self.assertEqual(config, {"num_classes": 3, "in_channels": 2, "channels": (2, 4)})
                buffer = io.BytesIO()
                torch.save(original.state_dict(), buffer)
                buffer.seek(0)
                restored = constructor(**config).eval()
                restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
                shape = (2, 2, 9, 11) if constructor is UNet2D else (2, 2, 3, 9, 11)
                images = torch.randn(*shape)
                with torch.no_grad():
                    torch.testing.assert_close(original(images), restored(images), rtol=0, atol=0)

    def test_invalid_configuration_is_rejected(self) -> None:
        invalid = (
            {"num_classes": 1}, {"num_classes": True}, {"num_classes": 2.5},
            {"in_channels": 0}, {"in_channels": False}, {"in_channels": 1.0},
            {"channels": ()}, {"channels": (2,)}, {"channels": (2, 0)},
            {"channels": (2, True)}, {"channels": (2, 4.0)},
            {"channels": None}, {"channels": 3}, {"channels": "24"},
        )
        for constructor in (UNet2D, SliceWiseUNet2D):
            for config in invalid:
                with self.subTest(constructor=constructor.__name__, config=config):
                    with self.assertRaises(ValueError):
                        constructor(**config)

    def test_unsafe_image_inputs_are_rejected(self) -> None:
        model = UNet2D(channels=(2, 4))
        invalid = (
            None, torch.empty(1, 1, 1, 9, 11), torch.empty(1, 2, 9, 11),
            torch.empty(0, 1, 9, 11), torch.empty(1, 1, 0, 11),
            torch.empty(1, 1, 1, 11), torch.empty(1, 1, 3, 3),
            torch.ones(1, 1, 9, 11, dtype=torch.long),
            torch.ones(1, 1, 9, 11, dtype=torch.complex64),
        )
        for images in invalid:
            with self.subTest(shape=None if images is None else images.shape):
                with self.assertRaises(ValueError):
                    model(images)

    def test_unsafe_volume_inputs_are_rejected(self) -> None:
        model = SliceWiseUNet2D(channels=(2, 4))
        invalid = (
            None, torch.empty(1, 1, 9, 11), torch.empty(1, 2, 3, 9, 11),
            torch.empty(0, 1, 3, 9, 11), torch.empty(1, 1, 0, 9, 11),
            torch.empty(1, 1, 3, 0, 11), torch.empty(1, 1, 3, 3, 3),
            torch.ones(1, 1, 3, 9, 11, dtype=torch.long),
        )
        for images in invalid:
            with self.subTest(shape=None if images is None else images.shape):
                with self.assertRaises(ValueError):
                    model(images)

    def test_architecture_has_only_2d_operators_and_no_dropout(self) -> None:
        model = SliceWiseUNet2D()
        modules = tuple(model.modules())
        self.assertTrue(any(isinstance(module, nn.ConvTranspose2d) for module in modules))
        self.assertTrue(any(isinstance(module, nn.MaxPool2d) for module in modules))
        forbidden = (nn.Conv3d, nn.ConvTranspose3d, nn.MaxPool3d, nn.InstanceNorm3d,
                     nn.BatchNorm2d, nn.BatchNorm3d, nn.Dropout, nn.Dropout2d, nn.Dropout3d)
        self.assertFalse(any(isinstance(module, forbidden) for module in modules))
        norms = [module for module in modules if isinstance(module, nn.InstanceNorm2d)]
        self.assertTrue(norms)
        self.assertTrue(all(module.affine and not module.track_running_stats for module in norms))
        self.assertEqual(model.get_config()["channels"], (16, 32, 64, 128))


if __name__ == "__main__":
    unittest.main()
