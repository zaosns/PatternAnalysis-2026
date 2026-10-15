"""Synthetic CPU checks for modules.py.

Run from the project directory: python -s -m unittest discover -s tests -v
"""

import unittest

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from modules import (
    Decoder3D,
    DeepSupervisionHead3D,
    Encoder3D,
    ResidualBlock3D,
    SegmentationHead3D,
    SegmentationModel3D,
)


class TestModelComponents(unittest.TestCase):
    def setUp(self) -> None:
        # Every case has its own seed and restores the caller's CPU RNG/thread state.
        rng = torch.random.fork_rng(devices=[])
        rng.__enter__()
        self.addCleanup(rng.__exit__, None, None, None)
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(2)
        torch.manual_seed(3710)

    def test_deep_supervision(self) -> None:
        """Check a hand-computed odd-size sum and gradients into every scale."""
        widths = (2, 3, 4)
        head = DeepSupervisionHead3D(widths, num_classes=2)
        features = []
        patterns = (torch.ones(9), torch.full((5,), 2.0), torch.arange(3, dtype=torch.float32))
        for width, pattern in zip(widths, patterns):
            feature = torch.zeros(1, width, 1, 1, pattern.numel())
            feature[:, 0] = pattern
            features.append(feature.requires_grad_())
        features = tuple(features)
        originals = tuple(feature.detach().clone() for feature in features)
        with torch.no_grad():
            for projection in head.heads:
                projection.projection.weight.zero_()
                projection.projection.bias.zero_()
                projection.projection.weight[0, 0, 0, 0, 0] = 1
                projection.projection.weight[1, 0, 0, 0, 0] = -1

        logits = head(features)
        # 3 -> 5 -> 9 nearest resizing differs from directly resizing 3 -> 9.
        values = torch.tensor([3., 3., 3., 3., 4., 4., 4., 4., 5.])
        expected = torch.stack((values, -values)).reshape(1, 2, 1, 1, 9)
        torch.testing.assert_close(logits, expected, rtol=0, atol=0)
        F.cross_entropy(logits, torch.ones(1, 1, 1, 9, dtype=torch.long)).backward()
        for feature, original in zip(features, originals):
            self.assertTrue(feature.grad is not None and torch.isfinite(feature.grad).all())
            self.assertTrue(feature.grad.abs().sum() > 0, "A decoder scale did not receive gradients.")
            torch.testing.assert_close(feature.detach(), original, rtol=0, atol=0)
        for name, parameter in head.named_parameters():
            self.assertTrue(parameter.grad is not None and torch.isfinite(parameter.grad).all(), name)
            self.assertTrue(parameter.grad.abs().sum() > 0, f"A segmentation head is disconnected: {name}")

        cases = (
            (features[:-1], "decoder features"),
            ((features[0][0], *features[1:]), "feature 0 must have shape"),
            ((features[0], features[1][:, :2], features[2]), "feature 1 must have shape"),
            ((features[0], features[1].expand(2, -1, -1, -1, -1), features[2]), "batch size"),
            ((features[0], features[1][..., :4], features[2]), "spatial pyramid"),
        )
        for invalid, message in cases:
            try:
                head(invalid)
            except ValueError as error:
                self.assertTrue(message in str(error), str(error))
            else:
                raise AssertionError(f"Expected an error containing: {message}")
        try:
            DeepSupervisionHead3D((), num_classes=2)
        except ValueError:
            pass
        else:
            raise AssertionError("An empty set of deep-supervision heads must be rejected.")

        # With one decoder level, enabled and disabled modes must be equivalent.
        summed = SegmentationModel3D(2, channels=(3, 5), dropout=0.0).eval()
        single = SegmentationModel3D(2, channels=(3, 5), dropout=0.0, deep_supervision=False).eval()
        single.encoder.load_state_dict(summed.encoder.state_dict())
        single.decoder.load_state_dict(summed.decoder.state_dict())
        single.head.load_state_dict(summed.head.heads[0].state_dict())
        with torch.no_grad():
            x = torch.randn(1, 1, 3, 5, 7)
            torch.testing.assert_close(summed(x), single(x), rtol=0, atol=0)
            voxel = torch.randn(1, 3, 1, 1, 1)
            torch.testing.assert_close(summed.head((voxel,)), single.head(voxel), rtol=0, atol=0)

    def test_segmentation(self) -> None:
        """Check the logits contract and cross-entropy gradients through the model."""
        # A known constant projection detects accidental softmax/sigmoid/argmax.
        head = SegmentationHead3D(3, 2)
        with torch.no_grad():
            head.projection.weight.zero_()
            head.projection.bias.copy_(torch.tensor([-2.0, 3.0]))
            for spatial in ((1, 3, 5), (1, 1, 1)):
                scores = head(torch.randn(2, 3, *spatial))
                expected = torch.tensor([-2.0, 3.0]).view(1, 2, 1, 1, 1).expand(2, 2, *spatial)
                torch.testing.assert_close(scores, expected, rtol=0, atol=0)

        for classes in (0, 1, 2.5, True):
            try:
                SegmentationHead3D(3, classes)
            except ValueError as error:
                self.assertTrue("num_classes" in str(error))
            else:
                raise AssertionError(f"Accepted invalid num_classes={classes}.")
        for shape in ((3, 3, 5, 7), (1, 4, 3, 5, 7), (1, 3, 0, 5, 7)):
            try:
                head(torch.empty(*shape))
            except ValueError as error:
                self.assertTrue("Head input" in str(error))
            else:
                raise AssertionError(f"Head accepted invalid input shape {shape}.")

        # Both class counts are synthetic examples, not assumptions about HipMRI.
        widths = (3, 5, 7, 9)
        cases = (
            ((1, 1, 8, 10, 12), 2, False),
            ((2, 2, 9, 11, 13), 4, False),
            ((1, 1, 8, 10, 12), 2, True),
            ((2, 2, 9, 11, 13), 4, True),
        )
        for shape, classes, supervision in cases:
            model = SegmentationModel3D(
                classes, shape[1], widths, dropout=0.0, deep_supervision=supervision
            )
            x = torch.randn(*shape, requires_grad=True)
            original = x.detach().clone()
            logits = model(x)
            self.assertTrue(logits.shape == (shape[0], classes, *shape[2:]))
            self.assertTrue(torch.isfinite(logits).all())
            # One integer class index per voxel, without a channel dimension.
            target = torch.randint(classes, (shape[0], *shape[2:]))
            loss = F.cross_entropy(logits, target)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0)
            for name, parameter in model.named_parameters():
                self.assertTrue(parameter.grad is not None, f"Missing segmentation gradient: {name}")
                self.assertTrue(torch.isfinite(parameter.grad).all(), f"Non-finite gradient: {name}")
                if parameter.ndim == 5:
                    self.assertTrue(parameter.grad.abs().sum() > 0, f"Zero convolution gradient: {name}")
            torch.testing.assert_close(x.detach(), original, rtol=0, atol=0)

            model.eval()
            restored = SegmentationModel3D(
                classes, shape[1], widths, dropout=0.0, deep_supervision=supervision
            ).eval()
            restored.load_state_dict(model.state_dict())
            with torch.no_grad():
                expected = model(x)
                torch.testing.assert_close(restored(x), expected)
                torch.testing.assert_close(model(x), expected)
                self.assertTrue(expected.argmax(dim=1).shape == target.shape)

        with torch.no_grad():
            default = SegmentationModel3D(num_classes=2).eval()
            scores = default(torch.randn(1, 1, 9, 11, 13))
            self.assertTrue(scores.shape == (1, 2, 9, 11, 13) and torch.isfinite(scores).all())
            thin = SegmentationModel3D(num_classes=2, channels=(3, 5), dropout=0.0).eval()
            scores = thin(torch.randn(1, 1, 1, 1, 3))
            self.assertTrue(scores.shape == (1, 2, 1, 1, 3) and torch.isfinite(scores).all())

    def test_decoder(self) -> None:
        """Check encoder/decoder integration and independent gradients for each skip."""
        widths = (3, 5, 7)
        for shape in ((1, 1, 12, 16, 20), (2, 1, 9, 11, 13)):
            model = nn.Sequential(Encoder3D(channels=widths, dropout=0.0), Decoder3D(widths))
            x = torch.randn(*shape, requires_grad=True)
            decoded = model(x)
            expected_shapes = (
                (shape[0], widths[0], *shape[2:]),
                (shape[0], widths[1], *((size + 1) // 2 for size in shape[2:])),
            )
            self.assertTrue(isinstance(decoded, tuple) and len(decoded) == len(expected_shapes))
            for feature, expected in zip(decoded, expected_shapes):
                self.assertTrue(feature.shape == expected and torch.isfinite(feature).all())
            target = torch.randn_like(decoded[0])
            (decoded[0] - target).square().mean().backward()
            self.assertTrue(x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0)
            for name, parameter in model.named_parameters():
                self.assertTrue(parameter.grad is not None, f"Missing encoder/decoder gradient: {name}")
                self.assertTrue(torch.isfinite(parameter.grad).all(), f"Non-finite gradient: {name}")
                if parameter.ndim == 5:
                    self.assertTrue(parameter.grad.abs().sum() > 0, f"Zero convolution gradient: {name}")

            model.eval()
            restored = nn.Sequential(Encoder3D(channels=widths, dropout=0.0), Decoder3D(widths)).eval()
            restored.load_state_dict(model.state_dict())
            with torch.no_grad():
                for expected, actual in zip(model(x), restored(x)):
                    torch.testing.assert_close(actual, expected)

            # Independent leaves prevent the encoder path from masking an unused skip.
            leaves = tuple(f.detach().requires_grad_() for f in model[0](x.detach()))
            originals = tuple(f.detach().clone() for f in leaves)
            (model[1](leaves)[0] - target).square().mean().backward()
            for leaf, original in zip(leaves, originals):
                self.assertTrue(leaf.grad is not None and torch.isfinite(leaf.grad).all())
                self.assertTrue(leaf.grad.abs().sum() > 0)
                torch.testing.assert_close(leaf.detach(), original, rtol=0, atol=0)

        with torch.no_grad():
            thin = Decoder3D(widths)(Encoder3D(channels=widths, dropout=0.0)(torch.randn(1, 1, 1, 1, 5)))
            self.assertTrue([tuple(f.shape[2:]) for f in thin] == [(1, 1, 5), (1, 1, 3)])
            minimal = Decoder3D((2, 4))(
                Encoder3D(channels=(2, 4), dropout=0.0)(torch.randn(1, 1, 3, 5, 7))
            )
            self.assertTrue(len(minimal) == 1 and minimal[0].shape == (1, 2, 3, 5, 7))
        try:
            Decoder3D((4,))
        except ValueError:
            pass
        else:
            raise AssertionError("A decoder must have at least two encoder levels.")

    def test_decoder_inputs(self) -> None:
        """Keep the default configuration and malformed feature checks repeatable."""
        with torch.no_grad():
            model = nn.Sequential(Encoder3D(), Decoder3D()).eval()
            outputs = model(torch.randn(1, 1, 9, 11, 13))
            self.assertTrue([tuple(f.shape) for f in outputs] == [
                (1, 16, 9, 11, 13), (1, 32, 5, 6, 7), (1, 64, 3, 3, 4)
            ])
            self.assertTrue(all(torch.isfinite(f).all() for f in outputs))

        decoder = Decoder3D((3, 5, 7))
        first = torch.randn(1, 3, 5, 7, 9)
        middle = torch.randn(1, 5, 3, 4, 5)
        last = torch.randn(1, 7, 2, 2, 3)
        cases = (
            ("feature count", (first, middle), "encoder features"),
            ("rank", (first[0], middle, last), "feature 0 must have shape"),
            ("channels", (first, middle[:, :4], last), "feature 1 must have shape"),
            ("batch", (first, middle.expand(2, -1, -1, -1, -1), last), "batch size"),
            ("spatial size", (first, middle[:, :, :2], last), "spatial pyramid"),
            ("single voxel", (first, middle, last[:, :, :1, :1, :1]), "spatial voxel"),
        )
        for label, features, message in cases:
            try:
                decoder(features)
            except ValueError as error:
                self.assertTrue(message in str(error), f"Wrong error for {label}: {error}")
            else:
                raise AssertionError(f"Decoder accepted an invalid {label}.")

        # Equal channel counts ensure this catches spatial order, not just channels.
        equal_width = (first, middle[:, :3], last[:, :3])
        try:
            Decoder3D((3, 3, 3))(tuple(reversed(equal_width)))
        except ValueError as error:
            self.assertTrue("spatial pyramid" in str(error))
        else:
            raise AssertionError("Decoder accepted reversed encoder features.")

    def test_encoder(self) -> None:
        """Check feature shapes and gradient flow from the bottleneck to the input."""
        widths = (4, 8, 16)
        cases = (
            ((1, 1, 12, 16, 20), ((12, 16, 20), (6, 8, 10), (3, 4, 5))),
            ((2, 2, 9, 11, 13), ((9, 11, 13), (5, 6, 7), (3, 3, 4))),
        )
        for input_shape, spatial_shapes in cases:
            encoder = Encoder3D(input_shape[1], channels=widths, dropout=0.0)
            x = torch.randn(*input_shape, requires_grad=True)
            original = x.detach().clone()
            features = encoder(x)
            self.assertTrue(isinstance(features, tuple) and len(features) == len(widths))
            for feature, width, spatial_shape in zip(features, widths, spatial_shapes):
                self.assertTrue(feature.shape == (input_shape[0], width, *spatial_shape))
                self.assertTrue(torch.isfinite(feature).all())
            torch.testing.assert_close(x.detach(), original, rtol=0, atol=0)

            # Using only the last feature ensures earlier levels remain connected.
            features[-1].square().mean().backward()
            self.assertTrue(x.grad is not None and torch.isfinite(x.grad).all())
            self.assertTrue(x.grad.abs().sum() > 0)
            for name, parameter in encoder.named_parameters():
                self.assertTrue(parameter.grad is not None, f"Missing encoder gradient: {name}")
                self.assertTrue(torch.isfinite(parameter.grad).all(), f"Non-finite gradient: {name}")
            for stage in encoder.stages:
                self.assertTrue(stage[0].weight.grad.abs().sum() > 0)
                self.assertTrue(stage[1].residual[-1].weight.grad.abs().sum() > 0)

            encoder.eval()
            restored = Encoder3D(input_shape[1], channels=widths, dropout=0.0).eval()
            restored.load_state_dict(encoder.state_dict())
            with torch.no_grad():
                for expected, actual in zip(encoder(x), restored(x)):
                    torch.testing.assert_close(actual, expected)

        encoder = Encoder3D(channels=widths, dropout=0.0)
        with torch.no_grad():
            thin_features = encoder(torch.randn(1, 1, 1, 1, 5))
            self.assertTrue([tuple(f.shape[2:]) for f in thin_features] == [(1, 1, 5), (1, 1, 3), (1, 1, 2)])
            single_level = Encoder3D(channels=(4,), dropout=0.0)(torch.randn(1, 1, 3, 5, 7))
            self.assertTrue(len(single_level) == 1 and single_level[0].shape == (1, 4, 3, 5, 7))
        for training in (True, False):
            encoder.train(training)
            try:
                encoder(torch.randn(1, 1, 1, 1, 4))
            except ValueError as error:
                self.assertTrue("bottleneck" in str(error))
            else:
                raise AssertionError("A single-voxel bottleneck must be rejected.")

    def test_dropout(self) -> None:
        """Exercise the encoder's default dropout in training and evaluation modes."""
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(3710)
            encoder = Encoder3D(channels=(8, 16)).train()
            x = torch.randn(2, 1, 3, 5, 7, requires_grad=True)

            def forward_with_seed(seed: int) -> tuple[Tensor, ...]:
                torch.manual_seed(seed)
                return encoder(x)

            first = forward_with_seed(17)
            repeated = forward_with_seed(17)
            changed = forward_with_seed(18)
            for expected, actual in zip(first, repeated):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertTrue(not torch.equal(first[-1], changed[-1]), "Training dropout had no effect.")
            self.assertTrue(all(torch.isfinite(f).all() for f in (*first, *repeated, *changed)))

            target = torch.randn_like(first[-1])
            (first[-1] - target).square().mean().backward()
            self.assertTrue(x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0)
            for name, parameter in encoder.named_parameters():
                self.assertTrue(parameter.grad is not None, f"Missing dropout-mode gradient: {name}")
                self.assertTrue(torch.isfinite(parameter.grad).all(), f"Non-finite gradient: {name}")

            encoder.eval()
            with torch.no_grad():
                first = forward_with_seed(17)
                changed = forward_with_seed(18)
            for expected, actual in zip(first, changed):
                self.assertTrue(torch.isfinite(actual).all())
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_residual_block(self) -> None:
        """Check residual shapes, gradients, state restoration and the shortcut."""

        for in_channels, out_channels in ((4, 4), (1, 4), (4, 8)):
            # Avoid randomly dropping every channel in these tiny gradient checks.
            block = ResidualBlock3D(in_channels, out_channels, dropout=0.0)
            x = torch.randn(1, in_channels, 5, 7, 9, requires_grad=True)
            output = block(x)
            self.assertTrue(output.shape == (1, out_channels, 5, 7, 9))
            self.assertTrue(torch.isfinite(output).all())
            output.square().mean().backward()
            self.assertTrue(x.grad is not None and torch.isfinite(x.grad).all())
            self.assertTrue(x.grad.abs().sum() > 0)
            for name, parameter in block.named_parameters():
                self.assertTrue(parameter.grad is not None, f"Missing gradient: {name}")
                self.assertTrue(torch.isfinite(parameter.grad).all(), f"Non-finite gradient: {name}")
            self.assertTrue(block.residual[-1].weight.grad.abs().sum() > 0)

            block.eval()
            restored = ResidualBlock3D(in_channels, out_channels, dropout=0.0).eval()
            restored.load_state_dict(block.state_dict())
            with torch.no_grad():
                expected = block(x)
                torch.testing.assert_close(restored(x), expected)
                torch.testing.assert_close(block(x), expected)

                # Removing the learned branch must leave exactly the shortcut.
                for parameter in block.residual.parameters():
                    parameter.zero_()
                torch.testing.assert_close(block(x), block.shortcut(x), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
