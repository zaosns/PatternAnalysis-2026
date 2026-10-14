"""Residual 3D segmentation prototype for the proposed Improved U-Net.

The pre-activation context-block design follows the architectural description
in Isensee et al. (2018), section 2.1: https://arxiv.org/abs/1802.10508.
This project implementation uses channel-wise Dropout3d and optionally projects
the shortcut to a different channel count. It is not a complete reproduction
of the paper's network. Run this file for a small CPU check with synthetic data.
The decoder uses nearest-neighbor resizing to each skip's actual spatial shape
and convolutional feature fusion, following the paper's localization pathway.
Multi-scale segmentation logits are added from coarse to fine for deep supervision.
"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ResidualBlock3D(nn.Module):
    """Extract volumetric features without changing spatial resolution.

    Input: floating-point tensor [N, in_channels, D, H, W].
    Output: tensor [N, out_channels, D, H, W].

    Each convolution is preceded by instance normalization and LeakyReLU.
    The result is added to the unmodified input, or to a 1x1x1 projection when
    channel counts differ. There is no activation after this addition.
    Instance normalization requires more than one spatial voxel, including in
    evaluation mode because this block does not track running statistics.

    Args:
        in_channels: Number of input feature channels (positive integer).
        out_channels: Number of output feature channels (positive integer).
        dropout: Probability of dropping a feature channel during training.
            This is a configurable starting value, not a tuned HipMRI setting.
    """

    def __init__(
        self, in_channels: int, out_channels: int, dropout: float = 0.3
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("Channel counts must be positive.")
        self.in_channels = in_channels

        self.residual = nn.Sequential(
            nn.InstanceNorm3d(in_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.Dropout3d(p=dropout),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        )
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Return learned features plus the shortcut, preserving D, H and W."""
        if x.ndim != 5:
            raise ValueError("Expected input shape [N, C, D, H, W].")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {x.shape[1]}.")
        if x.shape[2] * x.shape[3] * x.shape[4] <= 1:
            raise ValueError("Instance normalization needs more than one spatial voxel.")
        return self.shortcut(x) + self.residual(x)


class Encoder3D(nn.Module):
    """Build a feature pyramid for the U-Net decoder.

    Each level applies a 3x3x3 convolution followed by a residual block. The
    first convolution preserves resolution; later ones use stride two. With
    padding one, each downsampling maps an axis of size n to ceil(n / 2), so
    odd-sized inputs are accepted without cropping or padding them beforehand.

    The returned tuple is ordered from highest to lowest spatial resolution.
    Earlier features supply skip connections, and the last tensor
    is the bottleneck. These are feature maps, not segmentation predictions.

    Args:
        in_channels: Number of input image channels.
        channels: Positive feature-channel counts, one per encoder level.
            These defaults are provisional until the HipMRI resource audit.
        dropout: Feature-channel dropout probability in each residual block.

    Every level needs more than one spatial voxel for instance normalization.
    Thin volumes are allowed if their total spatial voxel count stays above one.
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: tuple[int, ...] = (16, 32, 64, 128),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.channels = tuple(channels)
        if in_channels <= 0 or not self.channels or any(c <= 0 for c in self.channels):
            raise ValueError("Specify positive input channels and at least one positive level width.")
        self.in_channels = in_channels
        self.stages = nn.ModuleList()
        previous_channels = in_channels
        for level, width in enumerate(self.channels):
            self.stages.append(
                nn.Sequential(
                    nn.Conv3d(
                        previous_channels, width, kernel_size=3,
                        stride=1 if level == 0 else 2, padding=1, bias=False,
                    ),
                    ResidualBlock3D(width, width, dropout=dropout),
                )
            )
            previous_channels = width

    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        """Return one feature tensor per level, including the bottleneck."""
        if x.ndim != 5:
            raise ValueError("Expected input shape [N, C, D, H, W].")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {x.shape[1]}.")

        # Check before convolutions so a small input fails with a useful message.
        divisor = 2 ** (len(self.channels) - 1)
        deepest_shape = tuple((size + divisor - 1) // divisor for size in x.shape[2:])
        if deepest_shape[0] * deepest_shape[1] * deepest_shape[2] <= 1:
            raise ValueError(
                f"Encoder bottleneck would have spatial shape {deepest_shape}; "
                "instance normalization needs more than one voxel. "
                "Use a larger input volume or fewer encoder levels."
            )

        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return tuple(features)


class DecoderBlock3D(nn.Module):
    """Upsample one level and fuse it with the corresponding encoder skip.

    The lower-resolution input must have spatial shape ceil(skip_shape / 2),
    matching two adjacent Encoder3D levels. Nearest-neighbor resizing uses the
    skip's exact shape, including odd sizes. The output has out_channels and
    the skip's spatial shape; it contains features, not segmentation scores.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        if min(in_channels, skip_channels, out_channels) <= 0:
            raise ValueError("Decoder channel counts must be positive.")
        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.upsample_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01),
        )
        joined_channels = out_channels + skip_channels
        self.localization = nn.Sequential(
            nn.Conv3d(joined_channels, joined_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(joined_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv3d(joined_channels, out_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.01),
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        """Combine a lower-resolution tensor with one encoder feature tensor."""
        if x.ndim != 5 or skip.ndim != 5:
            raise ValueError("Decoder inputs must have shape [N, C, D, H, W].")
        if x.shape[1] != self.in_channels or skip.shape[1] != self.skip_channels:
            raise ValueError(
                f"Expected {self.in_channels} input and {self.skip_channels} skip channels."
            )
        if x.shape[0] != skip.shape[0]:
            raise ValueError("Decoder input and skip must have the same batch size.")
        if skip.shape[2] * skip.shape[3] * skip.shape[4] <= 1:
            raise ValueError("The skip needs more than one spatial voxel for normalization.")
        expected_shape = tuple((size + 1) // 2 for size in skip.shape[2:])
        if tuple(x.shape[2:]) != expected_shape:
            raise ValueError("Decoder input and skip must be adjacent encoder resolutions.")

        x = F.interpolate(x, size=skip.shape[2:], mode="nearest")
        x = self.upsample_conv(x)
        return self.localization(torch.cat((x, skip), dim=1))


class Decoder3D(nn.Module):
    """Decode an Encoder3D feature tuple using all of its skip connections.

    Pass the same channel sequence as used by the encoder, ordered from highest
    to lowest resolution. At least two encoder levels are needed for a decoder.
    Output is a tuple of len(channels)-1 decoded tensors, highest resolution
    first. Output i matches encoder feature i in channel count and spatial size;
    the encoder bottleneck is consumed but is not included in this output tuple.
    The tuple supplies segmentation heads at multiple spatial resolutions.
    """

    def __init__(self, channels: tuple[int, ...] = (16, 32, 64, 128)) -> None:
        super().__init__()
        self.channels = tuple(channels)
        if len(self.channels) < 2 or any(c <= 0 for c in self.channels):
            raise ValueError("A decoder needs at least two positive encoder channel counts.")
        self.stages = nn.ModuleList(
            DecoderBlock3D(self.channels[level + 1], self.channels[level], self.channels[level])
            for level in range(len(self.channels) - 2, -1, -1)
        )

    def forward(self, features: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        """Return decoded features ordered from highest to lowest resolution."""
        if len(features) != len(self.channels):
            raise ValueError(f"Expected {len(self.channels)} encoder features, got {len(features)}.")
        for level, (feature, width) in enumerate(zip(features, self.channels)):
            if feature.ndim != 5 or feature.shape[1] != width:
                raise ValueError(f"Encoder feature {level} must have shape [N, {width}, D, H, W].")
            if feature.shape[0] != features[0].shape[0]:
                raise ValueError("All encoder features must have the same batch size.")
            if feature.shape[2] * feature.shape[3] * feature.shape[4] <= 1:
                raise ValueError(f"Encoder feature {level} needs more than one spatial voxel.")
            if level > 0:
                expected_shape = tuple((size + 1) // 2 for size in features[level - 1].shape[2:])
                if tuple(feature.shape[2:]) != expected_shape:
                    raise ValueError("Encoder features must follow the ceil-half spatial pyramid.")

        x = features[-1]
        decoded = []
        for stage, skip in zip(self.stages, reversed(features[:-1])):
            x = stage(x, skip)
            decoded.append(x)
        return tuple(reversed(decoded))


class SegmentationHead3D(nn.Module):
    """Map features to raw per-voxel class scores with a 1x1x1 convolution.

    Input is [N, in_channels, D, H, W]; output is [N, num_classes, D, H, W].
    num_classes includes background and must be supplied after defining the
    label mapping. This interface uses at least two mutually exclusive classes,
    including for binary segmentation (background and foreground).
    No normalization or activation is applied: cross entropy consumes logits.
    """

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        if isinstance(in_channels, bool) or not isinstance(in_channels, int) or in_channels <= 0:
            raise ValueError("Head input channels must be a positive integer.")
        if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 2:
            raise ValueError("num_classes must be an integer >= 2, including background.")
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.projection = nn.Conv3d(in_channels, num_classes, kernel_size=1, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        """Return unnormalized scores without changing the spatial dimensions."""
        if x.ndim != 5 or x.shape[1] != self.in_channels:
            raise ValueError(f"Head input must have shape [N, {self.in_channels}, D, H, W].")
        if any(size <= 0 for size in x.shape[2:]):
            raise ValueError("Head input spatial dimensions must be nonempty.")
        return self.projection(x)


class DeepSupervisionHead3D(nn.Module):
    """Project decoder features to logits and sum them from coarse to fine.

    channels and input features are ordered from highest to lowest resolution.
    Each feature receives a 1x1x1 segmentation head. At every step, the accumulated
    coarse logits are resized to the next finer feature's exact shape with
    nearest-neighbor interpolation and added to that level's logits.

    This follows the summed-output deep supervision in Isensee et al., section
    2.1. One loss on the final logits supplies gradients to every head; separate
    per-scale losses are not used here. A one-level input reduces to a single head.
    """

    def __init__(self, channels: tuple[int, ...], num_classes: int) -> None:
        super().__init__()
        self.channels = tuple(channels)
        if not self.channels:
            raise ValueError("Deep supervision needs at least one feature level.")
        self.heads = nn.ModuleList(
            SegmentationHead3D(width, num_classes) for width in self.channels
        )

    def forward(self, features: tuple[Tensor, ...]) -> Tensor:
        """Return summed raw logits at the highest feature resolution."""
        if len(features) != len(self.channels):
            raise ValueError(f"Expected {len(self.channels)} decoder features, got {len(features)}.")
        for level, (feature, width) in enumerate(zip(features, self.channels)):
            if feature.ndim != 5 or feature.shape[1] != width:
                raise ValueError(f"Decoder feature {level} must have shape [N, {width}, D, H, W].")
            if feature.shape[0] != features[0].shape[0]:
                raise ValueError("All decoder features must have the same batch size.")
            if any(size <= 0 for size in feature.shape[2:]):
                raise ValueError("Decoder feature spatial dimensions must be nonempty.")
            if level > 0:
                expected_shape = tuple((size + 1) // 2 for size in features[level - 1].shape[2:])
                if tuple(feature.shape[2:]) != expected_shape:
                    raise ValueError("Decoder features must follow the ceil-half spatial pyramid.")

        logits = self.heads[-1](features[-1])
        for level in range(len(self.heads) - 2, -1, -1):
            logits = F.interpolate(logits, size=features[level].shape[2:], mode="nearest")
            logits = logits + self.heads[level](features[level])
        return logits


class SegmentationModel3D(nn.Module):
    """Connect the residual encoder, decoder and configurable segmentation heads.

    Deep supervision is enabled by default: each decoder level contributes logits
    to a sum at full resolution. Disable it for a highest-resolution-only head.
    Both configurations return one logits tensor [N, num_classes, D, H, W] in
    training and evaluation. Class selection/probabilities remain outside the model.

    num_classes is required and includes background; no HipMRI label count is
    assumed. Other arguments follow Encoder3D, with at least two levels needed.
    With just two levels there is one decoder feature, so either setting uses
    one head. This project adaptation is not an exact paper reproduction.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        channels: tuple[int, ...] = (16, 32, 64, 128),
        dropout: float = 0.3,
        deep_supervision: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(deep_supervision, bool):
            raise ValueError("deep_supervision must be a bool.")
        self.deep_supervision = deep_supervision
        self.decoder = Decoder3D(channels)
        self.head = (
            DeepSupervisionHead3D(self.decoder.channels[:-1], num_classes)
            if deep_supervision
            else SegmentationHead3D(self.decoder.channels[0], num_classes)
        )
        self.encoder = Encoder3D(in_channels, self.decoder.channels, dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Produce full-resolution raw logits from an input image volume."""
        features = self.encoder(x)
        decoded = self.decoder(features)
        return self.head(decoded) if self.deep_supervision else self.head(decoded[0])


def _check_deep_supervision() -> None:
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
        assert feature.grad is not None and torch.isfinite(feature.grad).all()
        assert feature.grad.abs().sum() > 0, "A decoder scale did not receive gradients."
        torch.testing.assert_close(feature.detach(), original, rtol=0, atol=0)
    for name, parameter in head.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, f"A segmentation head is disconnected: {name}"

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
            assert message in str(error), str(error)
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
    print("PASS: deep supervision; odd-size sum, every scale/head gradient, single-head equivalence")


def _check_segmentation() -> None:
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
            assert "num_classes" in str(error)
        else:
            raise AssertionError(f"Accepted invalid num_classes={classes}.")
    for shape in ((3, 3, 5, 7), (1, 4, 3, 5, 7), (1, 3, 0, 5, 7)):
        try:
            head(torch.empty(*shape))
        except ValueError as error:
            assert "Head input" in str(error)
        else:
            raise AssertionError(f"Head accepted invalid input shape {shape}.")
    print("PASS: raw segmentation logits and invalid head inputs")

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
        assert logits.shape == (shape[0], classes, *shape[2:])
        assert torch.isfinite(logits).all()
        # One integer class index per voxel, without a channel dimension.
        target = torch.randint(classes, (shape[0], *shape[2:]))
        loss = F.cross_entropy(logits, target)
        assert torch.isfinite(loss)
        loss.backward()
        assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, f"Missing segmentation gradient: {name}"
            assert torch.isfinite(parameter.grad).all(), f"Non-finite gradient: {name}"
            if parameter.ndim == 5:
                assert parameter.grad.abs().sum() > 0, f"Zero convolution gradient: {name}"
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
            assert expected.argmax(dim=1).shape == target.shape
        print(
            f"PASS: segmentation {shape}, {classes} classes, deep_supervision={supervision}; "
            "CE gradients and state"
        )

    with torch.no_grad():
        default = SegmentationModel3D(num_classes=2).eval()
        scores = default(torch.randn(1, 1, 9, 11, 13))
        assert scores.shape == (1, 2, 9, 11, 13) and torch.isfinite(scores).all()
        thin = SegmentationModel3D(num_classes=2, channels=(3, 5), dropout=0.0).eval()
        scores = thin(torch.randn(1, 1, 1, 1, 3))
        assert scores.shape == (1, 2, 1, 1, 3) and torch.isfinite(scores).all()
    print("PASS: default four-level segmentation and a thin two-level volume")


def _check_decoder() -> None:
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
        assert isinstance(decoded, tuple) and len(decoded) == len(expected_shapes)
        for feature, expected in zip(decoded, expected_shapes):
            assert feature.shape == expected and torch.isfinite(feature).all()
        target = torch.randn_like(decoded[0])
        (decoded[0] - target).square().mean().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, f"Missing encoder/decoder gradient: {name}"
            assert torch.isfinite(parameter.grad).all(), f"Non-finite gradient: {name}"
            if parameter.ndim == 5:
                assert parameter.grad.abs().sum() > 0, f"Zero convolution gradient: {name}"

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
            assert leaf.grad is not None and torch.isfinite(leaf.grad).all()
            assert leaf.grad.abs().sum() > 0
            torch.testing.assert_close(leaf.detach(), original, rtol=0, atol=0)
        print(f"PASS: decoder {shape}; exact sizes, all skip gradients and state")

    with torch.no_grad():
        thin = Decoder3D(widths)(Encoder3D(channels=widths, dropout=0.0)(torch.randn(1, 1, 1, 1, 5)))
        assert [tuple(f.shape[2:]) for f in thin] == [(1, 1, 5), (1, 1, 3)]
        minimal = Decoder3D((2, 4))(
            Encoder3D(channels=(2, 4), dropout=0.0)(torch.randn(1, 1, 3, 5, 7))
        )
        assert len(minimal) == 1 and minimal[0].shape == (1, 2, 3, 5, 7)
    try:
        Decoder3D((4,))
    except ValueError:
        pass
    else:
        raise AssertionError("A decoder must have at least two encoder levels.")
    print("PASS: thin-volume decoding, two-level configuration and depth guard")


def _check_decoder_inputs() -> None:
    """Keep the default configuration and malformed feature checks repeatable."""
    with torch.no_grad():
        model = nn.Sequential(Encoder3D(), Decoder3D()).eval()
        outputs = model(torch.randn(1, 1, 9, 11, 13))
        assert [tuple(f.shape) for f in outputs] == [
            (1, 16, 9, 11, 13), (1, 32, 5, 6, 7), (1, 64, 3, 3, 4)
        ]
        assert all(torch.isfinite(f).all() for f in outputs)

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
            assert message in str(error), f"Wrong error for {label}: {error}"
        else:
            raise AssertionError(f"Decoder accepted an invalid {label}.")

    # Equal channel counts ensure this catches spatial order, not just channels.
    equal_width = (first, middle[:, :3], last[:, :3])
    try:
        Decoder3D((3, 3, 3))(tuple(reversed(equal_width)))
    except ValueError as error:
        assert "spatial pyramid" in str(error)
    else:
        raise AssertionError("Decoder accepted reversed encoder features.")
    print("PASS: default four-level model and seven malformed decoder inputs")


def _check_encoder() -> None:
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
        assert isinstance(features, tuple) and len(features) == len(widths)
        for feature, width, spatial_shape in zip(features, widths, spatial_shapes):
            assert feature.shape == (input_shape[0], width, *spatial_shape)
            assert torch.isfinite(feature).all()
        torch.testing.assert_close(x.detach(), original, rtol=0, atol=0)

        # Using only the last feature ensures earlier levels remain connected.
        features[-1].square().mean().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert x.grad.abs().sum() > 0
        for name, parameter in encoder.named_parameters():
            assert parameter.grad is not None, f"Missing encoder gradient: {name}"
            assert torch.isfinite(parameter.grad).all(), f"Non-finite gradient: {name}"
        for stage in encoder.stages:
            assert stage[0].weight.grad.abs().sum() > 0
            assert stage[1].residual[-1].weight.grad.abs().sum() > 0

        encoder.eval()
        restored = Encoder3D(input_shape[1], channels=widths, dropout=0.0).eval()
        restored.load_state_dict(encoder.state_dict())
        with torch.no_grad():
            for expected, actual in zip(encoder(x), restored(x)):
                torch.testing.assert_close(actual, expected)
        print(f"PASS: encoder {input_shape}; feature pyramid, gradients and state")

    encoder = Encoder3D(channels=widths, dropout=0.0)
    with torch.no_grad():
        thin_features = encoder(torch.randn(1, 1, 1, 1, 5))
        assert [tuple(f.shape[2:]) for f in thin_features] == [(1, 1, 5), (1, 1, 3), (1, 1, 2)]
        single_level = Encoder3D(channels=(4,), dropout=0.0)(torch.randn(1, 1, 3, 5, 7))
        assert len(single_level) == 1 and single_level[0].shape == (1, 4, 3, 5, 7)
    for training in (True, False):
        encoder.train(training)
        try:
            encoder(torch.randn(1, 1, 1, 1, 4))
        except ValueError as error:
            assert "bottleneck" in str(error)
        else:
            raise AssertionError("A single-voxel bottleneck must be rejected.")
    print("PASS: thin volumes, single-level encoder and bottleneck size guard")


def _check_dropout() -> None:
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
        assert not torch.equal(first[-1], changed[-1]), "Training dropout had no effect."
        assert all(torch.isfinite(f).all() for f in (*first, *repeated, *changed))

        target = torch.randn_like(first[-1])
        (first[-1] - target).square().mean().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
        for name, parameter in encoder.named_parameters():
            assert parameter.grad is not None, f"Missing dropout-mode gradient: {name}"
            assert torch.isfinite(parameter.grad).all(), f"Non-finite gradient: {name}"

        encoder.eval()
        with torch.no_grad():
            first = forward_with_seed(17)
            changed = forward_with_seed(18)
        for expected, actual in zip(first, changed):
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    print("PASS: default dropout; seeded training, finite gradients and deterministic evaluation")


def _smoke_test() -> None:
    """Check model components and segmentation logits using synthetic tensors."""
    torch.manual_seed(3710)
    torch.set_num_threads(2)
    for in_channels, out_channels in ((4, 4), (1, 4), (4, 8)):
        # Avoid randomly dropping every channel in these tiny gradient checks.
        block = ResidualBlock3D(in_channels, out_channels, dropout=0.0)
        x = torch.randn(1, in_channels, 5, 7, 9, requires_grad=True)
        output = block(x)
        assert output.shape == (1, out_channels, 5, 7, 9)
        assert torch.isfinite(output).all()
        output.square().mean().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert x.grad.abs().sum() > 0
        for name, parameter in block.named_parameters():
            assert parameter.grad is not None, f"Missing gradient: {name}"
            assert torch.isfinite(parameter.grad).all(), f"Non-finite gradient: {name}"
        assert block.residual[-1].weight.grad.abs().sum() > 0

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
        print(f"PASS: {in_channels} -> {out_channels} channels; shape, gradients, state, shortcut")
    _check_encoder()
    _check_decoder()
    _check_decoder_inputs()
    _check_dropout()
    _check_deep_supervision()
    _check_segmentation()
    print(f"Synthetic CPU checks passed (PyTorch {torch.__version__}); no MRI data used.")


if __name__ == "__main__":
    _smoke_test()
