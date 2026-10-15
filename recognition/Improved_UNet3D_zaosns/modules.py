"""Residual 3D segmentation prototype for the proposed Improved U-Net.

The pre-activation context-block design follows the architectural description
in Isensee et al. (2018), section 2.1: https://arxiv.org/abs/1802.10508.
This project implementation uses channel-wise Dropout3d and optionally projects
the shortcut to a different channel count. It is not a complete reproduction
of the paper's network. Synthetic regression checks live in tests/test_modules.py.
Run from this directory: python -s -m unittest discover -s tests -v.
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
        self._config = {
            "num_classes": num_classes,
            "in_channels": in_channels,
            "channels": self.decoder.channels,
            "dropout": float(dropout),
            "deep_supervision": deep_supervision,
        }

    def get_config(self) -> dict:
        """Return constructor settings for checkpoints; recreate to change them."""
        return dict(self._config)

    def forward(self, x: Tensor) -> Tensor:
        """Produce full-resolution raw logits from an input image volume."""
        features = self.encoder(x)
        decoded = self.decoder(features)
        return self.head(decoded) if self.deep_supervision else self.head(decoded[0])
