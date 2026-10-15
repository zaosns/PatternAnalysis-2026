"""Conventional 2D U-Net baseline, also exposed through a volume interface.

This implementation uses padded convolutions and affine instance normalization,
so it is an adaptation of the conventional U-Net encoder/decoder topology, not
an exact reproduction of the original 2015 architecture. It has no residual
blocks, dropout, deep supervision, or operations that mix adjacent slices.

The volume wrapper permits the same whole-case objective and one optimizer
update per case as the 3D model. Training code, rather than this module, defines
the loss, preprocessing, patient split, and checkpoint-selection policy.
"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

__all__ = ["UNet2D", "SliceWiseUNet2D"]


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


class _DoubleConv2D(nn.Sequential):
    """Two padded convolutions, each followed by instance norm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(),
        )


class UNet2D(nn.Module):
    """Map floating images [N, C, H, W] to raw logits [N, K, H, W].

    Each encoder level has two 3x3 convolutions. A 2x2 max pool separates
    consecutive levels. Each decoder level uses a learned 2x2 stride-two
    transposed convolution, a skip concatenation, and two 3x3 convolutions.
    Odd-sized inputs are supported by bilinearly resizing the upsampled feature
    map to the skip's exact H/W with ``align_corners=False``. This size-repair
    adaptation is only used when a transposed convolution cannot recover an odd
    size; even-sized matches are left unchanged.

    ``channels`` includes the bottleneck and must contain at least two positive
    integer widths. Pooling must leave both bottleneck axes nonempty and at
    least two spatial pixels overall for instance normalization, including in
    evaluation mode. Normalization uses each image independently; it has no
    running statistics. The output contains logits, with no softmax or argmax.
    """

    def __init__(
        self,
        num_classes: int = 6,
        in_channels: int = 1,
        channels: tuple[int, ...] = (16, 32, 64, 128),
    ) -> None:
        super().__init__()
        _positive_int(in_channels, "in_channels")
        _positive_int(num_classes, "num_classes")
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2, including background.")
        try:
            widths = tuple(channels)
        except TypeError as error:
            raise ValueError("channels must contain at least two positive integer widths.") from error
        if len(widths) < 2:
            raise ValueError("channels must contain at least two positive integer widths.")
        for width in widths:
            _positive_int(width, "Each channel width")

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.channels = widths
        self.encoder = nn.ModuleList()
        previous_width = in_channels
        for width in widths:
            self.encoder.append(_DoubleConv2D(previous_width, width))
            previous_width = width
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsamplers = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for level in range(len(widths) - 2, -1, -1):
            self.upsamplers.append(
                nn.ConvTranspose2d(widths[level + 1], widths[level], 2, stride=2, bias=False)
            )
            self.decoder.append(_DoubleConv2D(2 * widths[level], widths[level]))
        self.head = nn.Conv2d(widths[0], num_classes, kernel_size=1, bias=True)

    def get_config(self) -> dict:
        """Return only constructor arguments, sufficient to rebuild this model."""
        return {
            "num_classes": self.num_classes,
            "in_channels": self.in_channels,
            "channels": self.channels,
        }

    def forward(self, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor) or x.ndim != 4:
            raise ValueError("Expected input shape [N, C, H, W].")
        if not x.is_floating_point():
            raise ValueError("Input images must be floating-point tensors.")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {x.shape[1]}.")
        if any(size <= 0 for size in x.shape):
            raise ValueError("Input batch and spatial dimensions must be nonempty.")
        divisor = 2 ** (len(self.channels) - 1)
        bottleneck = tuple(size // divisor for size in x.shape[2:])
        if min(bottleneck) < 1 or bottleneck[0] * bottleneck[1] < 2:
            raise ValueError(
                f"Pooling would produce bottleneck shape {bottleneck}; each axis "
                "must be nonempty and instance normalization needs at least two pixels."
            )

        features = []
        for level, block in enumerate(self.encoder):
            if level:
                x = self.pool(x)
            x = block(x)
            features.append(x)
        for upsample, block, skip in zip(
            self.upsamplers, self.decoder, reversed(features[:-1])
        ):
            x = upsample(x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = block(torch.cat((x, skip), dim=1))
        return self.head(x)


class SliceWiseUNet2D(nn.Module):
    """Apply one shared UNet2D independently to every D slice of a volume.

    Input/output are [N, C, D, H, W] and [N, K, D, H, W]. Slices are flattened
    into the batch dimension, then restored in the original case/depth order.
    Only H/W have convolutions, pooling, or interpolation. Instance statistics
    are also computed separately per slice. D denotes the prepared tensor's
    depth axis; this wrapper does not infer or change anatomical orientation.
    """

    def __init__(
        self,
        num_classes: int = 6,
        in_channels: int = 1,
        channels: tuple[int, ...] = (16, 32, 64, 128),
    ) -> None:
        super().__init__()
        self.model = UNet2D(num_classes=num_classes, in_channels=in_channels, channels=channels)

    def get_config(self) -> dict:
        """Return constructor arguments for this wrapper and its shared model."""
        return self.model.get_config()

    def forward(self, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor) or x.ndim != 5:
            raise ValueError("Expected input shape [N, C, D, H, W].")
        if any(size <= 0 for size in x.shape):
            raise ValueError("Input batch, channels, depth and spatial dimensions must be nonempty.")
        n, c, d, h, w = x.shape
        slices = x.permute(0, 2, 1, 3, 4).reshape(n * d, c, h, w)
        logits = self.model(slices)
        return logits.reshape(n, d, self.model.num_classes, h, w).permute(0, 2, 1, 3, 4).contiguous()
