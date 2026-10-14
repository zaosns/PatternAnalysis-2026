"""Residual building block and encoder for the proposed 3D Improved U-Net.

The pre-activation context-block design follows the architectural description
in Isensee et al. (2018), section 2.1: https://arxiv.org/abs/1802.10508.
This project implementation uses channel-wise Dropout3d and optionally projects
the shortcut to a different channel count. It is not a complete reproduction
of the paper's network. Run this file for a small CPU check with synthetic data.
"""

import torch
from torch import Tensor, nn


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
    """Build a feature pyramid for a future U-Net decoder.

    Each level applies a 3x3x3 convolution followed by a residual block. The
    first convolution preserves resolution; later ones use stride two. With
    padding one, each downsampling maps an axis of size n to ceil(n / 2), so
    odd-sized inputs are accepted without cropping or padding them beforehand.

    The returned tuple is ordered from highest to lowest spatial resolution.
    Earlier features are intended for skip connections, and the last tensor
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


def _smoke_test() -> None:
    """Check the residual block and encoder using synthetic tensors on CPU."""
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
    print(f"Synthetic CPU checks passed (PyTorch {torch.__version__}); no MRI data used.")


if __name__ == "__main__":
    _smoke_test()
