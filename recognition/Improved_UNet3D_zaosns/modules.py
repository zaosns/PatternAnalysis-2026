"""Initial PyTorch building block for the proposed 3D Improved U-Net.

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


def _smoke_test() -> None:
    """Check shape, gradients, state restoration and shortcut behavior on CPU."""
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
    print(f"Synthetic CPU checks passed (PyTorch {torch.__version__}); no MRI data used.")


if __name__ == "__main__":
    _smoke_test()
