"""PyTorch-only Dice loss and per-volume overlap metrics for 3D segmentation.

Labels are contiguous class indices, with index 0 reserved for background when
excluding it. The source HipMRI label mapping is documented in README.md. Synthetic
checks live in tests/test_objectives.py and require no MRI data. Run from this
directory: python -s -m unittest discover -s tests -v.
"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _check_options(num_classes: int, include_background: bool) -> None:
    if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError("num_classes must be an integer >= 2.")
    if not isinstance(include_background, bool):
        raise ValueError("include_background must be a bool.")


def _check_labels(labels: Tensor, num_classes: int, name: str) -> None:
    if labels.ndim != 4 or any(size <= 0 for size in labels.shape):
        raise ValueError(f"{name} must have nonempty shape [N, D, H, W].")
    if labels.dtype != torch.long:
        raise ValueError(f"{name} must contain torch.long class indices.")
    if labels.min() < 0 or labels.max() >= num_classes:
        raise ValueError(f"{name} class indices must be in [0, {num_classes - 1}].")


class SoftDiceLoss(nn.Module):
    """Mean per-volume, per-class soft Dice loss from raw model logits.

    logits: floating tensor [N, C, D, H, W], with C >= 2.
    target: torch.long [N, D, H, W], class indices in [0, C-1].

    Compute softmax over C, then for each sample/class use
    1 - (2 * sum(p*y) + smooth) / (sum(p) + sum(y) + smooth).
    Sums cover only spatial axes. Average equally over samples and selected
    classes; this is not a batch-pooled or squared-denominator Dice loss.
    Empty target classes remain in this loss and penalize predicted probability
    mass. The positive smoothing term keeps it finite even for an empty class.

    All classes are included by default. include_background=False excludes index
    0 only; that index must first be established by the dataset label mapping.
    Half/bfloat16 inputs use float32 softmax and accumulation; float64 is retained.
    smooth must also fit the accumulation dtype's positive normal range.
    """

    def __init__(self, include_background: bool = True, smooth: float = 1e-5) -> None:
        super().__init__()
        _check_options(2, include_background)
        if isinstance(smooth, bool) or not isinstance(smooth, (int, float)):
            raise ValueError("smooth must be a finite positive number.")
        if not 0 < smooth < float("inf"):
            raise ValueError("smooth must be a finite positive number.")
        self.include_background = include_background
        self.smooth = float(smooth)

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        """Return a scalar differentiable loss; do not pass probabilities."""
        if logits.ndim != 5 or any(size <= 0 for size in logits.shape):
            raise ValueError("logits must have nonempty shape [N, C, D, H, W].")
        _check_options(logits.shape[1], self.include_background)
        if not logits.is_floating_point() or not torch.isfinite(logits).all():
            raise ValueError("logits must contain finite floating-point values.")
        if logits.device != target.device:
            raise ValueError("logits and target must be on the same device.")
        _check_labels(target, logits.shape[1], "target")
        if target.shape != (logits.shape[0], *logits.shape[2:]):
            raise ValueError("target must match the logits batch and spatial dimensions.")

        dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
        limits = torch.finfo(dtype)
        if not limits.tiny <= self.smooth <= limits.max:
            raise ValueError("smooth is outside the positive normal range of the accumulation dtype.")
        probabilities = F.softmax(logits, dim=1, dtype=dtype)
        encoded = F.one_hot(target, num_classes=logits.shape[1]).movedim(-1, 1).to(dtype)
        first_class = 0 if self.include_background else 1
        probabilities = probabilities[:, first_class:]
        encoded = encoded[:, first_class:]
        spatial = (2, 3, 4)
        intersection = (probabilities * encoded).sum(dim=spatial)
        total = probabilities.sum(dim=spatial) + encoded.sum(dim=spatial)
        dice = (2 * intersection + self.smooth) / (total + self.smooth)
        return (1 - dice).mean()


@torch.no_grad()
def dice_iou_per_class(
    prediction: Tensor,
    target: Tensor,
    num_classes: int,
    include_background: bool = True,
) -> dict[str, Tensor]:
    """Return hard Dice/IoU per sample/class from class-index masks.

    prediction and target are torch.long [N, D, H, W]. Convert model logits with
    argmax(dim=1) before calling. Scores are float32 [N, K], where K is C or C-1;
    class_ids identifies the original class index of each column.

    Dice = 2*intersection/(prediction_count+target_count).
    IoU = intersection/union. Counts are accumulated as int64, without smoothing.
    Both masks empty for a class: NaN and valid=False, not a perfect score.
    Exactly one mask empty: 0 and valid=True (including false positives).
    valid_count contains the number of defined sample scores for each class.

    There is no cross-sample reduction. For case-level reporting, each sample
    must be a reconstructed case volume, not an arbitrary slice or patch.
    Concatenate per-case results before reporting means and valid counts.
    """
    _check_options(num_classes, include_background)
    if prediction.device != target.device:
        raise ValueError("prediction and target must be on the same device.")
    _check_labels(prediction, num_classes, "prediction")
    _check_labels(target, num_classes, "target")
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape.")

    first_class = 0 if include_background else 1
    dice_scores, iou_scores, valid_masks = [], [], []
    for class_id in range(first_class, num_classes):
        predicted = prediction == class_id
        actual = target == class_id
        intersection = (predicted & actual).flatten(1).sum(dim=1)
        total = predicted.flatten(1).sum(dim=1) + actual.flatten(1).sum(dim=1)
        union = total - intersection
        valid = total > 0
        dice = 2 * intersection.float() / total.clamp_min(1).float()
        iou = intersection.float() / union.clamp_min(1).float()
        dice_scores.append(dice.masked_fill(~valid, float("nan")))
        iou_scores.append(iou.masked_fill(~valid, float("nan")))
        valid_masks.append(valid)
    valid = torch.stack(valid_masks, dim=1)
    return {
        "dice": torch.stack(dice_scores, dim=1),
        "iou": torch.stack(iou_scores, dim=1),
        "valid": valid,
        "valid_count": valid.sum(dim=0),
        "class_ids": torch.arange(first_class, num_classes, device=prediction.device),
    }
