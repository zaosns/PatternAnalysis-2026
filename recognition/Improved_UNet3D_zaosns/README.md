# HipMRI Segmentation with a 3D Improved U-Net

**Status: residual block and encoder.** `modules.py` contains a 3D residual block,
a configurable encoder and checks using synthetic tensors. The decoder,
segmentation heads, data loader, training and evaluation are not implemented.
No HipMRI experimental results or completed feasibility check-off are claimed.

## Proposed problem and approach

The project will investigate whether volumetric context improves prostate
segmentation at the base, apex and adjacent-organ boundaries enough to justify
its memory and inference cost. The intended prototype would produce contours
for human review.

The proposed main model is a PyTorch 3D Improved U-Net with residual units and
deep supervision on downsampled HipMRI volumes. The supplied assignment lists
this configuration as Hard difficulty. The architecture and compute budget will
be finalized after a data and resource audit.

## Planned experiments

- Implement a 2D U-Net baseline and the proposed 3D model. Use the same held-out
  patients and evaluate both on consistently reconstructed case volumes.
- If resources permit, add a plain 3D U-Net comparison to separate the effect of
  volumetric context from the effect of the architectural improvements.
- Report per-class Dice and IoU, parameter count, peak GPU memory and per-case
  inference time. Define model selection and evaluation before using the test set.
- Inspect 3-5 representative failures using input images, ground-truth masks,
  predictions and false-positive/false-negative overlays.
- Use the measured results to recommend where the prototype is useful, where
  human correction remains necessary and whether its additional cost is justified.

These are planned experiments, not completed work or assumed improvements.

## Development milestones

1. Confirm access to Rangpur GPU resources and the HipMRI data. A directory listing
   at `/home/groups/comp3710/HipMRI_Study_open` has succeeded, and PyTorch imports
   with `python -s` in the existing Conda environment. GPU execution and reading
   individual MRI/label files still need verification.
2. Audit patient identifiers, image/label pairs, label meanings, voxel spacing,
   orientation and spatial alignment.
3. Group all scans from each patient into one train/validation/test partition
   before constructing slices or patches. Document and reproduce the split.
4. Implement and run a small baseline experiment, measuring actual memory and time.
5. Prepare the one-page feasibility review: user need, 3-5 testable acceptance
   criteria, initial evidence, resource budget, risks and a fallback. Complete
   the required tutor check-off by the published course midpoint.
6. Implement the main model, run the comparisons and analyze failure cases.
7. Complete the report, reproduction instructions and final submission.

## Code structure

| File | Responsibility | Current state |
| --- | --- | --- |
| `modules.py` | PyTorch model components | Residual block and encoder |
| `dataset.py` | Loading, preprocessing, patient splits and augmentation | Not implemented |
| `train.py` | Training, validation, testing, checkpointing and learning curves | Not implemented |
| `predict.py` | Loading saved weights and visualizing test inference | Not implemented |

This README will develop into the final technical report. Training and inference
commands will be added once they exist and have been verified.

## Model components and synthetic checks

`ResidualBlock3D` accepts `[N, C, D, H, W]` feature tensors and preserves spatial
dimensions. It uses two pre-activation 3D convolutions with instance normalization
and LeakyReLU, inspired by the context modules described by Isensee et al. The
project uses configurable channel-wise dropout and a 1x1x1 shortcut projection
when the channel count changes. This is a building block, not a claim of a full
paper reproduction.

`Encoder3D` applies a full-resolution stem and residual block, then uses stride-two
3D convolutions and residual blocks to form a feature pyramid. It returns a tuple
from highest resolution to bottleneck, retaining earlier features for future
decoder skip connections. Each downsampled axis has size `ceil(n / 2)`; input
dimensions need not be divisible by powers of two. A bottleneck containing only
one spatial voxel is rejected before computation because instance normalization
cannot use it. Thin volumes with more than one voxel at the bottleneck are allowed.

The default widths `(16, 32, 64, 128)` and dropout probability `0.3` are configurable
starting points, not settings selected through HipMRI experiments. The decoder,
segmentation heads and deep supervision are pending; encoder outputs are features,
not class scores or masks.

From this project directory, with PyTorch available, run:

```bash
python -s modules.py
```

The synthetic CPU checks cover:

- The residual block: equal and changing channel counts, finite outputs and
  gradients, state restoration, and the shortcut with the learned branch zeroed out.
- The encoder: batches of one and two, even and odd spatial dimensions, single-
  and multi-channel inputs, the shapes of all returned features, and state
  restoration. A loss using only the bottleneck checks that gradients reach
  every encoder stage and the input.
- Thin volumes, a single encoder level, and rejection of a single-voxel
  bottleneck in both training and evaluation modes.

Dropout is disabled for these tiny checks to avoid randomly dropping every feature
channel. The checks do not load MRI data, train a segmentation model or measure
Dice/IoU.

## Environment and reproducibility

The residual-block and encoder checks passed locally on macOS CPU with Python
3.13.9 and PyTorch 2.14.0 on 14 October 2026, using seed 3710. Additional checks
covered the default four-level encoder, invalid channel configurations and
incorrect input rank/channel count. Single-voxel bottlenecks were rejected;
valid even-sized, odd-sized and thin volumes were accepted.

The training environment is not yet validated. The user's Rangpur terminal output
confirmed Python 3.11 with PyTorch 2.13.0+cu130 in the existing `torch` Conda
environment when user site-packages were excluded with `python -s`. Importing that
package does not establish GPU/driver compatibility; the CUDA execution check
has not completed. Local component checks and Rangpur training validation are
separate milestones.

Record tested versions, seeds, split-generation instructions, hyperparameters,
commands and checkpoint retrieval instructions as the implementation develops.

Keep datasets, patient-level metadata, model weights and raw experiment output
outside Git. The project `.gitignore` covers common locations and file types;
inspect staged changes because these patterns cannot identify every data file.
Reviewed figures for the report may be stored under `figures/`.

## Submission workflow

Develop on `topic-recognition`, committing meaningful changes as they are
completed. The final submission requires a pull request containing working code
to the course repository's `topic-recognition` branch, plus a PDF export of the
completed README for Turnitin. The one-page feasibility review must also appear
in the final report. This preparation commit is not a finished submission.

## Artificial Intelligence Usage Disclosure

Codex assisted with interpreting the assignment, comparing project options,
checking and configuring the GitHub fork, drafting this plan and the ignore rules,
and implementing the residual block, encoder and synthetic checks. The checks
exercise feature shapes, gradients through all encoder stages, state restoration,
input-size limits and residual-path behavior; they do not validate segmentation
quality. No training or empirical validation on HipMRI has been performed.
The student still needs to review and understand the
implementation. This section will be updated with the assistance received and
the checks actually performed during development.

## References

- Supplied COMP3710/COMP7710 report specification, sections 2.2-6. The supplied
  filename is v2.3 while its cover says DRAFT/Version 2.2; confirm deadlines and
  any revised requirements against the course announcements.
- Isensee et al., [Brain Tumor Segmentation and Radiomics Survival Prediction:
  Contribution to the BRATS 2017 Challenge](https://arxiv.org/abs/1802.10508).
  Proposed architecture reference.
- Cicek et al., [3D U-Net: Learning Dense Volumetric Segmentation from Sparse
  Annotation](https://arxiv.org/abs/1606.06650). Proposed comparison reference.
