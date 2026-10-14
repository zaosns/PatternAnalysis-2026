# HipMRI Segmentation with a 3D Improved U-Net

**Status: segmentation prototype with summed-output deep supervision.**
`modules.py` connects the residual encoder, skip-connected decoder and multi-scale
segmentation heads, with checks using synthetic inputs and labels. The data
loader, training and evaluation are not implemented. No HipMRI experimental
results or completed feasibility check-off are claimed.

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
| `modules.py` | PyTorch model components | Residual encoder, decoder, segmentation heads and summed-output deep supervision |
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
from highest resolution to bottleneck, retaining earlier features for decoder
skip connections. Each downsampled axis has size `ceil(n / 2)`; input
dimensions need not be divisible by powers of two. A bottleneck containing only
one spatial voxel is rejected before computation because instance normalization
cannot use it. Thin volumes with more than one voxel at the bottleneck are allowed.

`DecoderBlock3D` resizes a lower-resolution tensor to its skip's exact spatial
shape using nearest-neighbor interpolation. A 3x3x3 convolution adjusts its
channels, then the result is concatenated with the skip. A 3x3x3 convolution and
a 1x1x1 convolution fuse the features and reduce the concatenated channel count,
with instance normalization and LeakyReLU. This follows the localization-pathway
idea in the reference paper, extending the resizing to support odd input sizes.

`Decoder3D` consumes the entire encoder feature tuple and checks its level count,
channels, batches and adjacent spatial sizes. It returns decoded features from
highest to lowest resolution, excluding the consumed bottleneck. Output level
`i` matches encoder level `i` in spatial size and channels. At least two encoder
levels are required; a single-level encoder has no decoding stage. For four
encoder levels, three decoder features are returned for the segmentation heads.

The default widths `(16, 32, 64, 128)` and dropout probability `0.3` are configurable
starting points, not settings selected through HipMRI experiments.

`SegmentationHead3D` uses a 1x1x1 convolution to convert features into raw class
scores (logits). `DeepSupervisionHead3D` applies one such head to each decoder
level and combines their scores from coarse to fine. At each step, it resizes
the accumulated scores to the next finer feature's exact spatial dimensions
using nearest-neighbor interpolation, then adds that level's scores. Resizing
is performed between adjacent levels; for odd dimensions this can differ from
resizing every coarse output directly to the highest resolution.

This follows the elementwise-summed segmentation layers described in
[Isensee et al., section 2.1](https://arxiv.org/pdf/1802.10508). A loss on the
combined output supplies gradients to every head. This version does not add
separate losses at individual scales. Using every decoder level and exact-size
resizing is this project's adaptation, not a claim of an exact paper reproduction.

`SegmentationModel3D` joins the encoder, decoder and output heads. With the default
`deep_supervision=True`, the four encoder levels provide three decoder features
and three segmentation heads. Set `deep_supervision=False` to instantiate only
the highest-resolution head for a future controlled comparison. With only two
encoder levels, there is one decoder feature and either setting reduces to one
head. Both settings return one logits tensor `[N, num_classes, D, H, W]` at the
input's spatial resolution in training and evaluation modes. The encoder and
decoder still return features when used separately. State restoration requires
matching model settings, including the deep-supervision flag.

`num_classes` is required and includes background. This interface assumes
mutually exclusive labels and at least two classes; a binary task uses two scores
per voxel (background and foreground). The actual HipMRI label mapping and class
count will be decided after inspecting the data. The checks use two and four
classes only as synthetic examples.

The model returns logits without softmax or argmax, as expected by PyTorch's
[cross-entropy loss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html).
For class-index targets, labels have shape `[N, D, H, W]`, dtype `torch.long` and
values from `0` through `num_classes - 1`. An eventual inference caller can use
`logits.argmax(dim=1)` to obtain class indices. Cross entropy is used here to
check gradient flow, not as a finalized choice of training objective.

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
- The connected encoder and decoder: matching input resolution for even and odd
  inputs, non-doubling channel widths, finite gradients and state restoration.
  Encoder features are also detached into independent gradient-tracking tensors
  to check that every skip and the bottleneck affect the highest-resolution output.
- Thin-volume decoding, the minimum two-level configuration, and rejection of
  a decoder configuration with only one encoder level.
- The default four-level encoder/decoder and rejection of malformed feature
  counts, ranks, channels, batches, spatial sizes, single-voxel features and
  reversed feature ordering. The ordering check uses equal channel counts so
  it must detect the spatial error instead of only a channel mismatch.
- Default encoder dropout (`0.3`): training outputs repeat with the same seed
  and change with a different seed, gradients remain finite, and evaluation
  outputs remain identical when the seed changes.
- The segmentation head: known negative and greater-than-one scores are returned
  unchanged, including for a single voxel; invalid class counts and input shapes
  are rejected.
- The connected model with deep supervision enabled and disabled: even and odd
  spatial sizes, one and two input channels, two and four output classes, finite
  cross-entropy gradients through the encoder, decoder and every instantiated
  head, unchanged inputs, and state restoration.
  Forward checks also cover the default four-level model and a thin two-level
  volume.
- Deep supervision: a hand-computed three-scale sum with odd sizes, independent
  gradients into every scale and every head, invalid feature inputs, and exact
  equivalence to the single head when only one decoder level exists.

The gradient-connectivity checks disable dropout to avoid randomly dropping every
feature channel in a tiny block. The separate dropout check uses the default
probability with wider features and seeds 17 and 18. All of these checks run with
the command above. They do not load MRI data, train a segmentation model or
measure Dice/IoU.

## Environment and reproducibility

The component and segmentation model checks passed locally on macOS CPU with Python
3.13.9 and PyTorch 2.14.0 on 14 October 2026, using seed 3710. Additional checks
covered the default four-level encoder/decoder, invalid channel configurations,
incorrect feature counts, ranks, channels, batches and spatial ordering. A
double-precision numerical gradient check passed for both inputs of the decoder
block. Single-voxel bottlenecks were rejected; valid even-sized, odd-sized and
thin volumes were accepted.

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
and implementing the residual block, encoder, decoder, segmentation heads,
summed-output deep supervision and synthetic checks. The checks exercise
feature/logit shapes, gradients through the model, independent skip and scale
contributions, coarse-to-fine output summation, state restoration, input-size
limits, default dropout behavior and residual-path behavior; they do not validate
segmentation quality. No training or empirical validation on HipMRI has been
performed. The student still needs to review and understand the implementation.
This section will be updated with the assistance received and the checks actually
performed during development.

## References

- Supplied COMP3710/COMP7710 report specification, sections 2.2-6. The supplied
  filename is v2.3 while its cover says DRAFT/Version 2.2; confirm deadlines and
  any revised requirements against the course announcements.
- Isensee et al., [Brain Tumor Segmentation and Radiomics Survival Prediction:
  Contribution to the BRATS 2017 Challenge](https://arxiv.org/abs/1802.10508).
  Architecture reference, especially section 2.1; the project includes the
  implementation adaptations described above.
- Cicek et al., [3D U-Net: Learning Dense Volumetric Segmentation from Sparse
  Annotation](https://arxiv.org/abs/1606.06650). Proposed comparison reference.
