# HipMRI Segmentation with a 3D Improved U-Net

**Status: segmentation prototype and initial data preparation utilities.**
`modules.py` connects the residual encoder, skip-connected decoder and multi-scale
segmentation heads. `objectives.py` adds a soft Dice loss and per-class Dice/IoU,
checked with synthetic inputs and labels. `dataset.py` validates case manifests
and groups scans into patient-level splits. Checks are separate in `tests/`.
MRI voxel loading, preprocessing, the training loop and real-data evaluation are
not implemented. No HipMRI experimental results or completed feasibility
check-off are claimed.

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
| `objectives.py` | Segmentation loss and overlap metrics | Soft Dice loss and per-volume Dice/IoU |
| `dataset.py` | Loading, preprocessing, patient splits and augmentation | Manifest validation and patient-grouped splitting; voxel loading and transforms pending |
| `train.py` | Training, validation, testing, checkpointing and learning curves | Not implemented |
| `predict.py` | Loading saved weights and visualizing test inference | Not implemented |
| `tests/` | Synthetic checks using standard-library `unittest` | Model, objective and data-preparation tests |

This README will develop into the final technical report. Training and inference
commands will be added once they exist and have been verified.

## Model components

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

## Dice loss and overlap metrics

`SoftDiceLoss` accepts raw floating-point logits `[N, C, D, H, W]` and
`torch.long` class-index labels `[N, D, H, W]`. It applies softmax over classes
and computes the following for each sample and selected class, summing over
spatial dimensions only:

```text
soft_dice = (2 * sum(probability * target_one_hot) + smooth)
            / (sum(probability) + sum(target_one_hot) + smooth)
loss = mean(1 - soft_dice)
```

This uses the linear denominator of the multiclass Dice formulation in the
architecture reference and the supplied segmentation lecture (slide 28).
The per-sample averaging and smoothing policy are explicit project choices.
`smooth=1e-5` is added to both numerator and denominator. Selected classes remain
in the loss even when their target is empty, so predicted probability mass is
still penalized. Half/bfloat16 logits are converted to float32 before softmax
and accumulation; float64 inputs retain their precision. The smoothing value
must be finite, positive and within the accumulation dtype's normal range.

`dice_iou_per_class` accepts two class-index masks `[N, D, H, W]`, plus an explicit
`num_classes`. Convert predictions with `logits.argmax(dim=1)` before calling.
It returns `dice` and `iou` tensors `[N, K]`, a boolean `valid` tensor of the same
shape, `valid_count` for each class in the supplied batch, and `class_ids` identifying
the original label index for each column. These hard metrics use unsmoothed
integer overlap counts and are separate from the differentiable training loss.

| Class presence in a sample | Dice/IoU | Valid for averaging |
| --- | --- | --- |
| Present in both prediction and target | Computed from overlap | Yes |
| Present in only one | 0 | Yes, including false positives on empty targets |
| Absent in both | NaN (undefined) | No |

For both APIs, `include_background=True` includes all classes by default.
Setting it to `False` excludes index 0 only. A future data loader must first
verify and map labels to contiguous indices with background at 0. The actual
HipMRI class mapping is not yet established. Ignore labels and multilabel masks
are not supported by this initial interface.

For case-level evaluation, call the metric on reconstructed volumes. Retain all
per-case rows before computing per-class means and valid counts; do not average
batch means or slice/patch scores as if they were full-case scores. Exclude NaNs
when averaging, report the valid count, and keep the mean undefined when a class
has no valid cases. The empty-class policy is a documented project convention
to revisit if the course supplies a specific evaluation protocol.

Dice alone can give weak gradients for empty target classes; whether to combine
it with cross entropy must still be selected using training/validation experiments.
This step does not establish the final loss configuration or segmentation accuracy.

## Case manifests and patient splits

`load_manifest` reads a JSON list with exactly four fields per entry. For example,
the following illustrates one entry, using fictitious identifiers and paths:

```json
[
  {
    "case_id": "example_scan_01",
    "patient_id": "example_patient_01",
    "image_path": "images/example_scan_01.nii.gz",
    "label_path": "labels/example_scan_01.nii.gz"
  }
]
```

Keep real manifests in the ignored `data/` directory or outside the repository.
Relative image/label paths resolve against the manifest's directory; absolute
paths are also supported. Every referenced file must exist. Case IDs must be
unique, and IDs are case-sensitive with no surrounding whitespace. The current
interface requires one image and one label file per scan and rejects reuse of a
physical file, including symlink/hardlink aliases and image/label cross-use.

`split_cases_by_patient(cases, validation_patients=..., test_patients=..., seed=3710)`
returns `PatientSplit.train`, `.validation` and `.test`, each a tuple of
`CaseRecord` objects. Counts refer to patients, not scans. All scans sharing a
supplied patient ID stay together, and at least one patient must remain in training.
Zero-sized validation/test groups are allowed for preparation checks; experiment
holdout counts have not yet been selected.

Patient IDs are sorted before a local seeded shuffle, and cases are sorted within
each partition. The same records and seed reproduce the split in the same Python
environment, regardless of input order and without changing the global RNG.
Adding/removing patients can change assignments; save the final assignments
outside Git before experiments. This initial split is not stratified by labels.

These checks validate references and supplied IDs only. They cannot establish
actual patient identity, detect duplicate contents in separately copied files,
or verify MRI/label alignment. Real identifiers, label values, orientations and
voxel spacing must be audited before implementing voxel loading and transforms.

## Synthetic tests

Run all checks from this project directory, with PyTorch available:

```bash
python -s -m unittest discover -s tests -v
```

To run one group, use `python -s -m unittest tests.test_modules -v`, substituting
`tests.test_objectives` or `tests.test_dataset` as needed. No additional test
framework is required. Production files no longer contain executable self-tests.

- `test_modules.py`: residual shortcuts, feature/logit shapes, even/odd/thin inputs,
  invalid configurations, state restoration, dropout behavior, gradients through
  every encoder/decoder/head, independent skip contributions, hand-calculated
  coarse-to-fine sums, and equivalence of both head settings at one decoder level.
- `test_objectives.py`: hand-calculated soft/hard scores, false positives/negatives,
  empty classes, background selection, numerical gradient checking, half/bfloat16
  accumulation over 131072 voxels, extreme logits, invalid inputs, and Dice
  gradients with deep supervision enabled and disabled.
- `test_dataset.py`: relative/absolute paths, missing or reused files, invalid
  records, file aliases, complete and disjoint patient groups, reproducibility,
  unchanged global RNG state, and invalid split counts. These tests create and
  clean up temporary placeholder files; they do not contain real MRI volumes.

Gradient-connectivity checks disable dropout to avoid randomly dropping every
channel in a tiny block. A separate test checks default dropout (`0.3`) with wider
features and seeds 17 and 18. These tests do not train a model or establish
segmentation accuracy on HipMRI.

## Environment and reproducibility

After extracting the checks into `tests/` and adding manifest/split validation,
all 15 `unittest` methods passed locally on 15 October 2026 in the same macOS CPU
environment described below. Data-preparation tests used temporary synthetic
files, so this is not evidence of successful MRI loading or real-data evaluation.

The component and segmentation model checks passed locally on macOS CPU with Python
3.13.9 and PyTorch 2.14.0 on 14 October 2026, using seed 3710. Additional checks
covered the default four-level encoder/decoder, invalid channel configurations,
incorrect feature counts, ranks, channels, batches and spatial ordering. A
double-precision numerical gradient check passed for both inputs of the decoder
block. Single-voxel bottlenecks were rejected; valid even-sized, odd-sized and
thin volumes were accepted.

The objective checks also passed with the same local Python/PyTorch environment
on 14 October 2026, using seed 3710. No CUDA execution or real-data performance
has been validated by these checks.

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
summed-output deep supervision, Dice loss, overlap metrics, manifest validation
and patient-grouped splits. Codex also moved synthetic checks into `tests/`.
The checks exercise
feature/logit shapes, gradients through the model, independent skip and scale
contributions, coarse-to-fine output summation, state restoration, input-size
limits, default dropout behavior, residual-path behavior, known overlap scores,
empty-class policies, objective gradients, file references and patient-group
separation; they do not validate segmentation quality or real patient identity.
No training or empirical validation on HipMRI has been
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
- PyTorch documentation for [softmax](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.softmax.html)
  and [one-hot encoding](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.one_hot.html).
  Tensor conversion APIs used by the Dice loss.
