# HipMRI Segmentation with a 3D Improved U-Net

**Status: segmentation prototype with a complete training command and a passed small GPU continuation check.**
`modules.py` connects the residual encoder, skip-connected decoder and multi-scale
segmentation heads. `objectives.py` adds a soft Dice loss and per-class Dice/IoU,
checked with synthetic inputs and labels. `dataset.py` validates case manifests,
groups scans into patient-level splits, reads native-grid NIfTI pairs and prepares
normalized, downsampled volumes while preserving their outer field of view.
`train.py` connects lazy case loading to multi-epoch training, validation,
model selection, logs and resumable checkpoints. Checks are separate in `tests/`.
Strict deterministic mode passed a two-training-case/one-validation-case CUDA
continuation diagnostic; this is not a formal accuracy experiment.
Formal experiments, the 2D baseline, prediction visualization and held-out test
evaluation are still pending. The source-data audit, frozen patient split and preprocessing checks
are recorded below. No completed comparative experiments or completed
feasibility check-off are claimed.

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

1. Confirm access to Rangpur GPU resources and the HipMRI data. A small CUDA
   forward/backward/optimizer check passed on 15 October 2026. The data directories
   at `/home/groups/comp3710/HipMRI_Study_open` are readable; the data audit below
   records the checks completed so far.
2. Audit patient identifiers, image/label pairs, label meanings, voxel spacing,
   orientation and spatial alignment.
3. Group all scans from each patient into one train/validation/test partition
   before constructing slices or patches. The split is frozen at 28/5/5 patients
   (173/19/19 scans), with seed 3710; assignments are retained outside Git.
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
| `dataset.py` | Loading, preprocessing, patient splits and augmentation | Manifest and frozen-split validation, strict NIfTI loading, normalization, matched downsampling and lazy single-case batches; augmentation pending |
| `train.py` | Training, validation, testing and learning curves | Multi-epoch command, validation-loss model selection, resumable state and metrics JSON/CSV; held-out testing and curve rendering pending |
| `checkpoint.py` | Save and restore training state | Epoch-boundary model/optimizer/configuration/progress and RNG persistence |
| `predict.py` | Loading saved weights and visualizing test inference | Not implemented |
| `tests/` | Synthetic checks using standard-library `unittest` | Model, objective, data preparation, epoch functions, checkpoints and end-to-end runner tests |

This README will develop into the final technical report. The training command
is documented below; inference commands will follow when implemented and verified.

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
`get_config()` returns the five constructor settings used for checkpoint checks.

`num_classes` is required and includes background. This interface assumes
mutually exclusive labels and at least two classes; a binary task uses two scores
per voxel (background and foreground). The original HipMRI release documents six
labels including background, as listed in the data audit below. The intended
multiclass configuration therefore uses `num_classes=6`, with each loaded case
checked against the documented IDs. The component checks use two and four
classes as synthetic examples.

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
Setting it to `False` excludes index 0 only. The documented HipMRI IDs are already
contiguous, with background at 0; the loader checks allowed values and preserves
the source IDs. Ignore labels and multilabel masks are not supported by this
initial interface.

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
Zero-sized validation/test groups are allowed for preparation checks.

Patient IDs are sorted before a local seeded shuffle, and cases are sorted within
each partition. The same records and seed reproduce the split in the same Python
environment, regardless of input order and without changing the global RNG.
Adding/removing patients can change assignments; save the final assignments
outside Git before experiments. This initial split is not stratified by labels.

The HipMRI split was frozen on 15 October 2026 using
`validation_patients=5, test_patients=5, seed=3710` and the filename prefix before
`_Week` as the patient ID:

| Partition | Patients | Scans |
| --- | ---: | ---: |
| Train | 28 | 173 |
| Validation | 5 | 19 |
| Test | 5 | 19 |

All 211 paired scans are assigned once, and patient sets are disjoint. The private
manifest, its hash and the exact case assignments are saved together; later runs
must reuse these assignments rather than regenerate a more favorable split.
`load_patient_split(path, records, manifest_sha256=...)` loads these assignments
without reshuffling. Its version-1 JSON contains `format_version`, `seed`,
`validation_patients`, `test_patients`, `patient_key_rule`, `manifest_sha256`, and
the `train`, `validation`, `test` lists of case IDs. The loader checks the exact
manifest digest, full case coverage, duplicate/unknown IDs, disjoint patient
groups and recorded holdout counts. Train and validation must be nonempty.
The frozen school-data split JSON has SHA-256
`b6fb4ce7f1050344123e4314b26af09f05ac35346a7cbd2faf8dbfe6cdaee0b9`.

These checks validate references and supplied IDs only. They cannot establish
actual patient identity, detect duplicate contents in separately copied files,
or verify MRI/label alignment. The source-data audit below is a separate check;
the manifest utilities do not read voxel contents or spatial transforms.

## Initial Rangpur data audit

A read-only CPU audit on 15 October 2026 inspected `semantic_MRs` and
`semantic_labels_only` under `/home/groups/comp3710/HipMRI_Study_open`.
It used NumPy 2.4.6 and [NiBabel 5.4.2](https://pypi.org/project/nibabel/5.4.2/).

- All 211 MRI files matched a label file after removing the respective
  `_LFOV.nii.gz` and `_SEMANTIC.nii.gz` suffixes. No unmatched files or header
  read errors were found. The prefix before `_Week` gives 38 candidate patient
  identifiers, consistent with the assignment's counts; identity was not verified.
- In native NIfTI array order, 210 pairs have shape `256 x 256 x 128` and one
  has shape `256 x 256 x 144`. Shapes match within every pair. Header spacing
  matches within `rtol=atol=1e-5`; in-plane spacing varies between scans.
- Every pair has numerically identical effective affine matrices selected by
  NiBabel. All effective affines are finite and nonsingular, with axis codes
  `L, P, S`. No conflict between enabled qform and sform matrices was detected.
- All MRIs declare millimetres, but all labels leave the spatial unit unknown.
  The audit therefore returned a nonzero status for 211 unit metadata mismatches.
  For the school copy, the reader can explicitly adopt the paired MRI's units
  after checking shape and affine agreement, recording this assumption in the
  returned metadata. This does not repair or rewrite the source headers.
- Three complete image/label pairs from two candidate patient identifiers were
  read, covering a Week0 scan, a later scan and the 144-layer volume.
  All sampled voxels were finite `int32` values;
  each sampled label volume contained exactly `0, 1, 2, 3, 4, 5`.

The original publisher's [BriefDataDescription.txt](https://data.csiro.au/dap/ws/v2/collections/52549/support/5034)
for [Labelled weekly MR images of the male pelvis](https://doi.org/10.25919/45t8-p065)
explicitly documents this mapping:

| Label | Meaning |
| --- | --- |
| 0 | Background |
| 1 | Body |
| 2 | Bones |
| 3 | Bladder |
| 4 | Rectum |
| 5 | Prostate |

Diagnostic overlays were generated for the same three sampled pairs, selecting
the axial slice with the largest area of each nonzero label. Visual inspection
of those slices found no obvious gross displacement, and the displayed labels
were consistent with the published mapping. This is a limited visual check,
not a clinical review or proof of dataset-wide anatomical alignment or voxel
validity. The 211 volumes have not all undergone voxel-level inspection.

Data were provided by HipMRI Study; see the original dataset above and
[Dowling et al. (2015)](https://doi.org/10.1016/j.ijrobp.2015.08.045).
Raw audit output, private overlays and temporary diagnostic scripts are retained
locally under the ignored `private/` directory. Preprocessing checks are described
below; held-out segmentation evaluation remains pending.

## Native-grid NIfTI loading

`load_nifti_case(record, allowed_labels=..., assume_label_units_from_image=False,
max_voxels=64000000)` reads one image/label pair and returns a `LoadedNiftiCase`.
It requires NumPy, NiBabel and PyTorch; importing the manifest/split utilities
alone still uses only the standard library. With a prepared manifest:

```python
from dataset import load_manifest, load_nifti_case

records = load_manifest("data/manifest.json")
sample = load_nifti_case(
    records[0],
    allowed_labels=range(6),
    assume_label_units_from_image=True,  # explicit policy for the school copy
)
print(sample.image.shape, sample.target.shape)
```

The returned CPU tensors are contiguous: `image` is `float32 [1,D,H,W]` and
`target` is `int64 [D,H,W]`, without a batch axis. Source indices `(I,J,K)` are
permuted to `(D,H,W)=(K,J,I)`. `affine_ijk` maps `[i,j,k,1]` to source world
coordinates; `affine_dhw` maps `[d,h,w,1]` to those same coordinates. Both are
float64 matrices. This axis permutation preserves the native anatomical
orientation. `source_shape`, source units, effective `spatial_unit`, qform/sform
codes and `label_unit_assumed` retain the geometry and assumption provenance.

Both files must be nonempty 3D single-file NIfTI volumes with a coded spatial
transform and finite, nonsingular effective affines. Shapes must agree, and
affines must match with `rtol=0, atol=1e-5` in native units. Unknown image units
and conflicting known units are rejected. The explicit assumption switch only
allows an unknown label unit to adopt the known image unit after the geometry
checks; it records that decision without changing the source file.

NIfTI slope/intercept scaling is applied before validation. Labels must be finite
integers within the explicit `allowed_labels` vocabulary before int64 conversion;
each case may contain only a subset. Image values must remain finite after
float32 conversion. Values and class IDs are otherwise preserved. The voxel
limit is checked before reading arrays and is a size guard, not a guarantee of
peak memory consumption.

This reader does not normalize, resample or augment the data. The separate
preprocessing function below prepares each loaded case. `iter_case_batches`
connects the reader and preprocessing to the training command, one case at a time.

## Per-case normalization and downsampling

`preprocess_case(sample, downsample_factor=2)` returns a `PreprocessedCase` with
CPU `float32 image [1,D,H,W]`, `int64 target [D,H,W]`, the output `affine_dhw`,
an independent copy of the source affine, the source shape, unit provenance and
normalization statistics. For a loaded case:

```python
from dataset import preprocess_case

prepared = preprocess_case(sample, downsample_factor=2)
# Add the batch dimension for the current single-case training interface.
image_batch = prepared.image.unsqueeze(0)
target_batch = prepared.target.unsqueeze(0)
```

The mean and population standard deviation are computed in float64 over the
input MRI's nonzero voxels only. This support is not an anatomical foreground
mask; labels never determine intensity statistics. Nonzero values become
`(value - mean) / max(std, 1e-8)`, and original zeros remain zero at this stage.
An entirely zero image is rejected; a constant nonzero region maps to zero and
retains its measured zero standard deviation in the metadata. Statistics are
computed independently per case, with no fitted dataset-wide normalization.

For input size `N` along an axis, the output size is `M = ceil(N / factor)`.
Images use trilinear interpolation with `align_corners=False`. Labels stay in
int64 and use the nearest source index `floor((j + 0.5) * N / M)` at output index
`j`; half-integer ties select the higher source index. This matches the image's
half-pixel coordinate convention without interpolating numeric class IDs.
Downsampling may remove small label regions; retaining every class is not
guaranteed. Image interpolation can mix zero and nonzero neighbors, so the
resized nonzero voxels need not have mean zero or standard deviation one.

The output affine is `A_source @ T`, where each spatial diagonal of `T` is
`N / M` and its corresponding translation is `(N / M - 1) / 2`. Thus the outer
voxel edges and field of view are preserved, including odd input sizes. Voxel
centres and spacing change; this is not isotropic resampling or anatomical
reorientation. The source arrays are unchanged, and no antialiasing filter,
augmentation or intensity clipping is applied. Factor 1 still normalizes and
returns independent image/label storage. See PyTorch's
[interpolation conventions](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.interpolate.html).

Rangpur CPU job `663296` checked the same three previously audited pairs with
factor 2. Two image volumes became `[1,64,128,128]` and the deeper one became
`[1,72,128,128]`. All outputs were finite, all six integer labels remained in
these samples, and the maximum world-coordinate error across the eight outer
field-of-view corners was zero. Independent NumPy mean/std calculations agreed
within `rtol=1e-8, atol=1e-9`. The source code hash was recorded with the results.
A native/preprocessed overlay from the first training case showed no obvious
image/label displacement. These are geometry and numerical checks, not evidence
of segmentation quality. Figures and full case-level logs remain private.

## Single-epoch training and validation

`train_one_epoch` accepts a model, an iterable of `(image, target)` batches,
an optimizer and a criterion. Images must be finite floating-point tensors
`[N, C, D, H, W]`, and targets must be `torch.long [N, D, H, W]` class indices.
The caller moves the model to the chosen `device` before constructing the
optimizer; the function moves batches, enters training mode and performs one
update per batch. Gradients are cleared before each update. Nonfinite loss or
gradients stop the current update; earlier successful updates are retained.

The criterion must return a scalar loss averaged equally over samples, as
`SoftDiceLoss` does. The returned `loss` weights each batch by its sample count,
so a short final batch is not given the same weight as a full one. `num_samples`
records how many samples were processed. Scalar shape and finiteness are checked;
the caller remains responsible for the criterion's reduction meaning.

`validate_one_epoch` accepts the model, batches and criterion, with explicit
`num_classes`, optional `include_background` and `device`. It uses evaluation mode
and disables gradients. Each sample must represent a complete case volume;
patches must be reconstructed first. Different-sized cases can use separate
batches. A validity-mask policy for padded images/labels is not implemented.

Validation returns loss, sample count, all per-case `dice`, `iou` and `valid` rows
in iteration order, `class_ids`, `valid_count`, `mean_dice` and `mean_iou`.
Returned tensors are on CPU. Means are per class over valid cases: false positives
on empty targets count as zero, while classes absent throughout remain NaN.
Cases receive equal weight, which is not the same as equal weight per patient
when patients have different scan counts. Keep case IDs aligned with these rows
for later reporting. Both functions leave the model in their respective modes.

The caller must supply patient-disjoint partitions from data preparation. The
command below connects these helpers to data loading and multi-epoch training.
Held-out test evaluation, rendered learning curves, mixed precision, gradient
accumulation and augmentation remain outside this initial runner.

## Training command

Run from this project directory on an allocated compute node, with PyTorch,
NumPy and NiBabel available. Supply the already frozen private manifest and split;
the paths below illustrate files placed in the ignored `data/` directory:

```bash
python -s train.py \
  --manifest data/manifest.json --split data/split.json \
  --output-dir runs/initial_3d --epochs 2 --device cuda \
  --assume-label-units-from-image --deterministic
```

This example requests two epochs; it is not a tested real-data training budget.
The unit-assumption flag is required for the school labels' unknown units.
CUDA is selected explicitly and an unavailable GPU is an error. Use `--device cpu`
for small local checks. `python -s train.py --help` lists all options. The command
uses Linux/macOS file locking to prevent concurrent writers to the same run.

Defaults are six classes, encoder channels `(16,32,64,128)`, dropout `0.3`, summed
deep supervision, factor-2 preprocessing, Adam at `1e-3`, seed 3710 and one CPU
thread. The starting objective is equal-weight cross entropy plus
`SoftDiceLoss(include_background=False)`. Cross entropy includes background;
the Dice term includes classes 1-5, including Body. These settings have not been
selected through comparative HipMRI experiments.

`--deterministic` requests strict deterministic PyTorch algorithms and cuDNN
operations; unsupported operations raise an error. It is opt-in and may reduce
speed. cuDNN benchmarking is disabled in either mode. Before initializing CUDA,
the option supplies `CUBLAS_WORKSPACE_CONFIG=:4096:8` only when no value was
provided. The requested mode and actual algorithm/workspace settings are saved
in `run.json`; resume requires the same settings. Cross entropy now computes
unreduced voxel losses followed by an explicit mean, preserving the unweighted
valid-label objective while avoiding the spatial NLL atomic mean reduction.
This is not a guarantee of identical results across hardware or PyTorch versions.

`iter_case_batches` yields one complete preprocessed volume at a time with batch
size 1, no workers, padding or cache. This accommodates the differing volume
depths and releases the original full-resolution tensors before yielding. Each
training epoch shuffles its case list with a local `Random(seed + epoch)`, using
one-based epoch numbers. Validation preserves the frozen order. Test files are
checked for existence/identity by manifest validation, but their headers and
voxel arrays are never read by the training command.

Each epoch trains on all training cases and evaluates all validation cases.
The lowest validation loss selects the best checkpoint; an exact tie keeps the
earlier epoch. Validation Dice/IoU are reported per case and per foreground class
on the preprocessed grid, with equal case weighting. They are not native-grid
or held-out test scores. Undefined overlap scores are JSON `null` with their
valid masks/counts retained; CSV cells for these scores are empty.

The output directory must be new unless `--resume` is supplied. It contains:

- `run.json`: configuration, source/environment information, manifest/split and
  resolved-input hashes, completed epochs, best/latest checkpoint filenames,
  and complete training/validation metric history with validation case IDs.
- `checkpoint-NNNNNN.pt`: model, Adam and RNG state for an epoch. Normally only
  the latest and best files are retained; they can be the same file.
- `history.csv`: a table for plotting loss and per-class validation Dice/IoU.
  Epoch timing includes loading, preprocessing, training and validation, but
  excludes checkpoint/log writes. CUDA runs also record peak allocated tensor
  memory; this excludes CPU memory and is not total GPU reservation.

Keep this directory outside Git. `runs/` is ignored, and logs include case IDs.
The manifest/split hashes and resolved path digest bind the selected inputs;
the runner does not hash the contents of every MRI/label file.

To continue this example to a total of five epochs:

```bash
python -s train.py \
  --manifest data/manifest.json --split data/split.json \
  --output-dir runs/initial_3d --epochs 5 --device cuda \
  --assume-label-units-from-image --deterministic --resume
```

Repeat any nondefault training options used originally. Resume requires matching
settings, source hashes, environment, manifest/split and resolved data paths.
It restores the latest committed epoch, not the best epoch, and restores RNG
after constructing the model/optimizer. `--epochs` is the target total: a larger
value continues, the completed value performs no updates, and a smaller one is
rejected. Use a new output directory when changing the training recipe.

`run.json` is the commit point: the new checkpoint is saved first, then the
complete JSON state is atomically replaced, then old unreferenced checkpoints
are removed. If the process is interrupted before that replacement, resume uses
the prior committed epoch and discards any uncommitted checkpoint. CSV can be
rebuilt from JSON. An interruption before the first completed epoch restarts
from the recorded seed/configuration. No partial-epoch recovery is attempted.

## Checkpoint save and restore

`save_checkpoint(path, model, optimizer, completed_epochs=..., best_validation_loss=...)`
writes an epoch-boundary checkpoint. It stores model configuration and weights,
optimizer state, the number of completed epochs, an optional best validation loss,
and Python/Torch CPU random states. CUDA random states are included when CUDA has
already been initialized. Checkpoints belong in the ignored `checkpoints/`
directory or outside Git. A temporary file in the destination directory is fully
written before replacing the destination; a failed write leaves the old file intact.

To resume, create a fresh model with the same constructor settings, move it to
the intended device, and create its optimizer with the same class and parameter
group ordering. Call `load_checkpoint(path, model, optimizer)` and use its
`completed_epochs` result as the next zero-based epoch index. Saved optimizer
settings, including learning rate, replace the new optimizer's settings. Loading
checks configuration, tensor shape/dtype and optimizer parameter names/order;
discard the fresh objects if loading fails. The checkpoint contains state
dictionaries and primitive metadata and is read with `weights_only=True`.

Random states are restored after model/optimizer loading. `restore_rng=False`
keeps the caller's random state when transferring weights/state to a different
run. Reproducing the next update requires the same next batches, loss settings
and environment. DataLoader generators/workers, sampler position, NumPy/MPS RNG,
schedulers and AMP scalers are not saved. CUDA restoration requires the saved
number of devices. Restoration and continued updates have been exercised on
Rangpur; exact numerical reproduction additionally depends on the selected CUDA
algorithms, as documented below. Mid-epoch resume is not supported. Model modes
and gradients are handled by the epoch functions.

These low-level helpers do not select the best checkpoint themselves. The
training command above selects it using validation loss and excludes test data.

## Synthetic tests

Run all checks from this project directory, with PyTorch, NumPy and NiBabel
available. The tested NiBabel version is 5.4.2:

```bash
python -s -m pip install nibabel==5.4.2
python -s -m unittest discover -s tests -v
```

To run one group, use `python -s -m unittest tests.test_modules -v`, substituting
`tests.test_objectives`, `tests.test_dataset`, `tests.test_nifti`,
`tests.test_preprocessing`, `tests.test_training_data`, `tests.test_train`,
`tests.test_checkpoint` or `tests.test_runner` as needed.
No additional test framework is required. Production files no longer contain
executable self-tests.

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
- `test_nifti.py`: temporary synthetic NIfTI files check exact voxel/coordinate
  correspondence after axis permutation, slope/intercept scaling, explicit unit
  assumptions, invalid geometry, label vocabulary and integer precision,
  nonfinite/overflow rejection and size limits before voxel reads.
- `test_preprocessing.py`: hand-calculated normalization, label-independent
  statistics, empty/constant inputs, even/odd grid geometry, linear-ramp image
  interpolation, integer nearest-neighbor ties and precision, unchanged source
  arrays, independent output storage, and invalid metadata or tensors.
- `test_train.py`: model/optimizer integration, fresh gradients per batch,
  sample-weighted loss, hand-calculated validation summaries, batch regrouping,
  unchanged model state during validation, and invalid/nonfinite inputs.
- `test_checkpoint.py`: interrupted-versus-continuous synthetic training with
  dropout and optimizer state, configuration compatibility, RNG restoration and
  preservation of an existing checkpoint when a write fails.
- `test_training_data.py`: frozen-split integrity and hash binding, patient
  separation, lazy real NIfTI batches of different shapes, local shuffle RNG,
  volume release, test-data isolation and case-specific loading errors.
- `test_runner.py`: actual model training from temporary NIfTI files, CLI use,
  validation-only selection and ties, strict JSON missing-class metrics, exact
  CPU continuation with dropout, configuration/path protection and recovery
  after interrupted epoch commits.

Gradient-connectivity checks disable dropout to avoid randomly dropping every
channel in a tiny block. A separate test checks default dropout (`0.3`) with wider
features and seeds 17 and 18. Training-loop checks update models on synthetic
tensors only; they do not establish segmentation accuracy on HipMRI.

## Environment and reproducibility

With the complete training command added, all 66 `unittest` methods passed
locally on 15 October 2026 using Python 3.13.9, PyTorch 2.14.0, NumPy 2.1.3 and
NiBabel 5.4.2. The 19 new data-entry and runner tests use generated data. A real
model with nonzero dropout produced identical weights, Adam state, random states
and non-timing metrics for two uninterrupted CPU epochs and one epoch followed
by resume to two. Injected failures before the first epoch and before the second
epoch's JSON commit were also recoverable. These CPU results do not establish
CUDA continuation equivalence or real-data segmentation quality. After the
NIfTI fixture correction below, all 66 tests also passed on Rangpur in job
`663521`, using Python 3.11.15, PyTorch 2.13.0+cu130, NumPy 2.4.6 and NiBabel 5.4.2.
With the deterministic-mode and loss-equivalence tests added, all 68 tests passed
in the same software environment in job `663545`. The real-data CUDA continuation
comparison in that job passed exactly within the scope described below.

With preprocessing added, all 47 `unittest` methods passed locally on
15 October 2026 using Python 3.13.9, PyTorch 2.14.0, NumPy 2.1.3 and NiBabel 5.4.2.
The nine new preprocessing tests use synthetic tensors; the separate real-data
CPU check above used the existing Rangpur environment described below.

With native-grid NIfTI loading added, all 38 `unittest` methods passed locally on
15 October 2026 with Python 3.13.9, PyTorch 2.14.0, NumPy 2.1.3 and NiBabel 5.4.2.
The nine new reader tests use small generated volumes, separate from the private
HipMRI inspection figures and logs.

An earlier revision of `dataset.py`, verified by SHA-256 at the time, passed a
three-case reader check on Rangpur CPU node `vcpu-2` (job `663249`) on
15 October 2026, using Python
3.11.15, PyTorch 2.13.0+cu130, NumPy 2.4.6 and NiBabel 5.4.2. Two cases returned
images `[1,128,256,256]` and one returned `[1,144,256,256]`, with corresponding
three-dimensional targets. All voxel values matched the source arrays after
axis permutation, and checked corner/centre coordinates matched in world space.
Default rejection of unknown label units and the explicit adoption policy both
behaved as specified. This was a CPU loading check, without preprocessing or
model training.

With checkpoint support added, all 29 `unittest` methods passed locally on
15 October 2026. On the same CPU environment and next batch, restoring a model
with nonzero dropout and Adam state reproduced the uninterrupted next epoch's
loss, model/optimizer tensors and final random states exactly. This does not
establish reproducibility across devices, software versions or data loaders.

With the single-epoch functions added, all 22 `unittest` methods passed locally
on 15 October 2026 using the macOS CPU environment below. This includes actual
optimizer updates on synthetic tensors and validation without parameter updates.

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

On 15 October 2026, Rangpur job `663176` completed successfully on `a100-1`
with exit code `0:0`. The existing `torch` Conda environment uses Python 3.11
and PyTorch 2.13.0+cu130, launched with `python -s` to exclude user site-packages.
The allocated NVIDIA A100-PCIE-40GB reported driver 590.48.01 and CUDA was
available. A small `Conv3d` on a synthetic `[1, 1, 8, 16, 16]` input completed
forward propagation, backward propagation and an SGD update with finite loss
and gradients. This establishes basic CUDA execution in that environment;
the project model and realistic-volume GPU memory use need their own check.

A 20-update diagnostic using the project's full four-level 3D model and one
fixed training case completed as Rangpur job `663304` on `a100-9` on
15 October 2026, from 19:05:25 to 19:05:46 Brisbane time, with exit code `0:0`.
It used factor-2
preprocessing, batch size 1, seed 3710, Adam at `1e-3`, and cross entropy plus
foreground soft Dice loss. Dropout is disabled for this diagnostic; the model's
default remains `0.3`. Loss on that same training case fell from `3.070139` to
`0.834431`. The 20 updates took 2.643 seconds, excluding loading, initialization
and the final check; the full allocation lasted 21 seconds. PyTorch peak allocated
tensor memory was 2037.8 MiB during updates and the final check, not total GPU
usage or reserved memory. Rectum and prostate predictions were empty after these
updates. This demonstrates a working training path, not segmentation quality or
hyperparameter selection. No validation/test cases were used. It did not test
the new multi-epoch command or CUDA checkpoint restoration.

The first multi-epoch GPU diagnostic attempt, job `663499`, stopped in its
synthetic server test preflight before real-data CLI training. One test rewrote
a file-backed NIfTI fixture to its own path and lost its label values on Linux.
A separate CPU reproduction (`663504`) confirmed that disabling memory mapping
preserves those values. The two affected fixtures now use `mmap=False` for
header-only rewrites and assert that voxels are unchanged. The 21 affected local
tests passed after this test-only correction, followed by all 66 server tests
in job `663521`. Raw logs, source hashes and the reproduction are retained privately.

The corrected full-command diagnostic (`663521`) completed on `a100-9` from
19:40:25 to 19:41:17 Brisbane time on 15 October 2026, with exit code `0:0`.
It used two cases from distinct frozen training patients and one frozen validation
case, with no test cases, full model channels and dropout `0.3`. Three separate
processes performed two uninterrupted epochs, one epoch in another directory,
and restoration of that directory to two epochs. Training, validation, best/latest
checkpoint selection and continued parameter updates completed successfully.
Configurations and final Python/CPU/CUDA random states matched; source and
original split hashes remained unchanged. Peak allocated tensor memory was
approximately 2062 MiB.

This default-algorithm run did **not** pass the predeclared numerical comparison
(`rtol=1e-5`, `atol=1e-7`): the maximum checkpoint tensor difference was `0.003968`
and the maximum non-timing history difference was `0.001240`. Differences already
occurred in the first epoch, before checkpoint restoration. Its result remains
`completed_requires_numerical_review`, independently of the successful Slurm exit.
It demonstrates functional continuation, not exact continuation or segmentation
accuracy. The three-case subset is a diagnostic and does not replace the planned
173-case training and 19-case validation experiment.

A synthetic CUDA operator probe (`663535`) confirmed that the original 5D mean
cross entropy fails strict deterministic mode in this server environment at
`nll_loss2d_forward_out_cuda_template`. Unreduced voxel cross entropy followed by
an explicit mean completed forward/backward with finite, nonzero gradients.
The opt-in deterministic mode and explicit loss reduction are now implemented.
Sixteen targeted CPU tests passed, including loss/gradient equivalence against
the old expression and deterministic continuation/configuration protection.
Disabling cuDNN
benchmarking alone does not guarantee deterministic CUDA operations; see the
[PyTorch reproducibility guidance](https://docs.pytorch.org/docs/2.14/notes/randomness.html).

The revised strict-mode diagnostic (`663545`) then completed on `a100-3` from
19:51:55 to 19:52:52 Brisbane time on 15 October 2026, with exit code `0:0` and
result `passed_exact`. All 68 synthetic server tests passed before the three
real-data command processes. The same two-training-case/one-validation-case
selection rule, full model, dropout `0.3`, seed and optimizer settings were used.
Every command explicitly requested `--deterministic`; the saved policy confirms
strict algorithms, cuDNN determinism, disabled benchmarking and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`.

Continuous and resumed runs both completed two epochs and selected epoch 2.
All 246 checkpoint tensors, scalar progress fields, Python/CPU/CUDA random states
and non-timing epoch history matched exactly (maximum absolute difference zero).
Resuming changed the epoch-1 weights, so this was actual continued training.
Training losses were `2.688745` and `1.870584`; validation losses were `1.880930`
and `1.722612`. Peak allocated tensor memory was at most `2061.77 MiB` (about
2.01 GiB); the whole Slurm job lasted 57 seconds, including tests and three
process launches. Timing and memory measurements were deliberately excluded from
the equality comparison. Original source/split hashes were checked before and
after, and all downloaded log/JSON/CSV artifact hashes were verified.

This demonstrates exact continuation for this small run on the same hardware and
software configuration. It does not establish cross-device/version equivalence,
long-run stability or segmentation quality. The earlier default-mode differences
remain recorded. The next experiment should use the frozen training/validation
partitions to check learning and resource needs before selecting a longer budget;
the held-out test partition remains unused.

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
and patient-grouped splits, plus single-epoch training, validation and checkpoint
functions. Codex also moved synthetic checks into `tests/` and added training-loop
and checkpoint tests. Codex assisted with the Rangpur CUDA diagnostic, installing
the MRI-reading dependency and auditing source-file pairing, geometry metadata
and three sampled volumes. Codex located the publisher's label mapping and
generated and visually inspected diagnostic overlays of the sampled volumes.
Codex also implemented the native-grid NIfTI reader and its synthetic tests,
including explicit unit handling and preservation of voxel/world coordinates.
Codex implemented and tested nonzero-voxel normalization and geometry-preserving
downsampling, froze the patient split, and checked three preprocessed real cases
and a training-case overlay on Rangpur. Codex prepared and ran the completed
single-case GPU diagnostic described above, preserving its logs and results.
Codex implemented the complete training command, frozen-split loading, lazy
case batches, validation-loss model selection, metric logs and resumable run
state, and added synthetic end-to-end tests including interrupted writes and
nonzero-dropout continuation. Codex also reproduced and fixed the Linux NIfTI
fixture rewrite issue found during the first server preflight. Codex ran the
small real-data multi-epoch and CUDA continuation diagnostic and preserved its
functional success and failed numerical-equivalence comparison separately.
Codex investigated the difference using a synthetic strict-CUDA loss probe,
implemented the opt-in deterministic mode and explicit voxel-mean cross entropy,
added loss/gradient equivalence and reproducibility-policy tests, and ran the
strict GPU continuation diagnostic, which passed exactly on the small subset.
The checks exercise
feature/logit shapes, gradients through the model, independent skip and scale
contributions, coarse-to-fine output summation, state restoration, input-size
limits, default dropout behavior, residual-path behavior, known overlap scores,
empty-class policies, objective gradients, file references and patient-group
separation, optimizer updates, validation aggregation and checkpoint continuation;
they do not validate segmentation quality or real patient identity.
No held-out segmentation-quality evaluation on HipMRI has been performed.
The student still needs to review and understand the implementation.
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
- Dowling and Greer, [Labelled weekly MR images of the male pelvis](https://doi.org/10.25919/45t8-p065),
  CSIRO Data Access Portal. The attached
  [BriefDataDescription.txt](https://data.csiro.au/dap/ws/v2/collections/52549/support/5034)
  supplies the numeric label definitions.
- Dowling et al. (2015), [HipMRI study publication](https://doi.org/10.1016/j.ijrobp.2015.08.045).
- PyTorch documentation for [softmax](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.softmax.html)
  and [one-hot encoding](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.one_hot.html).
  Tensor conversion APIs used by the Dice loss.
