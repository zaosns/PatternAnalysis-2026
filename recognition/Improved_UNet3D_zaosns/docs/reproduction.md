# Reproduction guide

Run these commands from `recognition/Improved_UNet3D_zaosns/`.
They create new private runs; the results already reported in [README](../README.md) remain historical records.
Dataset access, original weights and private orchestration records are not included in Git.
Use the school-provided HipMRI data under its data-use agreement and retain its acknowledgment.

## Environment and execution

The recorded training/evaluation environment was:

| Component | Recorded version/device |
| --- | --- |
| Python | 3.11.15 |
| PyTorch | 2.13.0+cu130 |
| NumPy / NiBabel | 2.4.6 / 5.4.2 |
| CUDA runtime / cuDNN | 13.0 / 9.20.0 |
| NVIDIA driver | 590.48.01 |
| GPU | NVIDIA A100-PCIE-40GB |

On Rangpur, the existing `torch` environment was used:

```bash
source "$HOME/miniconda3/bin/activate"
conda activate torch
python -s -m pip install numpy==2.4.6 nibabel==5.4.2 matplotlib==3.11.2
python -s -c "import torch; print(torch.__version__, torch.version.cuda)"
```

This installs supporting packages, not PyTorch; a compatible PyTorch installation is a prerequisite.
Server prediction PNG metadata records Matplotlib 3.11.2; local report figures used 3.10.0.
This version evidence is not a complete environment lockfile for every historical run.
`python -s` excludes conflicting user-site packages.
Run CUDA commands, including CUDA evaluation-plan freezing, inside an allocated GPU job.
CPU prediction is supported; the GPU benchmark requires CUDA and has no CPU fallback.
Matching seeds alone does not guarantee identical results across software or hardware changes.

The optional connected-component diagnostics and their tests also require SciPy
(`python -s -m pip install scipy==1.17.1` for the measured server version).
This dependency is separate from model training.

The optional augmentation and required-file test entries were added after the
original test and unaugmented CV experiments. Source hashes remain strict: current code cannot
silently resume or evaluate a run made with different source files. Use the exact
archived source snapshot and environment for historical runs. For new experiments,
create new runs and new validation/test plans; do not edit stored source hashes
or replace the original test results.

## Manifest and original patient split

`dataset.py` exposes Python functions, not a data-preparation CLI.
A manifest is a JSON list whose records contain exactly `case_id`, `patient_id`, `image_path`, `label_path`.
Paths may be absolute or relative to the manifest file. Files must exist; duplicate cases or reused files are rejected.
The following preparation reads filenames and metadata only, writes new files exclusively, and preserves existing files:

```bash
python -s - <<'PY'
from pathlib import Path
import hashlib, json, re
from dataset import load_manifest, load_patient_split, split_cases_by_patient

root = Path('/home/groups/comp3710/HipMRI_Study_open')
out = Path('private')
out.mkdir(exist_ok=True)
def indexed(folder, suffix):
    result = {}
    for path in sorted((root / folder).glob('*' + suffix)):
        key = path.name[:-len(suffix)]
        assert re.fullmatch(r'[^_]+_Week\d+', key), key
        assert key not in result, key
        result[key] = str(path.resolve())
    return result
images = indexed('semantic_MRs', '_LFOV.nii.gz')
labels = indexed('semantic_labels_only', '_SEMANTIC.nii.gz')
assert images and images.keys() == labels.keys()
manifest = [dict(case_id=k, patient_id=k.split('_Week')[0],
                 image_path=images[k], label_path=labels[k]) for k in sorted(images)]
def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
manifest_path, split_path = out / 'manifest.json', out / 'split.json'
save(manifest_path, manifest)
records = load_manifest(manifest_path)
groups = split_cases_by_patient(records, validation_patients=5, test_patients=5, seed=3710)
digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
payload = dict(format_version=1, seed=3710, validation_patients=5, test_patients=5,
    patient_key_rule='Filename prefix before _Week; source identity not independently verified',
    manifest_sha256=digest,
    **{name: [c.case_id for c in getattr(groups, name)] for name in ('train', 'validation', 'test')})
save(split_path, payload)
load_patient_split(split_path, records, manifest_sha256=digest)
for name in ('train', 'validation', 'test'):
    cases = getattr(groups, name)
    print(name, len(cases), 'scans;', len({c.patient_id for c in cases}), 'patients')
PY
```

The original 211 scans from 38 filename-derived patient groups yielded 173/19/19 scans and 28/5/5 patients.
The split function sorts patient IDs, shuffles with `random.Random(3710)`, assigns the first five to validation,
the next five to test, and the remainder to training; each case list is sorted by case ID.
The saved split must contain exactly the nine fields shown above, with case IDs in the three lists.
Every case appears once, and all scans of a patient stay together. The filename rule is not independent identity verification.
Hashes bind exact manifest/split bytes; changing paths, formatting or resolved inputs prevents replay of an existing run.
For historical weights, retain their original files and paths instead of generating replacement metadata.

## Train the original eight-epoch comparison

These commands use new output directories, six classes, one CPU thread and deterministic algorithms:

```bash
python -s train_baseline2d.py \
  --manifest private/manifest.json --split private/split.json \
  --output-dir private/repro-2d --epochs 8 --device cuda \
  --channels 16 32 64 128 --downsample-factor 2 \
  --learning-rate 0.001 --seed 3710 --threads 1 \
  --deterministic --assume-label-units-from-image

python -s train.py \
  --manifest private/manifest.json --split private/split.json \
  --output-dir private/repro-3d --epochs 8 --device cuda \
  --channels 16 32 64 128 --dropout 0.3 --downsample-factor 2 \
  --learning-rate 0.001 --seed 3710 --threads 1 \
  --deterministic --assume-label-units-from-image
```

The 2D CLI has no `--dropout` option. The units flag adopts known image units for labels with unknown units
only after geometry checks; it does not bypass shape/affine checks or rewrite the source files.
Fresh output directories must not already exist. Each run saves `run.json`, `history.csv`, and named checkpoints.
`run.json` identifies the latest and minimum-validation-loss checkpoints; only those checkpoint files are retained.
Training uses the training partition and evaluates validation each epoch. It never evaluates the test partition.

### Optional augmentation for new training runs

Add `--augment` to either training command and choose a new output directory.
For matched 2D/3D experiments, enable it for both models with the same seed and
split. The paired evaluator rejects unequal augmentation settings.

The policy applies a width/source-I flip to both image and target with probability
0.5, then multiplies the prepared MRI intensities by a random factor in [0.9, 1.1].
Label IDs and zero image background are preserved. No interpolation, axis swap,
crop or affine resampling is added. These are temporary training tensors, not
newly saved images with altered physical coordinates. Width is an array axis,
not a claim about verified anatomical laterality.

The transforms run only on training records. Validation and test always use the
unchanged preparation pipeline. A local RNG keyed by the one-based epoch seed
(`seed + epoch`), case ID and policy version makes transforms reproducible and
independent of model dropout. `run.json` records whether augmentation is enabled
and the full policy. Resume requires the same settings, including `--augment`.
Omitting the flag leaves augmentation off. The original test and twelve-epoch
CV used it off. The supplementary augmented comparison used it on in all three
folds: 22 training patients/128 scans and 11 validation patients/64 scans per fold,
with the original five test patients excluded. Each model completed four epochs
from scratch, then resumed in a separate run directory to a fixed epoch-8 endpoint.
Compare each endpoint with the same unaugmented fold at epoch 8, not epoch 4,
epoch 12 or a best-validation checkpoint; see the results in [README](../README.md).

`--epochs` is the **target total**, including previously completed epochs.
To extend an explicitly chosen development run from eight to twelve, repeat its exact original command with
`--resume --epochs 12`; this adds four epochs. The same rule applies to both trainers.
Keep all other settings, source files, manifest/split bytes, resolved paths and environment unchanged.
Resumption restores model, optimizer and RNG state at a completed epoch boundary, not midway through an epoch.
Preserve `run.json` and both referenced checkpoints. Do not extend the already frozen formal-test models.
The historical unaugmented CV continuation preserved the eight-epoch runs.
For the augmented four-to-eight continuation, keep `--augment` and use
`--resume --epochs 8` with the exact frozen source and settings; preserve the
parent run, checkpoints and first four history rows before extending a copied run.

Export the training/validation loss and per-class validation Dice curves:

```bash
python -s plot_history.py private/repro-3d/run.json --output private/repro-3d-learning.png
```

Only `run.json` is needed, not MRI volumes or weights. Scores describe the prepared grid;
the output file must be new unless `--overwrite` is explicitly supplied.

## Inspect validation predictions

Replace `VALIDATION_CASE_ID` with an actual ID from `private/split.json`'s `validation` list:

```bash
python -s predict.py \
  --run-dir private/repro-3d \
  --manifest private/manifest.json --split private/split.json \
  --case-ids VALIDATION_CASE_ID --output-dir private/prediction-example \
  --device cpu --threads 1 --repeats 1 \
  --terminal-regions --include-rectum-panels
```

The script detects the model architecture and loads the saved **best** checkpoint, not necessarily the latest epoch.
It accepts validation cases only. `--panel-case-ids` can select a subset of the requested cases for rendering.
Outputs include `prediction.json` and panels; full case metadata remain private.
Terminal regions mean first/last 20% of target-bearing source-K planes, not anatomically identified base/apex.
Selected report PNGs require identifier and attribution review before copying into public `figures/`.

## Freeze, rehearse on all validation cases, then evaluate test

Freeze binds source, input, checkpoint and runtime identities. It does not read image voxels.
Both model runs must have completed the same training budget; selected weights are their best validation checkpoints.
Use a new plan filename and a new results directory for each run. Paste the SHA-256 printed by each freeze command:

```bash
python -s train.py evaluate freeze \
  --run-2d private/repro-2d --run-3d private/repro-3d \
  --manifest private/manifest.json --split private/split.json \
  --partition validation --device cuda --threads 1 \
  --output-plan private/validation-plan.json
python -s train.py evaluate run \
  --plan private/validation-plan.json \
  --plan-sha256 PASTE_PRINTED_VALIDATION_PLAN_SHA256 \
  --output-dir private/validation-evaluation

python -s train.py evaluate freeze \
  --run-2d private/repro-2d --run-3d private/repro-3d \
  --manifest private/manifest.json --split private/split.json \
  --partition test --device cuda --threads 1 \
  --validation-report private/validation-evaluation/evaluation.json \
  --output-plan private/test-plan.json
python -s predict.py test \
  --plan private/test-plan.json \
  --plan-sha256 PASTE_PRINTED_TEST_PLAN_SHA256 \
  --output-dir private/test-evaluation
```

The test gate requires a successful full-validation report with matching setup identities.
`train.py evaluate` delegates to the same frozen-plan implementation as
`evaluate.py`; it does not train while testing. `predict.py test` only accepts
a test-partition plan and evaluates every case in it for both models. It prints
the resulting metrics and saves `evaluation.json` plus anonymous 2D/3D panels
for the first test case by default. Optional `--panel-case-ids` chooses which
test cases receive images; it never narrows the scored cohort. Test panels are
explicitly labelled as test, and validation panels keep their existing labels.
The original `predict.py --run-dir ...` command remains validation-only.
Validation reloaded metrics must have identical validity flags and per-case/class Dice/IoU differences at most 0.005.
This tolerance is distinct from the stricter metric-match flags retained in the report.
Evaluation uses float32, disables autocast and TF32, and enables deterministic algorithms.
The original training records did not capture TF32 flags, so exact training/evaluation arithmetic identity is not established.
Freeze/run preserve failed attempts and refuse to overwrite existing outputs. A new reproduction is a new experiment;
it does not replace the historical test results or authorize selecting models from test performance.

## Historical GPU benchmark

The reported benchmark used archived **epoch-four** runs, all 19 validation cases, and four fresh processes
in 2D/3D/3D/2D order on one A100. Each case had five warm-ups and ten timed repeats.
One block uses this CLI; substitute the archived run, both recorded digests and every validation ID in split order:

```bash
python -s benchmark_gpu.py \
  --run-dir /path/to/archived-epoch4-2d-run \
  --manifest private/manifest.json --split private/split.json \
  --run-json-sha256 RECORDED_RUN_JSON_SHA256 \
  --checkpoint-sha256 RECORDED_BEST_CHECKPOINT_SHA256 \
  --case-ids ALL_VALIDATION_IDS_IN_SPLIT_ORDER \
  --warmups 5 --repeats 10 --threads 1 \
  --output-dir private/benchmark-block-01
```

`ALL_VALIDATION_IDS_IN_SPLIT_ORDER` means a space-separated list, not a literal ID.
Repeat with the corresponding model/archive and a new output directory for each block.
The timing measures forward inference, excluding file loading, host/device transfer and postprocessing.
Running this against newer checkpoints produces a new benchmark, not a replay of the epoch-four measurement.

## Connected-component diagnostics

`component_analysis.py` operates on complete prepared label volumes, without
changing predictions. After validation inference using the matching archived
source and checkpoint, pass the CPU integer arrays to:

```python
from component_analysis import component_metrics

report_26 = component_metrics(prediction.numpy(), prepared.target.numpy(), connectivity=26)
report_6 = component_metrics(prediction.numpy(), prepared.target.numpy(), connectivity=6)
```

Here `prepared` and `prediction` are the prepared case and CPU label tensor from
`predict.py`'s inference functions. Each report includes every foreground class,
component size, same-class ground-truth overlap, bounding box and representative
depth. Counts use prepared-grid voxels; no size cutoff or component removal is
applied. A zero-overlap prediction component is wholly false positive for that
class, but this measure does not count false-positive extensions attached to a
component that overlaps the ground truth. Non-largest components are not
automatically wrong, and ground truth can itself have multiple components.

`component_panels.render_component_panels` accepts the prepared case, paired
2D/3D NumPy predictions and their 26-neighbour reports. It uses a common crop and
planes, highlights zero-overlap components, and returns selection/geometry metadata.
An illustrative satellite requires both an overlapping and a zero-overlap
prediction component of the same class. Prostate has priority over rectum; within
that class the largest eligible component is selected, breaking ties by 2D then
3D and component ID. This purposeful illustration does not estimate frequency.
The reported check reused original epoch-8 weights and all 19 original validation
scans on four CPU cores (2 min 23 s, no GPU). Its 38 confusion matrices matched
the archived CPU predictions exactly; historical CPU/CUDA discrepancies remain.
Only the five existing failure cases were eligible for figures. Across those
cases, prostate again has priority over rectum, followed by largest component,
ascending example number, 2D before 3D and component ID. The published PNG is an
unchanged server output. The private records retain all five-class component
sizes, both connectivities, per-patient summaries and file/checkpoint hashes.

## Supplemental three-fold CV and reproducibility boundary

The public trainers can train supplied folds; private preparation, submission, provenance and aggregation scripts
are not distributed in Git. Consequently the public checkout is not a one-command replay of the historical six jobs.
The protocol below defines the split and endpoint choices without distributing patient IDs or checkpoints.

Exclude all five original test patients first. The remaining original training plus validation cohort contains
192 scans from 33 patients. Fold assignment uses only scan counts, with this deterministic algorithm:

```python
from collections import Counter
import random
# development: manifest records after excluding every original test patient.
counts = Counter(row['patient_id'] for row in development)
rng = random.Random(3710)
order = sorted(counts)
rng.shuffle(order)
order.sort(key=lambda patient: -counts[patient])  # Stable shuffled ties.
priorities = list(range(3))
rng.shuffle(priorities)
rank = {fold: position for position, fold in enumerate(priorities)}
groups, scans = [[], [], []], [0, 0, 0]
for patient in order:
    fold = min((i for i in range(3) if len(groups[i]) < 11),
               key=lambda i: (scans[i], len(groups[i]), rank[i]))
    groups[fold].append(patient)
    scans[fold] += counts[patient]
```

Each held-out fold has 11 patients/64 scans; its training complement has 22 patients/128 scans.
Save a development-only manifest and three split files using the same nine-field schema:
`validation_patients=11`, `test_patients=0`, `test=[]`, and sorted case IDs for training/validation.
Each split hashes the development manifest. Use those files in both training commands, with separate output directories.
For the historical unaugmented comparison, train six models from scratch for eight
epochs with augmentation off, then resume each designated run to a total of twelve.

CV metrics use fixed epoch-eight and epoch-twelve history rows, **not** the best-validation checkpoint.
In particular, the third-fold 3D run's best epoch is eleven, so `predict.py`/`evaluate.py` do not directly reproduce
its fixed epoch-twelve CV score. Read the endpoint's saved per-case `validation` metrics in `run.json` instead.
Aggregate valid scans per patient first, then equally weight patients; retain undefined classes as null rather than zero.
Report all folds and all 33 out-of-fold patients. Fold standard deviations are descriptive, not independent-study uncertainty.
CV followed inspection of the original test results; continuation to twelve was chosen after viewing epoch eight.
These are exploratory split-sensitivity results, not a new untouched test or a basis for reevaluating the original test set.

The augmented comparison retained the completed first fold and then trained
2D/3D on folds 2 and 3 with the same frozen policy, seed, source and fold assignments.
These four models used two four-epoch stages each, with separate parent and resumed
run directories. All eight sequential single-GPU jobs completed on 23 October;
records were collected and verified on 24 October. Each stayed below 30 minutes
(2 h 52 min 05 s total, within the 240-minute cap). Aggregate all three fixed epoch-8
endpoints, retaining every fold. This expansion followed inspection of fold 1,
so it remains exploratory. The original test was not rerun.
The augmented fold-2 2D model's best checkpoint is epoch 7; reproduce its fixed
epoch-8 score from history row 8 or the saved epoch-8 report, not the best checkpoint.
