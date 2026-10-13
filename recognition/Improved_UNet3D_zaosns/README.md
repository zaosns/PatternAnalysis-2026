# HipMRI Segmentation with a 3D Improved U-Net

**Status: project planning.** This branch currently contains a directory
placeholder, this initial plan and Git ignore rules. Model implementation, data
auditing, training and evaluation have not started. No experimental results or
completed feasibility check-off are claimed.

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

1. Confirm access to Rangpur GPU resources and the HipMRI data. The course-listed
   location is `/home/groups/comp3710/HipMRI_Study_open`; access has not been verified.
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

## Planned code structure

The following Python files are not yet included in this branch:

| File | Planned responsibility |
| --- | --- |
| `modules.py` | PyTorch model components |
| `dataset.py` | Loading, preprocessing, patient splits and augmentation |
| `train.py` | Training, validation, testing, checkpointing and learning curves |
| `predict.py` | Loading saved weights and visualizing test inference |

This README will develop into the final technical report. Training and inference
commands will be added once they exist and have been verified.

## Environment and reproducibility

The Python/PyTorch/CUDA environment and exact dependency versions have not yet
been selected or validated. Record tested versions, seeds, split-generation
instructions, hyperparameters, commands and checkpoint retrieval instructions
as the implementation is developed.

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
checking and configuring the GitHub fork, and drafting this plan and the ignore
rules. No model or data-processing implementation is included in this branch,
and no training or empirical model validation has been performed. The student
must review the plan and future implementation. This section will be updated
with the assistance received and the checks actually performed during development.

## References

- Supplied COMP3710/COMP7710 report specification, sections 2.2-6. The supplied
  filename is v2.3 while its cover says DRAFT/Version 2.2; confirm deadlines and
  any revised requirements against the course announcements.
- Isensee et al., [Brain Tumor Segmentation and Radiomics Survival Prediction:
  Contribution to the BRATS 2017 Challenge](https://arxiv.org/abs/1802.10508).
  Proposed architecture reference.
- Cicek et al., [3D U-Net: Learning Dense Volumetric Segmentation from Sparse
  Annotation](https://arxiv.org/abs/1606.06650). Proposed comparison reference.
