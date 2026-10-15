"""Explicit manifests, patient splits, native NIfTI loading and preprocessing.

Loading preserves voxel values and geometry; it does not infer patient identity,
remap labels, normalize intensities, reorient anatomy or resample volumes.
The separate preprocess_case function normalizes and downsamples a loaded case.
Keep real manifests outside Git, for example in the ignored data/ directory.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import TYPE_CHECKING, Iterator, Sequence

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class CaseRecord:
    """One scan with one image/label pair and an explicitly supplied patient ID."""

    case_id: str
    patient_id: str
    image_path: Path
    label_path: Path


@dataclass(frozen=True)
class PatientSplit:
    """Case tuples whose supplied patient IDs do not overlap between groups."""

    train: tuple[CaseRecord, ...]
    validation: tuple[CaseRecord, ...]
    test: tuple[CaseRecord, ...]


@dataclass(frozen=True)
class LoadedNiftiCase:
    """One CPU case, with tensors and affine matrices in explicitly named axes.

    image is float32 [1,D,H,W], target is int64 [D,H,W], and (D,H,W)=(K,J,I)
    for source NIfTI indices (I,J,K). This permutation is not anatomical
    reorientation. affine_ijk maps [i,j,k,1] to source world coordinates;
    affine_dhw maps [d,h,w,1] to the same coordinates. Both are float64.
    Spatial coordinates retain spatial_unit; no conversion to millimetres occurs.
    Frozen fields do not make the contained tensors immutable.
    """

    record: CaseRecord
    image: torch.Tensor
    target: torch.Tensor
    affine_ijk: torch.Tensor
    affine_dhw: torch.Tensor
    source_shape: tuple[int, int, int]
    image_spatial_unit: str
    label_spatial_unit: str
    spatial_unit: str
    label_unit_assumed: bool
    image_qform_code: int
    image_sform_code: int
    label_qform_code: int
    label_sform_code: int


@dataclass(frozen=True)
class PreprocessedCase:
    """A CPU case on its resized grid, retaining source geometry for restoration.

    image is float32 [1,D,H,W], target is int64 [D,H,W]. affine_dhw maps
    output voxel centres to world coordinates in spatial_unit. Source geometry
    describes the original tensor axes, not NIfTI (I,J,K). Normalization
    statistics describe only the original image's nonzero voxels. No original
    voxel arrays or LoadedNiftiCase reference are retained.
    """

    record: CaseRecord
    image: torch.Tensor
    target: torch.Tensor
    affine_dhw: torch.Tensor
    source_shape_dhw: tuple[int, int, int]
    source_affine_dhw: torch.Tensor
    spatial_unit: str
    label_unit_assumed: bool
    normalization_mean: float
    normalization_std: float
    normalization_scale: float
    nonzero_count: int
    downsample_factor: int


def _validate_cases(records: Sequence[CaseRecord]) -> tuple[CaseRecord, ...]:
    """Canonicalize paths and reject ambiguous IDs or reused physical files."""
    records = tuple(records)
    if not records:
        raise ValueError("At least one case is required.")
    case_ids = set()
    used_files = set()
    validated = []
    for record in records:
        if not isinstance(record, CaseRecord):
            raise ValueError("Each record must be a CaseRecord.")
        for field in ("case_id", "patient_id"):
            value = getattr(record, field)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"{field} must be a nonempty string without surrounding whitespace.")
        if record.case_id in case_ids:
            raise ValueError(f"Duplicate case_id: {record.case_id}")
        case_ids.add(record.case_id)
        paths = []
        for field in ("image_path", "label_path"):
            value = getattr(record, field)
            if not isinstance(value, (str, Path)) or not str(value).strip():
                raise ValueError(f"{field} must be a nonempty file path.")
            path = Path(value).resolve(strict=True)
            if not path.is_file():
                raise ValueError(f"{field} must refer to a regular file: {path}")
            stat = path.stat()
            identity = (stat.st_dev, stat.st_ino)
            if identity in used_files:
                raise ValueError(f"A file is reused as more than one image/label: {path}")
            used_files.add(identity)
            paths.append(path)
        validated.append(CaseRecord(record.case_id, record.patient_id, *paths))
    return tuple(validated)


def load_manifest(path: str | Path) -> tuple[CaseRecord, ...]:
    """Read a JSON list with exactly the four CaseRecord fields per entry.

    Relative file paths are resolved against the manifest's directory; absolute
    paths are also accepted. Returned paths are absolute. IDs are case-sensitive.
    This currently requires a one-to-one image/label pairing: file reuse, including
    symlink/hardlink aliases, is rejected even within the same patient.
    File contents and whether the supplied IDs represent real patients are not
    verified. Copies with different file identities are not detected as duplicates.
    """
    manifest = Path(path).resolve(strict=True)
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    fields = {"case_id", "patient_id", "image_path", "label_path"}
    if not isinstance(entries, list):
        raise ValueError("Manifest must be a JSON list of case records.")
    records = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != fields:
            raise ValueError(f"Manifest entry {index} must contain exactly {sorted(fields)}.")
        paths = []
        for field in ("image_path", "label_path"):
            value = entry[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Manifest entry {index}: {field} must be a nonempty string.")
            paths.append(manifest.parent / value)
        records.append(CaseRecord(entry["case_id"], entry["patient_id"], *paths))
    return _validate_cases(records)


def split_cases_by_patient(
    records: Sequence[CaseRecord],
    *,
    validation_patients: int,
    test_patients: int,
    seed: int = 3710,
) -> PatientSplit:
    """Partition complete patient groups using explicit held-out patient counts.

    Sort patient IDs before using a local seeded RNG, then sort each group's cases
    by case_id. Input order does not affect the result, and the global RNG is not
    changed. Every scan of one supplied patient stays in one partition. At least
    one training patient is required; zero validation/test counts are allowed.

    Counts are numbers of patients, not scans, and are not stratified by labels.
    The same manifest and seed reproduce the split in the same Python environment.
    Save actual assignments outside Git once a real-data split is chosen; adding
    or removing patients can change assignments even when the seed is unchanged.
    """
    for field, value in (("validation_patients", validation_patients), ("test_patients", test_patients)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{field} must be a nonnegative integer.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer.")
    cases = _validate_cases(records)
    patient_ids = sorted({case.patient_id for case in cases})
    held_out = validation_patients + test_patients
    if held_out >= len(patient_ids):
        raise ValueError("The split must leave at least one training patient.")
    random.Random(seed).shuffle(patient_ids)
    validation_ids = set(patient_ids[:validation_patients])
    test_ids = set(patient_ids[validation_patients:held_out])
    train_ids = set(patient_ids[held_out:])
    ordered = sorted(cases, key=lambda case: case.case_id)
    return PatientSplit(
        train=tuple(case for case in ordered if case.patient_id in train_ids),
        validation=tuple(case for case in ordered if case.patient_id in validation_ids),
        test=tuple(case for case in ordered if case.patient_id in test_ids),
    )


def load_patient_split(
    path: str | Path,
    records: Sequence[CaseRecord],
    *,
    manifest_sha256: str,
) -> PatientSplit:
    """Load frozen assignments, preserving order and never reshuffling patients.

    The caller supplies the SHA256 of the exact manifest bytes used to load
    records. The stored digest must match it; this function cannot reconstruct
    those bytes from CaseRecord objects. Every manifest case must appear exactly
    once, whole patient groups must stay together, and recorded holdout counts
    must agree. Training and validation are required; an empty test set is allowed.
    Files are checked for existence/identity, but no image headers or voxels are
    read. patient_key_rule is provenance text, not executable identity inference.
    """
    def valid_digest(value: object) -> bool:
        return (isinstance(value, str) and len(value) == 64
                and all(character in "0123456789abcdef" for character in value))

    if not valid_digest(manifest_sha256):
        raise ValueError("manifest_sha256 must be a lowercase SHA256 hexadecimal digest.")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    fields = {"format_version", "seed", "validation_patients", "test_patients",
              "patient_key_rule", "manifest_sha256", "train", "validation", "test"}
    if not isinstance(payload, dict) or set(payload) != fields:
        raise ValueError(f"Patient split must contain exactly {sorted(fields)}.")
    if type(payload["format_version"]) is not int or payload["format_version"] != 1:
        raise ValueError("Patient split format_version must be integer 1.")
    if isinstance(payload["seed"], bool) or not isinstance(payload["seed"], int):
        raise ValueError("Patient split seed must be an integer.")
    for field in ("validation_patients", "test_patients"):
        value = payload[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Patient split {field} must be a nonnegative integer.")
    rule = payload["patient_key_rule"]
    if not isinstance(rule, str) or not rule.strip() or rule != rule.strip():
        raise ValueError("patient_key_rule must be nonempty provenance text without surrounding whitespace.")
    if not valid_digest(payload["manifest_sha256"]):
        raise ValueError("Stored manifest_sha256 must be a lowercase SHA256 hexadecimal digest.")
    if payload["manifest_sha256"] != manifest_sha256:
        raise ValueError("Patient split manifest_sha256 does not match the supplied manifest.")

    cases = _validate_cases(records)
    by_id = {record.case_id: record for record in cases}
    assigned: set[str] = set()
    patient_groups: dict[str, str] = {}
    groups: dict[str, tuple[CaseRecord, ...]] = {}
    for name in ("train", "validation", "test"):
        identifiers = payload[name]
        if not isinstance(identifiers, list) or any(
            not isinstance(case_id, str) or not case_id or case_id != case_id.strip()
            for case_id in identifiers
        ):
            raise ValueError(f"Patient split {name} must be a list of nonempty case ID strings.")
        if name != "test" and not identifiers:
            raise ValueError(f"Patient split {name} must not be empty.")
        group = []
        for case_id in identifiers:
            if case_id not in by_id:
                raise ValueError(f"Unknown case ID in patient split {name}: {case_id}")
            if case_id in assigned:
                raise ValueError(f"Duplicate case ID in patient split: {case_id}")
            assigned.add(case_id)
            record = by_id[case_id]
            previous_group = patient_groups.setdefault(record.patient_id, name)
            if previous_group != name:
                raise ValueError(f"Patient {record.patient_id!r} appears in both {previous_group} and {name}.")
            group.append(record)
        groups[name] = tuple(group)
    missing = set(by_id) - assigned
    if missing:
        raise ValueError(f"Patient split is missing case IDs: {sorted(missing)}")
    for name in ("validation", "test"):
        actual_count = len({record.patient_id for record in groups[name]})
        if actual_count != payload[f"{name}_patients"]:
            raise ValueError(f"Patient split {name}_patients does not match its assignments.")
    return PatientSplit(**groups)


def iter_case_batches(
    records: Sequence[CaseRecord],
    *,
    downsample_factor: int = 2,
    assume_label_units_from_image: bool = False,
    num_classes: int = 6,
    seed: int | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield CPU batches of one preprocessed volume, with no workers or cache.

    Each batch is (float32 [1,1,D,H,W], int64 [1,D,H,W]); successive cases may
    have different spatial sizes. Only the provided records are read, so callers
    must pass the training or validation tuple, never concatenate held-out data.
    None preserves their order; an integer seed shuffles via an isolated RNG.
    Validation and I/O happen on iteration. Original volumes are released before
    yielding, and the generator releases its batch before loading the next case.
    A consumer may of course retain any yielded tensors itself.
    """
    if isinstance(downsample_factor, bool) or not isinstance(downsample_factor, int) or downsample_factor <= 0:
        raise ValueError("downsample_factor must be a positive integer.")
    if not isinstance(assume_label_units_from_image, bool):
        raise ValueError("assume_label_units_from_image must be a boolean.")
    if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError("num_classes must be an integer >= 2.")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise ValueError("seed must be an integer or None.")
    cases = list(records)
    if not cases or any(not isinstance(record, CaseRecord) for record in cases):
        raise ValueError("records must contain at least one CaseRecord.")
    if seed is not None:
        random.Random(seed).shuffle(cases)
    for record in cases:
        try:
            loaded = load_nifti_case(
                record, allowed_labels=range(num_classes),
                assume_label_units_from_image=assume_label_units_from_image,
            )
            prepared = preprocess_case(loaded, downsample_factor=downsample_factor)
            del loaded
        except Exception as error:
            raise RuntimeError(f"Case {record.case_id!r} could not be loaded or preprocessed: {error}") from error
        yield prepared.image.unsqueeze(0), prepared.target.unsqueeze(0)
        del prepared


def load_nifti_case(
    record: CaseRecord,
    *,
    allowed_labels: Sequence[int],
    assume_label_units_from_image: bool = False,
    max_voxels: int = 64_000_000,
) -> LoadedNiftiCase:
    """Read one aligned 3D NIfTI pair without preprocessing or label remapping.

    allowed_labels is an explicit vocabulary of nonnegative int64 class values;
    a case may contain only a subset. Values are checked after NIfTI intensity
    scaling and before conversion to int64. Image values include that scaling
    and are converted to float32 without normalization.

    Both files need a coded spatial transform. NiBabel's effective affine is
    used (sform before qform); shape and affine must match. Unknown image units
    are always rejected. Unknown label units require the explicit assumption
    switch and are then taken from the paired image, with this decision recorded.
    Matching headers cannot establish anatomical correctness of the labels.

    max_voxels limits each volume's header-declared size before reading voxels.
    This is not a peak-memory guarantee: conversion and validation make copies.
    NumPy, NiBabel and PyTorch are imported only when this reader is called.
    """
    if isinstance(max_voxels, bool) or not isinstance(max_voxels, int) or max_voxels <= 0:
        raise ValueError("max_voxels must be a positive integer.")
    if not isinstance(assume_label_units_from_image, bool):
        raise ValueError("assume_label_units_from_image must be a boolean.")
    try:
        values = tuple(allowed_labels)
    except TypeError as error:
        raise ValueError("allowed_labels must be a nonempty sequence of integers.") from error
    if not values or any(
        isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**63
        for value in values
    ):
        raise ValueError("allowed_labels must contain nonnegative int64 integers.")
    allowed = frozenset(values)
    case = _validate_cases((record,))[0]

    import nibabel as nib
    import numpy as np
    import torch

    files = [nib.load(str(case.image_path)), nib.load(str(case.label_path))]
    geometry = []
    for role, volume in zip(("image", "label"), files):
        if not isinstance(volume, (nib.Nifti1Image, nib.Nifti2Image)):
            raise ValueError(f"{role} must be a single-file NIfTI image.")
        shape = tuple(int(size) for size in volume.shape)
        if len(shape) != 3 or any(size <= 0 for size in shape):
            raise ValueError(f"{role} must have a nonempty 3D shape.")
        if math.prod(shape) > max_voxels:
            raise ValueError(f"{role} exceeds max_voxels={max_voxels}.")
        if volume.get_data_dtype().kind not in "iuf":
            raise ValueError(f"{role} must contain real integer or floating-point values.")
        qform_code = int(volume.header["qform_code"])
        sform_code = int(volume.header["sform_code"])
        if qform_code <= 0 and sform_code <= 0:
            raise ValueError(f"{role} needs a coded qform or sform; fallback geometry is not accepted.")
        affine = np.array(volume.affine, dtype=np.float64, copy=True)
        if (affine.shape != (4, 4) or not np.isfinite(affine).all()
                or not np.array_equal(affine[3], [0.0, 0.0, 0.0, 1.0])):
            raise ValueError(f"{role} affine must be a finite homogeneous 4x4 transform.")
        sign, log_determinant = np.linalg.slogdet(affine[:3, :3])
        if sign == 0 or not np.isfinite(log_determinant):
            raise ValueError(f"{role} affine must be nonsingular.")
        spatial_unit = volume.header.get_xyzt_units()[0]
        geometry.append((shape, affine, spatial_unit, qform_code, sform_code))

    image_shape, image_affine, image_unit, image_qform, image_sform = geometry[0]
    label_shape, label_affine, label_unit, label_qform, label_sform = geometry[1]
    if image_shape != label_shape:
        raise ValueError("Image and label shapes must match.")
    if not np.allclose(image_affine, label_affine, rtol=0.0, atol=1e-5):
        raise ValueError("Image and label affines must match on their native grid.")
    if image_unit not in ("mm", "meter", "micron"):
        raise ValueError("Image spatial units must be known.")
    label_unit_assumed = label_unit == "unknown" and assume_label_units_from_image
    if label_unit != image_unit and not label_unit_assumed:
        raise ValueError("Label spatial units must match the image; unknown label units need an explicit assumption.")

    label_values = np.asanyarray(files[1].dataobj)
    if not np.isfinite(label_values).all():
        raise ValueError("Label values must be finite.")
    unique_labels = np.unique(label_values)
    if label_values.dtype.kind == "f" and np.any(unique_labels != np.floor(unique_labels)):
        raise ValueError("Scaled label values must be integers.")
    # Python ints preserve uint64 values and avoid rounding int64's upper bound
    # during mixed float/integer comparisons before the eventual conversion.
    if any(int(value) not in allowed for value in unique_labels):
        raise ValueError("Label values are outside allowed_labels.")
    target_array = np.array(label_values.transpose(2, 1, 0), dtype=np.int64, order="C", copy=True)
    del label_values, unique_labels

    image_values = np.asanyarray(files[0].dataobj)
    with np.errstate(over="ignore", invalid="ignore"):
        image_array = np.array(image_values.transpose(2, 1, 0), dtype=np.float32, order="C", copy=True)
    del image_values
    if not np.isfinite(image_array).all():
        raise ValueError("Image values must remain finite after float32 conversion.")
    permutation = np.eye(4, dtype=np.float64)[:, [2, 1, 0, 3]]
    return LoadedNiftiCase(
        record=case,
        image=torch.from_numpy(image_array).unsqueeze(0),
        target=torch.from_numpy(target_array),
        affine_ijk=torch.from_numpy(image_affine),
        affine_dhw=torch.from_numpy(image_affine @ permutation),
        source_shape=image_shape,
        image_spatial_unit=image_unit,
        label_spatial_unit=label_unit,
        spatial_unit=image_unit,
        label_unit_assumed=label_unit_assumed,
        image_qform_code=image_qform,
        image_sform_code=image_sform,
        label_qform_code=label_qform,
        label_sform_code=label_sform,
    )


def preprocess_case(
    sample: LoadedNiftiCase,
    *,
    downsample_factor: int = 2,
) -> PreprocessedCase:
    """Normalize one native case and resize both volumes on a shared voxel grid.

    The image's nonzero voxels alone define the normalization mask; targets are
    never used to estimate intensity statistics. Population mean/std are computed
    in float64, with scale=max(std, 1e-8). Original zeros stay zero during
    normalization. An all-zero image is rejected; a constant nonzero region
    normalizes to zero. Trilinear resizing can mix zero/nonzero boundaries, so
    output background is not guaranteed to stay zero after downsampling.

    Each output axis has ceil(source_size/downsample_factor) voxels. Image
    interpolation uses align_corners=False; integer labels use nearest sampling
    on the same half-pixel grid, choosing the higher index at an exact tie.
    Labels never pass through floating point. The affine preserves the outer
    voxel edges, including for odd source sizes. There is no antialias filter,
    anatomical reorientation, crop, augmentation or label remapping.

    Inputs must be CPU tensors with the loader's dtypes and consistent metadata.
    Returned tensors are independent of input storage and have no autograd graph,
    even when factor=1. Preprocessing uses no other cases or split-wide statistics.
    """
    import torch
    import torch.nn.functional as functional

    if isinstance(downsample_factor, bool) or not isinstance(downsample_factor, int) or downsample_factor <= 0:
        raise ValueError("downsample_factor must be a positive integer.")
    if not isinstance(sample, LoadedNiftiCase) or not isinstance(sample.record, CaseRecord):
        raise ValueError("sample must be a LoadedNiftiCase with a CaseRecord.")
    image, target = sample.image, sample.target
    for name, value, dtype, rank in (("image", image, torch.float32, 4), ("target", target, torch.int64, 3)):
        if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                or value.layout != torch.strided or value.dtype != dtype or value.ndim != rank):
            raise ValueError(f"{name} must be a dense CPU {dtype} tensor with {rank} dimensions.")
    if image.shape[0] != 1 or tuple(image.shape[1:]) != tuple(target.shape) or any(size <= 0 for size in target.shape):
        raise ValueError("image [1,D,H,W] and target [D,H,W] must have matching nonempty shapes.")
    if not torch.isfinite(image).all():
        raise ValueError("Image values must be finite.")
    if torch.any(target < 0):
        raise ValueError("Target class values must be nonnegative.")
    shape = tuple(int(size) for size in target.shape)
    if (not isinstance(sample.source_shape, tuple) or len(sample.source_shape) != 3
            or any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in sample.source_shape)
            or sample.source_shape != shape[::-1]):
        raise ValueError("source_shape must agree with the original NIfTI (I,J,K) axes.")
    for name in ("affine_ijk", "affine_dhw"):
        affine = getattr(sample, name)
        if (not isinstance(affine, torch.Tensor) or affine.device.type != "cpu"
                or affine.layout != torch.strided or affine.dtype != torch.float64
                or affine.shape != (4, 4) or not torch.isfinite(affine).all()
                or not torch.equal(affine[3], torch.tensor([0., 0., 0., 1.], dtype=torch.float64))):
            raise ValueError(f"{name} must be a finite homogeneous CPU float64 4x4 transform.")
        sign, log_determinant = torch.linalg.slogdet(affine[:3, :3])
        if sign == 0 or not torch.isfinite(log_determinant):
            raise ValueError(f"{name} must be nonsingular.")
    if not torch.allclose(sample.affine_dhw, sample.affine_ijk[:, [2, 1, 0, 3]], rtol=0., atol=1e-5):
        raise ValueError("The IJK and DHW affines must describe the same source grid.")
    if sample.spatial_unit not in ("mm", "meter", "micron") or sample.image_spatial_unit != sample.spatial_unit:
        raise ValueError("Image and effective spatial units must be known and equal.")
    if not isinstance(sample.label_unit_assumed, bool):
        raise ValueError("label_unit_assumed must be a boolean.")
    expected_label_unit = "unknown" if sample.label_unit_assumed else sample.spatial_unit
    if sample.label_spatial_unit != expected_label_unit:
        raise ValueError("Label spatial unit and recorded unit assumption are inconsistent.")
    for role in ("image", "label"):
        codes = (getattr(sample, f"{role}_qform_code"), getattr(sample, f"{role}_sform_code"))
        if any(isinstance(code, bool) or not isinstance(code, int) or code < 0 for code in codes) or not any(codes):
            raise ValueError(f"{role} must retain a valid coded spatial transform.")

    with torch.no_grad():
        mask = image != 0
        values = image[mask].to(torch.float64)
        nonzero_count = values.numel()
        if nonzero_count == 0:
            raise ValueError("An all-zero image cannot be normalized.")
        mean = values.mean().item()
        std = values.std(unbiased=False).item()
        scale = max(std, 1e-8)
        normalized = torch.zeros_like(image)
        normalized[mask] = ((values - mean) / scale).to(torch.float32)
        del values, mask
        output_shape = tuple((size + downsample_factor - 1) // downsample_factor for size in shape)
        if downsample_factor == 1:
            resized_image = normalized
            resized_target = target.clone()
        else:
            resized_image = functional.interpolate(
                normalized.unsqueeze(0), size=output_shape, mode="trilinear", align_corners=False,
            ).squeeze(0)
            resized_target = target
            for axis, (source_size, output_size) in enumerate(zip(shape, output_shape)):
                # Integer arithmetic preserves large class IDs and exact tie
                # decisions; no floating-point coordinate rounding is involved.
                indices = torch.tensor([
                    ((2 * index + 1) * source_size) // (2 * output_size)
                    for index in range(output_size)
                ], dtype=torch.int64)
                resized_target = resized_target.index_select(axis, indices)
        transform = torch.eye(4, dtype=torch.float64)
        for axis, (source_size, output_size) in enumerate(zip(shape, output_shape)):
            ratio = source_size / output_size
            transform[axis, axis] = ratio
            transform[axis, 3] = (ratio - 1.) / 2.
        return PreprocessedCase(
            record=sample.record,
            image=resized_image,
            target=resized_target,
            affine_dhw=sample.affine_dhw @ transform,
            source_shape_dhw=shape,
            source_affine_dhw=sample.affine_dhw.detach().clone(),
            spatial_unit=sample.spatial_unit,
            label_unit_assumed=sample.label_unit_assumed,
            normalization_mean=mean,
            normalization_std=std,
            normalization_scale=scale,
            nonzero_count=nonzero_count,
            downsample_factor=downsample_factor,
        )
