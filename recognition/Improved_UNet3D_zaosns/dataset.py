"""Explicit case manifests and patient-grouped splits for the HipMRI project.

This preparation layer validates file references and supplied patient IDs. It
does not yet read MRI voxels, infer patient identity, remap labels or resample.
Keep real manifests outside Git, for example in the ignored data/ directory.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Sequence


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
