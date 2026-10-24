"""Single-epoch helpers and a complete-case, multi-epoch training command.

These functions connect an iterable of (image, target) tensor batches to the
existing model and objectives. Epoch-boundary persistence is in checkpoint.py.
The command uses a frozen patient split, batch size one and validation-loss
model selection. Training never evaluates test volumes. The explicit
``evaluate`` command delegates validation/test work to the frozen-plan evaluator.
Synthetic tests do not establish MRI accuracy.
"""

import argparse
from collections.abc import Iterable
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import sys
import tempfile
import time
import warnings

import torch
from torch import Tensor, nn

from checkpoint import load_checkpoint, save_checkpoint
from dataset import (iter_case_batches, load_manifest, load_patient_split,
                     training_augmentation_policy)
from modules import SegmentationModel3D
from objectives import SoftDiceLoss, dice_iou_per_class


def _prepare_batch(batch, device: str | torch.device) -> tuple[Tensor, Tensor]:
    """Move a nonempty image/label pair without silently converting its dtype."""
    if not isinstance(batch, (tuple, list)) or len(batch) != 2:
        raise ValueError("Each batch must be an (image, target) pair.")
    image, target = batch
    if not isinstance(image, Tensor) or not isinstance(target, Tensor):
        raise ValueError("image and target must be tensors.")
    if image.ndim != 5 or any(size <= 0 for size in image.shape):
        raise ValueError("image must have nonempty shape [N, C, D, H, W].")
    if not image.is_floating_point() or not torch.isfinite(image).all():
        raise ValueError("image must contain finite floating-point values.")
    if target.dtype != torch.long or target.shape != (image.shape[0], *image.shape[2:]):
        raise ValueError("target must be torch.long [N, D, H, W] matching image.")
    return image.to(device), target.to(device)


def _forward_loss(model, image, target, criterion, num_classes=None) -> tuple[Tensor, Tensor]:
    """Enforce the logits and scalar mean-loss contract before an update."""
    logits = model(image)
    if not isinstance(logits, Tensor) or logits.ndim != 5:
        raise ValueError("model must return logits [N, C, D, H, W].")
    if (logits.shape[1] < 2 or target.shape != (logits.shape[0], *logits.shape[2:])
            or (num_classes is not None and logits.shape[1] != num_classes)):
        raise ValueError("logits must match the target shape and fixed num_classes >= 2.")
    if not logits.is_floating_point() or not torch.isfinite(logits).all():
        raise ValueError("logits must contain finite floating-point values.")
    if target.min() < 0 or target.max() >= logits.shape[1]:
        raise ValueError("target class indices must be in [0, num_classes - 1].")
    loss = criterion(logits, target)
    if (not isinstance(loss, Tensor) or loss.ndim != 0
            or not loss.is_floating_point() or not torch.isfinite(loss)):
        raise ValueError("criterion must return a finite floating-point scalar mean loss.")
    return logits, loss


def train_one_epoch(
    model: nn.Module,
    batches: Iterable[tuple[Tensor, Tensor]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, float | int]:
    """Train once over batches; return sample-weighted mean loss and sample count.

    Move the model to device before constructing its optimizer. This function
    moves only batches and leaves the model in training mode. Each batch gets one
    optimizer step, with fresh gradients. criterion must average equally over
    samples, as SoftDiceLoss does; sum reduction is not supported. Shapes/dtypes
    are checked, but the caller is responsible for this loss-reduction meaning.

    Nonfinite loss/gradients stop the current step before optimizer.step(). Earlier
    successful steps are not rolled back. This initial loop has no mixed precision,
    gradient accumulation or scheduler. Call checkpoint.py helpers between epochs
    to save or restore progress; this loop does not write checkpoints itself.
    """
    model.train()
    loss_sum, sample_count = 0.0, 0
    num_classes = None
    for batch in batches:
        image, target = _prepare_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits, loss = _forward_loss(model, image, target, criterion, num_classes)
        num_classes = logits.shape[1]
        loss.backward()
        if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
               for parameter in model.parameters()):
            optimizer.zero_grad(set_to_none=True)
            raise ValueError("Nonfinite gradients; this optimizer step was skipped.")
        optimizer.step()
        loss_sum += loss.detach().item() * image.shape[0]
        sample_count += image.shape[0]
    if sample_count == 0:
        raise ValueError("Training requires at least one nonempty batch.")
    return {"loss": loss_sum / sample_count, "num_samples": sample_count}


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    batches: Iterable[tuple[Tensor, Tensor]],
    criterion: nn.Module,
    *,
    num_classes: int,
    include_background: bool = True,
    device: str | torch.device = "cpu",
) -> dict[str, Tensor | float | int]:
    """Validate complete case volumes, preserving every per-case metric row.

    The caller supplies one complete case volume per sample, after reconstructing
    any patches. This cannot be verified from tensor shape. Differently sized
    cases can be processed in separate batches; do not pad labels without a
    corresponding validity-mask policy (not implemented here).

    The model is left in evaluation mode; no gradients or optimizer updates are
    performed. The device and scalar mean-loss contract match train_one_epoch.
    dice/iou/valid retain [N, K] rows in iteration order; class_ids identifies K.
    All returned tensors are on CPU. mean_dice/mean_iou average valid cases per
    class, including zero-score false positives. An entirely absent class stays
    NaN with valid_count=0. Means weight cases equally, not patients or batches.
    Keep case IDs in this same order when attaching identities for later reports.
    """
    if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError("num_classes must be an integer >= 2.")
    if not isinstance(include_background, bool):
        raise ValueError("include_background must be a bool.")
    model.eval()
    loss_sum, sample_count = 0.0, 0
    rows = {key: [] for key in ("dice", "iou", "valid")}
    for batch in batches:
        image, target = _prepare_batch(batch, device)
        logits, loss = _forward_loss(model, image, target, criterion, num_classes)
        metrics = dice_iou_per_class(logits.argmax(dim=1), target, num_classes, include_background)
        for key in rows:
            rows[key].append(metrics[key].cpu())
        loss_sum += loss.item() * image.shape[0]
        sample_count += image.shape[0]
    if sample_count == 0:
        raise ValueError("Validation requires at least one nonempty batch.")
    result = {key: torch.cat(values, dim=0) for key, values in rows.items()}
    counts = result["valid"].sum(dim=0)
    for key in ("dice", "iou"):
        total = result[key].masked_fill(~result["valid"], 0).sum(dim=0, dtype=torch.float64)
        result[f"mean_{key}"] = (total / counts.clamp_min(1)).masked_fill(counts == 0, float("nan"))
    result.update(
        loss=loss_sum / sample_count,
        num_samples=sample_count,
        valid_count=counts,
        class_ids=torch.arange(0 if include_background else 1, num_classes),
    )
    return result


class CrossEntropyDiceLoss(nn.Module):
    """Equal-weight voxel cross entropy plus mean foreground soft Dice loss.

    Index zero is background. HipMRI foreground here includes Body (class 1)
    and classes 2-5. This is an explicit starting objective, not a tuned choice.
    Cross entropy averages unreduced voxel losses explicitly so strict CUDA
    determinism does not require the atomic reduction of spatial NLL loss.
    """

    def __init__(self) -> None:
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(reduction="none")
        self.dice = SoftDiceLoss(include_background=False)

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        return self.cross_entropy(logits, target).mean() + self.dice(logits, target)


def _atomic_json(path: Path, value: dict) -> None:
    """Commit state only after the whole strict-JSON document is on disk."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _json_metrics(value):
    """Represent undefined overlap scores as JSON null, retaining valid masks."""
    if isinstance(value, Tensor):
        return _json_metrics(value.tolist())
    if isinstance(value, dict):
        return {key: _json_metrics(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_metrics(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if not math.isfinite(value):
            raise ValueError("Infinite metric cannot be written to the run log.")
    return value


def _checkpoint_name(epoch: int) -> str:
    return f"checkpoint-{epoch:06d}.pt"


def _validate_run_state(state: dict, config: dict, output: Path) -> None:
    """Require a coherent committed history; never infer progress from filenames."""
    fields = {"format_version", "config", "completed_epochs", "best_epoch",
              "latest_checkpoint", "best_checkpoint", "history"}
    if (not isinstance(state, dict) or set(state) != fields
            or type(state["format_version"]) is not int or state["format_version"] != 1):
        raise ValueError("Unsupported run.json format.")
    if state["config"] != config:
        raise ValueError("Run configuration, source code, environment or data hashes changed; use a new output directory.")
    completed, history = state["completed_epochs"], state["history"]
    if (type(completed) is not int or completed < 0 or not isinstance(history, list)
            or len(history) != completed):
        raise ValueError("Run history does not match completed_epochs.")
    for epoch, row in enumerate(history, 1):
        if (not isinstance(row, dict) or type(row.get("epoch")) is not int
                or row["epoch"] != epoch):
            raise ValueError("Run history epochs must be consecutive, starting at one.")
        for key in ("train_loss", "validation_loss"):
            value = row.get(key)
            if (isinstance(value, bool) or not isinstance(value, (float, int))
                    or not math.isfinite(value)):
                raise ValueError("Run history must have finite training and validation losses.")
    if not history:
        if any(state[key] is not None for key in ("best_epoch", "latest_checkpoint", "best_checkpoint")):
            raise ValueError("An untrained run must not reference checkpoints.")
        return
    best = min(history, key=lambda row: row["validation_loss"])["epoch"]
    if (type(state["best_epoch"]) is not int or state["best_epoch"] != best
            or state["latest_checkpoint"] != _checkpoint_name(completed)
            or state["best_checkpoint"] != _checkpoint_name(best)):
        raise ValueError("Checkpoint references disagree with run history.")
    for key in ("latest_checkpoint", "best_checkpoint"):
        path = output / state[key]
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing or invalid committed checkpoint: {state[key]}")


def _write_history_csv(output: Path, state: dict) -> None:
    """Rebuild a compact plotting table from the committed authoritative JSON."""
    class_ids = list(range(1, state["config"]["model"]["num_classes"]))
    columns = ["epoch", "train_loss", "validation_loss", "train_samples",
               "validation_samples", "elapsed_seconds", "peak_cuda_allocated_mib"]
    columns += [f"validation_{metric}_class_{index}"
                for metric in ("dice", "iou") for index in class_ids]
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                         dir=output, prefix=".history.", delete=False) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for row in state["history"]:
                flattened = {key: row[key] for key in columns[:7]}
                for metric in ("dice", "iou"):
                    flattened.update({f"validation_{metric}_class_{index}": value
                                      for index, value in zip(class_ids, row["validation"][f"mean_{metric}"])})
                writer.writerow(flattened)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output / "history.csv")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _refresh_derived_files(output: Path, state: dict) -> None:
    # A failed export/cleanup must not make a committed epoch look uncommitted.
    try:
        _write_history_csv(output, state)
    except OSError as error:
        warnings.warn(f"Epoch state is saved, but history.csv could not be refreshed: {error}")
    keep = {state["latest_checkpoint"], state["best_checkpoint"]}
    for path in output.glob("checkpoint-*.pt"):
        if re.fullmatch(r"checkpoint-\d{6,}\.pt", path.name) and path.name not in keep:
            try:
                path.unlink()
            except OSError as error:
                warnings.warn(f"Could not remove an unreferenced checkpoint {path.name}: {error}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the 3D model on a frozen patient split (batch size 1).",
        epilog="For explicit validation/test evaluation: train.py evaluate freeze --help or train.py evaluate run --help.")
    parser.add_argument("--manifest", type=Path, required=True, help="Private case manifest JSON.")
    parser.add_argument("--split", type=Path, required=True, help="Frozen split JSON with the manifest SHA-256.")
    parser.add_argument("--output-dir", type=Path, required=True, help="New run directory, or an existing one with --resume.")
    parser.add_argument("--epochs", type=int, required=True, help="Target total completed epochs, including earlier epochs on resume.")
    parser.add_argument("--resume", action="store_true", help="Continue from the latest epoch referenced by run.json.")
    parser.add_argument("--augment", action="store_true",
                        help="Enable paired width flips and small MRI intensity scaling for training only.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda", help="Explicit device; no silent CPU fallback.")
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--channels", nargs="+", type=int, default=[16, 32, 64, 128])
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--downsample-factor", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=3710)
    parser.add_argument("--deterministic", action="store_true",
                        help="Require deterministic PyTorch algorithms; unsupported operations fail.")
    parser.add_argument("--threads", type=int, default=1, help="PyTorch CPU threads; data loading has no workers.")
    parser.add_argument("--assume-label-units-from-image", action="store_true",
                        help="Explicitly adopt image units for unknown label units after geometry checks.")
    return parser


def run_training(args: argparse.Namespace) -> dict:
    """Run or resume complete epochs; return committed run.json state.

    Only the requested total epoch count may change on resume. Files are kept
    private in the selected directory. Linux/macOS advisory locking prevents
    simultaneous writers. Latest and best refer to named checkpoint files in
    run.json, rather than independently updated best.pt/latest.pt aliases.
    """
    for key in ("epochs", "num_classes", "downsample_factor", "threads"):
        value = getattr(args, key)
        if type(value) is not int or value < 1:
            raise ValueError(f"{key} must be a positive integer.")
    if args.num_classes < 2:
        raise ValueError("num_classes must be at least two.")
    if type(args.seed) is not int or not 0 <= args.seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63).")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive.")
    if args.device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda.")
    if not isinstance(args.deterministic, bool):
        raise ValueError("deterministic must be a bool.")
    if type(args.augment) is not bool:
        raise ValueError("augment must be a bool.")
    # Configure before CUDA queries, seeding or model construction. Respect an
    # explicitly supplied cuBLAS workspace policy and record its actual value.
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(args.deterministic, warn_only=False)
    torch.backends.cudnn.deterministic = args.deterministic
    torch.backends.cudnn.benchmark = False
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; request a GPU or explicitly select --device cpu.")

    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    split_hash = hashlib.sha256(args.split.read_bytes()).hexdigest()
    cases = load_manifest(args.manifest)
    split = load_patient_split(args.split, cases, manifest_sha256=manifest_hash)
    # Identical manifest bytes can resolve to different data after a directory
    # move when the manifest contains relative paths. Bind resolved inputs too.
    resolved_records = [dict(case_id=case.case_id, patient_id=case.patient_id,
                             image_path=str(case.image_path), label_path=str(case.label_path))
                        for case in cases]
    resolved_hash = hashlib.sha256(json.dumps(resolved_records, sort_keys=True).encode("utf-8")).hexdigest()
    # Augmentation uses an isolated per-case RNG, separate from model dropout.
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = SegmentationModel3D(num_classes=args.num_classes, channels=tuple(args.channels),
                                dropout=args.dropout, deep_supervision=True).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = CrossEntropyDiceLoss()
    source = Path(__file__).resolve().parent
    config = _json_metrics({
        "manifest_sha256": manifest_hash, "split_sha256": split_hash,
        "resolved_records_sha256": resolved_hash,
        "model": model.get_config(), "optimizer": "Adam", "learning_rate": args.learning_rate,
        "loss": {"cross_entropy_weight": 1.0, "soft_dice_weight": 1.0,
                 "cross_entropy_reduction": "none_then_voxel_mean",
                 "dice_include_background": False, "dice_smooth": 1e-5},
        "selection": "minimum_validation_loss_first_tie", "batch_size": 1,
        "downsample_factor": args.downsample_factor,
        "assume_label_units_from_image": args.assume_label_units_from_image,
        "seed": args.seed, "shuffle": "local_Random(seed + one_based_epoch)",
        "augmentation": {"enabled": args.augment,
                         "policy": training_augmentation_policy() if args.augment else None},
        "device": args.device, "threads": args.threads,
        "reproducibility": {
            "deterministic_requested": args.deterministic,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
        "source_sha256": {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                          for name in ("train.py", "dataset.py", "modules.py", "objectives.py", "checkpoint.py")},
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "numpy": importlib.metadata.version("numpy"),
                        "nibabel": importlib.metadata.version("nibabel"),
                        "cuda_runtime": torch.version.cuda,
                        "device_name": torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu"},
    })
    output = args.output_dir.resolve()
    if args.resume:
        if not output.is_dir():
            raise ValueError("Resume requires an existing run directory.")
    else:
        # Never overwrite an earlier run, even if its directory happens to be empty.
        output.mkdir(parents=True, exist_ok=False)
    with (output / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another process is already writing this run directory.") from error
        if args.resume:
            def reject_constant(value):
                raise ValueError(f"Nonstandard JSON constant in run state: {value}")
            state = json.loads((output / "run.json").read_text(encoding="utf-8"),
                               parse_constant=reject_constant)
            _validate_run_state(state, config, output)
            if args.epochs < state["completed_epochs"]:
                raise ValueError("epochs cannot be less than the number already completed.")
            if state["completed_epochs"]:
                progress = load_checkpoint(output / state["latest_checkpoint"], model, optimizer)
                best_loss = state["history"][state["best_epoch"] - 1]["validation_loss"]
                if (progress["completed_epochs"] != state["completed_epochs"]
                        or progress["best_validation_loss"] != best_loss):
                    raise ValueError("Checkpoint progress does not match run.json.")
        else:
            state = dict(format_version=1, config=config, completed_epochs=0, best_epoch=None,
                         latest_checkpoint=None, best_checkpoint=None, history=[])
            _atomic_json(output / "run.json", state)
        _refresh_derived_files(output, state)
        batch_options = dict(downsample_factor=args.downsample_factor,
                             assume_label_units_from_image=args.assume_label_units_from_image,
                             num_classes=args.num_classes)
        for epoch in range(state["completed_epochs"] + 1, args.epochs + 1):
            if args.device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            trained = train_one_epoch(
                model, iter_case_batches(split.train, seed=args.seed + epoch,
                                         training=True, augment=args.augment, **batch_options),
                optimizer, criterion, device=args.device)
            validated = validate_one_epoch(
                model, iter_case_batches(split.validation, **batch_options), criterion,
                num_classes=args.num_classes, include_background=False, device=args.device)
            if args.device == "cuda":
                torch.cuda.synchronize()
            row = dict(epoch=epoch, train_loss=trained["loss"], validation_loss=validated["loss"],
                       train_samples=trained["num_samples"], validation_samples=validated["num_samples"],
                       validation=_json_metrics({"case_ids": [case.case_id for case in split.validation],
                                                 **{key: value for key, value in validated.items()
                                                    if key not in ("loss", "num_samples")}}),
                       elapsed_seconds=time.perf_counter() - started,
                       peak_cuda_allocated_mib=(torch.cuda.max_memory_allocated() / 1024**2
                                                if args.device == "cuda" else None))
            history = [*state["history"], row]
            best_row = min(history, key=lambda item: item["validation_loss"])
            current = dict(format_version=1, config=config, completed_epochs=epoch,
                           best_epoch=best_row["epoch"], latest_checkpoint=_checkpoint_name(epoch),
                           best_checkpoint=_checkpoint_name(best_row["epoch"]), history=history)
            # The previous run.json and its checkpoints remain valid if either
            # write fails. Only run.json's successful replace commits this epoch.
            save_checkpoint(output / current["latest_checkpoint"], model, optimizer,
                            completed_epochs=epoch, best_validation_loss=best_row["validation_loss"])
            _atomic_json(output / "run.json", current)
            state = current
            _refresh_derived_files(output, state)
            print(json.dumps({key: row[key] for key in ("epoch", "train_loss", "validation_loss",
                                                       "elapsed_seconds", "peak_cuda_allocated_mib")},
                             allow_nan=False), flush=True)
        print(f"Completed {state['completed_epochs']} epochs; best epoch: {state['best_epoch']}; output: {output}",
              flush=True)
        return state


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "evaluate":
        # Import lazily: evaluate imports these training helpers and predict.
        from evaluate import main as evaluate_main
        return evaluate_main(argv[1:])
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run_training(args)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(2, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
