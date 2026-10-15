"""Epoch-boundary checkpoints for SegmentationModel3D and its optimizer.

Only state dictionaries and primitive metadata are serialized. Keep checkpoint
files outside Git. See README.md for the scope and results of CPU/CUDA checks;
restoring RNG alone does not guarantee deterministic GPU computations.
"""

import math
import os
from pathlib import Path
import random
import tempfile

import torch

from modules import SegmentationModel3D


def _progress(completed_epochs, best_validation_loss) -> None:
    if isinstance(completed_epochs, bool) or not isinstance(completed_epochs, int) or completed_epochs < 0:
        raise ValueError("completed_epochs must be a nonnegative integer.")
    if best_validation_loss is not None and (
        isinstance(best_validation_loss, bool)
        or not isinstance(best_validation_loss, (int, float))
        or not math.isfinite(best_validation_loss)
    ):
        raise ValueError("best_validation_loss must be finite or None.")


def _optimizer_layout(model, optimizer) -> dict:
    """Record parameter names in order, since optimizer state loads positionally."""
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    groups, seen = [], set()
    for group in optimizer.param_groups:
        ordered = []
        for parameter in group["params"]:
            identity = id(parameter)
            if identity not in names or identity in seen:
                raise ValueError("Optimizer parameters must be unique parameters of this model.")
            seen.add(identity)
            ordered.append(names[identity])
        groups.append(ordered)
    cls = type(optimizer)
    return {"class": f"{cls.__module__}.{cls.__qualname__}", "parameter_groups": groups}


def save_checkpoint(
    path: str | Path,
    model: SegmentationModel3D,
    optimizer: torch.optim.Optimizer,
    *,
    completed_epochs: int,
    best_validation_loss: float | None = None,
) -> None:
    """Save after a completed epoch, replacing the destination only on success.

    Store constructor settings, model/optimizer state, progress, Python/CPU RNG
    and CUDA RNG when initialized. This does not select the best model: callers
    decide when to write latest.pt or best.pt using validation results only.
    Use ordinary PyTorch optimizers with tensor/primitive state. Model structure
    and constructor settings must not have been changed after construction.
    """
    if type(model) is not SegmentationModel3D:
        raise ValueError("Checkpoint model must be a SegmentationModel3D.")
    _progress(completed_epochs, best_validation_loss)
    payload = {
        "format_version": 1,
        "model_config": model.get_config(),
        "model_state_dict": model.state_dict(),
        "optimizer_layout": _optimizer_layout(model, optimizer),
        "optimizer_state_dict": optimizer.state_dict(),
        "completed_epochs": completed_epochs,
        "best_validation_loss": best_validation_loss,
        "python_rng_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.",
                                         suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            torch.save(payload, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_checkpoint(
    path: str | Path,
    model: SegmentationModel3D,
    optimizer: torch.optim.Optimizer,
    *,
    restore_rng: bool = True,
) -> dict:
    """Restore into fresh, compatible model/optimizer objects; return progress.

    Construct the model on its intended device before constructing the optimizer.
    Configurations, tensor shapes/dtypes, optimizer class and parameter-group
    order must match. Saved
    optimizer hyperparameters replace those passed to the new optimizer. Use
    fresh objects and discard them if loading fails; loading is not transactional.

    With restore_rng=True, restore Python, CPU and saved CUDA RNG states after
    loading. CUDA RNG restoration requires the saved number of devices. Disable
    this flag for a device transfer that does not aim to reproduce the next step.
    Only the same next batches/settings in the same environment can reproduce a
    continuation: loader generators/workers, samplers, NumPy/MPS RNG, schedulers,
    AMP scalers and criterion configuration are not captured. No mid-epoch resume.
    Model train/eval mode and gradients are not saved; the epoch functions set
    their mode and training clears gradients before the next optimizer step.
    """
    if type(model) is not SegmentationModel3D or not isinstance(restore_rng, bool):
        raise ValueError("Require a SegmentationModel3D and a boolean restore_rng.")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    fields = {"format_version", "model_config", "model_state_dict", "optimizer_layout",
              "optimizer_state_dict", "completed_epochs", "best_validation_loss",
              "python_rng_state", "torch_rng_state", "cuda_rng_states"}
    if (not isinstance(payload, dict) or set(payload) != fields
            or type(payload["format_version"]) is not int or payload["format_version"] != 1):
        raise ValueError("Unsupported checkpoint format.")
    _progress(payload["completed_epochs"], payload["best_validation_loss"])
    if payload["model_config"] != model.get_config():
        raise ValueError("Checkpoint model configuration does not match.")
    if payload["optimizer_layout"] != _optimizer_layout(model, optimizer):
        raise ValueError("Checkpoint optimizer class or parameter-group order does not match.")
    saved = payload["model_state_dict"]
    current = model.state_dict()
    if not isinstance(saved, dict) or saved.keys() != current.keys():
        raise ValueError("Checkpoint model state keys do not match.")
    for name, tensor in current.items():
        other = saved[name]
        if not isinstance(other, torch.Tensor) or other.shape != tensor.shape or other.dtype != tensor.dtype:
            raise ValueError(f"Checkpoint shape or dtype does not match for {name}.")
    if restore_rng:
        # Validate CPU/Python state using isolated generators before changing globals.
        random.Random().setstate(payload["python_rng_state"])
        torch.Generator(device="cpu").set_state(payload["torch_rng_state"])
        cuda_states = payload["cuda_rng_states"]
        if not isinstance(cuda_states, list) or any(
            not isinstance(state, torch.Tensor) or state.dtype != torch.uint8 or state.ndim != 1
            for state in cuda_states
        ):
            raise ValueError("CUDA RNG states must be a list of byte tensors.")
        if cuda_states and (not torch.cuda.is_available() or len(cuda_states) != torch.cuda.device_count()):
            raise ValueError("CUDA RNG device count differs; use restore_rng=False for transfer.")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    if restore_rng:
        random.setstate(payload["python_rng_state"])
        torch.set_rng_state(payload["torch_rng_state"])
        if payload["cuda_rng_states"]:
            torch.cuda.set_rng_state_all(payload["cuda_rng_states"])
    return {
        "completed_epochs": payload["completed_epochs"],
        "best_validation_loss": payload["best_validation_loss"],
        "model_config": payload["model_config"],
    }
