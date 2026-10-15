"""Render private validation figures from already prepared CPU case volumes.

These panels retain array axes: D/H/W correspond to source K/J/I, without an
anatomical reorientation. Planes are selected by ground-truth organ occupancy
and area, not a claimed anatomical base or apex.
No patient identifiers are read into titles and no volume files are written.
"""

from __future__ import annotations

import io
import math
from pathlib import Path

import numpy as np
import torch

from dataset import PreprocessedCase


CLASS_NAMES = {0: "Background", 1: "Body", 2: "Bones", 3: "Bladder", 4: "Rectum", 5: "Prostate"}
# Body and background remain transparent, so their extent cannot hide organs.
OVERLAY_COLORS = {2: "#e69f00", 3: "#0072b2", 4: "#b47cc7", 5: "#f0e442"}
ERROR_COLORS = {1: "#38c172", 2: "#ff4bcd", 3: "#22d3ee"}
ERROR_NAMES = {1: "True positive", 2: "False positive", 3: "False negative"}


def _selected_planes(target: torch.Tensor, include_rectum: bool = False) -> list[dict]:
    """Merge fixed candidates by depth while retaining every selection reason."""
    occupied = torch.nonzero((target == 5).any(dim=(1, 2)), as_tuple=False).flatten().tolist()
    candidates = []
    if not occupied:
        candidates.append((target.shape[0] // 2, "Prostate empty: middle fallback"))
    else:
        largest = int((target == 5).sum(dim=(1, 2)).argmax())
        candidates.extend(((occupied[0], "Prostate first"), (largest, "Prostate max area"),
                           (occupied[-1], "Prostate last")))
    bladder_area = (target == 3).sum(dim=(1, 2))
    if bladder_area.max() > 0:
        candidates.append((int(bladder_area.argmax()), "Bladder max area"))
    if include_rectum:
        rectum_area = (target == 4).sum(dim=(1, 2))
        if rectum_area.max() > 0:
            candidates.append((int(rectum_area.argmax()), "Rectum max area"))
    planes = {}
    for depth, reason in candidates:
        planes.setdefault(depth, []).append(reason)
    return [{"depth": depth, "selection_reasons": reasons} for depth, reasons in planes.items()]


def select_depths(target: torch.Tensor, *, include_rectum: bool = False) -> tuple[list[int], bool]:
    """Return fixed organ-based candidates, deduplicated in candidate order.

    Candidates: first prostate plane, maximum-area prostate plane, last prostate
    plane, maximum-area bladder plane. Area ties use the smallest depth index.
    Optionally append the maximum-area rectum plane. An empty prostate contributes
    the middle depth; empty bladder/rectum targets add no plane.
    """
    if type(include_rectum) is not bool:
        raise ValueError("include_rectum must be boolean.")
    return [plane["depth"] for plane in _selected_planes(target, include_rectum)], not bool((target == 5).any())


def organ_errors(prediction: np.ndarray, target: np.ndarray, class_id: int) -> np.ndarray:
    """Encode one organ: 0 transparent, 1 TP, 2 FP, 3 FN (no smoothing)."""
    predicted, actual = prediction == class_id, target == class_id
    result = np.zeros(target.shape, dtype=np.uint8)
    result[predicted & actual] = 1
    result[predicted & ~actual] = 2
    result[~predicted & actual] = 3
    return result


def _validate(prepared, prediction, case_label, metrics):
    if not isinstance(prepared, PreprocessedCase):
        raise ValueError("prepared must be a PreprocessedCase.")
    if (not isinstance(case_label, str) or not case_label.strip() or len(case_label) > 80
            or case_label != case_label.strip() or any(c in case_label for c in "\n\r/\\")):
        raise ValueError("case_label must be a short display label without paths or newlines.")
    for name, labels in (("target", prepared.target), ("prediction", prediction)):
        if (not isinstance(labels, torch.Tensor) or labels.device.type != "cpu"
                or labels.dtype != torch.long or labels.ndim != 3 or min(labels.shape) < 1):
            raise ValueError(f"{name} must be a CPU torch.long tensor with nonempty [D,H,W] shape.")
        if labels.min() < 0 or labels.max() > 5:
            raise ValueError(f"{name} class indices must lie in [0, 5].")
    if prediction.shape != prepared.target.shape:
        raise ValueError("prediction and target shapes must match.")
    image = prepared.image
    if (not isinstance(image, torch.Tensor) or image.device.type != "cpu"
            or not image.is_floating_point() or tuple(image.shape) != (1, *prediction.shape)
            or not torch.isfinite(image).all()):
        raise ValueError("image must be a finite CPU floating tensor with shape [1,D,H,W].")
    for name in ("affine_dhw", "source_affine_dhw"):
        affine = getattr(prepared, name)
        if (not isinstance(affine, torch.Tensor) or affine.device.type != "cpu"
                or affine.shape != (4, 4) or not affine.is_floating_point()
                or not torch.isfinite(affine).all()
                or not torch.allclose(affine[3], affine.new_tensor([0, 0, 0, 1]))
                or torch.linalg.det(affine[:3, :3]).abs() < 1e-12):
            raise ValueError(f"{name} must be a finite, nonsingular CPU spatial affine.")
    if prepared.spatial_unit not in ("mm", "meter", "micron", "unknown"):
        raise ValueError("spatial_unit must identify the NIfTI spatial units.")
    if metrics is not None:
        if not isinstance(metrics, dict) or metrics.get("class_ids") != [1, 2, 3, 4, 5]:
            raise ValueError("metrics must identify class_ids [1, 2, 3, 4, 5].")
        for name in ("dice", "iou"):
            values = metrics.get(name)
            if not isinstance(values, list) or len(values) != 5:
                raise ValueError(f"metrics.{name} must contain five single-volume scores.")
            for value in values:
                if value is not None and (type(value) not in (int, float)
                                          or not math.isfinite(value) or not 0 <= value <= 1):
                    raise ValueError(f"metrics.{name} must contain scores in [0, 1] or null.")


def _display_window(image):
    values = image[image != 0]
    if not values.size:
        return 0., 1., "all-zero fallback"
    lower, upper = (float(v) for v in np.percentile(values, [1, 99]))
    method = "whole prepared image nonzero voxel percentiles 1 and 99"
    if lower == upper:
        lower, upper = float(values.min()), float(values.max())
        method += "; equal percentiles use nonzero min/max"
    if lower == upper:
        radius = max(abs(lower) * .01, 1e-3)
        lower, upper = lower - radius, upper + radius
        method += "; constant image expands a symmetric window"
    return lower, upper, method


def render_case_panels(
    prepared: PreprocessedCase,
    prediction_cpu: torch.Tensor,
    output_path: str | Path,
    *,
    case_label: str,
    metrics: dict | None = None,
    include_rectum: bool = False,
) -> dict:
    """Save one PNG and return JSON-safe geometry, slice and display metadata.

    ``metrics`` optionally contains single-volume ``class_ids=[1,2,3,4,5]``
    plus flat ``dice`` and ``iou`` lists; undefined classes use JSON null.
    ``include_rectum`` appends a rectum error column and maximum-area target
    plane, retaining the original candidates for comparison with older panels.
    Labels are displayed with nearest interpolation. All columns share
    the same array orientation, intensity window and affine-derived pixel
    aspect ratio. Oblique/sheared volumes are not rectified: the in-plane
    angle and direction vectors are recorded for interpreting array planes.
    Existing output files are never overwritten, including races at creation.
    """
    output = Path(output_path)
    if output.suffix.lower() != ".png":
        raise ValueError("Prediction panels must be written to a .png path.")
    if output.exists():
        raise FileExistsError("Output already exists; choose a new prediction-panel path.")
    if type(include_rectum) is not bool:
        raise ValueError("include_rectum must be boolean.")
    _validate(prepared, prediction_cpu, case_label, metrics)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Patch

    image = prepared.image.detach().float().numpy()[0]
    target = prepared.target.numpy()
    prediction = prediction_cpu.numpy()
    planes = _selected_planes(prepared.target, include_rectum)
    depths = [plane["depth"] for plane in planes]
    empty_prostate = not bool((prepared.target == 5).any())
    lower, upper, window_method = _display_window(image)
    affine = prepared.affine_dhw.detach().double().numpy()
    source_affine = prepared.source_affine_dhw.detach().double().numpy()
    source_mapping = np.linalg.solve(source_affine, affine)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    direction = affine[:3, :3] / spacing
    aspect = float(spacing[1] / spacing[2])
    in_plane_angle = float(np.degrees(np.arccos(np.clip(direction[:, 1] @ direction[:, 2], -1, 1))))
    source_depths = [float((source_mapping @ np.array([d, (target.shape[1]-1)/2,
                                                       (target.shape[2]-1)/2, 1]))[0]) for d in depths]
    overlay_lookup = np.zeros((6, 4))
    for class_id, color in OVERLAY_COLORS.items():
        overlay_lookup[class_id] = to_rgba(color, alpha=.58)
    error_lookup = np.zeros((4, 4))
    for code, color in ERROR_COLORS.items():
        error_lookup[code] = to_rgba(color, alpha=.85)

    metadata = {
        "format_version": 1, "case_label": case_label, "evaluation_split": "validation",
        "grid": "prepared", "selected_depths": depths,
        "source_depth_centres": source_depths, "target_prostate_empty": empty_prostate,
        "target_bladder_empty": not bool((prepared.target == 3).any()),
        "target_rectum_empty": not bool((prepared.target == 4).any()),
        "include_rectum": include_rectum,
        "selected_planes": planes,
        "selection": "prostate first / max area / last, then bladder max area; deduplicate in candidate order; "
                     "area ties use smallest index; empty prostate uses middle fallback; empty bladder adds no plane"
                     + ("; then append rectum max area if present" if include_rectum else ""),
        "orientation": {
            "axes": {"D": "source K", "H": "source J", "W": "source I"},
            "origin": "upper", "flips": [], "anatomically_reoriented": False,
            "affine_dhw": affine.tolist(), "source_affine_dhw": source_affine.tolist(),
            "unit_direction_vectors_dhw": direction.T.tolist(),
            "spacing_dhw": spacing.tolist(), "spatial_unit": prepared.spatial_unit,
            "display_aspect_h_over_w": aspect, "in_plane_angle_degrees": in_plane_angle,
            "note": "Array planes retain source orientation, including obliquity/shear; no base/apex claim.",
        },
        "display_window": {"lower": lower, "upper": upper, "method": window_method},
        "legend": {"overlay": {str(c): {"name": CLASS_NAMES[c], "color": color}
                                for c, color in OVERLAY_COLORS.items()},
                   "hidden_overlay_classes": [0, 1],
                   "errors": {str(c): {"name": ERROR_NAMES[c], "color": color}
                              for c, color in ERROR_COLORS.items()}},
        "panel_error_counts": [],
    }
    error_classes = [5, 3] + ([4] if include_rectum else [])
    columns = 3 + len(error_classes)
    figure, axes = plt.subplots(len(depths), columns, figsize=(3 * columns, 3.1 * len(depths) + 2.1), squeeze=False)
    try:
        figure.suptitle(f"{case_label} | validation on prepared grid (not test)", fontsize=16, y=.985)
        if metrics is not None:
            def score(name, index):
                value = metrics[name][index]
                return "undefined" if value is None else f"{value:.3f}"
            subtitle = (f"Whole-volume Dice / IoU:  Prostate {score('dice', 4)} / {score('iou', 4)}"
                        f"    Bladder {score('dice', 2)} / {score('iou', 2)}")
            if include_rectum:
                subtitle += f"    Rectum {score('dice', 3)} / {score('iou', 3)}"
            figure.text(.5, .953, subtitle, ha="center", fontsize=11)
        for row, (depth, source_depth) in enumerate(zip(depths, source_depths)):
            for column, axis in enumerate(axes[row]):
                axis.imshow(image[depth], cmap="gray", vmin=lower, vmax=upper,
                            origin="upper", interpolation="nearest", aspect=aspect)
                axis.set_xticks([])
                axis.set_yticks([])
                for spine in axis.spines.values():
                    spine.set_visible(False)
            axes[row, 1].imshow(overlay_lookup[target[depth]], origin="upper", interpolation="nearest", aspect=aspect)
            axes[row, 2].imshow(overlay_lookup[prediction[depth]], origin="upper", interpolation="nearest", aspect=aspect)
            slice_counts = {"depth": depth}
            for column, class_id in enumerate(error_classes, 3):
                errors = organ_errors(prediction[depth], target[depth], class_id)
                axes[row, column].imshow(error_lookup[errors], origin="upper", interpolation="nearest", aspect=aspect)
                counts = {name: int(np.count_nonzero(errors == code))
                          for code, name in ((1, "tp"), (2, "fp"), (3, "fn"))}
                slice_counts[CLASS_NAMES[class_id].lower()] = counts
                axes[row, column].set_xlabel(f"TP {counts['tp']}   FP {counts['fp']}   FN {counts['fn']}", fontsize=9)
            metadata["panel_error_counts"].append(slice_counts)
            row_note = "\n".join(planes[row]["selection_reasons"])
            axes[row, 0].set_ylabel(f"{row_note}\nd={depth}; source k={source_depth:.1f}", fontsize=9)
        titles = ["MRI", "Ground truth overlay", "Prediction overlay"] + [
            f"{CLASS_NAMES[class_id]} errors" for class_id in error_classes]
        for axis, title in zip(axes[0], titles):
            axis.set_title(title, fontsize=12, pad=9)
        legend_handles = [Patch(facecolor=color, label=f"{class_id} {CLASS_NAMES[class_id]}")
                          for class_id, color in OVERLAY_COLORS.items()]
        legend_handles.extend(Patch(facecolor=color, label=ERROR_NAMES[code])
                              for code, color in ERROR_COLORS.items())
        figure.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(.5, .055),
                      ncol=7, frameon=False, fontsize=10)
        figure.text(.5, .035, "Background (0) and Body (1) overlays hidden. Colors denote classes or errors, not confidence.",
                    ha="center", fontsize=10)
        figure.text(.5, .014, f"Array axes: H down / W right; no anatomical reorientation. "
                    f"H/W spacing: {spacing[1]:.3g}/{spacing[2]:.3g} {prepared.spatial_unit}. "
                    "Planes are selected using ground truth.", ha="center", fontsize=9)
        figure.subplots_adjust(left=.08, right=.99, top=.905, bottom=.13, wspace=.06, hspace=.25)
        stream = io.BytesIO()
        figure.savefig(stream, format="png", dpi=120, facecolor="white")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as handle:
            handle.write(stream.getvalue())
    finally:
        plt.close(figure)
    return metadata
