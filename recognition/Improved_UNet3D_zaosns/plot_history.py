"""Export a HipMRI run.json learning curve without loading data or checkpoints.

Example (run.json remains private)::

    python -s plot_history.py private/example/run.json --output private/curve.png

PNG and PDF are supported. Existing files require --overwrite. Dice means
weight valid cases equally, as in train.py; undefined JSON nulls become gaps,
not zero scores. The figure shows validation on the preprocessed grid, never
test performance. Matplotlib is required only for this optional export command.
"""

import argparse
import io
import json
import math
import os
from pathlib import Path
import tempfile


CLASS_NAMES = ("Body", "Bones", "Bladder", "Rectum", "Prostate")


def _number(value, name, *, metric=False):
    if metric and value is None:
        return
    if (type(value) not in (int, float) or not math.isfinite(value)
            or (metric and not 0 <= value <= 1)):
        raise ValueError(f"{name} must be finite" + (" and in [0, 1], or null." if metric else "."))


def _vector(value, size, name):
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{name} must be a list of length {size}.")
    return value


def read_history(path):
    """Validate the version-1 HipMRI metric schema before labeling a figure."""
    def invalid_constant(value):
        raise ValueError(f"Nonfinite JSON constant {value} is not allowed.")

    with Path(path).open(encoding="utf-8") as stream:
        state = json.load(stream, parse_constant=invalid_constant)
    if (not isinstance(state, dict) or type(state.get("format_version")) is not int
            or state["format_version"] != 1):
        raise ValueError("Expected version-1 run.json.")
    model = state.get("config", {}).get("model", {})
    if type(model.get("num_classes")) is not int or model["num_classes"] != 6:
        raise ValueError("This HipMRI figure requires six model classes (background plus classes 1–5).")
    history = state.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError("No completed epochs to plot.")
    if type(state.get("completed_epochs")) is not int or state["completed_epochs"] != len(history):
        raise ValueError("completed_epochs must match the history length.")
    for epoch, row in enumerate(history, 1):
        if not isinstance(row, dict) or type(row.get("epoch")) is not int or row["epoch"] != epoch:
            raise ValueError("Epochs must be consecutive, starting at one.")
        for key in ("train_loss", "validation_loss"):
            _number(row.get(key), f"Epoch {epoch} {key}")
        for key in ("train_samples", "validation_samples"):
            if type(row.get(key)) is not int or row[key] < 1:
                raise ValueError(f"Epoch {epoch} {key} must be a positive integer.")
        validation = row.get("validation")
        if not isinstance(validation, dict):
            raise ValueError(f"Epoch {epoch} has no validation metrics.")
        class_ids = validation.get("class_ids")
        if (class_ids != [1, 2, 3, 4, 5]
                or any(type(value) is not int for value in class_ids)):
            raise ValueError(f"Epoch {epoch} class_ids must be [1, 2, 3, 4, 5].")
        count = row["validation_samples"]
        _vector(validation.get("case_ids"), count, "validation.case_ids")
        valid = _vector(validation.get("valid"), count, "validation.valid")
        for flags in valid:
            _vector(flags, 5, "validation.valid row")
            if any(type(flag) is not bool for flag in flags):
                raise ValueError("validation.valid must contain booleans.")
        counts = _vector(validation.get("valid_count"), 5, "validation.valid_count")
        if any(type(value) is not int or value != sum(flags[index] for flags in valid)
               for index, value in enumerate(counts)):
            raise ValueError("validation.valid_count disagrees with validation.valid.")
        for metric in ("dice", "iou"):
            values = _vector(validation.get(metric), count, f"validation.{metric}")
            means = _vector(validation.get(f"mean_{metric}"), 5, f"validation.mean_{metric}")
            for case, scores in enumerate(values):
                _vector(scores, 5, f"validation.{metric} row")
                for index, score in enumerate(scores):
                    _number(score, f"validation.{metric}", metric=True)
                    if (score is not None) != valid[case][index]:
                        raise ValueError(f"validation.{metric} nulls disagree with validation.valid.")
            for index, mean in enumerate(means):
                _number(mean, f"validation.mean_{metric}", metric=True)
                if (mean is None) != (counts[index] == 0):
                    raise ValueError(f"validation.mean_{metric} nulls disagree with valid_count.")
                if mean is not None:
                    expected = sum(scores[index] for scores in values if scores[index] is not None) / counts[index]
                    if not math.isclose(mean, expected, rel_tol=1e-6, abs_tol=1e-8):
                        raise ValueError(f"validation.mean_{metric} disagrees with its case scores.")
    best_epoch = min(history, key=lambda row: row["validation_loss"])["epoch"]
    if type(state.get("best_epoch")) is not int or state["best_epoch"] != best_epoch:
        raise ValueError("best_epoch must identify the earliest minimum validation loss.")
    return history, best_epoch


def export_history(history, best_epoch, output, *, overwrite=False):
    """Render to memory, then publish only the explicitly requested file."""
    output = Path(output)
    file_format = output.suffix.lower().removeprefix(".")
    if file_format not in ("png", "pdf"):
        raise ValueError("Output must have a .png or .pdf extension.")
    if output.exists() and not overwrite:
        raise FileExistsError("Output already exists; choose another path or use --overwrite.")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    try:
        figure.suptitle("HipMRI training and validation · preprocessed grid (not test)", fontsize=14, y=0.97)
        loss_ax, dice_ax = axes
        loss_ax.plot(epochs, [row["train_loss"] for row in history], "o-", label="Training", color="#2864a0", markersize=4)
        loss_ax.plot(epochs, [row["validation_loss"] for row in history], "s-", label="Validation", color="#c75c20", markersize=4)
        loss_ax.set(title="Cross entropy + foreground soft Dice loss", ylabel="Mean loss per case")
        loss_ax.legend(loc="best", frameon=False)
        colors = ("#2864a0", "#c75c20", "#278451", "#a94d84", "#7762a8")
        for index, (name, color) in enumerate(zip(CLASS_NAMES, colors)):
            scores = [row["validation"]["mean_dice"][index] for row in history]
            dice_ax.plot(epochs, [float("nan") if score is None else score for score in scores],
                         marker="o", markersize=4, label=f"{index + 1} {name}", color=color)
        dice_ax.set(title="Validation Dice by class", ylabel="Mean Dice over valid cases", ylim=(-0.025, 1.025))
        dice_ax.legend(loc="best", frameon=False, ncol=2, fontsize=9)
        for axis in axes:
            axis.axvline(best_epoch, color="#666666", linestyle="--", linewidth=1, alpha=0.8)
            axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
            axis.set(xlabel="Completed epoch", xlim=(0.8, max(1.2, epochs[-1] + 0.2)))
            axis.grid(alpha=0.18)
            axis.spines[["top", "right"]].set_visible(False)
        figure.text(0.5, 0.025,
                    f"Dashed line: best validation-loss epoch {best_epoch}.  "
                    "Dice excludes background; undefined scores appear as gaps.",
                    ha="center", fontsize=9, color="#555555")
        figure.tight_layout(rect=(0, 0.075, 1, 0.92))
        buffer = io.BytesIO()
        figure.savefig(buffer, format=file_format, dpi=160, facecolor="white")
    finally:
        plt.close(figure)
    if not overwrite:
        with output.open("xb") as stream:
            stream.write(buffer.getvalue())
    else:
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".curve-", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(buffer.getvalue())
            os.replace(temporary, output)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_json", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="Explicit .png or .pdf destination.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file.")
    args = parser.parse_args()
    try:
        history, best_epoch = read_history(args.run_json)
        export_history(history, best_epoch, args.output, overwrite=args.overwrite)
    except (ValueError, OSError, KeyError, TypeError, AttributeError) as error:
        parser.error(str(error))
    print(f"Saved {args.output}: {len(history)} epochs; best validation-loss epoch {best_epoch}.")


if __name__ == "__main__":
    main()
