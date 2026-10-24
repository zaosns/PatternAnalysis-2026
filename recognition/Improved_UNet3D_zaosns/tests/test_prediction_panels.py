"""Asymmetric phantoms check validation-plane selection and error display."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from dataset import CaseRecord, PreprocessedCase
from prediction_panels import organ_errors, render_case_panels, select_depths


def synthetic_case():
    d, h, w = torch.meshgrid(torch.arange(7), torch.arange(12), torch.arange(18), indexing="ij")
    image = (d * 100 + h * 10 + w).float().unsqueeze(0)
    target = torch.zeros((7, 12, 18), dtype=torch.long)
    target[:, 1:11, 1:17] = 1
    target[1, 3:6, 2:5] = 5
    target[3, 4:8, 3:7] = 5
    target[6, 3:6, 5:8] = 5
    target[1:7, 7:10, 12:16] = 3
    target[:, 2:5, 12:15] = 2
    target[:, 8:10, 6:8] = 4
    affine = torch.tensor([[0., 0., -2., 10.], [0., 3., 0., 4.], [4., 0., 0., -8.],
                           [0., 0., 0., 1.]], dtype=torch.float64)
    source = affine.clone()
    source[:3, :3] /= 2
    source[:3, 3] -= source[:3, :3] @ torch.full((3,), .5, dtype=torch.float64)
    return PreprocessedCase(
        record=CaseRecord("do-not-display-case", "do-not-display-patient", Path("image.nii"), Path("label.nii")),
        image=image, target=target, affine_dhw=affine, source_shape_dhw=(14, 24, 36),
        source_affine_dhw=source, spatial_unit="mm", label_unit_assumed=True,
        normalization_mean=0., normalization_std=1., normalization_scale=1.,
        nonzero_count=int((image != 0).sum()), downsample_factor=2,
    )


class PredictionPanelTests(unittest.TestCase):
    def setUp(self):
        self.sample = synthetic_case()
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.output = Path(self.folder.name) / "panel.png"

    def render(self, prepared=None, prediction=None, **kwargs):
        return render_case_panels(prepared or self.sample,
                                  self.sample.target.clone() if prediction is None else prediction,
                                  self.output, case_label="Validation example A", **kwargs)

    def test_sparse_occupied_depths_and_deduplication(self):
        self.assertEqual(select_depths(self.sample.target), ([1, 3, 6], False))
        labels = self.sample.target.clone()
        labels[2, 6:11, 11:17] = 3
        self.assertEqual(select_depths(labels), ([1, 3, 6, 2], False))
        labels = torch.zeros_like(self.sample.target)
        labels[2, 0, 0] = labels[5, 0, 0] = 5
        self.assertEqual(select_depths(labels), ([2, 5], False))
        labels[5] = 0
        self.assertEqual(select_depths(labels), ([2], False))
        labels[:] = 0
        self.assertEqual(select_depths(labels), ([3], True))

    def test_asymmetric_error_mapping(self):
        target = np.array([[5, 5, 3, 0], [0, 3, 0, 0]])
        prediction = np.array([[5, 0, 5, 0], [5, 0, 3, 0]])
        np.testing.assert_array_equal(organ_errors(prediction, target, 5), [[1, 3, 2, 0], [2, 0, 0, 0]])
        np.testing.assert_array_equal(organ_errors(prediction, target, 3), [[0, 0, 3, 0], [0, 3, 2, 0]])

    def test_render_preserves_array_orientation_and_physical_aspect(self):
        import matplotlib.axes
        original = matplotlib.axes.Axes.imshow
        calls = []

        def observed(axis, array, *args, **kwargs):
            calls.append((np.asarray(array).copy(), kwargs))
            return original(axis, array, *args, **kwargs)

        prediction = self.sample.target.clone()
        prediction[1, 3, 2] = 0  # FN, at an off-centre known coordinate.
        prediction[1, 1, 15] = 5  # FP on the opposite side of the same plane.
        with patch.object(matplotlib.axes.Axes, "imshow", new=observed):
            metadata = self.render(prediction=prediction)
        self.assertEqual(self.output.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(metadata["selected_depths"], [1, 3, 6])
        self.assertEqual(metadata["source_depth_centres"], [2.5, 6.5, 12.5])
        self.assertEqual(metadata["orientation"]["spacing_dhw"], [4., 3., 2.])
        self.assertEqual(metadata["orientation"]["display_aspect_h_over_w"], 1.5)
        self.assertEqual(metadata["orientation"]["flips"], [])
        grayscale = [(a, k) for a, k in calls if a.ndim == 2]
        self.assertEqual(len(grayscale), 15)
        for row, depth in enumerate([1, 3, 6]):
            for array, options in grayscale[row*5:(row+1)*5]:
                np.testing.assert_array_equal(array, self.sample.image[0, depth].numpy())
                self.assertEqual(options["origin"], "upper")
                self.assertEqual(options["interpolation"], "nearest")
                self.assertEqual(options["aspect"], 1.5)
        errors = metadata["panel_error_counts"][0]["prostate"]
        self.assertEqual(errors, {"tp": 8, "fp": 1, "fn": 1})
        self.assertNotIn("do-not-display", json.dumps(metadata, allow_nan=False))

    def test_empty_prostate_and_constant_window_are_explicit(self):
        sample = replace(self.sample, image=torch.zeros_like(self.sample.image), target=torch.zeros_like(self.sample.target))
        metadata = self.render(prepared=sample, prediction=sample.target)
        self.assertTrue(metadata["target_prostate_empty"])
        self.assertEqual(metadata["selected_depths"], [3])
        self.assertEqual(metadata["display_window"], {"lower": 0., "upper": 1., "method": "all-zero fallback"})

    def test_figure_and_metadata_identify_the_evaluation_split(self):
        import matplotlib.figure
        original = matplotlib.figure.Figure.suptitle
        titles = []

        def observed(figure, title, *args, **kwargs):
            titles.append(title)
            return original(figure, title, *args, **kwargs)

        with patch.object(matplotlib.figure.Figure, "suptitle", new=observed):
            validation = self.render()
            self.output.unlink()
            test = render_case_panels(self.sample, self.sample.target, self.output,
                                      case_label="Test example A", evaluation_split="test")
        self.assertEqual(validation["evaluation_split"], "validation")
        self.assertEqual(test["evaluation_split"], "test")
        self.assertEqual(titles, ["Validation example A | validation on prepared grid (not test)",
                                  "Test example A | test on prepared grid"])
        self.assertEqual(test["selected_depths"], validation["selected_depths"])
        self.assertEqual(test["panel_error_counts"], validation["panel_error_counts"])
        self.assertEqual(self.output.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_invalid_evaluation_split_does_not_create_a_panel(self):
        for split in ("train", "Test", "", None, True):
            with self.subTest(split=split), self.assertRaisesRegex(ValueError, "evaluation_split"):
                self.render(evaluation_split=split)
            self.assertFalse(self.output.exists())

    def test_rectum_option_adds_target_plane_and_sixth_error_column(self):
        import matplotlib.axes
        target = self.sample.target.clone()
        target[4, 8:11, 6:10] = 4
        sample = replace(self.sample, target=target)
        prediction = target.clone()
        prediction[4, 8, 6] = 0
        prediction[4, 0, 0] = 4
        self.assertEqual(select_depths(target), ([1, 3, 6], False))
        self.assertEqual(select_depths(target, include_rectum=True), ([1, 3, 6, 4], False))
        original = matplotlib.axes.Axes.imshow
        grayscale = []

        def observed(axis, array, *args, **kwargs):
            if np.asarray(array).ndim == 2:
                grayscale.append(np.asarray(array).copy())
            return original(axis, array, *args, **kwargs)

        with patch.object(matplotlib.axes.Axes, "imshow", new=observed):
            metadata = self.render(prepared=sample, prediction=prediction, include_rectum=True)
        self.assertTrue(metadata["include_rectum"])
        self.assertEqual(metadata["selected_depths"], [1, 3, 6, 4])
        self.assertEqual(metadata["selected_planes"][-1]["selection_reasons"], ["Rectum max area"])
        self.assertEqual(metadata["panel_error_counts"][-1]["rectum"], {"tp": 11, "fp": 1, "fn": 1})
        self.assertEqual(len(grayscale), 4 * 6)
        for row, depth in enumerate(metadata["selected_depths"]):
            for image in grayscale[row * 6:(row + 1) * 6]:
                np.testing.assert_array_equal(image, sample.image[0, depth].numpy())
        no_rectum = target.clone()
        no_rectum[no_rectum == 4] = 1
        self.assertEqual(select_depths(no_rectum, include_rectum=True), select_depths(no_rectum))

    def test_rejects_shapes_classes_and_nonfinite_inputs(self):
        invalid = [torch.zeros((1, 12, 18), dtype=torch.long), self.sample.target.float(),
                   torch.full_like(self.sample.target, 6), torch.full_like(self.sample.target, -1)]
        for prediction in invalid:
            with self.subTest(shape=prediction.shape, dtype=prediction.dtype):
                with self.assertRaises(ValueError):
                    self.render(prediction=prediction)
        image = self.sample.image.clone()
        image[0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            self.render(prepared=replace(self.sample, image=image))
        affine = self.sample.affine_dhw.clone()
        affine[:, 0] = 0
        with self.assertRaisesRegex(ValueError, "affine"):
            self.render(prepared=replace(self.sample, affine_dhw=affine))
        self.assertFalse(self.output.exists())

    def test_refuses_existing_output_and_invalid_metrics(self):
        self.output.write_bytes(b"keep this file")
        with self.assertRaises(FileExistsError):
            self.render()
        self.assertEqual(self.output.read_bytes(), b"keep this file")
        self.output.unlink()
        with self.assertRaisesRegex(ValueError, "metrics.dice"):
            self.render(metrics={"class_ids": [1, 2, 3, 4, 5], "dice": [0, 0, 0, 0, float("nan")], "iou": [0]*5})
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
