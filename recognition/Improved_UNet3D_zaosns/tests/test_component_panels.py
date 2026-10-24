"""Selection and rendering checks for full-volume component illustrations."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from component_analysis import component_metrics
from component_panels import render_component_panels, select_satellite


def synthetic_component_case():
    depth, height, width = np.indices((7, 12, 18))
    image = (depth * 100 + height * 10 + width).astype(np.float32)[None]
    target = np.zeros((7, 12, 18), dtype=np.int64)
    target[1:5, 2:5, 3:6] = 5
    pred_2d, pred_3d = target.copy(), target.copy()
    pred_2d[4:6, 8:10, 14:17] = 5
    pred_3d[3, 8, 14] = 5
    affine = np.array([[0., 0., -2., 10.], [0., 3., 0., 4.],
                       [4., 0., 0., -8.], [0., 0., 0., 1.]])
    prepared = SimpleNamespace(image=image, target=target, affine_dhw=affine)
    reports = {model: component_metrics(prediction, target)
               for model, prediction in (('2d', pred_2d), ('3d', pred_3d))}
    return prepared, pred_2d, pred_3d, reports


class ComponentPanelTests(unittest.TestCase):
    def reports(self, target, prediction_2d, prediction_3d):
        return {model: component_metrics(prediction, target)
                for model, prediction in (('2d', prediction_2d), ('3d', prediction_3d))}

    def test_prostate_has_priority_over_larger_rectum_satellite(self):
        target = np.zeros((8, 12, 12), dtype=np.int64)
        target[1:3, 1:3, 1:3] = 5
        target[1:3, 8:10, 1:3] = 4
        pred_2d, pred_3d = target.copy(), target.copy()
        pred_2d[6, 2, 8] = 5
        pred_3d[5:7, 8:10, 8:10] = 4
        selection = select_satellite(self.reports(target, pred_2d, pred_3d))
        self.assertEqual(selection['class_id'], 5)
        self.assertEqual(selection['model'], '2d')
        self.assertEqual(selection['component']['voxel_count'], 1)

    def test_largest_satellite_then_model_and_component_id_break_ties(self):
        target = np.zeros((8, 8, 8), dtype=np.int64)
        target[0, 0, 0] = 5
        pred_2d, pred_3d = target.copy(), target.copy()
        pred_2d[4, 1, 1] = pred_2d[4, 5, 5] = 5
        pred_3d[3, 3, 3] = 5
        selection = select_satellite(self.reports(target, pred_2d, pred_3d))
        self.assertEqual(selection['model'], '2d')
        self.assertEqual(selection['component']['component_id'], 2)
        self.assertEqual(selection['component']['bbox_dhw']['start'], [4, 1, 1])
        pred_3d[3, 3, 4] = 5
        selection = select_satellite(self.reports(target, pred_2d, pred_3d))
        self.assertEqual(selection['model'], '3d')
        self.assertEqual(selection['component']['voxel_count'], 2)

    def test_all_zero_overlap_prediction_is_not_called_a_satellite(self):
        target = np.zeros((8, 8, 8), dtype=np.int64)
        target[0, 0, 0] = 5
        pred_2d = np.zeros_like(target)
        pred_2d[5, 5, 5] = 5
        # GT overlap in the other model cannot qualify this model's fragment.
        reports = self.reports(target, pred_2d, target.copy())
        self.assertIsNone(select_satellite(reports))
        target[0, 1, 0] = 4
        pred_2d[0, 1, 0] = pred_2d[6, 6, 0] = 4
        selection = select_satellite(self.reports(target, pred_2d, target.copy()))
        self.assertEqual(selection['class_id'], 4)
        self.assertEqual(selection['model'], '2d')

    def test_selection_rejects_six_neighbour_reports(self):
        prepared, pred_2d, pred_3d, reports = synthetic_component_case()
        reports['3d'] = component_metrics(pred_3d, prepared.target, connectivity=6)
        with self.assertRaisesRegex(ValueError, '26-neighbour'):
            select_satellite(reports)

    def test_smoke_render_metadata_and_input_preservation(self):
        prepared, pred_2d, pred_3d, reports = synthetic_component_case()
        originals = [array.copy() for array in
                     (prepared.image, prepared.target, prepared.affine_dhw, pred_2d, pred_3d)]
        reports_before = copy.deepcopy(reports)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'synthetic-components.png'
            metadata = render_component_panels(prepared, pred_2d, pred_3d,
                reports['2d'], reports['3d'], output, case_label='Synthetic example')
            self.assertEqual(output.read_bytes()[:8], b'\x89PNG\r\n\x1a\n')
            self.assertEqual(metadata['status'], 'rendered')
            self.assertEqual(metadata['case_label'], 'Synthetic example')
            self.assertEqual(metadata['selection']['model'], '2d')
            self.assertEqual(metadata['selection']['class_id'], 5)
            self.assertEqual(metadata['selection']['component']['voxel_count'], 12)
            self.assertEqual(metadata['selected_depths'], [3, 4, 5])
            self.assertEqual(metadata['crop_hw'], dict(start=[0, 0], stop=[12, 18]))
            self.assertEqual(metadata['orientation']['spacing_dhw'], [4., 3., 2.])
            self.assertEqual(metadata['orientation']['display_aspect_h_over_w'], 1.5)
            self.assertFalse(metadata['orientation']['anatomically_reoriented'])
            self.assertEqual(metadata['full_volume_counts']['2d'], dict(
                predicted_component_count=2, zero_overlap_prediction_component_count=1,
                zero_overlap_prediction_voxels=12, false_positive_voxels=12,
                target_component_count=1))
            self.assertEqual(metadata['full_volume_counts']['3d']['zero_overlap_prediction_voxels'], 1)
            self.assertEqual(json.loads(json.dumps(metadata, allow_nan=False)), metadata)
            with self.assertRaises(FileExistsError):
                render_component_panels(prepared, pred_2d, pred_3d,
                    reports['2d'], reports['3d'], output, case_label='Synthetic example')
        for array, before in zip((prepared.image, prepared.target, prepared.affine_dhw,
                                  pred_2d, pred_3d), originals):
            np.testing.assert_array_equal(array, before)
        self.assertEqual(reports, reports_before)

    def test_stale_reports_rejected_before_writing(self):
        prepared, pred_2d, pred_3d, reports = synthetic_component_case()
        changed = pred_2d.copy()
        changed[0, 0, 0] = 5
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'rejected.png'
            with self.assertRaisesRegex(ValueError, 'does not match'):
                render_component_panels(prepared, changed, pred_3d,
                    reports['2d'], reports['3d'], output, case_label='Synthetic example')
            self.assertFalse(output.exists())
            altered_report = copy.deepcopy(reports['3d'])
            altered_report['classes']['5']['zero_overlap_prediction_voxels'] += 1
            with self.assertRaisesRegex(ValueError, 'does not match'):
                render_component_panels(prepared, pred_2d, pred_3d,
                    reports['2d'], altered_report, output, case_label='Synthetic example')
            self.assertFalse(output.exists())

    def test_no_eligible_satellite_returns_status_without_a_figure(self):
        prepared, pred_2d, pred_3d, reports = synthetic_component_case()
        pred_2d = prepared.target.copy()
        pred_3d = prepared.target.copy()
        reports = self.reports(prepared.target, pred_2d, pred_3d)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'no-satellite.png'
            metadata = render_component_panels(prepared, pred_2d, pred_3d,
                reports['2d'], reports['3d'], output, case_label='Synthetic example')
            self.assertEqual(metadata['status'], 'no_eligible_component')
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
