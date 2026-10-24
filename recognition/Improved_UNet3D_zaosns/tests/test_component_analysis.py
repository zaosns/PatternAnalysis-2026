"""Hand-counted and independent flood-fill checks for 3D components."""

import itertools
import json
import unittest

import numpy as np

from component_analysis import component_metrics, label_components


class ComponentAnalysisTests(unittest.TestCase):
    def empty(self, shape=(5, 5, 5)):
        return np.zeros(shape, dtype=np.int64)

    def test_cross_slice_face_connection_is_one_three_dimensional_component(self):
        truth = self.empty()
        truth[1:4, 2, 2] = 5
        for connectivity in (6, 26):
            result = component_metrics(truth, truth, connectivity=connectivity)
            item = result['classes']['5']
            self.assertEqual(item['predicted_component_count'], 1)
            self.assertEqual(item['target_component_count'], 1)
            self.assertEqual(item['same_class_overlap_voxels'], 3)
            component = item['prediction_components'][0]
            self.assertEqual(component['bbox_dhw'], dict(start=[1, 2, 2], stop=[4, 3, 3]))
            self.assertEqual(component['representative_depth'], 1)
            self.assertEqual(component['representative_depth_voxels'], 1)

    def test_diagonal_connections_differ_for_six_and_twenty_six(self):
        mask = self.empty()
        mask[1, 1, 1] = mask[2, 2, 2] = mask[3, 3, 2] = 5
        six = component_metrics(mask, mask, connectivity=6)['classes']['5']
        twenty_six = component_metrics(mask, mask)['classes']['5']
        self.assertEqual(six['predicted_component_count'], 3)
        self.assertEqual(six['target_component_count'], 3)
        self.assertAlmostEqual(six['largest_prediction_component_fraction'], 1 / 3)
        self.assertEqual(twenty_six['predicted_component_count'], 1)
        self.assertEqual(twenty_six['largest_prediction_component_fraction'], 1)

    def test_naturally_multiple_gt_components_are_not_assumed_errors(self):
        truth = self.empty()
        truth[0, 0, 0:2] = 2
        truth[4, 4, 4] = 2
        item = component_metrics(truth, truth)['classes']['2']
        self.assertEqual(item['predicted_component_count'], 2)
        self.assertEqual(item['target_component_count'], 2)
        self.assertEqual(item['zero_overlap_prediction_component_count'], 0)
        self.assertEqual(item['zero_overlap_prediction_voxels'], 0)
        self.assertEqual(item['false_positive_voxels'], 0)
        self.assertAlmostEqual(item['largest_prediction_component_fraction'], 2 / 3)

    def test_zero_overlap_and_partial_errors_remain_separate_even_if_largest_is_wrong(self):
        truth, pred = self.empty(), self.empty()
        truth[0, 0, 0:2] = 5
        truth[4, 0, 0] = 5
        pred[0, 0, 0] = pred[0, 1, 0] = 5  # two-voxel component with one overlap
        pred[4, 4, 1:4] = 5  # largest component is entirely incorrect
        item = component_metrics(pred, truth)['classes']['5']
        self.assertEqual(item['predicted_voxels'], 5)
        self.assertEqual(item['target_voxels'], 3)
        self.assertEqual(item['same_class_overlap_voxels'], 1)
        self.assertEqual(item['false_positive_voxels'], 4)
        self.assertEqual(item['false_negative_voxels'], 2)
        self.assertEqual(item['zero_overlap_prediction_component_count'], 1)
        self.assertEqual(item['zero_overlap_prediction_voxels'], 3)
        self.assertEqual(item['zero_overlap_target_component_count'], 1)
        self.assertEqual(item['zero_overlap_target_voxels'], 1)
        self.assertEqual(item['largest_prediction_component_fraction'], 3 / 5)
        self.assertEqual([x['nonoverlap_voxels'] for x in item['prediction_components']], [1, 3])

    def test_empty_classes_have_zero_counts_and_null_largest_fractions(self):
        empty = self.empty()
        result = component_metrics(empty, empty)
        self.assertEqual(set(result['classes']), {'1', '2', '3', '4', '5'})
        for item in result['classes'].values():
            self.assertEqual(item['predicted_component_count'], 0)
            self.assertEqual(item['target_component_count'], 0)
            self.assertEqual(item['prediction_components'], [])
            self.assertEqual(item['target_components'], [])
            self.assertIsNone(item['largest_prediction_component_fraction'])
            self.assertIsNone(item['largest_target_component_fraction'])
        pred = empty.copy()
        pred[2, 2, 2] = 1
        predicted_only = component_metrics(pred, empty)['classes']['1']
        target_only = component_metrics(empty, pred)['classes']['1']
        self.assertEqual(predicted_only['zero_overlap_prediction_voxels'], 1)
        self.assertEqual(predicted_only['largest_prediction_component_fraction'], 1)
        self.assertEqual(target_only['zero_overlap_target_voxels'], 1)
        self.assertEqual(target_only['largest_target_component_fraction'], 1)

    def test_overlap_requires_same_class(self):
        truth, pred = self.empty(), self.empty()
        truth[2, 2, 2] = 4
        pred[2, 2, 2] = 5
        result = component_metrics(pred, truth)
        self.assertEqual(result['classes']['5']['zero_overlap_prediction_component_count'], 1)
        self.assertEqual(result['classes']['5']['same_class_overlap_voxels'], 0)
        self.assertEqual(result['classes']['4']['zero_overlap_target_component_count'], 1)

    def test_label_helper_matches_report_ids_and_preserves_mask(self):
        truth = self.empty()
        truth[0, 0, :2] = truth[3, 4, 4] = truth[4, 4, 4] = 5
        mask = truth == 5
        mask_before = mask.copy()
        for connectivity in (6, 26):
            labels, count = label_components(mask, connectivity=connectivity)
            report = component_metrics(truth, truth, connectivity=connectivity)['classes']['5']
            self.assertEqual(count, report['predicted_component_count'])
            for component in report['prediction_components']:
                self.assertEqual(np.count_nonzero(labels == component['component_id']),
                                 component['voxel_count'])
            np.testing.assert_array_equal(labels > 0, mask)
        np.testing.assert_array_equal(mask, mask_before)
        for invalid in (truth, mask[0], mask[..., :0], mask.tolist()):
            with self.assertRaises(ValueError):
                label_components(invalid)
        for connectivity in (18, True, 26.0):
            with self.assertRaises(ValueError):
                label_components(mask, connectivity=connectivity)

    def test_representative_slice_selects_maximum_area_and_inputs_unchanged(self):
        truth = self.empty()
        truth[0:3, 1, 1] = 3
        truth[1:3, 1, 2] = 3
        truth_before = truth.copy()
        prediction = truth[:, :, ::-1]  # accept a non-contiguous array
        prediction_before = prediction.copy()
        result = component_metrics(prediction, truth)
        component = result['classes']['3']['target_components'][0]
        self.assertEqual(component['representative_depth'], 1)
        self.assertEqual(component['representative_depth_voxels'], 2)
        self.assertEqual(component['voxel_count'], 5)
        np.testing.assert_array_equal(truth, truth_before)
        np.testing.assert_array_equal(prediction, prediction_before)
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)

    def test_invalid_inputs_fail_instead_of_silently_casting(self):
        good = self.empty()
        invalid_arrays = [good.tolist(), good[0], good[..., :0], good[:1],
                          good.astype(float), good.astype(bool), good.astype(complex),
                          np.full_like(good, -1), np.full_like(good, 6)]
        for invalid in invalid_arrays:
            with self.subTest(dtype=getattr(invalid, 'dtype', None), shape=getattr(invalid, 'shape', None)):
                with self.assertRaises(ValueError):
                    component_metrics(invalid, good)
                with self.assertRaises(ValueError):
                    component_metrics(good, invalid)
        for connectivity in (0, 1, 18, True, 6.0, '26'):
            with self.assertRaises(ValueError):
                component_metrics(good, good, connectivity=connectivity)
        for num_classes in (0, 1, True, 6.0, '6'):
            with self.assertRaises(ValueError):
                component_metrics(good, good, num_classes=num_classes)

    def test_component_counts_and_overlap_match_independent_flood_fill(self):
        # An independent set-based traversal checks the library's 3D graph,
        # including border voxels, rather than reproducing ndimage internals.
        rng = np.random.default_rng(3710)
        for connectivity in (6, 26):
            offsets = [offset for offset in itertools.product((-1, 0, 1), repeat=3)
                       if any(offset) and (connectivity == 26 or sum(abs(x) for x in offset) == 1)]
            for _ in range(4):
                pred = rng.integers(0, 3, size=(4, 5, 3), dtype=np.int64)
                truth = rng.integers(0, 3, size=(4, 5, 3), dtype=np.int64)
                result = component_metrics(pred, truth, connectivity=connectivity, num_classes=3)
                for class_id in (1, 2):
                    for mask, other, key in ((pred, truth, 'prediction_components'),
                                             (truth, pred, 'target_components')):
                        pending = set(map(tuple, np.argwhere(mask == class_id)))
                        expected = []
                        while pending:
                            seed = pending.pop()
                            visited, frontier = {seed}, [seed]
                            while frontier:
                                current = frontier.pop()
                                for offset in offsets:
                                    neighbour = tuple(int(a + b) for a, b in zip(current, offset))
                                    if neighbour in pending:
                                        pending.remove(neighbour)
                                        visited.add(neighbour)
                                        frontier.append(neighbour)
                            expected.append((len(visited), sum(other[point] == class_id for point in visited)))
                        actual = [(item['voxel_count'], item['overlap_voxels'])
                                  for item in result['classes'][str(class_id)][key]]
                        self.assertEqual(sorted(actual), sorted(expected))


if __name__ == '__main__':
    unittest.main()
