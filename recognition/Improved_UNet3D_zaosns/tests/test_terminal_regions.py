"""Hand-counted checks for the fixed, target-defined prostate terminal protocol."""

import copy
import json
import unittest

import torch

import terminal_regions


class TerminalRegionTests(unittest.TestCase):
    @staticmethod
    def asymmetric_case():
        # Six occupied GT depths. All non-prostate anatomy remains negative
        # for this prostate-only metric, including class 1 below the extent.
        target = torch.zeros((10, 2, 3), dtype=torch.long)
        prediction = torch.zeros_like(target)
        target[0] = 1
        for depth, points in {
            1: [(0, 0), (0, 1)], 2: [(0, 0)], 4: [(0, 0)],
            5: [(0, 0), (0, 1)], 6: [(0, 0)], 8: [(0, 0)],
        }.items():
            for y, x in points:
                target[depth, y, x] = 5
        for depth, points in {
            0: [(1, 2)], 1: [(0, 0), (1, 2)], 3: [(0, 0), (1, 2)],
            4: [(0, 0)], 5: [(0, 0), (0, 1)], 6: [(0, 0), (1, 2)],
            7: [(1, 1)], 8: [(1, 2)], 9: [(0, 0), (0, 1), (0, 2)],
        }.items():
            for y, x in points:
                prediction[depth, y, x] = 5
        return prediction, target

    def assert_region(self, region, *, depths, target, predicted, tp, fp, fn,
                      dice, iou, precision, recall):
        self.assertEqual(region['depth_indices'], depths)
        self.assertEqual(region['num_planes'], len(depths))
        for key, expected in dict(target_voxels=target, predicted_voxels=predicted,
                                  tp=tp, fp=fp, fn=fn).items():
            self.assertEqual(region[key], expected, key)
        for key, expected in dict(dice=dice, iou=iou, precision=precision, recall=recall).items():
            self.assertIs(region['valid'][key], expected is not None)
            if expected is None:
                self.assertIsNone(region[key])
            else:
                self.assertAlmostEqual(region[key], expected, places=12, msg=key)

    def test_hand_counted_terminal_and_whole_volume_errors_include_full_planes(self):
        prediction, target = self.asymmetric_case()
        result = terminal_regions.terminal_region_metrics(prediction, target)
        self.assertEqual(result['shape_dhw'], [10, 2, 3])
        self.assertEqual(result['occupied_depths'], [1, 2, 4, 5, 6, 8])
        self.assertEqual(result['planes_per_end'], 2)
        self.assertEqual(result['overlap_depths'], [])
        self.assertFalse(result['target_empty'])
        self.assertTrue(result['extent_defined'])
        regions = result['regions']
        self.assert_region(regions['whole_volume'], depths=list(range(10)), target=8,
                           predicted=15, tp=5, fp=10, fn=3, dice=10 / 23, iou=5 / 18,
                           precision=1 / 3, recall=5 / 8)
        self.assert_region(regions['lower_terminal'], depths=[1, 2], target=3,
                           predicted=2, tp=1, fp=1, fn=2, dice=.4, iou=.25,
                           precision=.5, recall=1 / 3)
        self.assert_region(regions['upper_terminal'], depths=[6, 8], target=2,
                           predicted=3, tp=1, fp=2, fn=1, dice=.4, iou=.25,
                           precision=1 / 3, recall=.5)
        self.assert_region(regions['terminal_union'], depths=[1, 2, 6, 8], target=5,
                           predicted=5, tp=2, fp=3, fn=3, dice=.4, iou=.25,
                           precision=.4, recall=.4)
        self.assert_region(regions['occupied_interior'], depths=[4, 5], target=3,
                           predicted=3, tp=3, fp=0, fn=0, dice=1., iou=1.,
                           precision=1., recall=1.)
        # False positives on GT-negative pixels within a selected plane count.
        self.assertEqual(regions['lower_terminal']['fp'], 1)
        self.assertEqual(regions['upper_terminal']['fp'], 2)
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)

    def test_outside_extent_and_internal_gaps_preserve_all_false_positives(self):
        result = terminal_regions.terminal_region_metrics(*self.asymmetric_case())
        regions = result['regions']
        for name, depths, predicted in (
            ('outside_target_planes', [0, 3, 7, 9], 7),
            ('below_extent', [0], 1), ('above_extent', [9], 3),
            ('internal_gaps', [3, 7], 3),
        ):
            with self.subTest(region=name):
                self.assert_region(regions[name], depths=depths, target=0,
                                   predicted=predicted, tp=0, fp=predicted, fn=0,
                                   dice=0., iou=0., precision=0., recall=None)
        for key in ('target_voxels', 'predicted_voxels', 'tp', 'fp', 'fn'):
            self.assertEqual(regions['whole_volume'][key], sum(regions[name][key] for name in
                             ('terminal_union', 'occupied_interior', 'outside_target_planes')))
            self.assertEqual(regions['outside_target_planes'][key], sum(regions[name][key] for name in
                             ('below_extent', 'above_extent', 'internal_gaps')))

    def test_twenty_percent_rounding_at_zero_one_two_five_and_six_planes(self):
        # Expected depths are explicit to avoid reimplementing the ceil rule.
        expectations = {
            0: (0, [], [], [], []),
            1: (1, [1], [1], [1], []),
            2: (1, [1], [2], [], []),
            5: (1, [1], [5], [], [2, 3, 4]),
            6: (2, [1, 2], [5, 6], [], [3, 4]),
        }
        for n, (width, lower, upper, overlap, interior) in expectations.items():
            with self.subTest(occupied_planes=n):
                target = torch.zeros((n + 2, 2, 2), dtype=torch.long)
                target[1:n + 1, 0, 0] = 5
                result = terminal_regions.terminal_region_metrics(target.clone(), target)
                self.assertEqual(result['planes_per_end'], width)
                self.assertEqual(result['regions']['lower_terminal']['depth_indices'], lower)
                self.assertEqual(result['regions']['upper_terminal']['depth_indices'], upper)
                self.assertEqual(result['regions']['occupied_interior']['depth_indices'], interior)
                self.assertEqual(result['overlap_depths'], overlap)
                union = result['regions']['terminal_union']
                expected_union = sorted(set(lower + upper))
                self.assertEqual(union['depth_indices'], expected_union)
                self.assertEqual(union['tp'], len(expected_union))
                self.assertEqual(union['target_voxels'], len(expected_union))
                self.assertEqual(result['target_empty'], n == 0)

    def test_empty_target_has_no_invented_extent_and_retains_whole_volume_fp(self):
        target = torch.zeros((4, 2, 3), dtype=torch.long)
        prediction = target.clone()
        prediction[0, 0, 0] = prediction[2, 1, 2] = 5
        result = terminal_regions.terminal_region_metrics(prediction, target)
        self.assertTrue(result['target_empty'])
        self.assertFalse(result['extent_defined'])
        self.assertEqual(result['occupied_depths'], [])
        self.assertEqual(result['overlap_depths'], [])
        for name, region in result['regions'].items():
            if name in ('whole_volume', 'outside_target_planes'):
                self.assert_region(region, depths=[0, 1, 2, 3], target=0, predicted=2,
                                   tp=0, fp=2, fn=0, dice=0., iou=0., precision=0., recall=None)
            else:
                self.assert_region(region, depths=[], target=0, predicted=0, tp=0, fp=0, fn=0,
                                   dice=None, iou=None, precision=None, recall=None)
        both_empty = terminal_regions.terminal_region_metrics(target, target)['regions']['whole_volume']
        self.assert_region(both_empty, depths=[0, 1, 2, 3], target=0, predicted=0, tp=0, fp=0,
                           fn=0, dice=None, iou=None, precision=None, recall=None)

    def aggregation_cases(self):
        def case(case_id, patient_id, target_present, prediction_present):
            target = torch.zeros((3, 2, 2), dtype=torch.long)
            prediction = target.clone()
            if target_present:
                target[1, 0, 0] = 5
            if prediction_present:
                prediction[1, 0, 0] = 5
            return dict(case_id=case_id, patient_id=patient_id,
                        terminal_regions=terminal_regions.terminal_region_metrics(prediction, target))
        return [case('a1', 'A', True, True), case('a2', 'A', True, True),
                case('b1', 'B', True, False), case('b2', 'B', False, False),
                case('c1', 'C', False, True)]

    def test_equal_patient_macro_differs_from_case_mean_and_excludes_undefined_scores(self):
        cases = self.aggregation_cases()
        original = copy.deepcopy(cases)
        result = terminal_regions.summarize_terminal_regions(cases)
        self.assertEqual(cases, original)
        self.assertEqual(result['num_cases'], 5)
        self.assertEqual(result['num_patients'], 3)
        whole = result['regions']['whole_volume']
        self.assertEqual(whole['valid_case_count'], dict(dice=4, iou=4, precision=3, recall=3))
        self.assertEqual(whole['valid_patient_count'], dict(dice=3, iou=3, precision=2, recall=2))
        for metric in ('dice', 'iou'):
            self.assertAlmostEqual(whole['case_mean'][metric], .5)
            self.assertAlmostEqual(whole['patient_macro'][metric], 1 / 3)
        for metric in ('precision', 'recall'):
            self.assertAlmostEqual(whole['case_mean'][metric], 2 / 3)
            self.assertAlmostEqual(whole['patient_macro'][metric], .5)
        self.assertEqual(whole['per_patient']['A']['num_cases'], 2)
        self.assertEqual(whole['per_patient']['B']['num_cases'], 2)
        self.assertEqual(whole['per_patient']['B']['valid_case_count']['dice'], 1)
        self.assertEqual(whole['per_patient']['B']['mean']['dice'], 0.)
        self.assertIsNone(whole['per_patient']['B']['mean']['precision'])
        self.assertIsNone(whole['per_patient']['C']['mean']['recall'])
        self.assertEqual(whole['empty_target_cases'], 2)
        self.assertEqual(whole['empty_prediction_cases'], 2)
        self.assertEqual(whole['positive_fp_cases'], 1)
        self.assertEqual(whole['pooled_counts'], dict(tp=2, fp=1, fn=1,
                                                    target_voxels=3, predicted_voxels=3))
        terminal = result['regions']['terminal_union']
        self.assertAlmostEqual(terminal['case_mean']['dice'], 2 / 3)
        self.assertAlmostEqual(terminal['patient_macro']['dice'], .5)
        self.assertEqual(terminal['positive_fp_cases'], 0)
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)

    def test_all_undefined_region_scores_keep_null_means_and_zero_valid_counts(self):
        empty = torch.zeros((3, 2, 2), dtype=torch.long)
        cases = [dict(case_id='empty', patient_id='patient',
                      terminal_regions=terminal_regions.terminal_region_metrics(empty, empty))]
        summary = terminal_regions.summarize_terminal_regions(cases)
        for region in summary['regions'].values():
            self.assertEqual(region['valid_case_count'], dict(dice=0, iou=0, precision=0, recall=0))
            self.assertEqual(region['valid_patient_count'], dict(dice=0, iou=0, precision=0, recall=0))
            self.assertTrue(all(value is None for value in region['case_mean'].values()))
            self.assertTrue(all(value is None for value in region['patient_macro'].values()))
            self.assertEqual(region['empty_target_cases'], 1)
            self.assertEqual(region['empty_prediction_cases'], 1)
            self.assertEqual(region['positive_fp_cases'], 0)

    def test_invalid_masks_are_rejected_without_modifying_valid_inputs(self):
        valid = torch.zeros((3, 2, 2), dtype=torch.long)
        original = valid.clone()
        invalid = [valid.float(), valid.int(), valid.bool(), valid[0], valid.unsqueeze(0),
                   valid[:0], torch.zeros((3, 2, 3), dtype=torch.long),
                   torch.full_like(valid, -1), torch.full_like(valid, 6), valid.tolist()]
        for index, bad in enumerate(invalid):
            for side in ('prediction', 'target'):
                with self.subTest(index=index, side=side), self.assertRaises((ValueError, TypeError)):
                    terminal_regions.terminal_region_metrics(
                        bad if side == 'prediction' else valid, bad if side == 'target' else valid)
        torch.testing.assert_close(valid, original, rtol=0, atol=0)
        with self.assertRaises((ValueError, TypeError)):
            terminal_regions.terminal_region_metrics(valid.to('meta'), valid.to('meta'))

    def test_summary_rejects_duplicate_cases_and_inconsistent_scores_or_counts(self):
        cases = self.aggregation_cases()
        duplicated = [cases[0], {**cases[0], 'patient_id': 'another_patient'}]
        with self.assertRaises(ValueError):
            terminal_regions.summarize_terminal_regions(duplicated)
        for problem in ('nan', 'range', 'inconsistent_score', 'null_valid', 'negative_count',
                        'inconsistent_count', 'missing_region', 'empty_case_id', 'empty_patient_id'):
            with self.subTest(problem=problem):
                corrupted = copy.deepcopy(cases)
                region = corrupted[0]['terminal_regions']['regions']['whole_volume']
                if problem == 'nan':
                    region['dice'] = float('nan')
                elif problem == 'range':
                    region['iou'] = 1.01
                elif problem == 'inconsistent_score':
                    region['dice'] = .5
                elif problem == 'null_valid':
                    region['dice'] = None
                elif problem == 'negative_count':
                    region['fp'] = -1
                elif problem == 'inconsistent_count':
                    region['target_voxels'] += 1
                elif problem == 'missing_region':
                    del corrupted[0]['terminal_regions']['regions']['internal_gaps']
                elif problem == 'empty_case_id':
                    corrupted[0]['case_id'] = ''
                else:
                    corrupted[0]['patient_id'] = ''
                with self.assertRaises(ValueError):
                    terminal_regions.summarize_terminal_regions(corrupted)


if __name__ == '__main__':
    unittest.main()
