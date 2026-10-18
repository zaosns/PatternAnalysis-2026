"""Optional terminal diagnostics integrate with real 2D checkpoint prediction."""

from contextlib import redirect_stdout
import hashlib
import io
from pathlib import Path
import unittest

import predict
import terminal_regions
from tests import test_predict_baseline2d as baseline_fixtures


class TerminalPredictionIntegrationTests(unittest.TestCase):
    def setUp(self):
        # Compose the established synthetic NIfTI fixture rather than inheriting
        # its eight tests. Its cleanup also restores RNG, threads and policy.
        self.fixture = baseline_fixtures.Baseline2DPredictionTests()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()
        self.fixture.train_fixture()

    def test_default_prediction_omits_terminal_protocol_and_case_fields(self):
        with redirect_stdout(io.StringIO()):
            report = predict.run_prediction(self.fixture.arguments(self.fixture.root / 'default'))
        for key in ('terminal_region_protocol', 'terminal_region_summary'):
            self.assertNotIn(key, report)
        self.assertNotIn('terminal_regions.py', report['source_sha256'])
        self.assertTrue(all('terminal_regions' not in row for row in report['cases']))

    def test_enabled_prediction_records_protocol_source_summary_and_matching_whole_counts(self):
        with redirect_stdout(io.StringIO()):
            report = predict.run_prediction(self.fixture.arguments(self.fixture.root / 'terminal',
                extra=('--terminal-regions', '--panel-case-ids', 'validation_a')))
        self.assertEqual(report['terminal_region_protocol'], terminal_regions.PROTOCOL)
        self.assertEqual(report['terminal_region_summary'],
                         terminal_regions.summarize_terminal_regions(report['cases']))
        source = Path(terminal_regions.__file__)
        self.assertEqual(report['source_sha256']['terminal_regions.py'],
                         hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertEqual(len(report['cases']), 2)
        self.fixture.render.assert_called_once()
        for row in report['cases']:
            result = row['terminal_regions']
            self.assertEqual(result['shape_dhw'], row['shape_dhw'])
            whole = result['regions']['whole_volume']
            original = row['metrics']['voxel_counts']
            for key, source_key in (('tp', 'true_positive'), ('fp', 'false_positive'),
                                    ('fn', 'false_negative'), ('target_voxels', 'target_voxels'),
                                    ('predicted_voxels', 'predicted_voxels')):
                self.assertEqual(whole[key], original[source_key][5])
            for key in ('dice', 'iou'):
                expected = row['metrics'][key][4]
                if expected is None:
                    self.assertIsNone(whole[key])
                else:
                    self.assertAlmostEqual(whole[key], expected, places=6)


if __name__ == '__main__':
    unittest.main()
