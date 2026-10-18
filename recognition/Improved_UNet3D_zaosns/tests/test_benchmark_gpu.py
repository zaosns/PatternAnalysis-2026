"""Benchmark input isolation and GPU telemetry parsing, without GPU timing claims."""

import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import benchmark_gpu
from predict import sha256
from tests import test_predict_baseline2d as fixtures


class BenchmarkInputTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.Baseline2DPredictionTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.train_fixture()
        state = json.loads((self.fixture.run_dir / 'run.json').read_text())
        self.args = SimpleNamespace(run_dir=self.fixture.run_dir,
            run_json_sha256=sha256(self.fixture.run_dir / 'run.json'),
            checkpoint_sha256=sha256(self.fixture.run_dir / state['best_checkpoint']),
            manifest=self.fixture.manifest, split=self.fixture.split,
            case_ids=state['history'][state['best_epoch'] - 1]['validation']['case_ids'])

    def test_preflight_binds_complete_validation_without_reading_voxels_or_cuda(self):
        with (patch.object(benchmark_gpu, 'load_nifti_case', side_effect=AssertionError('Voxel read')),
              patch.object(benchmark_gpu.torch.cuda, 'is_available', side_effect=AssertionError('CUDA access'))):
            state, selected = benchmark_gpu.validate_inputs(self.args)
        self.assertEqual([case.case_id for case in selected], self.args.case_ids)
        self.assertEqual(len(selected), 3)
        self.assertEqual(state['best_epoch'], 2)

    def test_partial_shuffled_train_and_test_selections_are_rejected_before_voxel_read(self):
        for selection in (self.args.case_ids[:1], list(reversed(self.args.case_ids)), ['train'], ['test']):
            with self.subTest(selection=selection), patch.object(
                    benchmark_gpu, 'load_nifti_case', side_effect=AssertionError('Voxel read')):
                args = SimpleNamespace(**{**vars(self.args), 'case_ids': selection})
                with self.assertRaisesRegex(ValueError, 'every validation'):
                    benchmark_gpu.validate_inputs(args)

    def test_run_checkpoint_and_source_changes_are_rejected(self):
        for key in ('run_json_sha256', 'checkpoint_sha256'):
            args = SimpleNamespace(**{**vars(self.args), key: '0' * 64})
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'digest'):
                benchmark_gpu.validate_inputs(args)
        run_path = self.fixture.run_dir / 'run.json'
        state = json.loads(run_path.read_text())
        state['config']['source_sha256']['baseline2d.py'] = '0' * 64
        run_path.write_text(json.dumps(state))
        self.args.run_json_sha256 = sha256(run_path)
        with self.assertRaisesRegex(ValueError, 'Training source'):
            benchmark_gpu.validate_inputs(self.args)


class BenchmarkGpuSnapshotTests(unittest.TestCase):
    def test_bare_pytorch_uuid_matches_nvidia_prefix_for_gpu_and_process_checks(self):
        uuid = '37eceddc-2b16-fd43-25d3-5aebe19f3f41'
        gpu = f'0, GPU-{uuid}, A100, 580, P0, 40, 0, 300, 40960, 1410, 40, 250\n'
        apps = f'GPU-{uuid}, {os.getpid()}, 300\nGPU-{uuid}, -1, 100\n'
        with patch.object(benchmark_gpu.subprocess, 'run', side_effect=[
                SimpleNamespace(stdout=gpu), SimpleNamespace(stdout=apps)]):
            result = benchmark_gpu.gpu_snapshot(uuid.upper())
        self.assertEqual(result['gpu'][1], 'GPU-' + uuid)
        self.assertEqual(result['other_compute_processes_visible'], [['GPU-' + uuid, '-1', '100']])

    def test_uuid_filter_ignores_other_gpus_and_retains_own_process(self):
        gpu = '0, GPU-selected, A100, 580, P0, 40, 0, 300, 40960, 1410, 40, 250\n'
        apps = f'GPU-other, 999, 500\nGPU-selected, {os.getpid()}, 300\n'
        with patch.object(benchmark_gpu.subprocess, 'run', side_effect=[
                SimpleNamespace(stdout=gpu), SimpleNamespace(stdout=apps)]):
            result = benchmark_gpu.gpu_snapshot('GPU-selected')
        self.assertEqual(result['other_compute_processes_visible'], [])
        self.assertEqual(len(result['compute_processes']), 1)

    def test_unknown_gpu_and_foreign_process_are_not_silently_accepted(self):
        with patch.object(benchmark_gpu.subprocess, 'run', return_value=SimpleNamespace(stdout='')):
            with self.assertRaisesRegex(ValueError, 'identify'):
                benchmark_gpu.gpu_snapshot('GPU-selected')
        gpu = '0, GPU-selected, A100, 580, P0, 40, 0, 300, 40960, 1410, 40, 250\n'
        with patch.object(benchmark_gpu.subprocess, 'run', side_effect=[
                SimpleNamespace(stdout=gpu), SimpleNamespace(stdout='GPU-selected, -1, 300\n')]):
            result = benchmark_gpu.gpu_snapshot('GPU-selected')
        self.assertEqual(result['other_compute_processes_visible'], [['GPU-selected', '-1', '300']])


if __name__ == '__main__':
    unittest.main()
