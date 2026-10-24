"""Exercise documented training/test commands on tiny private synthetic volumes."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import nibabel as nib

import evaluate
import predict
import train
import train_baseline2d
from tests import test_evaluate as fixtures


class EvaluationEntrypointTests(unittest.TestCase):
    # Reuse the established tiny training fixture without inheriting its tests.
    train_fixture = fixtures.FrozenEvaluationTests.train_fixture
    sha256 = staticmethod(fixtures.FrozenEvaluationTests.sha256)
    write_plan = fixtures.FrozenEvaluationTests.write_plan
    freeze = fixtures.FrozenEvaluationTests.freeze
    arguments = staticmethod(fixtures.FrozenEvaluationTests.arguments)
    run_plan = fixtures.FrozenEvaluationTests.run_plan

    def setUp(self):
        fixtures.FrozenEvaluationTests.setUp(self)
        # Two test cases prove that one example panel does not subset metrics.
        records = json.loads(self.manifest.read_text())
        source = next(row for row in records if row['case_id'] == 'test')
        extra = dict(source, case_id='second_test', patient_id='second_test_patient')
        for key in ('image_path', 'label_path'):
            extra[key] = 'second_' + source[key]
            (self.root / extra[key]).write_bytes((self.root / source[key]).read_bytes())
        records.append(extra)
        self.manifest.write_text(json.dumps(records))
        split = json.loads(self.split.read_text())
        split.update(test_patients=2, test=['test', 'second_test'],
                     manifest_sha256=self.sha256(self.manifest))
        self.split.write_text(json.dumps(split))
        self.project = Path(evaluate.__file__).resolve().parent

    def script(self, name, *arguments):
        completed = subprocess.run(
            [sys.executable, '-s', str(self.project / name), *map(str, arguments)],
            cwd=self.project, capture_output=True, text=True, timeout=90)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        return completed

    def test_scripts_freeze_rehearse_and_test_full_cohort_with_default_paired_panels(self):
        self.train_fixture()
        original_runs = {kind: {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}
                         for kind, directory in self.run_dirs.items()}
        freeze_arguments = [
            'evaluate', 'freeze', '--run-2d', self.run_dirs['2d'],
            '--run-3d', self.run_dirs['3d'], '--manifest', self.manifest,
            '--split', self.split, '--device', 'cpu', '--threads', '1']
        validation_plan = self.root / 'validation-plan.json'
        self.script('train.py', *freeze_arguments, '--partition', 'validation',
                    '--output-plan', validation_plan)
        validation_output = self.root / 'validation-results'
        self.script('train.py', 'evaluate', 'run', '--plan', validation_plan,
                    '--plan-sha256', self.sha256(validation_plan), '--output-dir', validation_output)
        validation = json.loads((validation_output / 'evaluation.json').read_text())
        self.assertTrue(validation['eligible_for_test_plan'])
        self.assertEqual(list(validation_output.glob('*.png')), [])
        test_plan = self.root / 'test-plan.json'
        self.script('train.py', *freeze_arguments, '--partition', 'test',
                    '--validation-report', validation_output / 'evaluation.json',
                    '--output-plan', test_plan)
        output = self.root / 'test-results'
        completed = self.script('predict.py', 'test', '--plan', test_plan,
                                '--plan-sha256', self.sha256(test_plan), '--output-dir', output)
        report = json.loads((output / 'evaluation.json').read_text())
        printed = json.loads(completed.stdout.strip().splitlines()[-1])
        self.assertEqual(printed['accuracy_role'], 'cpu_diagnostic')
        self.assertEqual(printed['metrics'], {
            kind: result['summary'] for kind, result in report['models'].items()})
        self.assertEqual(report['status'], 'completed')
        self.assertEqual(report['partition'], 'test')
        self.assertEqual(report['training_updates'], 0)
        self.assertEqual(report['panel_case_ids'], ['test'])
        self.assertIn('prediction_panels.py', report['plan']['source_sha256'])
        self.assertEqual(len(list(output.glob('*.png'))), 2)
        for kind, result in report['models'].items():
            self.assertEqual(result['summary']['num_cases'], 2)
            self.assertEqual(result['summary']['num_patients'], 2)
            self.assertEqual([row['case_id'] for row in result['cases']], ['test', 'second_test'])
            first, second = result['cases']
            self.assertNotIn('figure', second)
            self.assertEqual(first['panel']['evaluation_split'], 'test')
            self.assertNotIn('patient', first['panel']['case_label'])
            self.assertEqual(first['figure'], f'test-{kind}-001.png')
            self.assertEqual(first['figure_sha256'], self.sha256(output / first['figure']))
            self.assertTrue((output / first['figure']).read_bytes().startswith(b'\x89PNG\r\n\x1a\n'))
        self.assertEqual(original_runs, {
            kind: {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}
            for kind, directory in self.run_dirs.items()})

    def assert_test_rejected_before_data(self, path, digest, output, *extra):
        with patch.object(evaluate, 'load_nifti_case') as loader, \
                patch.object(nib, 'load') as nifti_loader, \
                redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            predict.main(['test', '--plan', str(path), '--plan-sha256', digest,
                          '--output-dir', str(output), *extra])
        self.assertEqual(error.exception.code, 2)
        loader.assert_not_called()
        nifti_loader.assert_not_called()
        self.assertFalse(output.exists())

    def completed_test_plan(self):
        self.train_fixture()
        path, digest, _ = self.freeze(name='validation.json')
        output = self.root / 'validation-results'
        self.run_plan(path, digest, output)
        return self.freeze(partition='test', validation_report=output / 'evaluation.json',
                           name='test.json')

    def test_test_entrypoint_rejects_wrong_digest_partition_and_panel_subset_before_data(self):
        self.train_fixture()
        path, digest, _ = self.freeze()
        self.assert_test_rejected_before_data(path, '0' * 64, self.root / 'bad-digest')
        self.assert_test_rejected_before_data(path, digest, self.root / 'wrong-partition')
        validation_output = self.root / 'validation-results'
        self.run_plan(path, digest, validation_output)
        path, digest, _ = self.freeze(partition='test', name='test.json',
                                     validation_report=validation_output / 'evaluation.json')
        for index, ids in enumerate((['train'], ['validation_a'], ['test', 'test'])):
            with self.subTest(ids=ids):
                self.assert_test_rejected_before_data(
                    path, digest, self.root / f'bad-panels-{index}', '--panel-case-ids', *ids)

    def test_test_entrypoint_preserves_rehearsal_checkpoint_and_renderer_source_guards(self):
        path, digest, plan = self.completed_test_plan()
        for index, bound_path in enumerate((
                Path(plan['validation_evidence']['path']),
                self.run_dirs['3d'] / self.states['3d']['best_checkpoint'])):
            original = bound_path.read_bytes()
            try:
                bound_path.write_bytes(original + b'\n')
                self.assert_test_rejected_before_data(path, digest, self.root / f'changed-{index}')
            finally:
                bound_path.write_bytes(original)
        plan['validation_evidence'] = None
        no_evidence, no_evidence_digest = self.write_plan(plan, 'no-evidence.json')
        self.assert_test_rejected_before_data(no_evidence, no_evidence_digest,
                                             self.root / 'missing-evidence')
        original_read = Path.read_bytes
        source = self.project / 'prediction_panels.py'

        def changed_renderer(candidate):
            data = original_read(candidate)
            return data + b'\n# changed renderer\n' if candidate == source else data

        with patch.object(Path, 'read_bytes', new=changed_renderer):
            self.assert_test_rejected_before_data(path, digest, self.root / 'changed-renderer')

    def test_explicit_panel_subset_keeps_all_test_metrics(self):
        path, digest, _ = self.completed_test_plan()
        output = self.root / 'chosen-panels'
        with redirect_stdout(io.StringIO()):
            result = predict.main(['test', '--plan', str(path), '--plan-sha256', digest,
                                   '--output-dir', str(output), '--panel-case-ids', 'second_test'])
        self.assertEqual(result, 0)
        report = json.loads((output / 'evaluation.json').read_text())
        self.assertEqual(report['panel_case_ids'], ['second_test'])
        for kind, model in report['models'].items():
            self.assertEqual(model['summary']['num_cases'], 2)
            self.assertNotIn('figure', model['cases'][0])
            self.assertEqual(model['cases'][1]['figure'], f'test-{kind}-002.png')

    def test_original_prediction_command_still_rejects_test_case_ids(self):
        self.train_fixture()
        output = self.root / 'validation-command-with-test-case'
        with patch.object(predict, 'load_nifti_case') as loader, \
                patch.object(nib, 'load') as nifti_loader, \
                redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            predict.main(['--run-dir', str(self.run_dirs['3d']), '--manifest', str(self.manifest),
                          '--split', str(self.split), '--case-ids', 'test',
                          '--output-dir', str(output)])
        self.assertEqual(error.exception.code, 2)
        loader.assert_not_called()
        nifti_loader.assert_not_called()
        self.assertFalse(output.exists())

    def test_pairing_rejects_different_training_augmentation_before_voxels(self):
        for kind, trainer in (('2d', train_baseline2d), ('3d', train)):
            arguments = [
                '--manifest', str(self.manifest), '--split', str(self.split),
                '--output-dir', str(self.run_dirs[kind]), '--epochs', '1',
                '--device', 'cpu', '--num-classes', '6', '--channels', '2', '3',
                '--learning-rate', '0.002', '--downsample-factor', '2',
                '--threads', '1', '--seed', '3710', '--deterministic']
            if kind == '2d':
                arguments.append('--augment')
            with redirect_stdout(io.StringIO()):
                trainer.run_training(trainer.build_parser().parse_args(arguments))
        with patch.object(evaluate, 'load_nifti_case') as loader, \
                patch.object(nib, 'load') as nifti_loader, \
                self.assertRaisesRegex(ValueError, 'augmentation settings differ'):
            self.freeze()
        loader.assert_not_called()
        nifti_loader.assert_not_called()


if __name__ == '__main__':
    unittest.main()
