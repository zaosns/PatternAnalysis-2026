"""Check augmentation isolation and exact continuation through both trainers."""

import unittest
from unittest.mock import patch

import torch

import dataset
import train
import train_baseline2d
from tests import test_runner, test_baseline2d_runner


class TrainingAugmentationIntegrationTests(unittest.TestCase):
    def fixtures(self):
        return ((train, test_runner.TrainingRunnerTests),
                (train_baseline2d, test_baseline2d_runner.Baseline2DRunnerTests))

    def test_augmented_resume_is_exact_and_policy_changes_are_rejected(self):
        for module, fixture_type in self.fixtures():
            with self.subTest(trainer=module.__name__):
                fixture = fixture_type()
                try:
                    fixture.setUp()
                    continuous, resumed = fixture.root / 'continuous', fixture.root / 'resumed'
                    options = {'extra': ('--augment', '--deterministic')}
                    expected = fixture.run_training(continuous, **options)
                    fixture.run_training(resumed, epochs=1, **options)
                    actual = fixture.run_training(resumed, epochs=2, resume=True, **options)
                    fixture.assert_nested_equal(fixture.payload(resumed, actual),
                                                fixture.payload(continuous, expected))
                    self.assertEqual(fixture.stable_history(actual), fixture.stable_history(expected))
                    self.assertEqual(actual['config']['augmentation'], {
                        'enabled': True, 'policy': dataset.training_augmentation_policy()})
                    before = fixture.snapshot(resumed)
                    with patch('dataset.load_nifti_case') as reader:
                        with self.assertRaisesRegex(ValueError, 'configuration'):
                            fixture.run_training(resumed, epochs=3, resume=True,
                                                 extra=('--deterministic',))
                        reader.assert_not_called()
                    self.assertEqual(before, fixture.snapshot(resumed))
                finally:
                    fixture.doCleanups()

    def test_only_training_batches_are_augmented_and_validation_is_unchanged(self):
        for module, fixture_type in self.fixtures():
            with self.subTest(trainer=module.__name__):
                fixture = fixture_type()
                try:
                    fixture.setUp()
                    calls, validation_batches = [], []

                    def batches(records, **kwargs):
                        calls.append(([case.case_id for case in records], kwargs))
                        for image, target in dataset.iter_case_batches(records, **kwargs):
                            if not kwargs.get('training', False):
                                validation_batches.append((image.clone(), target.clone()))
                            yield image, target

                    with patch.object(module, 'iter_case_batches', side_effect=batches):
                        fixture.run_training(fixture.root / 'augmented', epochs=1,
                                             extra=('--augment',))
                    self.assertEqual(len(calls), 2)
                    self.assertCountEqual(calls[0][0], ['train_a', 'train_b'])
                    self.assertTrue(calls[0][1]['training'])
                    self.assertTrue(calls[0][1]['augment'])
                    self.assertEqual(calls[1][0], ['validation'])
                    self.assertFalse(calls[1][1].get('augment', False))
                    records = dataset.load_manifest(fixture.manifest)
                    validation = [case for case in records if case.case_id == 'validation']
                    expected = list(dataset.iter_case_batches(validation, num_classes=(
                        3 if module is train else 6)))
                    self.assertEqual(len(validation_batches), len(expected))
                    for actual_batch, expected_batch in zip(validation_batches, expected):
                        for actual_tensor, expected_tensor in zip(actual_batch, expected_batch):
                            torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)
                    # The fixtures' test files deliberately are not valid NIfTI.
                    # Completion also proves training did not load test voxels.
                finally:
                    fixture.doCleanups()

    def test_augmentation_is_opt_in_and_explicitly_recorded_disabled(self):
        for module, fixture_type in self.fixtures():
            with self.subTest(trainer=module.__name__):
                fixture = fixture_type()
                try:
                    fixture.setUp()
                    state = fixture.run_training(fixture.root / 'plain', epochs=1)
                    self.assertEqual(state['config']['augmentation'], {'enabled': False, 'policy': None})
                finally:
                    fixture.doCleanups()


if __name__ == '__main__':
    unittest.main()
