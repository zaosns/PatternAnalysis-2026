"""Boundary checks for inference measurement; mocked CUDA is not a GPU benchmark."""

from contextlib import ExitStack, nullcontext
import copy
import statistics
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import torch

import benchmark_inference as benchmark


class InferenceBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
        rng = torch.random.fork_rng(devices=devices)
        rng.__enter__()
        self.addCleanup(rng.__exit__, None, None, None)

    def controlled_benchmark(self, *, nonfinite_forward=None):
        """Run the actual loop with CPU tensors and independently traced CUDA stubs.

        Plain replacement functions intentionally avoid Mock's argument history:
        retaining a logits tensor in a mock would invalidate the lifetime check.
        """
        trace, references, observed = [], [], []
        tensor_forward = {}
        owner = self

        class RecordingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1., dtype=torch.float32))

            def forward(self, image):
                number = len(references) + 1
                owner.assertTrue(all(reference() is None for reference in references),
                                 'A previous forward output remained live at the next forward.')
                observed.append((self.training, torch.is_grad_enabled()))
                trace.append(('forward', number))
                logits = torch.zeros((1, 6, *image.shape[2:])) + self.weight
                logits[:, 5] = 2.
                if number == nonfinite_forward:
                    logits[:, 0, 0, 0, 0] = float('nan')
                references.append(weakref.ref(logits))
                tensor_forward[id(logits)] = number
                return logits

        model = RecordingModel().train()
        prepared = SimpleNamespace(image=torch.ones((1, 3, 4, 5)),
                                   target=torch.zeros((3, 4, 5), dtype=torch.long))

        def prepare(current_model, current_prepared):
            trace.append(('prepare', 0))
            current_model.eval()
            return current_prepared.image.unsqueeze(0), torch.device('cuda:0'), .75

        def validate(logits, image):
            trace.append(('validate', tensor_forward[id(logits)]))
            owner.assertEqual(logits.shape, (1, 6, *image.shape[2:]))
            owner.assertEqual(logits.dtype, torch.float32)

        original_finite, original_argmax = torch.isfinite, torch.Tensor.argmax

        def finite(tensor):
            if id(tensor) in tensor_forward:
                trace.append(('finite', tensor_forward[id(tensor)]))
            return original_finite(tensor)

        def argmax(tensor, *args, **kwargs):
            if id(tensor) in tensor_forward:
                trace.append(('argmax', tensor_forward[id(tensor)]))
            return original_argmax(tensor, *args, **kwargs)

        clock_values = iter((10., 10.125, 20., 20.25, 30., 30.375))
        event_durations_ms = iter((12., 24., 36.))

        def clock():
            value = next(clock_values)
            trace.append(('clock', value))
            return value

        class Event:
            def __init__(self, *args, **kwargs):
                trace.append(('event_create', len(references)))

            def record(self, *args, **kwargs):
                trace.append(('event_record', len(references)))

            def elapsed_time(self, other):
                trace.append(('event_elapsed', len(references)))
                return next(event_durations_ms)

        def synchronize(*args, **kwargs):
            trace.append(('synchronize', len(references)))

        def reset_peak(*args, **kwargs):
            trace.append(('reset_peak', len(references)))

        def allocated(*args, **kwargs):
            trace.append(('allocated', len(references)))
            return 200

        def reserved(*args, **kwargs):
            trace.append(('reserved', len(references)))
            return 300

        def peak_allocated(*args, **kwargs):
            trace.append(('peak_allocated', len(references)))
            return 1000 + 10 * len(references)

        def peak_reserved(*args, **kwargs):
            trace.append(('peak_reserved', len(references)))
            return 2000 + 10 * len(references)

        with ExitStack() as stack:
            replacements = (
                (benchmark, '_prepare_cuda_input', prepare),
                (benchmark, '_validate_logits', validate),
                (benchmark.time, 'perf_counter', clock),
                (torch, 'isfinite', finite), (torch.Tensor, 'argmax', argmax),
                (torch.cuda, 'Event', Event),
                (torch.cuda, 'device', lambda *args, **kwargs: nullcontext()),
                (torch.cuda, 'synchronize', synchronize),
                (torch.cuda, 'reset_peak_memory_stats', reset_peak),
                (torch.cuda, 'memory_allocated', allocated),
                (torch.cuda, 'memory_reserved', reserved),
                (torch.cuda, 'max_memory_allocated', peak_allocated),
                (torch.cuda, 'max_memory_reserved', peak_reserved),
            )
            for target, name, replacement in replacements:
                stack.enter_context(patch.object(target, name, new=replacement))
            if nonfinite_forward is not None:
                with self.assertRaises((ValueError, RuntimeError)):
                    benchmark.benchmark_prepared(model, prepared, warmups=2, repeats=3)
                return trace, references, observed
            prediction, timing = benchmark.benchmark_prepared(model, prepared, warmups=2, repeats=3)
        return prediction, timing, trace, references, observed

    def test_warmup_and_forward_timing_exclude_transfer_and_dispose_previous_logits(self):
        prediction, timing, trace, references, observed = self.controlled_benchmark()
        self.assertEqual(observed, [(False, False)] * 5)
        self.assertEqual(prediction.shape, (3, 4, 5))
        self.assertEqual(prediction.dtype, torch.long)
        self.assertEqual(prediction.device.type, 'cpu')
        self.assertTrue(torch.all(prediction == 5))
        self.assertTrue(all(reference() is None for reference in references))
        self.assertEqual(timing['warmup_forwards'], 2)
        self.assertEqual(timing['measured_forwards'], 3)
        self.assertEqual(timing['transfer_seconds'], .75)
        self.assertEqual(timing['wall_seconds'], [.125, .25, .375])
        self.assertEqual(timing['cuda_event_seconds'], [.012, .024, .036])
        self.assertEqual(timing['median_wall_seconds'], .25)
        self.assertEqual(timing['median_cuda_event_seconds'], .024)
        resets = [index for index, event in enumerate(trace) if event[0] == 'reset_peak']
        first_clock = next(index for index, event in enumerate(trace) if event[0] == 'clock')
        creations = [index for index, event in enumerate(trace) if event[0] == 'event_create']
        self.assertEqual(len(creations), 2)
        self.assertTrue(all(index < first_clock for index in creations))
        self.assertEqual(sum(event[0] == 'event_record' for event in trace[:first_clock]), 2)
        self.assertEqual(len(resets), 3)
        self.assertGreater(resets[0], trace.index(('forward', 2)))
        self.assertEqual([event for event in trace if event[0] == 'argmax'], [('argmax', 5)])
        self.assertEqual(timing['peak_allocated_bytes'], [1030, 1040, 1050])
        self.assertEqual(timing['peak_reserved_bytes'], [2030, 2040, 2050])
        self.assertEqual(timing['max_peak_allocated_bytes'], 1050)
        self.assertEqual(timing['max_peak_reserved_bytes'], 2050)
        self.assertEqual(timing['incremental_peak_allocated_bytes'], [830, 840, 850])
        self.assertEqual(timing['max_incremental_peak_allocated_bytes'], 850)

    def test_peak_reads_precede_logit_validation_and_postprocessing_for_every_measurement(self):
        _, timing, trace, _, _ = self.controlled_benchmark()
        clocks = [index for index, event in enumerate(trace) if event[0] == 'clock']
        for index, number in enumerate((3, 4, 5)):
            with self.subTest(measured_forward=number):
                forward = trace.index(('forward', number))
                start, stop = clocks[2 * index:2 * index + 2]
                allocated = trace.index(('peak_allocated', number))
                reserved = trace.index(('peak_reserved', number))
                validation = trace.index(('validate', number))
                finite = trace.index(('finite', number))
                self.assertLess(forward, allocated)
                self.assertLess(forward, reserved)
                self.assertLess(start, forward)
                self.assertLess(forward, stop)
                self.assertTrue(any(event[0] == 'synchronize' for event in trace[forward + 1:stop]))
                self.assertTrue(any(event[0] == 'event_record' for event in trace[start + 1:forward]))
                self.assertTrue(any(event[0] == 'event_record' for event in trace[forward + 1:stop]))
                self.assertLess(max(allocated, reserved), validation)
                self.assertLess(max(allocated, reserved), finite)
                following = trace[forward + 1:min(allocated, reserved)]
                self.assertTrue(any(event[0] == 'synchronize' for event in following))
                self.assertTrue(any(event[0] == 'clock' for event in following))
        self.assertLess(trace.index(('peak_reserved', 5)), trace.index(('argmax', 5)))
        self.assertEqual(timing['output_validation'], dict(
            finite=True, num_classes=6, shape=[1, 6, 3, 4, 5], prediction_dtype='torch.int64'))

    def test_nonfinite_measured_output_fails_after_peak_read_and_before_argmax(self):
        trace, _, observed = self.controlled_benchmark(nonfinite_forward=3)
        self.assertEqual(len(observed), 3)
        self.assertLess(trace.index(('peak_allocated', 3)), trace.index(('finite', 3)))
        self.assertLess(trace.index(('peak_reserved', 3)), trace.index(('finite', 3)))
        self.assertFalse(any(event[0] == 'argmax' for event in trace))

    def test_invalid_repeat_counts_and_unavailable_cuda_fail_before_forward(self):
        model = torch.nn.Conv3d(1, 6, 1)
        prepared = SimpleNamespace(image=torch.ones((1, 3, 4, 5)),
                                   target=torch.zeros((3, 4, 5), dtype=torch.long))
        for name in ('warmups', 'repeats'):
            for bad in (True, 0, -1, 1.5):
                with self.subTest(name=name, value=bad), \
                        patch.object(benchmark, '_prepare_cuda_input') as prepare, \
                        self.assertRaises((ValueError, TypeError)):
                    benchmark.benchmark_prepared(model, prepared, **{name: bad})
                prepare.assert_not_called()
        with patch.object(torch.cuda, 'is_available', return_value=False), \
                patch.object(model, 'forward') as forward, \
                self.assertRaises((ValueError, RuntimeError)):
            benchmark.benchmark_prepared(model, prepared, warmups=1, repeats=1)
        forward.assert_not_called()

    def test_invalid_prepared_images_are_rejected_before_transfer_or_forward(self):
        # Stub only CUDA parameter metadata so CPU-only test hosts can reach
        # the real prepared-image guard without allocating on a GPU.
        class DeviceMetadataModel(torch.nn.Module):
            def parameters(self, recurse=True):
                return iter([SimpleNamespace(dtype=torch.float32, device=torch.device('cuda:0'),
                    is_floating_point=lambda: True, is_complex=lambda: False)])

            def forward(self, image):
                raise AssertionError('Malformed image must never reach forward.')

        valid = torch.ones((1, 3, 4, 5), dtype=torch.float32)
        bad_images = [None, valid.tolist(), valid.double(), valid.long(), valid[0],
                      valid.unsqueeze(0), valid[:, :0], valid.repeat(2, 1, 1, 1),
                      torch.full_like(valid, float('nan'))]
        for index, bad in enumerate(bad_images):
            with self.subTest(index=index), \
                    patch.object(torch.cuda, 'is_available', return_value=True), \
                    patch.object(torch.cuda, 'synchronize') as synchronize, \
                    self.assertRaises(ValueError):
                benchmark.benchmark_prepared(DeviceMetadataModel(), SimpleNamespace(image=bad),
                                               warmups=1, repeats=1)
            synchronize.assert_not_called()

    def test_output_contract_rejects_wrong_grid_class_count_and_dtype(self):
        image = torch.ones((1, 1, 3, 4, 5), dtype=torch.float32)
        for bad in (None, torch.zeros((1, 5, 3, 4, 5)), torch.zeros((1, 6, 4, 4, 5)),
                    torch.zeros((1, 6, 3, 4, 5), dtype=torch.float64),
                    torch.zeros((1, 6, 3, 4, 5), dtype=torch.long)):
            with self.subTest(output_type=type(bad)), self.assertRaises(ValueError):
                benchmark._validate_logits(bad, image)

    @staticmethod
    def case(case_id, patient_id, durations):
        durations = list(durations)
        events = [value / 2 for value in durations]
        repeats = len(durations)
        timing = dict(warmup_forwards=2, measured_forwards=repeats, transfer_seconds=.1,
            wall_seconds=durations, cuda_event_seconds=events,
            median_wall_seconds=statistics.median(durations),
            median_cuda_event_seconds=statistics.median(events),
            baseline_allocated_bytes=100, baseline_reserved_bytes=200,
            repeat_baseline_allocated_bytes=[100] * repeats,
            repeat_baseline_reserved_bytes=[200] * repeats,
            peak_allocated_bytes=[150] * repeats, peak_reserved_bytes=[250] * repeats,
            max_peak_allocated_bytes=150, max_peak_reserved_bytes=250,
            incremental_peak_allocated_bytes=[50] * repeats, max_incremental_peak_allocated_bytes=50,
            output_validation=dict(finite=True, num_classes=6, shape=[1, 6, 3, 4, 5],
                                   prediction_dtype='torch.int64'))
        return dict(case_id=case_id, patient_id=patient_id, timing=timing)

    def test_summary_uses_case_medians_then_equal_patient_weight_and_absolute_peak(self):
        cases = [self.case('a1', 'A', [1., 1., 100.]),
                 self.case('a2', 'A', [2., 3., 4.]),
                 self.case('b1', 'B', [9., 10., 11.])]
        cases[1]['timing'].update(peak_allocated_bytes=[150, 175, 150],
            max_peak_allocated_bytes=175, incremental_peak_allocated_bytes=[50, 75, 50],
            max_incremental_peak_allocated_bytes=75)
        original = copy.deepcopy(cases)
        summary = benchmark.summarize_benchmark_cases(cases)
        self.assertEqual(cases, original)
        self.assertEqual(summary['num_cases'], 3)
        self.assertEqual(summary['num_patients'], 2)
        wall = summary['wall_seconds']
        self.assertAlmostEqual(wall['case_mean'], 14 / 3)
        self.assertEqual(wall['case_median'], 3.)
        self.assertEqual(wall['patient_macro'], 6.)
        self.assertEqual(wall['per_patient'], {'A': {'num_cases': 2, 'mean': 2.},
                                              'B': {'num_cases': 1, 'mean': 10.}})
        self.assertEqual(wall['slowest_case'], {'case_id': 'b1', 'seconds': 10.})
        events = summary['event_seconds']
        self.assertAlmostEqual(events['case_mean'], 7 / 3)
        self.assertEqual(events['patient_macro'], 3.)
        self.assertEqual(summary['max_peak_allocated_bytes'], 175)
        self.assertEqual(summary['max_peak_reserved_bytes'], 250)
        self.assertEqual(summary['max_incremental_peak_allocated_bytes'], 75)

    def test_summary_rejects_invalid_times_and_duplicate_case_records(self):
        case = self.case('case', 'patient', [1., 2., 3.])
        with self.assertRaises(ValueError):
            benchmark.summarize_benchmark_cases([case, copy.deepcopy(case)])
        for key in ('wall_seconds', 'cuda_event_seconds'):
            for value in (float('nan'), float('inf'), -1., True):
                with self.subTest(key=key, value=value):
                    corrupted = copy.deepcopy(case)
                    corrupted['timing'][key][0] = value
                    with self.assertRaises(ValueError):
                        benchmark.summarize_benchmark_cases([corrupted])
        for key, value in (('measured_forwards', 99), ('median_wall_seconds', 9.),
                           ('max_peak_allocated_bytes', 999), ('transfer_seconds', float('nan'))):
            with self.subTest(key=key):
                corrupted = copy.deepcopy(case)
                corrupted['timing'][key] = value
                with self.assertRaises(ValueError):
                    benchmark.summarize_benchmark_cases([corrupted])
        with self.assertRaises(ValueError):
            benchmark.summarize_benchmark_cases([])


if __name__ == '__main__':
    unittest.main()
