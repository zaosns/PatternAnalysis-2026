"""Synchronized CUDA FP32 forward benchmarks for prepared HipMRI volumes.

Loading, preprocessing, transfer, finite checks and argmax are outside forward
timing. Memory peaks are read before output postprocessing. Reserved memory
includes allocator caches; absolute allocated peaks include the model/input and
any other live CUDA allocations in the worker. Use an otherwise isolated worker
for each model. The caller controls GPU, TF32 policy and process-level settings.
"""

import math
import statistics
import time

import torch


def _positive_int(value):
    return type(value) is int and value > 0


def _prepare_cuda_input(model, prepared):
    """Validate FP32/CUDA prerequisites and transfer one case outside timing."""
    if not torch.cuda.is_available():
        raise ValueError('CUDA is required for this benchmark; CPU timing is not a substitute.')
    if not isinstance(model, torch.nn.Module):
        raise ValueError('model must be a torch.nn.Module on one CUDA device.')
    parameters = list(model.parameters())
    if not parameters or any(parameter.dtype != torch.float32 for parameter in parameters):
        raise ValueError('The benchmark model must have FP32 parameters.')
    device = parameters[0].device
    tensors = [*parameters, *model.buffers()]
    if (device.type != 'cuda' or any(tensor.device != device for tensor in tensors)
            or any(tensor.is_floating_point() and tensor.dtype != torch.float32 for tensor in tensors)
            or any(tensor.is_complex() for tensor in tensors)):
        raise ValueError('All model tensors must share one CUDA device, with floating tensors in FP32.')
    image = getattr(prepared, 'image', None)
    if (not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[0] != 1
            or image.numel() == 0 or image.dtype != torch.float32
            or image.device.type not in ('cpu', 'cuda')):
        raise ValueError('prepared.image must be a nonempty FP32 CPU/CUDA [1,D,H,W] tensor.')
    if not torch.isfinite(image).all().item():
        raise ValueError('Prepared image contains nonfinite values.')
    image = image.unsqueeze(0)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    image = image.to(device=device, non_blocking=False)
    torch.cuda.synchronize(device)
    transfer_seconds = time.perf_counter() - started
    return image, device, transfer_seconds


def _validate_logits(logits, image):
    if (not isinstance(logits, torch.Tensor) or logits.device != image.device
            or logits.dtype != torch.float32 or tuple(logits.shape) != (1, 6, *image.shape[2:])):
        raise ValueError('Forward must return FP32 CUDA logits [1,6,D,H,W] on the prepared grid.')


@torch.inference_mode()
def benchmark_prepared(model, prepared, *, warmups=5, repeats=10):
    """Return the final CPU prediction and raw CUDA forward measurements.

    The model is set to evaluation mode; autocast is disabled. Each measured
    forward has synchronized host-wall and CUDA-event timing. Wall time includes
    event recording, Python dispatch and final CUDA synchronization; event time
    measures the current-stream CUDA interval. Every old logits tensor is
    released before the next forward. No allocator cache is emptied.

    Each repeat records its own live-memory baseline, resets peaks immediately
    before forwarding, and captures peaks before finite checks or final argmax.
    Incremental allocated peaks subtract that repeat's baseline. Absolute
    allocated peaks retain the model/input baseline for resource-limit checks.
    """
    if not _positive_int(warmups) or not _positive_int(repeats):
        raise ValueError('warmups and repeats must be positive integers.')
    image, device, transfer_seconds = _prepare_cuda_input(model, prepared)
    model.eval()
    wall_seconds, event_seconds = [], []
    baseline_allocated, baseline_reserved = [], []
    peak_allocated, peak_reserved = [], []
    with torch.cuda.device(device), torch.autocast(device_type='cuda', enabled=False):
        for _ in range(warmups):
            logits = model(image)
            _validate_logits(logits, image)
            torch.cuda.synchronize(device)
            if not torch.isfinite(logits).all().item():
                raise ValueError('Warmup logits contain nonfinite values.')
            del logits
        # CUDA events initialize lazily on first record. Initialize one pair
        # outside the wall-time windows, then reuse it after synchronization.
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        end_event.record()
        torch.cuda.synchronize(device)
        initial_allocated = int(torch.cuda.memory_allocated(device))
        initial_reserved = int(torch.cuda.memory_reserved(device))

        for repeat in range(repeats):
            # The preceding output has been deleted, and the previous event
            # interval has completed before this pair is recorded again.
            torch.cuda.synchronize(device)
            baseline_allocated.append(int(torch.cuda.memory_allocated(device)))
            baseline_reserved.append(int(torch.cuda.memory_reserved(device)))
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            start_event.record()
            logits = model(image)
            end_event.record()
            torch.cuda.synchronize(device)
            wall_seconds.append(time.perf_counter() - started)
            event_seconds.append(start_event.elapsed_time(end_event) / 1000.)
            peak_allocated.append(int(torch.cuda.max_memory_allocated(device)))
            peak_reserved.append(int(torch.cuda.max_memory_reserved(device)))

            # Postprocessing can allocate CUDA tensors; it must happen only
            # after both forward peaks have been captured.
            _validate_logits(logits, image)
            if not torch.isfinite(logits).all().item():
                raise ValueError('Measured forward logits contain nonfinite values.')
            if repeat == repeats - 1:
                prediction = logits.argmax(dim=1)[0].cpu()
            del logits

    incremental = [peak - baseline for peak, baseline in zip(peak_allocated, baseline_allocated)]
    timing = dict(warmup_forwards=warmups, measured_forwards=repeats,
        transfer_seconds=transfer_seconds, wall_seconds=wall_seconds,
        cuda_event_seconds=event_seconds,
        median_wall_seconds=statistics.median(wall_seconds),
        median_cuda_event_seconds=statistics.median(event_seconds),
        baseline_allocated_bytes=initial_allocated, baseline_reserved_bytes=initial_reserved,
        repeat_baseline_allocated_bytes=baseline_allocated, repeat_baseline_reserved_bytes=baseline_reserved,
        peak_allocated_bytes=peak_allocated, peak_reserved_bytes=peak_reserved,
        max_peak_allocated_bytes=max(peak_allocated), max_peak_reserved_bytes=max(peak_reserved),
        incremental_peak_allocated_bytes=incremental, max_incremental_peak_allocated_bytes=max(incremental),
        output_validation=dict(finite=True, num_classes=6, shape=[1, 6, *image.shape[2:]],
                               prediction_dtype=str(prediction.dtype)))
    _validate_timing(timing)
    return prediction, timing


def _finite_nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _byte_count(value):
    return type(value) is int and value >= 0


def _validate_timing(timing):
    required = {
        'warmup_forwards', 'measured_forwards', 'transfer_seconds', 'wall_seconds',
        'cuda_event_seconds', 'median_wall_seconds', 'median_cuda_event_seconds',
        'baseline_allocated_bytes', 'baseline_reserved_bytes',
        'repeat_baseline_allocated_bytes', 'repeat_baseline_reserved_bytes',
        'peak_allocated_bytes', 'peak_reserved_bytes', 'max_peak_allocated_bytes',
        'max_peak_reserved_bytes', 'incremental_peak_allocated_bytes',
        'max_incremental_peak_allocated_bytes', 'output_validation',
    }
    if not isinstance(timing, dict) or not required.issubset(timing):
        raise ValueError('Benchmark timing is missing required measurements.')
    if not _positive_int(timing['warmup_forwards']) or not _positive_int(timing['measured_forwards']):
        raise ValueError('Benchmark forward counts must be positive integers.')
    repeats = timing['measured_forwards']
    for name in ('wall_seconds', 'cuda_event_seconds'):
        values = timing[name]
        median_key = 'median_wall_seconds' if name == 'wall_seconds' else 'median_cuda_event_seconds'
        if (not isinstance(values, list) or len(values) != repeats
                or not all(_finite_nonnegative(value) for value in values)
                or not _finite_nonnegative(timing[median_key])
                or not math.isclose(timing[median_key], statistics.median(values), rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError(f'Incoherent forward durations or median: {name}')
    if not _finite_nonnegative(timing['transfer_seconds']):
        raise ValueError('Transfer duration must be finite and nonnegative.')
    arrays = ('repeat_baseline_allocated_bytes', 'repeat_baseline_reserved_bytes',
              'peak_allocated_bytes', 'peak_reserved_bytes', 'incremental_peak_allocated_bytes')
    for name in arrays:
        if (not isinstance(timing[name], list) or len(timing[name]) != repeats
                or not all(_byte_count(value) for value in timing[name])):
            raise ValueError(f'Invalid per-repeat byte counts: {name}')
    for kind in ('allocated', 'reserved'):
        baseline = timing[f'baseline_{kind}_bytes']
        maximum = timing[f'max_peak_{kind}_bytes']
        if (not _byte_count(baseline) or baseline != timing[f'repeat_baseline_{kind}_bytes'][0]
                or not _byte_count(maximum) or maximum != max(timing[f'peak_{kind}_bytes'])):
            raise ValueError(f'Incoherent {kind} baseline or peak.')
    for allocated, reserved, peak_a, peak_r, increment in zip(*(timing[name] for name in arrays)):
        if (reserved < allocated or peak_a < allocated or peak_r < reserved or peak_r < peak_a
                or increment != peak_a - allocated):
            raise ValueError('Per-repeat CUDA memory peaks/baselines are inconsistent.')
    increment = timing['max_incremental_peak_allocated_bytes']
    if not _byte_count(increment) or increment != max(timing['incremental_peak_allocated_bytes']):
        raise ValueError('Incremental allocated maximum is inconsistent.')
    validation = timing['output_validation']
    if (not isinstance(validation, dict) or validation.get('finite') is not True
            or type(validation.get('num_classes')) is not int or validation['num_classes'] != 6
            or validation.get('prediction_dtype') != 'torch.int64'
            or not isinstance(validation.get('shape'), list) or len(validation['shape']) != 5
            or not all(_positive_int(value) for value in validation['shape'])
            or validation['shape'][:2] != [1, 6]):
        raise ValueError('Benchmark output validation is missing or invalid.')


def summarize_benchmark_cases(cases):
    """Summarize case-median latencies, keeping patient weighting explicit.

    Input records contain case_id, patient_id and timing returned above. Case
    medians, rather than raw repetitions, are the unit of cohort summaries.
    patient_macro averages case medians within each patient, then weights those
    patient means equally. No per-case raw timings or memory records are removed
    or overwritten by this function; callers retain them alongside the summary.
    """
    if not isinstance(cases, list) or not cases:
        raise ValueError('Provide a nonempty list of benchmark case records.')
    seen = set()
    patients = {}
    for case in cases:
        if (not isinstance(case, dict) or any(key not in case for key in ('case_id', 'patient_id', 'timing'))
                or any(not isinstance(case[key], str) or not case[key].strip() for key in ('case_id', 'patient_id'))):
            raise ValueError('Benchmark cases need nonempty case_id/patient_id strings and timing.')
        if case['case_id'] in seen:
            raise ValueError('Benchmark case IDs must be unique.')
        seen.add(case['case_id'])
        _validate_timing(case['timing'])
        patients.setdefault(case['patient_id'], []).append(case)

    result = dict(num_cases=len(cases), num_patients=len(patients))
    for label, key in (('wall_seconds', 'median_wall_seconds'), ('event_seconds', 'median_cuda_event_seconds')):
        values = [case['timing'][key] for case in cases]
        per_patient = {patient_id: dict(num_cases=len(records),
            mean=statistics.fmean(case['timing'][key] for case in records))
            for patient_id, records in sorted(patients.items())}
        slowest = max(cases, key=lambda case: case['timing'][key])
        result[label] = dict(case_mean=statistics.fmean(values), case_median=statistics.median(values),
            patient_macro=statistics.fmean(row['mean'] for row in per_patient.values()),
            per_patient=per_patient,
            slowest_case=dict(case_id=slowest['case_id'], seconds=slowest['timing'][key]))
    for name in ('max_peak_allocated_bytes', 'max_peak_reserved_bytes', 'max_incremental_peak_allocated_bytes'):
        result[name] = max(case['timing'][name] for case in cases)
    return result
