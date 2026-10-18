"""Benchmark one frozen checkpoint on the complete validation cohort using CUDA.

Run each model/block in a fresh process on the same allocated GPU. This command
never trains, renders images or reads held-out test voxels. It reports prepared-
volume forward latency and tensor-memory peaks, not end-to-end deployment cost.
"""

import argparse
import csv
import fcntl
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import torch

from benchmark_inference import benchmark_prepared, summarize_benchmark_cases
from dataset import load_manifest, load_patient_split, load_nifti_case, preprocess_case
from objectives import dice_iou_per_class
from predict import load_best_model, metric_comparison, select_validation_cases, sha256, voxel_confusion
from train import _atomic_json, _json_metrics, _validate_run_state


def validate_inputs(args):
    """Bind code, metadata and the complete validation selection before CUDA/I/O."""
    run_dir = args.run_dir.resolve(strict=True)
    run_path = run_dir / 'run.json'
    if sha256(run_path) != args.run_json_sha256:
        raise ValueError('run.json digest differs from the frozen benchmark input.')
    def reject_constant(value):
        raise ValueError(f'Nonfinite JSON constant: {value}')

    state = json.loads(run_path.read_text(), parse_constant=reject_constant)
    config = state['config']
    _validate_run_state(state, config, run_dir)
    if not state['completed_epochs'] or config['model']['num_classes'] != 6:
        raise ValueError('Require a completed six-class HipMRI checkpoint.')
    if sha256(run_dir / state['best_checkpoint']) != args.checkpoint_sha256:
        raise ValueError('Checkpoint digest differs from the frozen benchmark input.')
    source = Path(__file__).resolve().parent
    for name, expected in config['source_sha256'].items():
        if Path(name).name != name or sha256(source / name) != expected:
            raise ValueError(f'Training source differs from the checkpoint configuration: {name}')
    for name in ('manifest', 'split'):
        if sha256(getattr(args, name)) != config[name + '_sha256']:
            raise ValueError(f'{name} differs from the frozen training input.')
    records = load_manifest(args.manifest)
    split = load_patient_split(args.split, records, manifest_sha256=config['manifest_sha256'])
    reference = state['history'][state['best_epoch'] - 1]['validation']
    expected_ids = [case.case_id for case in split.validation]
    if args.case_ids != expected_ids or reference['case_ids'] != expected_ids:
        raise ValueError('Benchmark every validation case in the original frozen order.')
    selected = select_validation_cases(records, split, config, args.case_ids)
    return state, selected


def gpu_snapshot(gpu_uuid):
    """Capture allocation context; other GPUs' compute processes are not retained."""
    def uuid_key(value):
        # PyTorch may expose the bare UUID while nvidia-smi adds "GPU-".
        # Only normalize this prefix/case; never fall back to device ordinals.
        value = str(value).strip().lower()
        return value[4:] if value.startswith('gpu-') else value

    query = ['nvidia-smi', '--query-gpu=index,uuid,name,driver_version,pstate,temperature.gpu,'
             'utilization.gpu,memory.used,memory.total,clocks.current.sm,power.draw,power.limit',
             '--format=csv,noheader,nounits']
    result = subprocess.run(query, text=True, capture_output=True, check=True)
    rows = list(csv.reader(io.StringIO(result.stdout)))
    selected = [[value.strip() for value in row] for row in rows
                if len(row) >= 2 and uuid_key(row[1]) == uuid_key(gpu_uuid)]
    if len(selected) != 1:
        raise ValueError('Could not identify the allocated CUDA device by GPU UUID.')
    apps = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                           '--format=csv,noheader,nounits'], text=True, capture_output=True, check=True)
    processes = [[value.strip() for value in row] for row in csv.reader(io.StringIO(apps.stdout))
                 if row and uuid_key(row[0]) == uuid_key(gpu_uuid)]
    foreign = [row for row in processes if len(row) < 2 or row[1] != str(os.getpid())]
    return dict(gpu_fields=query[1].split('=', 1)[1].split(','), gpu=selected[0],
                compute_process_fields=['gpu_uuid', 'pid', 'used_gpu_memory_mib'],
                compute_processes=processes, other_compute_processes_visible=foreign,
                interpretation='Point-in-time check; it does not prove absence of contention throughout the run.')


def configure_cuda(config, threads):
    policy = config['reproducibility']
    if policy['cublas_workspace_config'] is not None:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = policy['cublas_workspace_config']
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(policy['deterministic_algorithms'],
                                      warn_only=policy['deterministic_warn_only'])
    torch.backends.cudnn.deterministic = policy['cudnn_deterministic']
    torch.backends.cudnn.benchmark = False
    # Use one precision-control API consistently; do not mix with fp32_precision.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('Allocate exactly one visible CUDA GPU; there is no CPU benchmark fallback.')
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    gpu_uuid = str(properties.uuid)
    environment = dict(python=platform.python_version(), torch=str(torch.__version__),
        cuda_runtime=torch.version.cuda, cudnn_version=torch.backends.cudnn.version(),
        device_name=properties.name, gpu_uuid=gpu_uuid, total_memory_bytes=properties.total_memory,
        capability=[properties.major, properties.minor], cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        autocast=False, compiled=False, tensor_dtype='torch.float32', batch_volumes=1,
        model_mode='eval', autograd_mode='inference_mode', hostname=platform.node(), pid=os.getpid(),
        slurm={key: os.environ.get(key) for key in ('SLURM_JOB_ID', 'SLURM_JOB_GPUS',
            'SLURM_CPUS_PER_TASK', 'SLURM_JOB_PARTITION', 'SLURM_NODELIST')})
    return gpu_uuid, environment


def tensor_hash(tensor):
    import hashlib
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def run_benchmark(args):
    for name in ('threads', 'warmups', 'repeats'):
        if type(getattr(args, name)) is not int or getattr(args, name) < 1:
            raise ValueError(f'{name} must be a positive integer.')
    run_dir = args.run_dir.resolve(strict=True)
    with (run_dir / '.run.lock').open('r') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('Training is still writing this run.') from error
        return _run_locked(args, run_dir)


def _run_locked(args, run_dir):
    state, selected = validate_inputs(args)
    config = state['config']
    source = Path(__file__).resolve().parent
    source_hashes = {name: sha256(source / name) for name in set(config['source_sha256']) |
                    {'predict.py', 'benchmark_gpu.py', 'benchmark_inference.py'}}
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(format_version=1, status='started', scope='complete validation; forward-only GPU benchmark',
        run_json_sha256=args.run_json_sha256, checkpoint_sha256=args.checkpoint_sha256,
        checkpoint_name=state['best_checkpoint'], best_epoch=state['best_epoch'],
        training_config=config, source_sha256=source_hashes, selected_case_ids=args.case_ids,
        warmups=args.warmups, repeats=args.repeats, cases=[], training_updates=0, test_voxels_read=False,
        interpretation='Each case is one prepared volume. The 2D wrapper batches all depth slices; no chunking. '
                       'Float32 tensors with TF32 disabled, no AMP/compile. Not end-to-end latency or total device memory.')
    _atomic_json(output / 'benchmark.json', report)
    try:
        gpu_uuid, environment = configure_cuda(config, args.threads)
        report['environment'] = environment
        report['gpu_before'] = gpu_snapshot(gpu_uuid)
        if report['gpu_before']['other_compute_processes_visible']:
            raise ValueError('Other compute processes are visible on the allocated GPU; do not benchmark contention silently.')
        model, _ = load_best_model(run_dir, state, device='cuda', expected_sha256=args.checkpoint_sha256)
        report['parameter_count'] = sum(parameter.numel() for parameter in model.parameters())
        report['parameter_bytes'] = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
        reference = state['history'][state['best_epoch'] - 1]['validation']
        for index, case in enumerate(selected):
            started = time.perf_counter()
            loaded = load_nifti_case(case, allowed_labels=range(6),
                assume_label_units_from_image=config['assume_label_units_from_image'])
            prepared = preprocess_case(loaded, downsample_factor=config['downsample_factor'])
            del loaded
            loading_seconds = time.perf_counter() - started
            image_hash, target_hash = tensor_hash(prepared.image), tensor_hash(prepared.target)
            prediction, timing = benchmark_prepared(model, prepared, warmups=args.warmups, repeats=args.repeats)
            metrics = _json_metrics(dice_iou_per_class(prediction.unsqueeze(0), prepared.target.unsqueeze(0),
                num_classes=6, include_background=False))
            flat = {name: metrics[name][0] for name in ('dice', 'iou', 'valid')}
            flat['class_ids'] = metrics['class_ids']
            comparison = metric_comparison(flat, reference, index)
            row = dict(case_id=case.case_id, patient_id=case.patient_id, shape_dhw=list(prepared.target.shape),
                affine_dhw=prepared.affine_dhw.tolist(), prepared_image_sha256=image_hash,
                prepared_target_sha256=target_hash, prediction_sha256=tensor_hash(prediction),
                loading_preprocessing_seconds=loading_seconds, timing=timing, metrics=flat,
                voxel_counts=voxel_confusion(prediction, prepared.target), saved_metric_comparison=comparison)
            report['cases'].append(row)
            _atomic_json(output / 'benchmark.json', report)
            print(json.dumps(dict(case=index + 1, median_forward_seconds=timing['median_wall_seconds'],
                                  peak_allocated_bytes=timing['max_peak_allocated_bytes'])), flush=True)
            del prepared, prediction
        report['summary'] = summarize_benchmark_cases(report['cases'])
        report['gpu_after'] = gpu_snapshot(gpu_uuid)
        if report['gpu_after']['other_compute_processes_visible']:
            raise ValueError('Other compute processes are visible at completion; benchmark requires contention review.')
        validate_inputs(args)
        if any(sha256(source / name) != digest for name, digest in source_hashes.items()):
            raise ValueError('Source changed during the benchmark.')
        report['all_saved_metrics_match'] = all(row['saved_metric_comparison']['matches'] for row in report['cases'])
        report['maximum_saved_metric_difference'] = max(row['saved_metric_comparison']['maximum_absolute_difference'] for row in report['cases'])
        report['status'] = 'completed'
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        _atomic_json(output / 'benchmark.json', report)
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir', 'manifest', 'split', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--run-json-sha256', required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--case-ids', nargs='+', required=True, help='All validation IDs in the frozen split order.')
    parser.add_argument('--warmups', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--threads', type=int, default=1)
    return parser


def main(argv=None):
    parser = build_parser()
    try:
        result = run_benchmark(parser.parse_args(argv))
    except (ValueError, OSError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError) as error:
        parser.error(str(error))
    print(f"Completed {len(result['cases'])} validation cases on {result['environment']['device_name']}.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
