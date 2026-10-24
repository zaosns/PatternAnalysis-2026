"""Inspect validation scans, or evaluate a frozen full test cohort with panels.

The original flag-based command remains validation-only. The separate `test`
subcommand uses the paired frozen evaluator and requires a matching completed
validation rehearsal. It evaluates every test case and writes private PNG
examples and JSON provenance. Output directories must be new. Neither command
exports native-grid segmentations.
"""

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import torch

from baseline2d import SliceWiseUNet2D
from dataset import load_manifest, load_patient_split, load_nifti_case, preprocess_case
from modules import SegmentationModel3D
from objectives import dice_iou_per_class
from train import _atomic_json, _json_metrics, _validate_run_state


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _model_kind(config):
    """An absent architecture tag identifies the original 3D run format."""
    if 'model_kind' not in config:
        return 'legacy_unet3d'
    if config['model_kind'] == 'slice_wise_unet2d':
        return 'slice_wise_unet2d'
    raise ValueError('Unsupported training model_kind; require a legacy 3D or slice-wise 2D run.')


def load_best_model(run_dir, state, *, device, expected_sha256=None):
    """Load only model weights; optimizer and saved RNG are not restored."""
    _validate_run_state(state, state['config'], run_dir)
    if not state['completed_epochs']:
        raise ValueError('Prediction requires a completed training epoch.')
    if state['config']['model']['num_classes'] != 6:
        raise ValueError('HipMRI panels require six classes with the documented label mapping.')
    model_kind = _model_kind(state['config'])
    path = run_dir / state['best_checkpoint']
    digest = sha256(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError('Best checkpoint SHA-256 does not match the supplied value.')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError('Checkpoint must contain a model state dictionary and configuration.')
    if model_kind == 'slice_wise_unet2d':
        if payload.get('model_kind') != model_kind or 'model_config' in payload:
            raise ValueError('Checkpoint architecture differs from the slice-wise 2D run.')
        config_matches = payload.get('config') == state['config']
        constructor = SliceWiseUNet2D
    else:
        if 'model_kind' in payload or 'config' in payload:
            raise ValueError('Checkpoint architecture differs from the legacy 3D run.')
        config_matches = _json_metrics(payload.get('model_config')) == state['config']['model']
        constructor = SegmentationModel3D
    if (payload.get('format_version') != 1 or not config_matches
            or payload.get('completed_epochs') != state['best_epoch']
            or payload.get('best_validation_loss') != state['history'][state['best_epoch'] - 1]['validation_loss']):
        raise ValueError('Checkpoint configuration/progress differs from run.json.')
    model = constructor(**state['config']['model'])
    current = model.state_dict()
    saved = payload.get('model_state_dict')
    if not isinstance(saved, dict) or saved.keys() != current.keys():
        raise ValueError('Checkpoint model keys differ.')
    for name, tensor in current.items():
        other = saved[name]
        if (not isinstance(other, torch.Tensor) or other.shape != tensor.shape
                or other.dtype != tensor.dtype or not torch.isfinite(other).all()):
            raise ValueError(f'Invalid checkpoint tensor: {name}')
    model.load_state_dict(saved, strict=True)
    model.to(device).eval()
    return model, digest


def select_validation_cases(records, split, config, case_ids):
    """Bind resolved paths and reject train/test IDs before any image loading."""
    resolved = [dict(case_id=case.case_id, patient_id=case.patient_id,
                     image_path=str(case.image_path), label_path=str(case.label_path))
                for case in records]
    digest = hashlib.sha256(json.dumps(resolved, sort_keys=True).encode('utf-8')).hexdigest()
    if digest != config['resolved_records_sha256']:
        raise ValueError('Resolved image/label paths differ from the training run.')
    if not case_ids or len(case_ids) != len(set(case_ids)):
        raise ValueError('Provide a nonempty list of unique validation case IDs.')
    validation = {case.case_id: case for case in split.validation}
    if any(case_id not in validation for case_id in case_ids):
        raise ValueError('Every requested case must belong to the frozen validation partition.')
    return [validation[case_id] for case_id in case_ids]


def voxel_confusion(prediction, target, num_classes=6):
    """Return integer voxel counts with ground-truth rows and predicted columns.

    The masks must have identical nonempty shapes and integer labels in
    [0, num_classes). Counts include background and describe the prepared grid.
    Zero true positives can therefore be distinguished from zero predictions.
    """
    if type(num_classes) is not int or num_classes < 1:
        raise ValueError('num_classes must be a positive integer.')
    integer_dtypes = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
                      torch.uint16, torch.uint32, torch.uint64)
    if (not isinstance(prediction, torch.Tensor) or not isinstance(target, torch.Tensor)
            or prediction.shape != target.shape or target.ndim == 0 or target.numel() == 0):
        raise ValueError('Prediction and target must have the same nonempty shape.')
    if prediction.dtype not in integer_dtypes or target.dtype not in integer_dtypes:
        raise ValueError('Prediction and target must contain integer class labels.')
    predicted = prediction.detach().to(device='cpu', dtype=torch.int64).reshape(-1)
    truth = target.detach().to(device='cpu', dtype=torch.int64).reshape(-1)
    for labels in (predicted, truth):
        if labels.min().item() < 0 or labels.max().item() >= num_classes:
            raise ValueError('Class labels must be in [0, num_classes).')
    matrix = torch.bincount(truth * num_classes + predicted,
                            minlength=num_classes ** 2).reshape(num_classes, num_classes)
    target_counts, predicted_counts = matrix.sum(dim=1), matrix.sum(dim=0)
    true_positive = matrix.diagonal()
    return dict(class_ids=list(range(num_classes)), confusion_matrix=matrix.tolist(),
        target_voxels=target_counts.tolist(), predicted_voxels=predicted_counts.tolist(),
        true_positive=true_positive.tolist(),
        false_positive=(predicted_counts - true_positive).tolist(),
        false_negative=(target_counts - true_positive).tolist())


@torch.no_grad()
def predict_prepared(model, prepared, *, device, repeats=3):
    """One warmup plus synchronized forward-only timing on a prepared volume.

    Timings exclude CPU loading, preprocessing, host/device transfers, argmax,
    metrics and rendering. They describe these selected cases, not a benchmark
    over the full validation cohort. Warmup output is discarded before timing.
    """
    if type(repeats) is not int or repeats < 1:
        raise ValueError('repeats must be a positive integer.')
    model.eval()
    image = prepared.image.unsqueeze(0).to(device)
    use_cuda = torch.device(device).type == 'cuda'

    def synchronize():
        if use_cuda:
            torch.cuda.synchronize()

    def forward():
        logits = model(image)
        if logits.shape != (1, 6, *prepared.target.shape):
            raise ValueError('Prediction logits do not match the six-class prepared grid.')
        return logits

    warmup = forward()
    synchronize()
    del warmup
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(repeats):
        synchronize()
        started = time.perf_counter()
        logits = forward()
        synchronize()
        times.append(time.perf_counter() - started)
        if not torch.isfinite(logits).all():
            raise ValueError('Prediction logits contain nonfinite values.')
        prediction = logits.argmax(dim=1).cpu()
        del logits
    peak = torch.cuda.max_memory_allocated() / 1024**2 if use_cuda else None
    metric = _json_metrics(dice_iou_per_class(
        prediction, prepared.target.unsqueeze(0), num_classes=6, include_background=False))
    flat = {name: metric[name][0] for name in ('dice', 'iou', 'valid')}
    flat['class_ids'] = metric['class_ids']
    flat['voxel_counts'] = voxel_confusion(prediction[0], prepared.target, num_classes=6)
    return prediction[0], flat, dict(warmup_forwards=1, measured_forwards=repeats,
        forward_seconds=times, median_forward_seconds=statistics.median(times),
        peak_cuda_allocated_mib=peak)


def metric_comparison(metrics, reference, index):
    """Compare to saved best-epoch scores, retaining any mismatch explicitly."""
    if reference['class_ids'] != [1, 2, 3, 4, 5] or metrics['class_ids'] != [1, 2, 3, 4, 5]:
        raise ValueError('Saved metric class order differs from prediction metrics.')
    differences = []
    saved_flags = reference['valid'][index]
    actual_flags = metrics['valid']
    for flags in (saved_flags, actual_flags):
        if len(flags) != 5 or any(type(flag) is not bool for flag in flags):
            raise ValueError('Metric validity flags must contain five booleans.')
    valid = saved_flags == actual_flags
    for metric in ('dice', 'iou'):
        if len(reference[metric][index]) != 5 or len(metrics[metric]) != 5:
            raise ValueError('Saved per-case metric width differs from prediction metrics.')
        for scores, flags in ((metrics[metric], actual_flags), (reference[metric][index], saved_flags)):
            for score, flag in zip(scores, flags):
                if ((score is not None) != flag or (score is not None and
                        (type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1))):
                    raise ValueError('Metric values must be finite scores in [0,1], or null exactly when invalid.')
        for actual, expected in zip(metrics[metric], reference[metric][index]):
            if actual is None or expected is None:
                valid = valid and actual is expected
            else:
                difference = abs(actual - expected)
                differences.append(difference)
                valid = valid and math.isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-6)
    return dict(matches=valid, maximum_absolute_difference=max(differences, default=0.),
                relative_tolerance=1e-5, absolute_tolerance=1e-6)


def run_prediction(args):
    if args.threads < 1 or args.repeats < 1:
        raise ValueError('threads and repeats must be positive.')
    run_dir = args.run_dir.resolve(strict=True)
    # Shared read lock refuses to inspect a run while train.py is updating it.
    with (run_dir / '.run.lock').open('r') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('The training run is still being written.') from error
        return _run_locked(args, run_dir)


def _run_locked(args, run_dir):
    def reject_constant(value):
        raise ValueError(f'Nonfinite JSON constant: {value}')

    run_path = run_dir / 'run.json'
    run_hash = sha256(run_path)
    state = json.loads(run_path.read_text(), parse_constant=reject_constant)
    config = state['config']
    _validate_run_state(state, config, run_dir)
    if not state['completed_epochs']:
        raise ValueError('Prediction requires a completed training epoch.')
    model_kind = _model_kind(config)
    source = Path(__file__).resolve().parent
    source_names = (('baseline2d.py', 'train_baseline2d.py', 'dataset.py', 'objectives.py', 'train.py')
                    if model_kind == 'slice_wise_unet2d'
                    else ('dataset.py', 'modules.py', 'objectives.py', 'train.py'))
    source_hashes = {name: sha256(source / name)
                     for name in source_names}
    if any(value != config['source_sha256'].get(name) for name, value in source_hashes.items()):
        raise ValueError('Model, preprocessing or metric source differs from the training run.')
    for name in ('manifest', 'split'):
        if sha256(getattr(args, name)) != config[name + '_sha256']:
            raise ValueError(f'{name} SHA-256 differs from the training run.')
    records = load_manifest(args.manifest)
    split = load_patient_split(args.split, records, manifest_sha256=config['manifest_sha256'])
    selected = select_validation_cases(records, split, config, args.case_ids)
    panel_case_ids = args.case_ids if args.panel_case_ids is None else args.panel_case_ids
    if (not panel_case_ids or len(panel_case_ids) != len(set(panel_case_ids))
            or not set(panel_case_ids).issubset(args.case_ids)):
        raise ValueError('panel_case_ids must be a nonempty unique subset of the selected case_ids.')
    if config['model']['num_classes'] != 6:
        raise ValueError('This inspection command requires the six-class HipMRI configuration.')
    reference = state['history'][state['best_epoch'] - 1]['validation']
    if reference['case_ids'] != [case.case_id for case in split.validation]:
        raise ValueError('Saved validation order differs from the frozen split.')
    policy = config['reproducibility']
    if policy['cublas_workspace_config'] is not None:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = policy['cublas_workspace_config']
    torch.use_deterministic_algorithms(policy['deterministic_algorithms'], warn_only=policy['deterministic_warn_only'])
    torch.backends.cudnn.deterministic = policy['cudnn_deterministic']
    torch.backends.cudnn.benchmark = policy['cudnn_benchmark']
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable; allocate a GPU or explicitly choose cpu.')
    torch.set_num_threads(args.threads)
    model, checkpoint_hash = load_best_model(run_dir, state, device=args.device,
                                              expected_sha256=args.checkpoint_sha256)
    from prediction_panels import render_case_panels
    terminal_enabled = getattr(args, 'terminal_regions', False)
    if terminal_enabled:
        from terminal_regions import PROTOCOL, terminal_region_metrics, summarize_terminal_regions
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(format_version=1, status='started', partition='validation', model_kind=model_kind,
        interpretation='Validation metrics cover the explicitly selected cases; rendered panels are diagnostic examples. No held-out test evaluation.',
        run_json_sha256=run_hash, checkpoint_sha256=checkpoint_hash,
        checkpoint_name=state['best_checkpoint'], best_epoch=state['best_epoch'],
        training_config=config, source_sha256={**source_hashes,
            **{name: sha256(source / name) for name in ('predict.py', 'prediction_panels.py')}},
        selected_case_ids=args.case_ids, panel_case_ids=panel_case_ids,
        include_rectum_panels=args.include_rectum_panels, cases=[],
        environment=dict(python=platform.python_version(), torch=str(torch.__version__),
            cuda_runtime=torch.version.cuda, device=args.device, threads=args.threads,
            device_name=torch.cuda.get_device_name(0) if args.device == 'cuda' else 'cpu'))
    if terminal_enabled:
        report['terminal_region_protocol'] = PROTOCOL
        report['source_sha256']['terminal_regions.py'] = sha256(source / 'terminal_regions.py')
    _atomic_json(output / 'prediction.json', report)
    try:
        for index, case in enumerate(selected):
            started = time.perf_counter()
            loaded = load_nifti_case(case, allowed_labels=range(6),
                assume_label_units_from_image=config['assume_label_units_from_image'])
            prepared = preprocess_case(loaded, downsample_factor=config['downsample_factor'])
            del loaded
            loading_seconds = time.perf_counter() - started
            prediction, metrics, timing = predict_prepared(model, prepared,
                device=args.device, repeats=args.repeats)
            name, panel, figure_hash = None, None, None
            if case.case_id in panel_case_ids:
                name = f'validation-{index + 1:02d}.png'
                panel_options = dict(include_rectum=True) if args.include_rectum_panels else {}
                panel = render_case_panels(prepared, prediction, output / name,
                    case_label=f"Validation example {index + 1} | best epoch {state['best_epoch']}",
                    metrics=metrics, **panel_options)
                figure_hash = sha256(output / name)
            comparison = metric_comparison(metrics, reference, reference['case_ids'].index(case.case_id))
            row = dict(case_id=case.case_id, patient_id=case.patient_id,
                shape_dhw=list(prepared.target.shape), spatial_unit=prepared.spatial_unit,
                affine_dhw=prepared.affine_dhw.tolist(), metrics=metrics, timing=timing,
                loading_preprocessing_seconds=loading_seconds, saved_metric_comparison=comparison,
                panel=panel, figure=name, figure_sha256=figure_hash)
            if terminal_enabled:
                row['terminal_regions'] = terminal_region_metrics(prediction, prepared.target)
            report['cases'].append(row)
            _atomic_json(output / 'prediction.json', report)
            print(json.dumps(dict(example=index + 1, metric_match=comparison['matches'],
                                  prostate_dice=metrics['dice'][4])), flush=True)
            del prepared, prediction
        if sha256(run_path) != run_hash or sha256(run_dir / state['best_checkpoint']) != checkpoint_hash:
            raise ValueError('Training run or checkpoint changed during inspection.')
        if terminal_enabled:
            report['terminal_region_summary'] = summarize_terminal_regions(report['cases'])
        report['status'] = 'completed'
        report['all_saved_metrics_match'] = all(row['saved_metric_comparison']['matches'] for row in report['cases'])
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        _atomic_json(output / 'prediction.json', report)
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
        epilog='For frozen full-cohort test evaluation and panels, use: python predict.py test --help')
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--case-ids', nargs='+', required=True)
    parser.add_argument('--panel-case-ids', nargs='+',
                        help='Unique subset of case IDs to render; default renders every selected case.')
    parser.add_argument('--include-rectum-panels', action='store_true',
                        help='Include additional rectum-focused diagnostic views in rendered panels.')
    parser.add_argument('--terminal-regions', action='store_true',
                        help='Measure prostate errors on the first/last 20%% of target-bearing source-K planes, '
                             'including case and patient summaries. This is not anatomical base/apex identification.')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--checkpoint-sha256', help='Optional expected best-checkpoint digest from preserved records.')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=3)
    return parser


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == 'test':
        return test_main(arguments[1:])
    parser = build_parser()
    try:
        result = run_prediction(parser.parse_args(arguments))
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(f"Completed {len(result['cases'])} validation examples; saved-metric match: {result['all_saved_metrics_match']}.")
    return 0


def test_main(argv=None):
    """Run the existing full-cohort test evaluator, with paired PNG examples."""
    # Lazy import avoids the evaluator/prediction helper import cycle.
    from evaluate import run_evaluation

    parser = argparse.ArgumentParser(
        description='Evaluate a frozen full test cohort after its matching validation rehearsal. '
                    'Render the first frozen test case for each model by default; '
                    'panel selection never reduces the evaluated cohort.')
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--panel-case-ids', nargs='+',
                        help='Optional unique subset of frozen test IDs to render for both models.')
    args = parser.parse_args(argv)
    args.render_panels = True
    try:
        result = run_evaluation(args, required_partition='test')
    except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError) as error:
        parser.error(str(error))
    print(json.dumps(dict(status=result['status'], partition=result['partition'],
        accuracy_role=result['accuracy_role'],
        evaluated_cases_per_model={kind: len(model['cases']) for kind, model in result['models'].items()},
        metrics={kind: model['summary'] for kind, model in result['models'].items()},
        rendered_cases_per_model=len(result['panel_case_ids']))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
