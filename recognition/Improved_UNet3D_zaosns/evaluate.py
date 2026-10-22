"""Freeze and execute a full-cohort, paired 2D/3D evaluation.

Freezing reads run/checkpoint metadata and hashes, never NIfTI bytes. A test run
requires a hash-bound successful validation rehearsal with identical setup.
Reports and plans contain private identifiers/paths and belong outside Git.
"""

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess

import torch

from dataset import load_manifest, load_patient_split, load_nifti_case, preprocess_case
from predict import load_best_model, metric_comparison, predict_prepared, sha256, _model_kind
from terminal_regions import PROTOCOL, terminal_region_metrics, summarize_terminal_regions
from train import _atomic_json, _validate_run_state


SOURCE_FILES = ('evaluate.py', 'predict.py', 'baseline2d.py', 'modules.py',
                'dataset.py', 'objectives.py', 'train.py', 'train_baseline2d.py',
                'checkpoint.py', 'terminal_regions.py')
POLICY = dict(selection='minimum validation loss over all completed epochs; earliest tie',
    preprocessing='unchanged training normalization and factor; prepared-grid scoring',
    prediction='float32 logits; argmax over six classes; no postprocessing',
    classes=[1, 2, 3, 4, 5], empty_class='null when prediction and target both absent; otherwise valid',
    aggregation='scan means; within-patient scan means followed by equal patient means',
    matmul_allow_tf32=False, cudnn_allow_tf32=False, autocast=False,
    deterministic_algorithms=True, deterministic_warn_only=False,
    cudnn_deterministic=True, cudnn_benchmark=False, cublas_workspace_config=':4096:8',
    validation_max_absolute_metric_difference=0.005, prostate_target_dice=0.70,
    primary_accuracy_device='cuda', cpu_scope='diagnostic only; no primary acceptance decision',
    terminal_protocol=PROTOCOL,
    terminal_scope='Apply the unchanged validation-defined convention to the chosen partition; no anatomical base/apex claim.')


def _read(path):
    def reject(value):
        raise ValueError(f'Nonfinite JSON constant: {value}')
    return json.loads(Path(path).read_text(), parse_constant=reject)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()


def _source_hashes():
    root = Path(__file__).resolve().parent
    return {name: sha256(root / name) for name in SOURCE_FILES}


def setup_fingerprint(plan):
    """Partition-independent identity required by the validation rehearsal."""
    return _digest({key: plan[key] for key in ('device', 'threads', 'policy',
        'source_sha256', 'manifest', 'split', 'models', 'runtime_identity')})


def _runtime_identity(device):
    """Bind numerical runtime/hardware class, not a particular job or GPU UUID."""
    identity = dict(python=platform.python_version(), torch=str(torch.__version__),
        numpy=importlib.metadata.version('numpy'), nibabel=importlib.metadata.version('nibabel'),
        cuda_runtime=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        machine=platform.machine(), device=device)
    if device == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA unavailable; freeze/run a CUDA plan inside a GPU allocation.')
        result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
            capture_output=True, text=True, check=True, timeout=10)
        versions = set(result.stdout.strip().splitlines())
        if len(versions) != 1 or not next(iter(versions)):
            raise ValueError('Cannot establish a unique NVIDIA driver version.')
        identity.update(device_name=torch.cuda.get_device_name(0),
            device_capability=list(torch.cuda.get_device_capability(0)), driver_version=versions.pop())
    return identity


def _locks(stack, run_dirs):
    for path in sorted({Path(value).resolve(strict=True) for value in run_dirs.values()}):
        lock = stack.enter_context((path / '.run.lock').open('r'))
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('A training run is still being written.') from error


def _inspect_inputs(plan):
    if (plan['format_version'] != 1 or plan['partition'] not in ('validation', 'test')
            or plan['device'] not in ('cpu', 'cuda') or type(plan['threads']) is not int
            or plan['threads'] < 1 or plan['policy'] != POLICY
            or set(plan['models']) != {'2d', '3d'}):
        raise ValueError('Invalid evaluation plan or unsupported policy.')
    sources = _source_hashes()
    if plan['source_sha256'] != sources:
        raise ValueError('Evaluation source SHA-256 differs from the frozen plan.')
    if plan['runtime_identity'] != _runtime_identity(plan['device']):
        raise ValueError('Runtime identity differs from the frozen plan; repeat validation rehearsal first.')
    for name in ('manifest', 'split'):
        if sha256(plan[name]['path']) != plan[name]['sha256']:
            raise ValueError(f'{name} SHA-256 differs from the frozen plan.')
    records = load_manifest(plan['manifest']['path'])
    split = load_patient_split(plan['split']['path'], records,
                              manifest_sha256=plan['manifest']['sha256'])
    resolved = [dict(case_id=c.case_id, patient_id=c.patient_id,
                    image_path=str(c.image_path), label_path=str(c.label_path)) for c in records]
    resolved_hash = hashlib.sha256(json.dumps(resolved, sort_keys=True).encode()).hexdigest()
    states = {}
    expected_kind = {'2d': 'slice_wise_unet2d', '3d': 'legacy_unet3d'}
    for kind, spec in plan['models'].items():
        run_dir = Path(spec['run_dir']).resolve(strict=True)
        if sha256(run_dir / 'run.json') != spec['run_json_sha256']:
            raise ValueError('Run SHA-256 differs from the frozen plan.')
        state = _read(run_dir / 'run.json')
        config = state['config']
        _validate_run_state(state, config, run_dir)
        if not state['completed_epochs']:
            raise ValueError('Evaluation requires completed training.')
        if _model_kind(config) != expected_kind[kind] or config['model']['num_classes'] != 6:
            raise ValueError('Expected paired six-class 2D/3D models.')
        for name in ('manifest', 'split'):
            if config[name + '_sha256'] != plan[name]['sha256']:
                raise ValueError('Training input hashes differ from the evaluation plan.')
        if config['resolved_records_sha256'] != resolved_hash:
            raise ValueError('Resolved dataset paths differ from training.')
        if any(name not in sources or sources[name] != value for name, value in config['source_sha256'].items()):
            raise ValueError('Training source SHA-256 changed.')
        if (spec['checkpoint_name'] != state['best_checkpoint'] or spec['best_epoch'] != state['best_epoch']
                or spec['completed_epochs'] != state['completed_epochs']
                or sha256(run_dir / state['best_checkpoint']) != spec['checkpoint_sha256']):
            raise ValueError('Selected checkpoint/progress differs from the frozen plan.')
        expected_validation = [case.case_id for case in split.validation]
        if any(row['validation']['case_ids'] != expected_validation for row in state['history']):
            raise ValueError('Training validation cohort differs from the frozen split.')
        states[kind] = state
    a, b = states['2d'], states['3d']
    if a['completed_epochs'] != b['completed_epochs']:
        raise ValueError('Paired evaluation requires equal completed training budgets.')
    for key in ('downsample_factor', 'assume_label_units_from_image', 'reproducibility'):
        if a['config'][key] != b['config'][key]:
            raise ValueError('Paired preparation/reproducibility settings differ.')
    selected = getattr(split, plan['partition'])
    cohort = [dict(case_id=c.case_id, patient_id=c.patient_id) for c in selected]
    if not selected or cohort != plan['cohort']:
        raise ValueError('The full frozen partition must be evaluated in its saved order.')
    return states, split, selected


def build_plan(run_dirs, manifest, split, *, partition, device='cuda', threads=1,
               validation_report=None):
    """Bind the full cohort and selected checkpoints without opening image bytes."""
    if set(run_dirs) != {'2d', '3d'}:
        raise ValueError('Provide both 2d and 3d runs.')
    with ExitStack() as stack:
        _locks(stack, run_dirs)
        manifest, split = Path(manifest).resolve(strict=True), Path(split).resolve(strict=True)
        records = load_manifest(manifest)
        assignments = load_patient_split(split, records, manifest_sha256=sha256(manifest))
        if partition not in ('validation', 'test'):
            raise ValueError('Partition must be validation or test.')
        plan = dict(format_version=1, created_at=datetime.now(timezone.utc).isoformat(),
            partition=partition, device=device, threads=threads,
            policy=json.loads(json.dumps(POLICY)), source_sha256=_source_hashes(),
            runtime_identity=_runtime_identity(device),
            manifest=dict(path=str(manifest), sha256=sha256(manifest)),
            split=dict(path=str(split), sha256=sha256(split)), models={},
            cohort=[dict(case_id=c.case_id, patient_id=c.patient_id)
                    for c in getattr(assignments, partition)], validation_evidence=None)
        for kind, directory in run_dirs.items():
            directory = Path(directory).resolve(strict=True)
            state = _read(directory / 'run.json')
            _validate_run_state(state, state['config'], directory)
            if not state['completed_epochs']:
                raise ValueError('Evaluation requires completed training.')
            plan['models'][kind] = dict(run_dir=str(directory),
                run_json_sha256=sha256(directory / 'run.json'),
                checkpoint_name=state['best_checkpoint'], best_epoch=state['best_epoch'],
                completed_epochs=state['completed_epochs'],
                checkpoint_sha256=sha256(directory / state['best_checkpoint']))
        states, assignments, _ = _inspect_inputs(plan)
        if validation_report is not None:
            path = Path(validation_report).resolve(strict=True)
            plan['validation_evidence'] = dict(path=str(path), sha256=sha256(path))
        _validate_rehearsal(plan, states, assignments)
        return plan


def summarize_cases(cases):
    """Keep empty-class support and equal-patient aggregation explicit."""
    if not cases or len({r['case_id'] for r in cases}) != len(cases):
        raise ValueError('Summary requires nonempty unique cases.')
    for row in cases:
        m = row['metrics']
        if m['class_ids'] != [1, 2, 3, 4, 5] or len(m['valid']) != 5:
            raise ValueError('Expected five foreground class metrics.')
        for metric in ('dice', 'iou'):
            if len(m[metric]) != 5:
                raise ValueError('Metric width differs.')
            for value, valid in zip(m[metric], m['valid']):
                if (type(valid) is not bool or (value is not None) != valid
                        or (valid and (type(value) not in (int, float) or not math.isfinite(value)
                                       or not 0 <= value <= 1))):
                    raise ValueError('Invalid metric score/validity.')
    def aggregate(rows):
        result = {}
        for metric in ('dice', 'iou'):
            columns = [[r['metrics'][metric][i] for r in rows if r['metrics']['valid'][i]] for i in range(5)]
            result[metric] = [math.fsum(values) / len(values) if values else None for values in columns]
        result['valid_count'] = [sum(r['metrics']['valid'][i] for r in rows) for i in range(5)]
        return result
    overall = aggregate(cases)
    patients = {patient: aggregate([r for r in cases if r['patient_id'] == patient])
                for patient in sorted({r['patient_id'] for r in cases})}
    macro = {}
    for metric in ('dice', 'iou'):
        columns = [[r[metric][i] for r in patients.values() if r['valid_count'][i]] for i in range(5)]
        macro[metric] = [math.fsum(values) / len(values) if values else None for values in columns]
    return dict(class_ids=[1, 2, 3, 4, 5], num_cases=len(cases), num_patients=len(patients),
        case_mean={m: overall[m] for m in ('dice', 'iou')}, patient_macro=macro,
        valid_case_count=overall['valid_count'],
        valid_patient_count=[sum(r['valid_count'][i] > 0 for r in patients.values()) for i in range(5)],
        per_patient={p: dict(num_cases=sum(r['patient_id'] == p for r in cases), **v) for p, v in patients.items()})


def _validate_rehearsal(plan, states, split):
    evidence = plan['validation_evidence']
    if plan['partition'] == 'validation':
        if evidence is not None:
            raise ValueError('A validation rehearsal cannot depend on another report.')
        return
    if not evidence or sha256(evidence['path']) != evidence['sha256']:
        raise ValueError('Test evaluation requires a hash-bound validation report.')
    report = _read(evidence['path'])
    if (report['status'] != 'completed' or report['partition'] != 'validation'
            or report['setup_fingerprint'] != setup_fingerprint(plan)
            or set(report['models']) != {'2d', '3d'}):
        raise ValueError('Validation rehearsal is incomplete or uses a different setup.')
    cohort = [(c.case_id, c.patient_id) for c in split.validation]
    for kind, model in report['models'].items():
        if (model['status'] != 'completed'
                or [(c['case_id'], c['patient_id']) for c in model['cases']] != cohort):
            raise ValueError('Validation rehearsal must cover the complete frozen cohort.')
        reference = states[kind]['history'][states[kind]['best_epoch'] - 1]['validation']
        for index, row in enumerate(model['cases']):
            comparison = metric_comparison(row['metrics'], reference, index)
            if (row['metrics']['valid'] != reference['valid'][index]
                    or comparison['maximum_absolute_difference'] > POLICY['validation_max_absolute_metric_difference']):
                raise ValueError('Validation rehearsal exceeds the fixed backend diagnostic bound.')


def _configure(plan):
    # This is a declared evaluation policy, not a claim about unlogged training TF32 flags.
    if os.environ.get('NVIDIA_TF32_OVERRIDE') not in (None, '0'):
        raise ValueError('NVIDIA_TF32_OVERRIDE conflicts with the declared TF32-off policy.')
    if plan['device'] == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; there is no automatic CPU fallback.')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = POLICY['cublas_workspace_config']
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(plan['threads'])


def _tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def run_evaluation(args):
    if sha256(args.plan) != args.plan_sha256:
        raise ValueError('Plan SHA-256 does not match the supplied digest.')
    plan = _read(args.plan)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError('Output exists; preserve earlier results and use a new directory.')
    with ExitStack() as stack:
        _locks(stack, {kind: spec['run_dir'] for kind, spec in plan['models'].items()})
        states, split, cases = _inspect_inputs(plan)
        _validate_rehearsal(plan, states, split)
        _configure(plan)
        # Validate BOTH checkpoint payloads before the first image is opened.
        models = {}
        for kind in ('2d', '3d'):
            spec = plan['models'][kind]
            models[kind], _ = load_best_model(Path(spec['run_dir']), states[kind],
                device='cpu', expected_sha256=spec['checkpoint_sha256'])
        output.mkdir(parents=True, exist_ok=False)
        report = dict(format_version=1, status='started', partition=plan['partition'],
            started_at=datetime.now(timezone.utc).isoformat(), plan=plan, plan_sha256=args.plan_sha256,
            setup_fingerprint=setup_fingerprint(plan), models={}, training_updates=0,
            accuracy_role='primary_cuda' if plan['device'] == 'cuda' else 'cpu_diagnostic',
            test_voxels_read=False, terminal_protocol=PROTOCOL,
            interpretation='Full frozen partition, prepared-grid accuracy. One warmup/one timed forward per case are diagnostic timing, not a new cost benchmark.',
            environment=dict(**plan['runtime_identity'], threads=plan['threads'],
                slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                nvidia_tf32_override=os.environ.get('NVIDIA_TF32_OVERRIDE')))
        persist = lambda: _atomic_json(output / 'evaluation.json', report)
        persist()
        try:
            paired = {}
            for kind in ('2d', '3d'):
                model = models.pop(kind).to(plan['device']).eval()
                state, config = states[kind], states[kind]['config']
                result = report['models'][kind] = dict(status='started', cases=[])
                persist()
                for index, case in enumerate(cases):
                    # Hashing image files is an explicit part of execution, never freezing.
                    if plan['partition'] == 'test':
                        report['test_voxels_read'] = True
                        persist()
                    input_hashes = {name: sha256(getattr(case, name + '_path')) for name in ('image', 'label')}
                    loaded = load_nifti_case(case, allowed_labels=range(6),
                        assume_label_units_from_image=config['assume_label_units_from_image'])
                    prepared = preprocess_case(loaded, downsample_factor=config['downsample_factor'])
                    del loaded
                    if input_hashes != {name: sha256(getattr(case, name + '_path')) for name in ('image', 'label')}:
                        raise ValueError('Image/label bytes changed during loading.')
                    identity = dict(patient_id=case.patient_id, input_sha256=input_hashes,
                        image_sha256=_tensor_hash(prepared.image), target_sha256=_tensor_hash(prepared.target),
                        shape_dhw=list(prepared.target.shape), affine_dhw=prepared.affine_dhw.tolist(),
                        spatial_unit=prepared.spatial_unit)
                    if kind == '2d':
                        paired[case.case_id] = identity
                    elif paired[case.case_id] != identity:
                        raise ValueError('Paired input/grid/target hashes differ.')
                    with torch.autocast(device_type=plan['device'], enabled=False):
                        prediction, metrics, timing = predict_prepared(model, prepared,
                            device=plan['device'], repeats=1)
                    row = dict(case_id=case.case_id, **identity, metrics=metrics, timing=timing,
                        terminal_regions=terminal_region_metrics(prediction, prepared.target))
                    if plan['partition'] == 'validation':
                        reference = state['history'][state['best_epoch'] - 1]['validation']
                        row['saved_metric_comparison'] = metric_comparison(metrics, reference, index)
                    result['cases'].append(row)
                    persist()
                    print(json.dumps(dict(model=kind, completed_cases=index+1, total_cases=len(cases))), flush=True)
                    del prepared, prediction
                result['summary'] = summarize_cases(result['cases'])
                result['terminal_summary'] = summarize_terminal_regions(result['cases'])
                result['status'] = 'completed'
                if plan['partition'] == 'validation':
                    result['all_saved_metrics_match'] = all(r['saved_metric_comparison']['matches'] for r in result['cases'])
                    result['within_diagnostic_bound'] = all(
                        r['metrics']['valid'] == reference['valid'][i] and
                        r['saved_metric_comparison']['maximum_absolute_difference'] <= POLICY['validation_max_absolute_metric_difference']
                        for i, r in enumerate(result['cases']))
                else:
                    score = result['summary']['case_mean']['dice'][4]
                    result['prostate_case_mean_target_met'] = (
                        score >= POLICY['prostate_target_dice']
                        if score is not None and plan['device'] == 'cuda' else None)
                del model
                if plan['device'] == 'cuda':
                    torch.cuda.empty_cache()
                persist()
            if sha256(args.plan) != args.plan_sha256:
                raise ValueError('Plan changed during execution.')
            _inspect_inputs(plan)
            _validate_rehearsal(plan, states, split)
            report['status'] = 'completed'
            report['paired_inputs_and_targets_match'] = True
            if plan['partition'] == 'validation':
                report['eligible_for_test_plan'] = all(r['within_diagnostic_bound'] for r in report['models'].values())
        except BaseException as error:
            report.update(status='failed' if isinstance(error, Exception) else 'interrupted',
                          error=f'{type(error).__name__}: {error}')
            raise
        finally:
            report['ended_at'] = datetime.now(timezone.utc).isoformat()
            persist()
        return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    freeze = sub.add_parser('freeze', help='Freeze full-cohort metadata without reading NIfTI contents.')
    freeze.add_argument('--run-2d', type=Path, required=True)
    freeze.add_argument('--run-3d', type=Path, required=True)
    freeze.add_argument('--manifest', type=Path, required=True)
    freeze.add_argument('--split', type=Path, required=True)
    freeze.add_argument('--partition', choices=('validation', 'test'), required=True)
    freeze.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    freeze.add_argument('--threads', type=int, default=1)
    freeze.add_argument('--validation-report', type=Path)
    freeze.add_argument('--output-plan', type=Path, required=True)
    run = sub.add_parser('run', help='Execute exactly the bound plan into a new directory.')
    run.add_argument('--plan', type=Path, required=True)
    run.add_argument('--plan-sha256', required=True)
    run.add_argument('--output-dir', type=Path, required=True)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == 'freeze':
            plan = build_plan({'2d':args.run_2d, '3d':args.run_3d}, args.manifest, args.split,
                partition=args.partition, device=args.device, threads=args.threads,
                validation_report=args.validation_report)
            with args.output_plan.open('x') as stream:
                json.dump(plan, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write('\n')
            print(json.dumps(dict(status='frozen_without_image_reads', plan_sha256=sha256(args.output_plan))))
        else:
            result = run_evaluation(args)
            print(json.dumps(dict(status=result['status'], partition=result['partition'])))
    except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError) as error:
        parser.error(str(error))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
