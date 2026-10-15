"""Train the slice-wise 2D U-Net on the same prepared volumes as the 3D model.

All slices of a case form one network batch; reconstructed volume logits use
the existing whole-case CE + foreground Dice objective and volume metrics.
One optimizer update is made per case. All slices, including empty-organ slices,
are retained. This is an adapted case-weighted baseline, not slice-minibatch
training or an exact reproduction of the original U-Net paper.
"""

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import tempfile
import time

import torch

from baseline2d import SliceWiseUNet2D
from dataset import iter_case_batches, load_manifest, load_patient_split
from train import (CrossEntropyDiceLoss, train_one_epoch, validate_one_epoch,
                   _atomic_json, _checkpoint_name, _json_metrics,
                   _refresh_derived_files, _validate_run_state)


MODEL_KIND = 'slice_wise_unet2d'


def _save_state(path, model, optimizer, config, epoch, best_loss):
    """Publish one complete, architecture-tagged baseline checkpoint atomically."""
    payload = dict(format_version=1, model_kind=MODEL_KIND, config=config,
        completed_epochs=epoch, best_validation_loss=best_loss,
        model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
        python_rng_state=random.getstate(), torch_rng_state=torch.get_rng_state(),
        cuda_rng_states=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.baseline-', delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _restore_state(path, model, optimizer, config, epoch, best_loss):
    """Restore only a matching baseline run, including epoch-boundary RNG state."""
    payload = torch.load(path, map_location='cpu', weights_only=True)
    fields = {'format_version', 'model_kind', 'config', 'completed_epochs',
              'best_validation_loss', 'model_state_dict', 'optimizer_state_dict',
              'python_rng_state', 'torch_rng_state', 'cuda_rng_states'}
    if (not isinstance(payload, dict) or set(payload) != fields
            or payload['format_version'] != 1 or payload['model_kind'] != MODEL_KIND):
        raise ValueError('Require a slice-wise 2D baseline checkpoint; other architectures cannot resume here.')
    if (payload['config'] != config or payload['completed_epochs'] != epoch
            or payload['best_validation_loss'] != best_loss):
        raise ValueError('Baseline checkpoint configuration/progress differs from run.json.')
    current, saved = model.state_dict(), payload['model_state_dict']
    if not isinstance(saved, dict) or saved.keys() != current.keys():
        raise ValueError('Baseline checkpoint parameter keys differ.')
    for name, expected in current.items():
        tensor = saved[name]
        if (not isinstance(tensor, torch.Tensor) or tensor.shape != expected.shape
                or tensor.dtype != expected.dtype or not torch.isfinite(tensor).all()):
            raise ValueError(f'Invalid baseline checkpoint tensor: {name}')
    random.Random().setstate(payload['python_rng_state'])
    torch.Generator(device='cpu').set_state(payload['torch_rng_state'])
    cuda_states = payload['cuda_rng_states']
    if (not isinstance(cuda_states, list) or any(not isinstance(state, torch.Tensor)
            or state.dtype != torch.uint8 or state.ndim != 1 for state in cuda_states)):
        raise ValueError('Invalid baseline checkpoint CUDA RNG state.')
    if cuda_states and (not torch.cuda.is_available() or len(cuda_states) != torch.cuda.device_count()):
        raise ValueError('CUDA device count differs from the baseline checkpoint.')
    model.load_state_dict(saved, strict=True)
    optimizer.load_state_dict(payload['optimizer_state_dict'])
    random.setstate(payload['python_rng_state'])
    torch.set_rng_state(payload['torch_rng_state'])
    if cuda_states:
        torch.cuda.set_rng_state_all(cuda_states)


def run_training(args):
    """Run fresh or resume full epochs, never reading test image/label voxels."""
    for key in ('epochs', 'num_classes', 'downsample_factor', 'threads'):
        if type(getattr(args, key)) is not int or getattr(args, key) < 1:
            raise ValueError(f'{key} must be a positive integer.')
    if args.num_classes < 2:
        raise ValueError('num_classes must be at least two.')
    if type(args.seed) is not int or not 0 <= args.seed < 2**63:
        raise ValueError('seed must be an integer in [0, 2**63).')
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError('learning_rate must be finite and positive.')
    if args.device not in ('cpu', 'cuda') or type(args.deterministic) is not bool:
        raise ValueError('Require cpu/cuda and a boolean deterministic flag.')
    if args.deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(args.deterministic, warn_only=False)
    torch.backends.cudnn.deterministic = args.deterministic
    torch.backends.cudnn.benchmark = False
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; allocate a GPU or explicitly choose cpu.')

    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    split_hash = hashlib.sha256(args.split.read_bytes()).hexdigest()
    records = load_manifest(args.manifest)
    split = load_patient_split(args.split, records, manifest_sha256=manifest_hash)
    resolved = [dict(case_id=case.case_id, patient_id=case.patient_id,
                     image_path=str(case.image_path), label_path=str(case.label_path))
                for case in records]
    resolved_hash = hashlib.sha256(json.dumps(resolved, sort_keys=True).encode()).hexdigest()
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = SliceWiseUNet2D(num_classes=args.num_classes, channels=tuple(args.channels)).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = CrossEntropyDiceLoss()
    source = Path(__file__).resolve().parent
    config = _json_metrics(dict(
        model_kind=MODEL_KIND, model=model.get_config(),
        manifest_sha256=manifest_hash, split_sha256=split_hash,
        resolved_records_sha256=resolved_hash,
        optimizer='Adam', learning_rate=args.learning_rate,
        loss=dict(cross_entropy_weight=1., soft_dice_weight=1.,
                  cross_entropy_reduction='none_then_voxel_mean',
                  dice_include_background=False, dice_smooth=1e-5),
        selection='minimum_validation_loss_first_tie', batch_size=1,
        slice_batching='all source-K planes of one prepared case; no slice filtering',
        objective_unit='complete reconstructed case volume',
        normalization='affine InstanceNorm2d per prepared slice',
        dropout=0., deep_supervision=False, residual_connections=False,
        downsample_factor=args.downsample_factor,
        assume_label_units_from_image=args.assume_label_units_from_image,
        seed=args.seed, shuffle='local_Random(seed + one_based_epoch)',
        device=args.device, threads=args.threads,
        reproducibility=dict(deterministic_requested=args.deterministic,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG')),
        source_sha256={name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in ('baseline2d.py', 'train_baseline2d.py', 'dataset.py', 'objectives.py', 'train.py')},
        environment=dict(python=platform.python_version(), torch=str(torch.__version__),
            numpy=importlib.metadata.version('numpy'), nibabel=importlib.metadata.version('nibabel'),
            cuda_runtime=torch.version.cuda,
            device_name=torch.cuda.get_device_name(0) if args.device == 'cuda' else 'cpu')))
    output = args.output_dir.resolve()
    if args.resume:
        if not output.is_dir():
            raise ValueError('Resume requires an existing baseline run directory.')
    else:
        output.mkdir(parents=True, exist_ok=False)
    with (output / '.run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('Another process is writing this run.') from error
        if args.resume:
            def reject_constant(value):
                raise ValueError(f'Nonfinite run-state constant: {value}')
            state = json.loads((output / 'run.json').read_text(), parse_constant=reject_constant)
            if (not isinstance(state, dict) or not isinstance(state.get('config'), dict)
                    or state['config'].get('model_kind') != MODEL_KIND):
                raise ValueError('Cannot resume a different architecture as a 2D baseline.')
            _validate_run_state(state, config, output)
            if args.epochs < state['completed_epochs']:
                raise ValueError('epochs cannot be less than completed_epochs.')
            if state['completed_epochs']:
                _restore_state(output / state['latest_checkpoint'], model, optimizer, config,
                    state['completed_epochs'], state['history'][state['best_epoch'] - 1]['validation_loss'])
        else:
            state = dict(format_version=1, config=config, completed_epochs=0, best_epoch=None,
                         latest_checkpoint=None, best_checkpoint=None, history=[])
            _atomic_json(output / 'run.json', state)
        _refresh_derived_files(output, state)
        options = dict(downsample_factor=args.downsample_factor,
                       assume_label_units_from_image=args.assume_label_units_from_image,
                       num_classes=args.num_classes)
        for epoch in range(state['completed_epochs'] + 1, args.epochs + 1):
            if args.device == 'cuda':
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            trained = train_one_epoch(model,
                iter_case_batches(split.train, seed=args.seed + epoch, **options),
                optimizer, criterion, device=args.device)
            validated = validate_one_epoch(model, iter_case_batches(split.validation, **options),
                criterion, num_classes=args.num_classes, include_background=False, device=args.device)
            if args.device == 'cuda':
                torch.cuda.synchronize()
            row = dict(epoch=epoch, train_loss=trained['loss'], validation_loss=validated['loss'],
                train_samples=trained['num_samples'], validation_samples=validated['num_samples'],
                validation=_json_metrics(dict(case_ids=[case.case_id for case in split.validation],
                    **{key: value for key, value in validated.items() if key not in ('loss', 'num_samples')})),
                elapsed_seconds=time.perf_counter() - started,
                peak_cuda_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2 if args.device == 'cuda' else None)
            history = [*state['history'], row]
            best = min(history, key=lambda item: item['validation_loss'])
            current = dict(format_version=1, config=config, completed_epochs=epoch,
                best_epoch=best['epoch'], latest_checkpoint=_checkpoint_name(epoch),
                best_checkpoint=_checkpoint_name(best['epoch']), history=history)
            _save_state(output / current['latest_checkpoint'], model, optimizer, config, epoch, best['validation_loss'])
            _atomic_json(output / 'run.json', current)
            state = current
            _refresh_derived_files(output, state)
            print(json.dumps({key: row[key] for key in ('epoch', 'train_loss', 'validation_loss',
                             'elapsed_seconds', 'peak_cuda_allocated_mib')}), flush=True)
    return state


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, required=True, help='Target total completed epochs.')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--num-classes', type=int, default=6)
    parser.add_argument('--channels', nargs='+', type=int, default=[16, 32, 64, 128])
    parser.add_argument('--downsample-factor', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--seed', type=int, default=3710)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--assume-label-units-from-image', action='store_true')
    return parser


def main(argv=None):
    parser = build_parser()
    try:
        result = run_training(parser.parse_args(argv))
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(2, f'error: {error}\n')
    print(f"Completed {result['completed_epochs']} baseline epochs; best epoch: {result['best_epoch']}.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
