"""Fixed prostate terminal-plane diagnostics on the prepared segmentation grid.

The first/last fifth of GT-occupied source-K planes are diagnostic regions,
not anatomical base/apex labels or regions of equal physical thickness. Every
selected plane contributes its entire H/W slab, including false positives.
Case scores and equally weighted patient means remain separate from pooled
voxel counts, since multiple scans from one patient are not independent people.
"""

import copy
import json
import math

import torch


METRIC_NAMES = ('dice', 'iou', 'precision', 'recall')
REGION_NAMES = (
    'lower_terminal', 'upper_terminal', 'terminal_union', 'occupied_interior',
    'outside_target_planes', 'below_extent', 'above_extent', 'internal_gaps',
    'whole_volume',
)
PROTOCOL = {
    'name': 'prostate_source_k_terminal_fifth_v1',
    'num_classes': 6,
    'target_class': 5,
    'depth_axis': 'prepared tensor D, corresponding to source K',
    'terminal_rule': 'first/last ceil(n/5) GT-occupied planes; n is the number of GT-occupied planes',
    'region_support': 'complete H/W slabs, including all predicted prostate false positives',
    'overlap_rule': 'terminal_union contains each selected plane once',
    'empty_target_rule': 'undefined extent; empty terminals/interior/extent subgroups; all planes outside target',
    'score_rule': 'unsmoothed counts; zero denominator gives null and valid=false',
    'patient_rule': 'mean defined case scores per patient, then mean defined patient means equally',
    'interpretation': 'validation diagnostic convention; no anatomical base/apex or equal physical thickness claim',
}

_COUNT_NAMES = ('target_voxels', 'predicted_voxels', 'tp', 'fp', 'fn')
_REGION_FIELDS = {'depth_indices', 'num_planes', *_COUNT_NAMES, *METRIC_NAMES, 'valid'}
_RESULT_FIELDS = {
    'format_version', 'protocol', 'shape_dhw', 'target_empty', 'occupied_depths',
    'planes_per_end', 'overlap_depths', 'extent_defined', 'regions',
}


def _partition(groups, expected, name):
    flattened = [depth for group in groups for depth in group]
    if len(flattened) != len(set(flattened)) or sorted(flattened) != expected:
        raise ValueError(f'{name} must form a disjoint, complete depth partition.')


def _depth_regions(depth_size, occupied):
    k = (len(occupied) + 4) // 5
    lower = occupied[:k]
    upper = occupied[-k:] if k else []
    union = sorted(set(lower) | set(upper))
    overlap = sorted(set(lower) & set(upper))
    interior = sorted(set(occupied) - set(union))
    all_depths = list(range(depth_size))
    outside = sorted(set(all_depths) - set(occupied))
    below = [depth for depth in outside if depth < occupied[0]] if occupied else []
    above = [depth for depth in outside if depth > occupied[-1]] if occupied else []
    gaps = [depth for depth in outside if occupied[0] < depth < occupied[-1]] if occupied else []
    _partition((union, interior, outside), all_depths, 'Terminal/interior/outside regions')
    if occupied:
        _partition((below, above, gaps), outside, 'Outside-target subgroups')
    return k, overlap, dict(zip(REGION_NAMES, (
        lower, upper, union, interior, outside, below, above, gaps, all_depths,
    )))


def _scores(tp, fp, fn):
    numerators = (2 * tp, tp, tp, tp)
    denominators = (2 * tp + fp + fn, tp + fp + fn, tp + fp, tp + fn)
    return {name: numerator / denominator if denominator else None
            for name, numerator, denominator in zip(METRIC_NAMES, numerators, denominators)}


def terminal_region_metrics(prediction, target):
    """Return fixed-region counts and scores for CPU long [D,H,W] label masks.

    Both masks must contain six-class indices 0..5. Only class 5 is scored.
    Plane selection uses ground truth only. With n occupied planes, each end
    contains ceil(n/5) planes; the union removes overlap (notably when n=1).
    Outside-target planes include internal GT gaps as well as both exterior
    extents. When GT is empty, only whole_volume/outside_target_planes have
    depth support; there is no defined lower or upper extent.
    """
    if (not isinstance(prediction, torch.Tensor) or not isinstance(target, torch.Tensor)
            or prediction.ndim != 3 or prediction.shape != target.shape
            or target.numel() == 0):
        raise ValueError('Prediction and target must have equal nonempty [D,H,W] shapes.')
    for mask in (prediction, target):
        if mask.device.type != 'cpu' or mask.dtype != torch.long:
            raise ValueError('Prediction and target must be CPU torch.long masks.')
        if mask.min().item() < 0 or mask.max().item() >= 6:
            raise ValueError('Mask labels must be integers in 0..5.')

    truth = target == 5
    predicted = prediction == 5
    target_by_depth = truth.flatten(1).sum(dim=1)
    predicted_by_depth = predicted.flatten(1).sum(dim=1)
    tp_by_depth = (truth & predicted).flatten(1).sum(dim=1)
    occupied = torch.nonzero(target_by_depth, as_tuple=False).flatten().tolist()
    k, overlap, depth_regions = _depth_regions(target.shape[0], occupied)
    regions = {}
    for name, depths in depth_regions.items():
        target_count = int(target_by_depth[depths].sum().item())
        predicted_count = int(predicted_by_depth[depths].sum().item())
        tp = int(tp_by_depth[depths].sum().item())
        fp, fn = predicted_count - tp, target_count - tp
        scores = _scores(tp, fp, fn)
        regions[name] = dict(depth_indices=depths, num_planes=len(depths),
            target_voxels=target_count, predicted_voxels=predicted_count,
            tp=tp, fp=fp, fn=fn, **scores,
            valid={metric: score is not None for metric, score in scores.items()})
    result = dict(format_version=1, protocol=copy.deepcopy(PROTOCOL),
        shape_dhw=list(target.shape), target_empty=not occupied,
        occupied_depths=occupied, planes_per_end=k, overlap_depths=overlap,
        extent_defined=bool(occupied), regions=regions)
    # The same consistency checks protect both fresh measurements and records
    # loaded later for aggregation, including additive region counts.
    _validate_result(result)
    return result


def _integer(value, *, minimum=0):
    return type(value) is int and value >= minimum


def _json(value):
    try:
        return json.dumps(value, allow_nan=False, sort_keys=True, separators=(',', ':'))
    except (TypeError, ValueError) as error:
        raise ValueError('Terminal-region records must contain strict JSON-compatible values.') from error


def _validate_result(result):
    if (not isinstance(result, dict) or set(result) != _RESULT_FIELDS
            or type(result['format_version']) is not int or result['format_version'] != 1
            or _json(result['protocol']) != _json(PROTOCOL)):
        raise ValueError('Terminal-region format/protocol differs from the fixed protocol.')
    shape = result['shape_dhw']
    if not isinstance(shape, list) or len(shape) != 3 or not all(_integer(x, minimum=1) for x in shape):
        raise ValueError('shape_dhw must contain three positive integers.')
    occupied = result['occupied_depths']
    if (not isinstance(occupied, list) or any(not _integer(x) or x >= shape[0] for x in occupied)
            or occupied != sorted(set(occupied))):
        raise ValueError('occupied_depths must be sorted, unique and within the recorded depth.')
    k, overlap, expected_regions = _depth_regions(shape[0], occupied)
    if (type(result['target_empty']) is not bool or result['target_empty'] != (not occupied)
            or type(result['extent_defined']) is not bool or result['extent_defined'] != bool(occupied)
            or not _integer(result['planes_per_end']) or result['planes_per_end'] != k
            or not isinstance(result['overlap_depths'], list)
            or any(not _integer(x) for x in result['overlap_depths'])
            or result['overlap_depths'] != overlap):
        raise ValueError('Terminal-region occupancy/extent metadata is inconsistent.')
    regions = result['regions']
    if not isinstance(regions, dict) or set(regions) != set(REGION_NAMES):
        raise ValueError('Terminal-region names differ from the fixed region schema.')
    area = shape[1] * shape[2]
    occupied_set = set(occupied)
    for name, depths in expected_regions.items():
        region = regions[name]
        if (not isinstance(region, dict) or set(region) != _REGION_FIELDS
                or not isinstance(region['depth_indices'], list)
                or any(not _integer(x) for x in region['depth_indices'])
                or region['depth_indices'] != depths
                or not _integer(region['num_planes']) or region['num_planes'] != len(depths)):
            raise ValueError(f'Invalid region schema or depth support: {name}')
        if any(not _integer(region[key]) for key in _COUNT_NAMES):
            raise ValueError(f'Region counts must be nonnegative integers: {name}')
        tp, fp, fn = (region[key] for key in ('tp', 'fp', 'fn'))
        occupied_planes = len(occupied_set.intersection(depths))
        if (region['target_voxels'] != tp + fn or region['predicted_voxels'] != tp + fp
                or tp + fp + fn > len(depths) * area
                or not occupied_planes <= region['target_voxels'] <= occupied_planes * area):
            raise ValueError(f'Incoherent voxel counts or occupancy: {name}')
        valid = region['valid']
        if not isinstance(valid, dict) or set(valid) != set(METRIC_NAMES):
            raise ValueError(f'Invalid score validity schema: {name}')
        expected_scores = _scores(tp, fp, fn)
        for metric, expected in expected_scores.items():
            actual = region[metric]
            if type(valid[metric]) is not bool or valid[metric] != (expected is not None):
                raise ValueError(f'Score validity disagrees with counts: {name}/{metric}')
            if expected is None:
                coherent = actual is None
            else:
                coherent = (type(actual) in (int, float) and math.isfinite(actual)
                            and 0 <= actual <= 1
                            and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12))
            if not coherent:
                raise ValueError(f'Score disagrees with voxel counts: {name}/{metric}')
    for key in _COUNT_NAMES:
        if sum(regions[name][key] for name in ('terminal_union', 'occupied_interior', 'outside_target_planes')) != regions['whole_volume'][key]:
            raise ValueError(f'Terminal/interior/outside counts do not reconstruct whole_volume: {key}')
        if occupied and sum(regions[name][key] for name in ('below_extent', 'above_extent', 'internal_gaps')) != regions['outside_target_planes'][key]:
            raise ValueError(f'Outside subgroup counts do not reconstruct outside_target_planes: {key}')
        if overlap:
            coherent = regions['lower_terminal'][key] == regions['upper_terminal'][key] == regions['terminal_union'][key]
        else:
            coherent = regions['lower_terminal'][key] + regions['upper_terminal'][key] == regions['terminal_union'][key]
        if not coherent:
            raise ValueError(f'Terminal union counts disagree with lower/upper regions: {key}')
    _json(result)


def _mean(values):
    defined = [value for value in values if value is not None]
    return math.fsum(defined) / len(defined) if defined else None


def summarize_terminal_regions(cases):
    """Aggregate compatible per-case measurements without pooling score weights.

    Each input record needs case_id, patient_id and terminal_regions; additional
    record fields are allowed. Scores exclude undefined (null) values. Per-patient
    scores first average that patient's defined scan scores, so patient_macro
    gives each patient with a defined score one equal vote. Valid counts accompany
    each metric. Pooled counts are reported separately and are not mean scores.
    """
    if not isinstance(cases, list) or not cases:
        raise ValueError('Provide a nonempty list of terminal-region case records.')
    seen = set()
    by_patient = {}
    for case in cases:
        if (not isinstance(case, dict) or any(key not in case for key in ('case_id', 'patient_id', 'terminal_regions'))
                or any(not isinstance(case[key], str) or not case[key].strip() for key in ('case_id', 'patient_id'))):
            raise ValueError('Every case needs nonempty case_id/patient_id strings and terminal_regions.')
        if case['case_id'] in seen:
            raise ValueError('Case IDs must be unique; duplicated scans cannot be counted twice.')
        seen.add(case['case_id'])
        _validate_result(case['terminal_regions'])
        by_patient.setdefault(case['patient_id'], []).append(case['terminal_regions']['regions'])

    summaries = {}
    for name in REGION_NAMES:
        regions = [case['terminal_regions']['regions'][name] for case in cases]
        patient_rows = {}
        for patient_id in sorted(by_patient):
            patient_regions = [record[name] for record in by_patient[patient_id]]
            patient_rows[patient_id] = dict(num_cases=len(patient_regions),
                mean={metric: _mean(row[metric] for row in patient_regions) for metric in METRIC_NAMES},
                valid_case_count={metric: sum(row['valid'][metric] for row in patient_regions)
                                  for metric in METRIC_NAMES})
        summaries[name] = dict(
            case_mean={metric: _mean(row[metric] for row in regions) for metric in METRIC_NAMES},
            patient_macro={metric: _mean(row['mean'][metric] for row in patient_rows.values())
                           for metric in METRIC_NAMES},
            per_patient=patient_rows,
            valid_case_count={metric: sum(row['valid'][metric] for row in regions) for metric in METRIC_NAMES},
            valid_patient_count={metric: sum(row['mean'][metric] is not None for row in patient_rows.values())
                                 for metric in METRIC_NAMES},
            empty_target_cases=sum(row['target_voxels'] == 0 for row in regions),
            empty_prediction_cases=sum(row['predicted_voxels'] == 0 for row in regions),
            positive_fp_cases=sum(row['fp'] > 0 for row in regions),
            pooled_counts={key: sum(row[key] for row in regions) for key in _COUNT_NAMES})
    result = dict(format_version=1, protocol=copy.deepcopy(PROTOCOL),
        num_cases=len(cases), num_patients=len(by_patient), regions=summaries)
    _json(result)
    return result
