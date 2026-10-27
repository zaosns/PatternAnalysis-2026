"""Paired validation illustrations of components measured in full 3D volumes."""
import io
from pathlib import Path

import numpy as np

from component_analysis import component_metrics, label_components


def select_satellite(reports):
    """Prefer prostate, then rectum; largest eligible component, deterministic ties.

    A satellite must have zero same-class GT overlap while another prediction
    component of that model/class has positive overlap. The selected plane has
    the largest satellite area. This is an illustration rule, not prevalence.
    """
    for class_id in (5, 4):
        candidates = []
        for order, model in enumerate(('2d', '3d')):
            report = reports[model]
            if report['protocol']['connectivity'] != 26:
                raise ValueError('Illustration selection requires 26-neighbour reports.')
            components = report['classes'][str(class_id)]['prediction_components']
            if any(c['overlap_voxels'] > 0 for c in components):
                for c in components:
                    if c['zero_overlap']:
                        candidates.append((-c['voxel_count'], order, c['component_id'], model, c))
        if candidates:
            _, _, _, model, component = min(candidates, key=lambda row: row[:3])
            return dict(model=model, class_id=class_id, component=dict(component))
    return None


def _numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def render_component_panels(prepared, prediction_2d, prediction_3d, components_2d,
                            components_3d, output_path, *, case_label):
    """Write a paired PNG, or report that no eligible satellite exists.

    Accept the historical prepared-case object by its data attributes, avoiding
    a dependency on the current preprocessing implementation. Labels and images
    are never changed or exported as volume files. Public titles must use an
    anonymous example label supplied by the caller, not a patient identifier.
    """
    path = Path(output_path)
    if path.suffix.lower() != '.png':
        raise ValueError('Expected a PNG output path.')
    if path.exists():
        raise FileExistsError(path)
    if (not isinstance(case_label, str) or not case_label.strip()
            or len(case_label) > 80 or any(c in case_label for c in '\n\r/\\')):
        raise ValueError('Use a short anonymous example label.')
    predictions = {'2d': prediction_2d, '3d': prediction_3d}
    reports = {'2d': components_2d, '3d': components_3d}
    target = _numpy(prepared.target)
    image = _numpy(prepared.image)
    if image.shape != (1, *target.shape) or not np.isfinite(image).all():
        raise ValueError('Prepared image must be finite [1,D,H,W].')
    image = image[0]
    # Bind the illustration to the supplied volumes, not a possibly stale report.
    for model in predictions:
        if component_metrics(predictions[model], target, connectivity=26) != reports[model]:
            raise ValueError('Component report does not match its prediction and target.')
    selection = select_satellite(reports)
    if selection is None:
        return dict(status='no_eligible_component', case_label=case_label,
                    selection_protocol='prostate_then_rectum_satellite_v1')
    class_id = selection['class_id']
    organ = {5: 'Prostate', 4: 'Rectum'}[class_id]
    component = selection['component']
    depth = component['representative_depth']
    depths = list(range(max(0, depth - 1), min(target.shape[0], depth + 2)))
    actual = target == class_id
    masks = {}
    volume_counts = {}
    for model, prediction in predictions.items():
        labels, _ = label_components(prediction == class_id, connectivity=26)
        summary = reports[model]['classes'][str(class_id)]
        zero_ids = [c['component_id'] for c in summary['prediction_components'] if c['zero_overlap']]
        zero_mask = np.isin(labels, zero_ids)
        masks[model] = (labels > 0, zero_mask)
        volume_counts[model] = {k: summary[k] for k in (
            'predicted_component_count', 'zero_overlap_prediction_component_count',
            'zero_overlap_prediction_voxels', 'false_positive_voxels', 'target_component_count')}
    # Shared crop contains the entire organ and every prediction in the volume.
    support = actual | masks['2d'][0] | masks['3d'][0]
    coords = np.argwhere(support)
    h0, w0 = np.maximum(coords[:, 1:].min(axis=0) - 8, 0)
    h1, w1 = np.minimum(coords[:, 1:].max(axis=0) + 9, target.shape[1:])
    crop = (slice(int(h0), int(h1)), slice(int(w0), int(w1)))
    nonzero = image[image != 0]
    low, high = np.percentile(nonzero, [1, 99]) if nonzero.size else (0., 1.)
    if high <= low:
        radius = max(abs(float(low)) * .01, 1e-3)
        low, high = float(low) - radius, float(low) + radius
    affine = _numpy(prepared.affine_dhw)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError('Prepared affine must be finite 4x4.')
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    if np.any(spacing <= 0):
        raise ValueError('Prepared affine must have nonzero spatial axes.')
    aspect = float(spacing[1] / spacing[2])

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch, Rectangle

    fig, axes = plt.subplots(len(depths), 4, figsize=(12, 2.75 * len(depths) + 1.7), squeeze=False)
    colors = ListedColormap(['#1bb6bf', '#f07840'])
    for row, d in enumerate(depths):
        for col, ax in enumerate(axes[row]):
            ax.imshow(image[d][crop], cmap='gray', vmin=low, vmax=high,
                      origin='upper', interpolation='nearest', aspect=aspect)
            if col == 1:
                mask = actual[d][crop]
                ax.imshow(np.ma.masked_where(~mask, mask), cmap=ListedColormap(['#80d98a']),
                          vmin=0, vmax=1, alpha=.7, origin='upper', interpolation='nearest', aspect=aspect)
            elif col >= 2:
                model = ('2d', '3d')[col - 2]
                predicted, zero = (m[d][crop] for m in masks[model])
                overlay = np.where(zero, 2, 1)
                ax.imshow(np.ma.masked_where(~predicted, overlay), cmap=colors, vmin=1, vmax=2,
                          alpha=.85, origin='upper', interpolation='nearest', aspect=aspect)
                truth = actual[d][crop]
                if min(truth.shape) >= 2 and truth.any() and not truth.all():
                    ax.contour(truth, levels=[.5], colors=['#80d98a'], linewidths=.65)
                if model == selection['model']:
                    start, stop = component['bbox_dhw']['start'], component['bbox_dhw']['stop']
                    if start[0] <= d < stop[0]:
                        x = max(float(start[2] - w0 - 1.5), -.35)
                        y = max(float(start[1] - h0 - 1.5), -.35)
                        right = min(float(stop[2] - w0 + .5), float(w1 - w0) - .65)
                        bottom = min(float(stop[1] - h0 + .5), float(h1 - h0) - .65)
                        width, height = right - x, bottom - y
                        ax.add_patch(Rectangle((x, y), width, height, fill=False,
                                               edgecolor='#ffef7d', linewidth=1.3))
            if row == 0:
                titles = ['MRI', f'{organ} ground truth', '2D prediction', '3D prediction']
                ax.set_title(titles[col], fontsize=11)
            if col == 0:
                ax.set_ylabel(f'Source-K plane d={d}', fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(f'{case_label}: disconnected {organ.lower()} prediction | original epoch-8 CPU validation\n'
                 f'Highlighted {selection["model"].upper()} component: {component["voxel_count"]} voxels, '
                 f'no same-class GT overlap, depths {component["bbox_dhw"]["start"][0]}–'
                 f'{component["bbox_dhw"]["stop"][0]-1}', fontsize=12, y=.985)
    fig.legend(handles=[Patch(color='#80d98a', label='Ground truth'),
                        Patch(color='#1bb6bf', label='Prediction component with some GT overlap'),
                        Patch(color='#f07840', label='Prediction component with zero GT overlap')],
               loc='lower center', bbox_to_anchor=(.5, .075), ncol=3, fontsize=8)
    count_text = ' | '.join(f'{m.upper()}: {v["zero_overlap_prediction_component_count"]} zero-overlap components, '
                            f'{v["zero_overlap_prediction_voxels"]} voxels' for m, v in volume_counts.items())
    fig.text(.5, .026, count_text + '\nCounts use full 3D volumes and 26-neighbour connectivity; predictions unchanged.'
             '\nCyan components can include false-positive voxels.',
             ha='center', fontsize=8)
    fig.subplots_adjust(top=.88, bottom=.14, left=.055, right=.99, hspace=.05, wspace=.05)
    try:
        data = io.BytesIO()
        fig.savefig(data, format='png', dpi=160)
        with path.open('xb') as stream:
            stream.write(data.getvalue())
    finally:
        plt.close(fig)
    return dict(status='rendered', case_label=case_label,
        selection_protocol='prostate_then_rectum_satellite_v1', selection=selection,
        selected_depths=depths, crop_hw=dict(start=[int(h0), int(w0)], stop=[int(h1), int(w1)]),
        display_window=dict(lower=float(low), upper=float(high), method='whole prepared nonzero image percentiles 1 and 99'),
        orientation=dict(axes='D/H/W = source K/J/I', anatomically_reoriented=False,
                         affine_dhw=affine.tolist(), spacing_dhw=spacing.tolist(), display_aspect_h_over_w=aspect),
        full_volume_counts=volume_counts, matplotlib_version=matplotlib.__version__,
        interpretation='Full-volume component labels establish separation; a single slice alone does not. '
            'Zero-overlap components exclude attached false positives; cyan components can also contain incorrect voxels.')
