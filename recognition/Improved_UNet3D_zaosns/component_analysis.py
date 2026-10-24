"""Connected-component diagnostics on complete prepared 3D label volumes.

This reports all component sizes without a "small" threshold or postprocessing.
A predicted component with no overlap with same-class ground truth is a
zero-overlap false-positive component. Other non-largest components are not
automatically errors: the ground truth itself can have multiple components.
Partial-overlap components can still contain false-positive voxels, which are
reported separately. No component is removed or relabelled.

NumPy is required; SciPy is an optional dependency for this diagnostic module.
Coordinates and volumes are in prepared-grid voxels, not physical millimetres.
"""

import numpy as np


def _ndimage():
    try:
        from scipy import ndimage
    except ImportError as error:
        raise ImportError('Connected-component diagnostics require SciPy; install scipy in the analysis environment.') from error
    return ndimage


def label_components(mask, *, connectivity=26):
    """Label a nonempty Boolean [D,H,W] mask using this report's component IDs.

    Return (integer labels, component count). Background is 0; components are
    numbered by SciPy's array traversal. The same mask and connectivity used
    in component_metrics gives the same component_id values. Input unchanged.
    """
    if type(connectivity) is not int or connectivity not in (6, 26):
        raise ValueError('connectivity must be 6 or 26.')
    if (not isinstance(mask, np.ndarray) or mask.ndim != 3 or not mask.size
            or mask.dtype != np.bool_):
        raise ValueError('Component mask must be a nonempty NumPy Boolean [D,H,W] array.')
    ndimage = _ndimage()
    return ndimage.label(mask, structure=ndimage.generate_binary_structure(
        3, 1 if connectivity == 6 else 3))


def _validate_masks(prediction, target, connectivity, num_classes):
    if type(connectivity) is not int or connectivity not in (6, 26):
        raise ValueError('connectivity must be 6 or 26.')
    if type(num_classes) is not int or num_classes < 2:
        raise ValueError('num_classes must be an integer of at least 2.')
    if (not isinstance(prediction, np.ndarray) or not isinstance(target, np.ndarray)
            or prediction.ndim != 3 or target.ndim != 3
            or prediction.shape != target.shape or prediction.size == 0):
        raise ValueError('Prediction and target must be NumPy arrays with equal nonempty [D,H,W] shapes.')
    for mask in (prediction, target):
        if not np.issubdtype(mask.dtype, np.integer):
            raise ValueError('Prediction and target must contain integer class indices, not probabilities or booleans.')
        if np.min(mask) < 0 or np.max(mask) >= num_classes:
            raise ValueError(f'Mask labels must be in 0..{num_classes - 1}.')


def _components(mask, other_mask, structure, ndimage):
    labels, count = ndimage.label(mask, structure=structure)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    overlaps = np.bincount(labels[other_mask], minlength=count + 1)
    boxes = ndimage.find_objects(labels, max_label=count)
    components = []
    for label_id, box in enumerate(boxes, start=1):
        # Every numbered component returned by ndimage.label is nonempty.
        in_box = labels[box] == label_id
        depth_counts = np.count_nonzero(in_box, axis=(1, 2))
        representative = int(np.argmax(depth_counts))
        size, overlap = int(sizes[label_id]), int(overlaps[label_id])
        components.append(dict(
            component_id=label_id,
            voxel_count=size,
            overlap_voxels=overlap,
            nonoverlap_voxels=size - overlap,
            zero_overlap=overlap == 0,
            bbox_dhw=dict(start=[int(axis.start) for axis in box],
                          stop=[int(axis.stop) for axis in box]),
            representative_depth=int(box[0].start) + representative,
            representative_depth_voxels=int(depth_counts[representative]),
        ))
    total = int(np.count_nonzero(mask))
    zero_overlap = [component for component in components if component['zero_overlap']]
    return dict(
        voxels=total,
        count=int(count),
        zero_overlap_count=len(zero_overlap),
        zero_overlap_voxels=sum(component['voxel_count'] for component in zero_overlap),
        largest_fraction=int(sizes[1:].max()) / total if total else None,
        components=components,
    )


def component_metrics(prediction, target, *, connectivity=26, num_classes=6):
    """Return JSON-compatible counts, sizes and locations for all foregrounds.

    Inputs are equal nonempty NumPy integer [D,H,W] arrays with labels in
    0..num_classes-1; class 0 is background. The arrays are never modified.
    26-neighbour connectivity joins touching faces, edges and corners in 3D;
    the optional 6-neighbour sensitivity check joins faces only. Connectivity
    is applied identically to prediction and target. IDs identify components
    within one mask/class/connectivity, not corresponding anatomical objects.

    classes uses string class IDs so JSON serialization preserves its keys.
    Each component's overlap_voxels counts overlap with the other mask's SAME
    class. A prediction component's nonoverlap_voxels are false positives;
    a target component's nonoverlap_voxels are false negatives. Bboxes use
    inclusive start and exclusive stop. representative_depth is the D plane
    containing most component voxels, choosing the smallest D on a tie.
    Largest fractions divide by that class's total volume and are null when
    that mask has no voxels of the class. No component-size cutoff is used.
    """
    _validate_masks(prediction, target, connectivity, num_classes)
    ndimage = _ndimage()
    structure = ndimage.generate_binary_structure(3, 1 if connectivity == 6 else 3)
    classes = {}
    for class_id in range(1, num_classes):
        predicted_mask, target_mask = prediction == class_id, target == class_id
        predicted = _components(predicted_mask, target_mask, structure, ndimage)
        truth = _components(target_mask, predicted_mask, structure, ndimage)
        overlap = int(np.count_nonzero(predicted_mask & target_mask))
        classes[str(class_id)] = dict(
            class_id=class_id,
            predicted_voxels=predicted['voxels'],
            target_voxels=truth['voxels'],
            same_class_overlap_voxels=overlap,
            false_positive_voxels=predicted['voxels'] - overlap,
            false_negative_voxels=truth['voxels'] - overlap,
            predicted_component_count=predicted['count'],
            target_component_count=truth['count'],
            zero_overlap_prediction_component_count=predicted['zero_overlap_count'],
            zero_overlap_prediction_voxels=predicted['zero_overlap_voxels'],
            zero_overlap_target_component_count=truth['zero_overlap_count'],
            zero_overlap_target_voxels=truth['zero_overlap_voxels'],
            largest_prediction_component_fraction=predicted['largest_fraction'],
            largest_target_component_fraction=truth['largest_fraction'],
            prediction_components=predicted['components'],
            target_components=truth['components'],
        )
    return dict(
        format_version=1,
        protocol=dict(
            name='prepared_3d_same_class_components_v1',
            connectivity=connectivity,
            num_classes=num_classes,
            background_class=0,
            axis_order='D,H,W on the complete prepared grid',
            size_unit='prepared-grid voxels',
            component_size_threshold=None,
            zero_overlap_rule='component has no voxel intersection with the other mask of the same class',
            largest_fraction_rule='largest component voxels / all class voxels; null when class absent',
            bbox_rule='start inclusive, stop exclusive; D,H,W',
            representative_rule='plane with most component voxels; smallest D breaks ties',
            interpretation='diagnostic counts; non-largest does not imply erroneous; predictions unchanged',
        ),
        shape_dhw=[int(size) for size in prediction.shape],
        classes=classes,
    )
