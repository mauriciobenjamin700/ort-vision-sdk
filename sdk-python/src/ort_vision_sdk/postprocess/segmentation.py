"""Segmentation head postprocessing: YOLO instance-segmentation decoding.

Works for any YOLO seg head sharing the v8 export layout — verified for
YOLOv8-seg and YOLOv11-seg, expected to also cover v9-seg/v12-seg if
exported. Two output tensors are required:

- ``output0`` of shape ``(1, 4 + num_classes + num_mask_coefs, num_anchors)``
  — the same per-anchor predictions as plain YOLOv8 detection plus an extra
  ``num_mask_coefs`` (typically 32) channels of mask coefficients.
- ``output1`` of shape ``(1, num_mask_coefs, mask_h, mask_w)`` — a set of
  "prototype" masks shared across all anchors.

Each instance's binary mask is reconstructed by linearly combining the
prototypes with that instance's coefficients (``coefs @ prototypes``),
applying sigmoid, then resizing to the instance's bounding box in
original-image coordinates and thresholding.

The prototype combination is restricted to the prototype region under each
instance's box **before** the sigmoid runs, rather than being evaluated over
every prototype pixel of every instance. Sigmoid is elementwise, so slicing
first is exact — it only skips the pixels that were about to be discarded,
which for typical box sizes is the overwhelming majority of them.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ort_vision_sdk.postprocess.detection import decode_yolo_anchors
from ort_vision_sdk.types import BoundingBox

DecodedSegmentation = tuple[BoundingBox, int, float, NDArray[np.uint8]]
"""``(bbox, class_id, confidence, mask)`` for a single segmented instance.

``mask`` is a binary (0/255) ``uint8`` array cropped to the bounding box
in original-image pixel coordinates. Shape is ``(bbox_h, bbox_w)``.
"""


def decode_yolo_seg(
    output: np.ndarray,
    prototypes: np.ndarray,
    *,
    num_classes: int,
    input_size: tuple[int, int],
    original_size: tuple[int, int],
    pad: tuple[int, int],
    scale: float,
    conf_threshold: float,
    iou_threshold: float,
    max_detections: int,
    mask_threshold: float = 0.5,
) -> list[DecodedSegmentation]:
    """Decode YOLO segmentation raw outputs into a list of segmented instances.

    Compatible with the YOLOv8-seg / YOLOv11-seg export layout (and any
    later seg head sharing it).

    Args:
        output: ``output0`` tensor of shape
            ``(1, 4 + num_classes + num_mask_coefs, num_anchors)`` or the same
            without the leading batch dim.
        prototypes: ``output1`` tensor of shape
            ``(1, num_mask_coefs, mask_h, mask_w)`` or without the batch dim.
        num_classes: Number of classes the model predicts. Required because the
            channel split between class scores and mask coefficients cannot be
            inferred from shapes alone.
        input_size: ``(width, height)`` of the model input (post-letterbox).
        original_size: ``(width, height)`` of the original image.
        pad: ``(pad_left, pad_top)`` letterbox padding in input-tensor pixels.
        scale: Letterbox scale factor.
        conf_threshold: Minimum class score to keep a candidate.
        iou_threshold: IoU threshold for non-maximum suppression.
        max_detections: Maximum number of instances to return after NMS.
        mask_threshold: Probability cutoff applied to the soft mask to obtain
            the final binary mask. Defaults to ``0.5``.

    Returns:
        List of ``(BoundingBox, class_id, confidence, mask)`` tuples in
        descending confidence order. Each ``mask`` is a binary ``uint8``
        array (0 or 255) cropped to its bounding box in original-image
        pixel coordinates.

    Raises:
        ValueError: If the per-anchor output channel count does not equal
            ``4 + num_classes + num_mask_coefs`` — i.e. the prototypes and
            per-anchor tensors come from different models, or ``num_classes``
            was misreported.
    """
    if prototypes.ndim == 4:
        prototypes = prototypes[0]
    num_mask_coefs, mask_h, mask_w = prototypes.shape

    output_2d = output[0] if output.ndim == 3 else output
    expected_channels = 4 + num_classes + num_mask_coefs
    if output_2d.shape[0] != expected_channels:
        raise ValueError(
            f"YOLO seg output channel count {output_2d.shape[0]} does not match "
            f"4 + num_classes ({num_classes}) + num_mask_coefs ({num_mask_coefs}) "
            f"= {expected_channels}."
        )

    anchor_indices, boxes_orig, class_ids, confidences = decode_yolo_anchors(
        output_2d,
        num_classes=num_classes,
        original_size=original_size,
        pad=pad,
        scale=scale,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        max_detections=max_detections,
    )

    if anchor_indices.size == 0:
        return []

    # Mask coefficients live in the per-anchor block, after the class scores.
    coefs = output_2d[
        4 + num_classes : 4 + num_classes + num_mask_coefs,
        anchor_indices,
    ].T  # (k, num_mask_coefs)

    input_w, input_h = input_size
    pad_left, pad_top = pad
    scale_x = mask_w / input_w
    scale_y = mask_h / input_h

    results: list[DecodedSegmentation] = []
    for k in range(anchor_indices.size):
        bx1, by1, bx2, by2 = boxes_orig[k]
        bbox = BoundingBox(x1=float(bx1), y1=float(by1), x2=float(bx2), y2=float(by2))

        bbox_w = max(0, int(bx2) - int(bx1))
        bbox_h = max(0, int(by2) - int(by1))

        # Bbox in input-tensor coords, then in low-res mask coords.
        ibx1 = bx1 * scale + pad_left
        iby1 = by1 * scale + pad_top
        ibx2 = bx2 * scale + pad_left
        iby2 = by2 * scale + pad_top

        mbx1 = max(0, int(np.floor(ibx1 * scale_x)))
        mby1 = max(0, int(np.floor(iby1 * scale_y)))
        mbx2 = min(mask_w, int(np.ceil(ibx2 * scale_x)))
        mby2 = min(mask_h, int(np.ceil(iby2 * scale_y)))

        mask_binary: np.ndarray
        if mbx2 <= mbx1 or mby2 <= mby1 or bbox_w == 0 or bbox_h == 0:
            mask_binary = np.zeros((bbox_h, bbox_w), dtype=np.uint8)
        else:
            region = prototypes[:, mby1:mby2, mbx1:mbx2].reshape(num_mask_coefs, -1)
            mask_crop = _sigmoid(coefs[k] @ region).reshape(mby2 - mby1, mbx2 - mbx1)
            mask_resized = _resize_bilinear(mask_crop, (bbox_w, bbox_h))
            mask_binary = ((mask_resized >= mask_threshold) * 255).astype(np.uint8)

        results.append((bbox, int(class_ids[k]), float(confidences[k]), mask_binary))

    return results


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid, evaluated branchlessly.

    ``exp(-|x|)`` never overflows, so the same expression covers both signs:
    ``1 / (1 + t)`` for ``x >= 0`` and ``t / (1 + t)`` for ``x < 0``. Writing it
    this way computes one exponential over the whole array instead of two
    partial ones behind a pair of boolean masks, and it keeps the formula
    identical to the web SDK's scalar ``sigmoid``.

    Args:
        x: Logits of any shape.

    Returns:
        A ``float32`` array of the same shape with values in ``(0, 1)``.
    """
    t = np.exp(-np.abs(x, dtype=np.float32))
    values: np.ndarray = np.where(x >= 0, 1.0 / (1.0 + t), t / (1.0 + t))
    return values.astype(np.float32, copy=False)


def _resize_bilinear(
    mask: np.ndarray,
    target_size: tuple[int, int],
) -> np.ndarray:
    """Resize a soft (float) mask to ``target_size = (width, height)``.

    Bilinear resampling with half-pixel centers (``(i + 0.5) * ratio - 0.5``)
    and edge clamping, sampling the four neighbours around each target pixel.

    This is deliberately hand-written rather than delegated to PIL. PIL cannot
    resample a float array without a round-trip through ``uint8``, and that
    quantization puts the input to a ``>= 0.5`` test on a grid of ``1/255``
    steps, flipping mask border pixels for no reason. It also keeps the result
    numerically aligned with the web SDK's ``resizeBilinear``, which
    implements this same formula — the two artifacts are supposed to produce
    the same mask for the same model output, and a shared algorithm is what
    makes that checkable.

    Args:
        mask: Source mask, shape ``(src_h, src_w)``.
        target_size: Target ``(width, height)`` in pixels.

    Returns:
        A ``float32`` array of shape ``(target_h, target_w)``.
    """
    target_w, target_h = target_size
    if mask.size == 0 or target_w == 0 or target_h == 0:
        return np.zeros((target_h, target_w), dtype=np.float32)

    source = mask.astype(np.float32, copy=False)
    src_h, src_w = source.shape
    if (src_w, src_h) == (target_w, target_h):
        return source

    x0, x1, wx = _sample_axis(src_w, target_w)
    y0, y1, wy = _sample_axis(src_h, target_h)

    top: np.ndarray = source[np.ix_(y0, x0)] * (1.0 - wx) + source[np.ix_(y0, x1)] * wx
    bottom: np.ndarray = source[np.ix_(y1, x0)] * (1.0 - wx) + source[np.ix_(y1, x1)] * wx
    blended: np.ndarray = top * (1.0 - wy[:, None]) + bottom * wy[:, None]
    return blended.astype(np.float32, copy=False)


def _sample_axis(
    src_length: int,
    target_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the bilinear sampling indices and weights for one axis.

    Args:
        src_length: Number of source pixels along the axis.
        target_length: Number of target pixels along the axis.

    Returns:
        ``(lower, upper, weight)`` — the two source indices to blend per target
        pixel and the ``float32`` weight of the upper one, all of length
        ``target_length``. The weight is derived from the **clamped** lower
        index, so edge pixels blend with themselves instead of extrapolating.
    """
    centers = (np.arange(target_length, dtype=np.float32) + 0.5) * (
        src_length / target_length
    ) - 0.5
    lower = np.maximum(0, np.floor(centers)).astype(np.int64)
    upper = np.minimum(src_length - 1, lower + 1)
    weight = np.clip(centers - lower, 0.0, 1.0).astype(np.float32)
    return lower, upper, weight
