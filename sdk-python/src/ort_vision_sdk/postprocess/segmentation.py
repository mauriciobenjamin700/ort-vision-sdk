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
applying sigmoid, then resizing to the original image and cropping to the
instance's bounding box.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray
from PIL import Image

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

    # Soft masks via a single matmul over all surviving instances.
    proto_flat = prototypes.reshape(num_mask_coefs, mask_h * mask_w)
    soft_masks = _sigmoid(coefs @ proto_flat).reshape(-1, mask_h, mask_w)

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

        if mbx2 <= mbx1 or mby2 <= mby1 or bbox_w == 0 or bbox_h == 0:
            mask_binary = np.zeros((bbox_h, bbox_w), dtype=np.uint8)
        else:
            mask_crop = soft_masks[k, mby1:mby2, mbx1:mbx2]
            mask_resized = _resize_soft_mask(mask_crop, (bbox_w, bbox_h))
            mask_binary = ((mask_resized >= mask_threshold) * 255).astype(np.uint8)

        results.append((bbox, int(class_ids[k]), float(confidences[k]), mask_binary))

    return results


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    neg_exp = np.exp(x[~pos])
    out[~pos] = neg_exp / (1.0 + neg_exp)
    return out


def _resize_soft_mask(
    mask: np.ndarray,
    target_size: tuple[int, int],
) -> np.ndarray:
    """Resize a soft (float) mask to ``target_size = (width, height)``.

    Uses bilinear resampling via PIL after a temporary uint8 quantization,
    which is precise enough to feed a 0.5 threshold.
    """
    target_w, target_h = target_size
    if mask.size == 0 or target_w == 0 or target_h == 0:
        return np.zeros((target_h, target_w), dtype=np.float32)

    quantized = (mask.clip(0.0, 1.0) * 255).astype(np.uint8)
    pil = Image.fromarray(quantized, mode="L")
    resized = pil.resize((target_w, target_h), resample=Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def decode_yolov8_seg(
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
    """Deprecated alias for :func:`decode_yolo_seg`. Will be removed in 0.3.0.

    Args:
        output: ``output0`` tensor, see :func:`decode_yolo_seg`.
        prototypes: ``output1`` tensor, see :func:`decode_yolo_seg`.
        num_classes: Number of classes the model predicts.
        input_size: ``(width, height)`` of the model input (post-letterbox).
        original_size: ``(width, height)`` of the original image.
        pad: ``(pad_left, pad_top)`` letterbox padding in input-tensor pixels.
        scale: Letterbox scale factor.
        conf_threshold: Minimum class score to keep a candidate.
        iou_threshold: IoU threshold for non-maximum suppression.
        max_detections: Maximum number of instances to return after NMS.
        mask_threshold: Probability cutoff applied to soft masks; defaults to ``0.5``.

    Returns:
        Identical to :func:`decode_yolo_seg` — a list of
        ``(BoundingBox, class_id, confidence, mask)`` tuples in descending
        confidence order.

    Raises:
        ValueError: Forwarded from :func:`decode_yolo_seg` on channel-count
            mismatch.
    """
    warnings.warn(
        "decode_yolov8_seg is deprecated since 0.2.0; use decode_yolo_seg "
        "(same behavior, covers v8/v11 seg heads). The alias will be removed in 0.3.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    return decode_yolo_seg(
        output,
        prototypes,
        num_classes=num_classes,
        input_size=input_size,
        original_size=original_size,
        pad=pad,
        scale=scale,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        max_detections=max_detections,
        mask_threshold=mask_threshold,
    )
