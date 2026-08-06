"""Detection head postprocessing: YOLO anchor-free decoding and non-maximum suppression.

The decoder works for any YOLO export that produces the post-v8 anchor-free
head shape ``(1, 4 + nc, N)``: **YOLOv8, YOLOv9, YOLOv10, YOLOv11, YOLOv12**.
Earlier YOLO families (v5, v6, v7) emit ``(1, N, 5 + nc)`` with explicit
objectness and need a different decoder; they are **not** handled here.
"""

from __future__ import annotations

import warnings

import numpy as np

from ort_vision_sdk.types import BoundingBox


def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """Greedy non-maximum suppression on axis-aligned bounding boxes.

    Signature mirrors :func:`torchvision.ops.nms` for drop-in compatibility:
    same argument order, same return semantics (kept indices).

    Args:
        boxes: Array of shape ``(N, 4)`` with boxes in ``(x1, y1, x2, y2)``.
        scores: Array of shape ``(N,)`` with detection scores.
        iou_threshold: Boxes with IoU above this value relative to a kept box
            are suppressed.

    Returns:
        Indices of kept boxes, in descending score order. Boxes tied on score
        are visited lowest-index first, so the survivor of a tie is
        deterministic and matches both ``torchvision`` and this SDK's web
        counterpart.

    Note:
        Two boxes that are both degenerate (zero area) have zero union, and
        their IoU is defined here as ``0`` — they do not suppress each other.
        The division is masked rather than computed and discarded, so no
        ``0 / 0`` is ever evaluated: letterbox padding routinely clips boxes
        down to zero area, and the discarded form emitted a ``RuntimeWarning``
        into the caller's logs on ordinary frames.
    """
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int64)

    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1).clip(min=0) * (y2 - y1).clip(min=0)
    order = np.argsort(-scores, kind="stable")

    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        w = (xx2 - xx1).clip(min=0)
        h = (yy2 - yy1).clip(min=0)
        inter = w * h
        union = areas[i] + areas[rest] - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        order = rest[iou <= iou_threshold]

    return np.asarray(keep, dtype=np.int64)


def batched_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    idxs: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """Per-class NMS — boxes are suppressed only by other boxes of the same class.

    Signature mirrors :func:`torchvision.ops.batched_nms`.

    Args:
        boxes: Array of shape ``(N, 4)`` with boxes in ``(x1, y1, x2, y2)``.
        scores: Array of shape ``(N,)`` with detection scores.
        idxs: Array of shape ``(N,)`` with the class index of each box. Boxes
            with different ``idxs`` never suppress each other.
        iou_threshold: IoU threshold for suppression within a class.

    Returns:
        Indices of kept boxes, sorted by descending score across all classes.
        Survivors from different classes that are tied on score are ordered
        lowest-index first — an explicit tie-break, because the order the
        per-class loop happens to emit them in is an implementation detail and
        the web counterpart iterates classes in a different order.
    """
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int64)

    keep: list[int] = []
    for cls in np.unique(idxs):
        cls_indices = np.where(idxs == cls)[0]
        cls_keep = nms(boxes[cls_indices], scores[cls_indices], iou_threshold)
        keep.extend(cls_indices[cls_keep].tolist())

    if not keep:
        return np.empty((0,), dtype=np.int64)
    keep_arr = np.asarray(keep, dtype=np.int64)
    order: np.ndarray = np.lexsort((keep_arr, -scores[keep_arr]))
    ordered: np.ndarray = keep_arr[order]
    return ordered


def decode_yolo_anchors(
    output: np.ndarray,
    *,
    num_classes: int,
    original_size: tuple[int, int],
    pad: tuple[int, int],
    scale: float,
    conf_threshold: float,
    iou_threshold: float,
    max_detections: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shared YOLO anchor-decode pipeline (v8/v9/v10/v11/v12 detect + seg heads).

    Performs the per-anchor steps that are identical across all YOLO heads
    sharing the anchor-free output layout: transpose, ``xywh → xyxy``, undo
    letterbox, per-class NMS, sort and cap.

    Args:
        output: Raw per-anchor output, shape ``(1, channels, N)`` or
            ``(channels, N)``. ``channels`` is ``4 + num_classes`` for plain
            detection and ``4 + num_classes + num_mask_coefs`` for segmentation;
            this function only touches the first ``4 + num_classes`` channels.
        num_classes: Number of class-score channels following the 4 box channels.
        original_size: ``(width, height)`` of the original image.
        pad: ``(pad_left, pad_top)`` letterbox padding in input-tensor pixels.
        scale: Letterbox scale factor.
        conf_threshold: Minimum class score to keep a candidate.
        iou_threshold: IoU threshold for non-maximum suppression.
        max_detections: Maximum number of detections to return after NMS.

    Returns:
        Four parallel arrays of length ``k`` (number of survivors), all
        ordered by descending confidence:

        - ``anchor_indices``: ``int64`` indices into the original ``N``-anchor
          axis. Used by callers (e.g. segmentation) that need to look up
          per-anchor channels not consumed here.
        - ``boxes_xyxy``: ``(k, 4)`` boxes in original-image pixel coordinates.
        - ``class_ids``: ``int64`` predicted class per survivor.
        - ``confidences``: ``float`` score per survivor.
    """
    if output.ndim == 3:
        output = output[0]
    preds = output.T  # (N, channels)

    box_xywh = preds[:, :4]
    class_scores = preds[:, 4 : 4 + num_classes]
    class_ids_all = np.argmax(class_scores, axis=1)
    confidences_all = class_scores[np.arange(class_scores.shape[0]), class_ids_all]

    conf_mask = confidences_all >= conf_threshold
    if not np.any(conf_mask):
        return _empty_decode_result()

    anchor_idx = np.where(conf_mask)[0]
    box_xywh = box_xywh[conf_mask]
    class_ids = class_ids_all[conf_mask]
    confidences = confidences_all[conf_mask]

    # (cx, cy, w, h) → (x1, y1, x2, y2) in input space.
    cx, cy, w, h = box_xywh.T
    boxes_input = np.stack(
        [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0],
        axis=1,
    )

    pad_left, pad_top = pad
    orig_w, orig_h = original_size
    boxes_orig = boxes_input.copy()
    boxes_orig[:, [0, 2]] = (boxes_orig[:, [0, 2]] - pad_left) / scale
    boxes_orig[:, [1, 3]] = (boxes_orig[:, [1, 3]] - pad_top) / scale
    boxes_orig[:, [0, 2]] = boxes_orig[:, [0, 2]].clip(0, orig_w)
    boxes_orig[:, [1, 3]] = boxes_orig[:, [1, 3]].clip(0, orig_h)

    keep = batched_nms(boxes_orig, confidences, class_ids, iou_threshold)
    if keep.size == 0:
        return _empty_decode_result()

    keep = keep[:max_detections]
    return (
        anchor_idx[keep].astype(np.int64, copy=False),
        boxes_orig[keep],
        class_ids[keep].astype(np.int64, copy=False),
        confidences[keep],
    )


def _empty_decode_result() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Empty parallel arrays returned when nothing survives."""
    return (
        np.empty((0,), dtype=np.int64),
        np.empty((0, 4), dtype=np.float64),
        np.empty((0,), dtype=np.int64),
        np.empty((0,), dtype=np.float64),
    )


def decode_yolo(
    output: np.ndarray,
    *,
    original_size: tuple[int, int],
    pad: tuple[int, int],
    scale: float,
    conf_threshold: float,
    iou_threshold: float,
    max_detections: int,
) -> list[tuple[BoundingBox, int, float]]:
    """Decode an anchor-free YOLO detection output into a list of detections.

    Works for **YOLOv8/v9/v10/v11/v12** detect heads, which all share the
    output shape ``(1, 4 + num_classes, N)``. The first 4 channels are
    ``(cx, cy, w, h)`` in input-tensor pixels; the remaining are per-class
    scores already passed through sigmoid (no separate objectness term).

    Args:
        output: Raw model output, shape ``(1, 4 + num_classes, N)`` or
            ``(4 + num_classes, N)``.
        original_size: ``(width, height)`` of the original image.
        pad: ``(pad_left, pad_top)`` letterbox padding in input-tensor pixels.
        scale: Letterbox scale factor.
        conf_threshold: Minimum class score to keep a candidate.
        iou_threshold: IoU threshold for non-maximum suppression.
        max_detections: Maximum number of detections to return after NMS.

    Returns:
        List of ``(BoundingBox, class_id, confidence)`` tuples in descending
        confidence order. Bounding boxes are in original-image pixel coordinates.

    Raises:
        ValueError: If the channel count is below 5 (i.e. fewer than ``4 + 1``
            channels), which would mean the tensor cannot be a valid
            anchor-free YOLO output.
    """
    channels = output.shape[1] if output.ndim == 3 else output.shape[0]
    num_classes = channels - 4
    if num_classes < 1:
        raise ValueError(f"decode_yolo: invalid output channel count {channels}; expected >= 5.")

    _, boxes, class_ids, confidences = decode_yolo_anchors(
        output,
        num_classes=num_classes,
        original_size=original_size,
        pad=pad,
        scale=scale,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        max_detections=max_detections,
    )

    return [
        (
            BoundingBox(
                x1=float(boxes[i, 0]),
                y1=float(boxes[i, 1]),
                x2=float(boxes[i, 2]),
                y2=float(boxes[i, 3]),
            ),
            int(class_ids[i]),
            float(confidences[i]),
        )
        for i in range(boxes.shape[0])
    ]


def decode_yolov8(
    output: np.ndarray,
    *,
    original_size: tuple[int, int],
    pad: tuple[int, int],
    scale: float,
    conf_threshold: float,
    iou_threshold: float,
    max_detections: int,
) -> list[tuple[BoundingBox, int, float]]:
    """Deprecated alias for :func:`decode_yolo` — same anchor-free decoder.

    The original name suggested v8-only support; in fact the same decoder
    handles v8/v9/v10/v11/v12 detect heads. Will be removed in 0.3.0.

    Args:
        output: Raw model output, shape ``(1, 4 + num_classes, N)`` or
            ``(4 + num_classes, N)``.
        original_size: ``(width, height)`` of the original image.
        pad: ``(pad_left, pad_top)`` letterbox padding in input-tensor pixels.
        scale: Letterbox scale factor.
        conf_threshold: Minimum class score to keep a candidate.
        iou_threshold: IoU threshold for non-maximum suppression.
        max_detections: Maximum number of detections to return after NMS.

    Returns:
        Identical to :func:`decode_yolo` — a list of
        ``(BoundingBox, class_id, confidence)`` tuples in descending
        confidence order.

    Raises:
        ValueError: Forwarded from :func:`decode_yolo` when the channel count
            is below 5.
    """
    warnings.warn(
        "decode_yolov8 is deprecated since 0.2.0; use decode_yolo (same behavior, "
        "covers v8/v9/v10/v11/v12). The alias will be removed in 0.3.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    return decode_yolo(
        output,
        original_size=original_size,
        pad=pad,
        scale=scale,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        max_detections=max_detections,
    )


def decode_yolov8_anchors(
    output: np.ndarray,
    *,
    num_classes: int,
    original_size: tuple[int, int],
    pad: tuple[int, int],
    scale: float,
    conf_threshold: float,
    iou_threshold: float,
    max_detections: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deprecated alias for :func:`decode_yolo_anchors`. Will be removed in 0.3.0.

    Args:
        output: Raw per-anchor output, shape ``(1, channels, N)`` or
            ``(channels, N)``.
        num_classes: Number of class-score channels following the 4 box channels.
        original_size: ``(width, height)`` of the original image.
        pad: ``(pad_left, pad_top)`` letterbox padding in input-tensor pixels.
        scale: Letterbox scale factor.
        conf_threshold: Minimum class score to keep a candidate.
        iou_threshold: IoU threshold for non-maximum suppression.
        max_detections: Maximum number of detections to return after NMS.

    Returns:
        Identical to :func:`decode_yolo_anchors` — four parallel arrays
        ``(anchor_indices, boxes_xyxy, class_ids, confidences)`` in descending
        confidence order.
    """
    warnings.warn(
        "decode_yolov8_anchors is deprecated since 0.2.0; use decode_yolo_anchors. "
        "The alias will be removed in 0.3.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    return decode_yolo_anchors(
        output,
        num_classes=num_classes,
        original_size=original_size,
        pad=pad,
        scale=scale,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        max_detections=max_detections,
    )
