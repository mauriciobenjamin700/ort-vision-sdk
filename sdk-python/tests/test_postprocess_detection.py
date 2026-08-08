"""Tests for ``ort_vision_sdk.postprocess.detection``.

Covers ``nms``, ``batched_nms``, ``decode_yolo_anchors`` (the shared helper),
and the high-level ``decode_yolo`` wrapper.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from ort_vision_sdk.postprocess.detection import (
    batched_nms,
    decode_yolo,
    decode_yolo_anchors,
    nms,
)
from ort_vision_sdk.types import BoundingBox


class TestNms:
    def test_empty_input(self) -> None:
        keep = nms(np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), 0.5)
        assert keep.shape == (0,)
        assert keep.dtype == np.int64

    def test_suppresses_overlapping_lower_score_box(self) -> None:
        # Two highly-overlapping boxes; keep only the higher-score one.
        boxes = np.array([[0, 0, 10, 10], [1, 1, 9, 9]], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)
        keep = nms(boxes, scores, iou_threshold=0.5)
        assert keep.tolist() == [0]

    def test_keeps_distant_boxes(self) -> None:
        boxes = np.array([[0, 0, 10, 10], [100, 100, 110, 110]], dtype=np.float32)
        scores = np.array([0.9, 0.7], dtype=np.float32)
        keep = nms(boxes, scores, iou_threshold=0.5)
        # Both kept, ordered by descending score.
        assert keep.tolist() == [0, 1]

    def test_tied_scores_keep_the_lowest_index(self) -> None:
        """Ties resolve to the lowest index, matching torchvision and the web SDK."""
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [2, 2, 12, 12]], dtype=np.float32)
        scores = np.array([0.5, 0.5, 0.5], dtype=np.float32)

        keep = nms(boxes, scores, iou_threshold=0.45)

        assert keep.tolist() == [0]

    def test_degenerate_boxes_do_not_warn(self) -> None:
        """Zero-area boxes have zero union, and that must not be a ``0 / 0``.

        Letterbox padding clips boxes down to zero area on ordinary frames, so a
        warning here would reach every caller's logs.
        """
        boxes = np.array([[5, 5, 5, 5], [5, 5, 5, 5], [0, 0, 10, 10]], dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            keep = nms(boxes, scores, iou_threshold=0.45)

        assert keep.tolist() == [0, 1, 2]

    def test_returns_descending_score_order(self) -> None:
        boxes = np.array([[0, 0, 5, 5], [10, 0, 15, 5], [20, 0, 25, 5]], dtype=np.float32)
        scores = np.array([0.3, 0.9, 0.6], dtype=np.float32)
        keep = nms(boxes, scores, iou_threshold=0.5)
        assert keep.tolist() == [1, 2, 0]


class TestBatchedNms:
    def test_per_class_isolation(self) -> None:
        # Two heavily-overlapping boxes belonging to *different* classes — both should survive.
        boxes = np.array([[0, 0, 10, 10], [1, 1, 9, 9]], dtype=np.float32)
        scores = np.array([0.9, 0.85], dtype=np.float32)
        idxs = np.array([0, 1], dtype=np.int64)
        keep = batched_nms(boxes, scores, idxs, iou_threshold=0.5)
        assert sorted(keep.tolist()) == [0, 1]

    def test_same_class_suppression(self) -> None:
        boxes = np.array([[0, 0, 10, 10], [1, 1, 9, 9]], dtype=np.float32)
        scores = np.array([0.9, 0.85], dtype=np.float32)
        idxs = np.array([0, 0], dtype=np.int64)
        keep = batched_nms(boxes, scores, idxs, iou_threshold=0.5)
        assert keep.tolist() == [0]

    def test_empty_input(self) -> None:
        keep = batched_nms(
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            0.5,
        )
        assert keep.shape == (0,)


def _make_yolo_output(
    *,
    num_classes: int,
    anchors: list[tuple[float, float, float, float, int, float]],
    extra_channels: int = 0,
) -> tuple[np.ndarray, int]:
    """Build a synthetic ``(1, 4 + num_classes + extra_channels, N)`` tensor.

    Each anchor is a tuple ``(cx, cy, w, h, class_id, score)`` set in the
    appropriate channels; everything else is zeroed (zero is below any
    realistic confidence threshold).
    """
    n = len(anchors)
    channels = 4 + num_classes + extra_channels
    out = np.zeros((1, channels, n), dtype=np.float32)
    for a, (cx, cy, w, h, cls, score) in enumerate(anchors):
        out[0, 0, a] = cx
        out[0, 1, a] = cy
        out[0, 2, a] = w
        out[0, 3, a] = h
        out[0, 4 + cls, a] = score
    return out, channels


class TestDecodeYoloAnchors:
    def test_filters_below_conf_threshold(self) -> None:
        out, _ = _make_yolo_output(
            num_classes=4,
            anchors=[
                (320, 320, 100, 100, 2, 0.9),
                (50, 50, 40, 40, 1, 0.05),  # below threshold
            ],
        )
        anchor_idx, boxes, cls, conf = decode_yolo_anchors(
            out,
            num_classes=4,
            original_size=(640, 480),
            pad=(0, 80),
            scale=1.0,
            conf_threshold=0.25,
            iou_threshold=0.45,
            max_detections=10,
        )
        assert anchor_idx.tolist() == [0]
        assert cls.tolist() == [2]
        assert conf == pytest.approx([0.9])

    def test_per_class_nms_keeps_distinct_classes(self) -> None:
        # Two overlapping boxes of different classes — both kept.
        out, _ = _make_yolo_output(
            num_classes=4,
            anchors=[
                (320, 320, 100, 100, 2, 0.9),
                (320, 320, 100, 100, 1, 0.85),
            ],
        )
        anchor_idx, _, cls, _ = decode_yolo_anchors(
            out,
            num_classes=4,
            original_size=(640, 480),
            pad=(0, 80),
            scale=1.0,
            conf_threshold=0.25,
            iou_threshold=0.45,
            max_detections=10,
        )
        assert sorted(cls.tolist()) == [1, 2]
        assert sorted(anchor_idx.tolist()) == [0, 1]

    def test_descending_confidence_order(self) -> None:
        out, _ = _make_yolo_output(
            num_classes=4,
            anchors=[
                (10, 10, 5, 5, 0, 0.4),
                (200, 200, 5, 5, 1, 0.9),
                (400, 400, 5, 5, 2, 0.6),
            ],
        )
        _, _, _, conf = decode_yolo_anchors(
            out,
            num_classes=4,
            original_size=(640, 480),
            pad=(0, 0),
            scale=1.0,
            conf_threshold=0.25,
            iou_threshold=0.45,
            max_detections=10,
        )
        assert conf.tolist() == sorted(conf.tolist(), reverse=True)

    def test_max_detections_caps_results(self) -> None:
        out, _ = _make_yolo_output(
            num_classes=4,
            anchors=[
                (10, 10, 5, 5, 0, 0.9),
                (200, 200, 5, 5, 1, 0.85),
                (400, 400, 5, 5, 2, 0.8),
            ],
        )
        anchor_idx, _, _, _ = decode_yolo_anchors(
            out,
            num_classes=4,
            original_size=(640, 480),
            pad=(0, 0),
            scale=1.0,
            conf_threshold=0.25,
            iou_threshold=0.45,
            max_detections=2,
        )
        assert anchor_idx.shape == (2,)

    def test_returns_empty_when_nothing_passes(self) -> None:
        out, _ = _make_yolo_output(
            num_classes=4, anchors=[(10, 10, 5, 5, 0, 0.05)]
        )
        anchor_idx, boxes, cls, conf = decode_yolo_anchors(
            out,
            num_classes=4,
            original_size=(640, 480),
            pad=(0, 0),
            scale=1.0,
            conf_threshold=0.25,
            iou_threshold=0.45,
            max_detections=10,
        )
        assert anchor_idx.size == 0
        assert boxes.size == 0
        assert cls.size == 0
        assert conf.size == 0


class TestDecodeYolo:
    def test_full_pipeline(self) -> None:
        # Three anchors: two overlapping high-score + one distant low-class.
        out, _ = _make_yolo_output(
            num_classes=4,
            anchors=[
                (320, 320, 100, 100, 2, 0.9),
                (330, 330, 100, 100, 2, 0.8),  # NMS-suppressed (same class, overlap)
                (50, 50, 40, 40, 1, 0.85),
            ],
        )
        dets = decode_yolo(
            out,
            original_size=(640, 480),
            pad=(0, 80),
            scale=1.0,
            conf_threshold=0.25,
            iou_threshold=0.45,
            max_detections=10,
        )
        assert len(dets) == 2
        bbox0, cid0, conf0 = dets[0]
        assert isinstance(bbox0, BoundingBox)
        assert cid0 == 2
        assert conf0 == pytest.approx(0.9)
        assert bbox0.as_xyxy() == (270.0, 190.0, 370.0, 290.0)

    def test_returns_empty_list_when_nothing_passes(self) -> None:
        out, _ = _make_yolo_output(
            num_classes=4, anchors=[(10, 10, 5, 5, 0, 0.05)]
        )
        dets = decode_yolo(
            out,
            original_size=(640, 480),
            pad=(0, 0),
            scale=1.0,
            conf_threshold=0.25,
            iou_threshold=0.45,
            max_detections=10,
        )
        assert dets == []

    def test_rejects_invalid_channel_count(self) -> None:
        # 4 channels = boxes only, no class scores → invalid.
        bad = np.zeros((1, 4, 3), dtype=np.float32)
        with pytest.raises(ValueError, match=r"channel count"):
            decode_yolo(
                bad,
                original_size=(640, 480),
                pad=(0, 0),
                scale=1.0,
                conf_threshold=0.25,
                iou_threshold=0.45,
                max_detections=10,
            )
