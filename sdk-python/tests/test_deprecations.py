"""Verify the back-compat aliases (``decode_yolov8*``) still work and warn."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from ort_vision_sdk.postprocess.detection import (
    decode_yolo,
    decode_yolo_anchors,
    decode_yolov8,
    decode_yolov8_anchors,
)
from ort_vision_sdk.postprocess.segmentation import decode_yolo_seg, decode_yolov8_seg


def _detection_output() -> np.ndarray:
    n = 1
    out = np.zeros((1, 4 + 4, n), dtype=np.float32)  # 4 classes
    out[0, 0, 0] = 320; out[0, 1, 0] = 320
    out[0, 2, 0] = 100; out[0, 3, 0] = 100
    out[0, 4 + 2, 0] = 0.9
    return out


class TestDeprecatedDetection:
    def test_decode_yolov8_emits_deprecation(self) -> None:
        out = _detection_output()
        with pytest.warns(DeprecationWarning, match="decode_yolov8"):
            decode_yolov8(
                out,
                original_size=(640, 480),
                pad=(0, 0), scale=1.0,
                conf_threshold=0.25, iou_threshold=0.45, max_detections=10,
            )

    def test_decode_yolov8_returns_same_as_decode_yolo(self) -> None:
        out = _detection_output()
        kwargs = dict(
            original_size=(640, 480),
            pad=(0, 0), scale=1.0,
            conf_threshold=0.25, iou_threshold=0.45, max_detections=10,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            old = decode_yolov8(out, **kwargs)
        new = decode_yolo(out, **kwargs)
        assert len(old) == len(new) == 1
        assert old[0][0].as_xyxy() == new[0][0].as_xyxy()
        assert old[0][1] == new[0][1]
        assert old[0][2] == new[0][2]

    def test_decode_yolov8_anchors_alias_works(self) -> None:
        out = _detection_output()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            anchor_idx, _, cls, conf = decode_yolov8_anchors(
                out,
                num_classes=4,
                original_size=(640, 480),
                pad=(0, 0), scale=1.0,
                conf_threshold=0.25, iou_threshold=0.45, max_detections=10,
            )
        ref_idx, _, ref_cls, ref_conf = decode_yolo_anchors(
            out,
            num_classes=4,
            original_size=(640, 480),
            pad=(0, 0), scale=1.0,
            conf_threshold=0.25, iou_threshold=0.45, max_detections=10,
        )
        assert anchor_idx.tolist() == ref_idx.tolist()
        assert cls.tolist() == ref_cls.tolist()
        assert conf.tolist() == ref_conf.tolist()


class TestDeprecatedSegmentation:
    def test_decode_yolov8_seg_emits_deprecation_and_matches_new(self) -> None:
        per_anchor = np.zeros((1, 4 + 4 + 32, 1), dtype=np.float32)
        prototypes = np.zeros((1, 32, 16, 16), dtype=np.float32)
        prototypes[0, 0, :8, :] = 5.0
        prototypes[0, 0, 8:, :] = -5.0
        per_anchor[0, 0, 0] = 32; per_anchor[0, 1, 0] = 32
        per_anchor[0, 2, 0] = 64; per_anchor[0, 3, 0] = 64
        per_anchor[0, 4 + 2, 0] = 0.9
        per_anchor[0, 4 + 4 + 0, 0] = 1.0

        kwargs = dict(
            num_classes=4, input_size=(64, 64), original_size=(64, 64),
            pad=(0, 0), scale=1.0,
            conf_threshold=0.25, iou_threshold=0.45, max_detections=10,
        )

        with pytest.warns(DeprecationWarning, match="decode_yolov8_seg"):
            old = decode_yolov8_seg(per_anchor, prototypes, **kwargs)
        new = decode_yolo_seg(per_anchor, prototypes, **kwargs)
        assert len(old) == len(new) == 1
        # Same bbox, class, confidence; mask binarization deterministic.
        assert old[0][0].as_xyxy() == new[0][0].as_xyxy()
        assert old[0][1] == new[0][1]
        assert old[0][2] == new[0][2]
        np.testing.assert_array_equal(old[0][3], new[0][3])
