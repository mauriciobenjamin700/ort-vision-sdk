"""Tests for ``ort_vision_sdk.postprocess.segmentation`` (YOLO-seg decode)."""

from __future__ import annotations

import numpy as np
import pytest

from ort_vision_sdk.postprocess.segmentation import decode_yolo_seg
from ort_vision_sdk.types import BoundingBox


def _build_outputs(
    *,
    num_classes: int = 4,
    num_mask_coefs: int = 32,
    mask_h: int = 16,
    mask_w: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Build empty per-anchor and prototype tensors for a 3-anchor synthetic test."""
    n = 3
    per_anchor = np.zeros((1, 4 + num_classes + num_mask_coefs, n), dtype=np.float32)
    prototypes = np.zeros((1, num_mask_coefs, mask_h, mask_w), dtype=np.float32)
    # Prototype 0: top half +5, bottom half -5 → sigmoid sharply ~1 / ~0.
    prototypes[0, 0, : mask_h // 2, :] = 5.0
    prototypes[0, 0, mask_h // 2 :, :] = -5.0
    return per_anchor, prototypes


class TestDecodeYoloSeg:
    def test_full_image_instance_mask_matches_prototype(self) -> None:
        per_anchor, prototypes = _build_outputs()
        # Anchor 0: bbox covers entire 64x64 image, class 2, conf 0.9, coef 0 = 1.
        per_anchor[0, 0, 0] = 32; per_anchor[0, 1, 0] = 32
        per_anchor[0, 2, 0] = 64; per_anchor[0, 3, 0] = 64
        per_anchor[0, 4 + 2, 0] = 0.9
        per_anchor[0, 4 + 4 + 0, 0] = 1.0  # mask coef 0

        decoded = decode_yolo_seg(
            per_anchor,
            prototypes,
            num_classes=4,
            input_size=(64, 64),
            original_size=(64, 64),
            pad=(0, 0),
            scale=1.0,
            conf_threshold=0.25,
            iou_threshold=0.45,
            max_detections=10,
        )
        assert len(decoded) == 1
        bbox, cid, conf, mask = decoded[0]
        assert isinstance(bbox, BoundingBox)
        assert cid == 2
        assert conf == pytest.approx(0.9)
        assert mask.shape == (64, 64)
        assert mask.dtype == np.uint8
        # Mask values must be strictly binary.
        assert set(np.unique(mask).tolist()).issubset({0, 255})
        # Top half ~all white, bottom half ~all black (matches prototype shape).
        assert (mask[:32, :] == 255).mean() > 0.95
        assert (mask[32:, :] == 0).mean() > 0.95

    def test_per_class_nms_keeps_overlapping_distinct_classes(self) -> None:
        per_anchor, prototypes = _build_outputs()
        # Two boxes covering same area, different classes → both survive.
        per_anchor[0, 0, 0] = 32; per_anchor[0, 1, 0] = 32
        per_anchor[0, 2, 0] = 64; per_anchor[0, 3, 0] = 64
        per_anchor[0, 4 + 2, 0] = 0.9
        per_anchor[0, 4 + 4 + 0, 0] = 1.0

        per_anchor[0, 0, 1] = 32; per_anchor[0, 1, 1] = 32
        per_anchor[0, 2, 1] = 64; per_anchor[0, 3, 1] = 64
        per_anchor[0, 4 + 1, 1] = 0.85
        per_anchor[0, 4 + 4 + 0, 1] = 1.0

        decoded = decode_yolo_seg(
            per_anchor, prototypes,
            num_classes=4, input_size=(64, 64), original_size=(64, 64),
            pad=(0, 0), scale=1.0,
            conf_threshold=0.25, iou_threshold=0.45, max_detections=10,
        )
        cls = sorted(d[1] for d in decoded)
        assert cls == [1, 2]

    def test_returns_descending_confidence_order(self) -> None:
        per_anchor, prototypes = _build_outputs()
        per_anchor[0, 0, 0] = 32; per_anchor[0, 1, 0] = 32
        per_anchor[0, 2, 0] = 64; per_anchor[0, 3, 0] = 64
        per_anchor[0, 4 + 2, 0] = 0.5
        per_anchor[0, 0, 1] = 16; per_anchor[0, 1, 1] = 16
        per_anchor[0, 2, 1] = 16; per_anchor[0, 3, 1] = 16
        per_anchor[0, 4 + 1, 1] = 0.95

        decoded = decode_yolo_seg(
            per_anchor, prototypes,
            num_classes=4, input_size=(64, 64), original_size=(64, 64),
            pad=(0, 0), scale=1.0,
            conf_threshold=0.25, iou_threshold=0.45, max_detections=10,
        )
        confs = [d[2] for d in decoded]
        assert confs == sorted(confs, reverse=True)

    def test_returns_empty_when_nothing_passes(self) -> None:
        per_anchor, prototypes = _build_outputs()
        per_anchor[0, 4 + 2, 0] = 0.05
        decoded = decode_yolo_seg(
            per_anchor, prototypes,
            num_classes=4, input_size=(64, 64), original_size=(64, 64),
            pad=(0, 0), scale=1.0,
            conf_threshold=0.25, iou_threshold=0.45, max_detections=10,
        )
        assert decoded == []

    def test_validates_channel_count(self) -> None:
        per_anchor, prototypes = _build_outputs(num_mask_coefs=32)
        with pytest.raises(ValueError, match="channel count"):
            # Lying about num_classes — channel total no longer matches.
            decode_yolo_seg(
                per_anchor, prototypes,
                num_classes=10,  # wrong: model has 4
                input_size=(64, 64), original_size=(64, 64),
                pad=(0, 0), scale=1.0,
                conf_threshold=0.25, iou_threshold=0.45, max_detections=10,
            )

    def test_mask_threshold_controls_binarization(self) -> None:
        # Use an all-zero prototype + zero coef → soft mask is sigmoid(0) = 0.5 everywhere.
        per_anchor = np.zeros((1, 4 + 4 + 32, 1), dtype=np.float32)
        prototypes = np.zeros((1, 32, 16, 16), dtype=np.float32)
        per_anchor[0, 0, 0] = 32; per_anchor[0, 1, 0] = 32
        per_anchor[0, 2, 0] = 64; per_anchor[0, 3, 0] = 64
        per_anchor[0, 4 + 0, 0] = 0.9

        # threshold=0.4 ⇒ keep (0.5 >= 0.4) → all 255
        decoded_low = decode_yolo_seg(
            per_anchor, prototypes,
            num_classes=4, input_size=(64, 64), original_size=(64, 64),
            pad=(0, 0), scale=1.0,
            conf_threshold=0.25, iou_threshold=0.45, max_detections=10,
            mask_threshold=0.4,
        )
        # threshold=0.6 ⇒ drop (0.5 < 0.6) → all 0
        decoded_high = decode_yolo_seg(
            per_anchor, prototypes,
            num_classes=4, input_size=(64, 64), original_size=(64, 64),
            pad=(0, 0), scale=1.0,
            conf_threshold=0.25, iou_threshold=0.45, max_detections=10,
            mask_threshold=0.6,
        )
        assert (decoded_low[0][3] == 255).all()
        assert (decoded_high[0][3] == 0).all()
