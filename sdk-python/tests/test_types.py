"""Tests for the public dataclasses in ``ort_vision_sdk.types``."""

from __future__ import annotations

import numpy as np
import pytest

from ort_vision_sdk.types import (
    BoundingBox,
    ClassProbability,
    ClassificationResult,
    DetectionResult,
    SegmentationResult,
)


class TestBoundingBox:
    """Geometry helpers on :class:`BoundingBox`."""

    def test_dimensions(self) -> None:
        b = BoundingBox(x1=10.0, y1=20.0, x2=60.0, y2=80.0)
        assert b.width == 50.0
        assert b.height == 60.0
        assert b.area == 3000.0

    def test_negative_dimensions_clamped(self) -> None:
        # Inverted box clamps to zero rather than producing negative width/height.
        b = BoundingBox(x1=60.0, y1=80.0, x2=10.0, y2=20.0)
        assert b.width == 0.0
        assert b.height == 0.0
        assert b.area == 0.0

    def test_format_conversions(self) -> None:
        b = BoundingBox(x1=1.5, y1=2.5, x2=11.7, y2=22.9)
        assert b.as_xyxy() == (1.5, 2.5, 11.7, 22.9)
        assert b.as_xywh() == pytest.approx((1.5, 2.5, 10.2, 20.4))
        assert b.as_int_xyxy() == (1, 2, 11, 22)


class TestClassProbability:
    def test_fields(self) -> None:
        p = ClassProbability(class_id=3, class_name="dog", probability=0.87)
        assert p.class_id == 3
        assert p.class_name == "dog"
        assert p.probability == 0.87

    def test_frozen(self) -> None:
        p = ClassProbability(class_id=0, class_name="x", probability=1.0)
        with pytest.raises((AttributeError, Exception)):
            p.class_id = 5  # type: ignore[misc]


class TestClassificationResult:
    def test_assembly(self) -> None:
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        probs = (
            ClassProbability(class_id=2, class_name="cat", probability=0.7),
            ClassProbability(class_id=0, class_name="dog", probability=0.3),
        )
        r = ClassificationResult(
            class_id=2, class_name="cat", confidence=0.7, image=img, probabilities=probs
        )
        assert r.class_id == 2
        assert r.confidence == 0.7
        assert r.probabilities[0].class_name == "cat"
        assert r.image.shape == (4, 4, 3)


class TestDetectionResult:
    def test_assembly(self) -> None:
        crop = np.zeros((6, 8, 3), dtype=np.uint8)
        bbox = BoundingBox(x1=0, y1=0, x2=8, y2=6)
        r = DetectionResult(
            class_id=1, class_name="cat", confidence=0.9, bbox=bbox, cropped_image=crop
        )
        assert r.bbox.area == 48
        assert r.cropped_image.shape == (6, 8, 3)


class TestSegmentationResult:
    def test_assembly(self) -> None:
        bbox = BoundingBox(x1=0, y1=0, x2=10, y2=10)
        mask = np.full((10, 10), 255, dtype=np.uint8)
        seg = np.zeros((10, 10, 3), dtype=np.uint8)
        r = SegmentationResult(
            class_id=0,
            class_name="person",
            confidence=0.8,
            bbox=bbox,
            mask=mask,
            segmented_image=seg,
        )
        assert r.mask.shape == (10, 10)
        assert r.mask.dtype == np.uint8
        assert r.segmented_image.shape == (10, 10, 3)
