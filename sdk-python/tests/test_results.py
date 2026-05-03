"""Tests for the Ultralytics-style result envelopes in ``ort_vision_sdk.results``."""

from __future__ import annotations

import numpy as np
import pytest

from ort_vision_sdk.results import (
    Boxes,
    ClassificationResults,
    DetectionResults,
    Masks,
    Probs,
    SegmentationResults,
)
from ort_vision_sdk.types import (
    BoundingBox,
    ClassProbability,
    ClassificationResult,
    DetectionResult,
    SegmentationResult,
)


class TestBoxes:
    def test_xyxy_xywh_consistency(self) -> None:
        xyxy = np.array([[10.0, 20.0, 60.0, 80.0], [0.0, 0.0, 100.0, 50.0]])
        cls = np.array([1, 2], dtype=np.int64)
        conf = np.array([0.9, 0.7])
        b = Boxes(xyxy=xyxy, cls=cls, conf=conf, orig_shape=(120, 200))

        assert len(b) == 2
        assert b.shape == (2, 4)
        np.testing.assert_array_almost_equal(
            b.xywh,
            [[35.0, 50.0, 50.0, 60.0], [50.0, 25.0, 100.0, 50.0]],
        )

    def test_normalized_variants(self) -> None:
        xyxy = np.array([[50.0, 30.0, 150.0, 90.0]])
        b = Boxes(
            xyxy=xyxy,
            cls=np.array([0], dtype=np.int64),
            conf=np.array([0.8]),
            orig_shape=(120, 200),  # (h, w)
        )
        # xyxyn: divide x by w=200, y by h=120
        np.testing.assert_array_almost_equal(b.xyxyn, [[0.25, 0.25, 0.75, 0.75]])
        # xywhn: cx=100/200, cy=60/120, w=100/200, h=60/120
        np.testing.assert_array_almost_equal(b.xywhn, [[0.5, 0.5, 0.5, 0.5]])

    def test_data_packs_xyxy_conf_cls(self) -> None:
        xyxy = np.array([[1.0, 2.0, 3.0, 4.0]])
        b = Boxes(
            xyxy=xyxy,
            cls=np.array([7], dtype=np.int64),
            conf=np.array([0.5]),
            orig_shape=(10, 10),
        )
        np.testing.assert_array_almost_equal(b.data, [[1.0, 2.0, 3.0, 4.0, 0.5, 7.0]])

    def test_empty(self) -> None:
        b = Boxes(
            xyxy=np.empty((0, 4), dtype=np.float64),
            cls=np.empty((0,), dtype=np.int64),
            conf=np.empty((0,), dtype=np.float64),
            orig_shape=(100, 100),
        )
        assert len(b) == 0
        assert b.xywh.shape == (0, 4)
        assert b.xyxyn.shape == (0, 4)
        assert b.data.shape == (0, 6)


class TestProbs:
    def test_top1_and_top1conf(self) -> None:
        data = np.array([0.1, 0.6, 0.2, 0.1])
        p = Probs(data=data)
        assert p.top1 == 1
        assert p.top1conf == pytest.approx(0.6)

    def test_top5_descending(self) -> None:
        data = np.array([0.05, 0.10, 0.50, 0.25, 0.05, 0.05])
        p = Probs(data=data)
        # Top-5 of 6 classes, descending.
        assert p.top5.tolist() == [2, 3, 1, 0, 4]
        np.testing.assert_array_almost_equal(p.top5conf, [0.50, 0.25, 0.10, 0.05, 0.05])

    def test_top5_truncates_when_fewer_classes(self) -> None:
        data = np.array([0.7, 0.3])
        p = Probs(data=data)
        assert p.top5.tolist() == [0, 1]
        assert p.top5conf.shape == (2,)

    def test_empty(self) -> None:
        p = Probs(data=np.empty((0,)))
        assert p.top1 == 0
        assert p.top1conf == 0.0
        assert p.top5.shape == (0,)


class TestMasks:
    def test_iteration(self) -> None:
        m1 = np.full((4, 4), 255, dtype=np.uint8)
        m2 = np.zeros((6, 6), dtype=np.uint8)
        masks = Masks(
            data=(m1, m2),
            xyxy=np.array([[0, 0, 4, 4], [10, 10, 16, 16]]),
            orig_shape=(20, 20),
        )
        assert len(masks) == 2
        assert masks.shape == (2,)
        assert [arr.shape for arr in masks] == [(4, 4), (6, 6)]


class TestDetectionResults:
    def test_iteration_and_indexing(self) -> None:
        bbox = BoundingBox(0, 0, 10, 10)
        crop = np.zeros((10, 10, 3), dtype=np.uint8)
        det = DetectionResult(
            class_id=1, class_name="cat", confidence=0.9, bbox=bbox, cropped_image=crop
        )
        results = DetectionResults(
            boxes=Boxes(
                xyxy=np.array([[0, 0, 10, 10]]),
                cls=np.array([1], dtype=np.int64),
                conf=np.array([0.9]),
                orig_shape=(20, 20),
            ),
            detections=(det,),
            names={1: "cat"},
            orig_img=np.zeros((20, 20, 3), dtype=np.uint8),
            orig_shape=(20, 20),
        )
        assert len(results) == 1
        assert results[0] is det
        assert list(results) == [det]

    def test_speed_default_dict(self) -> None:
        results = DetectionResults(
            boxes=Boxes(
                xyxy=np.empty((0, 4)),
                cls=np.empty((0,), dtype=np.int64),
                conf=np.empty((0,)),
                orig_shape=(0, 0),
            ),
            detections=(),
            names={},
            orig_img=np.zeros((1, 1, 3), dtype=np.uint8),
            orig_shape=(1, 1),
        )
        assert results.speed == {}


class TestClassificationResults:
    def test_aliases_match_top1(self) -> None:
        probs = Probs(data=np.array([0.1, 0.7, 0.2]))
        legacy = ClassificationResult(
            class_id=1,
            class_name="dog",
            confidence=0.7,
            image=np.zeros((4, 4, 3), dtype=np.uint8),
            probabilities=(
                ClassProbability(class_id=1, class_name="dog", probability=0.7),
                ClassProbability(class_id=2, class_name="cat", probability=0.2),
                ClassProbability(class_id=0, class_name="fish", probability=0.1),
            ),
        )
        results = ClassificationResults(
            probs=probs,
            result=legacy,
            names={0: "fish", 1: "dog", 2: "cat"},
            orig_img=np.zeros((4, 4, 3), dtype=np.uint8),
            orig_shape=(4, 4),
        )
        assert results.cls == 1
        assert results.conf == pytest.approx(0.7)
        assert results.name == "dog"
        assert results.probabilities[0].class_name == "dog"

    def test_unknown_class_falls_back_to_class_id_format(self) -> None:
        probs = Probs(data=np.array([0.0, 1.0]))
        legacy = ClassificationResult(
            class_id=1,
            class_name="?",
            confidence=1.0,
            image=np.zeros((1, 1, 3), dtype=np.uint8),
            probabilities=(),
        )
        results = ClassificationResults(
            probs=probs, result=legacy, names={},  # empty names map
            orig_img=np.zeros((1, 1, 3), dtype=np.uint8),
            orig_shape=(1, 1),
        )
        assert results.name == "class_1"


class TestSegmentationResults:
    def test_iteration(self) -> None:
        bbox = BoundingBox(0, 0, 10, 10)
        seg = SegmentationResult(
            class_id=0,
            class_name="person",
            confidence=0.95,
            bbox=bbox,
            mask=np.full((10, 10), 255, dtype=np.uint8),
            segmented_image=np.zeros((10, 10, 3), dtype=np.uint8),
        )
        results = SegmentationResults(
            boxes=Boxes(
                xyxy=np.array([[0, 0, 10, 10]]),
                cls=np.array([0], dtype=np.int64),
                conf=np.array([0.95]),
                orig_shape=(20, 20),
            ),
            masks=Masks(
                data=(seg.mask,),
                xyxy=np.array([[0, 0, 10, 10]]),
                orig_shape=(20, 20),
            ),
            detections=(seg,),
            names={0: "person"},
            orig_img=np.zeros((20, 20, 3), dtype=np.uint8),
            orig_shape=(20, 20),
        )
        assert len(results) == 1
        assert list(results)[0] is seg
        assert results[0] is seg
