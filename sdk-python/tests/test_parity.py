"""Regression tests over the shared Python/Web parity fixtures.

The fixtures in ``fixtures/parity/`` record inputs and the outputs the Python
implementation produced when they were generated. Reading them back here turns
any change in the Python postprocessing maths into a failing test, and
``sdk-js-web/test/parity.test.ts`` reads the same file to prove the web
implementation agrees.

The fixture lives at the repository root rather than inside the package,
because it belongs to both artifacts. That puts it outside an installed sdist,
so the module skips itself when the file is absent instead of failing a test
run that has no way to succeed.

Regenerate with ``scripts/gen_parity_fixtures.py``. Because the expectations
come from this implementation, a regenerated diff is a change of reference and
has to be read, not merely committed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ort_vision_sdk.graph import model_names
from ort_vision_sdk.postprocess.classification import softmax, topk
from ort_vision_sdk.postprocess.detection import batched_nms, decode_yolo, nms
from ort_vision_sdk.postprocess.segmentation import decode_yolo_seg

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "parity" / "postprocess.json"

pytestmark = pytest.mark.skipif(
    not FIXTURE.is_file(),
    reason=f"parity fixture not available at {FIXTURE} (expected outside an installed sdist)",
)

_DATA: dict[str, Any] = json.loads(FIXTURE.read_text(encoding="utf-8")) if FIXTURE.is_file() else {}


def cases(section: str) -> list[Any]:
    """Return the fixture cases for a section, as pytest parameters.

    Args:
        section: Top-level fixture key, e.g. ``"nms"``.

    Returns:
        A list of ``pytest.param`` objects, each named after its case.
    """
    return [pytest.param(case, id=case["name"]) for case in _DATA.get(section, [])]


def load_tensor(name: str) -> np.ndarray:
    """Materialize one of the fixture's shared tensors as a ``float32`` array."""
    entry = _DATA["tensors"][name]
    return np.asarray(entry["data"], dtype=np.float32).reshape(entry["dims"])


def expected_mask(instance: dict[str, Any]) -> np.ndarray:
    """Rebuild a binary 0/255 mask from a fixture's ``maskBits`` string."""
    bits = instance["maskBits"]
    values = np.frombuffer(bits.encode("ascii"), dtype=np.uint8) - ord("0")
    return (values * 255).astype(np.uint8).reshape(
        instance["maskHeight"], instance["maskWidth"]
    )


class TestNms:
    @pytest.mark.parametrize("case", cases("nms"))
    def test_matches_fixture(self, case: dict[str, Any]) -> None:
        boxes = np.asarray(case["boxes"], dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(case["scores"], dtype=np.float32)

        keep = nms(boxes, scores, case["iouThreshold"])

        assert [int(i) for i in keep] == case["expectedKeep"]


class TestBatchedNms:
    @pytest.mark.parametrize("case", cases("batchedNms"))
    def test_matches_fixture(self, case: dict[str, Any]) -> None:
        boxes = np.asarray(case["boxes"], dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(case["scores"], dtype=np.float32)
        class_ids = np.asarray(case["classIds"], dtype=np.int64)

        keep = batched_nms(boxes, scores, class_ids, case["iouThreshold"])

        assert [int(i) for i in keep] == case["expectedKeep"]


class TestDecodeYolo:
    @pytest.mark.parametrize("case", cases("decodeYolo"))
    def test_matches_fixture(self, case: dict[str, Any]) -> None:
        options = case["options"]

        decoded = decode_yolo(
            load_tensor(case["tensor"]),
            original_size=(options["originalWidth"], options["originalHeight"]),
            pad=(options["padLeft"], options["padTop"]),
            scale=options["scale"],
            conf_threshold=options["confThreshold"],
            iou_threshold=options["iouThreshold"],
            max_detections=options["maxDetections"],
        )

        assert len(decoded) == len(case["expected"])
        for (bbox, class_id, confidence), want in zip(decoded, case["expected"], strict=True):
            assert class_id == want["classId"]
            assert confidence == pytest.approx(want["confidence"], rel=1e-6)
            np.testing.assert_allclose(bbox.xyxy, want["bbox"], rtol=1e-6)


class TestDecodeYoloSeg:
    @pytest.mark.parametrize("case", cases("decodeYoloSeg"))
    def test_matches_fixture(self, case: dict[str, Any]) -> None:
        options = case["options"]

        decoded = decode_yolo_seg(
            load_tensor(case["perAnchorTensor"]),
            load_tensor(case["prototypeTensor"]),
            num_classes=options["numClasses"],
            input_size=(options["inputWidth"], options["inputHeight"]),
            original_size=(options["originalWidth"], options["originalHeight"]),
            pad=(options["padLeft"], options["padTop"]),
            scale=options["scale"],
            conf_threshold=options["confThreshold"],
            iou_threshold=options["iouThreshold"],
            max_detections=options["maxDetections"],
            mask_threshold=options["maskThreshold"],
        )

        assert len(decoded) == len(case["expected"])
        for (bbox, class_id, confidence, mask), want in zip(
            decoded, case["expected"], strict=True
        ):
            assert class_id == want["classId"]
            assert confidence == pytest.approx(want["confidence"], rel=1e-6)
            np.testing.assert_allclose(bbox.xyxy, want["bbox"], rtol=1e-6)
            np.testing.assert_array_equal(mask, expected_mask(want))


class TestSoftmax:
    @pytest.mark.parametrize("case", cases("softmax"))
    def test_matches_fixture(self, case: dict[str, Any]) -> None:
        logits = np.asarray(case["logits"], dtype=np.float32)

        np.testing.assert_allclose(softmax(logits), case["expected"], rtol=1e-6)


class TestTopk:
    @pytest.mark.parametrize("case", cases("topk"))
    def test_matches_fixture(self, case: dict[str, Any]) -> None:
        probabilities = np.asarray(case["probabilities"], dtype=np.float32)

        indices, values = topk(probabilities, case["k"])

        assert [int(i) for i in indices] == case["expectedIndices"]
        np.testing.assert_allclose(values, case["expectedValues"], rtol=1e-6)


class TestModelNames:
    @pytest.mark.parametrize("case", cases("modelNames"))
    def test_matches_fixture(self, case: dict[str, Any]) -> None:
        raw = case["raw"]

        parsed = model_names({"names": raw} if raw else {})

        if case["expected"] is None:
            assert parsed is None
        else:
            assert parsed == {int(k): v for k, v in case["expected"].items()}
