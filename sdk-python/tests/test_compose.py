"""Tests for fusing a detector and a classifier into a single ONNX pipeline.

The models are synthetic on purpose. A real YOLO export would make the fusion's
output impossible to predict, so the "detector" here emits a constant head with
boxes at known coordinates, and the "classifier" reduces each crop to its
per-channel mean. That makes every stage of the bridge observable from the
outside: the boxes tell us NMS and the ranking worked, and the classifier's
output tells us *which pixels were actually cropped* — the one thing a shape
assertion could never catch, and the thing a half-pixel error in RoiAlign would
silently ruin.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from ort_vision_sdk import DetectClassify
from ort_vision_sdk.compose import build_bridge, fuse_detect_classify
from ort_vision_sdk.core.exceptions import FusionError
from ort_vision_sdk.fusion import FusionSpec

_IMAGE_SIZE = 64
_CROP_SIZE = 8
_DETECTOR_NAMES = "{0: 'cat', 1: 'dog'}"
_CLASSIFIER_NAMES = "{0: 'red', 1: 'green', 2: 'blue'}"


def _write_detector(
    path: Path,
    *,
    opset: int = 17,
    output_rank: int = 3,
    size: int = _IMAGE_SIZE,
) -> Path:
    """Write a detector whose head is a constant with two confident boxes.

    The head declares four anchors: one at ``(8, 8, 24, 24)`` scoring 0.9 for
    class 0, one at ``(40, 40, 56, 56)`` scoring 0.8 for class 1, and two below
    any sane confidence threshold so the fusion has something to discard.

    Args:
        path: Where to write the ``.onnx``.
        opset: Opset to declare, so opset reconciliation can be exercised.
        output_rank: Rank of the declared output. Anything but 3 makes the
            model an invalid detector, which the fusion must reject.
        size: Spatial size of the declared input.

    Returns:
        Path: ``path``, for chaining.
    """
    head = np.zeros((1, 6, 4), dtype=np.float32)
    head[0, :4, 0] = [16, 16, 16, 16]
    head[0, 4, 0], head[0, 5, 0] = 0.9, 0.1
    head[0, :4, 1] = [48, 48, 16, 16]
    head[0, 4, 1], head[0, 5, 1] = 0.1, 0.8
    head[0, :4, 2] = head[0, :4, 3] = [5, 5, 4, 4]
    head[0, 4, 2] = head[0, 5, 2] = head[0, 4, 3] = head[0, 5, 3] = 0.01

    shape = [1, 6, 4] if output_rank == 3 else [1, 6, 4, 1][:output_rank]
    node = helper.make_node(
        "Constant", [], ["head"], value=numpy_helper.from_array(head, name="head_value")
    )
    graph = helper.make_graph(
        [node],
        "synthetic_detector",
        inputs=[helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, size, size])],
        outputs=[helper.make_tensor_value_info("head", TensorProto.FLOAT, shape)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 9
    model.metadata_props.add(key="names", value=_DETECTOR_NAMES)
    onnx.save(model, str(path))
    return path


def _write_edge_detector(path: Path, *, boxes: list[tuple[float, float, float, float]]) -> Path:
    """Write a detector whose head is exactly ``boxes``, all confident, all class 0.

    Separate from :func:`_write_detector` because the interesting cases here are
    boxes that leave the frame, which the shared fixture deliberately does not
    have — its two boxes sit comfortably inside the image so the ranking tests
    read cleanly.

    Args:
        path: Where to write the ``.onnx``.
        boxes: ``(cx, cy, w, h)`` per anchor, in the detector's own pixel space.

    Returns:
        Path: ``path``, for chaining.
    """
    head = np.zeros((1, 5, len(boxes)), dtype=np.float32)
    for index, box in enumerate(boxes):
        head[0, :4, index] = box
        head[0, 4, index] = 0.9

    node = helper.make_node(
        "Constant", [], ["head"], value=numpy_helper.from_array(head, name="head_value")
    )
    graph = helper.make_graph(
        [node],
        "edge_detector",
        inputs=[
            helper.make_tensor_value_info(
                "images", TensorProto.FLOAT, [1, 3, _IMAGE_SIZE, _IMAGE_SIZE]
            )
        ],
        outputs=[helper.make_tensor_value_info("head", TensorProto.FLOAT, [1, 5, len(boxes)])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.save(model, str(path))
    return path


def _write_classifier(path: Path, *, opset: int = 17, with_softmax: bool = False) -> Path:
    """Write a classifier that reports each crop's per-channel mean.

    Being a pure reduction makes it batch-agnostic — which is exactly what the
    fused graph needs — and makes its output readable as "what was in the crop".

    Args:
        path: Where to write the ``.onnx``.
        opset: Opset to declare.
        with_softmax: Append a ``Softmax``, so the fusion's auto-detection of an
            already-normalized output can be exercised.

    Returns:
        Path: ``path``, for chaining.
    """
    nodes = [
        helper.make_node("GlobalAveragePool", ["input"], ["pooled"]),
        helper.make_node("Flatten", ["pooled"], ["flat"], axis=1),
    ]
    final = "flat"
    if with_softmax:
        nodes.append(helper.make_node("Softmax", ["flat"], ["probabilities"], axis=1))
        final = "probabilities"
    graph = helper.make_graph(
        nodes,
        "synthetic_classifier",
        inputs=[
            helper.make_tensor_value_info(
                "input", TensorProto.FLOAT, [1, 3, _CROP_SIZE, _CROP_SIZE]
            )
        ],
        outputs=[helper.make_tensor_value_info(final, TensorProto.FLOAT, [1, 3])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 9
    model.metadata_props.add(key="names", value=_CLASSIFIER_NAMES)
    onnx.save(model, str(path))
    return path


def _corner_image() -> np.ndarray:
    """Build an NCHW image whose top-left 16x16 corner is pure red.

    Used by the clamping tests: a box that runs off the top-left corner clamps
    to exactly this region, so a crop taken from the clamped box averages 1.0
    while a crop taken from the raw box — which reaches into territory outside
    the image — cannot.
    """
    image = np.zeros((1, 3, _IMAGE_SIZE, _IMAGE_SIZE), dtype=np.float32)
    image[0, 0, 0:16, 0:16] = 1.0
    return image


def _marked_tensor() -> np.ndarray:
    """Build the NCHW image whose two box regions are pure red and pure green."""
    image = np.zeros((1, 3, _IMAGE_SIZE, _IMAGE_SIZE), dtype=np.float32)
    image[0, 0, 8:24, 8:24] = 1.0
    image[0, 1, 40:56, 40:56] = 1.0
    return image


def _marked_image() -> np.ndarray:
    """Build the same marked scene as an HWC uint8 image, for the runtime tests."""
    image = np.zeros((_IMAGE_SIZE, _IMAGE_SIZE, 3), dtype=np.uint8)
    image[8:24, 8:24, 0] = 255
    image[40:56, 40:56, 1] = 255
    return image


@pytest.fixture
def models(tmp_path: Path) -> tuple[Path, Path]:
    """A synthetic detector/classifier pair on disk."""
    return _write_detector(tmp_path / "det.onnx"), _write_classifier(tmp_path / "clf.onnx")


def _fuse(models: tuple[Path, Path], tmp_path: Path, **kwargs: object) -> Path:
    """Fuse the pair with identity normalization, so crop means survive unchanged."""
    detector, classifier = models
    output = tmp_path / "fused.onnx"
    defaults: dict[str, object] = {
        "mean": (0.0, 0.0, 0.0),
        "std": (1.0, 1.0, 1.0),
        "sampling_ratio": 1,
        "max_detections": 3,
    }
    defaults.update(kwargs)
    fuse_detect_classify(detector, classifier, output, **defaults)  # type: ignore[arg-type]
    return output


def _session(path: Path) -> ort.InferenceSession:
    """Open a fused model on the CPU provider."""
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


class TestFusedGraph:
    """The shape and content of what the fused graph computes."""

    def test_declares_the_pipeline_io(self, models: tuple[Path, Path], tmp_path: Path) -> None:
        session = _session(_fuse(models, tmp_path))
        assert [i.name for i in session.get_inputs()] == ["images"]
        assert [o.name for o in session.get_outputs()] == [
            "boxes",
            "scores",
            "classes",
            "num_detections",
            "probs",
        ]
        assert [o.shape for o in session.get_outputs()] == [[3, 4], [3], [3], [1], [3, 3]]

    def test_reports_the_detected_boxes_ranked_by_confidence(
        self, models: tuple[Path, Path], tmp_path: Path
    ) -> None:
        session = _session(_fuse(models, tmp_path))
        boxes, scores, classes, count, _ = session.run(None, {"images": _marked_tensor()})

        assert int(count[0]) == 2
        np.testing.assert_allclose(boxes[:2], [[8, 8, 24, 24], [40, 40, 56, 56]])
        np.testing.assert_allclose(scores[:2], [0.9, 0.8], rtol=1e-6)
        assert classes[:2].tolist() == [0, 1]

    def test_classifies_the_pixels_inside_each_box(
        self, models: tuple[Path, Path], tmp_path: Path
    ) -> None:
        """The crops must be the box regions themselves, not shifted neighbours.

        Each region is a single pure channel, so a correct crop reduces to a
        one-hot mean. A half-pixel misalignment or a swapped x/y would bleed
        black pixels in and drag the mean below 1.0.
        """
        session = _session(_fuse(models, tmp_path))
        probs = session.run(None, {"images": _marked_tensor()})[4]

        np.testing.assert_allclose(probs[0], [1.0, 0.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(probs[1], [0.0, 1.0, 0.0], atol=1e-5)

    def test_pads_the_unused_rows(self, models: tuple[Path, Path], tmp_path: Path) -> None:
        session = _session(_fuse(models, tmp_path, max_detections=5))
        boxes, scores, classes, count, probs = session.run(None, {"images": _marked_tensor()})

        assert int(count[0]) == 2
        assert boxes.shape == (5, 4)
        np.testing.assert_allclose(boxes[2:], 0.0)
        np.testing.assert_allclose(scores[2:], 0.0)
        assert classes[2:].tolist() == [0, 0, 0]
        np.testing.assert_allclose(probs[2:], 0.0, atol=1e-5)

    def test_reports_nothing_when_no_box_clears_the_threshold(
        self, models: tuple[Path, Path], tmp_path: Path
    ) -> None:
        """With a fixed row count, an empty result is still a full-shaped one.

        This is the case the padding exists for: the classifier stage still runs
        on ``K`` degenerate crops rather than being handed a zero-row batch.
        """
        session = _session(_fuse(models, tmp_path, conf_threshold=0.99))
        boxes, _, _, count, probs = session.run(None, {"images": _marked_tensor()})

        assert int(count[0]) == 0
        assert boxes.shape == (3, 4)
        assert probs.shape == (3, 3)

    def test_dynamic_mode_emits_exactly_the_surviving_rows(
        self, models: tuple[Path, Path], tmp_path: Path
    ) -> None:
        session = _session(_fuse(models, tmp_path, max_detections=None))
        boxes, _, _, count, probs = session.run(None, {"images": _marked_tensor()})

        assert int(count[0]) == 2
        assert boxes.shape == (2, 4)
        assert probs.shape == (2, 3)

    def test_dynamic_mode_survives_an_empty_selection(
        self, models: tuple[Path, Path], tmp_path: Path
    ) -> None:
        session = _session(_fuse(models, tmp_path, max_detections=None, conf_threshold=0.99))
        boxes, _, _, count, probs = session.run(None, {"images": _marked_tensor()})

        assert int(count[0]) == 0
        assert boxes.shape == (0, 4)
        assert probs.shape == (0, 3)


class TestOriginalCropSource:
    """Cropping from the full-resolution image instead of the letterboxed copy."""

    def test_declares_the_letterbox_inputs(self, models: tuple[Path, Path], tmp_path: Path) -> None:
        session = _session(_fuse(models, tmp_path, crop_source="original"))
        assert [i.name for i in session.get_inputs()] == [
            "images",
            "source_image",
            "letterbox_scale",
            "letterbox_pad",
        ]

    def test_crops_at_native_resolution_after_undoing_the_letterbox(
        self, models: tuple[Path, Path], tmp_path: Path
    ) -> None:
        """The graph must find the same objects in the doubled-size source.

        The detector's boxes live in 64x64 letterbox space; the source is
        128x128 at scale 0.5. If the in-graph inverse transform were wrong the
        crops would land on black background and the means would collapse.
        """
        session = _session(_fuse(models, tmp_path, crop_source="original"))
        source = np.zeros((1, 3, 128, 128), dtype=np.float32)
        source[0, 0, 16:48, 16:48] = 1.0
        source[0, 1, 80:112, 80:112] = 1.0

        boxes, _, _, count, probs = session.run(
            None,
            {
                "images": _marked_tensor(),
                "source_image": source,
                "letterbox_scale": np.asarray([0.5], dtype=np.float32),
                "letterbox_pad": np.asarray([0.0, 0.0], dtype=np.float32),
            },
        )

        assert int(count[0]) == 2
        np.testing.assert_allclose(boxes[:2], [[8, 8, 24, 24], [40, 40, 56, 56]])
        np.testing.assert_allclose(probs[0], [1.0, 0.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(probs[1], [0.0, 1.0, 0.0], atol=1e-5)

    def test_reports_boxes_in_letterbox_space_like_the_other_source(
        self, models: tuple[Path, Path], tmp_path: Path
    ) -> None:
        """Both crop sources must agree on coordinates, or the runtime would need two undos."""
        other = tmp_path / "other"
        other.mkdir()
        plain = _session(_fuse(models, tmp_path))
        original = _session(_fuse(models, other, crop_source="original"))

        from_plain = plain.run(None, {"images": _marked_tensor()})[0]
        from_original = original.run(
            None,
            {
                "images": _marked_tensor(),
                "source_image": np.zeros((1, 3, 128, 128), dtype=np.float32),
                "letterbox_scale": np.asarray([0.5], dtype=np.float32),
                "letterbox_pad": np.asarray([0.0, 0.0], dtype=np.float32),
            },
        )[0]
        np.testing.assert_allclose(from_plain, from_original)


class TestNormalization:
    """The classifier's own preprocessing, folded into the graph."""

    def test_applies_mean_and_std_to_the_crops(
        self, models: tuple[Path, Path], tmp_path: Path
    ) -> None:
        fused = _fuse(models, tmp_path, mean=(0.5, 0.25, 0.0), std=(2.0, 1.0, 1.0))
        probs = _session(fused).run(None, {"images": _marked_tensor()})[4]

        np.testing.assert_allclose(probs[0], [(1.0 - 0.5) / 2.0, -0.25, 0.0], atol=1e-5)

    def test_applies_the_input_scale_before_centring(
        self, models: tuple[Path, Path], tmp_path: Path
    ) -> None:
        fused = _fuse(models, tmp_path, input_scale=255.0)
        probs = _session(fused).run(None, {"images": _marked_tensor()})[4]

        np.testing.assert_allclose(probs[0], [255.0, 0.0, 0.0], atol=1e-3)


class TestMetadata:
    """What the fused file records about how it must be driven."""

    def test_round_trips_the_spec(self, models: tuple[Path, Path], tmp_path: Path) -> None:
        fused = _fuse(models, tmp_path, crop_source="original", conf_threshold=0.3)
        metadata = _session(fused).get_modelmeta().custom_metadata_map
        spec = FusionSpec.from_metadata(dict(metadata))

        assert spec is not None
        assert spec.input_size == (_IMAGE_SIZE, _IMAGE_SIZE)
        assert spec.crop_size == (_CROP_SIZE, _CROP_SIZE)
        assert spec.crop_source == "original"
        assert spec.max_detections == 3
        assert spec.conf_threshold == pytest.approx(0.3)
        assert spec.detector_names == {0: "cat", 1: "dog"}
        assert spec.classifier_names == {0: "red", 1: "green", 2: "blue"}
        assert spec.apply_softmax is True

    def test_keeps_the_detector_own_metadata(
        self, models: tuple[Path, Path], tmp_path: Path
    ) -> None:
        metadata = _session(_fuse(models, tmp_path)).get_modelmeta().custom_metadata_map
        assert metadata["names"] == _DETECTOR_NAMES

    def test_detects_a_classifier_that_already_normalizes(self, tmp_path: Path) -> None:
        detector = _write_detector(tmp_path / "det.onnx")
        classifier = _write_classifier(tmp_path / "clf.onnx", with_softmax=True)
        model = fuse_detect_classify(detector, classifier, tmp_path / "fused.onnx")

        spec = FusionSpec.from_metadata({e.key: e.value for e in model.metadata_props})
        assert spec is not None
        assert spec.apply_softmax is False

    def test_honours_an_explicit_softmax_override(self, tmp_path: Path) -> None:
        detector = _write_detector(tmp_path / "det.onnx")
        classifier = _write_classifier(tmp_path / "clf.onnx", with_softmax=True)
        model = fuse_detect_classify(
            detector, classifier, tmp_path / "fused.onnx", apply_softmax=True
        )

        spec = FusionSpec.from_metadata({e.key: e.value for e in model.metadata_props})
        assert spec is not None
        assert spec.apply_softmax is True

    def test_names_an_unnamed_stage_from_its_own_class_count(self, tmp_path: Path) -> None:
        """A fused file must carry names even when neither model declares any.

        The runtime cannot count either stage's classes — both heads are buried
        inside the merged graph — so a `None` recorded here becomes a fallback
        with no count behind it, which is the COCO preset. For a 2-class
        detector that silently mislabels every prediction.
        """
        detector = tmp_path / "det.onnx"
        _write_detector(detector)
        stripped = onnx.load(str(detector))
        del stripped.metadata_props[:]
        onnx.save(stripped, str(detector))
        classifier = _write_classifier(tmp_path / "clf.onnx")

        model = fuse_detect_classify(detector, classifier, tmp_path / "fused.onnx")
        spec = FusionSpec.from_metadata({e.key: e.value for e in model.metadata_props})

        assert spec is not None
        assert spec.detector_names == {0: "class_0", 1: "class_1"}

    def test_records_explicit_labels_over_the_model_metadata(
        self, models: tuple[Path, Path], tmp_path: Path
    ) -> None:
        fused = _fuse(models, tmp_path, detector_labels=["sheep", "goat"])
        spec = FusionSpec.from_metadata(dict(_session(fused).get_modelmeta().custom_metadata_map))
        assert spec is not None
        assert spec.detector_names == {0: "sheep", 1: "goat"}


class TestOpsetReconciliation:
    """Two models exported at different opsets still fuse."""

    def test_upgrades_the_older_stage(self, tmp_path: Path) -> None:
        detector = _write_detector(tmp_path / "det.onnx", opset=17)
        classifier = _write_classifier(tmp_path / "clf.onnx", opset=13)
        model = fuse_detect_classify(detector, classifier, tmp_path / "fused.onnx")

        default = next(entry for entry in model.opset_import if entry.domain in ("", "ai.onnx"))
        assert default.version >= 17

    def test_lifts_both_stages_to_the_bridge_minimum(self, tmp_path: Path) -> None:
        """RoiAlign needs opset 16, so a pair exported below it must be raised."""
        detector = _write_detector(tmp_path / "det.onnx", opset=13)
        classifier = _write_classifier(tmp_path / "clf.onnx", opset=13)
        model = fuse_detect_classify(detector, classifier, tmp_path / "fused.onnx")

        default = next(entry for entry in model.opset_import if entry.domain in ("", "ai.onnx"))
        assert default.version >= 16


class TestRejections:
    """What the fusion refuses, and why."""

    def test_missing_detector_file(self, tmp_path: Path) -> None:
        classifier = _write_classifier(tmp_path / "clf.onnx")
        with pytest.raises(FusionError, match="detector model not found"):
            fuse_detect_classify(tmp_path / "absent.onnx", classifier)

    def test_missing_classifier_file(self, tmp_path: Path) -> None:
        detector = _write_detector(tmp_path / "det.onnx")
        with pytest.raises(FusionError, match="classifier model not found"):
            fuse_detect_classify(detector, tmp_path / "absent.onnx")

    def test_detector_head_of_the_wrong_rank(self, tmp_path: Path) -> None:
        detector = _write_detector(tmp_path / "det.onnx", output_rank=2)
        classifier = _write_classifier(tmp_path / "clf.onnx")
        with pytest.raises(FusionError, match="anchor-free YOLO layout"):
            fuse_detect_classify(detector, classifier)

    def test_dynamic_detector_resolution_without_an_override(self, tmp_path: Path) -> None:
        detector = tmp_path / "det.onnx"
        _write_detector(detector)
        model = onnx.load(str(detector))
        model.graph.input[0].type.tensor_type.shape.dim[2].dim_param = "height"
        model.graph.input[0].type.tensor_type.shape.dim[3].dim_param = "width"
        onnx.save(model, str(detector))
        classifier = _write_classifier(tmp_path / "clf.onnx")

        with pytest.raises(FusionError, match="Pass input_size"):
            fuse_detect_classify(detector, classifier)

    def test_multi_input_classifier(self, tmp_path: Path) -> None:
        detector = _write_detector(tmp_path / "det.onnx")
        classifier = tmp_path / "clf.onnx"
        _write_classifier(classifier)
        model = onnx.load(str(classifier))
        model.graph.input.append(helper.make_tensor_value_info("extra", TensorProto.FLOAT, [1, 3]))
        onnx.save(model, str(classifier))

        with pytest.raises(FusionError, match="exactly one input"):
            fuse_detect_classify(detector, classifier)


class TestBridgeArguments:
    """Argument validation on the bridge builder itself."""

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"max_detections": 0}, "max_detections"),
            ({"max_boxes_per_class": 0}, "max_boxes_per_class"),
            ({"crop_size": (0, 8)}, "crop_size"),
            ({"channels": 0}, "channels"),
            ({"std": (1.0, 0.0, 1.0)}, "std"),
        ],
    )
    def test_rejects_out_of_range_values(self, kwargs: dict[str, object], message: str) -> None:
        defaults: dict[str, object] = {
            "detector_output": "head",
            "detector_input": "images",
            "classifier_input": "input",
            "crop_size": (8, 8),
            "channels": 3,
            "crop_source": "detector_input",
            "max_detections": 4,
            "conf_threshold": 0.25,
            "iou_threshold": 0.45,
            "max_boxes_per_class": 16,
            "mean": (0.0, 0.0, 0.0),
            "std": (1.0, 1.0, 1.0),
            "input_scale": 1.0,
            "sampling_ratio": 0,
        }
        defaults.update(kwargs)
        with pytest.raises(ValueError, match=message):
            build_bridge(**defaults)  # type: ignore[arg-type]


class TestRuntimeIntegration:
    """The fused file driven through :class:`DetectClassify`."""

    def test_reports_detections_with_their_classification(
        self, models: tuple[Path, Path], tmp_path: Path
    ) -> None:
        pipeline = DetectClassify(_fuse(models, tmp_path), providers=["cpu"])
        result = pipeline.predict(_marked_image())[0]

        assert len(result) == 2
        first, second = result[0], result[1]
        assert (first.name, second.name) == ("cat", "dog")
        assert first.classification is not None
        assert second.classification is not None
        assert first.classification.name == "red"
        assert second.classification.name == "green"

    def test_maps_boxes_back_through_the_letterbox(
        self, models: tuple[Path, Path], tmp_path: Path
    ) -> None:
        """A source image twice the model's size must yield boxes at twice the coordinates."""
        pipeline = DetectClassify(_fuse(models, tmp_path), providers=["cpu"])
        doubled = np.zeros((128, 128, 3), dtype=np.uint8)
        doubled[16:48, 16:48, 0] = 255

        result = pipeline.predict(doubled)[0]
        assert result[0].box.as_int_xyxy() == (16, 16, 48, 48)
        assert result[0].cropped_image.shape == (32, 32, 3)

    def test_carries_the_two_label_spaces_separately(
        self, models: tuple[Path, Path], tmp_path: Path
    ) -> None:
        pipeline = DetectClassify(_fuse(models, tmp_path), providers=["cpu"])
        result = pipeline.predict(_marked_image())[0]

        assert result.names == {0: "cat", 1: "dog"}
        assert result.classifier_names == {0: "red", 1: "green", 2: "blue"}

    def test_ignores_the_padded_rows(self, models: tuple[Path, Path], tmp_path: Path) -> None:
        pipeline = DetectClassify(_fuse(models, tmp_path, max_detections=8), providers=["cpu"])
        assert len(pipeline.predict(_marked_image())[0]) == 2

    def test_rejects_a_model_that_is_not_a_pipeline(
        self, models: tuple[Path, Path], tmp_path: Path
    ) -> None:
        with pytest.raises(FusionError, match="no fused-pipeline metadata"):
            DetectClassify(models[0], providers=["cpu"])


class TestBoxClamping:
    """What the pipeline reports versus what it actually cropped."""

    def _fused_edge(
        self,
        tmp_path: Path,
        *,
        boxes: list[tuple[float, float, float, float]],
        **kwargs: object,
    ) -> Path:
        """Fuse an edge-box detector with the identity classifier.

        Args:
            tmp_path: Directory to write the three models into.
            boxes: ``(cx, cy, w, h)`` anchors for the detector head.
            **kwargs: Overrides forwarded to ``fuse_detect_classify``.

        Returns:
            Path: The fused model.
        """
        detector = _write_edge_detector(tmp_path / "edge_det.onnx", boxes=boxes)
        classifier = _write_classifier(tmp_path / "edge_clf.onnx")
        output = tmp_path / "edge_fused.onnx"
        defaults: dict[str, object] = {
            "mean": (0.0, 0.0, 0.0),
            "std": (1.0, 1.0, 1.0),
            "sampling_ratio": 1,
            "max_detections": 2,
        }
        defaults.update(kwargs)
        fuse_detect_classify(detector, classifier, output, **defaults)  # type: ignore[arg-type]
        return output

    def test_reports_the_clamped_box_that_was_cropped(self, tmp_path: Path) -> None:
        """A box running off the top-left corner is reported as the region cropped."""
        fused = self._fused_edge(tmp_path, boxes=[(0.0, 0.0, 32.0, 32.0)])
        boxes, _, _, count, probs = _session(fused).run(None, {"images": _corner_image()})

        assert count[0] == 1
        np.testing.assert_allclose(boxes[0], [0.0, 0.0, 16.0, 16.0])
        np.testing.assert_allclose(probs[0][0], 1.0, atol=1e-5)

    def test_clamps_the_far_corner_to_the_full_extent(self, tmp_path: Path) -> None:
        """The far edge clips to ``W``, not ``W - 1``: the last column is valid image."""
        fused = self._fused_edge(tmp_path, boxes=[(56.0, 56.0, 32.0, 32.0)])
        boxes, _, _, _, _ = _session(fused).run(None, {"images": _corner_image()})

        np.testing.assert_allclose(boxes[0], [40.0, 40.0, 64.0, 64.0])

    def test_leaves_a_box_inside_the_frame_untouched(self, tmp_path: Path) -> None:
        fused = self._fused_edge(tmp_path, boxes=[(16.0, 16.0, 16.0, 16.0)])
        boxes, _, _, _, _ = _session(fused).run(None, {"images": _corner_image()})

        np.testing.assert_allclose(boxes[0], [8.0, 8.0, 24.0, 24.0])

    def test_keeps_the_padded_rows_zero(self, tmp_path: Path) -> None:
        """Clamping runs before padding, so surplus rows are still all-zero."""
        fused = self._fused_edge(tmp_path, boxes=[(0.0, 0.0, 32.0, 32.0)], max_detections=4)
        boxes, scores, classes, count, _ = _session(fused).run(None, {"images": _corner_image()})

        assert count[0] == 1
        np.testing.assert_allclose(boxes[1:], 0.0)
        np.testing.assert_allclose(scores[1:], 0.0)
        np.testing.assert_array_equal(classes[1:], 0)

    def test_reports_letterbox_coordinates_for_the_original_crop_source(
        self, tmp_path: Path
    ) -> None:
        """With ``crop_source="original"`` the reported box is the ROI, mapped back."""
        fused = self._fused_edge(tmp_path, boxes=[(0.0, 0.0, 32.0, 32.0)], crop_source="original")
        source = np.zeros((1, 3, _IMAGE_SIZE * 2, _IMAGE_SIZE * 2), dtype=np.float32)
        source[0, 0, 0:32, 0:32] = 1.0
        boxes, _, _, _, probs = _session(fused).run(
            None,
            {
                "images": _corner_image(),
                "source_image": source,
                "letterbox_scale": np.asarray([0.5], dtype=np.float32),
                "letterbox_pad": np.asarray([0.0, 0.0], dtype=np.float32),
            },
        )

        np.testing.assert_allclose(boxes[0], [0.0, 0.0, 16.0, 16.0])
        np.testing.assert_allclose(probs[0][0], 1.0, atol=1e-5)

