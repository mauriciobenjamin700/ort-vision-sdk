"""Tests for the :class:`~ort_vision_sdk.tasks.pipeline.DetectClassify` runtime.

The graph is replaced by a stub backend here, so these tests are about the code
*around* the model: which feeds get built, how the graph's letterboxed boxes are
mapped back onto the caller's image, which rows are read, and how the two label
spaces are kept apart. The graph itself is covered end to end in
``test_compose.py``, against a real fused file.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from ort_vision_sdk import DetectClassify
from ort_vision_sdk.core.exceptions import FusionError
from ort_vision_sdk.fusion import (
    INPUT_IMAGE,
    INPUT_PAD,
    INPUT_SCALE,
    INPUT_SOURCE,
    OUTPUT_BOXES,
    OUTPUT_CLASSES,
    OUTPUT_NUM_DETECTIONS,
    OUTPUT_PROBS,
    OUTPUT_SCORES,
    FusionSpec,
)


class StubBackend:
    """An :class:`~ort_vision_sdk.core.backend.InferenceBackend` returning canned outputs.

    Records the feeds it was handed, so a test can assert on what the runtime
    decided to send rather than only on what it did with the answer.
    """

    def __init__(
        self,
        *,
        outputs: dict[str, np.ndarray],
        metadata: dict[str, str],
        probs_classes: int | str = 3,
    ) -> None:
        """Initialize the stub.

        Args:
            outputs: Canned output arrays, keyed by the pipeline's output names.
            metadata: The custom metadata map the runtime reads its spec from.
            probs_classes: Last dimension declared for the ``probs`` output —
                a string to simulate a graph that leaves it dynamic.
        """
        self._outputs = outputs
        self._metadata = metadata
        self._probs_classes = probs_classes
        self.feeds: dict[str, np.ndarray] = {}

    @property
    def input_names(self) -> list[str]:
        """Names of the pipeline's inputs."""
        return [INPUT_IMAGE, INPUT_SOURCE, INPUT_SCALE, INPUT_PAD]

    @property
    def input_name(self) -> str:
        """Name of the first input."""
        return INPUT_IMAGE

    @property
    def input_shapes(self) -> list[tuple[int | str, ...]]:
        """Declared input shapes."""
        return [(1, 3, 64, 64), (1, 3, "h", "w"), (1,), (2,)]

    @property
    def input_shape(self) -> tuple[int | str, ...]:
        """Declared shape of the first input."""
        return (1, 3, 64, 64)

    @property
    def output_names(self) -> list[str]:
        """Names of the pipeline's outputs."""
        return list(self._outputs)

    @property
    def output_shapes(self) -> list[tuple[int | str, ...]]:
        """Declared output shapes, with ``probs`` carrying the class count."""
        return [
            (2, 4),
            (2,),
            (2,),
            (1,),
            (2, self._probs_classes),
        ]

    @property
    def metadata(self) -> dict[str, str]:
        """The model's custom metadata map."""
        return dict(self._metadata)

    def run(
        self,
        feeds: dict[str, np.ndarray],
        *,
        output_names: list[str] | None = None,
    ) -> list[np.ndarray]:
        """Record the feeds and return the canned outputs in the requested order."""
        self.feeds = dict(feeds)
        names = output_names if output_names is not None else list(self._outputs)
        return [self._outputs[name] for name in names]

    async def async_run(
        self,
        feeds: dict[str, np.ndarray],
        *,
        output_names: list[str] | None = None,
    ) -> list[np.ndarray]:
        """Async passthrough to :meth:`run`."""
        return self.run(feeds, output_names=output_names)

    async def ort_async_run(
        self,
        feeds: dict[str, np.ndarray],
        *,
        output_names: list[str] | None = None,
    ) -> list[np.ndarray]:
        """Async passthrough to :meth:`run`."""
        return self.run(feeds, output_names=output_names)


def _spec(**overrides: object) -> FusionSpec:
    """Build a pipeline spec with defaults matching the stub's canned outputs."""
    defaults: dict[str, object] = {
        "input_size": (64, 64),
        "crop_size": (8, 8),
        "crop_source": "detector_input",
        "max_detections": 4,
        "conf_threshold": 0.25,
        "iou_threshold": 0.45,
        "apply_softmax": True,
        "detector_names": {0: "cat", 1: "dog"},
        "classifier_names": {0: "red", 1: "green", 2: "blue"},
        "sdk_version": "0.0.0",
    }
    defaults.update(overrides)
    return FusionSpec(**defaults)  # type: ignore[arg-type]


def _outputs(
    *,
    boxes: np.ndarray | None = None,
    scores: np.ndarray | None = None,
    classes: np.ndarray | None = None,
    count: int = 2,
    probs: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Build a canned output set with four rows, two of them real."""
    return {
        OUTPUT_BOXES: (
            boxes
            if boxes is not None
            else np.asarray(
                [[8, 8, 24, 24], [40, 40, 56, 56], [0, 0, 0, 0], [0, 0, 0, 0]],
                dtype=np.float32,
            )
        ),
        OUTPUT_SCORES: (
            scores if scores is not None else np.asarray([0.9, 0.4, 0.0, 0.0], dtype=np.float32)
        ),
        OUTPUT_CLASSES: (
            classes if classes is not None else np.asarray([0, 1, 0, 0], dtype=np.int64)
        ),
        OUTPUT_NUM_DETECTIONS: np.asarray([count], dtype=np.int64),
        OUTPUT_PROBS: (
            probs
            if probs is not None
            else np.asarray(
                [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                dtype=np.float32,
            )
        ),
    }


def _pipeline(backend: StubBackend, **kwargs: object) -> DetectClassify:
    """Construct a pipeline over the stub backend."""
    return DetectClassify("unused.onnx", backend=backend, **kwargs)  # type: ignore[arg-type]


def _image(size: int = 64) -> np.ndarray:
    """A blank square HWC uint8 image."""
    return np.zeros((size, size, 3), dtype=np.uint8)


class TestConstruction:
    """Reading the pipeline's configuration out of the model."""

    def test_rejects_a_model_without_pipeline_metadata(self) -> None:
        backend = StubBackend(outputs=_outputs(), metadata={"names": "{0: 'cat'}"})
        with pytest.raises(FusionError, match="no fused-pipeline metadata"):
            _pipeline(backend)

    def test_rejects_a_pipeline_missing_a_contract_output(self) -> None:
        """A load-time failure beats a per-call one when the file is simply wrong."""
        outputs = _outputs()
        del outputs[OUTPUT_PROBS]
        backend = StubBackend(outputs=outputs, metadata=_spec().to_metadata())

        with pytest.raises(FusionError, match="does not declare probs"):
            _pipeline(backend)

    def test_reads_the_spec_from_the_model(self) -> None:
        backend = StubBackend(outputs=_outputs(), metadata=_spec().to_metadata())
        pipeline = _pipeline(backend)

        assert pipeline.spec.crop_size == (8, 8)
        assert pipeline.input_size == (64, 64)

    def test_exposes_both_label_spaces(self) -> None:
        backend = StubBackend(outputs=_outputs(), metadata=_spec().to_metadata())
        pipeline = _pipeline(backend)

        assert pipeline.names == {0: "cat", 1: "dog"}
        assert pipeline.classifier_names == {0: "red", 1: "green", 2: "blue"}

    def test_falls_back_to_coco_for_an_unnamed_detection_stage(self) -> None:
        metadata = _spec(detector_names=None).to_metadata()
        pipeline = _pipeline(StubBackend(outputs=_outputs(), metadata=metadata))

        assert pipeline.labels[0] == "person"
        assert len(pipeline.labels) == 80

    def test_generates_names_for_an_unnamed_classification_stage(self) -> None:
        """The class count comes from the graph's ``probs`` output, not from guesswork."""
        metadata = _spec(classifier_names=None).to_metadata()
        pipeline = _pipeline(StubBackend(outputs=_outputs(), metadata=metadata))

        assert pipeline.classifier_labels == ("class_0", "class_1", "class_2")

    def test_caller_labels_win_over_the_recorded_ones(self) -> None:
        backend = StubBackend(outputs=_outputs(), metadata=_spec().to_metadata())
        pipeline = _pipeline(backend, labels=["sheep", "goat"], classifier_labels=["a", "b", "c"])

        assert pipeline.names == {0: "sheep", 1: "goat"}
        assert pipeline.classifier_names == {0: "a", 1: "b", 2: "c"}


class TestFeeds:
    """What the runtime sends into the graph."""

    def test_sends_only_the_letterboxed_image_by_default(self) -> None:
        backend = StubBackend(outputs=_outputs(), metadata=_spec().to_metadata())
        _pipeline(backend).predict(_image())

        assert list(backend.feeds) == [INPUT_IMAGE]
        assert backend.feeds[INPUT_IMAGE].shape == (1, 3, 64, 64)

    def test_sends_the_source_image_and_its_letterbox_for_the_original_crop_source(self) -> None:
        """The graph undoes the letterbox itself, so it needs the parameters that made it."""
        metadata = _spec(crop_source="original").to_metadata()
        backend = StubBackend(outputs=_outputs(), metadata=metadata)
        _pipeline(backend).predict(_image(128))

        assert set(backend.feeds) == {INPUT_IMAGE, INPUT_SOURCE, INPUT_SCALE, INPUT_PAD}
        assert backend.feeds[INPUT_SOURCE].shape == (1, 3, 128, 128)
        assert backend.feeds[INPUT_SCALE].tolist() == [0.5]
        assert backend.feeds[INPUT_PAD].tolist() == [0.0, 0.0]

    def test_letterboxes_a_non_square_image_with_padding(self) -> None:
        metadata = _spec(crop_source="original").to_metadata()
        backend = StubBackend(outputs=_outputs(), metadata=metadata)
        _pipeline(backend).predict(np.zeros((32, 64, 3), dtype=np.uint8))

        assert backend.feeds[INPUT_SCALE].tolist() == [1.0]
        assert backend.feeds[INPUT_PAD].tolist() == [0.0, 16.0]


class TestResults:
    """How the graph's rows become detections."""

    def test_reads_only_the_reported_rows(self) -> None:
        backend = StubBackend(outputs=_outputs(count=2), metadata=_spec().to_metadata())
        assert len(_pipeline(backend).predict(_image())[0]) == 2

    def test_a_count_of_zero_yields_an_empty_envelope(self) -> None:
        backend = StubBackend(outputs=_outputs(count=0), metadata=_spec().to_metadata())
        result = _pipeline(backend).predict(_image())[0]

        assert len(result) == 0
        assert result.boxes.xyxy.shape == (0, 4)

    def test_a_count_beyond_the_rows_is_clamped(self) -> None:
        """A corrupt count must not index past the arrays the graph actually returned."""
        backend = StubBackend(outputs=_outputs(count=99), metadata=_spec().to_metadata())
        assert len(_pipeline(backend).predict(_image())[0]) == 4

    def test_maps_boxes_out_of_letterbox_space(self) -> None:
        backend = StubBackend(outputs=_outputs(), metadata=_spec().to_metadata())
        result = _pipeline(backend).predict(_image(128))[0]

        assert result[0].box.as_int_xyxy() == (16, 16, 48, 48)

    def test_clips_boxes_to_the_source_image(self) -> None:
        boxes = np.asarray(
            [[-20, -20, 200, 200], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float32
        )
        backend = StubBackend(
            outputs=_outputs(boxes=boxes, count=1), metadata=_spec().to_metadata()
        )
        result = _pipeline(backend).predict(_image())[0]

        assert result[0].box.as_int_xyxy() == (0, 0, 64, 64)

    def test_carries_the_crop_on_each_detection(self) -> None:
        backend = StubBackend(outputs=_outputs(), metadata=_spec().to_metadata())
        result = _pipeline(backend).predict(_image())[0]

        assert result[0].cropped_image.shape == (16, 16, 3)

    def test_exposes_the_bulk_box_view(self) -> None:
        backend = StubBackend(outputs=_outputs(), metadata=_spec().to_metadata())
        result = _pipeline(backend).predict(_image())[0]

        assert result.boxes.xyxy.shape == (2, 4)
        assert result.boxes.cls.tolist() == [0, 1]

    def test_times_every_stage(self) -> None:
        backend = StubBackend(outputs=_outputs(), metadata=_spec().to_metadata())
        result = _pipeline(backend).predict(_image())[0]

        assert set(result.speed) == {"load", "preprocess", "inference", "postprocess"}


class TestClassification:
    """The second stage's answer, attached to each detection."""

    def test_applies_softmax_when_the_spec_says_to(self) -> None:
        backend = StubBackend(outputs=_outputs(), metadata=_spec().to_metadata())
        result = _pipeline(backend).predict(_image())[0]

        classification = result[0].classification
        assert classification is not None
        assert classification.name == "red"
        assert classification.conf == pytest.approx(0.9094, abs=1e-3)

    def test_skips_softmax_for_a_graph_that_already_normalizes(self) -> None:
        probs = np.tile(np.asarray([0.7, 0.2, 0.1], dtype=np.float32), (4, 1))
        backend = StubBackend(
            outputs=_outputs(probs=probs), metadata=_spec(apply_softmax=False).to_metadata()
        )
        result = _pipeline(backend).predict(_image())[0]

        classification = result[0].classification
        assert classification is not None
        assert classification.conf == pytest.approx(0.7)

    def test_each_detection_gets_its_own_row(self) -> None:
        backend = StubBackend(outputs=_outputs(), metadata=_spec().to_metadata())
        result = _pipeline(backend).predict(_image())[0]

        names = [d.classification.name for d in result if d.classification is not None]
        assert names == ["red", "green"]

    def test_truncates_the_probability_tuple_to_top_k(self) -> None:
        backend = StubBackend(outputs=_outputs(), metadata=_spec().to_metadata())
        result = _pipeline(backend).predict(_image(), top_k=1)[0]

        classification = result[0].classification
        assert classification is not None
        assert len(classification.probabilities) == 1

    def test_names_a_class_the_label_map_does_not_cover(self) -> None:
        """A shorter recorded map must degrade to a generated name, not raise."""
        probs = np.zeros((4, 3), dtype=np.float32)
        probs[:, 2] = 1.0
        metadata = _spec(classifier_names={0: "red"}).to_metadata()
        backend = StubBackend(outputs=_outputs(probs=probs), metadata=metadata, probs_classes="n")
        result = _pipeline(backend).predict(_image())[0]

        classification = result[0].classification
        assert classification is not None
        assert classification.name == "class_2"

    def test_carries_the_crop_it_describes(self) -> None:
        backend = StubBackend(outputs=_outputs(), metadata=_spec().to_metadata())
        result = _pipeline(backend).predict(_image())[0]

        detection = result[0]
        assert detection.classification is not None
        assert detection.classification.image.shape == detection.cropped_image.shape


class TestFiltering:
    """Narrowing the result set after the graph has run."""

    def test_drops_detections_below_an_extra_confidence_floor(self) -> None:
        backend = StubBackend(outputs=_outputs(), metadata=_spec().to_metadata())
        result = _pipeline(backend).predict(_image(), conf_threshold=0.5)[0]

        assert [d.name for d in result] == ["cat"]

    def test_keeps_only_the_allowed_detector_classes(self) -> None:
        backend = StubBackend(outputs=_outputs(), metadata=_spec().to_metadata())
        result = _pipeline(backend).predict(_image(), classes=[1])[0]

        assert [d.name for d in result] == ["dog"]

    def test_an_empty_allowlist_keeps_nothing(self) -> None:
        backend = StubBackend(outputs=_outputs(), metadata=_spec().to_metadata())
        assert len(_pipeline(backend).predict(_image(), classes=[])[0]) == 0


class TestAsync:
    """Both async entry points must agree with the synchronous one."""

    @pytest.mark.parametrize("method", ["async_predict", "ort_async_predict"])
    def test_matches_predict(self, method: str) -> None:
        backend = StubBackend(outputs=_outputs(), metadata=_spec().to_metadata())
        pipeline = _pipeline(backend)
        expected = [d.name for d in pipeline.predict(_image())[0]]

        results = asyncio.run(getattr(pipeline, method)(_image()))
        assert [d.name for d in results[0]] == expected


class TestCallableAlias:
    """``pipeline(image)`` must behave exactly like ``pipeline.predict(image)``."""

    def test_call_forwards_to_predict(self) -> None:
        backend = StubBackend(outputs=_outputs(), metadata=_spec().to_metadata())
        pipeline = _pipeline(backend)

        assert [d.name for d in pipeline(_image(), classes=[1])[0]] == ["dog"]
