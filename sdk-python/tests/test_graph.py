"""Tests for reading a model's own declarations (input resolution, class names).

These cover the pure helpers in :mod:`ort_vision_sdk.graph` plus the two places
a task consumes them: the input size it preprocesses to, and the labels it
resolves when the caller supplies none.
"""

from __future__ import annotations

import numpy as np
import pytest

from ort_vision_sdk import Classifier, Detector
from ort_vision_sdk.core.backend import read_metadata
from ort_vision_sdk.graph import model_names, resolve_input_size, spatial_input_size
from tests.test_backend import FakeBackend


class MetadataFakeBackend(FakeBackend):
    """A :class:`FakeBackend` that also declares an input shape and metadata."""

    def __init__(
        self,
        output_shapes: list[tuple[int | str, ...]],
        outputs: list[np.ndarray],
        *,
        input_shape: tuple[int | str, ...] = (1, 3, 224, 224),
        metadata: dict[str, str] | None = None,
    ) -> None:
        """Store the declared input shape and metadata alongside the outputs.

        Args:
            output_shapes: Declared output shapes (drives ``num_classes``).
            outputs: The arrays ``run`` returns, one per output.
            input_shape: Shape the fake graph declares for its image input.
            metadata: Custom metadata map the fake export carries.
        """
        super().__init__(output_shapes, outputs)
        self._input_shape = input_shape
        self._metadata = metadata or {}

    @property
    def input_shapes(self) -> list[tuple[int | str, ...]]:
        """The declared input shape, as a single-entry list."""
        return [self._input_shape]

    @property
    def input_shape(self) -> tuple[int | str, ...]:
        """Shape the fake graph declares for its image input."""
        return self._input_shape

    @property
    def metadata(self) -> dict[str, str]:
        """Custom metadata map the fake export carries."""
        return self._metadata


class TestSpatialInputSize:
    """Reading ``(width, height)`` out of a declared shape."""

    def test_reads_static_nchw_axes(self) -> None:
        """A pinned NCHW shape yields its spatial size, width first."""
        assert spatial_input_size((1, 3, 224, 224)) == (224, 224)
        assert spatial_input_size((1, 3, 480, 640)) == (640, 480)

    def test_rejects_shapes_that_pin_no_resolution(self) -> None:
        """Dynamic axes, wrong rank and no shape all read as undeclared."""
        assert spatial_input_size((1, 3, "height", "width")) is None
        assert spatial_input_size((1, 3, 224)) is None
        assert spatial_input_size(()) is None
        assert spatial_input_size(None) is None


class TestResolveInputSize:
    """Precedence between the graph, the caller and the fallback."""

    def test_graph_wins_over_the_requested_size(self) -> None:
        """A static graph overrides the caller and says so."""
        with pytest.warns(UserWarning, match="224x224"):
            resolved = resolve_input_size(
                graph_shape=(1, 3, 224, 224),
                requested=(640, 640),
                fallback=(640, 640),
            )
        assert resolved == (224, 224)

    def test_stays_quiet_when_the_request_already_matches(
        self, recwarn: pytest.WarningsRecorder
    ) -> None:
        """Agreement between caller and graph warns about nothing."""
        assert resolve_input_size(
            graph_shape=(1, 3, 224, 224),
            requested=(224, 224),
            fallback=(640, 640),
        ) == (224, 224)
        assert len(recwarn) == 0

    def test_falls_back_to_request_then_default_for_a_dynamic_graph(self) -> None:
        """With nothing pinned, the caller decides; with no caller, the default."""
        assert resolve_input_size(
            graph_shape=(1, 3, "h", "w"),
            requested=(512, 512),
            fallback=(640, 640),
        ) == (512, 512)
        assert resolve_input_size(
            graph_shape=(1, 3, "h", "w"),
            requested=None,
            fallback=(640, 640),
        ) == (640, 640)


class TestModelNames:
    """Parsing the ``names`` map an exporter baked into the model."""

    def test_parses_the_ultralytics_repr(self) -> None:
        """The dict repr Ultralytics writes becomes a class map."""
        assert model_names({"names": "{0: 'deworm', 1: 'not_deworm'}"}) == {
            0: "deworm",
            1: "not_deworm",
        }

    def test_rejects_unusable_values(self) -> None:
        """Anything not a contiguous int-keyed string dict is refused whole."""
        assert model_names(None) is None
        assert model_names({}) is None
        assert model_names({"task": "classify"}) is None
        assert model_names({"names": "not a dict"}) is None
        assert model_names({"names": "{0: 'a', 2: 'b'}"}) is None
        assert model_names({"names": "{'0': 'a'}"}) is None
        assert model_names({"names": "{0: 1}"}) is None
        assert model_names({"names": "{}"}) is None

    def test_never_evaluates_the_value(self) -> None:
        """A hostile value is rejected, not executed."""
        assert model_names({"names": "__import__('os').system('true')"}) is None


class TestReadMetadata:
    """Probing a backend for the optional metadata capability."""

    def test_returns_the_map_when_the_backend_has_one(self) -> None:
        """A backend satisfying ``MetadataBackend`` hands its map over."""
        backend = MetadataFakeBackend(
            [(1, 2)], [np.array([[0.2, 0.8]], dtype=np.float32)], metadata={"task": "classify"}
        )
        assert read_metadata(backend) == {"task": "classify"}

    def test_returns_empty_for_a_backend_without_metadata(self) -> None:
        """A backend from before the capability existed still works."""
        backend = FakeBackend([(1, 2)], [np.array([[0.2, 0.8]], dtype=np.float32)])
        assert read_metadata(backend) == {}


class TestTasksReadTheirModel:
    """The two places a task consumes what the model declares."""

    def test_classifier_preprocesses_at_the_declared_size(self) -> None:
        """The graph's 224x224 wins over a 640x640 request."""
        backend = MetadataFakeBackend([(1, 2)], [np.array([[0.2, 0.8]], dtype=np.float32)])
        with pytest.warns(UserWarning, match="224x224"):
            clf = Classifier(
                "unused.onnx", backend=backend, labels=["a", "b"], input_size=(640, 640)
            )
        assert clf.input_size == (224, 224)

    def test_classifier_takes_labels_from_the_model(self) -> None:
        """With no labels passed, the export's own ``names`` are used."""
        backend = MetadataFakeBackend(
            [(1, 2)],
            [np.array([[0.2, 0.8]], dtype=np.float32)],
            metadata={"names": "{0: 'deworm', 1: 'not_deworm'}"},
        )
        clf = Classifier("unused.onnx", backend=backend)
        assert clf.labels == ("deworm", "not_deworm")
        assert clf.names == {0: "deworm", 1: "not_deworm"}

    def test_explicit_labels_override_the_model(self) -> None:
        """A caller-supplied spec wins over what the model declares."""
        backend = MetadataFakeBackend(
            [(1, 2)],
            [np.array([[0.2, 0.8]], dtype=np.float32)],
            metadata={"names": "{0: 'deworm', 1: 'not_deworm'}"},
        )
        clf = Classifier("unused.onnx", backend=backend, labels=["sick", "healthy"])
        assert clf.labels == ("sick", "healthy")

    def test_detector_falls_back_to_coco_without_model_names(self) -> None:
        """A model carrying no ``names`` keeps the COCO preset default."""
        backend = MetadataFakeBackend(
            [(1, 84, 8400)],
            [np.zeros((1, 84, 8400), dtype=np.float32)],
            input_shape=(1, 3, 640, 640),
        )
        det = Detector("unused.onnx", backend=backend)
        assert det.labels[0] == "person"
        assert det.input_size == (640, 640)

    def test_detector_reads_a_single_class_model_from_its_names(self) -> None:
        """A custom export resolves its own label instead of failing on COCO."""
        backend = MetadataFakeBackend(
            [(1, 5, 8400)],
            [np.zeros((1, 5, 8400), dtype=np.float32)],
            input_shape=(1, 3, 640, 640),
            metadata={"names": "{0: 'ocular-mucosa'}"},
        )
        det = Detector("unused.onnx", backend=backend)
        assert det.labels == ("ocular-mucosa",)
