"""Tests for treating an empty detection result as an error, on request.

Finding nothing is a successful inference — a photo of an empty field is a valid
photo — so the SDK returns an empty envelope by default and every test here that
does not opt in asserts exactly that. ``raise_on_empty`` exists for the opposite
situation: a step whose precondition is that something is there, where carrying
on with zero rows is worse than stopping.

The three tasks that can come back empty share one helper, so the interesting
cases are (a) the helper's own decision and message, and (b) that each task
actually routes its per-call override into it.
"""

from __future__ import annotations

import numpy as np
import pytest

from ort_vision_sdk import Detector, Segmenter, require_detections
from ort_vision_sdk.core.exceptions import NoDetectionsError


class StubBackend:
    """An inference backend returning canned outputs, with no metadata map.

    Deliberately does not implement
    :class:`~ort_vision_sdk.core.backend.MetadataBackend`, so the tasks fall back
    to the labels they are handed rather than reading names off a model.
    """

    def __init__(
        self,
        *,
        outputs: list[np.ndarray],
        output_shapes: list[tuple[int | str, ...]],
    ) -> None:
        """Initialize the stub.

        Args:
            outputs: Arrays returned by every ``run`` variant.
            output_shapes: Declared output shapes, which is where the tasks read
                their class count from.
        """
        self._outputs = outputs
        self._output_shapes = output_shapes

    @property
    def input_names(self) -> list[str]:
        """Names of the model's inputs."""
        return ["images"]

    @property
    def input_name(self) -> str:
        """Name of the first input."""
        return "images"

    @property
    def input_shapes(self) -> list[tuple[int | str, ...]]:
        """Declared input shapes."""
        return [(1, 3, 64, 64)]

    @property
    def input_shape(self) -> tuple[int | str, ...]:
        """Declared shape of the first input."""
        return (1, 3, 64, 64)

    @property
    def output_names(self) -> list[str]:
        """Names of the model's outputs."""
        return [f"output{index}" for index in range(len(self._outputs))]

    @property
    def output_shapes(self) -> list[tuple[int | str, ...]]:
        """Declared output shapes."""
        return list(self._output_shapes)

    def run(
        self,
        feeds: dict[str, np.ndarray],
        *,
        output_names: list[str] | None = None,
    ) -> list[np.ndarray]:
        """Return the canned outputs."""
        return list(self._outputs)

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


def _image() -> np.ndarray:
    """A blank 64x64 HWC uint8 image."""
    return np.zeros((64, 64, 3), dtype=np.uint8)


def _detector_head(*, score: float) -> np.ndarray:
    """Build a ``(1, 4 + 2, 8)`` YOLO head whose best anchor scores ``score``."""
    head = np.zeros((1, 6, 8), dtype=np.float32)
    head[0, :4, 0] = [16.0, 16.0, 12.0, 12.0]
    head[0, 4, 0] = score
    head[0, 5, 0] = score / 3.0
    return head


def _detector(*, score: float, **kwargs: object) -> Detector:
    """Build a Detector over a stub whose single candidate scores ``score``."""
    backend = StubBackend(
        outputs=[_detector_head(score=score)],
        output_shapes=[(1, 6, 8)],
    )
    return Detector("unused.onnx", backend=backend, labels=["cat", "dog"], **kwargs)  # type: ignore[arg-type]


def _segmenter(*, score: float, **kwargs: object) -> Segmenter:
    """Build a Segmenter over a stub whose single candidate scores ``score``.

    The per-anchor tensor carries ``4 + num_classes + 32`` channels, the layout
    a YOLO-seg export produces and the one the task infers its class count from.
    """
    per_anchor = np.zeros((1, 38, 64), dtype=np.float32)
    per_anchor[0, :4, 0] = [16.0, 16.0, 12.0, 12.0]
    per_anchor[0, 4, 0] = score
    prototypes = np.zeros((1, 32, 16, 16), dtype=np.float32)
    backend = StubBackend(
        outputs=[per_anchor, prototypes],
        output_shapes=[(1, 38, 64), (1, 32, 16, 16)],
    )
    return Segmenter("unused.onnx", backend=backend, labels=["cat", "dog"], **kwargs)  # type: ignore[arg-type]


class TestRequireDetections:
    """The shared decision, exercised directly."""

    def test_stays_quiet_when_the_flag_is_off(self) -> None:
        require_detections(0, raise_on_empty=False, conf_threshold=0.25, classes=None, path=None)

    def test_stays_quiet_when_something_was_found(self) -> None:
        require_detections(3, raise_on_empty=True, conf_threshold=0.9, classes=None, path=None)

    def test_raises_on_an_empty_result(self) -> None:
        with pytest.raises(NoDetectionsError):
            require_detections(
                0, raise_on_empty=True, conf_threshold=0.25, classes=None, path=None
            )

    def test_names_the_threshold_that_produced_the_emptiness(self) -> None:
        """A bare "no detections" cannot distinguish a blank image from a bad threshold."""
        with pytest.raises(NoDetectionsError, match="conf_threshold=0.9"):
            require_detections(0, raise_on_empty=True, conf_threshold=0.9, classes=None, path=None)

    def test_names_the_image_when_the_input_was_a_path(self) -> None:
        with pytest.raises(NoDetectionsError, match="in flock.jpg"):
            require_detections(
                0, raise_on_empty=True, conf_threshold=0.25, classes=None, path="flock.jpg"
            )

    def test_names_the_class_filter_when_one_narrowed_the_search(self) -> None:
        with pytest.raises(NoDetectionsError, match=r"among classes \[0, 3\]"):
            require_detections(
                0, raise_on_empty=True, conf_threshold=0.25, classes=[3, 0], path=None
            )


class TestPublicSurface:
    """The helper is reachable without importing a submodule.

    ``VisionTask`` is public, so writing a task of your own is a supported thing
    to do — and such a task needs this helper to raise the same error the
    built-in ones raise. Reaching into ``ort_vision_sdk.tasks.base`` for it left
    the caller either breaking the project's own import convention or writing a
    second wording of the same error.
    """

    def test_exported_from_the_package_root(self) -> None:
        import ort_vision_sdk

        assert "require_detections" in ort_vision_sdk.__all__

    def test_exported_from_the_tasks_barrel(self) -> None:
        from ort_vision_sdk import tasks

        assert "require_detections" in tasks.__all__
        assert tasks.require_detections is require_detections


class TestThresholdFormatting:
    """The threshold in the message is rendered identically in both SDKs.

    The same table lives in the web suite
    (``test/raiseOnEmpty.test.ts``). Python's own ``str(float)`` would print a
    whole threshold as ``1.0`` against JavaScript's ``1``, and would switch to
    ``1e-05`` where JavaScript still writes ``0.00001`` — two SDKs describing one
    run with two numbers. If either side's formatting drifts, one of these two
    tables fails.
    """

    @pytest.mark.parametrize(
        ("threshold", "rendered"),
        [
            (0.0, "0"),
            (1.0, "1"),
            (0.25, "0.25"),
            (0.5, "0.5"),
            (0.9, "0.9"),
            (0.001, "0.001"),
            (0.00001, "0.00001"),
            (0.123456, "0.123456"),
        ],
    )
    def test_renders_the_threshold_like_the_web_sdk(
        self, threshold: float, rendered: str
    ) -> None:
        with pytest.raises(NoDetectionsError) as caught:
            require_detections(
                0, raise_on_empty=True, conf_threshold=threshold, classes=None, path=None
            )

        assert str(caught.value) == f"No detections: nothing cleared conf_threshold={rendered}."


class TestDetector:
    """``Detector`` routes the flag through every entry point."""

    def test_returns_an_empty_envelope_by_default(self) -> None:
        """The default must stay a successful empty result, not an error."""
        result = _detector(score=0.01).predict(_image())[0]

        assert len(result) == 0
        assert result.boxes.xyxy.shape == (0, 4)

    def test_raises_when_the_constructor_asked_for_it(self) -> None:
        with pytest.raises(NoDetectionsError):
            _detector(score=0.01, raise_on_empty=True).predict(_image())

    def test_stays_quiet_when_something_was_detected(self) -> None:
        result = _detector(score=0.9, raise_on_empty=True).predict(_image())[0]

        assert len(result) == 1

    def test_per_call_override_turns_it_on(self) -> None:
        with pytest.raises(NoDetectionsError):
            _detector(score=0.01).predict(_image(), raise_on_empty=True)

    def test_per_call_override_turns_it_off(self) -> None:
        detector = _detector(score=0.01, raise_on_empty=True)

        assert len(detector.predict(_image(), raise_on_empty=False)[0]) == 0

    def test_a_stricter_per_call_threshold_can_empty_the_result(self) -> None:
        """Raising the bar per call must raise, and the message must show that bar."""
        detector = _detector(score=0.5, raise_on_empty=True)

        with pytest.raises(NoDetectionsError, match="conf_threshold=0.8"):
            detector.predict(_image(), conf_threshold=0.8)

    def test_a_class_filter_that_empties_the_result_raises(self) -> None:
        detector = _detector(score=0.9, raise_on_empty=True)

        with pytest.raises(NoDetectionsError, match=r"among classes \[1\]"):
            detector.predict(_image(), classes=[1])

    def test_the_callable_alias_forwards_the_flag(self) -> None:
        with pytest.raises(NoDetectionsError):
            _detector(score=0.01)(_image(), raise_on_empty=True)

    @pytest.mark.parametrize("method", ["async_predict", "ort_async_predict"])
    async def test_async_variants_forward_the_flag(self, method: str) -> None:
        detector = _detector(score=0.01)

        with pytest.raises(NoDetectionsError):
            await getattr(detector, method)(_image(), raise_on_empty=True)


class TestSegmenter:
    """``Segmenter`` behaves identically — same helper, same routing."""

    def test_returns_an_empty_envelope_by_default(self) -> None:
        result = _segmenter(score=0.01).predict(_image())[0]

        assert len(result) == 0

    def test_raises_when_the_constructor_asked_for_it(self) -> None:
        with pytest.raises(NoDetectionsError):
            _segmenter(score=0.01, raise_on_empty=True).predict(_image())

    def test_stays_quiet_when_something_was_detected(self) -> None:
        assert len(_segmenter(score=0.9, raise_on_empty=True).predict(_image())[0]) == 1

    def test_per_call_override_turns_it_off(self) -> None:
        segmenter = _segmenter(score=0.01, raise_on_empty=True)

        assert len(segmenter.predict(_image(), raise_on_empty=False)[0]) == 0

    def test_per_call_override_turns_it_on(self) -> None:
        with pytest.raises(NoDetectionsError):
            _segmenter(score=0.01).predict(_image(), raise_on_empty=True)

    @pytest.mark.parametrize("method", ["async_predict", "ort_async_predict"])
    async def test_async_variants_forward_the_flag(self, method: str) -> None:
        segmenter = _segmenter(score=0.01)

        with pytest.raises(NoDetectionsError):
            await getattr(segmenter, method)(_image(), raise_on_empty=True)
