"""Tests for picking the preprocessing a classifier was actually trained with.

The decision is invisible from the outside — a classifier fed the wrong tensor
returns a prediction of exactly the right shape, just a worse one — so these
tests read the tensor the backend receives rather than the prediction that comes
back. That is the only place the choice is observable.
"""

from __future__ import annotations

import numpy as np
import pytest

from ort_vision_sdk import Classifier
from ort_vision_sdk.normalization import (
    IDENTITY_MEAN,
    IDENTITY_STD,
    IMAGENET_MEAN,
    IMAGENET_STD,
    is_ultralytics_classifier,
    resolve_normalization,
)

_ULTRALYTICS = {"author": "Ultralytics", "task": "classify", "names": "{0: 'a', 1: 'b'}"}
_TORCHVISION: dict[str, str] = {}


class RecordingBackend:
    """A backend that carries metadata and remembers the tensor it was fed."""

    def __init__(self, metadata: dict[str, str], num_classes: int = 2) -> None:
        """Store the metadata to report and size the canned output.

        Args:
            metadata: The custom metadata map to expose.
            num_classes: Width of the logits this backend returns.
        """
        self.metadata: dict[str, str] = dict(metadata)
        self.fed: np.ndarray | None = None
        self._num_classes = num_classes

    @property
    def input_names(self) -> list[str]:
        """Single input named ``images``."""
        return ["images"]

    @property
    def input_name(self) -> str:
        """Name of the first input."""
        return "images"

    @property
    def input_shapes(self) -> list[tuple[int | str, ...]]:
        """One 224x224 RGB input."""
        return [(1, 3, 224, 224)]

    @property
    def input_shape(self) -> tuple[int | str, ...]:
        """Shape of the first input."""
        return (1, 3, 224, 224)

    @property
    def output_names(self) -> list[str]:
        """Single output named ``output0``."""
        return ["output0"]

    @property
    def output_shapes(self) -> list[tuple[int | str, ...]]:
        """Declared output shape, which drives the class count."""
        return [(1, self._num_classes)]

    def run(
        self,
        feeds: dict[str, np.ndarray],
        *,
        output_names: list[str] | None = None,
    ) -> list[np.ndarray]:
        """Record the input tensor and return constant logits.

        Args:
            feeds: Input feeds; the first is kept for inspection.
            output_names: Ignored.

        Returns:
            list[np.ndarray]: One ``(1, num_classes)`` array of zeros.
        """
        self.fed = next(iter(feeds.values()))
        return [np.zeros((1, self._num_classes), dtype=np.float32)]


def _white_image() -> np.ndarray:
    """An all-white HWC uint8 image, so every channel arrives as 1.0 before centring."""
    return np.full((224, 224, 3), 255, dtype=np.uint8)


def _fed_value(backend: RecordingBackend) -> float:
    """The value every pixel of the recorded tensor carries in channel 0."""
    assert backend.fed is not None
    return float(backend.fed[0, 0, 0, 0])


class TestDetection:
    """Reading the family off the model's own export metadata."""

    def test_recognizes_an_ultralytics_classification_export(self) -> None:
        assert is_ultralytics_classifier(_ULTRALYTICS)

    def test_is_case_and_whitespace_insensitive(self) -> None:
        assert is_ultralytics_classifier({"author": " ultralytics ", "task": "Classify"})

    def test_rejects_an_ultralytics_detector(self) -> None:
        assert not is_ultralytics_classifier({"author": "Ultralytics", "task": "detect"})

    def test_rejects_a_model_with_no_metadata(self) -> None:
        assert not is_ultralytics_classifier({})


class TestResolution:
    """Turning a request plus metadata into the numbers to apply."""

    def test_auto_picks_identity_for_ultralytics(self) -> None:
        name, mean, std = resolve_normalization(
            _ULTRALYTICS, normalization="auto", mean=None, std=None
        )

        assert (name, mean, std) == ("ultralytics", IDENTITY_MEAN, IDENTITY_STD)

    def test_auto_picks_imagenet_for_everything_else(self) -> None:
        name, mean, std = resolve_normalization(
            _TORCHVISION, normalization="auto", mean=None, std=None
        )

        assert (name, mean, std) == ("imagenet", IMAGENET_MEAN, IMAGENET_STD)

    def test_none_and_ultralytics_are_the_same_arithmetic(self) -> None:
        as_none = resolve_normalization(
            _TORCHVISION, normalization="none", mean=None, std=None
        )
        as_vendor = resolve_normalization(
            _TORCHVISION, normalization="ultralytics", mean=None, std=None
        )

        assert as_none[1:] == as_vendor[1:]
        assert (as_none[0], as_vendor[0]) == ("none", "ultralytics")

    def test_an_explicit_mean_keeps_the_preset_deviation(self) -> None:
        name, mean, std = resolve_normalization(
            _TORCHVISION, normalization="auto", mean=(0.0, 0.0, 0.0), std=None
        )

        assert (name, mean, std) == ("custom", (0.0, 0.0, 0.0), IMAGENET_STD)

    def test_rejects_a_preset_alongside_explicit_values(self) -> None:
        with pytest.raises(ValueError, match="not both"):
            resolve_normalization(
                _TORCHVISION, normalization="imagenet", mean=(0.0, 0.0, 0.0), std=None
            )

    def test_rejects_an_unknown_preset(self) -> None:
        with pytest.raises(ValueError, match="normalization must be one of"):
            resolve_normalization(
                _TORCHVISION,
                normalization="torchvision",  # type: ignore[arg-type]
                mean=None,
                std=None,
            )

    def test_warns_when_an_ultralytics_model_is_normalized(self) -> None:
        with pytest.warns(UserWarning, match="Ultralytics export"):
            resolve_normalization(
                _ULTRALYTICS, normalization="auto", mean=IMAGENET_MEAN, std=IMAGENET_STD
            )

    def test_stays_quiet_when_the_override_is_the_identity(
        self, recwarn: pytest.WarningsRecorder
    ) -> None:
        resolve_normalization(
            _ULTRALYTICS, normalization="auto", mean=IDENTITY_MEAN, std=IDENTITY_STD
        )

        assert not [w for w in recwarn if issubclass(w.category, UserWarning)]


class TestClassifierPreprocessing:
    """What the task actually feeds the model."""

    def test_an_ultralytics_model_is_fed_raw_values(self) -> None:
        backend = RecordingBackend(_ULTRALYTICS)
        classifier = Classifier("unused.onnx", backend=backend)

        classifier.predict(_white_image())

        assert classifier.normalization == "ultralytics"
        assert _fed_value(backend) == pytest.approx(1.0)

    def test_anything_else_is_fed_imagenet_normalized_values(self) -> None:
        backend = RecordingBackend(_TORCHVISION)
        classifier = Classifier("unused.onnx", backend=backend)

        classifier.predict(_white_image())

        assert classifier.normalization == "imagenet"
        assert _fed_value(backend) == pytest.approx((1.0 - 0.485) / 0.229, abs=1e-5)

    def test_an_explicit_preset_overrides_the_detection(self) -> None:
        backend = RecordingBackend(_ULTRALYTICS)
        classifier = Classifier("unused.onnx", backend=backend, normalization="imagenet")

        classifier.predict(_white_image())

        assert _fed_value(backend) == pytest.approx((1.0 - 0.485) / 0.229, abs=1e-5)

    def test_explicit_values_still_win(self) -> None:
        backend = RecordingBackend(_TORCHVISION)
        classifier = Classifier(
            "unused.onnx", backend=backend, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
        )

        classifier.predict(_white_image())

        assert classifier.normalization == "custom"
        assert _fed_value(backend) == pytest.approx(1.0, abs=1e-5)

    def test_warns_when_normalizing_an_ultralytics_model(self) -> None:
        backend = RecordingBackend(_ULTRALYTICS)

        with pytest.warns(UserWarning, match="Ultralytics export"):
            Classifier("unused.onnx", backend=backend, mean=IMAGENET_MEAN, std=IMAGENET_STD)
