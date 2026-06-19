"""Tests for the pluggable :class:`InferenceBackend` seam (v0.4.0).

These prove that a task can be driven by an injected backend — the path the
browser (``onnxruntime-web``) and Android (``onnxruntime-android`` AAR) bridges
use — so inference never touches the in-process ``onnxruntime`` wheel while the
SDK's preprocessing/postprocessing still run in Python.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np

from ort_vision_sdk import (
    Classifier,
    ClassificationResults,
    Detector,
    DetectionResults,
    InferenceBackend,
    OrtSession,
)


class FakeBackend:
    """A minimal in-memory backend that returns canned outputs.

    Implements the :class:`InferenceBackend` protocol without any ONNX Runtime,
    so it stands in for a bridged native/remote runtime in tests.
    """

    def __init__(
        self,
        output_shapes: list[tuple[int | str, ...]],
        outputs: list[np.ndarray],
    ) -> None:
        """Store the declared output shapes and the canned outputs to return.

        Args:
            output_shapes: Declared output shapes (drives ``num_classes``).
            outputs: The arrays :meth:`run` returns, one per output.
        """
        self._output_shapes = output_shapes
        self._outputs = outputs
        self.calls = 0

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
        """One 640x640 RGB input."""
        return [(1, 3, 640, 640)]

    @property
    def input_shape(self) -> tuple[int | str, ...]:
        """Shape of the first input."""
        return (1, 3, 640, 640)

    @property
    def output_names(self) -> list[str]:
        """Single output named ``output0``."""
        return ["output0"]

    @property
    def output_shapes(self) -> list[tuple[int | str, ...]]:
        """Declared output shapes."""
        return self._output_shapes

    def run(
        self,
        feeds: dict[str, np.ndarray],
        *,
        output_names: list[str] | None = None,
    ) -> list[np.ndarray]:
        """Return the canned outputs, recording the call.

        Args:
            feeds: Input feeds (ignored; recorded as a call).
            output_names: Ignored.

        Returns:
            The canned outputs.
        """
        self.calls += 1
        return self._outputs

    async def async_run(
        self,
        feeds: dict[str, np.ndarray],
        *,
        output_names: list[str] | None = None,
    ) -> list[np.ndarray]:
        """Async delegate to :meth:`run`."""
        return self.run(feeds, output_names=output_names)

    async def ort_async_run(
        self,
        feeds: dict[str, np.ndarray],
        *,
        output_names: list[str] | None = None,
    ) -> list[np.ndarray]:
        """Async delegate to :meth:`run`."""
        return self.run(feeds, output_names=output_names)


def test_ort_session_class_exposes_the_backend_surface() -> None:
    """``OrtSession`` structurally provides every ``InferenceBackend`` member."""
    for member in (
        "input_names",
        "input_name",
        "input_shapes",
        "input_shape",
        "output_names",
        "output_shapes",
        "run",
        "async_run",
        "ort_async_run",
    ):
        assert hasattr(OrtSession, member), f"OrtSession missing {member!r}"


def test_fake_backend_satisfies_protocol() -> None:
    """A structural backend is recognized by the runtime-checkable protocol."""
    fake = FakeBackend([(1, 84, 8400)], [np.zeros((1, 84, 8400), dtype=np.float32)])
    assert isinstance(fake, InferenceBackend)


def test_detector_runs_through_injected_backend() -> None:
    """A ``Detector`` drives an injected backend end to end (no ONNX Runtime).

    The fake returns an all-zero YOLO output ``(1, 4+nc, N)``; preprocessing and
    postprocessing run in Python, and the backend's ``run`` is invoked exactly
    once, yielding an (empty) :class:`DetectionResults`.
    """
    fake = FakeBackend([(1, 84, 8400)], [np.zeros((1, 84, 8400), dtype=np.float32)])
    det = Detector("unused.onnx", backend=fake)

    # num_classes inferred from the output shape -> 80 COCO labels.
    assert len(det.labels) == 80
    assert det.session is fake

    results = det.predict(np.zeros((120, 160, 3), dtype=np.uint8))
    assert isinstance(results, list)
    assert isinstance(results[0], DetectionResults)
    assert fake.calls == 1


def test_classifier_runs_through_injected_backend() -> None:
    """A ``Classifier`` drives an injected backend end to end (no ONNX Runtime)."""
    logits = np.zeros((1, 1000), dtype=np.float32)
    logits[0, 7] = 10.0  # a clear top-1
    fake = FakeBackend([(1, 1000)], [logits])
    clf = Classifier("unused.onnx", backend=fake)

    results = clf.predict(np.zeros((64, 64, 3), dtype=np.uint8))
    assert isinstance(results, list)
    assert isinstance(results[0], ClassificationResults)
    assert fake.calls == 1
    assert results[0].probs.top1 == 7


def test_package_and_injected_backend_work_without_onnxruntime() -> None:
    """``import ort_vision_sdk`` + injected-backend inference work with no ORT.

    Runs in a subprocess where ``import onnxruntime`` is forced to fail, proving
    the bridged path (browser/Android) never needs the wheel.
    """
    script = (
        "import sys;"
        "sys.modules['onnxruntime'] = None;"  # any `import onnxruntime` now raises
        "import numpy as np;"
        "import ort_vision_sdk as v;"
        "from test_backend import FakeBackend;"
        "f = FakeBackend([(1, 84, 8400)], [np.zeros((1, 84, 8400), dtype=np.float32)]);"
        "d = v.Detector('unused.onnx', backend=f);"
        "r = d.predict(np.zeros((120, 160, 3), dtype=np.uint8));"
        "assert f.calls == 1;"
        "print('OK')"
    )
    import os
    from pathlib import Path

    tests_dir = Path(__file__).parent
    src_dir = tests_dir.parent / "src"
    env = dict(os.environ)
    # Absolute src on PYTHONPATH so the subprocess imports the package regardless
    # of its cwd; prepend so it wins over any installed copy.
    env["PYTHONPATH"] = os.pathsep.join(
        [str(src_dir), str(tests_dir), env.get("PYTHONPATH", "")]
    )
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        cwd=str(tests_dir),
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert "OK" in proc.stdout
