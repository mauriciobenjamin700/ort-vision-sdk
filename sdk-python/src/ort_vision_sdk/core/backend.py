"""The inference-backend seam that decouples the SDK from ONNX Runtime.

Every task (:class:`~ort_vision_sdk.tasks.detector.Detector`,
:class:`~ort_vision_sdk.tasks.classifier.Classifier`,
:class:`~ort_vision_sdk.tasks.segmenter.Segmenter`) drives inference through a
single object — an :class:`InferenceBackend`. The default implementation is
:class:`~ort_vision_sdk.core.session.OrtSession` (ONNX Runtime in-process), but
because the tasks only ever touch the methods declared here — never
``onnxruntime`` directly — any object that satisfies this protocol can be
injected instead.

That is what makes the SDK runnable where the ``onnxruntime`` Python wheel does
not exist: in a browser the inference is bridged to ``onnxruntime-web`` (JS), and
on Android it is bridged to the native ``onnxruntime-android`` AAR. In both cases
the SDK's preprocessing, postprocessing and result parsing run unchanged in
Python (NumPy); only ``run`` crosses the bridge to the native runtime.

The protocol is intentionally the subset of :class:`OrtSession`'s public surface
that tasks actually use — input/output metadata plus the three ``run`` variants —
and deliberately omits ONNX-Runtime-specific members (such as ``raw``), so an
alternative backend never has to fabricate an ``onnxruntime.InferenceSession``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np

__all__ = ["InferenceBackend"]


@runtime_checkable
class InferenceBackend(Protocol):
    """The inference interface every task depends on.

    Implement this to run the SDK against a runtime other than the in-process
    ONNX Runtime — e.g. ``onnxruntime-web`` in the browser or the
    ``onnxruntime-android`` AAR over a native bridge. The default backend is
    :class:`~ort_vision_sdk.core.session.OrtSession`.

    A backend exposes the model's input/output metadata (so tasks can build
    feeds and infer ``num_classes`` from output shapes) and runs inference on a
    mapping of input name to NumPy array, returning one array per output.
    """

    @property
    def input_names(self) -> list[str]:
        """Names of the model's inputs, in declaration order."""
        ...

    @property
    def input_name(self) -> str:
        """Name of the first (and usually only) input."""
        ...

    @property
    def input_shapes(self) -> list[tuple[int | str, ...]]:
        """Declared shapes of the inputs (dynamic dims appear as strings)."""
        ...

    @property
    def input_shape(self) -> tuple[int | str, ...]:
        """Declared shape of the first input."""
        ...

    @property
    def output_names(self) -> list[str]:
        """Names of the model's outputs, in declaration order."""
        ...

    @property
    def output_shapes(self) -> list[tuple[int | str, ...]]:
        """Declared shapes of the outputs (dynamic dims appear as strings)."""
        ...

    def run(
        self,
        feeds: dict[str, np.ndarray],
        *,
        output_names: list[str] | None = None,
    ) -> list[np.ndarray]:
        """Run inference and return the requested outputs.

        Args:
            feeds: Mapping of input name to NumPy array.
            output_names: Outputs to fetch; ``None`` fetches all in order.

        Returns:
            One NumPy array per requested output, in order.
        """
        ...

    async def async_run(
        self,
        feeds: dict[str, np.ndarray],
        *,
        output_names: list[str] | None = None,
    ) -> list[np.ndarray]:
        """Async variant of :meth:`run` (one worker thread per inference).

        Args:
            feeds: Mapping of input name to NumPy array.
            output_names: Outputs to fetch; ``None`` fetches all in order.

        Returns:
            One NumPy array per requested output, in order.
        """
        ...

    async def ort_async_run(
        self,
        feeds: dict[str, np.ndarray],
        *,
        output_names: list[str] | None = None,
    ) -> list[np.ndarray]:
        """High-concurrency async variant of :meth:`run`.

        Backends without a native async path may simply delegate to
        :meth:`async_run`.

        Args:
            feeds: Mapping of input name to NumPy array.
            output_names: Outputs to fetch; ``None`` fetches all in order.

        Returns:
            One NumPy array per requested output, in order.
        """
        ...
