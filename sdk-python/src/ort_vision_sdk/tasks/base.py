"""Common base class shared by all vision tasks.

Provides the ``ONNX Runtime`` session lifecycle (load + provider resolution)
plus the ``session`` property. Each subclass adds its own preprocessing,
postprocessing, label resolution, and ``predict()`` signature, since those
diverge enough between classification, detection and segmentation that
forcing a common abstract method would just push a generic ``Any`` return
type onto callers.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ort_vision_sdk.core.backend import InferenceBackend
from ort_vision_sdk.core.session import OrtSession

if TYPE_CHECKING:
    # Only used in type annotations; imported lazily by OrtSession at runtime so
    # the package imports without an onnxruntime wheel (Pyodide/WASM).
    import onnxruntime as ort


class VisionTask:
    """Common foundation for task-oriented vision SDK objects.

    Subclasses (:class:`~ort_vision_sdk.tasks.classifier.Classifier`,
    :class:`~ort_vision_sdk.tasks.detector.Detector`, ...) wire their own
    preprocessing and postprocessing on top of an
    :class:`~ort_vision_sdk.core.backend.InferenceBackend`. The base class only
    owns the inference backend — label resolution lives in each task because the
    way ``num_classes`` is read from the model differs per task.

    By default the backend is an :class:`OrtSession` (in-process ONNX Runtime).
    Pass ``backend`` to run inference through a different runtime — e.g.
    ``onnxruntime-web`` in the browser, or the ``onnxruntime-android`` AAR over a
    native bridge — while the SDK's preprocessing/postprocessing still run in
    Python. When ``backend`` is given, ``model_path``/``providers``/
    ``session_options`` are ignored (the backend owns model loading).

    This class is not intended to be instantiated directly; instantiate one
    of the concrete task classes instead.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        providers: list[str] | None = None,
        session_options: ort.SessionOptions | None = None,
        backend: InferenceBackend | None = None,
    ) -> None:
        """Load the ONNX model into an inference backend.

        Args:
            model_path: Path to the ``.onnx`` model file. Ignored when
                ``backend`` is provided.
            providers: Execution providers in preference order. ``None``
                (default) auto-selects the best available accelerator. Ignored
                when ``backend`` is provided.
            session_options: Optional ORT session options. Ignored when
                ``backend`` is provided.
            backend: An explicit :class:`~ort_vision_sdk.core.backend.InferenceBackend`
                to run inference through. ``None`` (default) builds an
                :class:`OrtSession` from ``model_path`` — the in-process ONNX
                Runtime path. Provide one to bridge inference to a non-ORT
                runtime (browser/Android) without an ``onnxruntime`` wheel.
        """
        self._session: InferenceBackend = (
            backend
            if backend is not None
            else OrtSession(
                model_path,
                providers=providers,
                session_options=session_options,
            )
        )

    @property
    def session(self) -> InferenceBackend:
        """The inference backend used to run inference.

        Returns the injected :class:`~ort_vision_sdk.core.backend.InferenceBackend`,
        or the default :class:`OrtSession` when none was provided.
        """
        return self._session
