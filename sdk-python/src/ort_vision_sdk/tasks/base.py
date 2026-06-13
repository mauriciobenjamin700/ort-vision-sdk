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

from ort_vision_sdk.core.session import OrtSession

if TYPE_CHECKING:
    # Only used in type annotations; imported lazily by OrtSession at runtime so
    # the package imports without an onnxruntime wheel (Pyodide/WASM).
    import onnxruntime as ort


class VisionTask:
    """Common foundation for task-oriented vision SDK objects.

    Subclasses (:class:`~ort_vision_sdk.tasks.classifier.Classifier`,
    :class:`~ort_vision_sdk.tasks.detector.Detector`, ...) wire their own
    preprocessing and postprocessing on top of an :class:`OrtSession`. The
    base class only owns the inference session — label resolution lives in
    each task because the way ``num_classes`` is read from the model differs
    per task.

    This class is not intended to be instantiated directly; instantiate one
    of the concrete task classes instead.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        providers: list[str] | None = None,
        session_options: ort.SessionOptions | None = None,
    ) -> None:
        """Load the ONNX model into an :class:`OrtSession`.

        Args:
            model_path: Path to the ``.onnx`` model file.
            providers: Execution providers in preference order. ``None``
                (default) auto-selects the best available accelerator.
            session_options: Optional ORT session options.
        """
        self._session: OrtSession = OrtSession(
            model_path,
            providers=providers,
            session_options=session_options,
        )

    @property
    def session(self) -> OrtSession:
        """The underlying :class:`OrtSession` used to run inference."""
        return self._session
