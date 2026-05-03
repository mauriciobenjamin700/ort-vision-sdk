"""Abstract base class shared by all vision tasks."""

from __future__ import annotations

from abc import ABC
from pathlib import Path

import onnxruntime as ort

from ort_vision_sdk.core.session import OrtSession


class VisionTask(ABC):
    """Common foundation for task-oriented vision SDK objects.

    Subclasses (:class:`~ort_vision_sdk.tasks.classifier.Classifier`,
    :class:`~ort_vision_sdk.tasks.detector.Detector`, ...) wire their own
    preprocessing and postprocessing on top of an :class:`OrtSession`. The
    base class only owns the inference session — label resolution lives in
    each task because the way ``num_classes`` is read from the model differs
    per task.
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
