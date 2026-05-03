"""Thin wrapper around ``onnxruntime.InferenceSession`` with typed metadata."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

from ort_vision_sdk.core.exceptions import InferenceError, ModelLoadError
from ort_vision_sdk.core.providers import resolve_providers


class OrtSession:
    """Wrap an ONNX Runtime ``InferenceSession`` with convenient metadata access.

    The wrapper exposes input/output names and shapes, manages execution-provider
    selection, and provides a typed :meth:`run` method that operates on
    dictionaries of NumPy arrays.

    Attributes:
        model_path: Filesystem path of the loaded ONNX model.
        providers: Resolved execution providers, in preference order.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        providers: list[str] | None = None,
        session_options: ort.SessionOptions | None = None,
    ) -> None:
        """Load an ONNX model into an ORT inference session.

        Args:
            model_path: Path to the ``.onnx`` model file.
            providers: Execution providers to use, in preference order. ``None``
                selects the best available provider automatically.
            session_options: Optional ``SessionOptions`` to customize the session
                (graph optimization level, threading, etc.).

        Raises:
            ModelLoadError: If the model file does not exist or cannot be loaded.
            ProviderNotAvailableError: If a requested provider is not installed.
        """
        path = Path(model_path)
        if not path.is_file():
            raise ModelLoadError(f"Model file not found: {path}")

        self.model_path: Path = path
        self.providers: list[str] = resolve_providers(providers)

        try:
            self._session: ort.InferenceSession = ort.InferenceSession(
                str(path),
                sess_options=session_options,
                providers=self.providers,
            )
        except Exception as exc:
            raise ModelLoadError(f"Failed to load ONNX model from {path}: {exc}") from exc

        self._input_names: list[str] = [i.name for i in self._session.get_inputs()]
        self._output_names: list[str] = [o.name for o in self._session.get_outputs()]
        self._input_shapes: list[tuple[int | str, ...]] = [
            tuple(i.shape) for i in self._session.get_inputs()
        ]

    @property
    def input_names(self) -> list[str]:
        """Names of the model's inputs, in declaration order."""
        return list(self._input_names)

    @property
    def input_name(self) -> str:
        """Name of the first (and usually only) input."""
        return self._input_names[0]

    @property
    def output_names(self) -> list[str]:
        """Names of the model's outputs, in declaration order."""
        return list(self._output_names)

    @property
    def input_shapes(self) -> list[tuple[int | str, ...]]:
        """Declared shapes of the model's inputs (dynamic dims appear as strings)."""
        return [tuple(s) for s in self._input_shapes]

    @property
    def input_shape(self) -> tuple[int | str, ...]:
        """Declared shape of the first input."""
        return tuple(self._input_shapes[0])

    @property
    def raw(self) -> ort.InferenceSession:
        """The underlying ``onnxruntime.InferenceSession``, for advanced use cases."""
        return self._session

    def run(
        self,
        feeds: dict[str, np.ndarray],
        *,
        output_names: list[str] | None = None,
    ) -> list[np.ndarray]:
        """Run inference and return the requested outputs.

        Args:
            feeds: Mapping of input name to NumPy array. Keys must match
                :pyattr:`input_names`.
            output_names: Names of the outputs to fetch. ``None`` (default)
                fetches all outputs in declared order.

        Returns:
            A list of NumPy arrays, one per output, in the order requested.

        Raises:
            InferenceError: If ORT raises any error during execution.
        """
        try:
            return self._session.run(output_names, feeds)
        except Exception as exc:
            raise InferenceError(f"Inference failed: {exc}") from exc
