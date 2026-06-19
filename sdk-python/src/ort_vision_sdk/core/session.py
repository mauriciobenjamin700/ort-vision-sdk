"""Thin wrapper around ``onnxruntime.InferenceSession`` with typed metadata."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ort_vision_sdk.core.exceptions import InferenceError, ModelLoadError
from ort_vision_sdk.core.providers import resolve_providers

if TYPE_CHECKING:
    # onnxruntime is imported lazily at runtime (inside ``OrtSession.__init__``) so
    # the rest of the SDK — preprocessing/postprocessing, types, labels — imports
    # in environments without an onnxruntime wheel (e.g. Pyodide/WASM in a browser,
    # where inference is bridged to onnxruntime-web instead).
    import onnxruntime as ort


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
        import onnxruntime as ort

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
        self._output_shapes: list[tuple[int | str, ...]] = [
            tuple(o.shape) for o in self._session.get_outputs()
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
    def output_shapes(self) -> list[tuple[int | str, ...]]:
        """Declared shapes of the model's outputs (dynamic dims appear as strings).

        Exposed so tasks can introspect output metadata (e.g. inferring
        ``num_classes`` from the last static dim) without reaching into the
        backend-specific :attr:`raw` session — which lets an alternative
        :class:`~ort_vision_sdk.core.backend.InferenceBackend` (Pyodide/WASM,
        a native Android bridge) satisfy the same interface.
        """
        return [tuple(s) for s in self._output_shapes]

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
            outputs: list[np.ndarray] = self._session.run(output_names, feeds)
        except Exception as exc:
            raise InferenceError(f"Inference failed: {exc}") from exc
        return outputs

    async def async_run(
        self,
        feeds: dict[str, np.ndarray],
        *,
        output_names: list[str] | None = None,
    ) -> list[np.ndarray]:
        """Async wrapper around :meth:`run` using ``asyncio.to_thread``.

        Off-loads the synchronous ORT call to the asyncio default executor's
        thread pool, so a single inference does not block the event loop.
        This is the right choice for typical async code (FastAPI handlers,
        AnyIO tasks, scripts that intermix I/O and inference): one thread per
        in-flight inference, fully portable across ORT versions.

        For high-throughput concurrency where many awaits should share a
        single thread pool, prefer :meth:`ort_async_run`.

        Args:
            feeds: Mapping of input name to NumPy array.
            output_names: Names of the outputs to fetch. ``None`` (default)
                fetches all outputs in declared order.

        Returns:
            A list of NumPy arrays, one per output, in the order requested.

        Raises:
            InferenceError: If ORT raises any error during execution.
        """
        return await asyncio.to_thread(self.run, feeds, output_names=output_names)

    async def ort_async_run(
        self,
        feeds: dict[str, np.ndarray],
        *,
        output_names: list[str] | None = None,
    ) -> list[np.ndarray]:
        """Async wrapper around ORT's native ``InferenceSession.run_async``.

        Schedules inference on the ONNX Runtime internal thread pool
        (configured via ``intra_op_num_threads`` / ``inter_op_num_threads``
        on your ``SessionOptions``). The returned future resolves when ORT
        invokes its callback on a worker thread; the result is hopped back
        to the event loop via ``loop.call_soon_threadsafe``.

        Use this for **high-concurrency** workloads — dozens or hundreds of
        simultaneous awaits all share the ORT pool, instead of each call
        spawning a Python thread (which is what :meth:`async_run` does). For
        typical one-off async calls, :meth:`async_run` is simpler and equally
        non-blocking.

        Requires ``onnxruntime>=1.16``.

        Args:
            feeds: Mapping of input name to NumPy array.
            output_names: Names of the outputs to fetch. ``None`` (default)
                fetches all outputs in declared order.

        Returns:
            A list of NumPy arrays, one per output.

        Raises:
            InferenceError: If ORT signals an error in the callback, or if
                the installed ORT version does not expose ``run_async``.
        """
        run_async = getattr(self._session, "run_async", None)
        if run_async is None:
            raise InferenceError(
                "InferenceSession.run_async is not available in this onnxruntime "
                "version. Upgrade to onnxruntime>=1.16, or use async_run instead."
            )

        loop = asyncio.get_running_loop()
        future: asyncio.Future[list[np.ndarray]] = loop.create_future()
        fetch = output_names if output_names is not None else self._output_names

        def _callback(
            outputs: list[np.ndarray],
            user_data: object,
            error: str | None,
        ) -> None:
            if future.done():
                return
            if error:
                loop.call_soon_threadsafe(
                    future.set_exception, InferenceError(f"Inference failed: {error}")
                )
            else:
                loop.call_soon_threadsafe(future.set_result, outputs)

        try:
            run_async(fetch, feeds, _callback, None)
        except Exception as exc:
            raise InferenceError(f"Inference failed: {exc}") from exc

        return await future
