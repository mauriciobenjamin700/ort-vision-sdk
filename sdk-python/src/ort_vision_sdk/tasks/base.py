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

import numpy as np

from ort_vision_sdk.core.backend import InferenceBackend
from ort_vision_sdk.core.exceptions import NoDetectionsError
from ort_vision_sdk.core.providers import ProviderSpec
from ort_vision_sdk.core.session import OrtSession
from ort_vision_sdk.dtypes import as_float32, numpy_dtype_for

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
        providers: list[ProviderSpec] | None = None,
        session_options: ort.SessionOptions | None = None,
        backend: InferenceBackend | None = None,
    ) -> None:
        """Load the ONNX model into an inference backend.

        Args:
            model_path: Path to the ``.onnx`` model file. Ignored when
                ``backend`` is provided.
            providers: Execution providers in preference order, as names or
                ``(name, options)`` pairs (see :data:`ProviderSpec`). ``None``
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
        self._feed_dtypes: dict[str, np.dtype] = {
            name: numpy_dtype_for(declared)
            for name, declared in zip(
                self._session.input_names, self._session.input_dtypes, strict=True
            )
        }

    @property
    def session(self) -> InferenceBackend:
        """The inference backend used to run inference.

        Returns the injected :class:`~ort_vision_sdk.core.backend.InferenceBackend`,
        or the default :class:`OrtSession` when none was provided.
        """
        return self._session

    @property
    def input_size(self) -> tuple[int, int]:
        """The ``(width, height)`` this task preprocesses to.

        Raises:
            NotImplementedError: Always; every concrete task overrides it.
        """
        raise NotImplementedError

    def warmup(self, runs: int = 1) -> None:
        """Run the model on zero-filled input, paying one-time costs up front.

        The first inference of a session is not representative. On CUDA it
        allocates the device arena and runs cuDNN's algorithm search; TensorRT
        may build its engine there; even the CPU provider faults in its memory
        arena. Calling this while the service is starting — before the first
        request — moves that cost out of someone's latency. Mirrors the web
        SDK's ``warmup()``.

        The feeds are cast to the element type each input declares, so a
        half-precision export is warmed with the dtype it will really run on.

        Args:
            runs (int): How many warm-up inferences to run. One is usually
                enough; defaults to ``1``.
        """
        feeds = self._as_feeds(self._warmup_feeds())
        for _ in range(runs):
            self._session.run(feeds)

    def _warmup_feeds(self) -> dict[str, np.ndarray]:
        """Zero-filled feeds of the shape :meth:`predict` produces.

        Returns:
            dict[str, np.ndarray]: One ``(1, 3, H, W)`` ``float32`` tensor for
            the image input.
        """
        width, height = self.input_size
        return {self._session.input_name: np.zeros((1, 3, height, width), dtype=np.float32)}

    @property
    def input_dtype(self) -> np.dtype:
        """NumPy dtype the graph's first input is fed with.

        ``float32`` for a normal export, ``float16`` for one exported with
        ``half=True``. Preprocessing always runs in ``float32``; this is the
        type the tensor is cast to at the feed boundary.
        """
        return self._feed_dtypes[self._session.input_name]

    def _as_feeds(self, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Cast each feed to the element type its input declares.

        ONNX Runtime matches feed dtypes against the graph exactly — a
        ``float32`` tensor against a ``half=True`` export fails the run with
        ``Unexpected input data type``. Preprocessing stays in ``float32``
        because that is where the normalization arithmetic belongs, so the
        conversion happens here, once, at the boundary.

        Args:
            feeds (dict[str, np.ndarray]): Preprocessed tensors keyed by input
                name.

        Returns:
            dict[str, np.ndarray]: The same mapping with each array cast to the
            declared dtype. Arrays already carrying it are passed through
            untouched.
        """
        return {
            name: array.astype(self._feed_dtypes[name], copy=False)
            if name in self._feed_dtypes
            else array
            for name, array in feeds.items()
        }

    @staticmethod
    def _as_float32_outputs(outputs: list[np.ndarray]) -> list[np.ndarray]:
        """Bring half-precision outputs back to ``float32`` before decoding.

        Float16 resolves to 0.5 px around coordinate 640 and to 1.0 px around
        1280, so decoding boxes in that type quantises every coordinate to the
        grid before NMS and the scale-back to original-image pixels ever run.
        Widening once here keeps the decoders working at the precision they
        were written for; integer outputs (class ids, detection counts) pass
        through unchanged.

        Args:
            outputs (list[np.ndarray]): Raw arrays as the backend returned them.

        Returns:
            list[np.ndarray]: The same arrays with floating-point ones in
            ``float32``.
        """
        return [as_float32(output) for output in outputs]


def _format_threshold(value: float) -> str:
    """Render a confidence threshold the way the web SDK renders it.

    Python and JavaScript disagree on how a number becomes text. A whole
    threshold is ``1.0`` here and ``1`` there; Python switches to an exponent
    at ``1e-05`` while JavaScript holds off until ``1e-07``. A fused pipeline
    is built once and runs under both runtimes from the same file, so a message
    that quotes the threshold has to quote it identically — otherwise the two
    SDKs describe the same run with two different numbers.

    Six decimals with the trailing zeros trimmed covers every threshold a
    caller can meaningfully set and agrees byte for byte with
    ``formatThreshold`` in the web SDK's ``tasks/base.ts``. The pairing is
    fixed by a shared table in both test suites, so a change on one side that
    is not mirrored on the other fails.

    Args:
        value (float): The threshold to render.

    Returns:
        The threshold as text, without trailing zeros or a trailing dot.
    """
    return f"{value:.6f}".rstrip("0").rstrip(".")


def require_detections(
    count: int,
    *,
    raise_on_empty: bool,
    conf_threshold: float,
    classes: list[int] | None,
    path: str | None,
) -> None:
    """Turn an empty result into an error, when the caller asked for that.

    Shared by every task that can come back with nothing —
    :class:`~ort_vision_sdk.tasks.detector.Detector`,
    :class:`~ort_vision_sdk.tasks.segmenter.Segmenter` and
    :class:`~ort_vision_sdk.tasks.pipeline.DetectClassify` — so the three agree
    on when they raise and on what the message says. The message names the two
    settings that decide the outcome, because "no detections" on its own leaves
    the reader unable to tell a blank image from a threshold set too high.

    Args:
        count (int): How many detections survived every filter.
        raise_on_empty (bool): Whether an empty result is an error for this
            call. ``False`` makes this a no-op.
        conf_threshold (float): The threshold actually applied, after any
            per-call override — reported so the message reflects the run rather
            than the constructor.
        classes (list[int] | None): The class allowlist applied, if any.
        path (str | None): Source path of the image, when the input was one.

    Raises:
        NoDetectionsError: If ``raise_on_empty`` is set and ``count`` is zero.
    """
    if not raise_on_empty or count:
        return
    where = f" in {path}" if path else ""
    narrowed = f" among classes {sorted(classes)}" if classes is not None else ""
    raise NoDetectionsError(
        f"No detections{where}{narrowed}: "
        f"nothing cleared conf_threshold={_format_threshold(conf_threshold)}."
    )
