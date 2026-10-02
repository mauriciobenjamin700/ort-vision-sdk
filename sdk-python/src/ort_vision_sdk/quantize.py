"""Static INT8 quantization, calibrated with the SDK's own preprocessing.

Quantizing a vision model is easy to get subtly wrong in one specific place:
the calibration tensors. ONNX Runtime picks each activation's int8 range from
the values it sees during calibration, so they have to be exactly what
inference will feed — a detector's letterboxed ``[0, 1]`` tensor, a
classifier's resized and mean/std-normalized one. A hand-written calibration
loop that resizes instead of letterboxing, or forgets the normalization,
produces a model that loads and runs and is quietly worse. Here the
calibration images go through the same task code ``predict()`` uses.

Measured on a YOLO11n-seg 640x640 (per-channel QDQ, 24 calibration views):

============================  ==========  ==========
                              FP32        INT8
============================  ==========  ==========
File size                     11.7 MB     3.5 MB
Native CPU inference          87 ms       30.5 ms
WASM inference (4 threads)    68.4 ms     52.7 ms
============================  ==========  ==========

The accuracy cost depends on the model and on how representative the
calibration set is: validate the quantized model on your own data before
shipping it.

Like :mod:`ort_vision_sdk.optimize` and :mod:`ort_vision_sdk.compose`, this is
a build step. It needs the ``[quantize]`` extra, because ONNX Runtime's
quantization tooling imports ``onnx``; running the quantized model needs
nothing beyond the base install.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from ort_vision_sdk.io.image import ImageInput, load_image
from ort_vision_sdk.optimize import _metadata_entry

if TYPE_CHECKING:
    from onnxruntime.quantization import CalibrationDataReader

__all__ = ["QUANTIZATION_KEY", "QuantizeTask", "quantize_model"]

QUANTIZATION_KEY = "ort_vision_sdk.quantization"
"""Metadata key :func:`quantize_model` writes, naming the scheme it applied.

The web SDK reads it and keeps a marked model off WebGPU. ONNX Runtime Web's
WebGPU ``DequantizeLinear`` rejects the quantized bias this produces (``scale
and zero-point inputs must have the same rank``), and with the bias left in
float the same model ran but returned different detections than WASM. Mirrored
by ``QUANTIZATION_KEY`` in the web SDK's ``core/session.ts``; both test suites
pin the string.
"""

QuantizeTask = Literal["detect", "segment", "classify"]
"""Which task's preprocessing the calibration images go through."""


def quantize_model(
    model: str | Path,
    output: str | Path,
    calibration: Iterable[ImageInput],
    *,
    task: QuantizeTask | None = None,
    per_channel: bool = True,
) -> Path:
    """Write a statically INT8-quantized copy of a model.

    The output uses the QDQ format — ``QuantizeLinear``/``DequantizeLinear``
    pairs around each quantized operator, uint8 activations and int8 weights —
    which every ONNX Runtime execution provider understands, the WASM backend
    included. Dynamic quantization is not offered: it turns convolutions into
    ``ConvInteger``, which ran *slower* than FP32 on the same model (134 ms vs
    87 ms on CPU).

    The output carries :data:`QUANTIZATION_KEY` in its metadata, valued
    ``"qdq-u8s8-per-channel"`` or ``"qdq-u8s8-per-tensor"``. The web SDK runs
    a marked model on WASM even when WebGPU was asked for — see the key for
    why.

    Do not pass the output to :func:`~ort_vision_sdk.optimize_model` for the
    browser. ONNX Runtime fuses quantized operators at load time in ways the
    offline ``"extended"`` level does not reproduce: the pre-optimized INT8
    model was created 79 ms faster but inferred 19% slower under WASM.

    Args:
        model (str | Path): Source FP32 ``.onnx`` file.
        output (str | Path): Where to write the quantized model. Overwritten
            if it exists.
        calibration (Iterable[ImageInput]): Representative images — paths,
            bytes, arrays or PIL images, anything ``predict()`` accepts. A few
            dozen to a few hundred drawn from the deployment distribution is
            the usual range; the int8 ranges are only as good as this set.
        task (QuantizeTask | None): The task whose preprocessing calibration
            uses. ``None`` (default) reads the ``task`` key Ultralytics bakes
            into the model's metadata.
        per_channel (bool): Quantize weights per output channel rather than
            per tensor. More accurate and, on the model above, also faster
            (30.5 ms vs 44 ms on CPU). Defaults to ``True``.

    Returns:
        Path: ``output``, as a :class:`~pathlib.Path`.

    Raises:
        FileNotFoundError: If ``model`` does not exist.
        ImportError: If the ``[quantize]`` extra is not installed.
        ValueError: If the task cannot be determined or is not one of
            ``"detect"``, ``"segment"``, ``"classify"``; if the model's input
            is not float32; or if ``calibration`` yields no images.
    """
    source = Path(model)
    if not source.is_file():
        raise FileNotFoundError(f"Model file not found: {source}")
    try:
        from onnxruntime.quantization import (
            QuantFormat,
            QuantType,
            quant_pre_process,
            quantize_static,
        )
    except ImportError as exc:
        raise ImportError(
            "quantize_model needs ONNX Runtime's quantization tooling, which imports onnx. "
            'Install it with: pip install "ort-vision-sdk[quantize]"'
        ) from exc

    preprocess, input_name = _task_preprocess(source, task)
    feeds = [preprocess(load_image(image)) for image in calibration]
    if not feeds:
        raise ValueError("calibration yielded no images; pass at least one.")

    destination = Path(output)
    with tempfile.TemporaryDirectory() as scratch:
        prepared = Path(scratch) / "prepared.onnx"
        quant_pre_process(str(source), str(prepared), skip_symbolic_shape=True)
        quantize_static(
            str(prepared),
            str(destination),
            _reader(input_name, feeds),
            quant_format=QuantFormat.QDQ,
            per_channel=per_channel,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
        )
    scheme = "qdq-u8s8-per-channel" if per_channel else "qdq-u8s8-per-tensor"
    with destination.open("ab") as handle:
        handle.write(_metadata_entry(QUANTIZATION_KEY, scheme))
    return destination


def _task_preprocess(
    model: Path,
    task: QuantizeTask | None,
) -> tuple[Callable[[np.ndarray], np.ndarray], str]:
    """Build the task for ``model`` and return its preprocessing.

    Constructing the real task — rather than restating its preprocessing here
    — is what keeps calibration in lockstep with inference: input size read
    off the graph, letterbox fill, classifier normalization resolved from the
    metadata, all of it.

    Args:
        model: Source model.
        task: Requested task, or ``None`` to read it from the metadata.

    Returns:
        A function from an RGB image to the batched input tensor, and the name
        of the input it feeds.

    Raises:
        ValueError: If the task cannot be determined or is unsupported, or if
            the model's input is not float32.
    """
    from ort_vision_sdk.tasks.classifier import Classifier
    from ort_vision_sdk.tasks.detector import Detector
    from ort_vision_sdk.tasks.segmenter import Segmenter

    detector: Detector | Segmenter
    if task is None:
        import onnxruntime as ort

        metadata = (
            ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
            .get_modelmeta()
            .custom_metadata_map
        )
        declared = metadata.get("task")
        if declared not in ("detect", "segment", "classify"):
            raise ValueError(
                f"Cannot tell which preprocessing {model.name} expects: its metadata has "
                f"task={declared!r}. Pass task='detect', 'segment' or 'classify'."
            )
        task = declared
    if task == "classify":
        classifier = Classifier(model, providers=["cpu"])
        _require_float32(classifier.session.input_dtype, model)
        return classifier._preprocess, classifier.session.input_name
    if task == "detect":
        detector = Detector(model, providers=["cpu"])
    elif task == "segment":
        detector = Segmenter(model, providers=["cpu"])
    else:
        raise ValueError(f"task must be 'detect', 'segment' or 'classify', got {task!r}.")
    _require_float32(detector.session.input_dtype, model)
    letterboxed = detector._preprocess
    return (lambda image: letterboxed(image)[0]), detector.session.input_name


def _require_float32(dtype: str, model: Path) -> None:
    """Refuse a model whose input is not float32.

    Args:
        dtype: The input element type as ONNX Runtime spells it.
        model: The model, named in the message.

    Raises:
        ValueError: If ``dtype`` is not ``"tensor(float)"``.
    """
    if dtype != "tensor(float)":
        raise ValueError(
            f"{model.name} takes {dtype} input; quantize_model starts from a float32 export."
        )


def _reader(input_name: str, feeds: list[np.ndarray]) -> CalibrationDataReader:
    """Wrap preprocessed tensors in ONNX Runtime's calibration reader protocol.

    Args:
        input_name: Graph input each tensor feeds.
        feeds: Batched input tensors.

    Returns:
        A reader yielding one feed dict per tensor, then ``None``.
    """
    from onnxruntime.quantization import CalibrationDataReader

    class _Feeds(CalibrationDataReader):  # type: ignore[misc]
        """Yield each calibration tensor once."""

        def __init__(self) -> None:
            """Start at the first tensor."""
            self._items: Iterator[np.ndarray] = iter(feeds)

        def get_next(self) -> dict[str, np.ndarray] | None:
            """Return the next feed, or ``None`` when calibration is done."""
            item = next(self._items, None)
            return None if item is None else {input_name: item}

    return _Feeds()
