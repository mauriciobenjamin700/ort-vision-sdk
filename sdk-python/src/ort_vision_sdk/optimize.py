"""Offline graph optimization: pay ONNX Runtime's optimizer once, at build time.

Every ``InferenceSession`` runs the graph optimizer before it can infer —
constant folding, node fusion, redundant-node elimination. On a YOLO11n-seg that
is most of what session creation costs: measured in Chromium on the WASM
backend, creating the session from the original file took 29 ms and from a
pre-optimized one, loaded with optimization disabled, 12 ms, with identical
inference time afterwards. On a phone, or for a page that builds several
sessions, that difference is the part of start-up nobody needs to wait for.

:func:`optimize_model` writes the optimized graph and marks it, in the model's
own metadata, with the level it was optimized at. The web SDK reads that mark
and loads the file with ``graphOptimizationLevel: "disabled"`` on its own, so a
pre-optimized model is a drop-in replacement for the original. The Python SDK
does not act on the mark — it reads metadata only after the session exists, too
late to change how it is built — but a pre-optimized file still loads and runs
correctly here; pass ``SessionOptions`` with ``ORT_DISABLE_ALL`` to skip the
redundant pass.

This is a build step, like :mod:`ort_vision_sdk.compose`: run it once next to
the export, ship its output.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal

__all__ = ["GRAPH_OPTIMIZATION_KEY", "optimize_model"]

GRAPH_OPTIMIZATION_KEY = "ort_vision_sdk.graph_optimization"
"""Metadata key naming the level a model was optimized at offline.

Mirrored by ``GRAPH_OPTIMIZATION_KEY`` in the web SDK's ``core/session.ts``;
both test suites pin the same string, so a rename on one side fails.
"""

OptimizationLevel = Literal["basic", "extended"]

_METADATA_PROPS_TAG = 0x72
"""Protobuf tag of ``ModelProto.metadata_props``: field 14, length-delimited."""


def optimize_model(
    model: str | Path,
    output: str | Path,
    *,
    level: OptimizationLevel = "extended",
) -> Path:
    """Write a graph-optimized copy of a model, marked as such in its metadata.

    ``"all"`` is deliberately not offered. Beyond ``"extended"`` ONNX Runtime
    applies layout transformations (NCHWc blocking) chosen for the instruction
    set of the CPU doing the optimizing, which another machine — and the WASM
    backend in particular — cannot be assumed to share. ``"extended"`` fusions
    are expressed in standard and ``com.microsoft`` operators that every CPU and
    WASM build implements; the WebGPU backend's coverage of the latter has not
    been measured here.

    The custom metadata of the source (Ultralytics' ``names``, ``imgsz``, ...)
    survives: ONNX Runtime copies it into the optimized file. The mark is then
    appended to the serialized model. Protobuf merges a repeated field spread
    across the stream, so appending one more ``metadata_props`` entry is a valid
    edit that needs no ``onnx`` dependency.

    Args:
        model (str | Path): Source ``.onnx`` file.
        output (str | Path): Where to write the optimized model. Overwritten if
            it exists.
        level (OptimizationLevel): ``"basic"`` (constant folding, redundant
            node elimination) or ``"extended"`` (adds operator fusions).
            Defaults to ``"extended"``.

    Returns:
        Path: ``output``, as a :class:`~pathlib.Path`.

    Raises:
        FileNotFoundError: If ``model`` does not exist.
        ValueError: If ``level`` is not ``"basic"`` or ``"extended"``.

    Warns:
        UserWarning: If ``model`` is quantized (contains ``QuantizeLinear``).
            ONNX Runtime fuses quantized operators at load time in ways the
            offline levels do not reproduce: on an INT8 YOLO11n-seg under WASM
            the pre-optimized file was created 79 ms faster but inferred 19%
            slower (``"basic"``: 85% slower). Worth it only when start-up
            matters more than per-frame time.
    """
    import onnxruntime as ort

    source = Path(model)
    if not source.is_file():
        raise FileNotFoundError(f"Model file not found: {source}")
    levels = {
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    }
    if level not in levels:
        raise ValueError(f"level must be 'basic' or 'extended', got {level!r}.")

    if b"QuantizeLinear" in source.read_bytes():
        warnings.warn(
            f"{source.name} is quantized. Loading a pre-optimized INT8 model with the optimizer "
            "off skips fusions ONNX Runtime only applies at load time: measured under WASM, "
            "creation was 79 ms faster but every inference 19% slower. Ship the quantized "
            "file as is unless start-up matters more than per-frame time.",
            UserWarning,
            stacklevel=2,
        )

    destination = Path(output)
    options = ort.SessionOptions()
    options.graph_optimization_level = levels[level]
    options.optimized_model_filepath = str(destination)
    ort.InferenceSession(str(source), options, providers=["CPUExecutionProvider"])

    with destination.open("ab") as handle:
        handle.write(_metadata_entry(GRAPH_OPTIMIZATION_KEY, level))
    return destination


def _metadata_entry(key: str, value: str) -> bytes:
    """Serialize one ``metadata_props`` entry of a ``ModelProto``.

    Args:
        key (str): Entry key.
        value (str): Entry value.

    Returns:
        bytes: The tagged, length-prefixed ``StringStringEntryProto``.
    """
    key_bytes = key.encode("utf-8")
    value_bytes = value.encode("utf-8")
    entry = (
        b"\x0a"
        + _varint(len(key_bytes))
        + key_bytes
        + b"\x12"
        + _varint(len(value_bytes))
        + value_bytes
    )
    return bytes([_METADATA_PROPS_TAG]) + _varint(len(entry)) + entry


def _varint(value: int) -> bytes:
    """Encode a non-negative integer as a protobuf base-128 varint.

    Args:
        value (int): The integer to encode.

    Returns:
        bytes: Its varint encoding.
    """
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)
