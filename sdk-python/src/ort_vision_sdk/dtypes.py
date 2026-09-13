"""What element type the graph declares, and what NumPy type that means.

A model exported with ``half=True`` declares its input as ``tensor(float16)``,
and ONNX Runtime refuses a ``float32`` feed against it:

``Unexpected input data type. Actual: (tensor(float)) , expected: (tensor(float16))``

The tasks preprocess in ``float32`` because that is what the arithmetic wants —
``(value / 255 - mean) / std`` in half precision loses resolution on the small
differences normalization exists to preserve. So the cast belongs at the feed
boundary: preprocess in ``float32``, hand ONNX Runtime whatever the graph asked
for, and bring the outputs back to ``float32`` before decoding them.

The reverse direction matters just as much, for a quieter reason. Float16
carries 11 bits of mantissa, so its resolution at pixel scale is coarse:
measured with ``np.spacing``, neighbouring values are 0.25 px apart around
320, 0.5 px around 640, 1.0 px around 1280 and 2.0 px around 2048. Decoding
boxes in that type quantises every coordinate to the grid — ``640.3`` is
already ``640.5`` — and the error rides through NMS and the scale-back to
original-image coordinates. So :func:`as_float32` widens every floating-point
output before the decoders touch it.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "ORT_FLOAT",
    "ORT_TO_NUMPY",
    "as_float32",
    "numpy_dtype_for",
]

ORT_FLOAT = "tensor(float)"
"""The element type every task preprocesses to, and the fallback when unknown."""

ORT_TO_NUMPY: dict[str, np.dtype] = {
    "tensor(float)": np.dtype(np.float32),
    "tensor(float16)": np.dtype(np.float16),
    "tensor(double)": np.dtype(np.float64),
    "tensor(bfloat16)": np.dtype(np.float32),
    "tensor(int8)": np.dtype(np.int8),
    "tensor(uint8)": np.dtype(np.uint8),
    "tensor(int16)": np.dtype(np.int16),
    "tensor(uint16)": np.dtype(np.uint16),
    "tensor(int32)": np.dtype(np.int32),
    "tensor(uint32)": np.dtype(np.uint32),
    "tensor(int64)": np.dtype(np.int64),
    "tensor(uint64)": np.dtype(np.uint64),
    "tensor(bool)": np.dtype(np.bool_),
}
"""ONNX Runtime element type names → the NumPy dtype a feed must carry.

``bfloat16`` maps to ``float32`` on purpose: NumPy has no bfloat16, and a
bfloat16 graph accepts a float32 feed through ORT's own conversion.
"""


def numpy_dtype_for(ort_type: str) -> np.dtype:
    """Resolve an ONNX Runtime element type name to the NumPy dtype to feed.

    Args:
        ort_type (str): Element type as ONNX Runtime spells it, e.g.
            ``"tensor(float16)"``.

    Returns:
        np.dtype: The dtype a feed for that input must carry. Unknown names
        resolve to ``float32``, which is what every task produces anyway — an
        unrecognised type should surface as ORT's own type error against the
        real graph, not as a failure to look a string up.
    """
    return ORT_TO_NUMPY.get(ort_type, ORT_TO_NUMPY[ORT_FLOAT])


def as_float32(array: np.ndarray) -> np.ndarray:
    """Return ``array`` as ``float32``, without copying when it already is.

    Args:
        array (np.ndarray): Any model output.

    Returns:
        np.ndarray: The same values in ``float32``. Integer outputs (class ids,
        detection counts) are left alone — they carry no precision to recover
        and are indexed with, not measured.
    """
    if array.dtype.kind != "f" or array.dtype == np.float32:
        return array
    return array.astype(np.float32, copy=False)
