"""Image loading utilities — convert any supported input into a canonical ndarray.

The SDK works internally with ``ImageArray`` (HWC uint8 RGB). :func:`load_image`
is the single entry point that accepts files, bytes, NumPy arrays, or PIL
images and produces the canonical format.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import TypeAlias

import numpy as np
from PIL import Image, UnidentifiedImageError

from ort_vision_sdk.core.exceptions import ImageLoadError
from ort_vision_sdk.types import ImageArray

ImageInput: TypeAlias = str | Path | bytes | np.ndarray | Image.Image
"""Anything the SDK accepts as an image input.

- ``str`` / ``Path``: filesystem path to an image file.
- ``bytes``: raw encoded bytes (PNG, JPEG, etc).
- ``numpy.ndarray``: an existing HWC uint8 RGB array.
- ``PIL.Image.Image``: a PIL image (any mode — converted to RGB).
"""


def load_image(source: ImageInput) -> ImageArray:
    """Load an image from any supported source into a HWC uint8 RGB array.

    Args:
        source: Image source. See :data:`ImageInput`.

    Returns:
        A ``numpy.ndarray`` of shape ``(H, W, 3)``, dtype ``uint8``, channel
        order RGB.

    Raises:
        ImageLoadError: If the source cannot be decoded or has an unsupported
            shape or dtype.
    """
    if isinstance(source, np.ndarray):
        return _validate_array(source)

    if isinstance(source, Image.Image):
        return _from_pil(source)

    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise ImageLoadError(f"Image file not found: {path}")
        try:
            with Image.open(path) as img:
                return _from_pil(img)
        except UnidentifiedImageError as exc:
            raise ImageLoadError(f"Cannot identify image file: {path}") from exc
        except Exception as exc:
            raise ImageLoadError(f"Failed to read image file {path}: {exc}") from exc

    if isinstance(source, bytes):
        try:
            with Image.open(io.BytesIO(source)) as img:
                return _from_pil(img)
        except Exception as exc:
            raise ImageLoadError(f"Failed to decode image bytes: {exc}") from exc

    raise ImageLoadError(
        f"Unsupported image source type: {type(source).__name__}. "
        "Expected str, Path, bytes, numpy.ndarray, or PIL.Image.Image."
    )


def _from_pil(image: Image.Image) -> ImageArray:
    """Convert a PIL image to a contiguous HWC uint8 RGB ndarray."""
    rgb = image.convert("RGB")
    array = np.asarray(rgb, dtype=np.uint8)
    return np.ascontiguousarray(array)


def _validate_array(array: np.ndarray) -> ImageArray:
    """Validate that a user-supplied ndarray matches the canonical format."""
    if array.ndim != 3 or array.shape[2] != 3:
        raise ImageLoadError(f"Expected an HWC RGB array with 3 channels, got shape {array.shape}.")
    if array.dtype != np.uint8:
        raise ImageLoadError(
            f"Expected dtype uint8, got {array.dtype}. "
            "Convert normalized float images back to uint8 before passing them in."
        )
    return np.ascontiguousarray(array)
