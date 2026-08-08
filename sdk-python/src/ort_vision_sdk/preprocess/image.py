"""Composable image preprocessing primitives (resize, normalize, letterbox, layout).

These helpers operate on the canonical :data:`~ort_vision_sdk.types.ImageArray`
(HWC uint8 RGB) and produce either uint8 or float32 arrays depending on the
operation. Tasks chain them to build their own pipelines.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from ort_vision_sdk.types import ImageArray

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
"""Per-channel RGB mean of ImageNet, the normalization torchvision classifiers assume."""

IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
"""Per-channel RGB standard deviation of ImageNet."""


def reduction_factor(source_size: tuple[int, int], target_size: tuple[int, int]) -> int:
    """Largest integer factor an image can be box-reduced by before resampling.

    Chosen as ``min(src // target)`` over both axes, which keeps the reduced
    image at least as large as the target on both: :meth:`PIL.Image.Image.reduce`
    divides with a ceiling, so ``ceil(src / factor) >= src / factor >= target``.
    The final resample therefore stays a downscale (or a no-op) and never
    invents detail.

    Args:
        source_size: Source ``(width, height)`` in pixels.
        target_size: Target ``(width, height)`` in pixels.

    Returns:
        The factor to pass to ``reduce``. ``1`` means no reduction applies —
        either the target is larger than the source, or it is less than half
        the source on one axis.
    """
    source_width, source_height = source_size
    target_width, target_height = target_size
    if target_width < 1 or target_height < 1:
        return 1
    return max(1, min(source_width // target_width, source_height // target_height))


def resize(
    image: ImageArray,
    size: tuple[int, int],
    *,
    resample: Image.Resampling = Image.Resampling.BILINEAR,
) -> ImageArray:
    """Resize an HWC uint8 image to ``(width, height) = size``.

    A downscale of 2x or more runs in two steps: an integer box reduction via
    :meth:`PIL.Image.Image.reduce`, then the requested resampling onto the exact
    target. This is the same optimization PIL exposes as ``reducing_gap``, done
    explicitly so the reduction actually engages at the ratios this SDK sees.

    The gain is speed, and it is large: letterboxing 1920x1080 to 640x640 goes
    from 7.8 ms to 3.5 ms, and 3840x2160 from 29.8 ms to 10.5 ms.

    The gain is **not** fidelity, and it is worth being precise about that. Box
    reduction is the better antialiasing filter on most content — measured
    against a LANCZOS reference it beats the single pass on photographic detail
    and on most periodic patterns. But when the content's period resonates with
    the reduction factor it is markedly worse: 2-pixel stripes every 6 rows,
    reduced by 3, score an MSE of 106 against the single pass's 13. White noise
    likewise favours the single pass. Across six content types the two paths
    split three-three, so this changes which artifacts a downscale produces
    rather than removing them.

    ``NEAREST`` is exempt: a caller asking for it wants unblended pixels, and a
    box reduction would blend them.

    Args:
        image: Source image (HWC uint8 RGB).
        size: Target ``(width, height)`` in pixels.
        resample: PIL resampling filter.

    Returns:
        The resized image as an HWC uint8 RGB ndarray.
    """
    pil = Image.fromarray(image)
    if resample != Image.Resampling.NEAREST:
        factor = reduction_factor(pil.size, size)
        if factor > 1:
            pil = pil.reduce(factor)
    resized = pil if pil.size == size else pil.resize(size, resample=resample)
    return np.asarray(resized, dtype=np.uint8)


def normalize(
    image: ImageArray,
    *,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    scale: float = 1.0 / 255.0,
) -> np.ndarray:
    """Convert a uint8 image to a normalized float32 array.

    Applies ``(image * scale - mean) / std`` channel-wise.

    Args:
        image: Source image (HWC uint8 RGB).
        mean: Per-channel mean (RGB).
        std: Per-channel standard deviation (RGB).
        scale: Multiplier applied before mean subtraction (default ``1/255``).

    Returns:
        A ``float32`` ndarray with the same HWC layout as ``image``.
    """
    arr = image.astype(np.float32) * scale
    arr -= np.array(mean, dtype=np.float32)
    arr /= np.array(std, dtype=np.float32)
    return arr


def to_chw(image: np.ndarray) -> np.ndarray:
    """Transpose an HWC array to CHW layout.

    Args:
        image: An ``(H, W, C)`` array.

    Returns:
        A contiguous ``(C, H, W)`` array.
    """
    return np.ascontiguousarray(np.transpose(image, (2, 0, 1)))


def to_tensor(image: ImageArray) -> np.ndarray:
    """Convert an HWC uint8 image to a CHW ``float32`` array scaled to ``[0, 1]``.

    Mirrors the semantics of ``torchvision.transforms.ToTensor()``:
    HWC → CHW, ``uint8 → float32 / 255``. The result is the standard input
    format for a PyTorch-style classification or detection model **before**
    any ``Normalize`` step, and the format ORT expects from CHW float models
    that don't require ImageNet normalization (e.g. YOLO).

    Args:
        image: HWC uint8 RGB array.

    Returns:
        Contiguous ``(C, H, W)`` ``float32`` array with values in ``[0, 1]``.
    """
    chw = to_chw(image)
    return np.ascontiguousarray(chw.astype(np.float32) / 255.0)


def from_cv2(bgr_image: np.ndarray) -> ImageArray:
    """Convert an OpenCV ``HWC BGR uint8`` image to the SDK's canonical RGB format.

    Use this when you read images with ``cv2.imread`` (which returns BGR) and
    want to feed them to the SDK without going through PIL.

    Args:
        bgr_image: HWC uint8 BGR array.

    Returns:
        HWC uint8 RGB array (the SDK's canonical :data:`ImageArray`).

    Raises:
        ValueError: If ``bgr_image`` is not a 3-D array with exactly 3 channels.
    """
    if bgr_image.ndim != 3 or bgr_image.shape[2] != 3:
        raise ValueError(f"from_cv2: expected an HWC 3-channel array, got shape {bgr_image.shape}.")
    return np.ascontiguousarray(bgr_image[..., ::-1])


def to_cv2(image: ImageArray) -> np.ndarray:
    """Convert the SDK's HWC uint8 RGB image to OpenCV's HWC BGR layout.

    Use this when you want to display or save the SDK's output with
    ``cv2.imshow`` / ``cv2.imwrite`` (both expect BGR).

    Args:
        image: HWC uint8 RGB array.

    Returns:
        HWC uint8 BGR array.

    Raises:
        ValueError: If ``image`` is not a 3-D array with exactly 3 channels.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"to_cv2: expected an HWC 3-channel array, got shape {image.shape}.")
    return np.ascontiguousarray(image[..., ::-1])


def add_batch_dim(array: np.ndarray) -> np.ndarray:
    """Prepend a batch dimension of size 1.

    Args:
        array: Source array.

    Returns:
        ``array`` with a leading axis of size 1.
    """
    return np.expand_dims(array, axis=0)


def letterbox(
    image: ImageArray,
    size: tuple[int, int],
    *,
    fill: tuple[int, int, int] = (114, 114, 114),
) -> tuple[ImageArray, float, tuple[int, int]]:
    """Resize preserving aspect ratio, padding to ``size`` with a constant color.

    This is the standard preprocessing for YOLO-family detectors: the original
    image is rescaled by a single factor that keeps proportions intact, and
    the remaining canvas is filled with a neutral colour. Returning the
    ``scale`` and ``pad`` values lets callers map detections back to the
    original image coordinates.

    Args:
        image: Source image (HWC uint8 RGB).
        size: Target ``(width, height)`` in pixels.
        fill: RGB fill colour for padding (default YOLO grey).

    Returns:
        A tuple ``(letterboxed, scale, (pad_left, pad_top))``:

        - ``letterboxed``: image of shape ``(size[1], size[0], 3)``, uint8 RGB.
        - ``scale``: factor applied to the original image (``< 1`` if downscaled).
        - ``(pad_left, pad_top)``: integer padding offsets in pixels.
    """
    target_w, target_h = size
    src_h, src_w = image.shape[:2]
    scale = min(target_w / src_w, target_h / src_h)
    new_w = round(src_w * scale)
    new_h = round(src_h * scale)

    resized = resize(image, (new_w, new_h))

    canvas = np.full((target_h, target_w, 3), fill, dtype=np.uint8)
    pad_left = (target_w - new_w) // 2
    pad_top = (target_h - new_h) // 2
    canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized

    return canvas, scale, (pad_left, pad_top)
