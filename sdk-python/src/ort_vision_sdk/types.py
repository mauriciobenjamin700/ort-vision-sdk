"""Public output types returned by the SDK's vision tasks.

These types form the contract between the SDK and its callers. They are frozen
dataclasses (immutable, slot-based) so they are cheap to construct in tight
inference loops and safe to pass between threads.

Naming is intentionally compatible with the Ultralytics / torchvision idiom
(``cls``, ``conf``, ``box``, ``xyxy``, ``xywh``, normalized variants) so code
ported from those projects keeps working with minimal edits. The original
verbose attribute names (``class_id``, ``class_name``, ``confidence``,
``bbox``) are preserved as the canonical fields; the short Ultralytics-style
names are exposed as read-only properties.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

ImageArray = NDArray[np.uint8]
"""HWC uint8 RGB image array — the canonical image format used across the SDK."""


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Axis-aligned bounding box in absolute pixel coordinates (xyxy format).

    Coordinates refer to the original input image (before any internal resize),
    so callers can map detections back onto their source image without
    additional bookkeeping.

    Attributes:
        x1: Left edge, in pixels.
        y1: Top edge, in pixels.
        x2: Right edge, in pixels.
        y2: Bottom edge, in pixels.
    """

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        """Box width in pixels (clamped to non-negative)."""
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        """Box height in pixels (clamped to non-negative)."""
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        """Box area in pixels squared."""
        return self.width * self.height

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        """The box as ``(x1, y1, x2, y2)`` in absolute pixels (Ultralytics-style)."""
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def xywh(self) -> tuple[float, float, float, float]:
        """The box as ``(cx, cy, w, h)`` with ``(cx, cy)`` at the center.

        This matches Ultralytics' ``boxes.xywh`` and YOLO's native head format.
        For the top-left ``(x, y, w, h)`` convention, use :meth:`as_xywh`.
        """
        return (
            (self.x1 + self.x2) / 2.0,
            (self.y1 + self.y2) / 2.0,
            self.width,
            self.height,
        )

    def xyxyn(self, orig_shape: tuple[int, int]) -> tuple[float, float, float, float]:
        """The box as ``(x1, y1, x2, y2)`` normalized to ``[0, 1]``.

        Args:
            orig_shape: ``(height, width)`` of the source image, in pixels.

        Returns:
            Normalized ``(x1, y1, x2, y2)``.
        """
        h, w = orig_shape
        if w <= 0 or h <= 0:
            return (0.0, 0.0, 0.0, 0.0)
        return (self.x1 / w, self.y1 / h, self.x2 / w, self.y2 / h)

    def xywhn(self, orig_shape: tuple[int, int]) -> tuple[float, float, float, float]:
        """The box as ``(cx, cy, w, h)`` normalized to ``[0, 1]``.

        Args:
            orig_shape: ``(height, width)`` of the source image, in pixels.

        Returns:
            Normalized ``(cx, cy, w, h)``.
        """
        h, w = orig_shape
        if w <= 0 or h <= 0:
            return (0.0, 0.0, 0.0, 0.0)
        cx, cy, bw, bh = self.xywh
        return (cx / w, cy / h, bw / w, bh / h)

    def as_xyxy(self) -> tuple[float, float, float, float]:
        """Return the box as ``(x1, y1, x2, y2)``."""
        return self.xyxy

    def as_xywh(self) -> tuple[float, float, float, float]:
        """Return the box as ``(x, y, width, height)`` with ``(x, y)`` at the top-left.

        Note: this is the **top-left** convention. Ultralytics' ``xywh``
        property uses **center** coordinates — for that, use :pyattr:`xywh`.
        """
        return (self.x1, self.y1, self.width, self.height)

    def as_int_xyxy(self) -> tuple[int, int, int, int]:
        """Return the box as integer ``(x1, y1, x2, y2)``, useful for slicing arrays."""
        return (int(self.x1), int(self.y1), int(self.x2), int(self.y2))


@dataclass(frozen=True, slots=True)
class ClassProbability:
    """Probability assigned to a single class for a classification prediction.

    Attributes:
        class_id: Integer index of the class as produced by the model.
        class_name: Human-readable label resolved from the SDK's label map.
        probability: Probability in ``[0.0, 1.0]``.
    """

    class_id: int
    class_name: str
    probability: float

    @property
    def cls(self) -> int:
        """Alias for :pyattr:`class_id` (Ultralytics-style)."""
        return self.class_id

    @property
    def name(self) -> str:
        """Alias for :pyattr:`class_name`."""
        return self.class_name

    @property
    def conf(self) -> float:
        """Alias for :pyattr:`probability` (Ultralytics-style)."""
        return self.probability


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    """Output of an image classification inference.

    Attributes:
        class_id: Integer index of the top-1 predicted class.
        class_name: Human-readable label of the top-1 class.
        confidence: Probability of the top-1 class in ``[0.0, 1.0]``.
        image: The original input image as a HWC uint8 RGB array.
        probabilities: Probabilities for every class the model can predict,
            sorted in descending order. The first entry mirrors ``class_id``,
            ``class_name``, and ``confidence``.
    """

    class_id: int
    class_name: str
    confidence: float
    image: ImageArray
    probabilities: tuple[ClassProbability, ...]

    @property
    def cls(self) -> int:
        """Alias for :pyattr:`class_id` (Ultralytics-style)."""
        return self.class_id

    @property
    def name(self) -> str:
        """Alias for :pyattr:`class_name`."""
        return self.class_name

    @property
    def conf(self) -> float:
        """Alias for :pyattr:`confidence` (Ultralytics-style)."""
        return self.confidence


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """Single detected object produced by an object-detection model.

    A detection inference call typically returns a per-image
    :class:`~ort_vision_sdk.results.DetectionResults` whose ``detections``
    field is a tuple of these dataclasses — one per surviving box after NMS.

    Attributes:
        class_id: Integer index of the predicted class.
        class_name: Human-readable label of the predicted class.
        confidence: Detection score in ``[0.0, 1.0]`` (objectness * class probability).
        bbox: Bounding box in original-image pixel coordinates.
        cropped_image: The original image cropped to ``bbox``, as a HWC uint8
            RGB array. Empty boxes (zero area) yield a zero-sized array.
        classification: What a second, classification stage predicted **for
            this crop** — populated only by
            :class:`~ort_vision_sdk.tasks.pipeline.DetectClassify`, and ``None``
            for a plain detector. Kept as its own field rather than folded into
            ``class_id``/``class_name`` because the two answers are different
            questions: the detector says *what kind of object this is*, the
            classifier says *which sub-category the object belongs to*, and
            collapsing them would lose one of the two.
    """

    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox
    cropped_image: ImageArray
    classification: ClassificationResult | None = None

    @property
    def cls(self) -> int:
        """Alias for :pyattr:`class_id` (Ultralytics-style)."""
        return self.class_id

    @property
    def name(self) -> str:
        """Alias for :pyattr:`class_name`."""
        return self.class_name

    @property
    def conf(self) -> float:
        """Alias for :pyattr:`confidence` (Ultralytics-style)."""
        return self.confidence

    @property
    def box(self) -> BoundingBox:
        """Alias for :pyattr:`bbox` (Ultralytics-style)."""
        return self.bbox


@dataclass(frozen=True, slots=True)
class SegmentationResult:
    """Single segmented instance produced by an instance-segmentation model.

    An instance-segmentation inference call returns a per-image
    :class:`~ort_vision_sdk.results.SegmentationResults` whose ``detections``
    field is a tuple of these dataclasses. Each entry mirrors a
    :class:`DetectionResult` but adds the per-instance binary mask and a
    "ready-to-display" background-removed crop.

    All spatial fields share the same ``(bbox.height, bbox.width)`` shape, so
    callers can index them in lockstep.

    Attributes:
        class_id: Integer index of the predicted class.
        class_name: Human-readable label of the predicted class.
        confidence: Instance score in ``[0.0, 1.0]``.
        bbox: Bounding box in original-image pixel coordinates.
        mask: Binary mask cropped to ``bbox``. Shape ``(bbox.height, bbox.width)``,
            dtype ``uint8``, values are ``0`` (background) or ``255`` (foreground).
            Empty boxes yield a zero-sized array.
        segmented_image: The original image cropped to ``bbox`` with
            background pixels (where ``mask == 0``) zeroed out. Shape
            ``(bbox.height, bbox.width, 3)``, dtype ``uint8``, channel order RGB.
            Empty boxes yield a zero-sized array.
    """

    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox
    mask: NDArray[np.uint8]
    segmented_image: ImageArray

    @property
    def cls(self) -> int:
        """Alias for :pyattr:`class_id` (Ultralytics-style)."""
        return self.class_id

    @property
    def name(self) -> str:
        """Alias for :pyattr:`class_name`."""
        return self.class_name

    @property
    def conf(self) -> float:
        """Alias for :pyattr:`confidence` (Ultralytics-style)."""
        return self.confidence

    @property
    def box(self) -> BoundingBox:
        """Alias for :pyattr:`bbox` (Ultralytics-style)."""
        return self.bbox
