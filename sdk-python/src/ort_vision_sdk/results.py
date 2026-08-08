"""Per-image result envelopes — Ultralytics-style ``Results`` for the SDK.

Each ``predict()`` call returns ``list[Results]`` (length 1 for a single
image), mirroring how Ultralytics' ``YOLO("img.jpg")`` returns a list. The
envelope holds:

- A bulk-array view of the predictions (``Boxes``, ``Probs``, ``Masks``)
  with the exact attribute names Ultralytics uses (``xyxy``, ``xywh``,
  ``xyxyn``, ``xywhn``, ``cls``, ``conf``, ``data``, ``top1``, ``top5``).
- The per-instance dataclasses (``DetectionResult``, ``SegmentationResult``,
  ``ClassProbability``) for callers who prefer the OO interface.
- ``names``: ``dict[int, str]`` of class id → label, matching Ultralytics'
  ``model.names``.
- ``orig_img`` / ``orig_shape`` / ``path``: provenance for the original input.

The envelope is iterable and indexable for backwards compatibility — iterating
yields the per-instance dataclasses, so ``for d in results[0]: ...`` reads
just like the older "list of detections" API.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import overload

import numpy as np
from numpy.typing import NDArray

from ort_vision_sdk.types import (
    ClassificationResult,
    ClassProbability,
    DetectionResult,
    ImageArray,
    SegmentationResult,
)


@dataclass(frozen=True, slots=True)
class Boxes:
    """Bulk numpy view of detected boxes for a single image.

    Mirrors Ultralytics' ``Boxes`` interface so code written against
    ``result.boxes.xyxy`` / ``.cls`` / ``.conf`` ports without changes.

    Attributes:
        xyxy: ``(N, 4)`` float array of boxes in original-image absolute pixel
            coordinates ``(x1, y1, x2, y2)``.
        cls: ``(N,)`` int64 array of predicted class indices.
        conf: ``(N,)`` float array of detection confidences in ``[0, 1]``.
        orig_shape: ``(height, width)`` of the original image, used to compute
            normalized variants.
    """

    xyxy: NDArray[np.float64]
    cls: NDArray[np.int64]
    conf: NDArray[np.float64]
    orig_shape: tuple[int, int]

    @property
    def xywh(self) -> NDArray[np.float64]:
        """Boxes as ``(N, 4)`` ``(cx, cy, w, h)`` arrays in absolute pixels."""
        if self.xyxy.size == 0:
            return np.empty((0, 4), dtype=np.float64)
        x1, y1, x2, y2 = self.xyxy.T
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
        return np.stack([cx, cy, w, h], axis=1)

    @property
    def xyxyn(self) -> NDArray[np.float64]:
        """Boxes as ``(N, 4)`` ``(x1, y1, x2, y2)`` normalized to ``[0, 1]``."""
        h, w = self.orig_shape
        if self.xyxy.size == 0 or w <= 0 or h <= 0:
            return np.empty((0, 4), dtype=np.float64)
        out = self.xyxy.astype(np.float64, copy=True)
        out[:, [0, 2]] /= float(w)
        out[:, [1, 3]] /= float(h)
        return out

    @property
    def xywhn(self) -> NDArray[np.float64]:
        """Boxes as ``(N, 4)`` ``(cx, cy, w, h)`` normalized to ``[0, 1]``."""
        h, w = self.orig_shape
        xywh = self.xywh
        if xywh.size == 0 or w <= 0 or h <= 0:
            return np.empty((0, 4), dtype=np.float64)
        out: NDArray[np.float64] = xywh.copy()
        out[:, [0, 2]] /= float(w)
        out[:, [1, 3]] /= float(h)
        return out

    @property
    def data(self) -> NDArray[np.float64]:
        """Concatenated ``(N, 6)`` array of ``[x1, y1, x2, y2, conf, cls]``.

        Matches Ultralytics' ``boxes.data``.
        """
        if self.xyxy.size == 0:
            return np.empty((0, 6), dtype=np.float64)
        return np.concatenate(
            [
                self.xyxy.astype(np.float64, copy=False),
                self.conf.reshape(-1, 1).astype(np.float64, copy=False),
                self.cls.reshape(-1, 1).astype(np.float64, copy=False),
            ],
            axis=1,
        )

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the underlying ``xyxy`` array, ``(N, 4)``."""
        return tuple(self.xyxy.shape)

    def __len__(self) -> int:
        """Number of detected boxes."""
        return int(self.xyxy.shape[0])


@dataclass(frozen=True, slots=True)
class Probs:
    """Top-k classification probabilities for a single image.

    Mirrors Ultralytics' ``Probs`` interface. ``data`` is the full per-class
    probability vector; ``top1``/``top5`` are derived on access.

    Attributes:
        data: ``(num_classes,)`` float array of per-class probabilities,
            indexed by class id (not sorted).
    """

    data: NDArray[np.float64]

    @property
    def top1(self) -> int:
        """Index of the most probable class."""
        if self.data.size == 0:
            return 0
        return int(np.argmax(self.data))

    @property
    def top1conf(self) -> float:
        """Probability of the top-1 class."""
        if self.data.size == 0:
            return 0.0
        return float(self.data[self.top1])

    @property
    def top5(self) -> NDArray[np.int64]:
        """Indices of the top-5 most probable classes, descending."""
        if self.data.size == 0:
            return np.empty((0,), dtype=np.int64)
        n = min(5, int(self.data.shape[0]))
        return np.argsort(-self.data, kind="stable")[:n].astype(np.int64, copy=False)

    @property
    def top5conf(self) -> NDArray[np.float64]:
        """Probabilities of the top-5 classes, descending."""
        if self.data.size == 0:
            return np.empty((0,), dtype=np.float64)
        return self.data[self.top5].astype(np.float64, copy=False)

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the underlying probability vector, ``(num_classes,)``."""
        return tuple(self.data.shape)

    def __len__(self) -> int:
        """Number of classes in the probability vector."""
        return int(self.data.shape[0])


@dataclass(frozen=True, slots=True)
class Masks:
    """Per-instance binary masks for a single image.

    Each mask is cropped to its instance's bounding box (HWC layout). To
    obtain a full-image-sized mask, paint each crop onto a zero canvas at
    the corresponding ``xyxy`` location.

    Attributes:
        data: Tuple of binary uint8 masks, one per instance. Each mask has
            shape ``(bbox_h, bbox_w)`` and values in ``{0, 255}``.
        xyxy: ``(N, 4)`` float array of the bounding box of each mask in
            original-image pixel coordinates — provided for convenience so
            callers can paint masks onto a full image.
        orig_shape: ``(height, width)`` of the original image.
    """

    data: tuple[NDArray[np.uint8], ...]
    xyxy: NDArray[np.float64]
    orig_shape: tuple[int, int]

    @property
    def shape(self) -> tuple[int, ...]:
        """Number of masks as a 1-element tuple ``(N,)``."""
        return (len(self.data),)

    def __len__(self) -> int:
        """Number of instance masks."""
        return len(self.data)

    def __iter__(self) -> Iterator[NDArray[np.uint8]]:
        """Iterate over per-instance binary masks."""
        return iter(self.data)


@dataclass(frozen=True, slots=True)
class DetectionResults:
    """Per-image detection envelope (Ultralytics-style ``Results``).

    Iterating or indexing yields :class:`~ort_vision_sdk.types.DetectionResult`
    instances, so legacy code that did ``for d in detector.predict(img)`` only
    needs an extra ``[0]`` to bridge:

    ```python
    for d in detector.predict(img)[0]:
        print(d.cls, d.conf, d.box.xyxy)
    ```

    For numpy-array access, use the ``boxes`` collection:

    ```python
    res = detector.predict(img)[0]
    print(res.boxes.xyxy.shape, res.boxes.cls, res.boxes.conf)
    ```

    Attributes:
        boxes: Bulk-array view of all detections (``Boxes``).
        detections: Tuple of per-instance :class:`DetectionResult` dataclasses.
        names: Class id → class name dict, matching Ultralytics' ``model.names``.
        orig_img: The original input image as HWC uint8 RGB.
        orig_shape: ``(height, width)`` of ``orig_img``.
        path: Source path of the input image, or ``None`` if it wasn't a path.
        speed: Per-stage durations in milliseconds — ``load``,
            ``preprocess``, ``inference`` and ``postprocess`` — as measured by
            the ``predict()`` call that produced this envelope. Empty for
            envelopes built by hand.
    """

    boxes: Boxes
    detections: tuple[DetectionResult, ...]
    names: dict[int, str]
    orig_img: ImageArray
    orig_shape: tuple[int, int]
    path: str | None = None
    speed: dict[str, float] = field(default_factory=dict)

    def __len__(self) -> int:
        """Number of surviving detections."""
        return len(self.detections)

    def __iter__(self) -> Iterator[DetectionResult]:
        """Iterate over per-instance detections."""
        return iter(self.detections)

    @overload
    def __getitem__(self, index: int) -> DetectionResult: ...
    @overload
    def __getitem__(self, index: slice) -> tuple[DetectionResult, ...]: ...
    def __getitem__(self, index: int | slice) -> DetectionResult | tuple[DetectionResult, ...]:
        """Index into the per-instance detections."""
        return self.detections[index]


@dataclass(frozen=True, slots=True)
class DetectClassifyResults:
    """Per-image envelope for a fused detect→classify pipeline.

    Structurally a :class:`DetectionResults` with a second class map: every
    detection it yields carries a populated
    :pyattr:`~ort_vision_sdk.types.DetectionResult.classification`, and the two
    stages have their own, unrelated label spaces — a detector that finds
    ``sheep`` feeding a classifier that answers ``famacha_3`` shares no class
    ids with it. Merging them into one ``names`` dict would make ``cls`` and
    ``classification.cls`` look comparable when they are not.

    ```python
    result = pipeline.predict("flock.jpg")[0]
    for detection in result:
        print(detection.name, detection.conf, detection.classification.name)
    ```

    Attributes:
        boxes: Bulk-array view of the detections (``Boxes``), covering the
            detection stage only.
        detections: Tuple of per-instance
            :class:`~ort_vision_sdk.types.DetectionResult` dataclasses, each
            with its ``classification`` filled in.
        names: Detector class id → class name.
        classifier_names: Classifier class id → class name.
        orig_img: The original input image as HWC uint8 RGB.
        orig_shape: ``(height, width)`` of ``orig_img``.
        path: Source path of the input image, or ``None`` if it wasn't a path.
        speed: Per-stage durations in milliseconds — ``load``, ``preprocess``,
            ``inference`` and ``postprocess``. The ``inference`` figure covers
            detection *and* classification, since the pipeline runs them as one
            graph and no boundary between them is observable from outside.
    """

    boxes: Boxes
    detections: tuple[DetectionResult, ...]
    names: dict[int, str]
    classifier_names: dict[int, str]
    orig_img: ImageArray
    orig_shape: tuple[int, int]
    path: str | None = None
    speed: dict[str, float] = field(default_factory=dict)

    def __len__(self) -> int:
        """Number of surviving detections."""
        return len(self.detections)

    def __iter__(self) -> Iterator[DetectionResult]:
        """Iterate over per-instance detections, each carrying its classification."""
        return iter(self.detections)

    @overload
    def __getitem__(self, index: int) -> DetectionResult: ...
    @overload
    def __getitem__(self, index: slice) -> tuple[DetectionResult, ...]: ...
    def __getitem__(self, index: int | slice) -> DetectionResult | tuple[DetectionResult, ...]:
        """Index into the per-instance detections."""
        return self.detections[index]


@dataclass(frozen=True, slots=True)
class ClassificationResults:
    """Per-image classification envelope (Ultralytics-style ``Results``).

    Attributes:
        probs: Bulk view of per-class probabilities (``Probs``).
        result: The legacy :class:`~ort_vision_sdk.types.ClassificationResult`
            for callers who want the per-class probability tuple with names
            already resolved.
        names: Class id → class name dict, matching Ultralytics' ``model.names``.
        orig_img: The original input image as HWC uint8 RGB.
        orig_shape: ``(height, width)`` of ``orig_img``.
        path: Source path of the input image, or ``None`` if it wasn't a path.
        speed: Per-stage durations in milliseconds — ``load``,
            ``preprocess``, ``inference`` and ``postprocess`` — as measured by
            the ``predict()`` call that produced this envelope. Empty for
            envelopes built by hand.
    """

    probs: Probs
    result: ClassificationResult
    names: dict[int, str]
    orig_img: ImageArray
    orig_shape: tuple[int, int]
    path: str | None = None
    speed: dict[str, float] = field(default_factory=dict)

    @property
    def cls(self) -> int:
        """Top-1 class index (Ultralytics-style alias)."""
        return self.probs.top1

    @property
    def conf(self) -> float:
        """Top-1 confidence (Ultralytics-style alias)."""
        return self.probs.top1conf

    @property
    def name(self) -> str:
        """Top-1 class name."""
        return self.names.get(self.cls, f"class_{self.cls}")

    @property
    def probabilities(self) -> tuple[ClassProbability, ...]:
        """Per-class probability tuple, sorted descending (legacy field)."""
        return self.result.probabilities


@dataclass(frozen=True, slots=True)
class SegmentationResults:
    """Per-image instance-segmentation envelope (Ultralytics-style ``Results``).

    Iterating or indexing yields :class:`~ort_vision_sdk.types.SegmentationResult`
    instances. ``boxes`` and ``masks`` mirror Ultralytics' bulk-array views.

    Attributes:
        boxes: Bulk-array view of all instance boxes (``Boxes``).
        masks: Bulk view of per-instance binary masks (``Masks``).
        detections: Tuple of per-instance :class:`SegmentationResult` dataclasses.
        names: Class id → class name dict.
        orig_img: The original input image as HWC uint8 RGB.
        orig_shape: ``(height, width)`` of ``orig_img``.
        path: Source path of the input image, or ``None`` if it wasn't a path.
        speed: Per-stage durations in milliseconds — ``load``,
            ``preprocess``, ``inference`` and ``postprocess`` — as measured by
            the ``predict()`` call that produced this envelope. Empty for
            envelopes built by hand.
    """

    boxes: Boxes
    masks: Masks
    detections: tuple[SegmentationResult, ...]
    names: dict[int, str]
    orig_img: ImageArray
    orig_shape: tuple[int, int]
    path: str | None = None
    speed: dict[str, float] = field(default_factory=dict)

    def __len__(self) -> int:
        """Number of surviving instances."""
        return len(self.detections)

    def __iter__(self) -> Iterator[SegmentationResult]:
        """Iterate over per-instance results."""
        return iter(self.detections)

    @overload
    def __getitem__(self, index: int) -> SegmentationResult: ...
    @overload
    def __getitem__(self, index: slice) -> tuple[SegmentationResult, ...]: ...
    def __getitem__(
        self, index: int | slice
    ) -> SegmentationResult | tuple[SegmentationResult, ...]:
        """Index into the per-instance results."""
        return self.detections[index]


__all__: list[str] = [
    "Boxes",
    "ClassificationResults",
    "DetectionResults",
    "Masks",
    "Probs",
    "SegmentationResults",
]
