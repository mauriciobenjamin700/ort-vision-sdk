"""Instance-segmentation task using YOLO seg ONNX models (v8/v11 and forward)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from ort_vision_sdk.core.backend import read_metadata
from ort_vision_sdk.core.timing import SpeedTimer
from ort_vision_sdk.graph import model_names, resolve_input_size
from ort_vision_sdk.io.image import ImageInput, load_image
from ort_vision_sdk.labels import LabelSpec, default_labels, resolve_labels
from ort_vision_sdk.postprocess.segmentation import decode_yolo_seg
from ort_vision_sdk.preprocess.image import add_batch_dim, letterbox, to_tensor
from ort_vision_sdk.results import Boxes, Masks, SegmentationResults
from ort_vision_sdk.tasks.base import VisionTask
from ort_vision_sdk.types import (
    BoundingBox,
    ImageArray,
    SegmentationResult,
)

if TYPE_CHECKING:
    # Annotation-only; OrtSession imports onnxruntime lazily at runtime.
    import onnxruntime as ort

    from ort_vision_sdk.core.backend import InferenceBackend

SegmenterHead = Literal["yolo-seg"]
"""Decoder family for the segmentation head.

- ``"yolo-seg"``: YOLO instance-segmentation head with two outputs —
  ``(1, 4 + nc + nm, N)`` per-anchor predictions plus ``(1, nm, mh, mw)``
  prototype masks. Covers YOLOv8-seg, v11-seg, v26-seg.

The SDK does **not** auto-detect this — the caller is responsible for
picking a head that matches their export.
"""


class Segmenter(VisionTask):
    """Instance segmenter for YOLO seg ONNX models (v8-seg / v11-seg / ...).

    The model is expected to expose two outputs:

    1. ``output0``: ``(1, 4 + num_classes + num_mask_coefs, num_anchors)`` —
       per-anchor predictions (boxes, class scores, mask coefficients).
    2. ``output1``: ``(1, num_mask_coefs, mask_h, mask_w)`` — prototype masks.

    The names need not be exactly ``output0``/``output1``; the segmenter picks
    the per-anchor tensor as the 3-D output and the prototypes as the 4-D
    output, regardless of name.

    ``predict()`` returns ``list[SegmentationResults]`` (length 1 for a
    single image), mirroring Ultralytics' API. The envelope exposes:

    - ``boxes``: bulk numpy view (``xyxy``, ``xywh``, ``xyxyn``, ``xywhn``,
      ``cls``, ``conf``).
    - ``masks``: per-instance binary masks cropped to each box.
    - per-instance :class:`SegmentationResult` dataclasses via iteration /
      indexing.

    Defaults match Ultralytics YOLOv8/v11 segmentation models (640x640 input,
    COCO 80 classes). Override via constructor arguments for other models.

    Example:
        >>> seg = Segmenter("yolov8n-seg.onnx")
        >>> r = seg.predict("street.jpg")[0]
        >>> r.boxes.xyxy, r.boxes.cls, r.boxes.conf
        >>> for inst in r:
        ...     print(inst.cls, inst.conf, inst.box.xyxy, inst.mask.shape)
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        head: SegmenterHead = "yolo-seg",
        labels: LabelSpec = None,
        providers: list[str] | None = None,
        session_options: ort.SessionOptions | None = None,
        backend: InferenceBackend | None = None,
        input_size: tuple[int, int] | None = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 300,
        mask_threshold: float = 0.5,
    ) -> None:
        """Initialize the segmenter.

        Args:
            model_path: Path to the ``.onnx`` model. Ignored when ``backend``
                is provided.
            head: Decoder family for the model's segmentation head — see
                :data:`SegmenterHead`. Default ``"yolo-seg"`` covers
                YOLOv8-seg/v11-seg/v26-seg.
            labels: Class label spec (see :func:`resolve_labels`). ``None``
                (default) reads the class names the export baked into the model
                metadata (Ultralytics' ``names``), falling back to the 80-class
                COCO preset when the model carries none. Pass a spec to override
                what the model declares.
            providers: Execution providers in preference order. Accepts short
                aliases (``"cuda"``, ``"cpu"``, ...) or canonical ORT names.
                Auto if ``None``. Ignored when ``backend`` is provided.
            session_options: Optional ORT session options. Ignored when
                ``backend`` is provided.
            backend: An explicit
                :class:`~ort_vision_sdk.core.backend.InferenceBackend` to run
                inference through (browser/Android bridge). ``None`` (default)
                uses the in-process ONNX Runtime via :class:`OrtSession`.
            input_size: Model input ``(width, height)`` for letterboxing. Only
                used when the model's graph leaves its spatial axes dynamic: a
                graph that declares a static size always wins, since that is the
                only shape ONNX Runtime will accept. ``None`` (default) means
                "ask the graph, fall back to ``(640, 640)``".
            conf_threshold: Default minimum class score to keep a candidate.
            iou_threshold: Default IoU threshold for non-maximum suppression.
            max_detections: Maximum number of instances to return per image.
            mask_threshold: Probability cutoff applied to soft masks to obtain
                the binary mask. Defaults to ``0.5``.

        Raises:
            ValueError: If ``head`` is not a recognised value.
        """
        if head != "yolo-seg":
            raise ValueError(f"Unsupported segmenter head {head!r}. Supported: 'yolo-seg'.")
        super().__init__(
            model_path,
            providers=providers,
            session_options=session_options,
            backend=backend,
        )
        self._head: SegmenterHead = head
        self._input_size: tuple[int, int] = resolve_input_size(
            graph_shape=self._session.input_shape,
            requested=input_size,
            fallback=(640, 640),
        )
        self._conf_threshold: float = conf_threshold
        self._iou_threshold: float = iou_threshold
        self._max_detections: int = max_detections
        self._mask_threshold: float = mask_threshold

        num_classes = self._infer_num_classes()
        spec: LabelSpec = (
            labels
            if labels is not None
            else model_names(read_metadata(self._session)) or default_labels(num_classes)
        )
        self._labels: tuple[str, ...] = resolve_labels(spec, num_classes=num_classes)
        self._names: dict[int, str] = {i: name for i, name in enumerate(self._labels)}

    @property
    def head(self) -> SegmenterHead:
        """The decoder family used to interpret the model's output."""
        return self._head

    @property
    def input_size(self) -> tuple[int, int]:
        """The ``(width, height)`` this task preprocesses to.

        Resolved at construction time from the model's graph when it declares a
        static input, so reading it back tells you the resolution inference
        really runs at — not merely what was requested.
        """
        return self._input_size

    @property
    def labels(self) -> tuple[str, ...]:
        """Class labels indexed by class id."""
        return self._labels

    @property
    def names(self) -> dict[int, str]:
        """Class id → class name dict (matches Ultralytics' ``model.names``)."""
        return self._names

    @property
    def num_classes(self) -> int:
        """Number of classes the model predicts."""
        return len(self._labels)

    def __call__(
        self,
        image: ImageInput,
        *,
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
        classes: list[int] | None = None,
    ) -> list[SegmentationResults]:
        """Alias for :meth:`predict` — call the segmenter like a torch ``nn.Module``."""
        return self.predict(
            image,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            classes=classes,
        )

    def predict(
        self,
        image: ImageInput,
        *,
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
        classes: list[int] | None = None,
    ) -> list[SegmentationResults]:
        """Run instance segmentation on a single image (synchronous).

        Args:
            image: Image source (path, bytes, ``np.ndarray``, or ``PIL.Image``).
            conf_threshold: Override the default confidence threshold.
            iou_threshold: Override the default IoU threshold.
            classes: If set, keep only instances whose ``class_id`` is in this
                list (mirrors Ultralytics' ``model.predict(img, classes=[0, 16])``).
                ``None`` (default) keeps all classes.

        Returns:
            A 1-element list containing a :class:`SegmentationResults`
            envelope.
        """
        timer = SpeedTimer()
        path = str(image) if isinstance(image, (str, Path)) else None
        original = load_image(image)
        timer.stage("load")
        tensor, scale, pad = self._preprocess(original)
        timer.stage("preprocess")
        outputs = self._session.run({self._session.input_name: tensor})
        timer.stage("inference")
        return self._build_results(
            outputs,
            timer=timer,
            original=original,
            path=path,
            scale=scale,
            pad=pad,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            classes=classes,
        )

    async def async_predict(
        self,
        image: ImageInput,
        *,
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
        classes: list[int] | None = None,
    ) -> list[SegmentationResults]:
        """Async segmentation via ``asyncio.to_thread``.

        Off-loads the entire :meth:`predict` pipeline to the asyncio default
        executor's thread pool. Use in FastAPI/AnyIO handlers. For
        high-concurrency workloads, see :meth:`ort_async_predict`.

        Args and return type match :meth:`predict` exactly.
        """
        return await asyncio.to_thread(
            self.predict,
            image,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            classes=classes,
        )

    async def ort_async_predict(
        self,
        image: ImageInput,
        *,
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
        classes: list[int] | None = None,
    ) -> list[SegmentationResults]:
        """Async segmentation using ORT's native ``run_async`` for the model step.

        Pre-/post-processing run on the event loop thread; the model run is
        dispatched to the ONNX Runtime internal thread pool (configured via
        ``SessionOptions``). Prefer this for high-concurrency workloads where
        many awaits should share the ORT pool. Requires ``onnxruntime>=1.16``.

        Args and return type match :meth:`predict` exactly.
        """
        timer = SpeedTimer()
        path = str(image) if isinstance(image, (str, Path)) else None
        original = load_image(image)
        timer.stage("load")
        tensor, scale, pad = self._preprocess(original)
        timer.stage("preprocess")
        outputs = await self._session.ort_async_run({self._session.input_name: tensor})
        timer.stage("inference")
        return self._build_results(
            outputs,
            timer=timer,
            original=original,
            path=path,
            scale=scale,
            pad=pad,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            classes=classes,
        )

    def _build_results(
        self,
        outputs: list[np.ndarray],
        *,
        timer: SpeedTimer,
        original: ImageArray,
        path: str | None,
        scale: float,
        pad: tuple[int, int],
        conf_threshold: float | None,
        iou_threshold: float | None,
        classes: list[int] | None,
    ) -> list[SegmentationResults]:
        """Decode raw outputs + NMS + masks into a :class:`SegmentationResults`.

        Shared between :meth:`predict`, :meth:`async_predict` and
        :meth:`ort_async_predict`.
        """
        per_anchor, prototypes = self._split_outputs(outputs)

        decoded = decode_yolo_seg(
            per_anchor,
            prototypes,
            num_classes=len(self._labels),
            input_size=self._input_size,
            original_size=(original.shape[1], original.shape[0]),
            pad=pad,
            scale=scale,
            conf_threshold=(conf_threshold if conf_threshold is not None else self._conf_threshold),
            iou_threshold=(iou_threshold if iou_threshold is not None else self._iou_threshold),
            max_detections=self._max_detections,
            mask_threshold=self._mask_threshold,
        )

        if classes is not None:
            allowed = set(classes)
            decoded = [d for d in decoded if d[1] in allowed]

        detections = tuple(
            self._build_instance(original, bbox, class_id, confidence, mask)
            for bbox, class_id, confidence, mask in decoded
        )

        orig_shape = (int(original.shape[0]), int(original.shape[1]))
        boxes = self._build_boxes(detections, orig_shape=orig_shape)
        masks = self._build_masks(detections, orig_shape=orig_shape)
        timer.stage("postprocess")

        return [
            SegmentationResults(
                boxes=boxes,
                masks=masks,
                detections=detections,
                names=self._names,
                orig_img=original,
                orig_shape=orig_shape,
                path=path,
                speed=timer.speed(),
            )
        ]

    def _preprocess(self, image: ImageArray) -> tuple[np.ndarray, float, tuple[int, int]]:
        """Letterbox → CHW float32/255 → batch."""
        boxed, scale, pad = letterbox(image, self._input_size)
        tensor = to_tensor(boxed)
        return np.ascontiguousarray(add_batch_dim(tensor)), scale, pad

    def _split_outputs(self, outputs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Identify which output is the per-anchor tensor and which is the prototypes.

        Per-anchor output is 3-D ``(1, channels, anchors)``; prototypes are
        4-D ``(1, num_mask_coefs, mask_h, mask_w)``.
        """
        per_anchor: np.ndarray | None = None
        prototypes: np.ndarray | None = None
        for out in outputs:
            if out.ndim == 3 and per_anchor is None:
                per_anchor = out
            elif out.ndim == 4 and prototypes is None:
                prototypes = out
        if per_anchor is None or prototypes is None:
            shapes = [tuple(o.shape) for o in outputs]
            raise ValueError(f"Segmenter expected one 3-D and one 4-D output, got shapes={shapes}.")
        return per_anchor, prototypes

    def _build_instance(
        self,
        original: ImageArray,
        bbox: BoundingBox,
        class_id: int,
        confidence: float,
        mask: NDArray[np.uint8],
    ) -> SegmentationResult:
        """Assemble a :class:`SegmentationResult` and the masked crop."""
        h, w = original.shape[:2]
        x1, y1, x2, y2 = bbox.as_int_xyxy()
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 > x1 and y2 > y1 and mask.size > 0:
            crop = np.ascontiguousarray(original[y1:y2, x1:x2])
            # Defensive: clamp mask shape to crop shape if rounding produced a 1px diff.
            mh = min(mask.shape[0], crop.shape[0])
            mw = min(mask.shape[1], crop.shape[1])
            mask = mask[:mh, :mw]
            crop = crop[:mh, :mw]
            segmented = crop.copy()
            segmented[mask == 0] = 0
        else:
            mask = np.zeros((0, 0), dtype=np.uint8)
            segmented = np.zeros((0, 0, 3), dtype=np.uint8)

        class_name = self._names.get(class_id, f"class_{class_id}")

        return SegmentationResult(
            class_id=class_id,
            class_name=class_name,
            confidence=confidence,
            bbox=bbox,
            mask=np.ascontiguousarray(mask),
            segmented_image=np.ascontiguousarray(segmented),
        )

    def _build_boxes(
        self,
        detections: tuple[SegmentationResult, ...],
        *,
        orig_shape: tuple[int, int],
    ) -> Boxes:
        """Assemble the bulk-array ``Boxes`` view from per-instance dataclasses."""
        if not detections:
            return Boxes(
                xyxy=np.empty((0, 4), dtype=np.float64),
                cls=np.empty((0,), dtype=np.int64),
                conf=np.empty((0,), dtype=np.float64),
                orig_shape=orig_shape,
            )
        xyxy = np.asarray(
            [d.bbox.xyxy for d in detections],
            dtype=np.float64,
        )
        cls = np.asarray([d.class_id for d in detections], dtype=np.int64)
        conf = np.asarray([d.confidence for d in detections], dtype=np.float64)
        return Boxes(xyxy=xyxy, cls=cls, conf=conf, orig_shape=orig_shape)

    def _build_masks(
        self,
        detections: tuple[SegmentationResult, ...],
        *,
        orig_shape: tuple[int, int],
    ) -> Masks:
        """Assemble the bulk ``Masks`` view from per-instance dataclasses."""
        if not detections:
            return Masks(
                data=(),
                xyxy=np.empty((0, 4), dtype=np.float64),
                orig_shape=orig_shape,
            )
        xyxy = np.asarray(
            [d.bbox.xyxy for d in detections],
            dtype=np.float64,
        )
        return Masks(
            data=tuple(d.mask for d in detections),
            xyxy=xyxy,
            orig_shape=orig_shape,
        )

    def _infer_num_classes(self) -> int | None:
        """Best-effort inference of ``num_classes`` from the per-anchor output shape.

        Per-anchor output has channels = ``4 + num_classes + num_mask_coefs``.
        Without prototypes available at construction time we cannot know
        ``num_mask_coefs`` precisely, so we assume the standard YOLO seg value
        of 32. If the user passes the wrong labels the constructor's
        validation will catch it; otherwise ``predict`` will raise.
        """
        output_shapes = self._session.output_shapes
        if not output_shapes:
            return None
        for shape in output_shapes:
            if len(shape) != 3:
                continue
            int_dims = [d for d in shape if isinstance(d, int) and d > 1]
            if not int_dims:
                return None
            channels = min(int_dims)
            inferred = channels - 4 - 32  # standard YOLO seg num_mask_coefs
            return int(inferred) if inferred > 0 else None
        return None
