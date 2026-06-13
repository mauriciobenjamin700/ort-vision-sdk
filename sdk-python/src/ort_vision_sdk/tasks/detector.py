"""Object detection task using anchor-free YOLO ONNX models (v8/v9/v10/v11/v12/v26)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from ort_vision_sdk.io.image import ImageInput, load_image
from ort_vision_sdk.labels import LabelSpec, resolve_labels
from ort_vision_sdk.postprocess.detection import decode_yolo
from ort_vision_sdk.preprocess.image import add_batch_dim, letterbox, to_tensor
from ort_vision_sdk.results import Boxes, DetectionResults
from ort_vision_sdk.tasks.base import VisionTask
from ort_vision_sdk.types import BoundingBox, DetectionResult, ImageArray

if TYPE_CHECKING:
    # Annotation-only; OrtSession imports onnxruntime lazily at runtime.
    import onnxruntime as ort

DetectorHead = Literal["yolo"]
"""Decoder family for the detection head.

- ``"yolo"``: anchor-free YOLO head with output shape ``(1, 4 + nc, N)`` —
  covers YOLOv8, v9, v10, v11, v12, v26 detect exports.

This list is intentionally explicit. The SDK does **not** auto-detect the
head layout from the model — the caller is responsible for picking a head
that matches their export. Future families (v5/v6/v7 ``(1, N, 5+nc)``) will
be added as new literal members when their decoders land.
"""


class Detector(VisionTask):
    """Object detector for anchor-free YOLO ONNX models (v8/v9/v10/v11/v12).

    The detector applies letterbox preprocessing, runs inference, decodes the
    raw output into per-class candidates, runs non-maximum suppression and
    returns detections with bounding boxes mapped back to the original image
    coordinates plus the corresponding cropped regions.

    ``predict()`` returns ``list[DetectionResults]`` (length 1 for a single
    image input), mirroring Ultralytics' ``YOLO("img.jpg")`` API. Iterate the
    envelope for per-instance dataclasses, or use the bulk ``boxes`` view
    (``.xyxy``, ``.xywh``, ``.xyxyn``, ``.xywhn``, ``.cls``, ``.conf``).

    Default labels are the 80-class COCO preset; pass ``labels=`` to override
    (a list, dict, file path, or another preset name).

    Example:
        >>> det = Detector("yolov8n.onnx")
        >>> results = det.predict("street.jpg")
        >>> r = results[0]
        >>> r.boxes.xyxy.shape, r.boxes.cls, r.boxes.conf
        >>> for d in r:
        ...     print(d.cls, d.conf, d.box.xyxy)
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        head: DetectorHead = "yolo",
        labels: LabelSpec = "coco",
        providers: list[str] | None = None,
        session_options: ort.SessionOptions | None = None,
        input_size: tuple[int, int] = (640, 640),
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 300,
    ) -> None:
        """Initialize the detector.

        Args:
            model_path: Path to the ``.onnx`` model.
            head: Decoder family for the model's detection head — see
                :data:`DetectorHead`. Default ``"yolo"`` covers YOLOv8/v9/v10/v11/v12/v26.
            labels: Class label spec (see :func:`resolve_labels`). Defaults to
                the 80-class COCO preset.
            providers: Execution providers in preference order. Accepts short
                aliases (``"cuda"``, ``"cpu"``, ``"tensorrt"``, ...) as well
                as canonical ORT names. Auto if ``None``.
            session_options: Optional ORT session options.
            input_size: Model input ``(width, height)`` for letterboxing.
            conf_threshold: Default minimum class score to keep a candidate.
                Can be overridden per :meth:`predict` call.
            iou_threshold: Default IoU threshold for non-maximum suppression.
                Can be overridden per :meth:`predict` call.
            max_detections: Maximum number of detections to return per image.

        Raises:
            ValueError: If ``head`` is not a recognised value.
        """
        if head != "yolo":
            raise ValueError(f"Unsupported detector head {head!r}. Supported: 'yolo'.")
        super().__init__(
            model_path,
            providers=providers,
            session_options=session_options,
        )
        self._head: DetectorHead = head
        self._input_size: tuple[int, int] = input_size
        self._conf_threshold: float = conf_threshold
        self._iou_threshold: float = iou_threshold
        self._max_detections: int = max_detections

        num_classes = self._infer_num_classes()
        self._labels: tuple[str, ...] = resolve_labels(labels, num_classes=num_classes)
        self._names: dict[int, str] = {i: name for i, name in enumerate(self._labels)}

    @property
    def head(self) -> DetectorHead:
        """The decoder family used to interpret the model's output."""
        return self._head

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
    ) -> list[DetectionResults]:
        """Alias for :meth:`predict` — call the detector like a torch ``nn.Module``."""
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
    ) -> list[DetectionResults]:
        """Run detection on a single image (synchronous).

        Args:
            image: Image source (path, bytes, ``np.ndarray``, or ``PIL.Image``).
            conf_threshold: Override the default confidence threshold.
            iou_threshold: Override the default IoU threshold.
            classes: If set, keep only detections whose ``class_id`` is in this
                list (mirrors Ultralytics' ``model.predict(img, classes=[0, 16])``).
                ``None`` (default) keeps all classes.

        Returns:
            A 1-element list containing a :class:`DetectionResults` envelope.
            Iterate the envelope to access per-instance
            :class:`DetectionResult` dataclasses, or use the bulk-array
            ``boxes`` view.
        """
        path = str(image) if isinstance(image, (str, Path)) else None
        original = load_image(image)
        tensor, scale, pad = self._preprocess(original)
        outputs = self._session.run({self._session.input_name: tensor})
        return self._build_results(
            outputs,
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
    ) -> list[DetectionResults]:
        """Async detection via ``asyncio.to_thread``.

        Off-loads :meth:`predict` (preprocess + ORT run + decode + NMS) to the
        asyncio default executor's thread pool, freeing the event loop. Use in
        FastAPI/AnyIO handlers and similar async contexts. For
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
    ) -> list[DetectionResults]:
        """Async detection using ORT's native ``run_async`` for the model step.

        Letterboxing and decode/NMS run on the event loop thread (NumPy ops);
        the model run is dispatched to the ONNX Runtime internal thread pool
        (configured by your ``SessionOptions``). Prefer this for
        high-throughput concurrency where many awaits should share the ORT
        pool. Requires ``onnxruntime>=1.16``.

        Args and return type match :meth:`predict` exactly.
        """
        path = str(image) if isinstance(image, (str, Path)) else None
        original = load_image(image)
        tensor, scale, pad = self._preprocess(original)
        outputs = await self._session.ort_async_run({self._session.input_name: tensor})
        return self._build_results(
            outputs,
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
        original: ImageArray,
        path: str | None,
        scale: float,
        pad: tuple[int, int],
        conf_threshold: float | None,
        iou_threshold: float | None,
        classes: list[int] | None,
    ) -> list[DetectionResults]:
        """Decode raw outputs + NMS into a :class:`DetectionResults` envelope.

        Shared between :meth:`predict`, :meth:`async_predict` and
        :meth:`ort_async_predict`.
        """
        decoded = decode_yolo(
            outputs[0],
            original_size=(original.shape[1], original.shape[0]),
            pad=pad,
            scale=scale,
            conf_threshold=(conf_threshold if conf_threshold is not None else self._conf_threshold),
            iou_threshold=(iou_threshold if iou_threshold is not None else self._iou_threshold),
            max_detections=self._max_detections,
        )

        if classes is not None:
            allowed = set(classes)
            decoded = [d for d in decoded if d[1] in allowed]

        detections = tuple(
            self._build_detection(original, bbox, class_id, confidence)
            for bbox, class_id, confidence in decoded
        )

        orig_shape: tuple[int, int] = (int(original.shape[0]), int(original.shape[1]))
        boxes = self._build_boxes(detections, orig_shape=orig_shape)
        return [
            DetectionResults(
                boxes=boxes,
                detections=detections,
                names=self._names,
                orig_img=original,
                orig_shape=orig_shape,
                path=path,
            )
        ]

    def _preprocess(self, image: ImageArray) -> tuple[np.ndarray, float, tuple[int, int]]:
        """Letterbox → CHW float32/255 → batch."""
        boxed, scale, pad = letterbox(image, self._input_size)
        tensor = to_tensor(boxed)
        return np.ascontiguousarray(add_batch_dim(tensor)), scale, pad

    def _build_detection(
        self,
        original: ImageArray,
        bbox: BoundingBox,
        class_id: int,
        confidence: float,
    ) -> DetectionResult:
        """Crop the bbox region from the original image and assemble the dataclass."""
        h, w = original.shape[:2]
        x1, y1, x2, y2 = bbox.as_int_xyxy()
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        if x2 > x1 and y2 > y1:
            cropped = np.ascontiguousarray(original[y1:y2, x1:x2])
        else:
            cropped = np.zeros((0, 0, 3), dtype=np.uint8)

        class_name = self._names.get(class_id, f"class_{class_id}")

        return DetectionResult(
            class_id=class_id,
            class_name=class_name,
            confidence=confidence,
            bbox=bbox,
            cropped_image=cropped,
        )

    def _build_boxes(
        self,
        detections: tuple[DetectionResult, ...],
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

    def _infer_num_classes(self) -> int | None:
        """Infer ``num_classes`` from the YOLO output shape ``(B, 4 + nc, N)``."""
        outputs = self._session.raw.get_outputs()
        if not outputs:
            return None
        shape = tuple(outputs[0].shape)
        non_batch = [d for d in shape if isinstance(d, int) and d > 1]
        if not non_batch:
            return None
        candidate = min(non_batch)
        if candidate < 5:
            return None
        return candidate - 4
