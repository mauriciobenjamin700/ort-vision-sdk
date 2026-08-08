"""Run a fused detect→classify pipeline built by :mod:`ort_vision_sdk.compose`."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ort_vision_sdk.core.backend import read_metadata
from ort_vision_sdk.core.exceptions import FusionError
from ort_vision_sdk.core.timing import SpeedTimer
from ort_vision_sdk.fusion import (
    INPUT_IMAGE,
    INPUT_PAD,
    INPUT_SCALE,
    INPUT_SOURCE,
    OUTPUT_BOXES,
    OUTPUT_CLASSES,
    OUTPUT_NUM_DETECTIONS,
    OUTPUT_PROBS,
    OUTPUT_SCORES,
    FusionSpec,
)
from ort_vision_sdk.io.image import ImageInput, load_image
from ort_vision_sdk.labels import LabelSpec, default_labels, resolve_labels
from ort_vision_sdk.postprocess.classification import softmax, topk
from ort_vision_sdk.preprocess.image import add_batch_dim, letterbox, to_tensor
from ort_vision_sdk.results import Boxes, DetectClassifyResults
from ort_vision_sdk.tasks.base import VisionTask, require_detections
from ort_vision_sdk.types import (
    BoundingBox,
    ClassificationResult,
    ClassProbability,
    DetectionResult,
    ImageArray,
)

if TYPE_CHECKING:
    # Annotation-only; OrtSession imports onnxruntime lazily at runtime.
    import onnxruntime as ort

    from ort_vision_sdk.core.backend import InferenceBackend

_OUTPUTS = [
    OUTPUT_BOXES,
    OUTPUT_SCORES,
    OUTPUT_CLASSES,
    OUTPUT_NUM_DETECTIONS,
    OUTPUT_PROBS,
]


class DetectClassify(VisionTask):
    """Detector and classifier running as a single ONNX model.

    Takes a pipeline produced by
    :func:`~ort_vision_sdk.compose.detect_classify.fuse_detect_classify` and
    drives it end to end: letterbox the image, one inference, and per-detection
    results whose ``classification`` is already filled in. There is no second
    session, no second model load, and no Python round trip for the crops —
    detection, cropping, resizing and classification all happen inside the
    graph.

    Everything the pipeline needs to know about itself — the resolution to
    letterbox to, whether it wants the full-resolution image as well, whether
    its classifier output still needs a softmax, the class names of both
    stages — was written into the file at fusion time and is read back here.
    Nothing is restated, and nothing can drift.

    Example:
        >>> pipeline = DetectClassify("pipeline.onnx")
        >>> result = pipeline.predict("flock.jpg")[0]
        >>> for detection in result:
        ...     print(detection.name, detection.conf, detection.classification.name)
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        labels: LabelSpec = None,
        classifier_labels: LabelSpec = None,
        raise_on_empty: bool = False,
        providers: list[str] | None = None,
        session_options: ort.SessionOptions | None = None,
        backend: InferenceBackend | None = None,
    ) -> None:
        """Load a fused pipeline.

        Args:
            model_path: Path to the fused ``.onnx``. Ignored when ``backend``
                is provided.
            labels: Class label spec for the **detection** stage — see
                :func:`~ort_vision_sdk.labels.resolve_labels`. ``None``
                (default) uses the names recorded at fusion time. A fusion
                records a name for every class it can count, so the fallback
                below it — :func:`~ort_vision_sdk.labels.default_labels` with no
                count, i.e. the COCO preset — is reached only by a pipeline
                built before that was true.
            classifier_labels: Class label spec for the **classification**
                stage. ``None`` uses the recorded names, falling back to
                generated ``class_<id>`` names.
            raise_on_empty: If ``True``, a run that finds nothing raises
                :class:`~ort_vision_sdk.core.exceptions.NoDetectionsError`
                instead of returning an empty envelope. Default ``False``,
                because looking and finding nothing is a successful inference.
                Turn it on when an empty result means the surrounding pipeline
                should stop rather than carry on with zero rows. Can be
                overridden per :meth:`predict` call.
            providers: Execution providers in preference order. Auto if
                ``None``. Ignored when ``backend`` is provided.
            session_options: Optional ORT session options. Ignored when
                ``backend`` is provided.
            backend: An explicit
                :class:`~ort_vision_sdk.core.backend.InferenceBackend` to run
                inference through. It must expose the model's metadata, since
                that is where the pipeline's configuration lives.

        Raises:
            FusionError: If the model carries no pipeline metadata — i.e. it is
                a plain detector or classifier rather than something
                :mod:`ort_vision_sdk.compose` produced, or the backend cannot
                read metadata at all — or if it declares the metadata but not
                every output the pipeline contract requires.
        """
        super().__init__(
            model_path,
            providers=providers,
            session_options=session_options,
            backend=backend,
        )
        spec = FusionSpec.from_metadata(read_metadata(self._session))
        if spec is None:
            raise FusionError(
                "This model carries no fused-pipeline metadata, so DetectClassify cannot tell "
                "how to drive it. Build one with ort_vision_sdk.compose.fuse_detect_classify, "
                "or load a plain model with Detector/Classifier instead."
            )
        missing = [name for name in _OUTPUTS if name not in self._session.output_names]
        if missing:
            raise FusionError(
                f"This model claims to be a fused pipeline but does not declare "
                f"{', '.join(missing)}. It was most likely produced by a different — or newer — "
                f"version of ort_vision_sdk.compose; re-fuse it with this one."
            )
        self._spec: FusionSpec = spec
        self._raise_on_empty: bool = raise_on_empty
        self._labels: tuple[str, ...] = resolve_labels(
            labels if labels is not None else spec.detector_names or default_labels(None)
        )
        self._names: dict[int, str] = dict(enumerate(self._labels))
        self._classifier_labels: tuple[str, ...] = resolve_labels(
            classifier_labels if classifier_labels is not None else spec.classifier_names,
            num_classes=self._classifier_classes(),
        )
        self._classifier_names: dict[int, str] = dict(enumerate(self._classifier_labels))

    @property
    def spec(self) -> FusionSpec:
        """The pipeline configuration recorded in the model at fusion time."""
        return self._spec

    @property
    def input_size(self) -> tuple[int, int]:
        """The ``(width, height)`` the detection stage runs at."""
        return self._spec.input_size

    @property
    def labels(self) -> tuple[str, ...]:
        """Detector class labels indexed by class id."""
        return self._labels

    @property
    def names(self) -> dict[int, str]:
        """Detector class id → class name (matches Ultralytics' ``model.names``)."""
        return self._names

    @property
    def classifier_labels(self) -> tuple[str, ...]:
        """Classifier class labels indexed by class id."""
        return self._classifier_labels

    @property
    def classifier_names(self) -> dict[int, str]:
        """Classifier class id → class name."""
        return self._classifier_names

    def __call__(
        self,
        image: ImageInput,
        *,
        conf_threshold: float | None = None,
        classes: list[int] | None = None,
        top_k: int | None = None,
        raise_on_empty: bool | None = None,
    ) -> list[DetectClassifyResults]:
        """Alias for :meth:`predict` — call the pipeline like a torch ``nn.Module``."""
        return self.predict(
            image,
            conf_threshold=conf_threshold,
            classes=classes,
            top_k=top_k,
            raise_on_empty=raise_on_empty,
        )

    def predict(
        self,
        image: ImageInput,
        *,
        conf_threshold: float | None = None,
        classes: list[int] | None = None,
        top_k: int | None = None,
        raise_on_empty: bool | None = None,
    ) -> list[DetectClassifyResults]:
        """Run the pipeline on a single image (synchronous).

        Args:
            image: Image source (path, bytes, ``np.ndarray``, or ``PIL.Image``).
            conf_threshold: Drop detections scoring below this. The graph's own
                NMS threshold was fixed at fusion time and cannot be lowered
                here — this only filters further, so passing a value below
                ``spec.conf_threshold`` changes nothing.
            classes: If set, keep only detections whose detector ``class_id`` is
                in this list. ``None`` (default) keeps all classes.
            top_k: Truncate each detection's ``classification.probabilities``
                tuple to its top-k entries. ``None`` keeps every class.
            raise_on_empty: Override the constructor's setting for this call.

        Returns:
            A 1-element list containing a :class:`DetectClassifyResults`
            envelope.

        Raises:
            NoDetectionsError: If nothing survives the thresholds and
                ``raise_on_empty`` is in effect for this call.
        """
        timer = SpeedTimer()
        path = str(image) if isinstance(image, (str, Path)) else None
        original = load_image(image)
        timer.stage("load")
        feeds, scale, pad = self._preprocess(original)
        timer.stage("preprocess")
        outputs = self._session.run(feeds, output_names=_OUTPUTS)
        timer.stage("inference")
        return self._build_results(
            outputs,
            timer=timer,
            original=original,
            path=path,
            scale=scale,
            pad=pad,
            conf_threshold=conf_threshold,
            classes=classes,
            top_k=top_k,
            raise_on_empty=raise_on_empty,
        )

    async def async_predict(
        self,
        image: ImageInput,
        *,
        conf_threshold: float | None = None,
        classes: list[int] | None = None,
        top_k: int | None = None,
        raise_on_empty: bool | None = None,
    ) -> list[DetectClassifyResults]:
        """Async pipeline run via ``asyncio.to_thread``.

        Off-loads the whole of :meth:`predict` to the asyncio default
        executor's thread pool, freeing the event loop. Use in FastAPI/AnyIO
        handlers. For high-concurrency workloads, see :meth:`ort_async_predict`.

        Args and return type match :meth:`predict` exactly.
        """
        return await asyncio.to_thread(
            self.predict,
            image,
            conf_threshold=conf_threshold,
            classes=classes,
            top_k=top_k,
            raise_on_empty=raise_on_empty,
        )

    async def ort_async_predict(
        self,
        image: ImageInput,
        *,
        conf_threshold: float | None = None,
        classes: list[int] | None = None,
        top_k: int | None = None,
        raise_on_empty: bool | None = None,
    ) -> list[DetectClassifyResults]:
        """Async pipeline run using ORT's native ``run_async`` for the model step.

        Letterboxing and result assembly run on the event loop thread (NumPy
        ops); the model run — which here covers detection, cropping and
        classification in one call — is dispatched to the ONNX Runtime internal
        thread pool. Requires ``onnxruntime>=1.16``.

        Args and return type match :meth:`predict` exactly.
        """
        timer = SpeedTimer()
        path = str(image) if isinstance(image, (str, Path)) else None
        original = load_image(image)
        timer.stage("load")
        feeds, scale, pad = self._preprocess(original)
        timer.stage("preprocess")
        outputs = await self._session.ort_async_run(feeds, output_names=_OUTPUTS)
        timer.stage("inference")
        return self._build_results(
            outputs,
            timer=timer,
            original=original,
            path=path,
            scale=scale,
            pad=pad,
            conf_threshold=conf_threshold,
            classes=classes,
            top_k=top_k,
            raise_on_empty=raise_on_empty,
        )

    def _preprocess(
        self, image: ImageArray
    ) -> tuple[dict[str, np.ndarray], float, tuple[int, int]]:
        """Letterbox the image and build the graph's feeds.

        A pipeline fused with ``crop_source="original"`` takes the untouched
        image as a second input, plus the scale and padding of the letterbox —
        that is what lets the graph undo the letterbox transform internally and
        crop at native resolution instead of from the downscaled copy.

        Args:
            image: The source image, HWC uint8 RGB.

        Returns:
            tuple[dict[str, np.ndarray], float, tuple[int, int]]: The feeds, the
            letterbox scale, and the ``(pad_left, pad_top)`` offsets.
        """
        boxed, scale, pad = letterbox(image, self._spec.input_size)
        feeds: dict[str, np.ndarray] = {
            INPUT_IMAGE: np.ascontiguousarray(add_batch_dim(to_tensor(boxed)))
        }
        if self._spec.needs_source_image:
            feeds[INPUT_SOURCE] = np.ascontiguousarray(add_batch_dim(to_tensor(image)))
            feeds[INPUT_SCALE] = np.asarray([scale], dtype=np.float32)
            feeds[INPUT_PAD] = np.asarray(pad, dtype=np.float32)
        return feeds, scale, pad

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
        classes: list[int] | None,
        top_k: int | None,
        raise_on_empty: bool | None,
    ) -> list[DetectClassifyResults]:
        """Assemble the envelope from the graph's five outputs.

        Only the first ``num_detections`` rows are read: the rest are the
        padding that keeps the graph's shapes static, and their boxes and
        probabilities are meaningless.

        Args:
            outputs: The graph's ``boxes``, ``scores``, ``classes``,
                ``num_detections`` and ``probs``, in that order.
            timer: The stage timer to close out.
            original: The source image.
            path: Source path, when the input was one.
            scale: Letterbox scale factor.
            pad: ``(pad_left, pad_top)`` letterbox offsets.
            conf_threshold: Optional extra confidence filter.
            classes: Optional detector-class allowlist.
            top_k: Optional truncation of each classification's probability tuple.
            raise_on_empty: Whether an empty result is an error for this call.

        Returns:
            A 1-element list containing the envelope.

        Raises:
            NoDetectionsError: If nothing survives and ``raise_on_empty`` is in
                effect for this call.
        """
        boxes_raw, scores_raw, classes_raw, count_raw, probs_raw = outputs
        valid = min(int(count_raw.reshape(-1)[0]), boxes_raw.shape[0])
        allowed = set(classes) if classes is not None else None
        minimum = conf_threshold if conf_threshold is not None else 0.0

        orig_shape: tuple[int, int] = (int(original.shape[0]), int(original.shape[1]))
        detections: list[DetectionResult] = []
        for row in range(valid):
            class_id = int(classes_raw[row])
            confidence = float(scores_raw[row])
            if confidence < minimum or (allowed is not None and class_id not in allowed):
                continue
            bbox = self._to_original(boxes_raw[row], scale=scale, pad=pad, orig_shape=orig_shape)
            crop = _crop(original, bbox)
            detections.append(
                DetectionResult(
                    class_id=class_id,
                    class_name=self._names.get(class_id, f"class_{class_id}"),
                    confidence=confidence,
                    bbox=bbox,
                    cropped_image=crop,
                    classification=self._to_classification(probs_raw[row], crop, top_k=top_k),
                )
            )

        frozen = tuple(detections)
        require_detections(
            len(frozen),
            raise_on_empty=(raise_on_empty if raise_on_empty is not None else self._raise_on_empty),
            conf_threshold=max(minimum, self._spec.conf_threshold),
            classes=classes,
            path=path,
        )
        timer.stage("postprocess")
        return [
            DetectClassifyResults(
                boxes=_bulk_boxes(frozen, orig_shape=orig_shape),
                detections=frozen,
                names=self._names,
                classifier_names=self._classifier_names,
                orig_img=original,
                orig_shape=orig_shape,
                path=path,
                speed=timer.speed(),
            )
        ]

    def _to_original(
        self,
        box: np.ndarray,
        *,
        scale: float,
        pad: tuple[int, int],
        orig_shape: tuple[int, int],
    ) -> BoundingBox:
        """Map one letterboxed xyxy box back onto the original image.

        The graph always reports boxes in the detector's letterboxed pixel
        space, whichever crop source it was fused with, so this undo is the same
        arithmetic :func:`~ort_vision_sdk.postprocess.detection.decode_yolo`
        applies — and both crop sources therefore agree on the coordinates.

        Args:
            box: The ``(4,)`` xyxy row, in letterboxed pixels.
            scale: Letterbox scale factor.
            pad: ``(pad_left, pad_top)`` letterbox offsets.
            orig_shape: ``(height, width)`` of the original image.

        Returns:
            BoundingBox: The box in original-image pixel coordinates, clipped to
            the image.
        """
        pad_left, pad_top = pad
        height, width = orig_shape
        x1 = float(np.clip((float(box[0]) - pad_left) / scale, 0.0, width))
        y1 = float(np.clip((float(box[1]) - pad_top) / scale, 0.0, height))
        x2 = float(np.clip((float(box[2]) - pad_left) / scale, 0.0, width))
        y2 = float(np.clip((float(box[3]) - pad_top) / scale, 0.0, height))
        return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)

    def _to_classification(
        self,
        row: np.ndarray,
        crop: ImageArray,
        *,
        top_k: int | None,
    ) -> ClassificationResult:
        """Turn one row of the classifier output into a result dataclass.

        Args:
            row: The ``(num_classes,)`` output row for this detection.
            crop: The crop the row describes, carried on the result so callers
                can display what was classified. This is the crop taken in
                NumPy from the original image, not the resampled tensor the
                graph fed its classifier.
            top_k: Optional truncation of the probability tuple.

        Returns:
            ClassificationResult: Top-1 plus the per-class probabilities.
        """
        scores = softmax(row) if self._spec.apply_softmax else row.astype(np.float32, copy=False)
        indices, values = topk(scores, k=top_k)
        probabilities = tuple(
            ClassProbability(
                class_id=int(index),
                class_name=self._classifier_label(int(index)),
                probability=float(value),
            )
            for index, value in zip(indices, values, strict=True)
        )
        top = probabilities[0]
        return ClassificationResult(
            class_id=top.class_id,
            class_name=top.class_name,
            confidence=top.probability,
            image=crop,
            probabilities=probabilities,
        )

    def _classifier_label(self, class_id: int) -> str:
        """Name a classifier class, tolerating a label map shorter than the output."""
        if 0 <= class_id < len(self._classifier_labels):
            return self._classifier_labels[class_id]
        return f"class_{class_id}"

    def _classifier_classes(self) -> int | None:
        """Read the classifier stage's class count off the ``probs`` output shape.

        Returns:
            int | None: The class count, or ``None`` when the graph leaves that
            axis dynamic or does not declare a ``probs`` output — in which case
            label resolution falls back to whatever the fusion recorded.
        """
        names = self._session.output_names
        if OUTPUT_PROBS not in names:
            return None
        shape = self._session.output_shapes[names.index(OUTPUT_PROBS)]
        if not shape:
            return None
        last = shape[-1]
        return int(last) if isinstance(last, int) else None


def _crop(image: ImageArray, bbox: BoundingBox) -> ImageArray:
    """Cut the box region out of the original image.

    Args:
        image: The source image, HWC uint8 RGB.
        bbox: The box, in original-image pixel coordinates.

    Returns:
        ImageArray: The cropped region, or a zero-sized array for a box with no
        area.
    """
    height, width = image.shape[:2]
    x1, y1, x2, y2 = bbox.as_int_xyxy()
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((0, 0, 3), dtype=np.uint8)
    return np.ascontiguousarray(image[y1:y2, x1:x2])


def _bulk_boxes(
    detections: tuple[DetectionResult, ...],
    *,
    orig_shape: tuple[int, int],
) -> Boxes:
    """Assemble the bulk-array ``Boxes`` view from per-instance dataclasses.

    Args:
        detections: The surviving detections.
        orig_shape: ``(height, width)`` of the original image, needed by the
            view's normalized accessors.

    Returns:
        Boxes: The bulk view, empty-shaped when nothing survived.
    """
    if not detections:
        return Boxes(
            xyxy=np.empty((0, 4), dtype=np.float64),
            cls=np.empty((0,), dtype=np.int64),
            conf=np.empty((0,), dtype=np.float64),
            orig_shape=orig_shape,
        )
    return Boxes(
        xyxy=np.asarray([d.bbox.xyxy for d in detections], dtype=np.float64),
        cls=np.asarray([d.class_id for d in detections], dtype=np.int64),
        conf=np.asarray([d.confidence for d in detections], dtype=np.float64),
        orig_shape=orig_shape,
    )
