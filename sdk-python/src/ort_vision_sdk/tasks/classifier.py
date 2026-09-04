"""Image classification task using ONNX Runtime."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ort_vision_sdk.core.backend import read_metadata
from ort_vision_sdk.core.timing import SpeedTimer
from ort_vision_sdk.graph import model_names, resolve_input_size
from ort_vision_sdk.io.image import ImageInput, load_image
from ort_vision_sdk.labels import LabelSpec, resolve_labels
from ort_vision_sdk.normalization import (
    Normalization,
    is_ultralytics_classifier,
    resolve_normalization,
)
from ort_vision_sdk.postprocess.classification import softmax, topk
from ort_vision_sdk.preprocess.image import add_batch_dim, normalize, resize, to_chw
from ort_vision_sdk.results import ClassificationResults, Probs
from ort_vision_sdk.tasks.base import VisionTask
from ort_vision_sdk.types import ClassificationResult, ClassProbability, ImageArray

if TYPE_CHECKING:
    # Annotation-only; OrtSession imports onnxruntime lazily at runtime.
    import onnxruntime as ort

    from ort_vision_sdk.core.backend import InferenceBackend


class Classifier(VisionTask):
    """Image classifier wrapping an ONNX model with ImageNet-style preprocessing.

    The default configuration matches the most common torchvision/ImageNet
    convention: 224x224 RGB input, ``float32`` normalized with ImageNet mean
    and standard deviation, NCHW layout, batch size 1, softmax applied to the
    raw output.

    ``predict()`` returns ``list[ClassificationResults]`` (length 1 for a
    single image), mirroring Ultralytics' API. The envelope exposes a
    ``probs`` collection (``top1``, ``top1conf``, ``top5``, ``top5conf``,
    ``data``) and a per-image ``result`` dataclass with the resolved
    per-class probabilities.

    Override the constructor arguments when working with models that expect a
    different input size or normalization, or skip the softmax for models
    that already output probabilities.

    Example:
        >>> clf = Classifier("resnet50.onnx", labels="imagenet_labels.txt")
        >>> results = clf.predict("dog.jpg")
        >>> r = results[0]
        >>> print(r.cls, r.conf, r.name)
        >>> print(r.probs.top5, r.probs.top5conf)
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        labels: LabelSpec = None,
        providers: list[str] | None = None,
        session_options: ort.SessionOptions | None = None,
        backend: InferenceBackend | None = None,
        input_size: tuple[int, int] | None = None,
        normalization: Normalization = "auto",
        mean: tuple[float, float, float] | None = None,
        std: tuple[float, float, float] | None = None,
        apply_softmax: bool | None = None,
    ) -> None:
        """Initialize the classifier.

        Args:
            model_path: Path to the ``.onnx`` model. Ignored when ``backend``
                is provided.
            labels: Class label spec — see :func:`resolve_labels`. ``None``
                (default) reads the class names the export baked into the model
                metadata (Ultralytics' ``names``), and only falls back to
                generated ``class_<id>`` names when the model carries none.
            providers: Execution providers in preference order. Accepts short
                aliases (``"cuda"``, ``"cpu"``, ...) or canonical ORT names.
                Auto if ``None``. Ignored when ``backend`` is provided.
            session_options: Optional ORT session options. Ignored when
                ``backend`` is provided.
            backend: An explicit
                :class:`~ort_vision_sdk.core.backend.InferenceBackend` to run
                inference through (browser/Android bridge). ``None`` (default)
                uses the in-process ONNX Runtime via :class:`OrtSession`.
            input_size: Model input ``(width, height)`` in pixels. Only used
                when the model's graph leaves its spatial axes dynamic: a graph
                that declares a static size always wins, since that is the only
                shape ONNX Runtime will accept. ``None`` (default) means "ask
                the graph, fall back to ``(224, 224)``".
            normalization: Which preprocessing this model expects — see
                :data:`~ort_vision_sdk.normalization.Normalization`. ``"auto"``
                (default) reads the model's own export metadata and picks
                ``"ultralytics"`` (raw ``[0, 1]``) for an Ultralytics
                classification head, ``"imagenet"`` for everything else.
            mean: Per-channel RGB mean, overriding the preset. ``None``
                (default) takes it from ``normalization``.
            std: Per-channel RGB standard deviation, overriding the preset.
                ``None`` (default) takes it from ``normalization``.
            apply_softmax: Whether the model's output still needs a softmax.
                ``None`` (default) reads the model's metadata and answers
                ``False`` for an Ultralytics classification export, whose graph
                already ends in one — applying a second softmax to a probability
                vector keeps the ranking but flattens the confidences, so the
                top-1 stays right while every number attached to it is wrong.
                Detection covers that family; for any other model that already
                emits probabilities, pass ``False`` explicitly.

        Raises:
            ValueError: If ``normalization`` names an unknown preset, or names
                one while ``mean``/``std`` are also given.

        Warns:
            UserWarning: If the model is an Ultralytics export and the
                ``mean``/``std`` supplied are not the identity it was trained
                with.
        """
        super().__init__(
            model_path,
            providers=providers,
            session_options=session_options,
            backend=backend,
        )
        self._input_size: tuple[int, int] = resolve_input_size(
            graph_shape=self._session.input_shape,
            requested=input_size,
            fallback=(224, 224),
        )
        metadata = read_metadata(self._session)
        self._normalization: str
        self._mean: tuple[float, float, float]
        self._std: tuple[float, float, float]
        self._normalization, self._mean, self._std = resolve_normalization(
            metadata, normalization=normalization, mean=mean, std=std
        )
        self._apply_softmax: bool = (
            apply_softmax if apply_softmax is not None else not is_ultralytics_classifier(metadata)
        )

        num_classes = self._infer_num_classes()
        spec: LabelSpec = labels if labels is not None else model_names(metadata)
        self._labels: tuple[str, ...] = resolve_labels(spec, num_classes=num_classes)
        self._names: dict[int, str] = {i: name for i, name in enumerate(self._labels)}

    @property
    def applies_softmax(self) -> bool:
        """Whether a softmax is applied to the model's output before ranking.

        Resolved once at construction. Worth reading when confidences look
        compressed: a second softmax over an already-normalized vector leaves the
        ordering intact and the numbers meaningless, which is invisible to any
        check that only looks at the predicted class.

        Returns:
            bool: ``True`` when the task softmaxes the raw output itself.
        """
        return self._apply_softmax

    @property
    def normalization(self) -> str:
        """Which preprocessing this classifier applies to every image.

        One of the :data:`~ort_vision_sdk.normalization.Normalization` preset
        names, or ``"custom"`` when the caller supplied ``mean``/``std``
        directly. Worth reading when a model underperforms: feeding a classifier
        a differently prepared tensor than it was trained on degrades it without
        raising anything, so "what does this assume" is the first question.

        Returns:
            str: The name of the normalization in effect.
        """
        return self._normalization

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
        """Number of classes the model can predict."""
        return len(self._labels)

    def __call__(
        self,
        image: ImageInput,
        *,
        top_k: int | None = None,
    ) -> list[ClassificationResults]:
        """Alias for :meth:`predict` — call the classifier like a torch ``nn.Module``."""
        return self.predict(image, top_k=top_k)

    def predict(
        self,
        image: ImageInput,
        *,
        top_k: int | None = None,
    ) -> list[ClassificationResults]:
        """Run classification on a single image (synchronous).

        Args:
            image: Image source (path, bytes, ``np.ndarray``, or ``PIL.Image``).
            top_k: If set, the per-class probability tuple in
                ``results[0].result.probabilities`` is truncated to the top-k
                entries. The bulk ``probs`` view always exposes the full
                vector, regardless of ``top_k``.

        Returns:
            A 1-element list containing a :class:`ClassificationResults`
            envelope.
        """
        timer = SpeedTimer()
        path = str(image) if isinstance(image, (str, Path)) else None
        original = load_image(image)
        timer.stage("load")
        tensor = self._preprocess(original)
        timer.stage("preprocess")
        outputs = self._session.run({self._session.input_name: tensor})
        timer.stage("inference")
        return self._build_results(
            outputs,
            timer=timer,
            original=original,
            path=path,
            top_k=top_k,
        )

    async def async_predict(
        self,
        image: ImageInput,
        *,
        top_k: int | None = None,
    ) -> list[ClassificationResults]:
        """Async classification via ``asyncio.to_thread``.

        Off-loads the entire :meth:`predict` pipeline to the asyncio default
        executor's thread pool, freeing the event loop. Use in FastAPI/AnyIO
        handlers, or any async code where you don't want a single inference
        to block the loop. For high concurrency, see :meth:`ort_async_predict`.

        Args and return type match :meth:`predict` exactly.
        """
        return await asyncio.to_thread(self.predict, image, top_k=top_k)

    async def ort_async_predict(
        self,
        image: ImageInput,
        *,
        top_k: int | None = None,
    ) -> list[ClassificationResults]:
        """Async classification using ORT's native ``run_async`` for inference.

        Pre-/post-processing run on the event loop thread (cheap NumPy ops);
        the model run is dispatched to the ONNX Runtime internal thread pool
        (configured by your ``SessionOptions``). Prefer this over
        :meth:`async_predict` for high-throughput concurrent workloads where
        you want all in-flight inferences to share the ORT pool. Requires
        ``onnxruntime>=1.16``.

        Args and return type match :meth:`predict` exactly.
        """
        timer = SpeedTimer()
        path = str(image) if isinstance(image, (str, Path)) else None
        original = load_image(image)
        timer.stage("load")
        tensor = self._preprocess(original)
        timer.stage("preprocess")
        outputs = await self._session.ort_async_run({self._session.input_name: tensor})
        timer.stage("inference")
        return self._build_results(
            outputs,
            timer=timer,
            original=original,
            path=path,
            top_k=top_k,
        )

    def _build_results(
        self,
        outputs: list[np.ndarray],
        *,
        timer: SpeedTimer,
        original: ImageArray,
        path: str | None,
        top_k: int | None,
    ) -> list[ClassificationResults]:
        """Postprocess raw outputs into a :class:`ClassificationResults` envelope.

        Shared between :meth:`predict`, :meth:`async_predict` and
        :meth:`ort_async_predict` so the result-building logic stays in one
        place regardless of how the model run was scheduled.
        """
        full_probs = self._postprocess(outputs[0])

        indices, values = topk(full_probs, k=top_k)
        probabilities = tuple(
            ClassProbability(
                class_id=int(idx),
                class_name=self._labels[int(idx)],
                probability=float(val),
            )
            for idx, val in zip(indices, values, strict=True)
        )

        top = probabilities[0]
        result = ClassificationResult(
            class_id=top.class_id,
            class_name=top.class_name,
            confidence=top.probability,
            image=original,
            probabilities=probabilities,
        )
        probs_view = Probs(data=full_probs.astype(np.float64, copy=False))
        timer.stage("postprocess")

        return [
            ClassificationResults(
                probs=probs_view,
                result=result,
                names=self._names,
                orig_img=original,
                orig_shape=(int(original.shape[0]), int(original.shape[1])),
                path=path,
                speed=timer.speed(),
            )
        ]

    def _preprocess(self, image: ImageArray) -> np.ndarray:
        """Resize → normalize → CHW → batch."""
        resized = resize(image, self._input_size)
        normalized = normalize(resized, mean=self._mean, std=self._std)
        chw = to_chw(normalized)
        return add_batch_dim(chw).astype(np.float32, copy=False)

    def _postprocess(self, output: np.ndarray) -> np.ndarray:
        """Squeeze batch dim and (optionally) apply softmax.

        Args:
            output: Raw classifier output, ``(1, num_classes)`` or
                ``(num_classes,)``.

        Returns:
            A 1-D ``float32`` per-class probability vector. Softmax is
            applied when :pyattr:`_apply_softmax` is ``True``; otherwise the
            input is returned cast to ``float32``.

        Raises:
            ValueError: If the output cannot be squeezed down to a 1-D vector
                (i.e. the model emits an unexpected shape).
        """
        scores = np.squeeze(output, axis=0) if output.ndim == 2 else output
        if scores.ndim != 1:
            raise ValueError(
                f"Expected classifier output to reduce to a 1-D vector, got shape {scores.shape}."
            )
        return softmax(scores) if self._apply_softmax else scores.astype(np.float32, copy=False)

    def _infer_num_classes(self) -> int | None:
        """Read num_classes from the model's first output last static dim."""
        output_shapes = self._session.output_shapes
        if not output_shapes:
            return None
        last_dim = output_shapes[0][-1]
        return int(last_dim) if isinstance(last_dim, int) else None
