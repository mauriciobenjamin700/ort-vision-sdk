"""The contract a fused pipeline carries inside its own ``.onnx`` file.

A pipeline produced by :mod:`ort_vision_sdk.compose` is a single graph that
already contains the detector, the crop-and-resize bridge and the classifier.
Everything the runtime needs to drive it — the letterbox resolution, the crop
resolution, where the crops come from, how many detections the graph emits,
the class names of both stages — is decided at fusion time and would otherwise
have to be restated by whoever loads the file. A restatement drifts.

So the fusion writes it into the model's own metadata, and
:class:`~ort_vision_sdk.tasks.pipeline.DetectClassify` reads it back. This
module owns both directions and deliberately imports nothing from ``onnx``:
building a pipeline is a build-time step that needs the ``[compose]`` extra,
but *running* one must work on a plain ``onnxruntime`` install.

The same idea already applies to single-stage models — see
:mod:`ort_vision_sdk.graph`, which reads the input resolution and the class
names straight out of the graph rather than trusting configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ort_vision_sdk.graph import parse_names

__all__ = [
    "FUSION_KIND_DETECT_CLASSIFY",
    "INPUT_IMAGE",
    "INPUT_PAD",
    "INPUT_SCALE",
    "INPUT_SOURCE",
    "METADATA_PREFIX",
    "OUTPUT_BOXES",
    "OUTPUT_CLASSES",
    "OUTPUT_NUM_DETECTIONS",
    "OUTPUT_PROBS",
    "OUTPUT_SCORES",
    "CropSource",
    "FusionSpec",
]

CropSource = Literal["detector_input", "original"]
"""Which tensor the bridge crops the detected boxes out of.

- ``"detector_input"``: the letterboxed tensor already fed to the detector. The
  fused graph then has a **single** image input, but a small object is cropped
  out of its downscaled copy — a 40x40 px box in a 640x640 letterbox is
  upsampled to the classifier's resolution from 40x40 pixels of real detail.
- ``"original"``: a second, full-resolution image input. The bridge undoes the
  letterbox transform in-graph (using the ``letterbox_scale`` and
  ``letterbox_pad`` inputs) and crops at native resolution. Two tensors to
  feed, still one session and one model load.
"""

FUSION_KIND_DETECT_CLASSIFY = "detect_classify"
"""Value of the ``ovs.kind`` metadata key for a detector→classifier pipeline."""

METADATA_PREFIX = "ovs."
"""Namespace for every metadata key the fusion writes.

Namespaced on purpose: the detector's own Ultralytics metadata (``names``,
``task``, ``imgsz``) is carried over into the fused model, and an un-prefixed
key would either collide with it or be mistaken for it.
"""

INPUT_IMAGE = "images"
"""Name of the fused graph's letterboxed detector input, ``(1, 3, H, W)`` float32 in ``[0, 1]``."""

INPUT_SOURCE = "source_image"
"""Name of the full-resolution input, ``(1, 3, H, W)`` float32 in ``[0, 1]``.

Present only when ``crop_source == "original"``.
"""

INPUT_SCALE = "letterbox_scale"
"""Name of the ``(1,)`` float32 letterbox scale factor. Only with ``crop_source == "original"``."""

INPUT_PAD = "letterbox_pad"
"""Name of the ``(2,)`` float32 ``(pad_left, pad_top)``. Only with ``crop_source == "original"``."""

OUTPUT_BOXES = "boxes"
"""Name of the ``(K, 4)`` float32 xyxy output, in **letterboxed** input pixels."""

OUTPUT_SCORES = "scores"
"""Name of the ``(K,)`` float32 detection-confidence output."""

OUTPUT_CLASSES = "classes"
"""Name of the ``(K,)`` int64 detector-class output."""

OUTPUT_NUM_DETECTIONS = "num_detections"
"""Name of the ``(1,)`` int64 output holding how many of the ``K`` rows are real."""

OUTPUT_PROBS = "probs"
"""Name of the ``(K, num_classifier_classes)`` float32 classifier output, one row per box."""

_KEY_KIND = "kind"
_KEY_SDK_VERSION = "sdk_version"
_KEY_INPUT_SIZE = "input_size"
_KEY_CROP_SIZE = "crop_size"
_KEY_CROP_SOURCE = "crop_source"
_KEY_MAX_DETECTIONS = "max_detections"
_KEY_CONF_THRESHOLD = "conf_threshold"
_KEY_IOU_THRESHOLD = "iou_threshold"
_KEY_APPLY_SOFTMAX = "apply_softmax"
_KEY_DETECTOR_NAMES = "detector_names"
_KEY_CLASSIFIER_NAMES = "classifier_names"

_DYNAMIC = "dynamic"


def _encode_size(size: tuple[int, int]) -> str:
    """Encode a ``(width, height)`` pair as ``"640,640"``."""
    return f"{size[0]},{size[1]}"


def _decode_size(raw: str | None) -> tuple[int, int] | None:
    """Decode a ``"640,640"`` pair, returning ``None`` when malformed."""
    if not raw:
        return None
    parts = raw.split(",")
    if len(parts) != 2:
        return None
    try:
        width, height = int(parts[0]), int(parts[1])
    except ValueError:
        return None
    if width < 1 or height < 1:
        return None
    return width, height


def _decode_float(raw: str | None, fallback: float) -> float:
    """Decode a float, falling back when the value is missing or malformed."""
    if not raw:
        return fallback
    try:
        return float(raw)
    except ValueError:
        return fallback


@dataclass(frozen=True, slots=True)
class FusionSpec:
    """Everything a fused pipeline declares about how it must be driven.

    Attributes:
        input_size: ``(width, height)`` the detector stage expects, i.e. the
            resolution the caller must letterbox to.
        crop_size: ``(width, height)`` the classifier stage expects, i.e. the
            size every crop is resampled to inside the graph.
        crop_source: Which tensor the crops are taken from — see
            :data:`CropSource`.
        max_detections: Fixed number of rows ``K`` every output carries, with
            the surplus zero-padded and :data:`OUTPUT_NUM_DETECTIONS` saying how
            many are real. ``None`` means the graph emits exactly as many rows
            as survived NMS (dynamic shapes).
        conf_threshold: Score threshold baked into the graph's NMS node.
        iou_threshold: IoU threshold baked into the graph's NMS node.
        apply_softmax: Whether the classifier stage emits logits that still
            need a softmax. ``False`` when the classifier's own graph ends in
            one.
        detector_names: Class id → name for the detector stage, or ``None``
            when the source model carried none.
        classifier_names: Class id → name for the classifier stage, or ``None``.
        sdk_version: Version of ``ort-vision-sdk`` that produced the file.
    """

    input_size: tuple[int, int]
    crop_size: tuple[int, int]
    crop_source: CropSource
    max_detections: int | None
    conf_threshold: float
    iou_threshold: float
    apply_softmax: bool
    detector_names: dict[int, str] | None
    classifier_names: dict[int, str] | None
    sdk_version: str
    kind: str = FUSION_KIND_DETECT_CLASSIFY

    @property
    def needs_source_image(self) -> bool:
        """Whether driving this pipeline requires feeding the full-resolution input.

        Returns:
            bool: ``True`` when the graph declares :data:`INPUT_SOURCE`,
            :data:`INPUT_SCALE` and :data:`INPUT_PAD` alongside
            :data:`INPUT_IMAGE`.
        """
        return self.crop_source == "original"

    def to_metadata(self) -> dict[str, str]:
        """Encode the spec as ONNX custom metadata entries.

        Every key is namespaced with :data:`METADATA_PREFIX` so it cannot be
        confused with the detector's own Ultralytics metadata, which the fusion
        copies over untouched.

        Returns:
            dict[str, str]: The ``ovs.*`` map to write into ``metadata_props``.
        """
        entries: dict[str, str] = {
            _KEY_KIND: self.kind,
            _KEY_SDK_VERSION: self.sdk_version,
            _KEY_INPUT_SIZE: _encode_size(self.input_size),
            _KEY_CROP_SIZE: _encode_size(self.crop_size),
            _KEY_CROP_SOURCE: self.crop_source,
            _KEY_MAX_DETECTIONS: (
                _DYNAMIC if self.max_detections is None else str(self.max_detections)
            ),
            _KEY_CONF_THRESHOLD: repr(self.conf_threshold),
            _KEY_IOU_THRESHOLD: repr(self.iou_threshold),
            _KEY_APPLY_SOFTMAX: "1" if self.apply_softmax else "0",
        }
        if self.detector_names is not None:
            entries[_KEY_DETECTOR_NAMES] = repr(self.detector_names)
        if self.classifier_names is not None:
            entries[_KEY_CLASSIFIER_NAMES] = repr(self.classifier_names)
        return {f"{METADATA_PREFIX}{key}": value for key, value in entries.items()}

    @classmethod
    def from_metadata(cls, metadata: dict[str, str] | None) -> FusionSpec | None:
        """Read the spec back out of a model's custom metadata.

        Args:
            metadata (dict[str, str] | None): The model's custom metadata map,
                as exposed by :func:`~ort_vision_sdk.core.backend.read_metadata`.

        Returns:
            FusionSpec | None: The decoded spec, or ``None`` when the model is
            not a fused pipeline — i.e. it carries no ``ovs.kind`` entry, or one
            naming a pipeline kind this version does not know how to drive.
            Individual malformed entries fall back to the value a fusion would
            have used by default, since a single bad float is not a reason to
            reject an otherwise loadable pipeline; a malformed
            ``input_size``/``crop_size`` is, because there is no safe default
            for a resolution.
        """
        if not metadata:
            return None
        read = {
            key[len(METADATA_PREFIX) :]: value
            for key, value in metadata.items()
            if key.startswith(METADATA_PREFIX)
        }
        if read.get(_KEY_KIND) != FUSION_KIND_DETECT_CLASSIFY:
            return None

        input_size = _decode_size(read.get(_KEY_INPUT_SIZE))
        crop_size = _decode_size(read.get(_KEY_CROP_SIZE))
        if input_size is None or crop_size is None:
            return None

        raw_max = read.get(_KEY_MAX_DETECTIONS, _DYNAMIC)
        max_detections: int | None
        if raw_max == _DYNAMIC:
            max_detections = None
        else:
            try:
                max_detections = int(raw_max)
            except ValueError:
                max_detections = None

        crop_source: CropSource = (
            "original" if read.get(_KEY_CROP_SOURCE) == "original" else "detector_input"
        )

        return cls(
            input_size=input_size,
            crop_size=crop_size,
            crop_source=crop_source,
            max_detections=max_detections,
            conf_threshold=_decode_float(read.get(_KEY_CONF_THRESHOLD), 0.25),
            iou_threshold=_decode_float(read.get(_KEY_IOU_THRESHOLD), 0.45),
            apply_softmax=read.get(_KEY_APPLY_SOFTMAX, "1") != "0",
            detector_names=parse_names(read.get(_KEY_DETECTOR_NAMES)),
            classifier_names=parse_names(read.get(_KEY_CLASSIFIER_NAMES)),
            sdk_version=read.get(_KEY_SDK_VERSION, ""),
            kind=FUSION_KIND_DETECT_CLASSIFY,
        )
