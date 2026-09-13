"""ort-vision-sdk: high-level Python SDK for computer vision inference with ONNX Runtime.

Everything the package considers public is importable from this module. That is
a contract with two halves, and the second is what makes it worth stating:

- **Nothing below the root needs importing.** A consumer never has to know that
  ``NoDetectionsError`` lives in ``core.exceptions`` or that ``decode_yolo``
  lives in ``postprocess.detection``. Submodule paths are an implementation
  detail and are free to move.
- **The surface mirrors the web SDK.** ``@mauriciobenjamin700/ort-vision-sdk-web``
  exports the same names from its own root, so a symbol reachable there is
  reachable here under this project's naming convention (``decode_yolo`` for
  ``decodeYolo``). A name exported on one side and missing on the other is a
  defect, not a language difference.

Re-exports are written as ``from x import Y as Y`` alongside ``__all__`` because
strict type checkers treat a plain ``from x import Y`` in a package root as a
private import and flag every consumer that uses it.
"""

from ort_vision_sdk.core import STAGES as STAGES
from ort_vision_sdk.core import FusionError as FusionError
from ort_vision_sdk.core import ImageLoadError as ImageLoadError
from ort_vision_sdk.core import InferenceBackend as InferenceBackend
from ort_vision_sdk.core import InferenceError as InferenceError
from ort_vision_sdk.core import LabelMapError as LabelMapError
from ort_vision_sdk.core import MetadataBackend as MetadataBackend
from ort_vision_sdk.core import ModelLoadError as ModelLoadError
from ort_vision_sdk.core import NoDetectionsError as NoDetectionsError
from ort_vision_sdk.core import OrtSession as OrtSession
from ort_vision_sdk.core import OrtVisionError as OrtVisionError
from ort_vision_sdk.core import ProviderNotAvailableError as ProviderNotAvailableError
from ort_vision_sdk.core import SpeedTimer as SpeedTimer
from ort_vision_sdk.core import Stage as Stage
from ort_vision_sdk.core import available_providers as available_providers
from ort_vision_sdk.core import read_metadata as read_metadata
from ort_vision_sdk.core import resolve_providers as resolve_providers
from ort_vision_sdk.dtypes import as_float32 as as_float32
from ort_vision_sdk.dtypes import numpy_dtype_for as numpy_dtype_for
from ort_vision_sdk.fusion import FUSION_KIND_DETECT_CLASSIFY as FUSION_KIND_DETECT_CLASSIFY
from ort_vision_sdk.fusion import (
    FUSION_KIND_DETECT_SEGMENT_CLASSIFY as FUSION_KIND_DETECT_SEGMENT_CLASSIFY,
)
from ort_vision_sdk.fusion import INPUT_IMAGE as INPUT_IMAGE
from ort_vision_sdk.fusion import INPUT_PAD as INPUT_PAD
from ort_vision_sdk.fusion import INPUT_SCALE as INPUT_SCALE
from ort_vision_sdk.fusion import INPUT_SOURCE as INPUT_SOURCE
from ort_vision_sdk.fusion import METADATA_PREFIX as METADATA_PREFIX
from ort_vision_sdk.fusion import OUTPUT_BOXES as OUTPUT_BOXES
from ort_vision_sdk.fusion import OUTPUT_CLASSES as OUTPUT_CLASSES
from ort_vision_sdk.fusion import OUTPUT_MASKS as OUTPUT_MASKS
from ort_vision_sdk.fusion import OUTPUT_NUM_DETECTIONS as OUTPUT_NUM_DETECTIONS
from ort_vision_sdk.fusion import OUTPUT_PROBS as OUTPUT_PROBS
from ort_vision_sdk.fusion import OUTPUT_SCORES as OUTPUT_SCORES
from ort_vision_sdk.fusion import CropSource as CropSource
from ort_vision_sdk.fusion import FusionSpec as FusionSpec
from ort_vision_sdk.graph import model_names as model_names
from ort_vision_sdk.graph import parse_names as parse_names
from ort_vision_sdk.graph import resolve_input_size as resolve_input_size
from ort_vision_sdk.graph import spatial_input_size as spatial_input_size
from ort_vision_sdk.io import ImageInput as ImageInput
from ort_vision_sdk.io import load_image as load_image
from ort_vision_sdk.labels import COCO_CLASSES as COCO_CLASSES
from ort_vision_sdk.labels import LabelSpec as LabelSpec
from ort_vision_sdk.labels import default_labels as default_labels
from ort_vision_sdk.labels import resolve_labels as resolve_labels
from ort_vision_sdk.normalization import CUSTOM_NORMALIZATION as CUSTOM_NORMALIZATION
from ort_vision_sdk.normalization import IDENTITY_MEAN as IDENTITY_MEAN
from ort_vision_sdk.normalization import IDENTITY_STD as IDENTITY_STD
from ort_vision_sdk.normalization import IMAGENET_MEAN as IMAGENET_MEAN
from ort_vision_sdk.normalization import IMAGENET_STD as IMAGENET_STD
from ort_vision_sdk.normalization import NORMALIZATION_PRESETS as NORMALIZATION_PRESETS
from ort_vision_sdk.normalization import Normalization as Normalization
from ort_vision_sdk.normalization import is_ultralytics_classifier as is_ultralytics_classifier
from ort_vision_sdk.normalization import resolve_normalization as resolve_normalization
from ort_vision_sdk.postprocess import DecodedSegmentation as DecodedSegmentation
from ort_vision_sdk.postprocess import batched_nms as batched_nms
from ort_vision_sdk.postprocess import decode_yolo as decode_yolo
from ort_vision_sdk.postprocess import decode_yolo_anchors as decode_yolo_anchors
from ort_vision_sdk.postprocess import decode_yolo_seg as decode_yolo_seg
from ort_vision_sdk.postprocess import nms as nms
from ort_vision_sdk.postprocess import softmax as softmax
from ort_vision_sdk.postprocess import topk as topk
from ort_vision_sdk.preprocess import add_batch_dim as add_batch_dim
from ort_vision_sdk.preprocess import from_cv2 as from_cv2
from ort_vision_sdk.preprocess import letterbox as letterbox
from ort_vision_sdk.preprocess import normalize as normalize
from ort_vision_sdk.preprocess import reduction_factor as reduction_factor
from ort_vision_sdk.preprocess import resize as resize
from ort_vision_sdk.preprocess import to_chw as to_chw
from ort_vision_sdk.preprocess import to_cv2 as to_cv2
from ort_vision_sdk.preprocess import to_tensor as to_tensor
from ort_vision_sdk.results import Boxes as Boxes
from ort_vision_sdk.results import ClassificationResults as ClassificationResults
from ort_vision_sdk.results import DetectClassifyResults as DetectClassifyResults
from ort_vision_sdk.results import DetectionResults as DetectionResults
from ort_vision_sdk.results import Masks as Masks
from ort_vision_sdk.results import Probs as Probs
from ort_vision_sdk.results import SegmentationResults as SegmentationResults
from ort_vision_sdk.tasks import Classifier as Classifier
from ort_vision_sdk.tasks import DetectClassify as DetectClassify
from ort_vision_sdk.tasks import Detector as Detector
from ort_vision_sdk.tasks import DetectorHead as DetectorHead
from ort_vision_sdk.tasks import Segmenter as Segmenter
from ort_vision_sdk.tasks import SegmenterHead as SegmenterHead
from ort_vision_sdk.tasks import VisionTask as VisionTask
from ort_vision_sdk.tasks import require_detections as require_detections
from ort_vision_sdk.types import BoundingBox as BoundingBox
from ort_vision_sdk.types import ClassificationResult as ClassificationResult
from ort_vision_sdk.types import ClassProbability as ClassProbability
from ort_vision_sdk.types import DetectionResult as DetectionResult
from ort_vision_sdk.types import ImageArray as ImageArray
from ort_vision_sdk.types import SegmentationResult as SegmentationResult

__version__: str = "0.11.0"

__all__: list[str] = [
    "COCO_CLASSES",
    "CUSTOM_NORMALIZATION",
    "FUSION_KIND_DETECT_CLASSIFY",
    "FUSION_KIND_DETECT_SEGMENT_CLASSIFY",
    "IDENTITY_MEAN",
    "IDENTITY_STD",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "INPUT_IMAGE",
    "INPUT_PAD",
    "INPUT_SCALE",
    "INPUT_SOURCE",
    "METADATA_PREFIX",
    "NORMALIZATION_PRESETS",
    "OUTPUT_BOXES",
    "OUTPUT_CLASSES",
    "OUTPUT_MASKS",
    "OUTPUT_NUM_DETECTIONS",
    "OUTPUT_PROBS",
    "OUTPUT_SCORES",
    "STAGES",
    "BoundingBox",
    "Boxes",
    "ClassProbability",
    "ClassificationResult",
    "ClassificationResults",
    "Classifier",
    "CropSource",
    "DecodedSegmentation",
    "DetectClassify",
    "DetectClassifyResults",
    "DetectionResult",
    "DetectionResults",
    "Detector",
    "DetectorHead",
    "FusionError",
    "FusionSpec",
    "ImageArray",
    "ImageInput",
    "ImageLoadError",
    "InferenceBackend",
    "InferenceError",
    "LabelMapError",
    "LabelSpec",
    "Masks",
    "MetadataBackend",
    "ModelLoadError",
    "NoDetectionsError",
    "Normalization",
    "OrtSession",
    "OrtVisionError",
    "Probs",
    "ProviderNotAvailableError",
    "SegmentationResult",
    "SegmentationResults",
    "Segmenter",
    "SegmenterHead",
    "SpeedTimer",
    "Stage",
    "VisionTask",
    "__version__",
    "add_batch_dim",
    "as_float32",
    "available_providers",
    "batched_nms",
    "decode_yolo",
    "decode_yolo_anchors",
    "decode_yolo_seg",
    "default_labels",
    "from_cv2",
    "is_ultralytics_classifier",
    "letterbox",
    "load_image",
    "model_names",
    "nms",
    "normalize",
    "numpy_dtype_for",
    "parse_names",
    "read_metadata",
    "reduction_factor",
    "require_detections",
    "resize",
    "resolve_input_size",
    "resolve_labels",
    "resolve_normalization",
    "resolve_providers",
    "softmax",
    "spatial_input_size",
    "to_chw",
    "to_cv2",
    "to_tensor",
    "topk",
]
