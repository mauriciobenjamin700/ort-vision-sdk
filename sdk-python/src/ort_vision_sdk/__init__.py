"""ort-vision-sdk: high-level Python SDK for computer vision inference with ONNX Runtime."""

from ort_vision_sdk.core import (
    InferenceBackend,
    MetadataBackend,
    OrtSession,
    read_metadata,
)
from ort_vision_sdk.fusion import CropSource, FusionSpec
from ort_vision_sdk.graph import model_names, parse_names, resolve_input_size, spatial_input_size
from ort_vision_sdk.io import ImageInput, load_image
from ort_vision_sdk.labels import COCO_CLASSES, LabelSpec, default_labels, resolve_labels
from ort_vision_sdk.normalization import (
    IDENTITY_MEAN,
    IDENTITY_STD,
    IMAGENET_MEAN,
    IMAGENET_STD,
    Normalization,
    is_ultralytics_classifier,
    resolve_normalization,
)
from ort_vision_sdk.results import (
    Boxes,
    ClassificationResults,
    DetectClassifyResults,
    DetectionResults,
    Masks,
    Probs,
    SegmentationResults,
)
from ort_vision_sdk.tasks import (
    Classifier,
    DetectClassify,
    Detector,
    DetectorHead,
    Segmenter,
    SegmenterHead,
    VisionTask,
    require_detections,
)
from ort_vision_sdk.types import (
    BoundingBox,
    ClassificationResult,
    ClassProbability,
    DetectionResult,
    ImageArray,
    SegmentationResult,
)

__version__: str = "0.8.0"

__all__: list[str] = [
    "COCO_CLASSES",
    "IDENTITY_MEAN",
    "IDENTITY_STD",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "BoundingBox",
    "Boxes",
    "ClassProbability",
    "ClassificationResult",
    "ClassificationResults",
    "Classifier",
    "CropSource",
    "DetectClassify",
    "DetectClassifyResults",
    "DetectionResult",
    "DetectionResults",
    "Detector",
    "DetectorHead",
    "FusionSpec",
    "ImageArray",
    "ImageInput",
    "InferenceBackend",
    "LabelSpec",
    "Masks",
    "MetadataBackend",
    "Normalization",
    "OrtSession",
    "Probs",
    "SegmentationResult",
    "SegmentationResults",
    "Segmenter",
    "SegmenterHead",
    "VisionTask",
    "__version__",
    "default_labels",
    "is_ultralytics_classifier",
    "load_image",
    "model_names",
    "parse_names",
    "read_metadata",
    "require_detections",
    "resolve_input_size",
    "resolve_labels",
    "resolve_normalization",
    "spatial_input_size",
]
