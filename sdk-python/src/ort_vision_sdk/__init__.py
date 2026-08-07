"""ort-vision-sdk: high-level Python SDK for computer vision inference with ONNX Runtime."""

from ort_vision_sdk.core import (
    InferenceBackend,
    MetadataBackend,
    OrtSession,
    read_metadata,
)
from ort_vision_sdk.graph import model_names, resolve_input_size, spatial_input_size
from ort_vision_sdk.io import ImageInput, load_image
from ort_vision_sdk.labels import COCO_CLASSES, LabelSpec, default_labels, resolve_labels
from ort_vision_sdk.results import (
    Boxes,
    ClassificationResults,
    DetectionResults,
    Masks,
    Probs,
    SegmentationResults,
)
from ort_vision_sdk.tasks import (
    Classifier,
    Detector,
    DetectorHead,
    Segmenter,
    SegmenterHead,
    VisionTask,
)
from ort_vision_sdk.types import (
    BoundingBox,
    ClassificationResult,
    ClassProbability,
    DetectionResult,
    ImageArray,
    SegmentationResult,
)

__version__: str = "0.6.0"

__all__: list[str] = [
    "COCO_CLASSES",
    "BoundingBox",
    "Boxes",
    "ClassProbability",
    "ClassificationResult",
    "ClassificationResults",
    "Classifier",
    "DetectionResult",
    "DetectionResults",
    "Detector",
    "DetectorHead",
    "ImageArray",
    "ImageInput",
    "InferenceBackend",
    "LabelSpec",
    "Masks",
    "MetadataBackend",
    "OrtSession",
    "Probs",
    "SegmentationResult",
    "SegmentationResults",
    "Segmenter",
    "SegmenterHead",
    "VisionTask",
    "__version__",
    "default_labels",
    "load_image",
    "model_names",
    "read_metadata",
    "resolve_input_size",
    "resolve_labels",
    "spatial_input_size",
]
