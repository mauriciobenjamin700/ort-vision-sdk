"""ort-vision-sdk: high-level Python SDK for computer vision inference with ONNX Runtime."""

from ort_vision_sdk.io import ImageInput, load_image
from ort_vision_sdk.labels import COCO_CLASSES, LabelSpec, resolve_labels
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

__version__: str = "0.3.2"

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
    "LabelSpec",
    "Masks",
    "Probs",
    "SegmentationResult",
    "SegmentationResults",
    "Segmenter",
    "SegmenterHead",
    "VisionTask",
    "__version__",
    "load_image",
    "resolve_labels",
]
