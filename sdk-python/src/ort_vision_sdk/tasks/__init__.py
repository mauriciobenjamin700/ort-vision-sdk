"""Task-oriented public API: classification, detection, segmentation."""

from ort_vision_sdk.tasks.base import VisionTask
from ort_vision_sdk.tasks.classifier import Classifier
from ort_vision_sdk.tasks.detector import Detector, DetectorHead
from ort_vision_sdk.tasks.segmenter import Segmenter, SegmenterHead

__all__: list[str] = [
    "Classifier",
    "Detector",
    "DetectorHead",
    "Segmenter",
    "SegmenterHead",
    "VisionTask",
]
