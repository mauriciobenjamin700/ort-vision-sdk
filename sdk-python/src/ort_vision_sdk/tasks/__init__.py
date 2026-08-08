"""Task-oriented public API: classification, detection, segmentation, fused pipelines."""

from ort_vision_sdk.tasks.base import VisionTask, require_detections
from ort_vision_sdk.tasks.classifier import Classifier
from ort_vision_sdk.tasks.detector import Detector, DetectorHead
from ort_vision_sdk.tasks.pipeline import DetectClassify
from ort_vision_sdk.tasks.segmenter import Segmenter, SegmenterHead

__all__: list[str] = [
    "Classifier",
    "DetectClassify",
    "Detector",
    "DetectorHead",
    "Segmenter",
    "SegmenterHead",
    "VisionTask",
    "require_detections",
]
