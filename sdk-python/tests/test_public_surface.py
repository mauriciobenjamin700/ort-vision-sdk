"""The two SDKs must export the same public surface.

This repository ships one API in two languages, and the failure mode it exists
to prevent is a name that reaches one root and not the other. Each suite only
knows its own side, so nothing catches that — until a consumer ports code
between the SDKs and finds the symbol missing.

The check reads ``sdk-js-web/src/index.ts`` and pairs every export with the
Python name in :data:`ort_vision_sdk.__all__`. Anything legitimately one-sided
is listed in :data:`WEB_ONLY` or :data:`PYTHON_ONLY` **with its reason**, so the
asymmetry is a decision on the record rather than an oversight. Adding an export
to one SDK and not the other fails here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import ort_vision_sdk

WEB_INDEX = Path(__file__).resolve().parents[2] / "sdk-js-web" / "src" / "index.ts"

EQUIVALENTS: dict[str, str] = {
    "BoundingBox": "BoundingBox",
    "Boxes": "Boxes",
    "COCO_CLASSES": "COCO_CLASSES",
    "CUSTOM_NORMALIZATION": "CUSTOM_NORMALIZATION",
    "ClassProbability": "ClassProbability",
    "ClassificationResult": "ClassificationResult",
    "ClassificationResults": "ClassificationResults",
    "Classifier": "Classifier",
    "CropSource": "CropSource",
    "DecodedSegmentation": "DecodedSegmentation",
    "DetectClassify": "DetectClassify",
    "DetectClassifyResults": "DetectClassifyResults",
    "DetectionResult": "DetectionResult",
    "DetectionResults": "DetectionResults",
    "Detector": "Detector",
    "DetectorHead": "DetectorHead",
    "FUSION_KIND_DETECT_CLASSIFY": "FUSION_KIND_DETECT_CLASSIFY",
    "FusionError": "FusionError",
    "FusionSpec": "FusionSpec",
    "IDENTITY_MEAN": "IDENTITY_MEAN",
    "IDENTITY_STD": "IDENTITY_STD",
    "IMAGENET_MEAN": "IMAGENET_MEAN",
    "IMAGENET_STD": "IMAGENET_STD",
    "INPUT_IMAGE": "INPUT_IMAGE",
    "INPUT_PAD": "INPUT_PAD",
    "INPUT_SCALE": "INPUT_SCALE",
    "INPUT_SOURCE": "INPUT_SOURCE",
    "ImageInput": "ImageInput",
    "ImageLoadError": "ImageLoadError",
    "InferenceError": "InferenceError",
    "LabelMapError": "LabelMapError",
    "LabelSpec": "LabelSpec",
    "METADATA_PREFIX": "METADATA_PREFIX",
    "Masks": "Masks",
    "ModelLoadError": "ModelLoadError",
    "NoDetectionsError": "NoDetectionsError",
    "Normalization": "Normalization",
    "OUTPUT_BOXES": "OUTPUT_BOXES",
    "OUTPUT_CLASSES": "OUTPUT_CLASSES",
    "OUTPUT_NUM_DETECTIONS": "OUTPUT_NUM_DETECTIONS",
    "OUTPUT_PROBS": "OUTPUT_PROBS",
    "OUTPUT_SCORES": "OUTPUT_SCORES",
    "OrtSession": "OrtSession",
    "OrtVisionError": "OrtVisionError",
    "Probs": "Probs",
    "ProviderNotAvailableError": "ProviderNotAvailableError",
    "SegmentationResult": "SegmentationResult",
    "SegmentationResults": "SegmentationResults",
    "Segmenter": "Segmenter",
    "SegmenterHead": "SegmenterHead",
    "Speed": "Stage",
    "SpeedTimer": "SpeedTimer",
    "VERSION": "__version__",
    "VisionTask": "VisionTask",
    "batchedNms": "batched_nms",
    "decodeYolo": "decode_yolo",
    "decodeYoloAnchors": "decode_yolo_anchors",
    "decodeYoloSeg": "decode_yolo_seg",
    "defaultLabels": "default_labels",
    "detectProviders": "available_providers",
    "fromCv2": "from_cv2",
    "isUltralyticsClassifier": "is_ultralytics_classifier",
    "letterbox": "letterbox",
    "loadImage": "load_image",
    "modelNames": "model_names",
    "nms": "nms",
    "normalize": "normalize",
    "parseNames": "parse_names",
    "readModelMetadata": "read_metadata",
    "requireDetections": "require_detections",
    "resize": "resize",
    "resolveInputSize": "resolve_input_size",
    "resolveLabels": "resolve_labels",
    "resolveNormalization": "resolve_normalization",
    "resolveProviders": "resolve_providers",
    "softmax": "softmax",
    "spatialInputSize": "spatial_input_size",
    "toCHW": "to_chw",
    "toCv2": "to_cv2",
    "toTensor": "to_tensor",
    "topK": "topk",
}

WEB_ONLY: dict[str, str] = {
    "ClassifierOptions": "Named options object; Python takes keyword arguments.",
    "ClassifierPredictOptions": "Named options object; Python takes keyword arguments.",
    "DetectClassifyOptions": "Named options object; Python takes keyword arguments.",
    "DetectClassifyPredictOptions": "Named options object; Python takes keyword arguments.",
    "DetectorOptions": "Named options object; Python takes keyword arguments.",
    "DetectorPredictOptions": "Named options object; Python takes keyword arguments.",
    "SegmenterOptions": "Named options object; Python takes keyword arguments.",
    "SegmenterPredictOptions": "Named options object; Python takes keyword arguments.",
    "DecodeYoloAnchorsOptions": "Named options object; Python takes keyword arguments.",
    "DecodeYoloOptions": "Named options object; Python takes keyword arguments.",
    "DecodeYoloSegOptions": "Named options object; Python takes keyword arguments.",
    "ResolveInputSizeOptions": "Named options object; Python takes keyword arguments.",
    "ResolveLabelsOptions": "Named options object; Python takes keyword arguments.",
    "OrtSessionOptions": "Named options object; Python takes keyword arguments.",
    "DecodedAnchors": "TS return-shape alias; Python returns a documented tuple.",
    "DecodedDetection": "TS return-shape alias; Python returns a documented tuple.",
    "ModelSource": "Union of browser model sources; Python takes a path or bytes.",
    "DEFAULT_PROVIDERS": "ORT web needs a static provider list; Python asks the runtime.",
    "DeclaredDim": "TypeScript shape aliases; Python annotates with int | None inline.",
    "DeclaredShape": "TypeScript shape aliases; Python annotates with tuple inline.",
    "FusedLetterboxResult": "Reusable-buffer pipeline, browser-only.",
    "FusedResizeResult": "Reusable-buffer pipeline, browser-only.",
    "LetterboxPipeline": "Reusable-buffer pipeline, browser-only; NumPy needs no buffer reuse.",
    "LetterboxResult": "TS return-shape alias; Python returns a documented tuple.",
    "Mask": "Wraps ImageData; Python masks are ndarray.",
    "RGBImage": "Wraps browser pixel buffers; Python uses the ImageArray ndarray.",
    "ResizePipeline": "Reusable-buffer pipeline, browser-only; NumPy needs no buffer reuse.",
    "ResolvedNormalization": "TS return-shape alias; Python returns a documented tuple.",
    "TopKResult": "TS return-shape alias; Python returns a documented tuple.",
    "classificationNumClasses": "Reads class count off a declared shape; Python reads it "
    "from the session inside the task.",
    "declaredShapesFrom": "ORT web metadata adapter; Python reads session.get_inputs().",
    "detectionNumClasses": "Reads class count off a declared shape; Python reads it "
    "from the session inside the task.",
    "letterboxToTensorData": "Reusable-buffer pipeline, browser-only.",
    "readFusionSpec": "Python spells it FusionSpec.from_metadata, a classmethod.",
    "resizeToTensorData": "Reusable-buffer pipeline, browser-only.",
    "toFloat32": "Typed-array conversion; NumPy has astype.",
    "toFloat32Tensor": "Typed-array conversion; NumPy has astype.",
    "writePlanarFloat32": "Reusable-buffer pipeline, browser-only.",
    "zeroTensorData": "Reusable-buffer pipeline, browser-only.",
}

PYTHON_ONLY: dict[str, str] = {
    "DecodedSegmentation": "Exported on both sides; listed here because the web export "
    "is a type-only alias the parser reports under the same name.",
    "ImageArray": "The NumPy canonical image type; the web equivalent is RGBImage.",
    "InferenceBackend": "Pluggable backend protocol; the web SDK has one runtime.",
    "MetadataBackend": "Pluggable backend protocol; the web SDK has one runtime.",
    "NORMALIZATION_PRESETS": "Preset table exported for introspection; the web SDK keeps "
    "it module-private.",
    "STAGES": "Stage-name tuple; the web SDK expresses it as the Speed interface keys.",
    "Stage": "Literal of stage names; paired with the web Speed interface.",
    "add_batch_dim": "NumPy layout helper; the browser builds the batch axis in the "
    "pipeline buffer.",
    "reduction_factor": "Box-reduction step of the NumPy resize; the browser resamples "
    "through canvas.",
}

_EXPORT_BLOCK = re.compile(r"export\s*\{(.*?)\}\s*from", re.DOTALL)
_EXPORT_CONST = re.compile(r"export\s+const\s+([A-Za-z_][A-Za-z0-9_]*)")


def _web_exports() -> set[str]:
    """Parse the names the web SDK exports from its package root.

    Returns:
        set[str]: Every identifier re-exported by ``sdk-js-web/src/index.ts``,
        with the ``type`` modifier stripped.
    """
    source = WEB_INDEX.read_text(encoding="utf-8")
    names: set[str] = set(_EXPORT_CONST.findall(source))
    for block in _EXPORT_BLOCK.findall(source):
        for entry in block.split(","):
            cleaned = entry.strip().removeprefix("type ").strip()
            if cleaned:
                names.add(cleaned)
    return names


pytestmark = pytest.mark.skipif(
    not WEB_INDEX.exists(), reason="web SDK sources are not present in this checkout"
)


def test_every_web_export_has_a_python_counterpart() -> None:
    """Fail when the web SDK exports a name the Python root does not."""
    unpaired = sorted(name for name in _web_exports() if name not in EQUIVALENTS | WEB_ONLY)
    assert unpaired == [], (
        "web exports with no Python counterpart: "
        f"{unpaired}. Export them from ort_vision_sdk, or record the reason in WEB_ONLY."
    )


def test_every_python_export_has_a_web_counterpart() -> None:
    """Fail when the Python root exports a name the web SDK does not."""
    paired = set(EQUIVALENTS.values())
    unpaired = sorted(
        name for name in ort_vision_sdk.__all__ if name not in paired | PYTHON_ONLY.keys()
    )
    assert unpaired == [], (
        "Python exports with no web counterpart: "
        f"{unpaired}. Export them from the web SDK, or record the reason in PYTHON_ONLY."
    )


def test_every_mapped_python_name_is_actually_exported() -> None:
    """Fail when the equivalence table names a Python symbol that is not public."""
    exported = set(ort_vision_sdk.__all__)
    broken = sorted(
        f"{web} -> {python}" for web, python in EQUIVALENTS.items() if python not in exported
    )
    assert broken == [], f"EQUIVALENTS points at names missing from __all__: {broken}"
