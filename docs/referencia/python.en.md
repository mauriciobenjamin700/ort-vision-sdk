# Reference — Python API

Public surface of the `ort-vision-sdk` package (everything is importable
directly from `ort_vision_sdk`).

## Tasks

| Class | Description |
| --- | --- |
| `Classifier` | Image classification (output `(1, num_classes)`). |
| `Detector` | Object detection (anchor-free YOLO heads). |
| `Segmenter` | Instance segmentation (YOLO-seg heads). |
| `DetectClassify` | Fused detector → classifier pipeline, in a single `.onnx`. |
| `VisionTask` | Common base class (do not instantiate directly). |
| `DetectorHead` | Type of the detection decoder families (e.g. `"yolo"`). |
| `SegmenterHead` | Type of the segmentation decoder families (e.g. `"yolo-seg"`). |

Each task exposes three inference variants with the **same signature**:
`predict()`, `async_predict()` (`asyncio.to_thread`) and `ort_async_predict()`
(`InferenceSession.run_async`). All return `list[Results]` of length 1 per image.

### Constructors (summary)

```python
Classifier(model_path, *, labels=None, providers=None, session_options=None,
           backend=None, input_size=None, mean=..., std=..., apply_softmax=True)

Detector(model_path, *, head="yolo", labels=None, providers=None,
         session_options=None, backend=None, input_size=None,
         conf_threshold=0.25, iou_threshold=0.45, max_detections=300,
         raise_on_empty=False)

Segmenter(model_path, *, head="yolo-seg", labels=None, providers=None,
          session_options=None, backend=None, input_size=None,
          conf_threshold=0.25, iou_threshold=0.45, max_detections=300,
          mask_threshold=0.5, raise_on_empty=False)

DetectClassify(model_path, *, labels=None, classifier_labels=None,
               raise_on_empty=False, providers=None, session_options=None,
               backend=None)
```

`DetectClassify` takes only these parameters because everything else — the
resolution, the crop size, the thresholds, the softmax, the class names of both
stages — was written into the file at fusion time. See
[Fused pipelines](../guia/pipeline.md).

All three constructors accept `backend=` (v0.4.0): inject an `InferenceBackend`
to run inference outside the in-process ONNX Runtime (browser, Android). When
given, `model_path`/`providers`/`session_options` are ignored. See the
[backends guide](../guia/backends.en.md).

`input_size=None` and `labels=None` (v0.6.0) mean "ask the model": the
resolution comes from the shape the graph declares and the names come from the
`names` metadata, with the previous defaults (224/640, COCO preset) as
fallbacks. See [The model decides](../guia/modelo.en.md).

`Detector.predict()` and `Segmenter.predict()` accept per-call overrides:
`conf_threshold`, `iou_threshold`, `classes`.

## Inference backends

| Symbol | Description |
| --- | --- |
| `InferenceBackend` | Inference-engine protocol — metadata (`input_name`/`input_shape`/`output_names`/`output_shapes`) + `run`/`async_run`/`ort_async_run`. |
| `MetadataBackend` | Capability protocol: backends that can read the model's metadata map. Separate because a native bridge may not be able to. |
| `read_metadata(backend)` | Reads a backend's metadata map, returning `{}` when it does not offer the capability. |
| `OrtSession` | Default backend (in-process ONNX Runtime); satisfies both protocols. |
| `OrtSession.metadata` | The model's custom metadata map (`names`, `task`, `imgsz`, ...). |
| `OrtSession.input_shape` | Declared shape of the first input (dynamic axes as strings). |
| `OrtSession.providers` | Providers ORT **registered** — the answer to "where is this running". |
| `OrtSession.requested_providers` | Providers that were **asked for**, after alias expansion and auto-selection. |

## Result envelopes

| Envelope | Bulk view | Iterating yields | Notable fields |
| --- | --- | --- | --- |
| `ClassificationResults` | `probs` | n/a (single result) | `cls`, `conf`, `name`, `probabilities` |
| `DetectionResults` | `boxes` | `DetectionResult` | `cls`, `conf`, `box.xyxy`, `cropped_image` |
| `DetectClassifyResults` | `boxes` | `DetectionResult` | + `classification`, plus `classifier_names` on the envelope |
| `SegmentationResults` | `boxes`, `masks` | `SegmentationResult` | `cls`, `conf`, `box.xyxy`, `mask`, `segmented_image` |

Every envelope also exposes `names`, `orig_img`, `orig_shape`, `path`, and
`speed` — a `dict[str, float]` holding `load`, `preprocess`, `inference` and
`postprocess` in milliseconds, filled in by every `predict()`. See
[Inference cost](../guia/velocidade.md).

## Bulk views (Ultralytics-style)

| Class | Attributes |
| --- | --- |
| `Boxes` | `xyxy`, `xywh`, `xyxyn`, `xywhn`, `cls`, `conf`, `data` |
| `Probs` | `top1`, `top5`, `top1conf`, `top5conf`, `data` |
| `Masks` | `data`, `xyxy` |

## Per-instance types

| Type | Canonical fields | Ultralytics aliases |
| --- | --- | --- |
| `DetectionResult` | `class_id`, `class_name`, `confidence`, `bbox`, `cropped_image`, `classification` | `cls`, `name`, `conf`, `box` |
| `SegmentationResult` | + `mask`, `segmented_image` | `cls`, `name`, `conf`, `box` |
| `ClassificationResult` | `class_id`, `class_name`, `confidence` | `cls`, `name`, `conf` |
| `ClassProbability` | `class_id`, `class_name`, `probability` | `cls`, `name` |
| `BoundingBox` | `x1`, `y1`, `x2`, `y2` + `xyxy` | — |

## Empty results

`Detector`, `Segmenter` and `DetectClassify` take `raise_on_empty` on the
constructor and as a per-call override on `predict()`. Default `False`: finding
nothing returns an empty envelope, not an error. With `True`, it raises
`NoDetectionsError` — see
[When finding nothing is an error](../guia/deteccao.md#when-finding-nothing-is-an-error).

| Symbol | Description |
| --- | --- |
| `raise_on_empty` | Constructor and `predict()` argument; the per-call value wins. |
| `NoDetectionsError` | Raised when nothing survives and the flag is in effect. Exported from `ort_vision_sdk.core`. |
| `require_detections(count, ...)` | The helper the three tasks share, exported for anyone building their own task. |

## Composing pipelines (`[compose]` extra)

| Symbol | Description |
| --- | --- |
| `compose.fuse_detect_classify(...)` | Fuses a YOLO detector and a classifier into one `.onnx`, and validates the result by running it. |
| `compose.build_bridge(...)` | Builds just the bridge subgraph (NMS → RoiAlign → normalization). Useful for inspection. |
| `compose.MIN_OPSET` | Lowest opset the bridge requires (16, because of `RoiAlign`). |
| `compose.Normalization` | `"auto"`, `"imagenet"`, `"ultralytics"` or `"none"` — which preprocessing the classifier expects. |
| `FusionError` | Raised when two models cannot be fused, or a loaded file is not a pipeline. |

This module is the only one that imports `onnx`, and it ships only with
`pip install "ort-vision-sdk[compose]"`. Running the fused model does not need
it. See [Fused pipelines](../guia/pipeline.md).

## Images and labels

| Symbol | Description |
| --- | --- |
| `load_image(image)` | Loads any supported input into an HWC uint8 RGB `ndarray`. |
| `ImageInput` | Union type of the inputs accepted by `predict()`. |
| `ImageArray` | Alias for the HWC uint8 RGB `ndarray`. |
| `resolve_labels(spec, ...)` | Resolves a `LabelSpec` into `dict[int, str]`. |
| `LabelSpec` | Union type accepted by `labels=` (preset, list, dict, path, None). |
| `COCO_CLASSES` | Tuple with the 80 classes of the COCO preset. |

## What the model declares

| Symbol | Description |
| --- | --- |
| `spatial_input_size(shape)` | Pulls `(width, height)` out of a static NCHW shape; `None` when the axes are dynamic. |
| `resolve_input_size(...)` | Applies the graph → caller → fallback precedence, warning when the caller contradicts a static graph. |
| `model_names(metadata)` | Parses Ultralytics' `names` (the `repr` of a `dict[int, str]`) with `ast.literal_eval`; `None` when absent or unusable. |
| `parse_names(raw)` | The same parser over any string — used for both class maps of a fused pipeline. |
| `FusionSpec` | What a fused pipeline declares about itself; `FusionSpec.from_metadata(...)` reads it back. |
| `CropSource` | `"detector_input"` or `"original"` — where the bridge crops the boxes from. |
| `task.input_size` | The resolution the task actually preprocesses to. |

!!! note "Source of truth"
    The full signatures, with types and docstrings, live in the source at
    [`sdk-python/src/ort_vision_sdk/`](https://github.com/mauriciobenjamin700/ort-vision-sdk/tree/main/sdk-python/src/ort_vision_sdk).
    This page summarizes the public surface exported in `__init__.py`.
