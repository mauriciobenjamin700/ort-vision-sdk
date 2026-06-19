# Reference — Python API

Public surface of the `ort-vision-sdk` package (everything is importable
directly from `ort_vision_sdk`).

## Tasks

| Class | Description |
| --- | --- |
| `Classifier` | Image classification (output `(1, num_classes)`). |
| `Detector` | Object detection (anchor-free YOLO heads). |
| `Segmenter` | Instance segmentation (YOLO-seg heads). |
| `VisionTask` | Common base class (do not instantiate directly). |
| `DetectorHead` | Type of the detection decoder families (e.g. `"yolo"`). |
| `SegmenterHead` | Type of the segmentation decoder families (e.g. `"yolo-seg"`). |

Each task exposes three inference variants with the **same signature**:
`predict()`, `async_predict()` (`asyncio.to_thread`) and `ort_async_predict()`
(`InferenceSession.run_async`). All return `list[Results]` of length 1 per image.

### Constructors (summary)

```python
Classifier(model_path, *, labels=None, providers=None, session_options=None,
           backend=None, input_size=(224, 224), mean=..., std=..., apply_softmax=True)

Detector(model_path, *, head="yolo", labels="coco", providers=None,
         session_options=None, backend=None, input_size=(640, 640),
         conf_threshold=0.25, iou_threshold=0.45, max_detections=300)

Segmenter(model_path, *, head="yolo-seg", labels="coco", providers=None,
          session_options=None, backend=None, input_size=(640, 640),
          conf_threshold=0.25, iou_threshold=0.45, max_detections=300,
          mask_threshold=0.5)
```

All three constructors accept `backend=` (v0.4.0): inject an `InferenceBackend`
to run inference outside the in-process ONNX Runtime (browser, Android). When
given, `model_path`/`providers`/`session_options` are ignored. See the
[backends guide](../guia/backends.en.md).

`Detector.predict()` and `Segmenter.predict()` accept per-call overrides:
`conf_threshold`, `iou_threshold`, `classes`.

## Inference backends

| Symbol | Description |
| --- | --- |
| `InferenceBackend` | Inference-engine protocol — metadata (`input_name`/`input_shape`/`output_names`/`output_shapes`) + `run`/`async_run`/`ort_async_run`. |
| `OrtSession` | Default backend (in-process ONNX Runtime); satisfies the protocol. |

## Result envelopes

| Envelope | Bulk view | Iterating yields | Notable fields |
| --- | --- | --- | --- |
| `ClassificationResults` | `probs` | n/a (single result) | `cls`, `conf`, `name`, `probabilities` |
| `DetectionResults` | `boxes` | `DetectionResult` | `cls`, `conf`, `box.xyxy`, `cropped_image` |
| `SegmentationResults` | `boxes`, `masks` | `SegmentationResult` | `cls`, `conf`, `box.xyxy`, `mask`, `segmented_image` |

Every envelope also exposes `names`, `orig_img`, `orig_shape`, `path`, and an
optional `speed` timings dict.

## Bulk views (Ultralytics-style)

| Class | Attributes |
| --- | --- |
| `Boxes` | `xyxy`, `xywh`, `xyxyn`, `xywhn`, `cls`, `conf`, `data` |
| `Probs` | `top1`, `top5`, `top1conf`, `top5conf`, `data` |
| `Masks` | `data`, `xyxy` |

## Per-instance types

| Type | Canonical fields | Ultralytics aliases |
| --- | --- | --- |
| `DetectionResult` | `class_id`, `class_name`, `confidence`, `bbox`, `cropped_image` | `cls`, `name`, `conf`, `box` |
| `SegmentationResult` | + `mask`, `segmented_image` | `cls`, `name`, `conf`, `box` |
| `ClassificationResult` | `class_id`, `class_name`, `confidence` | `cls`, `name`, `conf` |
| `ClassProbability` | `class_id`, `class_name`, `probability` | `cls`, `name` |
| `BoundingBox` | `x1`, `y1`, `x2`, `y2` + `xyxy` | — |

## Images and labels

| Symbol | Description |
| --- | --- |
| `load_image(image)` | Loads any supported input into an HWC uint8 RGB `ndarray`. |
| `ImageInput` | Union type of the inputs accepted by `predict()`. |
| `ImageArray` | Alias for the HWC uint8 RGB `ndarray`. |
| `resolve_labels(spec, ...)` | Resolves a `LabelSpec` into `dict[int, str]`. |
| `LabelSpec` | Union type accepted by `labels=` (preset, list, dict, path, None). |
| `COCO_CLASSES` | Tuple with the 80 classes of the COCO preset. |

!!! note "Source of truth"
    The full signatures, with types and docstrings, live in the source at
    [`sdk-python/src/ort_vision_sdk/`](https://github.com/mauriciobenjamin700/ort-vision-sdk/tree/main/sdk-python/src/ort_vision_sdk).
    This page summarizes the public surface exported in `__init__.py`.
