# Reference — Web API

Public surface of the `@mauriciobenjamin700/ort-vision-sdk-web` package
(everything is exported from `@mauriciobenjamin700/ort-vision-sdk-web`).

## Tasks

| Class | Creation | Description |
| --- | --- | --- |
| `Classifier` | `await Classifier.create(model, options)` | Image classification. |
| `Detector` | `await Detector.create(model, options)` | Object detection (YOLO). |
| `Segmenter` | `await Segmenter.create(model, options)` | Instance segmentation (YOLO-seg). |
| `VisionTask` | — | Common base class. |

`predict()` is always `async` and returns `Promise<...Results[]>` of length 1
per image. Each task exposes a `run()` alias.

### Option types

| Type | For |
| --- | --- |
| `ClassifierOptions` / `ClassifierPredictOptions` | `Classifier` construction / `predict` (`labels`, `numClasses`, `inputSize`, `applySoftmax`, `providers`; `topK` on predict) |
| `DetectorOptions` / `DetectorPredictOptions` | `Detector` (`head`, `labels`, `inputSize`, `confThreshold`, `iouThreshold`; overrides + `classes` on predict) |
| `SegmenterOptions` / `SegmenterPredictOptions` | `Segmenter` (+ `maskThreshold`) |
| `DetectorHead` (`"yolo"`) / `SegmenterHead` (`"yolo-seg"`) | decoder families |

## Results

| Envelope | Bulk view | Iterating yields |
| --- | --- | --- |
| `ClassificationResults` | `probs` | n/a (single result) |
| `DetectionResults` | `boxes` | `DetectionResult` |
| `SegmentationResults` | `boxes`, `masks` | `SegmentationResult` |

Every envelope exposes `names`, `origImg`, `origShape`, `path` and `speed` —
a `Speed` object holding `load`, `preprocess`, `inference` and `postprocess`
in milliseconds, filled in by every `predict()`. See
[Inference cost](../guia/velocidade.md).

Bulk views: `Boxes`, `Probs`, `Masks` (same attributes as Python).

Per-instance types/classes: `DetectionResult`, `SegmentationResult`,
`ClassificationResult`, `ClassProbability` (with `classId`/`className`/
`confidence` and the aliases `cls`/`name`/`conf`/`box`), plus `BoundingBox`
(`asXyxy()`, `asXywh()`), `Mask` (`data`/`width`/`height`) and `RGBImage`.

## Images, labels and providers

| Symbol | Description |
| --- | --- |
| `loadImage(image)` | Loads any supported input into an `RGBImage`. |
| `ImageInput` | Union type of the inputs accepted by `predict()`. |
| `resolveLabels(spec, options)` | Resolves a `LabelSpec` into the class mapping. |
| `LabelSpec` / `ResolveLabelsOptions` | Label-resolution types. |
| `COCO_CLASSES` | The 80 classes of the COCO preset. |
| `DEFAULT_PROVIDERS` | `["webgpu", "wasm"]`. |
| `resolveProviders(...)` | Resolves the provider list into ORT-Web names. |
| `OrtSession` / `OrtSessionOptions` / `ModelSource` | Low-level session. |

## Errors

Exported exception hierarchy: `OrtVisionError` (base), `ImageLoadError`,
`InferenceError`, `LabelMapError`, `ModelLoadError`,
`ProviderNotAvailableError`.

## Pre/post-processing utilities

The package also exports low-level helpers for callers building their own
pipeline: `letterbox`, `resize`, `normalize`, `toCHW`, `toTensor`,
`toFloat32`/`toFloat32Tensor`, `fromCv2`/`toCv2`, `softmax`, `topK`, `nms`,
`batchedNms`, `decodeYolo`/`decodeYoloV8`, `decodeYoloAnchors`/
`decodeYoloV8Anchors`, `decodeYoloSeg`/`decodeYoloV8Seg`.

!!! note "Source of truth"
    The full signatures live in the source at
    [`sdk-js-web/src/`](https://github.com/mauriciobenjamin700/ort-vision-sdk/tree/main/sdk-js-web/src).
    This page summarizes the public surface exported in `index.ts`.
