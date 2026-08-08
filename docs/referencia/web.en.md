# Reference — Web API

Public surface of the `@mauriciobenjamin700/ort-vision-sdk-web` package
(everything is exported from `@mauriciobenjamin700/ort-vision-sdk-web`).

## Tasks

| Class | Creation | Description |
| --- | --- | --- |
| `Classifier` | `await Classifier.create(model, options)` | Image classification. |
| `Detector` | `await Detector.create(model, options)` | Object detection (YOLO). |
| `Segmenter` | `await Segmenter.create(model, options)` | Instance segmentation (YOLO-seg). |
| `DetectClassify` | `await DetectClassify.create(model, options)` | Fused detector → classifier pipeline, in a single `.onnx`. |
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
| `DetectClassifyOptions` / `DetectClassifyPredictOptions` | `DetectClassify` (`labels`, `classifierLabels`; `confThreshold`, `classes`, `topK` on predict) |

All three detection types additionally take `raiseOnEmpty` (construction and
predict) — see [Empty results](#empty-results).

## Results

| Envelope | Bulk view | Iterating yields |
| --- | --- | --- |
| `ClassificationResults` | `probs` | n/a (single result) |
| `DetectionResults` | `boxes` | `DetectionResult` |
| `DetectClassifyResults` | `boxes` | `DetectionResult` with `classification` filled in (+ `classifierNames` on the envelope) |
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
| `OrtSession.inputShape` / `.inputShapes` | Shapes the graph declares, dynamic axes as `null`. |
| `OrtSession.release()` | Frees the native session (needed when discarding a session while the page lives on). |
| `task.inputSize` | The resolution the task actually preprocesses to. |
| `task.warmup(runs?)` | Runs the model on a zero-filled tensor to pay shader compilation up front. |
| `spatialInputSize` / `resolveInputSize` / `declaredShapesFrom` | Pure helpers behind the graph → caller → fallback precedence. |
| `DeclaredShape` / `DeclaredDim` | A declared shape and one dimension (`number`, or `null` when symbolic). |

## Errors

Exported exception hierarchy: `OrtVisionError` (base), `ImageLoadError`,
`InferenceError`, `LabelMapError`, `ModelLoadError`,
`ProviderNotAvailableError`, `FusionError`, `NoDetectionsError`.

## Empty results

`Detector`, `Segmenter` and `DetectClassify` take `raiseOnEmpty` in both their
construction and `predict()` options. Default `false`: finding nothing returns an
empty envelope. With `true`, it throws `NoDetectionsError` — see
[When finding nothing is an error](../guia/deteccao.md#when-finding-nothing-is-an-error).

| Symbol | Description |
| --- | --- |
| `raiseOnEmpty` | Construction and `predict()` option; the per-call value wins. |
| `NoDetectionsError` | Thrown when nothing survives and the flag is in effect. |
| `requireDetections(count, options)` | The helper the three tasks share, exported for anyone building their own task. |

## Fused pipelines

| Symbol | Description |
| --- | --- |
| `readFusionSpec(metadata)` | Reads what a fused pipeline declares about itself; `null` when the model is not a pipeline. |
| `FusionSpec` / `CropSource` | The decoded contract, and where the crops come from. |
| `INPUT_IMAGE` / `INPUT_SOURCE` / `INPUT_SCALE` / `INPUT_PAD` | Names of the fused graph's inputs. |
| `OUTPUT_BOXES` / `OUTPUT_SCORES` / `OUTPUT_CLASSES` / `OUTPUT_NUM_DETECTIONS` / `OUTPUT_PROBS` | Names of its outputs. |
| `METADATA_PREFIX` / `FUSION_KIND_DETECT_CLASSIFY` | The `ovs.` namespace and the pipeline family. |
| `parseNames(raw)` | Parses a `repr`-encoded class map. |

Fusing models is a **Python-side** build step (the `[compose]` extra); the
browser only loads the resulting `.onnx`. See
[Fused pipelines](../guia/pipeline.md).

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
