# Changelog

All notable changes to `@mauriciobenjamin700/ort-vision-sdk-web` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.2] - 2026-05-31

### Changed

- README now links to the bilingual (PT-BR + EN-US) documentation site on
  GitHub Pages (<https://mauriciobenjamin700.github.io/ort-vision-sdk/>) so the
  npm package page points readers to the full guide and API reference.
  Documentation-only release; no public API changes.

## [0.2.1] - 2026-05-03

First **published** release on npm. The previous `0.2.0` tag never produced a
package because the CI smoke-install step still referenced the old
`@ort-vision-sdk/web` scope after the rename to
`@mauriciobenjamin700/ort-vision-sdk-web`. No public API changes.

### Changed

- README rewritten with badges, GitHub repo links, npm install snippets and
  imports updated to the new `@mauriciobenjamin700/ort-vision-sdk-web` scope.

### Fixed

- CI smoke-install step now captures the packed tarball name dynamically and
  resolves the package name via `package.json`, so it survives future
  package-name changes.

## [0.2.0] - 2026-05-02

This release brings the public API in line with the Ultralytics / PyTorch
idiom, mirroring the Python `ort-vision-sdk` 0.2.0 changes.

### Added

- **Per-image `Results` envelopes** (`DetectionResults`, `ClassificationResults`,
  `SegmentationResults`) returned by every `predict()` call as a 1-element
  array, mirroring `YOLO("img.jpg")`. Each envelope holds:
  - `boxes` / `probs` / `masks` — bulk typed-array views with Ultralytics-style
    accessors (`xyxy`, `xywh`, `xyxyn`, `xywhn`, `cls`, `conf`, `data`,
    `top1`, `top1conf`, `top5`, `top5conf`).
  - `detections` — array of per-instance objects (the previous return type).
  - `names` — `Record<number, string>` matching `model.names`.
  - `origImg`, `origShape`, `path`, `speed`.
  - `[Symbol.iterator]` — `for (const d of results[0])` works directly.
- **Ultralytics-style aliases** on `BoundingBox`, `ClassProbability`,
  `ClassificationResult`, `DetectionResult`, `SegmentationResult`:
  `cls`, `conf`, `name`, `box`, `xyxy`, `xywh` (center coords),
  plus `xyxyn(origShape)` / `xywhn(origShape)` methods on `BoundingBox`.
- **`call(image)` method** on every task class — delegates to `predict()`,
  for parity with PyTorch `nn.Module.__call__`.
- **`names: Record<number, string>` getter** on every task class alongside
  `labels: readonly string[]`.
- **`batchedNms(boxes, scores, idxs, iouThreshold)`** matching
  `torchvision.ops.batched_nms`.
- **Generic YOLO decoder names**: `decodeYolo`, `decodeYoloAnchors`,
  `decodeYoloSeg` — same code, accurate name (covers v8/v9/v10/v11/v12).
- **Preprocess helpers** mirroring torchvision/OpenCV:
  - `toTensor(image)` → CHW `Float32Array / 255` (ToTensor parity).
  - `fromCv2(bgr, w, h)` / `toCv2(image)` — channel-order bridges.
- **Explicit `head` option** on `Detector.create()` and `Segmenter.create()`
  — caller declares which decoder family the model uses (`"yolo"` for
  detect heads of v8/v9/v10/v11/v12/v26; `"yolo-seg"` for the matching seg
  heads). The SDK does **not** auto-detect — passing an unsupported head
  throws. New types `DetectorHead`, `SegmenterHead`.
- **`classes` filter** on `Detector.predict()`/`call()` and
  `Segmenter.predict()`/`call()` predict-options — keeps only results
  whose `classId` is in the supplied array, matching Ultralytics'
  `model.predict(img, classes=[0, 16])`.

### Changed

- `Detector.predict(img)` now returns `Promise<DetectionResults[]>` (1
  element) instead of `Promise<DetectionResult[]>`. Iterate the envelope
  to recover the old shape: `for (const d of (await det.predict(img))[0]) ...`.
- `Classifier.predict(img)` now returns `Promise<ClassificationResults[]>`.
- `Segmenter.predict(img)` now returns `Promise<SegmentationResults[]>`.
- The detection decoder is funnelled through `batchedNms` instead of an
  inline per-class loop.

### Deprecated

- `decodeYoloV8`, `decodeYoloV8Anchors`, `decodeYoloV8Seg` — log a one-time
  `console.warn` and delegate to the generic `decodeYolo*` versions.
  Will be removed in 0.3.0. The corresponding option types
  (`DecodeYoloV8Options`, `DecodeYoloV8AnchorsOptions`,
  `DecodeYoloV8SegOptions`) are now type aliases of the new names.

### Migration

```typescript
// Before (0.1.0)
const detections = await det.predict("street.jpg");
for (const d of detections) {
  console.log(d.classId, d.className, d.confidence, d.bbox.asXyxy());
}

// After (0.2.0)
const results = await det.predict("street.jpg");  // length 1
const r = results[0];

// Per-instance entries (legacy fields still present):
for (const d of r) {
  console.log(d.classId, d.className, d.confidence, d.bbox.asXyxy());
}
// or with the new short aliases:
for (const d of r) {
  console.log(d.cls, d.name, d.conf, d.box.xyxy);
}

// Bulk typed-array access (matches Ultralytics):
console.log(r.boxes.xyxy, r.boxes.cls, r.boxes.conf, r.names);
```

## [0.1.0] - 2026-05-02

### Added

- Initial alpha release.
- `Classifier` and `Detector` task classes wrapping `onnxruntime-web` sessions.
- Browser-friendly image loading (`loadImage`) accepting URLs, `Blob`, `File`, `HTMLImageElement`, `HTMLCanvasElement`, `OffscreenCanvas`, `ImageBitmap`, `ImageData`, and `RGBImage`.
- Preprocessing helpers: `letterbox`, `resize`, `normalize`, `toCHW`, `toFloat32`, `toFloat32Tensor`.
- Postprocessing helpers: `softmax`, `topK`, `decodeYoloV8`, `nms`.
- Execution-provider resolution defaulting to `["webgpu", "wasm"]`.
- Public types mirroring the Python SDK: `BoundingBox`, `ClassProbability`, `ClassificationResult`, `DetectionResult`, `RGBImage`.

[Unreleased]: https://github.com/mauriciobenjamin700/ort-vision-sdk/compare/web-v0.2.0...HEAD
[0.2.0]: https://github.com/mauriciobenjamin700/ort-vision-sdk/releases/tag/web-v0.2.0
[0.1.0]: https://github.com/mauriciobenjamin700/ort-vision-sdk/releases/tag/web-v0.1.0
