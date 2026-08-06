# Changelog

All notable changes to `@mauriciobenjamin700/ort-vision-sdk-web` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **Score ties in `nms` and `batchedNms` break by index explicitly.** Both
  relied on `Array.prototype.sort` being stable to order tied scores. It is, in
  every engine that matters, but leaning on it left the tie behaviour implicit —
  and in `batchedNms` the surviving order also depended on `Map` insertion order,
  which the Python SDK (iterating classes in sorted order) does not share. The
  comparators now fall back to `a - b`, so a tie resolves to the lowest index on
  both sides, matching `torchvision`.

  Only exact ties are affected. The Python SDK was the one actually producing a
  different survivor — it visited ties in descending index order — and this is
  the other half of that fix.

### Internal

- **Parity tests against shared fixtures** (`test/parity.test.ts`, reading
  `fixtures/parity/` at the repository root). They feed the web implementation
  the same inputs the Python SDK was given and require the same outputs, with
  mask bitmaps compared pixel for pixel. Two published artifacts promising the
  same numbers had nothing checking the promise; this is that check, and it
  found a mask-resampling divergence on its first run (fixed on the Python
  side, see that package's changelog).

## [0.5.1] - 2026-08-05

### Fixed

- **A model no longer costs twice its size at the peak of session creation.**
  Reading the metadata map (0.5.0) fetches a URL model into a `Uint8Array`, and
  that read sat *after* `InferenceSession.create` — so the JavaScript buffer
  stayed reachable while ORT copied the model into its WASM heap and allocated
  the graph and the weights on top of it. A 5 MB `.onnx` therefore held 5 MB of
  JS heap plus 5 MB of WASM heap plus the weights at the same instant. On a phone
  loading two models that was enough for ORT's allocator to give up with
  `Can't create a session. failed to allocate a buffer of size 5355557`.

  The metadata is now read before the session is built, so the buffer is
  collectable as soon as ORT has copied it. `test/session.test.ts` pins the order.

  `readMetadata: false` remains the escape hatch for a device that cannot afford
  the bytes at all: ORT loads from the URL and nothing in the SDK ever holds the
  model. Only the class names are lost — the input size still comes from the
  graph — so that route has to pass `labels` itself.

## [0.5.0] - 2026-08-03

### Added

- **Tasks read their class names off the model, matching the Python SDK.**
  `labels` is now optional on all three tasks: when omitted, the names the
  exporter baked into the model are used (Ultralytics writes them as `names` in
  the metadata map), and only a model carrying none falls back to the COCO
  preset for detection/segmentation or generated `class_<id>` labels for
  classification. Passing `labels` still wins, for a model whose names are
  wrong or absent.

  ```typescript
  const det = await Detector.create("/models/detect.onnx");
  console.log(det.labels); // ["ocular-mucosa"] — from the model, not a preset
  ```

  This also fixes a trap: a single-class detector used to **fail** without an
  explicit `labels`, because the 80-name COCO default disagreed with its class
  count.

- **`numClasses` is inferred from the declared output shape.** A YOLO head
  declares `(B, 4 + nc, N)` and a classifier `(B, nc)`, so the count no longer
  has to be supplied. Passing it still validates the labels against the model.

- **`OrtSession.metadata`** exposes the model's custom metadata map (`names`,
  `task`, `imgsz`, ...). `onnxruntime-web` does not surface that map — unlike
  Python's `custom_metadata_map` — so it is read from the model's own bytes at
  load time by walking `metadata_props` in the ModelProto. A truncated or
  unexpected file yields an empty map instead of an error, and every caller
  falls back to what it was given.

- **`OrtSession.outputShapes` / `.outputShape`**, the shapes the graph declares
  for its outputs, with dynamic axes as `null` — the same treatment
  `inputShapes` already got.

- **`detectionNumClasses`, `classificationNumClasses`, `readModelMetadata`,
  `modelNames`**: the pure helpers behind the above, exported for anyone
  assembling their own pipeline.

### Changed

- **A URL model is fetched by the SDK instead of by ORT**, so its bytes are
  available to read metadata from. It is the same single download, and
  `readMetadata: false` on the session options restores the previous path (ORT
  fetches the URL, `metadata` stays empty). A failed fetch falls back to handing
  ORT the URL, so a model that ORT could load still loads.

## [0.4.0] - 2026-08-03

### Added

- **Tasks read their input resolution off the model instead of trusting
  configuration.** `Classifier`, `Detector` and `Segmenter` now ask the ONNX
  graph what shape it declares and preprocess to that. The resolution a session
  must be fed at is a property of the export — feeding a 640x640 tensor to a
  graph exported at 224x224 makes ORT abort mid-run with:

  ```text
  Inference failed: failed to call OrtRun(). ERROR_CODE: 2, ERROR_MESSAGE: Got
  invalid dimensions for input: images for the following indices index: 2
  Got: 640 Expected: 224
  ```

  A caller had no way to see that coming: the number lives in the file. So it is
  read from there now.

  ```typescript
  // An Ultralytics -cls export is 224; nothing to configure, nothing to get wrong
  const clf = await Classifier.create("/models/classify.onnx", { labels: LABELS });
  console.log(clf.inputSize); // [224, 224]
  ```

  `inputSize` is now a *fallback*, used only when the graph leaves its spatial
  axes dynamic. Passing a size that contradicts a static graph logs a warning
  and the graph wins — honoring the caller there would only turn a fixable
  mismatch into a failed run.

- **`inputSize` getter on every task**, so callers can read back the resolution
  inference actually runs at rather than the one they asked for.

- **`OrtSession.inputShapes` / `OrtSession.inputShape`** expose what the graph
  declares, with dynamic axes as `null`. Empty when the runtime reports no
  metadata (`onnxruntime-web` older than 1.21).

- **`OrtSession.release()`** frees the native session. Needed whenever a session
  is discarded while the page lives on — rebuilding a task at a different input
  size, swapping in a newer model — which previously required reaching into
  `session.raw`.

- **`declaredShapesFrom`, `spatialInputSize`, `resolveInputSize`** (plus the
  `DeclaredShape` / `DeclaredDim` types): the pure helpers behind the above,
  exported so callers building their own pipeline can reuse the same precedence
  rules without importing `onnxruntime-web` types themselves.

## [0.3.0] - 2026-08-02

### Added

- **`predict()` now reports where the time went.** Every `Results` envelope
  already carried a `speed` field, mirroring Ultralytics' `results[0].speed` —
  and it was always empty, because no task ever filled it. `Classifier`,
  `Detector` and `Segmenter` now time each stage and hand the breakdown to the
  envelope:

  ```typescript
  const results = await det.predict("/images/street.jpg");
  console.log(results[0].speed);
  // { load: 84.2, preprocess: 11.7, inference: 118.9, postprocess: 6.4 }
  ```

  Four keys instead of Ultralytics' three: `preprocess`, `inference` and
  `postprocess` measure the same boundaries Ultralytics does, and `load` covers
  the fetch/decode this SDK performs inside `predict()` — on a cold cache it
  dominates everything else, and folding it into `preprocess` would misreport
  where the cost is.

  New exports `Speed` (the four-key breakdown) and `SpeedTimer` (the stage
  accumulator) let callers time their own pipeline stages around the SDK calls
  using the same boundaries.

### Changed

- `Results.speed` is typed `Readonly<Speed>` instead of
  `Readonly<Record<string, number>>`, so `speed.inference` is a `number` rather
  than `number | undefined`. Envelopes built by hand default to all-zeros.

### Deprecated

- The `decodeYoloV8` / `decodeYoloV8Anchors` / `decodeYoloV8Seg` aliases were
  announced for removal in `0.3.0`. They survive this release — dropping them
  would break `tempest-react-sdk`, which re-exports all three from its vendored
  copy. The warning now points at `0.4.0`.

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
