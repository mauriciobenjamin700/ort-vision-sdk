# Changelog

All notable changes to `@mauriciobenjamin700/ort-vision-sdk-web` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.7.1] - 2026-08-14

### Fixed

- **A pipeline buffer that a consumer transferred away is replaced instead of
  written into.** `LetterboxPipeline` and `ResizePipeline` hold one
  `Float32Array` and hand the same one out every call, which is the whole point
  of them — but `ort.env.wasm.proxy` (ONNX Runtime in a worker) posts the input
  tensors with their `ArrayBuffer`s in the transfer list, and that **detaches**
  them on this side. A detached `Float32Array` is silently `0` long: the writes
  land nowhere and the next `InferenceSession.run` rejects with

  ```text
  Tensor's size(1228800) does not match data length(0).
  ```

  on every other call — the detached buffer throws, the throw leaves the claim
  outstanding, so the call after it allocates a fresh array and succeeds.
  Measured in Chromium against a 640×640 YOLO detector: run 1 ok (195 ms), run 2
  rejected, run 3 ok (206 ms).

  The buffer holder now treats a length that no longer matches the target as
  spent and allocates a replacement, so the cost is one allocation per transfer
  instead of a failure every second inference — and reuse still holds for every
  consumer that copies the tensor rather than transferring it. Nothing about the
  public shape changed: `run()` still reports `reused: true` when the returned
  data is the pipeline's own buffer.

  Without this, `env.wasm.proxy` is unusable with the built-in tasks, and that
  flag is the only way to keep inference off the browser's main thread — measured
  at 805 ms of frozen UI for one warm-up on a 32-core desktop, against 18 ms with
  the worker.

## [0.7.0] - 2026-08-08

### Added

- **`Classifier.warmup(runs = 1)`.** The classifier was the one task left
  without it — the documentation already claimed all four had it. The cost it
  moves is the same one the other three move (WebGPU compiling shaders, the WASM
  backend faulting in its arenas), and it lands worse here: an app running a
  detector and then a classifier could warm the first and not the second, so an
  unavoidable first-run cost sat right before the answer appears on screen.

- **`ResizePipeline`, the classification counterpart of `LetterboxPipeline`**
  (plus `resizeToTensorData`, its one-shot form, and `writePlanarFloat32`, the
  loop both pipelines share). A classifier stretches to the model's input rather
  than letterboxing into it — no padding, no scale to invert afterwards — so it
  could not reuse the letterbox path as it stands.

### Changed

- **`Classifier` preprocessing allocates nothing per frame and walks the image
  once.** It was still on the composable route (`resize` → `normalize` →
  `toCHW`), which allocates an `RGBImage` and two `Float32Array`s and scans each
  end to end: roughly 1.4 MB of fresh garbage per 224×224 `predict()`, produced
  exactly when a phone near its ONNX memory ceiling can least afford it. It now
  goes through `ResizePipeline`, which resizes with one `drawImage` and writes
  normalized planar float32 into a buffer held across calls.

  **The output is bit-identical** to the composable path — asserted value by
  value against `normalize` → `toCHW`, including the `mean=[0,0,0]`,
  `std=[1,1,1]` configuration an Ultralytics `-cls` export uses, where the old
  route walked the whole buffer and allocated another one to change no value at
  all.

### Fixed

- **The web reference stopped listing `decodeYoloV8`, `decodeYoloV8Anchors` and
  `decodeYoloV8Seg`.** They were removed in 0.6.0; the page kept advertising
  them as current API. It documents both preprocessing pipelines now.

## [0.6.1] - 2026-08-08

### Fixed

- **The threshold quoted in a `NoDetectionsError` now reads the same in both
  SDKs.** This side rendered a whole threshold as `confThreshold=1` where the
  Python SDK rendered `conf_threshold=1.0`, and wrote `0.0000001` where Python
  switched to `1e-07`. A fused pipeline is built once and runs under both
  runtimes from the same file, so the two SDKs were describing one run with two
  different numbers. Both sides now render six decimals with the trailing zeros
  trimmed, and the same table in both test suites fails if either side drifts.

- **The changelog's link table is no longer stuck at 0.2.0.** `[Unreleased]`
  compared against `web-v0.2.0`, so following it showed four releases of shipped
  history as unreleased work, and `[0.3.0]` through `[0.6.0]` had no link
  definition at all — they rendered as literal `[0.6.0]` text. The table is
  rebuilt through 0.6.1. `[0.1.0]` loses its definition on purpose: no
  `web-v0.1.0` tag was ever pushed, so that link was a 404.

## [0.6.0] - 2026-08-07

### Added

- **`DetectClassify` runs through the fused letterbox pipeline, and can be
  warmed up.** It was written against the pre-`LetterboxPipeline` preprocessing
  and would otherwise have been the only task left on the slow path — and the
  only one unable to pay its shader compilation up front, which is the task that
  needs it most: two models plus the bridge compile on the first inference.

- **`raiseOnEmpty` — opt in to treating an empty result as an error.** Mirrors
  the Python SDK: `Detector`, `Segmenter` and `DetectClassify` resolve to an
  empty envelope by default, and throw the new `NoDetectionsError` when the flag
  is set. Available on the construction options and as a per-call override, with
  the same message — which names the threshold that applied, plus the image and
  the class filter when either narrowed the search.

  ```typescript
  const det = await Detector.create("/models/yolov8n.onnx", {
    confThreshold: 0.7,
    raiseOnEmpty: true,
  });
  await det.predict("/img.jpg");
  // NoDetectionsError: No detections in /img.jpg: nothing cleared confThreshold=0.7.
  ```

- **`DetectClassify` — run a fused detect→classify pipeline in the browser.**
  A pipeline built by the Python SDK's `ort_vision_sdk.compose` (0.7.0) already
  contains both models plus the crop-and-resize bridge between them. That matters
  more in a tab than anywhere else: two models mean two `.onnx` downloads, two
  WASM/WebGPU session initializations, and a per-crop round trip through
  JavaScript to slice, resize and restack the regions before the second model can
  see them. A fused pipeline has one download, one session, and no round trip.

  ```typescript
  import { DetectClassify } from "@mauriciobenjamin700/ort-vision-sdk-web";

  const pipeline = await DetectClassify.create("/models/pipeline.onnx");
  for (const d of (await pipeline.predict("/images/flock.jpg"))[0]) {
    console.log(d.name, d.conf, d.classification?.name);
  }
  ```

- **`readFusionSpec()` and the pipeline contract** (`FusionSpec`, `CropSource`,
  the `INPUT_*` / `OUTPUT_*` names, `METADATA_PREFIX`). The letterbox resolution,
  the crop size, whether the classifier output still needs a softmax and the class
  names of both stages are read out of the model's own `ovs.` metadata — the same
  keys, encodings and fallbacks the Python side writes. A pipeline fused once
  therefore behaves identically in both runtimes, off the same file.

- **`DetectionResult.classification`** (optional, absent for a plain detector) and
  the `DetectClassifyResults` envelope, which carries `names` and
  `classifierNames` separately because the two stages have unrelated label spaces.

- **`FusionError`** and `parseNames()`, split out of `modelNames()` so both class
  maps of a pipeline can be parsed with the same reader.

- **`warmup()` on `Detector` and `Segmenter`.** The first inference of a session
  is not representative: WebGPU compiles its shaders on it and the WASM backend
  faults in its arenas, so on a phone the first frame can take seconds while
  every later frame takes tens of milliseconds. `await det.warmup()` runs the
  model once on a zero tensor, moving that cost to wherever the user is already
  watching a spinner.

- **`LetterboxPipeline`, exported.** The fused preprocessing path the tasks now
  take, available to a custom pipeline that wants the same allocation-free
  behaviour. `letterboxToTensorData` is the one-shot form.

### Changed

- **Preprocessing is ~2x faster and allocates nothing per frame.** Chaining the
  composable primitives cost eleven full-buffer passes and six large allocations
  per frame: `getImageData` → RGBA→RGB → RGB→RGBA → `putImageData` →
  `drawImage` → `getImageData` → RGBA→RGB → fill → row copies → `toFloat32` →
  `toCHW`. The tasks now go through `LetterboxPipeline`, which collapses the
  second half into one `drawImage` — resizing *and* positioning the content
  inside the padded target in a single accelerated operation — plus one loop
  that reads the resulting RGBA and writes planar float32 straight into a buffer
  reused across frames.

  Measured in Chromium, median of 31 runs, letterboxing into 640x640:

  | Source | Before | After | Speedup |
  | --- | --- | --- | --- |
  | 1920x1080 | 19.8 ms | 10.7 ms | 1.85x |
  | 1280x720 | 13.8 ms | 7.8 ms | 1.77x |
  | 640x480 | 6.8 ms | 3.1 ms | 2.19x |

  **The output is bit-identical**, verified in a real browser against the
  primitive-chained path across four source sizes — maximum difference 0.

  The primitives are unchanged and still exported: they are what makes a custom
  pipeline writable. Only the path the built-in tasks take has changed.

- **The preprocessing canvas is created with `willReadFrequently`.** It is read
  back with `getImageData` on every frame, which is what the hint exists for.
  Measured effect on speed in Chromium: none beyond noise. What it removes is
  the `Canvas2D: Multiple readback operations using getImageData are faster with
  the willReadFrequently attribute set to true` warning the browser was printing
  into every consumer's console, once per frame.

### Removed

- **`decodeYoloV8`, `decodeYoloV8Anchors`, `decodeYoloV8Seg` and the
  `DecodeYoloV8*Options` type aliases are gone.** Deprecated in 0.2.0 with
  "will be removed in 0.4.0", and still shipping at 0.5.1. Use `decodeYolo`,
  `decodeYoloAnchors`, `decodeYoloSeg` and their `DecodeYolo*Options` types:
  same functions, honest names, since the decoder covers every anchor-free YOLO
  head from v8 through v12.

  `test/deprecations.test.ts` now guards the removal rather than the
  deprecation, so a re-add fails loudly.

### Fixed

- **A custom model with no baked-in `names` no longer fails to create.** Same
  defect as the Python SDK: the fallback was the 80-name COCO preset, so a
  3-class detector threw `Resolved 80 labels but the model has 3 classes`.
  Labels now come up as `class_0`, `class_1`, ..., and the COCO preset applies
  only when the model really does have 80 classes. `defaultLabels` is exported.

- **A model URL that cannot be fetched for its metadata now says so.** The
  fallback — hand the URL to ORT and let it load the model itself — is right,
  but it was silent, and the symptom it produces is remote from the cause:
  class names come back as `class_0`, `class_1`, ... with nothing explaining
  why. It warns now, and names `labels` as the way out.

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

### Notes

- Fusing models stays a Python-side build step; there is no ONNX protobuf writer
  here and no reason for one. The browser only loads the result.
- ORT Web returns `int64` outputs as `BigInt64Array`. The pipeline's class ids and
  detection count are widened to `number` internally, so no caller has to handle
  `BigInt`.

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

[Unreleased]: https://github.com/mauriciobenjamin700/ort-vision-sdk/compare/web-v0.7.0...HEAD
[0.7.0]: https://github.com/mauriciobenjamin700/ort-vision-sdk/compare/web-v0.6.1...web-v0.7.0
[0.6.1]: https://github.com/mauriciobenjamin700/ort-vision-sdk/compare/web-v0.6.0...web-v0.6.1
[0.6.0]: https://github.com/mauriciobenjamin700/ort-vision-sdk/compare/web-v0.5.1...web-v0.6.0
[0.5.1]: https://github.com/mauriciobenjamin700/ort-vision-sdk/compare/web-v0.5.0...web-v0.5.1
[0.5.0]: https://github.com/mauriciobenjamin700/ort-vision-sdk/compare/web-v0.4.0...web-v0.5.0
[0.4.0]: https://github.com/mauriciobenjamin700/ort-vision-sdk/compare/web-v0.3.0...web-v0.4.0
[0.3.0]: https://github.com/mauriciobenjamin700/ort-vision-sdk/compare/web-v0.2.2...web-v0.3.0
[0.2.2]: https://github.com/mauriciobenjamin700/ort-vision-sdk/compare/web-v0.2.1...web-v0.2.2
[0.2.1]: https://github.com/mauriciobenjamin700/ort-vision-sdk/compare/web-v0.2.0...web-v0.2.1
[0.2.0]: https://github.com/mauriciobenjamin700/ort-vision-sdk/releases/tag/web-v0.2.0
