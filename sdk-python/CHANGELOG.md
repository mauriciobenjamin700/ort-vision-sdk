# Changelog

All notable changes to `ort-vision-sdk` (Python) are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.8.0] - 2026-08-08

### Added

- **`require_detections` is part of the public API.** `VisionTask` has always
  been exported, so writing a task of your own is a supported thing to do — but
  the helper the built-in tasks use to raise `NoDetectionsError` was only
  reachable as `ort_vision_sdk.tasks.base.require_detections`. A custom task
  therefore either imported from a submodule, against the convention every other
  import in this package follows, or wrote a second wording of the same error —
  the exact outcome a single shared helper exists to prevent. It is now exported
  from `ort_vision_sdk` and `ort_vision_sdk.tasks`, next to `VisionTask`. The web
  SDK already exported `requireDetections`; this closes the gap between the two.

### Fixed

- **The threshold quoted in a `NoDetectionsError` now reads the same in both
  SDKs.** Python rendered a whole threshold as `conf_threshold=1.0` where the web
  SDK rendered `confThreshold=1`, and switched to `1e-05` where the web SDK still
  wrote `0.00001`. A fused pipeline is built once and runs under both runtimes
  from the same file, so the two SDKs were describing one run with two different
  numbers. Both sides now render six decimals with the trailing zeros trimmed,
  and the same table in both test suites fails if either side drifts.

- **The changelog's link table is no longer stuck at 0.2.0.** `[Unreleased]`
  compared against `v0.2.0`, so following it showed five releases of shipped
  history as unreleased work, and `[0.3.0]` through `[0.7.0]` had no link
  definition at all — they rendered as literal `[0.7.0]` text. The table is
  rebuilt through 0.8.0. `[0.1.0]` loses its definition on purpose: no `v0.1.0`
  tag was ever pushed, so that link was a 404.

## [0.7.0] - 2026-08-07

### Added

- **`raise_on_empty` — opt in to treating an empty result as an error.**
  `Detector`, `Segmenter` and `DetectClassify` return an empty envelope when
  nothing is detected, and that stays the default: the model looked and found
  nothing is a successful inference, not a failure. But a step whose
  *precondition* is that something is there needs the opposite, and previously
  had to re-check the length at every call site. The flag is available on the
  constructor and as a per-call override, and raises the new
  `NoDetectionsError`:

  ```python
  det = Detector("yolov8n.onnx", conf_threshold=0.7, raise_on_empty=True)
  det.predict("img.jpg")
  # NoDetectionsError: No detections in img.jpg: nothing cleared conf_threshold=0.7.
  ```

  There is no separate threshold for the error, because `conf_threshold` is
  already what decides whether something counts as a detection — "nothing
  detected" and "nothing confident enough" are one condition. The message names
  the threshold that applied to the call, plus the image and the class filter
  when either narrowed the search, since a bare "no detections" cannot
  distinguish a blank image from a threshold set too high.

- **`ort_vision_sdk.compose` — fuse a detector and a classifier into one
  `.onnx`.** Chaining the two tasks meant two sessions, two model loads, and a
  round trip through Python for every crop: slice the image, resize each region,
  restack a batch, call the second runtime. On a phone or in a browser tab that
  round trip is frequently the dominant cost, and the second load is what makes a
  page slow to become interactive.

  `fuse_detect_classify()` rewrites both protobufs into a single graph — the
  detector's nodes, a bridge, then the classifier's nodes — so it is one file,
  one session, one load, and the crops never leave the runtime:

  ```python
  from ort_vision_sdk.compose import fuse_detect_classify

  fuse_detect_classify("yolov8n.onnx", "resnet18.onnx", "pipeline.onnx", max_detections=20)
  ```

  The bridge is authored in plain ONNX operators: `NonMaxSuppression` with
  `center_point_box=1` (which eats the YOLO head's `(cx, cy, w, h)` directly),
  `TopK` + `Pad` to a fixed row count, and `RoiAlign` at `spatial_scale=1.0` to
  crop and resample every box in one batched call.

- **`DetectClassify` — the runtime for a fused pipeline.** Loads the file and
  configures itself from it; every detection it yields carries a populated
  `classification`. Same `predict()` / `async_predict()` / `ort_async_predict()`
  triad as the other tasks:

  ```python
  from ort_vision_sdk import DetectClassify

  pipeline = DetectClassify("pipeline.onnx")
  for d in pipeline.predict("flock.jpg")[0]:
      print(d.name, d.conf, d.classification.name)
  ```

- **`crop_source="original"` — classify at native resolution.** Cropping from the
  letterboxed tensor classifies a small object from its downscaled copy: a 40x40
  px box inside a 640 letterbox reaches the classifier upsampled from 40x40
  pixels of real detail. This mode adds a full-resolution input plus the
  letterbox parameters, and the bridge inverts the letterbox transform in-graph
  so the crop comes out of the original photo. Still one session, still one load.

- **`FusionSpec` — the pipeline's configuration lives in the pipeline.** The
  letterbox resolution, the crop size, the crop source, the thresholds, whether
  the classifier output still needs a softmax, and the class names of *both*
  stages are written into the model's metadata under an `ovs.` namespace and read
  back at load time. Nothing is restated by the caller, so nothing can drift —
  the same principle already applied to single models in 0.6.0.

- **`DetectionResult.classification`** (optional, `None` for a plain detector) and
  the `DetectClassifyResults` envelope, which carries `names` and
  `classifier_names` separately. The two stages have unrelated label spaces — a
  detector finding `sheep` feeding a classifier answering `famacha_3` shares no
  class ids with it — and merging them would make `cls` and `classification.cls`
  look comparable when they are not.

- **`FusionError`**, `graph.parse_names()`, and `preprocess.IMAGENET_MEAN` /
  `IMAGENET_STD` (previously private to `Classifier`, now shared with the fusion's
  defaults).

- **The `[compose]` extra** (`pip install "ort-vision-sdk[compose]"`), which adds
  `onnx`. It is needed only to *build* a pipeline — a step you run once next to
  your export pipeline. Running the fused `.onnx` needs nothing beyond the
  `onnxruntime` already in the base install, which is what lets the same file run
  in a browser through `@mauriciobenjamin700/ort-vision-sdk-web`.

### Changed

- **`decode_yolo` is up to 8x faster on frames with few or no detections.** The
  decoder transposed the model's `(channels, anchors)` output to
  `(anchors, channels)` before reducing, which made every reduction walk a
  column with a stride of `num_anchors` — a cache line fetched per element. It
  now reduces over the channel axis in the model's own layout, reading
  contiguous rows, and asks `argmax` which class won only for the anchors that
  passed the confidence threshold instead of for all 8400.

  | Candidates above threshold | Before | After | Speedup |
  | --- | --- | --- | --- |
  | none | 0.52 ms | 0.06 ms | 8.10x |
  | 50 | 0.87 ms | 0.47 ms | 1.84x |
  | 500 | 5.63 ms | 5.15 ms | 1.09x |
  | 2000 | 19.52 ms | 19.22 ms | 1.02x |

  The gain lands where the decode itself is the cost, which is the ordinary
  case: a frame with nothing in it, or with a handful of objects. Once hundreds
  of candidates survive, NMS dominates and this changes little.

  Output is unchanged — the parity fixtures are what says so, rather than an
  argument about equivalence.

- **Downscaling by 2x or more runs in two steps, roughly halving `preprocess`.**
  `resize` now applies an integer box reduction (`PIL.Image.reduce`) before
  resampling onto the exact target — the optimization PIL exposes as
  `reducing_gap`, applied explicitly so it engages at the ratios this SDK
  actually sees. Letterboxing 1920x1080 into 640x640 went from 7.8 ms to 3.5 ms
  (2.25x), 1280x720 from 4.8 ms to 2.3 ms (2.08x), 3840x2160 from 29.8 ms to
  10.5 ms (2.84x).

  Where the reduction applies, **the pixels fed to the model change, and so do
  the detections**. Below a 2x downscale, and for every upscale, the output is
  byte-identical to before — pinned by tests. `NEAREST` is exempt entirely,
  since a caller asking for it wants unblended pixels.

  This is not a quality improvement, and the distinction is worth stating
  plainly. Against a LANCZOS reference, box reduction wins on photographic
  content but loses badly when the content's period resonates with the
  reduction factor: 2 px stripes every 6 rows, reduced by 3, score an MSE of
  106 against the single pass's 13. Across six content types the two paths
  split three-three. What changes is *which* artifacts a downscale produces,
  not how many — `test_resonant_content_is_the_known_weak_case` pins the losing
  case so it stays visible.

  The new `reduction_factor` helper is exported, so a caller can ask whether a
  given source/target pair takes the two-step path.

- **`decode_yolo_seg` is 1.3x–6.7x faster**, depending on how many instances
  survive and how large they are. The prototype combination and the sigmoid ran
  over every prototype pixel of every instance, and the result was cropped to
  the bounding box afterwards. They now run on the cropped region only — sigmoid
  is elementwise, so slicing first is exact, and it skips the pixels that were
  about to be discarded. Measured on 8400 anchors with 160x160 prototypes: 6.7x
  at 100 instances of 60px, 2.0x at 30 instances of 80px, 1.3x at 30 instances
  of 200px.

### Removed

- **`decode_yolov8`, `decode_yolov8_anchors` and `decode_yolov8_seg` are gone.**
  Deprecated in 0.2.0 with "will be removed in 0.3.0", and still shipping at
  0.6.0 — a deprecation nobody enforces stops being one. Use `decode_yolo`,
  `decode_yolo_anchors` and `decode_yolo_seg`: same functions, honest names.
  The decoder was never v8-specific; it covers every anchor-free YOLO head from
  v8 through v12.

  `tests/test_deprecations.py` now guards the removal rather than the
  deprecation, so a re-add fails loudly.

- **Python 3.10 is no longer supported.** `requires-python` is `>=3.11`.
  `onnxruntime` stopped publishing cp310 wheels at 1.24, so a 3.10 install was
  already pinned to `onnxruntime <= 1.23` without saying so — a floor that
  quietly decays is worse than one that is stated. The CI matrix moves to
  3.11 / 3.12 / 3.13.

### Fixed

- **A custom model with no baked-in `names` no longer fails to construct.** The
  fallback for a model that declares no class names was the COCO preset, which
  names exactly 80 classes — so a 3-class detector raised `Resolved 80 labels
  but the model has 3 classes` and could not be built at all without passing
  `labels`. An export without `names` is an ordinary thing to have; the labels
  now come up as `class_0`, `class_1`, ... and the COCO preset is used only when
  the model really does have 80 classes. `default_labels` is exported for
  callers who want the same decision.

- **Automatic provider selection no longer tries TensorRT.** `onnxruntime-gpu`
  reports `TensorrtExecutionProvider` as available whenever it was compiled in,
  not when it can load — so on a machine without the TensorRT shared libraries,
  which is the common case for that wheel, every session printed four lines of
  failed provider registration and fallback notices to stderr before recovering
  on CUDA:

  ```text
  *************** EP Error ***************
  EP Error ... Please install TensorRT libraries as mentioned in the GPU
  requirements page ... when using ['TensorrtExecutionProvider', ...]
  Falling back to ['CUDAExecutionProvider', 'CPUExecutionProvider'] and retrying.
  ****************************************
  ```

  TensorRT also builds an engine on first run that can take minutes, which is
  not a cost to opt somebody into by default. `providers=["tensorrt"]` still
  selects it, and still raises `ProviderNotAvailableError` when the installed
  build does not carry it.

- **Instance masks no longer lose precision to a `uint8` round-trip.** Resizing
  a soft mask to its bounding box went through PIL, which cannot resample a
  float array — so the mask was quantized to `uint8` first. That put the input
  to the `>= mask_threshold` test on a grid of `1/255` steps and flipped border
  pixels for no reason. Resampling is now a bilinear pass in `float32`, and it
  agrees with a `float64` reference on **100%** of pixels where the old path
  agreed on 99.7%.

  This also removes a divergence between the two published SDKs: the web
  implementation always resampled in float, so the same model on the same image
  produced different masks in Python and in the browser. They now produce the
  same bitmap, and `fixtures/parity/` checks it.

  Masks stored by 0.6.0 or earlier will differ from new ones by a few border
  pixels.

- **Score ties in `nms` and `batched_nms` resolve deterministically, to the
  lowest index.** `nms` ordered candidates with `scores.argsort()[::-1]`, which
  reverses a stable sort — so tied scores were visited in *descending* index
  order, the opposite of `torchvision` and of the web SDK. Two boxes tied on
  score therefore produced a different survivor in each SDK. `batched_nms` had
  the same problem across classes, where the surviving order also depended on
  the class-iteration order. Both now break ties by index explicitly.

- **`nms` no longer emits `RuntimeWarning: invalid value encountered in
  divide`.** The IoU was computed for every pair and then masked, so a pair of
  zero-area boxes evaluated `0 / 0` before the result was discarded. Letterbox
  padding clips boxes down to zero area on ordinary frames, so the warning
  reached callers' logs during normal use. The division is now masked instead of
  discarded.

### Notes

- A fixed `max_detections` keeps every shape in the fused graph static, which is
  what TensorRT/NNAPI/WebGPU need in order to compile it, and removes the
  zero-detection edge case (the classifier is never handed an empty batch).
  `max_detections=None` trades that for exact-sized outputs.
- `fuse_detect_classify` runs the fused graph once before returning. This catches
  a classifier whose own graph only accepts the batch size it was exported with —
  a hardcoded `Reshape`, typically — which rewriting the declared input shape does
  not fix. Pass `validate=False` to skip.
- The fused NMS scores every class independently, while
  `postprocess.decode_yolo` collapses each anchor to its `argmax` first. An anchor
  clearing the threshold for two classes yields two rows in a pipeline and one in
  a standalone `Detector`.

### Internal

- **End-to-end tests against real ONNX Runtime sessions**
  (`tests/test_e2e_onnx.py`). The suite previously drove every task through a
  fake backend, so nothing checked that a task can load a file, feed ORT a
  tensor it accepts and read the outputs back — `ort_async_run` in particular had
  never run against real ORT. Five tiny `.onnx` fixtures with `Constant` outputs
  make the expected predictions exact and need no download;
  `scripts/gen_test_models.py` regenerates them (it needs `onnx`, which stays out
  of the package's dependencies).

- **Shared Python/Web parity fixtures** (`fixtures/parity/`) covering `nms`,
  `batched_nms`, `decode_yolo`, `decode_yolo_seg`, `softmax`, `topk` and
  `model_names`. Both suites read the same committed numbers, which is how the
  mask and tie-break divergences above were found.

- **Benchmark harness** (`scripts/bench.py`, `make bench-python`) with a
  committed baseline, so the remaining optimization work is measurable. The
  baseline is a local reference, not a CI gate. Cases whose median falls under
  `NOISE_FLOOR_MS` are reported but never counted as regressions — `softmax`
  runs in ~5 microseconds, where a context switch reads as a "+253% regression"
  on code nobody touched.

- **Direct tests for the preprocessing primitives**
  (`tests/test_preprocess_image.py`). `resize`, `letterbox`, `to_tensor`,
  `to_chw`, `normalize` and the cv2 interop helpers had no tests of their own —
  they were only reachable through a task, so a change to any of them surfaced
  as a puzzling assertion about bounding boxes.

## [0.6.0] - 2026-08-03

### Added

- **Tasks read their input resolution off the model instead of trusting
  configuration.** `Classifier`, `Detector` and `Segmenter` now ask the ONNX
  graph what shape it declares and preprocess to that. The resolution a session
  must be fed at is a property of the export — feeding a 640x640 tensor to a
  graph exported at 224x224 makes ORT abort mid-run with:

  ```text
  [ONNXRuntimeError] Got invalid dimensions for input: images for the following
  indices index: 2 Got: 640 Expected: 224
  ```

  A caller had no way to see that coming: the number lives in the file. So it is
  read from there now.

  ```python
  # An Ultralytics -cls export is 224; nothing to configure, nothing to get wrong
  clf = Classifier("classify.onnx")
  print(clf.input_size)  # (224, 224)
  ```

  `input_size` became `None` by default and is now a *fallback*, used only when
  the graph leaves its spatial axes dynamic. Passing a size that contradicts a
  static graph emits a `UserWarning` and the graph wins — honoring the caller
  there would only turn a fixable mismatch into a failed run.

- **`input_size` property on every task**, so callers can read back the
  resolution inference actually runs at rather than the one they asked for.

- **Class names come from the model when the caller passes none.** Ultralytics
  bakes `names` into the model metadata; a hand-maintained list beside it can be
  reordered by accident, which silently swaps predictions between classes instead
  of failing. `labels=None` (now the default for `Detector` and `Segmenter` too)
  reads `names` from the export, falling back to the COCO preset for detection
  and generated `class_<id>` names for classification.

  ```python
  det = Detector("detect.onnx")   # a single-class custom export
  print(det.labels)               # ("ocular-mucosa",) — read from the model
  ```

  This also fixes a papercut: a custom detector used to *fail* without explicit
  labels, because the 80-name COCO default disagreed with its class count.

- **`OrtSession.metadata`** exposes the model's custom metadata map (`names`,
  `task`, `imgsz`, ...), so tasks can read what a model says about itself
  without reaching into the ORT-specific `raw` session.

- **`MetadataBackend` + `read_metadata`** (`ort_vision_sdk.core.backend`): the
  metadata map is a *capability*, declared as its own protocol rather than added
  to `InferenceBackend`. A bridge that only forwards tensors to a native runtime
  cannot read it, and requiring it of every backend would break implementations
  that are otherwise complete. Existing backends keep satisfying
  `InferenceBackend` unchanged.

- **`ort_vision_sdk.graph`**: `spatial_input_size`, `resolve_input_size` and
  `model_names` — the pure helpers behind all of the above, exported so callers
  building their own pipeline can reuse the same precedence rules.

### Changed

- `Detector`/`Segmenter` default `labels` from `"coco"` to `None`, which means
  "read the model, fall back to COCO". A COCO-trained export resolves to the
  same 80 names as before; a custom export now resolves to its own.

## [0.5.0] - 2026-08-02

### Added

- **`predict()` now fills the `speed` field it always advertised.** Every
  `Results` dataclass declared `speed: dict[str, float]` and documented it as
  the Ultralytics-style timing breakdown — and it was always `{}`, because no
  task ever populated it. `Classifier`, `Detector` and `Segmenter` now time
  each stage, in all three scheduling modes (`predict`, `async_predict`,
  `ort_async_predict`):

  ```python
  results = detector.predict("street.jpg")
  print(results[0].speed)
  # {"load": 84.2, "preprocess": 11.7, "inference": 118.9, "postprocess": 6.4}
  ```

  Four keys instead of Ultralytics' three: `preprocess`, `inference` and
  `postprocess` measure the same boundaries Ultralytics does, and `load`
  covers the read/decode this SDK performs inside `predict()` — on a cold page
  cache it dominates everything else, and folding it into `preprocess` would
  misreport where the cost is.

  `SpeedTimer` and `STAGES` are exported from `ort_vision_sdk.core` so callers
  can time their own pipeline stages around the SDK calls using the same
  boundaries. This matches the `@mauriciobenjamin700/ort-vision-sdk-web@0.3.0`
  release, keeping the two SDKs' surfaces aligned.

## [0.4.0] - 2026-06-19

### Added

- **Pluggable inference backend.** A new `InferenceBackend` protocol
  (`ort_vision_sdk.core.backend`) formalizes the interface every task drives
  inference through — input/output metadata plus `run` / `async_run` /
  `ort_async_run`. The default backend is `OrtSession` (unchanged), but tasks now
  accept a `backend=` argument to run inference through a **different runtime**
  without an `onnxruntime` Python wheel: `onnxruntime-web` in the browser, or the
  native `onnxruntime-android` AAR over a bridge. Preprocessing, postprocessing
  and result parsing still run in Python (NumPy); only `run` crosses the bridge.

  ```python
  # Default: in-process ONNX Runtime (unchanged)
  det = Detector("yolov8n.onnx")

  # Bridged: inference runs on a native/remote runtime, pre/post stays in Python
  det = Detector("yolov8n.onnx", backend=my_backend)
  ```

- `OrtSession.output_shapes` — declared output shapes, so tasks infer
  `num_classes` from output metadata through the backend interface instead of the
  ONNX-Runtime-specific `OrtSession.raw`.
- `InferenceBackend` and `OrtSession` are now re-exported at the package root.

### Changed

- `VisionTask` (and `Detector` / `Classifier` / `Segmenter`) take an optional
  `backend: InferenceBackend | None = None`. When given, `model_path` /
  `providers` / `session_options` are ignored (the backend owns model loading).
  `Task.session` now returns an `InferenceBackend`. **Fully backward compatible**:
  omit `backend` and behavior is identical to 0.3.x.

## [0.3.2] - 2026-06-13

### Changed

- **`onnxruntime` is now an optional / lazy import.** Preprocessing,
  postprocessing, `load_image`, types and labels now import in environments
  **without** an `onnxruntime` wheel (e.g. Pyodide/WASM in a browser, where
  inference is bridged to `onnxruntime-web`). `onnxruntime` is imported lazily
  inside `OrtSession.__init__` and `providers.available_providers`, and is
  annotation-only (guarded by `TYPE_CHECKING`) in the task modules.
- Behavior is **unchanged** when `onnxruntime` is installed: constructing
  `Detector` / `Classifier` / `Segmenter` still requires it (the lazy import
  raises a clear `ImportError` if absent). All existing tests pass.

## [0.3.1] - 2026-05-31

### Changed

- README now links to the bilingual (PT-BR + EN-US) documentation site on
  GitHub Pages (<https://mauriciobenjamin700.github.io/ort-vision-sdk/>) so the
  PyPI project page points readers to the full guide and API reference.
  Documentation-only release; no public API changes.

## [0.3.0] - 2026-05-03

### Added

- **Async inference API.** Each task and the underlying `OrtSession` now
  expose two async variants of the existing sync entrypoints, with explicit
  prefixes so callers know which threading model they are opting into:
  - `async_*` — `asyncio.to_thread` wrappers. Default async path: dispatches
    the sync call to the asyncio executor's thread pool, freeing the event
    loop. Right for FastAPI/AnyIO handlers and most async code.
    - `OrtSession.async_run`
    - `Classifier.async_predict`, `Detector.async_predict`,
      `Segmenter.async_predict`
  - `ort_async_*` — uses ORT's native `InferenceSession.run_async` callback,
    so all in-flight inferences share the ONNX Runtime internal thread pool
    (configured via `SessionOptions`). Right for high-concurrency workloads
    where you don't want a Python thread per await. Requires
    `onnxruntime>=1.16`.
    - `OrtSession.ort_async_run`
    - `Classifier.ort_async_predict`, `Detector.ort_async_predict`,
      `Segmenter.ort_async_predict`
- `pytest-asyncio` added to `[dev]` extras (with `asyncio_mode = "auto"`) so
  async tests are recognised without per-test markers.

### Changed

- Each task's `predict()` now delegates the result-building logic to a
  private `_build_results()` helper, so the sync and async variants share
  one implementation of decode/NMS/result envelope construction. Public
  behaviour is unchanged.

## [0.2.1] - 2026-05-03

Patch release focused on documentation and CI hardening — no public API changes.

### Changed

- README rewritten as a comprehensive PyPI landing page: badges, "Why this
  SDK" comparison, task/result tables, runnable quick-start for `Classifier`,
  `Detector`, `Segmenter`, sections on inputs, label specs, execution
  providers, and "Common patterns" recipes.

### Fixed

- `mypy` strict mode now passes on Python 3.10/3.11/3.12. `numpy>=2.x` marks
  `ndarray` as generic but its type-parameter defaults rely on PEP 696 (Python
  3.13+); we disabled `disallow_any_generics` to keep the SDK building under
  the supported Python range. Two real shape inference issues in
  `postprocess/segmentation.py` and `tasks/detector.py` were also fixed via
  explicit annotations.
- CI smoke-install step now captures the packed tarball name dynamically and
  resolves the package name via `package.json`, fixing failures introduced by
  the previous web-package scope rename.

## [0.2.0] - 2026-05-02

This release brings the public API in line with the Ultralytics / PyTorch
idiom, so code ported from those ecosystems works with minimal changes.

### Added

- **Per-image `Results` envelopes** (`DetectionResults`, `ClassificationResults`,
  `SegmentationResults`) returned by every `predict()` call as a 1-element
  list, mirroring `YOLO("img.jpg")`. Each envelope holds:
  - `boxes` / `probs` / `masks` — bulk numpy views with Ultralytics-style
    accessors (`xyxy`, `xywh`, `xyxyn`, `xywhn`, `cls`, `conf`, `data`,
    `top1`, `top1conf`, `top5`, `top5conf`).
  - `detections` — tuple of per-instance dataclasses (the previous return
    type).
  - `names` — `dict[int, str]` matching `model.names`.
  - `orig_img`, `orig_shape`, `path`, `speed`.
- **Ultralytics-style aliases** on `BoundingBox`, `ClassProbability`,
  `ClassificationResult`, `DetectionResult`, `SegmentationResult`:
  `cls`, `conf`, `name`, `box`, `xyxy`, `xywh` (center coords),
  plus `xyxyn(orig_shape)` / `xywhn(orig_shape)` methods on `BoundingBox`.
- **`__call__` on every task class** — `Detector(model)(image)` now works
  like `torch.nn.Module.__call__`, delegating to `predict()`.
- **`names: dict[int, str]` property** on every task class alongside
  `labels: tuple[str, ...]`.
- **Short device aliases** in `providers=` — `"cpu"`, `"cuda"`, `"gpu"`,
  `"tensorrt"` / `"trt"`, `"coreml"` / `"mps"`, `"dml"` / `"directml"`,
  `"openvino"` are expanded to the canonical `*ExecutionProvider` names.
  New helper `normalize_provider(name)`.
- **Explicit `head=` parameter** on `Detector` and `Segmenter` constructors —
  caller declares which decoder family the model uses (`"yolo"` for
  detect heads of v8/v9/v10/v11/v12/v26; `"yolo-seg"` for the matching
  seg heads). The SDK does **not** auto-detect — wrong head raises
  `ValueError`. New types `DetectorHead`, `SegmenterHead`.
- **`classes` filter** on `Detector.predict()` / `Detector.__call__()` and
  `Segmenter.predict()` / `Segmenter.__call__()` — keeps only results whose
  `class_id` is in the supplied list, matching Ultralytics'
  `model.predict(img, classes=[0, 16])`.
- **`postprocess.batched_nms(boxes, scores, idxs, iou_threshold)`** matching
  `torchvision.ops.batched_nms`.
- **Generic YOLO decoder names**: `decode_yolo`, `decode_yolo_anchors`,
  `decode_yolo_seg` — same code, accurate name (covers v8/v9/v10/v11/v12).
- **Preprocess helpers** mirroring torchvision/OpenCV:
  - `to_tensor(image)` → CHW `float32 / 255` (ToTensor parity).
  - `from_cv2(bgr)` / `to_cv2(rgb)` — channel-order bridges.

### Changed

- `Detector.predict(img)` now returns `list[DetectionResults]` (1 element)
  instead of `list[DetectionResult]`. Iterate the envelope to recover the
  old shape: `for d in detector.predict(img)[0]: ...`.
- `Classifier.predict(img)` now returns `list[ClassificationResults]`.
- `Segmenter.predict(img)` now returns `list[SegmentationResults]`.
- `nms(boxes, scores, iou_threshold)` — first parameter renamed from
  `boxes_xyxy` to `boxes` for `torchvision.ops.nms` parity.
- The detection decoder no longer applies the per-class loop inline — it
  delegates to `batched_nms`.

### Deprecated

- `decode_yolov8`, `decode_yolov8_anchors`, `decode_yolov8_seg` — emit
  `DeprecationWarning`; will be removed in 0.3.0. Use `decode_yolo`,
  `decode_yolo_anchors`, `decode_yolo_seg` instead.

### Migration

```python
# Before (0.1.0)
detections = detector.predict("street.jpg")
for d in detections:
    print(d.class_id, d.class_name, d.confidence, d.bbox.as_xyxy())

# After (0.2.0)
results = detector.predict("street.jpg")  # list[DetectionResults], len 1
r = results[0]

# Per-instance dataclasses (legacy fields still work):
for d in r:
    print(d.class_id, d.class_name, d.confidence, d.bbox.as_xyxy())
# or with the new short aliases:
for d in r:
    print(d.cls, d.name, d.conf, d.box.xyxy)

# Bulk numpy access (matches Ultralytics):
print(r.boxes.xyxy.shape, r.boxes.cls, r.boxes.conf, r.names)
```

## [0.1.0] - 2026-05-02

### Added

- Initial alpha release.
- `Classifier` and `Detector` task classes wrapping `onnxruntime.InferenceSession`.
- Image I/O helpers (`load_image`, `ImageInput`).
- Default label maps (`COCO_CLASSES`, ImageNet via `resolve_labels`).
- Public types: `BoundingBox`, `ClassProbability`, `ClassificationResult`, `DetectionResult`, `ImageArray`.
- Optional extras: `gpu` (onnxruntime-gpu), `opencv` (opencv-python), `dev` (test/lint tooling).

[Unreleased]: https://github.com/mauriciobenjamin700/ort-vision-sdk/compare/v0.8.0...HEAD
[0.8.0]: https://github.com/mauriciobenjamin700/ort-vision-sdk/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/mauriciobenjamin700/ort-vision-sdk/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/mauriciobenjamin700/ort-vision-sdk/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/mauriciobenjamin700/ort-vision-sdk/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/mauriciobenjamin700/ort-vision-sdk/compare/v0.3.2...v0.4.0
[0.3.2]: https://github.com/mauriciobenjamin700/ort-vision-sdk/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/mauriciobenjamin700/ort-vision-sdk/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/mauriciobenjamin700/ort-vision-sdk/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/mauriciobenjamin700/ort-vision-sdk/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/mauriciobenjamin700/ort-vision-sdk/releases/tag/v0.2.0
