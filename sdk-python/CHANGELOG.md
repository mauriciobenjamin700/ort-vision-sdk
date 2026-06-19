# Changelog

All notable changes to `ort-vision-sdk` (Python) are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/mauriciobenjamin700/ort-vision-sdk/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/mauriciobenjamin700/ort-vision-sdk/releases/tag/v0.2.0
[0.1.0]: https://github.com/mauriciobenjamin700/ort-vision-sdk/releases/tag/v0.1.0
