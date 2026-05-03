# ort-vision-sdk

[![PyPI](https://img.shields.io/pypi/v/ort-vision-sdk.svg)](https://pypi.org/project/ort-vision-sdk/)
[![Python](https://img.shields.io/pypi/pyversions/ort-vision-sdk.svg)](https://pypi.org/project/ort-vision-sdk/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/mauriciobenjamin700/ort-vision-sdk/blob/main/LICENSE)

A high-level Python SDK for computer-vision inference on top of [ONNX Runtime](https://onnxruntime.ai/).

`ort-vision-sdk` wraps the low-level `InferenceSession` API behind task-oriented classes — `Classifier`, `Detector`, `Segmenter` — that handle preprocessing, execution-provider selection, and postprocessing for you. You go from a raw image (path, bytes, NumPy array, or PIL image) to a typed result in **one call**, with an output shape that matches the [Ultralytics](https://docs.ultralytics.com/) idiom (`boxes.xyxy`, `cls`, `conf`, `names`, ...) so existing code ports over with minimal edits.

```python
from ort_vision_sdk import Detector

det = Detector("yolov8n.onnx")           # any anchor-free YOLO export (v8/v9/v10/v11/v12)
result = det.predict("street.jpg")[0]    # list[DetectionResults], length 1 per image
print(result.boxes.xyxy)                 # (N, 4) float64 array, original-image pixels
print(result.boxes.cls, result.boxes.conf)
for d in result:                          # per-instance dataclasses
    print(d.name, d.conf, d.box.xyxy)
```

---

## Why this SDK

Using `onnxruntime` directly means you have to:

- pick and configure execution providers (CPU / CUDA / TensorRT / ...),
- letterbox / resize / normalize / `to_chw` / batch your image,
- decode the model output (anchor grids, NMS, mask prototypes),
- map boxes back from the letterboxed input to the original image,
- resolve class indices to human-readable labels,
- repeat all of the above per task family.

`ort-vision-sdk` does all of that and gives you a typed, dataclass-based result. The internals are explicit and overridable — you can pass your own `mean` / `std`, `input_size`, `conf_threshold`, `iou_threshold`, `providers`, or a pre-built `ort.SessionOptions`.

## What's in the box

| Task             | Class        | Models supported                                                              |
| ---------------- | ------------ | ----------------------------------------------------------------------------- |
| Classification   | `Classifier` | Any ONNX classifier with output shape `(1, num_classes)` (torchvision-style)  |
| Object detection | `Detector`   | Anchor-free YOLO heads: v8, v9, v10, v11, v12, v26 (`(1, 4 + nc, N)`)         |
| Inst. seg.       | `Segmenter`  | YOLO seg heads: v8-seg, v11-seg, v26-seg (`(1, 4 + nc + nm, N)` + prototypes) |

All three return the same envelope shape — a `list[Results]` of length 1 per image — so you can switch between tasks without rewriting your downstream code.

## Installation

```bash
pip install ort-vision-sdk             # CPU only (default)
pip install "ort-vision-sdk[gpu]"      # adds onnxruntime-gpu for CUDA / TensorRT
pip install "ort-vision-sdk[opencv]"   # adds OpenCV image backend
pip install "ort-vision-sdk[dev]"      # ruff, mypy, pytest, build, twine
```

Requires Python **3.10+**.

---

## Quick start

### Classification

```python
from ort_vision_sdk import Classifier

clf = Classifier(
    "resnet50.onnx",
    labels="imagenet_labels.txt",   # one class per line, or pass a list/dict
    input_size=(224, 224),          # default
    apply_softmax=True,             # set False if your model already outputs probs
)

results = clf.predict("dog.jpg")
r = results[0]

print(r.cls, r.conf, r.name)        # top-1 — Ultralytics-style
print(r.probs.top5)                 # array of top-5 class indices
print(r.probs.top5conf)             # corresponding probabilities
print(r.probabilities[:5])          # tuple of ClassProbability dataclasses
```

### Object detection

```python
from ort_vision_sdk import Detector

det = Detector(
    "yolov8n.onnx",
    labels="coco",                   # default — 80-class COCO preset
    input_size=(640, 640),
    conf_threshold=0.25,
    iou_threshold=0.45,
)

result = det.predict("street.jpg")[0]

# Bulk numpy view — Ultralytics' Boxes interface
print(result.boxes.xyxy.shape)       # (N, 4) absolute pixels
print(result.boxes.xywhn)            # (N, 4) normalized (cx, cy, w, h)
print(result.boxes.cls)              # (N,) int64
print(result.boxes.conf)             # (N,) float64
print(result.boxes.data)             # (N, 6) [x1, y1, x2, y2, conf, cls]

# Per-instance dataclasses
for d in result:
    print(d.name, d.conf, d.box.xyxy)
    # d.cropped_image is an HWC uint8 RGB ndarray of the box crop
```

### Instance segmentation

```python
from ort_vision_sdk import Segmenter

seg = Segmenter(
    "yolov8n-seg.onnx",
    labels="coco",
    mask_threshold=0.5,              # cutoff for soft → binary mask
)

result = seg.predict("street.jpg")[0]

# Same Boxes view as the detector …
print(result.boxes.xyxy, result.boxes.cls, result.boxes.conf)

# … plus per-instance binary masks
for inst in result:
    print(inst.name, inst.conf, inst.box.xyxy)
    print(inst.mask.shape)           # (h, w) uint8 ∈ {0, 255}, cropped to bbox
    print(inst.segmented_image.shape) # (h, w, 3) RGB with background zeroed out
```

---

## Inputs

Every `predict()` accepts the same set of image inputs:

```python
from pathlib import Path
from PIL import Image
import numpy as np

clf.predict("dog.jpg")               # str path
clf.predict(Path("dog.jpg"))         # pathlib
clf.predict(open("dog.jpg", "rb").read())   # raw bytes (PNG, JPEG, ...)
clf.predict(Image.open("dog.jpg"))   # PIL — any mode, converted to RGB
clf.predict(np.zeros((480, 640, 3), dtype=np.uint8))  # HWC uint8 RGB ndarray
```

Need to load an image once and reuse it? Use the same loader the SDK uses internally:

```python
from ort_vision_sdk import load_image
img = load_image("dog.jpg")          # HWC uint8 RGB
clf.predict(img)
```

---

## Labels

Tasks resolve labels at construction time via `resolve_labels`:

```python
from ort_vision_sdk import Classifier, COCO_CLASSES, resolve_labels

# 1) Built-in preset (currently: "coco")
det = Detector("yolov8n.onnx", labels="coco")

# 2) Explicit list / tuple
clf = Classifier("model.onnx", labels=["cat", "dog", "fox"])

# 3) Sparse dict — gaps filled with "class_<id>"
clf = Classifier("model.onnx", labels={0: "cat", 2: "fox"})

# 4) File path — one class per line
clf = Classifier("model.onnx", labels="imagenet_labels.txt")

# 5) None — auto-generates "class_0", "class_1", ... (only works when the
#    model's output shape is statically known)
clf = Classifier("model.onnx", labels=None)
```

`names` on every result is the canonical `dict[int, str]` mapping (mirrors Ultralytics' `model.names`).

---

## Execution providers

By default the SDK picks the first available provider in ORT's preference order. To pin a specific backend, pass `providers=` with either short aliases or canonical ORT names:

```python
det = Detector("yolov8n.onnx", providers=["cuda", "cpu"])
det = Detector("yolov8n.onnx", providers=["tensorrt", "cuda", "cpu"])
det = Detector("yolov8n.onnx", providers=["CUDAExecutionProvider"])  # canonical name
```

Aliases supported: `"cpu"`, `"cuda"`, `"tensorrt"`, `"directml"`, `"coreml"`, `"openvino"`, `"rocm"`. Anything else is forwarded verbatim to ORT.

For fine-grained control (graph optimization, threading, profiling) pass an `ort.SessionOptions` instance:

```python
import onnxruntime as ort

opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
opts.intra_op_num_threads = 4

det = Detector("yolov8n.onnx", session_options=opts)
```

---

## Result objects

Each `predict()` call returns `list[Results]` of length 1 (per image), so the typical pattern is `results[0]`. The envelope is **iterable and indexable** — iterating yields per-instance dataclasses, so legacy "list of detections" code works with one extra `[0]`.

| Envelope                | Bulk view        | Iterating yields     | Notable fields                                       |
| ----------------------- | ---------------- | -------------------- | ---------------------------------------------------- |
| `ClassificationResults` | `probs`          | n/a (single result)  | `cls`, `conf`, `name`, `probabilities`               |
| `DetectionResults`      | `boxes`          | `DetectionResult`    | `cls`, `conf`, `box.xyxy`, `cropped_image`           |
| `SegmentationResults`   | `boxes`, `masks` | `SegmentationResult` | `cls`, `conf`, `box.xyxy`, `mask`, `segmented_image` |

Every envelope also exposes `names`, `orig_img`, `orig_shape`, `path`, and an optional `speed` timings dict.

The bulk views (`Boxes`, `Probs`, `Masks`) match Ultralytics one-to-one: `boxes.xyxy`, `boxes.xywh`, `boxes.xyxyn`, `boxes.xywhn`, `boxes.cls`, `boxes.conf`, `boxes.data`; `probs.top1`, `probs.top5`, `probs.top1conf`, `probs.top5conf`, `probs.data`; `masks.data`, `masks.xyxy`.

Per-instance dataclasses (`DetectionResult`, `SegmentationResult`, `ClassProbability`, `ClassificationResult`) carry the verbose names (`class_id`, `class_name`, `confidence`, `bbox`) as canonical fields and expose Ultralytics aliases (`cls`, `name`, `conf`, `box`) as read-only properties — pick whichever style your codebase already uses.

---

## Async inference

Each task class exposes two async variants of `predict()` that match the sync signature exactly. Pick the one that matches your concurrency profile:

| Method | Mechanism | Use when |
| --- | --- | --- |
| `predict()` | Synchronous | Scripts, notebooks, batch pipelines without an event loop |
| `async_predict()` | `asyncio.to_thread` | Default async path — FastAPI/AnyIO/Quart handlers and other async code. Off-loads the whole pipeline (preprocess + run + postprocess) to the asyncio default executor's thread pool, freeing the event loop. One Python thread per in-flight inference. |
| `ort_async_predict()` | `InferenceSession.run_async` | High-throughput concurrency — many simultaneous awaits should share a single thread pool. Pre-/post-processing run on the event loop thread; the model run is dispatched to the ONNX Runtime internal pool you configured via `SessionOptions.intra_op_num_threads` / `inter_op_num_threads`. Requires `onnxruntime>=1.16`. |

The same split is available on the underlying session — `OrtSession.async_run` / `OrtSession.ort_async_run` — for callers building their own pipelines.

### FastAPI handler (default async)

```python
from fastapi import FastAPI, UploadFile
from ort_vision_sdk import Detector

app = FastAPI()
det = Detector("yolov8n.onnx")

@app.post("/detect")
async def detect(file: UploadFile) -> dict[str, list[dict[str, float | int | str]]]:
    image_bytes = await file.read()
    result = (await det.async_predict(image_bytes))[0]
    return {
        "detections": [
            {"name": d.name, "conf": d.conf, "x1": d.box.x1, "y1": d.box.y1,
             "x2": d.box.x2, "y2": d.box.y2}
            for d in result
        ]
    }
```

### High-concurrency batch (ORT pool)

```python
import asyncio
import onnxruntime as ort
from ort_vision_sdk import Detector

# Configure the ORT thread pool — all in-flight inferences share these threads.
opts = ort.SessionOptions()
opts.intra_op_num_threads = 4
opts.inter_op_num_threads = 1

det = Detector("yolov8n.onnx", session_options=opts)

async def detect_all(paths: list[str]) -> list[list]:
    # Hundreds of concurrent awaits, all sharing the ORT pool of 4 threads —
    # no Python thread spawned per await.
    return await asyncio.gather(*(det.ort_async_predict(p) for p in paths))

results = asyncio.run(detect_all([f"img_{i}.jpg" for i in range(200)]))
```

### Rule of thumb

- **One-off async call inside a request handler** → `async_predict`
- **Hundreds of concurrent inferences** (queue worker, batch endpoint) → `ort_async_predict`

## Common patterns

### Iterate detections only

```python
for d in det.predict("img.jpg")[0]:
    print(d.name, d.conf, d.box.xyxy)
```

### Filter by class

```python
result = det.predict("img.jpg")[0]
people = [d for d in result if d.name == "person"]
```

### Save crops

```python
from PIL import Image
for i, d in enumerate(det.predict("img.jpg")[0]):
    Image.fromarray(d.cropped_image).save(f"crop_{i}.png")
```

### Batch over a folder

```python
from pathlib import Path
for path in Path("images").glob("*.jpg"):
    result = det.predict(path)[0]
    print(path.name, len(result), "detections")
```

### Override per-call thresholds (detector)

```python
result = det.predict("img.jpg", conf_threshold=0.4, iou_threshold=0.5)[0]
```

---

## Status

This project is **alpha** — the public API is stable enough to build against, but minor versions may introduce breaking changes during the pre-1.0 phase. Pin the version range you build against.

- Source code & issues: <https://github.com/mauriciobenjamin700/ort-vision-sdk>
- Changelog: <https://github.com/mauriciobenjamin700/ort-vision-sdk/blob/main/sdk-python/CHANGELOG.md>
- Browser counterpart (TypeScript): [`@mauriciobenjamin700/ort-vision-sdk-web`](https://www.npmjs.com/package/@mauriciobenjamin700/ort-vision-sdk-web)

## License

MIT — see [LICENSE](https://github.com/mauriciobenjamin700/ort-vision-sdk/blob/main/sdk-python/LICENSE).
