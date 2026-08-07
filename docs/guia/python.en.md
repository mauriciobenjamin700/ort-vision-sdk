# Python guide

Specifics of the Python `ort-vision-sdk` package: accepted inputs, label
resolution, execution providers, and the three inference variants.

## Accepted inputs

Every `predict()` accepts the same set of image inputs:

```python
from pathlib import Path
from PIL import Image
import numpy as np

clf.predict("dog.jpg")                              # str path
clf.predict(Path("dog.jpg"))                        # pathlib
clf.predict(open("dog.jpg", "rb").read())           # raw bytes (PNG, JPEG, ...)
clf.predict(Image.open("dog.jpg"))                  # PIL — converted to RGB
clf.predict(np.zeros((480, 640, 3), dtype=np.uint8))  # HWC uint8 RGB ndarray
```

To load an image once and reuse it, use the same internal loader:

```python
from ort_vision_sdk import load_image
img = load_image("dog.jpg")   # HWC uint8 RGB
clf.predict(img)
```

## Labels

Tasks resolve labels at construction time via `resolve_labels`:

```python
from ort_vision_sdk import Classifier, Detector, COCO_CLASSES, resolve_labels

# 1) Built-in preset (currently: "coco")
det = Detector("yolov8n.onnx", labels="coco")

# 2) Explicit list / tuple
clf = Classifier("model.onnx", labels=["cat", "dog", "fox"])

# 3) Sparse dict — gaps filled with "class_<id>"
clf = Classifier("model.onnx", labels={0: "cat", 2: "fox"})

# 4) File path — one class per line
clf = Classifier("model.onnx", labels="imagenet_labels.txt")

# 5) None (default) — reads the `names` baked into the model metadata; without
#    them, auto-generates "class_0", "class_1", ... (classification) or uses the
#    COCO preset (detection/segmentation)
clf = Classifier("model.onnx", labels=None)
```

`names` on every result is the canonical `dict[int, str]` mapping (mirrors
Ultralytics' `model.names`).

!!! tip "The model already knows its own names"
    An Ultralytics export carries `names` in its metadata, and the SDK uses it
    when you pass no `labels` — so a hand-maintained list cannot be reordered by
    accident and silently swap your predictions between classes. The same goes
    for the input resolution. See [The model decides](modelo.md).

## Execution providers

By default the SDK picks the first available provider in ORT's preference order
— CUDA, CoreML, DirectML, OpenVINO, CPU. **TensorRT is left out of the automatic
order** and has to be asked for by name: the `onnxruntime-gpu` wheel reports it
as available even without the TensorRT libraries installed, and it builds an
engine on first run that can take minutes.
To pin a specific backend, pass `providers=` with short aliases or canonical ORT
names:

```python
det = Detector("yolov8n.onnx", providers=["cuda", "cpu"])
det = Detector("yolov8n.onnx", providers=["tensorrt", "cuda", "cpu"])
det = Detector("yolov8n.onnx", providers=["CUDAExecutionProvider"])  # canonical name
```

Supported aliases: `"cpu"`, `"cuda"`, `"tensorrt"`, `"directml"`, `"coreml"`,
`"openvino"`, `"rocm"`. Anything else is forwarded verbatim to ORT.

For fine-grained control (graph optimization, threading, profiling) pass an
`ort.SessionOptions`:

```python
import onnxruntime as ort

opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
opts.intra_op_num_threads = 4

det = Detector("yolov8n.onnx", session_options=opts)
```

## Async inference

Each class exposes two async variants of `predict()` matching the sync
signature. Pick the one that matches your concurrency profile:

| Method | Mechanism | Use when |
| --- | --- | --- |
| `predict()` | Synchronous | Scripts, notebooks, batch pipelines without an event loop. |
| `async_predict()` | `asyncio.to_thread` | Default async path — FastAPI/AnyIO/Quart handlers. Off-loads the whole pipeline (pre + run + post) to the asyncio default executor's thread pool, freeing the event loop. One Python thread per in-flight inference. |
| `ort_async_predict()` | `InferenceSession.run_async` | High concurrency — many simultaneous awaits share a single thread pool. Pre-/post-processing run on the event-loop thread; the model run is dispatched to the ONNX Runtime internal pool configured via `SessionOptions`. Requires `onnxruntime>=1.16`. |

The same split exists on the underlying session — `OrtSession.async_run` /
`OrtSession.ort_async_run` — for callers building their own pipelines.

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

opts = ort.SessionOptions()
opts.intra_op_num_threads = 4
opts.inter_op_num_threads = 1

det = Detector("yolov8n.onnx", session_options=opts)

async def detect_all(paths: list[str]) -> list[list]:
    return await asyncio.gather(*(det.ort_async_predict(p) for p in paths))

results = asyncio.run(detect_all([f"img_{i}.jpg" for i in range(200)]))
```

### Rule of thumb

- **One-off async call inside a request handler** → `async_predict`.
- **Hundreds of concurrent inferences** (queue worker, batch endpoint) →
  `ort_async_predict`.

## See also

- [Python API reference](../referencia/python.md)
- [Web guide](web.md) — the browser counterpart.
