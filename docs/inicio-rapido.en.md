# Quick start

This guide shows the shortest path to a typed result on both platforms. Every
`predict()` returns a **list of length-1 envelopes per image** — hence the `[0]`
pattern. The envelope is iterable: looping over it yields the detected instances.

## Python

### Classification

```python
from ort_vision_sdk import Classifier

clf = Classifier(
    "resnet50.onnx",
    labels="imagenet_labels.txt",  # one class per line, or a list/dict
    input_size=(224, 224),         # default
    apply_softmax=True,            # False if your model already outputs probs
)

r = clf.predict("dog.jpg")[0]
print(r.cls, r.conf, r.name)       # top-1 — Ultralytics-style
print(r.probs.top5)                # array of the top-5 indices
print(r.probs.top5conf)            # corresponding probabilities
```

### Detection

```python
from ort_vision_sdk import Detector

det = Detector(
    "yolov8n.onnx",
    labels="coco",                 # default — 80-class COCO preset
    conf_threshold=0.25,
    iou_threshold=0.45,
)

result = det.predict("street.jpg")[0]

# Bulk view — Ultralytics' Boxes interface
print(result.boxes.xyxy.shape)     # (N, 4) absolute pixels
print(result.boxes.cls, result.boxes.conf)

# Per-instance dataclasses
for d in result:
    print(d.name, d.conf, d.box.xyxy)
    # d.cropped_image is an HWC uint8 RGB ndarray of the box crop
```

### Segmentation

```python
from ort_vision_sdk import Segmenter

seg = Segmenter("yolov8n-seg.onnx", labels="coco", mask_threshold=0.5)
result = seg.predict("street.jpg")[0]

for inst in result:
    print(inst.name, inst.conf, inst.box.xyxy)
    print(inst.mask.shape)          # (h, w) uint8 ∈ {0, 255}, cropped to the bbox
    print(inst.segmented_image.shape)  # (h, w, 3) RGB with background zeroed
```

### Async inference

Each class exposes `async_predict()` (off-loads via `asyncio.to_thread`, the
default async path) and `ort_async_predict()` (`InferenceSession.run_async`, for
high concurrency). See the [Python guide](guia/python.md) for the difference.

```python
result = (await det.async_predict("street.jpg"))[0]
```

## Web (browser)

The API mirrors Python. Tasks are created with `await Task.create(...)` (loads
the model asynchronously) and `predict()` is always `async`.

### Classification

```typescript
import { Classifier } from "@mauriciobenjamin700/ort-vision-sdk-web";

const clf = await Classifier.create("/models/resnet50.onnx", {
  labels: ["tench", "goldfish", /* ... 1000 ImageNet labels */],
});

const r = (await clf.predict("/images/dog.jpg", { topK: 5 }))[0];
console.log(r.cls, r.conf, r.name);
console.log(r.probs.top5, r.probs.top5conf);
```

### Detection

```typescript
import { Detector } from "@mauriciobenjamin700/ort-vision-sdk-web";

// labels defaults to "coco" (80 classes)
const det = await Detector.create("/models/yolov8n.onnx");

const result = (await det.predict("/images/street.jpg", { confThreshold: 0.4 }))[0];
for (const d of result) {
  console.log(d.className, d.confidence, d.bbox.asXyxy());
  // d.croppedImage is an RGBImage of just that bounding box region
}
```

### Segmentation

```typescript
import { Segmenter } from "@mauriciobenjamin700/ort-vision-sdk-web";

const seg = await Segmenter.create("/models/yolov8n-seg.onnx", { maskThreshold: 0.5 });
const result = (await seg.predict("/images/street.jpg"))[0];
for (const inst of result) {
  console.log(inst.className, inst.confidence, inst.bbox.asXyxy());
  console.log(inst.mask.width, inst.mask.height);
}
```

## Common problems

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: ort_vision_sdk` | Package not installed in the env. Run `pip install ort-vision-sdk`. See [Installation](instalacao.md). |
| `onnxruntime-web` import error in the browser | It's a peer dependency — install it alongside the SDK and serve the matching `.wasm` files. |
| Model file not found / load failure | Check the `.onnx` path. Export one with `yolo export model=yolov8n.pt format=onnx` (see [Installation](instalacao.md)). |
| `predict(...)` seems to return a weird "list" | By design: every `predict` returns a **list of one envelope per image** — take `[0]`. |
| Labels come out as `class_0`, `class_1`, … | The model carried no names and no `labels` was passed. Pass `labels="coco"`, a list, a dict, or a file — see [Python guide](guia/python.md#labels). |
| Inference freezes the `async` server | Don't call sync `predict` in an `async` handler; use `async_predict`. See [Python guide](guia/python.md#async-inference). |

## Recap

- Each task is a class (`Classifier`, `Detector`, `Segmenter`); `predict` returns
  a **list of one envelope** per image — use `[0]`.
- Iterate the envelope for items, or use the bulk views (`.boxes`, `.probs`).
- The API is the same in Python and the browser; only `snake_case` ↔ `camelCase`
  changes.

## Next steps

- Per-task guides: [classification](guia/classificacao.md),
  [detection](guia/deteccao.md), [segmentation](guia/segmentacao.md).
- Platform differences: [Python](guia/python.md) and [Web](guia/web.md).
- Full reference: [Python API](referencia/python.md) and
  [Web API](referencia/web.md).
