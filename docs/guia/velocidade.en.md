# Measuring inference cost

You have the model running. The next question is always the same: **why did
that take so long?** 🐌

A 2-second `predict()` might be a heavy forward pass — or a 12 MB download
that happened once and never again. The two call for opposite fixes, and
without numbers you cannot tell which one you are looking at.

Every result envelope carries a `speed` breakdown of each stage.

## The basics

```python
from ort_vision_sdk import Detector

det = Detector("yolov8n.onnx", labels="coco")
results = det.predict("street.jpg")

print(results[0].speed)
# {"load": 84.2, "preprocess": 11.7, "inference": 118.9, "postprocess": 6.4}
```

Values are in **milliseconds**. Same thing in the browser:

```typescript
import { Detector } from "@mauriciobenjamin700/ort-vision-sdk-web";

const det = await Detector.create("/models/yolov8n.onnx");
const results = await det.predict("/images/street.jpg");

console.log(results[0].speed);
// { load: 84.2, preprocess: 11.7, inference: 118.9, postprocess: 6.4 }
```

!!! tip "Same shape in both SDKs"
    Python returns a `dict[str, float]`, TypeScript returns a `Speed` object.
    The keys and the measurement boundaries are identical — one dashboard can
    treat both the same way.

## What each stage covers

| Key | Covers |
| --- | --- |
| `load` | Reading/fetching the input and decoding it to `RGBImage`/`ndarray` |
| `preprocess` | Letterbox/resize, normalization and tensor packing |
| `inference` | The ONNX Runtime forward pass |
| `postprocess` | Decoding raw outputs — NMS, mask assembly, top-k |

`preprocess`, `inference` and `postprocess` are exactly the three keys
Ultralytics reports, measured over the same boundaries. `load` is ours:
`predict()` accepts a path, bytes or URL and decodes it internally. Folding
that into `preprocess` would misreport where the cost is — on a first call
with a cold cache, `load` is usually the largest slice of all.

!!! warning "The stages tile the call, not your wall clock"
    The four stages cover the whole `predict()` with no gaps. What they do
    **not** include is `Detector(...)`/`Detector.create(...)` — loading the
    model and building the ORT session happens before, exactly once, and
    never shows up in `speed`.

## Reading the result

```python
speed = results[0].speed
total = sum(speed.values())

for stage, ms in speed.items():
    print(f"{stage:12} {ms:7.1f} ms  {ms / total:5.1%}")
```

```text
load            84.2 ms  38.5%
preprocess      11.7 ms   5.3%
inference      118.9 ms  54.3%
postprocess      6.4 ms   2.9%
```

Three readings that show up in practice:

- **`load` dominates** — you are fetching the image over the network on every
  call. Decode once and pass the ready array/`RGBImage` in.
- **`inference` dominates** — that is the model itself. Quantize, shrink
  `inputSize`, or check that the provider you asked for is the one ORT
  actually resolved (`session.providers`).
- **`postprocess` dominates** — almost always NMS with a `conf_threshold` set
  too low, letting thousands of candidates through.

## Timing your own pipeline alongside

Your app is rarely just `predict()`. If you crop a ROI between a detection and
a classification, that crop costs something too — and each `predict()`'s
`speed` cannot see what happens between them.

`SpeedTimer` is the same piece the tasks use internally, exported so you can
measure with the same boundaries:

```python
from ort_vision_sdk.core import SpeedTimer

timer = SpeedTimer()
image = load_my_image(path)
timer.stage("load")

roi = detector.predict(image)[0]
timer.stage("inference")

crop = crop_to_box(image, roi.boxes.xyxy[0])
timer.stage("postprocess")

print(timer.speed())
```

```typescript
import { SpeedTimer } from "@mauriciobenjamin700/ort-vision-sdk-web";

const timer = new SpeedTimer();
const image = await loadMyImage(url);
timer.stage("load");

const roi = (await detector.predict(image))[0];
timer.stage("inference");

const crop = cropToBox(image, roi.boxes.xyxy);
timer.stage("postprocess");

console.log(timer.speed());
```

Each `stage()` closes the previous interval and credits the time to the name
given — no start/stop pairs to forget. Calling the same name twice
**accumulates**, so you can fold two inference passes into one key.

## Recap

- `results[0].speed` reports `load`, `preprocess`, `inference` and
  `postprocess` in milliseconds, in Python and in the browser. ✅
- The three Ultralytics keys measure the same boundaries; `load` is ours,
  because `predict()` decodes the input internally.
- Loading the model is **not** in `speed` — that is startup cost.
- `SpeedTimer` measures your own pipeline stages under the same rules.
