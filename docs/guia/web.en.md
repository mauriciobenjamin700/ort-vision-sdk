# Web guide (browser)

Specifics of the TypeScript `@mauriciobenjamin700/ort-vision-sdk-web` package.
The API mirrors Python; the relevant differences are listed below.

## Async creation

In the browser, loading the model is asynchronous — so tasks are created with
`await Task.create(...)` instead of a constructor, and `predict()` is always
`async`:

```typescript
import { Detector } from "@mauriciobenjamin700/ort-vision-sdk-web";

const det = await Detector.create("/models/yolov8n.onnx");
const result = (await det.predict("/images/street.jpg"))[0];
```

Like Python, `predict()` returns a length-1 list (`Promise<DetectionResults[]>`);
use `[0]`. Each task also exposes a `run()` alias (parity with PyTorch's
`nn.Module.__call__`).

## Accepted inputs

`predict(image)` and `loadImage(image)` both accept:

- `string` — a URL fetched via `fetch()`.
- `Blob` / `File` — for `<input type="file">` uploads.
- `HTMLImageElement` — an existing `<img>` tag.
- `HTMLCanvasElement` / `OffscreenCanvas` — an already-rendered canvas.
- `ImageBitmap` — from `createImageBitmap()`.
- `ImageData` — a raw pixel buffer (RGBA from canvas `getImageData()`).
- `RGBImage` — the SDK's canonical HWC RGB `Uint8Array` wrapper.

## Labels

Label resolution mirrors Python via `resolveLabels`:

```typescript
import { Detector, Classifier, COCO_CLASSES } from "@mauriciobenjamin700/ort-vision-sdk-web";

// 1) Built-in preset
const det = await Detector.create("/models/yolov8n.onnx", { labels: "coco" });

// 2) Explicit list
const clf = await Classifier.create("/m.onnx", { labels: ["cat", "dog", "fox"] });

// 3) Sparse dict — gaps become "class_<id>"
const clf2 = await Classifier.create("/m.onnx", { labels: { 0: "cat", 2: "fox" } });

// 4) null — auto-generates "class_0", "class_1", ... (pass numClasses)
const clf3 = await Classifier.create("/m.onnx", { labels: null, numClasses: 1000 });
```

## Execution providers

The default provider order is `["webgpu", "wasm"]` — ONNX Runtime tries WebGPU
first and silently falls back to WebAssembly if WebGPU isn't available. You can
override per task:

```typescript
const clf = await Classifier.create(model, {
  labels,
  providers: ["wasm"], // force CPU
});
```

For WebGPU to actually engage you need a recent ORT-Web build, a Chromium-based
browser with WebGPU enabled, and a secure context (`https://` or `localhost`) —
or the right COOP/COEP headers if you also want `SharedArrayBuffer`-based wasm
threading.

## Results

The result shapes mirror Python:

- `result.boxes` — bulk view (`xyxy`, `xywh`, `xyxyn`, `xywhn`, `cls`, `conf`,
  `data`).
- `result.probs` (classification) — `top1`, `top5`, `top1conf`, `top5conf`,
  `data`.
- `result.masks` (segmentation) — `data`, `xyxy`.
- Iterating the envelope yields per-instance objects with `classId`/`className`/
  `confidence`/`bbox` and the aliases `cls`/`name`/`conf`/`box`. `BoundingBox`
  exposes `asXyxy()` and `asXywh()`.

## See also

- [Web API reference](../referencia/web.md)
- [Python guide](python.md) — the backend counterpart.
