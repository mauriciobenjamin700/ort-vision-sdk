# @mauriciobenjamin700/ort-vision-sdk-web

[![npm](https://img.shields.io/npm/v/@mauriciobenjamin700/ort-vision-sdk-web.svg)](https://www.npmjs.com/package/@mauriciobenjamin700/ort-vision-sdk-web)
[![GitHub](https://img.shields.io/badge/github-mauriciobenjamin700%2Fort--vision--sdk-181717?logo=github)](https://github.com/mauriciobenjamin700/ort-vision-sdk)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/mauriciobenjamin700/ort-vision-sdk/blob/main/sdk-js-web/LICENSE)

High-level TypeScript SDK for browser computer vision inference on top of [ONNX Runtime Web](https://onnxruntime.ai/docs/get-started/with-javascript/web.html).

Mirrors the Python [`ort-vision-sdk`](https://pypi.org/project/ort-vision-sdk/) API: task-oriented classes (`Classifier`, `Detector`) that handle image loading, preprocessing, execution-provider selection and postprocessing. Output is the same typed shape as the Python version (`ClassificationResult`, `DetectionResult`, `BoundingBox`).

## Installation

```bash
npm install @mauriciobenjamin700/ort-vision-sdk-web onnxruntime-web
```

`onnxruntime-web` is a peer dependency — you bring your own version and ship the matching `.wasm` files yourself.

## Quick start

### Image classification

```typescript
import { Classifier } from "@mauriciobenjamin700/ort-vision-sdk-web";

const clf = await Classifier.create("/models/resnet50.onnx", {
  labels: ["tench", "goldfish", /* ... 1000 ImageNet labels */],
});

const result = await clf.predict("/images/dog.jpg", { topK: 5 });

console.log(result.className, result.confidence);
console.log(result.probabilities);
// result.image is an RGBImage (HWC RGB Uint8Array) — the original input.
```

### Object detection

```typescript
import { Detector } from "@mauriciobenjamin700/ort-vision-sdk-web";

// labels defaults to "coco" (80 classes)
const det = await Detector.create("/models/yolov8n.onnx");

const detections = await det.predict("/images/street.jpg", {
  confThreshold: 0.4,
});

for (const d of detections) {
  console.log(d.className, d.confidence, d.bbox.asXyxy());
  // d.croppedImage is an RGBImage of just that bounding box region.
}
```

## Accepted image inputs

`predict(image)` and `loadImage(image)` both accept:

- `string` — a URL fetched via `fetch()`.
- `Blob` / `File` — for `<input type="file">` uploads.
- `HTMLImageElement` — an existing `<img>` tag.
- `HTMLCanvasElement` / `OffscreenCanvas` — already-rendered canvas.
- `ImageBitmap` — from `createImageBitmap()`.
- `ImageData` — raw pixel buffer (RGBA from canvas `getImageData()`).
- `RGBImage` — the SDK's canonical HWC RGB Uint8Array wrapper.

## Execution providers

The default provider order is `["webgpu", "wasm"]` — ONNX Runtime tries WebGPU first and silently falls back to WebAssembly if WebGPU isn't available. You can override per task:

```typescript
const clf = await Classifier.create(model, {
  labels,
  providers: ["wasm"], // force CPU
});
```

For WebGPU to actually engage you need a recent ORT-Web build, a Chromium-based browser with WebGPU enabled, and either secure context (`https://` or `localhost`) or the right COOP/COEP headers if you also want SharedArrayBuffer-based wasm threading.

## Status

This project is **alpha** — the public API is stable enough to build against, but minor versions may introduce breaking changes during the pre-1.0 phase. Pin the version range you build against.

- Source code & issues: <https://github.com/mauriciobenjamin700/ort-vision-sdk>
- Changelog: <https://github.com/mauriciobenjamin700/ort-vision-sdk/blob/main/sdk-js-web/CHANGELOG.md>
- Python counterpart: [`ort-vision-sdk`](https://pypi.org/project/ort-vision-sdk/)

## License

MIT — see [LICENSE](https://github.com/mauriciobenjamin700/ort-vision-sdk/blob/main/sdk-js-web/LICENSE).
