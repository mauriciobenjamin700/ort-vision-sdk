# @ort-vision-sdk/web

High-level TypeScript SDK for browser computer vision inference on top of [ONNX Runtime Web](https://onnxruntime.ai/docs/get-started/with-javascript/web.html).

Mirrors the Python `ort-vision-sdk` API: task-oriented classes (`Classifier`, `Detector`) that handle image loading, preprocessing, execution-provider selection and postprocessing. Output is the same typed shape as the Python version (`ClassificationResult`, `DetectionResult`, `BoundingBox`).

## Installation

```bash
npm install @ort-vision-sdk/web onnxruntime-web
```

`onnxruntime-web` is a peer dependency — you bring your own version and ship the matching `.wasm` files yourself.

## Quick start

### Image classification

```typescript
import { Classifier } from "@ort-vision-sdk/web";

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
import { Detector } from "@ort-vision-sdk/web";

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

Alpha — API may change. See [`package.json`](package.json) for supported versions.
