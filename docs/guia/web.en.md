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

## Input resolution

`inputSize` is optional and acts as a fallback: the resolution comes from the
shape the graph declares.

```typescript
const clf = await Classifier.create("/models/classify.onnx", { labels: LABELS });
console.log(clf.inputSize); // [224, 224] — read from the .onnx, not configured
```

!!! danger "Why this matters"
    An Ultralytics `-cls` export comes out at 224×224 and a detector at 640×640.
    Feeding the wrong one makes ORT abort with `Got invalid dimensions for input:
    images ... Got: 640 Expected: 224` — and the number only exists inside the
    file, so no configuration could get it right on its own.

Passing an `inputSize` that contradicts a static graph logs a warning and is
ignored (ORT would reject it anyway). On dynamic-axis models your value stands,
with `[224, 224]`/`[640, 640]` as the last resort. See
[The model decides](modelo.en.md).

```typescript
console.log(clf.session.inputShape); // [1, 3, 224, 224] — null on a dynamic axis
await clf.session.release();         // frees the native session
```

## Labels

**`labels` is optional: without it, the names the model declares are used.**
Ultralytics writes `names` into the `.onnx` metadata, and a list kept by hand
alongside can be reordered by accident — nothing fails, the predictions just
swap classes.

```typescript
import { Detector } from "@mauriciobenjamin700/ort-vision-sdk-web";

const det = await Detector.create("/models/detect.onnx");
console.log(det.labels); // ["ocular-mucosa"] — from the model, not a preset
console.log(det.numClasses); // 1 — inferred from the (B, 4 + nc, N) output shape
```

!!! check "This also fixes an old trap"
    A single-class detector used to **fail** without an explicit `labels`: the
    default was the 80-name COCO preset, which disagreed with the model's class
    count.

!!! note "Where the metadata comes from in the browser"
    `onnxruntime-web` does not expose the model's metadata map — unlike Python's
    `custom_metadata_map`. The SDK reads `metadata_props` out of the `.onnx`
    bytes at load time, which is why it now fetches the model itself when you
    pass a URL (same single download, different fetcher). Pass
    `readMetadata: false` to keep the previous path. A truncated or unexpected
    file yields an empty map, never an error.

!!! warning "Low-memory phones"
    ORT copies the model into its WASM heap and allocates the graph and the
    weights **on top of** that copy. While that happens, the bytes the SDK fetched
    are alive in the JS heap too — a 5 MB `.onnx` costs 5 MB + 5 MB + weights at
    the same instant. The SDK reads the metadata **before** building the session
    precisely so that buffer dies as early as possible (0.5.1 — before that it
    survived the whole build).

    On a device where the numbers still do not add up, ORT gives up with
    `Can't create a session. failed to allocate a buffer of size N`. Two ways out,
    in order: load one model at a time (never two concurrent `create` calls) and
    free what you are not using with `session.release()`; if that is not enough,
    pass `readMetadata: false` **together with explicit `labels`** — then ORT
    fetches the model itself and nothing in the SDK holds the bytes. The input
    size still comes from the graph; only the class names are lost.

Precedence matches Python: what you pass wins, then the model's `names`, then
the preset:

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

## Getting off the main thread (`env.wasm.proxy`)

The WASM backend runs on whichever thread called it. That thread is the main one,
so both session creation and every `predict()` **block the UI** while they run —
and not by a little: measured on a 32-core desktop, one detector + classifier
`warmup()` froze the page for **805 ms**; on a 4-core / 2 GB phone, a single
analysis takes 50 to 103 s.

ONNX Runtime has a flag for this: `env.wasm.proxy`. With it on, ORT creates its
own Web Worker (`onnxruntime-web-proxy-worker`) and forwards create, run and
release over `postMessage`. Same warmup as above: worst frame **18 ms**, zero
frames over 50 ms.

Turn it on **once, before the first session**:

```typescript
import { env } from "onnxruntime-web";
import { Detector } from "@mauriciobenjamin700/ort-vision-sdk-web";

env.wasm.proxy = true; // before the first Detector.create / Classifier.create

const det = await Detector.create("/models/yolov8n.onnx", { providers: ["wasm"] });
const result = (await det.predict("/images/img.jpg"))[0];
for (const d of result) console.log(d.className, d.confidence, d.bbox.asXyxy());
```

!!! warning "Before the first session, not after"
    ORT reads `env.wasm` when it initialises the WASM runtime, which happens
    inside the first `InferenceSession.create`. Setting the flag after that is
    silently ignored — inference goes back to the main thread with nothing to
    say so.

!!! note "The worker makes nothing cheaper"
    It changes *where* the cost is paid, not how much it is. The WASM heap and the
    pthread build's shared-memory reservation simply move threads; a device that
    cannot create the session on the main thread cannot create it in the worker
    either.

??? info "Technical details: why this needed an SDK fix (0.7.1)"
    The proxy posts the input tensors with their `ArrayBuffer`s in the transfer
    list, which **detaches** them on the sender's side. Since `LetterboxPipeline`
    and `ResizePipeline` hold a `Float32Array` and hand the same one out on every
    `run()`, the next predict wrote into a detached buffer — silently
    `length === 0` — and ORT rejected with
    `Tensor's size(1228800) does not match data length(0).` on every other call.
    Since 0.7.1 the buffer is replaced once it has been detached. On earlier
    versions, `env.wasm.proxy` does not work with the built-in tasks.

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
