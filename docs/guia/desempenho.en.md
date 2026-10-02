# Optimizing creation and inference

On the previous page you learned to [measure](velocidade.md) where the time
goes. This page is the next step: **what to do with that number.** 🚀

There are two different costs, and each has its own levers:

- **Creating the task** (`Detector.create(...)` / `Detector(...)`): downloading
  the model and building the ONNX Runtime session. It happens once, but it is
  what the user waits through before seeing anything.
- **Every `predict()`**: decoding the image, preprocessing, running the model,
  decoding the output. It happens on every frame.

!!! info "Every number here was measured"
    Unless stated otherwise, the numbers come from headless Chromium
    (Playwright) on a 12-logical-core machine, WASM backend, with a
    YOLO11n-seg 640×640 (11.7 MB). Each value is the **median of 5 to 12
    runs** after one warm-up run. On a phone the absolute numbers differ; the
    ratios tend to hold.

## Creation: keep the model in the browser

On a page load, the biggest creation cost is almost always **downloading the
model**. It is megabytes, and the server does not always send cache headers
the browser honours for a file that size.

The `cache` option keeps the model in the browser's Cache Storage. From the
second visit on, the bytes come from disk, not the network:

```typescript hl_lines="4"
import { Detector } from "@mauriciobenjamin700/ort-vision-sdk-web";

const det = await Detector.create("/models/yolov8n.v1.onnx", {
  cache: true,
});
const result = (await det.predict("/images/street.jpg"))[0];
for (const d of result) console.log(d.className, d.confidence, d.bbox.asXyxy());
```

`cache: true` uses the `DEFAULT_MODEL_CACHE` bucket (`"ort-vision-sdk-models"`).
A string picks another name: `cache: "my-app-models"`.

!!! warning "The URL is the key, and nothing expires on its own"
    If you publish a new model **at the same URL**, returning visitors keep the
    old one. Version the URL (`yolov8n.v2.onnx`, `?v=2`), change the bucket
    name, or delete the bucket with `caches.delete(name)`.

!!! note "Where there is no Cache Storage, nothing breaks"
    Cache Storage only exists in a secure context (`https://` or `localhost`)
    and may be blocked (private window, site data disabled). In those cases
    the SDK just downloads the model from the network, as if `cache` had not
    been asked for.

## Creation: download the runtime alongside the model

Besides the model, a page's first session downloads and compiles the **ONNX
Runtime runtime**, a `.wasm` of several megabytes. It used to start only after
the model download finished, so the page's two largest downloads ran one after
the other.

The SDK now starts the runtime **while** the model downloads. You change
nothing. Measured with bandwidth capped at 5 MB/s (11.7 MB model, 12.8 MB
runtime), on the page's first creation:

| | First creation |
| --- | --- |
| Before (one after the other) | 5020–5145 ms |
| Now (in parallel) | **2648–2656 ms** |

### Not using WebGPU? Download half the runtime

The default `onnxruntime-web` import brings the runtime **with** WebGPU: 25.9 MB
(6.0 MB gzipped). The `onnxruntime-web/wasm` subpath brings WASM only: 12.8 MB
(3.3 MB gzipped). If your app runs on WASM only, point the import at it. In
Vite, in `vite.config.ts`:

```typescript
import { defineConfig } from "vite";

export default defineConfig({
  resolve: {
    alias: [{ find: /^onnxruntime-web$/, replacement: "onnxruntime-web/wasm" }],
  },
});
```

The SDK uses that build too, since it imports `onnxruntime-web`. Ask for
`providers: ["wasm"]` in the tasks: the WASM-only build has no WebGPU to offer.

## Creation: optimize the graph once, at build time

Every ONNX Runtime session runs the **graph optimizer** before it can infer:
it folds constants, fuses operators, removes redundant nodes. That is most of
the cost of building a session, and it produces the same result every time.

So you can pay it **once, in your build**, with the Python SDK:

```python
from ort_vision_sdk import optimize_model

optimize_model("yolov8n.onnx", "yolov8n.opt.onnx")
```

Ship `yolov8n.opt.onnx` instead of the original. Done! The web SDK reads, in
the file's metadata, the mark `optimize_model` left, and builds the session
**without** running the optimizer again. You change no line in the browser:

```typescript
import { Segmenter } from "@mauriciobenjamin700/ort-vision-sdk-web";

const seg = await Segmenter.create("/models/yolov8n-seg.opt.onnx");
```

| File | Session creation (ORT only) | Full `Segmenter.create` | Inference |
| --- | --- | --- | --- |
| Original | 29.4 ms | 52.6 ms | 65.9 ms |
| `optimize_model(...)` | **12.4 ms** | **33.6 ms** | 65.2 ms |

Inference does not change: the graph is the same one ORT would build on its
own, just built earlier.

!!! tip "Why `"extended"` and not `"all"`"
    The default is `level="extended"`. Beyond that, ORT applies layout
    transformations chosen for the instruction set of the **CPU doing the
    optimizing** (AVX2, AVX-512…), which the browser's WASM does not have. That
    is why `"all"` is not offered at all. `"basic"` exists for constant folding
    only.

??? info "Technical details"
    - The export's metadata (`names`, `imgsz`…) is preserved; the mark is the
      `GRAPH_OPTIMIZATION_KEY` key (`"ort_vision_sdk.graph_optimization"`).
    - An explicit `graphOptimizationLevel` in `sessionOptions` always wins over
      the mark.
    - With `readMetadata: false` the mark is not read, and ORT optimizes as
      usual.
    - `extended` fusions use `com.microsoft` operators that every CPU and WASM
      build implements. On **WebGPU**, some have no kernel: on the
      YOLO11n-seg, ORT placed 2 nodes on the CPU for the optimized file and
      none for the original, which costs GPU↔CPU copies on every inference.
      The SDK warns in the console when a pre-optimized model is headed for
      WebGPU. **For WebGPU, ship the original export.**
    - The Python SDK does not apply the mark on its own: it only reads metadata
      once the session exists. The optimized file runs normally; to skip the
      repeated optimization, pass `SessionOptions` with
      `graph_optimization_level = ORT_DISABLE_ALL`.

## Creation: per-provider options

Each execution provider has its own options, and some change creation and
inference time a lot. Pass an object (web) or a `(name, options)` pair
(Python) instead of the name:

=== "Web"

    ```typescript hl_lines="4"
    import { Detector } from "@mauriciobenjamin700/ort-vision-sdk-web";

    const det = await Detector.create("/models/yolov8n.onnx", {
      providers: [{ name: "webgpu", preferredLayout: "NHWC" }, "wasm"],
    });
    console.log(det.session.providers); // ["webgpu", "wasm"] — always names
    ```

=== "Python"

    ```python hl_lines="6-7"
    from ort_vision_sdk import Detector

    det = Detector(
        "yolov8n.onnx",
        providers=[
            ("tensorrt", {"trt_engine_cache_enable": True, "trt_engine_cache_path": "./trt"}),
            ("cuda", {"cudnn_conv_algo_search": "HEURISTIC"}),
            "cpu",
        ],
    )
    print(det.session.requested_providers)
    # ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    ```

In Python, the name in a pair accepts the usual aliases (`"cuda"`,
`"tensorrt"`…). The options reach ONNX Runtime untouched.

!!! tip "The TensorRT engine cache pays the most"
    Without a cache, TensorRT rebuilds its engine **in every process**, and for
    a large model that takes minutes. With `trt_engine_cache_enable`, only the
    first process pays.

## Inference: quantize to INT8

Quantizing swaps float32 weights and activations for 8-bit integers. The model
gets ~3× smaller, and the CPU runs integer kernels, which are much cheaper. It
is also a build step in the Python SDK, with the `[quantize]` extra:

```python
from pathlib import Path

from ort_vision_sdk import quantize_model

calibration = sorted(Path("calibration/").glob("*.jpg"))
quantize_model("yolov8n-seg.onnx", "yolov8n-seg.int8.onnx", calibration)
```

The calibration images set each activation's int8 range, and go through the
**same preprocessing** as `predict()`: letterbox for detection and
segmentation, resize and normalization for classification. The task comes from
the `task` field of the Ultralytics metadata, or from `task="detect" |
"segment" | "classify"`.

Measured on the YOLO11n-seg (per-channel QDQ, 24 calibration images):

| | FP32 | INT8 |
| --- | --- | --- |
| File | 11.7 MB | **3.5 MB** |
| Inference, native CPU (Python ORT) | 87 ms | **30.5 ms** |
| Full `predict()`, Python SDK | 37.5 ms | **16.8 ms** |
| Inference, WASM (4 threads) | 68.4 ms | **52.7 ms** |

!!! warning "Validate accuracy on your own data"
    Quantization costs some accuracy, and how much depends on the model and on
    how representative the calibration images are. On this model the main
    detection stayed the same (confidence 0.663 → 0.659), but measure on your
    validation set before shipping.

!!! note "In the browser, INT8 runs on WASM"
    ONNX Runtime Web's WebGPU backend cannot run this format's quantized
    operators: its `DequantizeLinear` rejects the quantized bias.
    `quantize_model` marks the file, and the web SDK drops `webgpu` from the
    providers of a marked model, with a console warning. On WASM, which is
    where INT8 pays, it runs normally.

??? info "Why not chain it with `optimize_model`"
    For an INT8 model, ORT applies fusions **at load time** that the offline
    `extended` level does not reproduce. Measured on WASM: the pre-optimized
    INT8 model was created 79 ms faster, but every inference got 19% slower
    (60.8 vs 51.1 ms). `optimize_model` raises a `UserWarning` when given a
    quantized model.

## Inference: WASM threads

The WASM backend uses several threads, but **only** when the page is isolated
(`crossOriginIsolated === true`). Without isolation, it runs on one thread.
Measured on the YOLO11n-seg:

| `env.wasm.numThreads` | Inference |
| --- | --- |
| 1 (page not isolated) | 221 ms |
| 4 (ORT default) | 65 ms |
| **6** | **49 ms** |
| 10 | 58 ms |

Two lessons here:

1. **Isolating the page is worth 3.4×.** It is the biggest win on this whole
   page, and it changes no line of your code: it is two HTTP headers.
2. ORT's default caps at 4 threads. On this machine, **half the cores** was
   best; going past that got worse.

To isolate, serve the page with:

```text
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

And, if you want to tune the threads, do it **before the first session**:

```typescript
import { env } from "onnxruntime-web";
import { Detector } from "@mauriciobenjamin700/ort-vision-sdk-web";

env.wasm.numThreads = Math.max(1, Math.floor(navigator.hardwareConcurrency / 2));

const det = await Detector.create("/models/yolov8n.onnx", { providers: ["wasm"] });
```

!!! warning "Why the SDK does not do this for you"
    `env.wasm` is **global** configuration of `onnxruntime-web`, which belongs
    to your app (it is a peer dependency), and the best number depends on the
    device and on what else runs on the page. The SDK leaves it alone.

!!! note "COEP has a price"
    With `require-corp`, every cross-origin resource (CDN image, font, script)
    must come with `Cross-Origin-Resource-Policy` or CORS. Check before turning
    it on in production.

## Inference: two `predict` calls in flight

With `env.wasm.proxy = true`, inference runs in a worker, and the main thread is
free while it happens. You can put that time to use: while the model infers
frame N, the SDK already decodes and preprocesses frame N+1.

Just keep **two** `predict()` calls going. The SDK queues the runs of one
session (ONNX Runtime Web refuses two at once), so only the model step waits
its turn:

```typescript
import { env } from "onnxruntime-web";
import { Segmenter } from "@mauriciobenjamin700/ort-vision-sdk-web";

env.wasm.proxy = true;

const seg = await Segmenter.create("/models/yolov8n-seg.onnx", { providers: ["wasm"] });
await seg.warmup();

const frames: HTMLCanvasElement[] = [...document.querySelectorAll("canvas")];
let next = 0;
async function worker(): Promise<void> {
  while (next < frames.length) {
    const frame = frames[next++]!;
    const result = (await seg.predict(frame))[0];
    console.log(result.length, "instances");
  }
}
await Promise.all([worker(), worker()]);
```

Measured on a 1080p frame, with the proxy:

| `predict()` in flight | ms per frame |
| --- | --- |
| 1 | 82–86 |
| **2** | **70–73** |
| 3 | 72 |

Three gain nothing more: the model itself becomes the bottleneck. Without
`env.wasm.proxy`, inference holds the main thread and two in flight gain
nothing (84.6 vs 86 ms), but they do not break either.

## Inference: hand the video over directly

In a camera loop, pass the `HTMLVideoElement` (or a `VideoFrame`) straight to
`predict()`, without drawing it onto a canvas first:

```typescript
import { Detector } from "@mauriciobenjamin700/ort-vision-sdk-web";

const video = document.querySelector("video")!;
video.srcObject = await navigator.mediaDevices.getUserMedia({ video: true });
await new Promise((resolve) => video.addEventListener("loadeddata", resolve, { once: true }));
await video.play();

const det = await Detector.create("/models/yolov8n.onnx", { cache: true });
await det.warmup();

async function loop(): Promise<void> {
  const result = (await det.predict(video))[0];
  console.log(result.length, "objects", result.speed);
  requestAnimationFrame(loop);
}
loop();
```

The decoded frame is the one on screen at the time of the call. A video with no
frame yet (`readyState < 2`) throws `ImageLoadError`, explaining that it must
wait for the `loadeddata` event.

With a camera or screen capture (`srcObject` is a `MediaStream`), the SDK does
not even read the frame back from the GPU: it draws it onto a copy and
preprocesses from that. The RGB pixels are only built if something reads
`origImg`, `croppedImage` or `segmentedImage`. Measured on a 1080p frame,
`load` went from 8.5–10.7 ms to **2.0–2.5 ms**. The same holds for an
alpha-less `VideoFrame` (`I420`, `NV12`, `RGBX`…) and for JPEG.

??? info "Why only these sources"
    Preprocessing straight from the copy is identical to the normal path only
    when no pixel is translucent, and usually only reading the pixels tells.
    `MediaStream` frames, alpha-less `VideoFrame`s and JPEGs **cannot** carry
    alpha, so they skip the read. A `<video>` playing a file may have alpha
    (VP9), so it is still read right away.

## What the SDK already does for you

Some optimizations ask nothing of you. They are worth knowing about, because
they explain numbers in `speed`:

- **Preprocessing without an extra copy.** When the input is drawable (image,
  canvas, `ImageBitmap`, video, `Blob`, URL), the pipeline resizes straight
  from the canvas the image was decoded onto. Measured on a 1080p frame,
  `preprocess` went from ~10 ms to ~5 ms. An image with **transparency** takes
  the old path, because compositing its alpha would change the colours.
- **`croppedImage` and `segmentedImage` on demand.** The per-detection crops
  are only built when you read the field. If you only use box and class, they
  never cost anything.
- **Faster postprocessing**, with output bit-identical to before: decode
  25–54% faster, NMS 28–40%, mask assembly 71% on the web; in Python, decode
  43–80%, NMS 78–87% (up to 512 boxes per class) and masks 81%.
- **`warmup()` in both SDKs.** In Python it pays the CUDA arena allocation,
  cuDNN's algorithm search and the TensorRT engine before the first request.

!!! warning "Reusing the same `RGBImage` across frames? Crops are built right away"
    If you pass your own `RGBImage` and rewrite its buffer on every frame, a
    crop built later would read the next frame. So with a caller's `RGBImage`,
    the SDK builds the crops **immediately**. On-demand applies only when the
    SDK allocated the pixels itself.

## Recap

- **Creation:** `cache: true` takes the network out of the way from the second
  visit on; the runtime now downloads alongside the model (−48% on the first
  creation); `onnxruntime-web/wasm` halves the runtime; `optimize_model` at
  build time cuts ~60% of session building (for WASM); per-provider options via
  an object or a `(name, options)` pair. ✅
- **Inference:** `quantize_model` makes the model ~3× smaller and 2.2× faster in
  Python; isolate the page (COOP/COEP) for up to 3.4× on WASM; with
  `env.wasm.proxy`, two `predict()` calls in flight yield ~15% more.
- **Video:** pass the `HTMLVideoElement` straight to `predict()`; with a camera,
  `load` drops ~75%.
- The SDK already avoids copies in preprocessing and builds crops only when
  you ask for them.

## See also

- [Inference cost](velocidade.md): measure before you optimize.
- [Web guide](web.md): `env.wasm.proxy` to move inference off the main thread.
