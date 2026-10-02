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
      build implements. Their coverage on the WebGPU backend was not measured
      here.
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
  25–54% faster, NMS 28–40%, mask assembly 61%.

!!! warning "Reusing the same `RGBImage` across frames? Crops are built right away"
    If you pass your own `RGBImage` and rewrite its buffer on every frame, a
    crop built later would read the next frame. So with a caller's `RGBImage`,
    the SDK builds the crops **immediately**. On-demand applies only when the
    SDK allocated the pixels itself.

## Recap

- **Creation:** `cache: true` takes the network out of the way from the second
  visit on; `optimize_model` at build time cuts ~60% of session building;
  per-provider options (TensorRT cache, WebGPU layout) via an object or a
  `(name, options)` pair. ✅
- **Inference:** isolate the page (COOP/COEP) for up to 3.4× on WASM, and tune
  `env.wasm.numThreads` before the first session.
- **Video:** pass the `HTMLVideoElement` straight to `predict()`.
- The SDK already avoids copies in preprocessing and builds crops only when
  you ask for them.

## See also

- [Inference cost](velocidade.md): measure before you optimize.
- [Web guide](web.md): `env.wasm.proxy` to move inference off the main thread.
