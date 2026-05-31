# ort-vision-sdk

High-level SDKs for **computer vision inference** on top of the
[ONNX Runtime](https://onnxruntime.ai/). The repository ships two sibling
packages — the same task-oriented API (`Classifier`, `Detector`, `Segmenter`)
and the same typed result shapes — one for Python (servers/scripts) and one for
the browser (TypeScript).

| Package | Registry | Directory | Install |
| --- | --- | --- | --- |
| `ort-vision-sdk` | PyPI | [`sdk-python/`](https://github.com/mauriciobenjamin700/ort-vision-sdk/tree/main/sdk-python) | `pip install ort-vision-sdk` |
| `@mauriciobenjamin700/ort-vision-sdk-web` | npm | [`sdk-js-web/`](https://github.com/mauriciobenjamin700/ort-vision-sdk/tree/main/sdk-js-web) | `npm install @mauriciobenjamin700/ort-vision-sdk-web onnxruntime-web` |

!!! info "Idioma / Language"
    This documentation is bilingual. Use the language selector at the top of the
    page to switch between **Português (BR)** and **English (US)**.

## What it is

Using `onnxruntime` directly forces you to pick and configure execution
providers, letterbox/resize/normalize/`to_chw`/batch your image, decode the model
output (anchor grids, NMS, mask prototypes), map boxes back from the letterboxed
input to the original image, and resolve class indices to human-readable labels —
all repeated per task family.

`ort-vision-sdk` does all of that for you and returns a **typed** result in the
[Ultralytics](https://docs.ultralytics.com/) idiom (`boxes.xyxy`, `cls`, `conf`,
`names`, ...) so existing code ports over with minimal edits. From a raw image
(path, bytes, NumPy array, or PIL image in Python; URL/Blob/canvas/etc. in the
browser) to a typed result in **one call**.

## What's in the box

| Task | Class | Models supported |
| --- | --- | --- |
| Classification | `Classifier` | Any ONNX classifier with output `(1, num_classes)` (torchvision-style) |
| Object detection | `Detector` | Anchor-free YOLO heads: v8, v9, v10, v11, v12, v26 |
| Instance segmentation | `Segmenter` | YOLO-seg heads: v8-seg, v11-seg, v26-seg (+ prototypes) |

All three tasks return the same envelope shape — a list (`list[Results]` in
Python, `Results[]` in Web) of length 1 per image — so you switch tasks without
rewriting the code that consumes the result.

## Quick install

=== "Python"

    ```bash
    pip install ort-vision-sdk            # CPU only (default)
    pip install "ort-vision-sdk[gpu]"     # adds onnxruntime-gpu (CUDA / TensorRT)
    pip install "ort-vision-sdk[opencv]"  # adds the OpenCV image backend
    ```

    Requires Python **3.10+**.

=== "Web (browser)"

    ```bash
    npm install @mauriciobenjamin700/ort-vision-sdk-web onnxruntime-web
    ```

    `onnxruntime-web` is a **peer dependency** — you bring your own version and
    ship the matching `.wasm` files.

## First steps

=== "Python"

    ```python
    from ort_vision_sdk import Detector

    det = Detector("yolov8n.onnx")          # labels="coco" by default
    result = det.predict("street.jpg")[0]   # list[DetectionResults], length 1
    for d in result:
        print(d.name, d.conf, d.box.xyxy)
    ```

=== "Web (browser)"

    ```typescript
    import { Detector } from "@mauriciobenjamin700/ort-vision-sdk-web";

    const det = await Detector.create("/models/yolov8n.onnx");
    const result = (await det.predict("/images/street.jpg"))[0];
    for (const d of result) {
      console.log(d.className, d.confidence, d.bbox.asXyxy());
    }
    ```

Continue with [Installation](instalacao.md) and [Quick start](inicio-rapido.md).

## Status

Alpha — the public API is stable enough to build against, but minor versions may
introduce breaking changes until 1.0. Pin the version range you build against.

- Source code & issues: <https://github.com/mauriciobenjamin700/ort-vision-sdk>
- Python package: <https://pypi.org/project/ort-vision-sdk/>
- Web package: <https://www.npmjs.com/package/@mauriciobenjamin700/ort-vision-sdk-web>

## License

[MIT](https://github.com/mauriciobenjamin700/ort-vision-sdk/blob/main/LICENSE).
