# Installation

`ort-vision-sdk` ships as two independent packages. Install the one that matches
your platform — or both, if you share models between a Python backend and a
browser frontend.

## Python (PyPI)

Requires **Python 3.11+**.

```bash
pip install ort-vision-sdk            # CPU only (default)
pip install "ort-vision-sdk[gpu]"     # adds onnxruntime-gpu (CUDA / TensorRT)
pip install "ort-vision-sdk[opencv]"  # adds the OpenCV image backend
pip install "ort-vision-sdk[compose]" # adds onnx, to fuse models into one pipeline
pip install "ort-vision-sdk[dev]"     # ruff, mypy, pytest, build, twine
```

Base dependencies: `onnxruntime>=1.17.0`, `numpy>=1.24.0`, `pillow>=10.0.0`.

### Extras

| Extra | Adds | Use when |
| --- | --- | --- |
| `gpu` | `onnxruntime-gpu` | NVIDIA GPU inference via CUDA / TensorRT. |
| `opencv` | `opencv-python` | OpenCV image backend (alternative to Pillow). |
| `compose` | `onnx` | Fusing a detector plus a classifier into one `.onnx` — see [Fused pipelines](guia/pipeline.md). |
| `dev` | ruff, mypy, pytest, build, twine | Contributing to the package. |

!!! info "`compose` is build-time only"
    `onnx` (the library that rewrites the protobuf) is needed only to **fuse**
    models, a step you run once alongside your export pipeline. **Running** the
    fused `.onnx` needs nothing beyond the `onnxruntime` already in the base
    install.

!!! warning "CPU vs. GPU"
    `onnxruntime` (CPU) and `onnxruntime-gpu` must not coexist in the same
    environment. To use the GPU, install the `gpu` extra in a clean environment
    (without the CPU `onnxruntime` already present), or uninstall it first.

!!! danger "Available GPU ≠ loadable GPU"
    `onnxruntime.get_available_providers()` answers **"this was compiled into the
    wheel"**, not "this can load". `onnxruntime-gpu` always lists
    `CUDAExecutionProvider`, and still registers CPU when the dynamic loader
    cannot find `libcudnn.so.9`. The result is a deployment that asked for GPU,
    got CPU with no error at all, and only surfaces weeks later on the latency
    bill.

    The case is slipperier than it looks: **importing `torch` first** makes CUDA
    load, because the torch wheel ships cuDNN and loads it into the process. The
    same code works or does not depending on the import order of a library the
    SDK does not even depend on.

    From 0.9.0 the SDK reconciles this: `session.providers` reads back what ORT
    **registered**, and asking for a provider by name and not getting it emits a
    `UserWarning` instead of silence.

### Verify the install

```bash
python -c "from ort_vision_sdk import Classifier, Detector, Segmenter; print('OK')"
```

And when the intent is to run on GPU, confirm **where** it actually ran:

```python
from ort_vision_sdk import OrtSession

session = OrtSession("yolov8n.onnx", providers=["cuda"])

print(session.requested_providers)  # ['CUDAExecutionProvider'] — what was asked for
print(session.providers)            # what ORT actually registered
```

If the second line prints only `['CPUExecutionProvider']`, cuDNN is not
reachable — install it, or point `LD_LIBRARY_PATH` at where it lives.

!!! tip "The tasks too"
    `Detector`, `Classifier` and `Segmenter` build an `OrtSession` underneath and
    expose it as `.session`, so `detector.session.providers` answers the same
    question whenever you have not injected a backend of your own.

## Web (npm)

```bash
npm install @mauriciobenjamin700/ort-vision-sdk-web onnxruntime-web
```

`onnxruntime-web` is a **peer dependency** (accepted range: `>=1.17.0`). You pick
the version and ship the matching `.wasm` files — the SDK does not bundle the
runtime so that you stay in control of the version and the bundle.

!!! tip ".wasm files and WebGPU"
    For WebGPU to actually engage (the default provider order is
    `["webgpu", "wasm"]`), you need a recent ORT-Web build, a Chromium-based
    browser with WebGPU enabled, and a secure context (`https://` or
    `localhost`). Otherwise the runtime silently falls back to WebAssembly.

### Verify the install

```bash
node -e "import('@mauriciobenjamin700/ort-vision-sdk-web').then(m => console.log(Object.keys(m)))"
```

## Next steps

- [Quick start](inicio-rapido.md) — first examples side by side.
- The [classification](guia/classificacao.md), [detection](guia/deteccao.md) and
  [segmentation](guia/segmentacao.md) guides.
- [Fused pipelines](guia/pipeline.md) — two models, one file, one session.
