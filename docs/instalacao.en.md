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
pip install "ort-vision-sdk[dev]"     # ruff, mypy, pytest, build, twine
```

Base dependencies: `onnxruntime>=1.17.0`, `numpy>=1.24.0`, `pillow>=10.0.0`.

### Extras

| Extra | Adds | Use when |
| --- | --- | --- |
| `gpu` | `onnxruntime-gpu` | NVIDIA GPU inference via CUDA / TensorRT. |
| `opencv` | `opencv-python` | OpenCV image backend (alternative to Pillow). |
| `dev` | ruff, mypy, pytest, build, twine | Contributing to the package. |

!!! warning "CPU vs. GPU"
    `onnxruntime` (CPU) and `onnxruntime-gpu` must not coexist in the same
    environment. To use the GPU, install the `gpu` extra in a clean environment
    (without the CPU `onnxruntime` already present), or uninstall it first.

### Verify the install

```bash
python -c "from ort_vision_sdk import Classifier, Detector, Segmenter; print('OK')"
```

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
