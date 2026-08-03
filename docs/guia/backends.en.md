# Inference backends

By default `ort-vision-sdk` runs inference with the in-process **ONNX Runtime**
(the `OrtSession` class). But everything else in the SDK — preprocessing,
postprocessing, result decoding — is **pure NumPy**. As of **v0.4.0** you can
swap just the inference engine for your own **backend** while keeping the
pipeline in Python.

!!! tip "Why this matters"
    Some environments have **no `onnxruntime` Python wheel**:

    - in the **browser** (Pyodide/WASM) inference runs on `onnxruntime-web` (JS);
    - on **Android** it runs on the native `onnxruntime-android` AAR.

    In both, the SDK runs normally in Python and only the inference call crosses
    the bridge to the native runtime. 🚀

## The `InferenceBackend` protocol

A backend is any object that satisfies the `InferenceBackend` protocol — it
exposes the model metadata and runs inference over a `{input_name: ndarray}`
mapping:

```python
from typing import Protocol
import numpy as np

class InferenceBackend(Protocol):
    @property
    def input_name(self) -> str: ...
    @property
    def input_shape(self) -> tuple[int | str, ...]: ...
    @property
    def output_names(self) -> list[str]: ...
    @property
    def output_shapes(self) -> list[tuple[int | str, ...]]: ...

    def run(self, feeds: dict[str, np.ndarray], *,
            output_names: list[str] | None = None) -> list[np.ndarray]: ...
    async def async_run(self, feeds, *, output_names=None) -> list[np.ndarray]: ...
    async def ort_async_run(self, feeds, *, output_names=None) -> list[np.ndarray]: ...
```

!!! note
    The default backend, `OrtSession`, already satisfies this protocol. You only
    implement one when you want to run outside the in-process ONNX Runtime.

## Complete example: an echo backend

A minimal, runnable backend that returns a fixed output (handy for tests):

```python
import numpy as np
from ort_vision_sdk import Detector

class EchoBackend:
    """Test backend — always returns the same YOLO output (no detections)."""

    def __init__(self) -> None:
        self._outputs = [np.zeros((1, 84, 8400), dtype=np.float32)]  # 4 + 80 classes

    @property
    def input_name(self) -> str:
        return "images"

    @property
    def input_shape(self) -> tuple[int | str, ...]:
        return (1, 3, 640, 640)

    @property
    def input_names(self) -> list[str]:
        return ["images"]

    @property
    def input_shapes(self) -> list[tuple[int | str, ...]]:
        return [(1, 3, 640, 640)]

    @property
    def output_names(self) -> list[str]:
        return ["output0"]

    @property
    def output_shapes(self) -> list[tuple[int | str, ...]]:
        return [(1, 84, 8400)]

    def run(self, feeds, *, output_names=None):
        return self._outputs

    async def async_run(self, feeds, *, output_names=None):
        return self.run(feeds, output_names=output_names)

    async def ort_async_run(self, feeds, *, output_names=None):
        return self.run(feeds, output_names=output_names)


# Inject the backend — `model_path` is ignored (the backend owns loading).
det = Detector("unused.onnx", backend=EchoBackend())
results = det.predict(np.zeros((480, 640, 3), dtype=np.uint8))
print(len(results[0]))   # 0 detections (zeroed output)
```

Pre/post (letterbox, normalization, NMS, parsing) ran in Python; only `run`
crossed the backend.

!!! info "Real bridge (Android/web)"
    In a bridging backend, `run` serializes the input `ndarray`, sends it to the
    native runtime (AAR / `onnxruntime-web`) and deserializes the output back —
    the metadata (`input_name`, `output_shapes`, ...) comes from the model loaded
    on the other side of the bridge.

## Injecting into any task

`Detector`, `Classifier` and `Segmenter` accept the same argument:

```python
Classifier("m.onnx", backend=my_backend)
Detector("m.onnx", backend=my_backend)
Segmenter("m.onnx", backend=my_backend)
```

!!! warning
    When you pass `backend=`, the `model_path`, `providers` and `session_options`
    arguments are **ignored** — the backend is responsible for loading the model
    and choosing the accelerator.

## Recap

- The SDK separates the **pipeline** (NumPy, always in Python) from the
  **inference engine** (the backend).
- Implement `InferenceBackend` to run where there is no `onnxruntime` wheel
  (browser, Android) or to plug in another runtime.
- Inject via `backend=` on any task; the default stays `OrtSession` (in-process
  ONNX Runtime), **100% backward compatible**.
