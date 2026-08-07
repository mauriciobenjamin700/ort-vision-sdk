"""Execution-provider resolution for ONNX Runtime sessions.

Encapsulates the logic of picking the best available accelerator (CUDA,
TensorRT, CoreML, DirectML, OpenVINO) and falling back to CPU. Callers can
pass an explicit list of providers when they need full control.

Short Ultralytics-style aliases (``"cpu"``, ``"cuda"``, ``"tensorrt"``,
``"coreml"``, ``"dml"`` / ``"directml"``, ``"openvino"``) are accepted in
addition to the canonical ORT names (``"CPUExecutionProvider"``,
``"CUDAExecutionProvider"``, ...).
"""

from __future__ import annotations

from ort_vision_sdk.core.exceptions import ProviderNotAvailableError

_PRIORITY: tuple[str, ...] = (
    "CUDAExecutionProvider",
    "CoreMLExecutionProvider",
    "DmlExecutionProvider",
    "OpenVINOExecutionProvider",
    "CPUExecutionProvider",
)
"""Default preference order, from most to least accelerated.

TensorRT is deliberately absent and has to be asked for by name. Two reasons,
either of which would be enough on its own:

- ``onnxruntime-gpu`` lists ``TensorrtExecutionProvider`` as available whenever
  it was *compiled in*, not when it can actually load. On a machine without the
  TensorRT shared libraries — the common case for that wheel — auto-selecting it
  made every session print a failed provider registration and a fallback notice
  to stderr before recovering. Alarming output for a session that was always
  going to run on CUDA.
- Where it does load, TensorRT builds an engine on the first run, which can take
  minutes for a large model. That is not a cost to opt somebody into silently.

``providers=["tensorrt"]`` still selects it, and still raises
:class:`~ort_vision_sdk.core.exceptions.ProviderNotAvailableError` when the
installed build does not carry it.
"""

_ALIASES: dict[str, str] = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "gpu": "CUDAExecutionProvider",
    "tensorrt": "TensorrtExecutionProvider",
    "trt": "TensorrtExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "mps": "CoreMLExecutionProvider",
    "dml": "DmlExecutionProvider",
    "directml": "DmlExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
}
"""Short device aliases → canonical ORT provider names. Lookup is case-insensitive."""


def available_providers() -> list[str]:
    """Return the execution providers available in this ORT build.

    Returns:
        List of provider names exactly as ONNX Runtime reports them.
    """
    import onnxruntime as ort

    return list(ort.get_available_providers())


def normalize_provider(name: str) -> str:
    """Expand a short device alias to its canonical ORT provider name.

    Names that already end in ``ExecutionProvider`` are returned unchanged
    (case-preserving). Short aliases are looked up case-insensitively.

    Args:
        name: Either a short alias (``"cpu"``, ``"cuda"``, ``"tensorrt"``,
            ``"coreml"``, ``"dml"``, ``"openvino"``, ...) or a canonical ORT
            provider name (``"CPUExecutionProvider"`` etc.).

    Returns:
        The canonical ORT provider name. If ``name`` is already canonical,
        it is returned as-is.
    """
    if name.endswith("ExecutionProvider"):
        return name
    return _ALIASES.get(name.lower(), name)


def resolve_providers(requested: list[str] | None = None) -> list[str]:
    """Resolve the execution providers to use for an inference session.

    Args:
        requested: Explicit list of providers in preference order. Each entry
            may be a canonical ORT name (``"CUDAExecutionProvider"``) or a
            short alias (``"cuda"``, ``"cpu"``, ``"tensorrt"``, ...). ``None``
            (default) auto-selects the best available accelerator with CPU
            as the final fallback.

    Returns:
        Ordered list of canonical providers to pass to
        ``onnxruntime.InferenceSession``. Always non-empty (CPU is always
        available).

    Raises:
        ProviderNotAvailableError: If any explicitly requested provider is
            not available in this ORT build.
    """
    available = set(available_providers())
    if requested is None:
        ordered = [p for p in _PRIORITY if p in available]
        return ordered or ["CPUExecutionProvider"]

    canonical = [normalize_provider(p) for p in requested]
    missing = [p for p in canonical if p not in available]
    if missing:
        raise ProviderNotAvailableError(
            f"Requested execution provider(s) not available: {missing}. "
            f"Available providers: {sorted(available)}."
        )
    return canonical
