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

from collections.abc import Sequence
from typing import Any, TypeAlias, overload

from ort_vision_sdk.core.exceptions import ProviderNotAvailableError

ProviderSpec: TypeAlias = str | tuple[str, dict[str, Any]]
"""One execution provider: its name, or a ``(name, options)`` pair.

The pair form is what ``onnxruntime.InferenceSession`` itself accepts, and it
is the only way to reach provider options — a TensorRT engine cache, the CUDA
convolution search strategy, an OpenVINO cache directory::

    providers=[
        ("tensorrt", {"trt_engine_cache_enable": True, "trt_engine_cache_path": "./trt"}),
        ("cuda", {"cudnn_conv_algo_search": "HEURISTIC"}),
        "cpu",
    ]

The name in a pair accepts the same short aliases as a bare name.
"""

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


def provider_name(spec: ProviderSpec) -> str:
    """Return the provider name a spec refers to, as written.

    Args:
        spec: A provider name or a ``(name, options)`` pair.

    Returns:
        The name part of the spec, before alias expansion.
    """
    return spec if isinstance(spec, str) else spec[0]


@overload
def resolve_providers(requested: list[str] | None = None) -> list[str]: ...


@overload
def resolve_providers(requested: Sequence[ProviderSpec]) -> list[ProviderSpec]: ...


def resolve_providers(
    requested: Sequence[ProviderSpec] | None = None,
) -> list[str] | list[ProviderSpec]:
    """Resolve the execution providers to use for an inference session.

    Args:
        requested: Explicit list of providers in preference order. Each entry
            may be a canonical ORT name (``"CUDAExecutionProvider"``), a short
            alias (``"cuda"``, ``"cpu"``, ``"tensorrt"``, ...), or a
            ``(name, options)`` pair whose name is either — see
            :data:`ProviderSpec`. ``None`` (default) auto-selects the best
            available accelerator with CPU as the final fallback.

    Returns:
        Ordered list to pass to ``onnxruntime.InferenceSession``, with every
        name canonical and every options dict carried through unchanged. Always
        non-empty (CPU is always available).

    Raises:
        ProviderNotAvailableError: If any explicitly requested provider is
            not available in this ORT build.
    """
    available = set(available_providers())
    if requested is None:
        ordered: list[ProviderSpec] = [p for p in _PRIORITY if p in available]
        return ordered or ["CPUExecutionProvider"]

    canonical: list[ProviderSpec] = [
        normalize_provider(spec)
        if isinstance(spec, str)
        else (normalize_provider(spec[0]), spec[1])
        for spec in requested
    ]
    missing = [provider_name(p) for p in canonical if provider_name(p) not in available]
    if missing:
        raise ProviderNotAvailableError(
            f"Requested execution provider(s) not available: {missing}. "
            f"Available providers: {sorted(available)}."
        )
    return canonical
