"""Tests for what an :class:`OrtSession` says about where it is running.

``onnxruntime.get_available_providers()`` answers "was this compiled in", not
"can this load". A CUDA build whose cuDNN the dynamic loader cannot find lists
``CUDAExecutionProvider`` as available, accepts it, and then registers CPU —
which is why the session reads its providers back from ORT instead of trusting
the list it asked for.

The drop is forced here with a stub session rather than an environment, since
reproducing it for real needs a broken CUDA install.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import onnxruntime as ort
import pytest

from ort_vision_sdk.core.session import OrtSession

MODEL = Path(__file__).parent / "fixtures" / "models" / "tiny_identity.onnx"

_REAL_SESSION = ort.InferenceSession
"""The genuine class, captured before any test swaps the attribute out."""


class _FallbackSession:
    """An ``InferenceSession`` that quietly registers fewer providers than asked.

    Wraps a real session so every piece of metadata the wrapper reads stays
    truthful, and overrides only :meth:`get_providers` — exactly the divergence
    a provider that fails to load produces.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        """Build the underlying CPU session, ignoring the requested providers.

        Args:
            *args: Positional arguments for ``ort.InferenceSession``.
            **kwargs: Keyword arguments for it; ``providers`` is replaced.
        """
        kwargs["providers"] = ["CPUExecutionProvider"]
        self._inner = _REAL_SESSION(*args, **kwargs)

    def get_providers(self) -> list[str]:
        """Report the fallback ORT actually ended up with."""
        return ["CPUExecutionProvider"]

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        """Forward everything else to the real session.

        Args:
            name: Attribute being looked up.

        Returns:
            The underlying session's attribute.
        """
        return getattr(self._inner, name)


@pytest.fixture
def pretend_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make CUDA look available and make every session fall back to CPU."""
    monkeypatch.setattr(
        ort, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    monkeypatch.setattr(ort, "InferenceSession", _FallbackSession)


class TestProviderReconciliation:
    """``providers`` answers "where is this running", not "what did I ask for"."""

    def test_reports_what_ort_registered(self) -> None:
        session = OrtSession(MODEL, providers=["cpu"])

        assert session.providers == ["CPUExecutionProvider"]
        assert session.requested_providers == ["CPUExecutionProvider"]

    def test_reports_the_fallback_rather_than_the_request(self, pretend_cuda: None) -> None:
        with pytest.warns(UserWarning):
            session = OrtSession(MODEL, providers=["cuda"])

        assert session.requested_providers == ["CUDAExecutionProvider"]
        assert session.providers == ["CPUExecutionProvider"]

    def test_warns_when_an_explicit_provider_is_dropped(self, pretend_cuda: None) -> None:
        with pytest.warns(UserWarning, match="CUDAExecutionProvider"):
            OrtSession(MODEL, providers=["cuda"])

    def test_stays_quiet_when_auto_selection_falls_back(
        self, pretend_cuda: None, recwarn: pytest.WarningsRecorder
    ) -> None:
        """Auto-selection walks a priority list; falling back is what it is for."""
        session = OrtSession(MODEL)

        assert session.providers == ["CPUExecutionProvider"]
        assert not [w for w in recwarn if issubclass(w.category, UserWarning)]
