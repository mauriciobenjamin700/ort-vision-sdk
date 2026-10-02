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

from ort_vision_sdk.core.exceptions import ProviderNotAvailableError
from ort_vision_sdk.core.providers import resolve_providers
from ort_vision_sdk.core.session import OrtSession

MODEL = Path(__file__).parent / "fixtures" / "models" / "tiny_identity.onnx"


class _Meta:
    """The shape ``get_modelmeta()`` returns, reduced to what the wrapper reads."""

    custom_metadata_map: dict[str, str] = {}


class _Value:
    """An input/output descriptor, reduced to what the wrapper reads."""

    def __init__(self, name: str, shape: list[int], type_: str = "tensor(float)") -> None:
        """Store the declared name, shape and element type.

        Args:
            name: Tensor name.
            shape: Declared dimensions.
            type_: Element type as ONNX Runtime spells it.
        """
        self.name = name
        self.shape = shape
        self.type = type_


class _FallbackSession:
    last: _FallbackSession | None = None

    """An ``InferenceSession`` that registers fewer providers than it was given.

    Deliberately pure Python: it starts no ONNX Runtime session at all. An
    earlier version wrapped a real one to keep its metadata honest, which cost
    nothing in correctness and a great deal in stability — every test left a live
    ORT session with its own thread pool un-released, and the suite began hanging
    intermittently (4 runs in 6) somewhere unrelated. Nothing here needs a
    runtime: the behaviour under test is which list the wrapper reports.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        """Record the requested providers without starting anything.

        Args:
            *args: Positional arguments ORT would take; ignored.
            **kwargs: Keyword arguments ORT would take; ignored.
        """
        self.requested: list[Any] = list(kwargs.get("providers") or [])
        _FallbackSession.last = self

    def get_providers(self) -> list[str]:
        """Report the CPU fallback, whatever was asked for."""
        return ["CPUExecutionProvider"]

    def get_inputs(self) -> list[_Value]:
        """Declare one NCHW image input."""
        return [_Value("images", [1, 3, 64, 64])]

    def get_outputs(self) -> list[_Value]:
        """Declare one output."""
        return [_Value("output0", [1, 3])]

    def get_modelmeta(self) -> _Meta:
        """Report an empty custom-metadata map."""
        return _Meta()


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


class TestProviderSpecs:
    """``(name, options)`` pairs reach ORT intact; names are what the session reports."""

    def test_resolve_expands_the_alias_and_keeps_the_options(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            ort, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        options = {"cudnn_conv_algo_search": "HEURISTIC"}

        resolved = resolve_providers([("cuda", options), "cpu"])

        assert resolved == [("CUDAExecutionProvider", options), "CPUExecutionProvider"]
        assert resolved[0][1] is options

    def test_resolve_refuses_a_pair_naming_a_missing_provider(self) -> None:
        with pytest.raises(ProviderNotAvailableError, match="NoSuchExecutionProvider"):
            resolve_providers([("NoSuchExecutionProvider", {})])

    def test_session_hands_the_pair_to_ort(self, pretend_cuda: None) -> None:
        options = {"device_id": 0}
        with pytest.warns(UserWarning):
            session = OrtSession(MODEL, providers=[("cuda", options)])

        assert _FallbackSession.last is not None
        assert _FallbackSession.last.requested == [("CUDAExecutionProvider", options)]
        assert session.requested_providers == ["CUDAExecutionProvider"]

    def test_a_pair_runs_on_a_real_session(self) -> None:
        session = OrtSession(MODEL, providers=[("cpu", {})])

        assert session.providers == ["CPUExecutionProvider"]
        assert session.requested_providers == ["CPUExecutionProvider"]
