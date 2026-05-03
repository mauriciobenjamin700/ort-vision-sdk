"""Tests for the async inference variants on ``OrtSession`` and the tasks.

These tests bypass ``OrtSession.__init__`` (which loads a real ONNX file) by
constructing instances via ``object.__new__`` and wiring a ``MagicMock`` as
the underlying ``ort.InferenceSession``. We verify the async machinery
itself — not the model — so a mock is the right tool.
"""

from __future__ import annotations

import inspect
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from ort_vision_sdk import Classifier, Detector, Segmenter
from ort_vision_sdk.core.exceptions import InferenceError
from ort_vision_sdk.core.session import OrtSession


def _make_session(
    *,
    run_outputs: list[np.ndarray] | None = None,
    run_async_outputs: list[np.ndarray] | None = None,
    run_async_error: str | None = None,
    run_raises: BaseException | None = None,
    run_async_raises: BaseException | None = None,
    has_run_async: bool = True,
) -> OrtSession:
    """Build an ``OrtSession`` with a mocked underlying ORT session.

    Args:
        run_outputs: Outputs returned by the synchronous ``run`` call.
        run_async_outputs: Outputs delivered via the ``run_async`` callback.
        run_async_error: Error string passed to the callback (None for success).
        run_raises: Exception raised synchronously by the underlying ``run``.
        run_async_raises: Exception raised synchronously by ``run_async`` itself.
        has_run_async: If ``False``, the underlying session has no ``run_async``
            attribute (simulates an old ORT version).
    """
    default_outputs: list[np.ndarray] = [np.zeros((1, 1000), dtype=np.float32)]
    sess: OrtSession = object.__new__(OrtSession)
    raw = MagicMock()

    if run_raises is not None:
        raw.run.side_effect = run_raises
    else:
        raw.run.return_value = run_outputs if run_outputs is not None else default_outputs

    if has_run_async:
        if run_async_raises is not None:
            raw.run_async.side_effect = run_async_raises
        else:
            cb_outputs = (
                run_async_outputs if run_async_outputs is not None else default_outputs
            )
            cb_error = run_async_error

            def _stub(
                output_names: list[str] | None,
                feed: dict[str, np.ndarray],
                callback: Any,
                user_data: Any,
                run_options: Any = None,
            ) -> None:
                callback(cb_outputs, user_data, cb_error)

            raw.run_async.side_effect = _stub
    else:
        # Delete the attribute so getattr() returns None — simulates ORT < 1.16.
        del raw.run_async

    sess._session = raw  # type: ignore[attr-defined]
    sess._input_names = ["input"]  # type: ignore[attr-defined]
    sess._output_names = ["output"]  # type: ignore[attr-defined]
    sess._input_shapes = [(1, 3, 224, 224)]  # type: ignore[attr-defined]
    sess.providers = ["CPUExecutionProvider"]
    return sess


class TestAsyncRun:
    """``OrtSession.async_run`` — asyncio.to_thread wrapper around ``run``."""

    async def test_returns_same_as_run(self) -> None:
        expected: list[np.ndarray] = [np.array([[1.0, 2.0, 3.0]], dtype=np.float32)]
        sess = _make_session(run_outputs=expected)
        outputs = await sess.async_run(
            {"input": np.zeros((1, 3, 224, 224), dtype=np.float32)}
        )
        np.testing.assert_array_equal(outputs[0], expected[0])

    async def test_propagates_errors_as_inference_error(self) -> None:
        sess = _make_session(run_raises=RuntimeError("kernel boom"))
        with pytest.raises(InferenceError, match="kernel boom"):
            await sess.async_run(
                {"input": np.zeros((1, 3, 224, 224), dtype=np.float32)}
            )

    async def test_passes_output_names_through(self) -> None:
        sess = _make_session()
        await sess.async_run(
            {"input": np.zeros((1, 3, 224, 224), dtype=np.float32)},
            output_names=["custom_out"],
        )
        passed_names = sess._session.run.call_args.args[0]  # type: ignore[attr-defined]
        assert passed_names == ["custom_out"]


class TestOrtAsyncRun:
    """``OrtSession.ort_async_run`` — wraps ORT's native ``run_async`` callback."""

    async def test_returns_callback_outputs(self) -> None:
        expected: list[np.ndarray] = [np.array([[7.0, 8.0]], dtype=np.float32)]
        sess = _make_session(run_async_outputs=expected)
        outputs = await sess.ort_async_run(
            {"input": np.zeros((1, 3, 224, 224), dtype=np.float32)}
        )
        np.testing.assert_array_equal(outputs[0], expected[0])

    async def test_callback_error_becomes_inference_error(self) -> None:
        sess = _make_session(run_async_error="kernel-side failure")
        with pytest.raises(InferenceError, match="kernel-side failure"):
            await sess.ort_async_run(
                {"input": np.zeros((1, 3, 224, 224), dtype=np.float32)}
            )

    async def test_sync_raise_in_run_async_is_wrapped(self) -> None:
        sess = _make_session(run_async_raises=RuntimeError("dispatch failed"))
        with pytest.raises(InferenceError, match="dispatch failed"):
            await sess.ort_async_run(
                {"input": np.zeros((1, 3, 224, 224), dtype=np.float32)}
            )

    async def test_missing_run_async_raises_clear_error(self) -> None:
        sess = _make_session(has_run_async=False)
        with pytest.raises(InferenceError, match="run_async is not available"):
            await sess.ort_async_run(
                {"input": np.zeros((1, 3, 224, 224), dtype=np.float32)}
            )

    async def test_uses_default_output_names_when_none_passed(self) -> None:
        sess = _make_session()
        await sess.ort_async_run(
            {"input": np.zeros((1, 3, 224, 224), dtype=np.float32)}
        )
        passed_output_names = sess._session.run_async.call_args.args[0]  # type: ignore[attr-defined]
        assert passed_output_names == ["output"]


class TestTaskAsyncSurface:
    """Sanity check: each task class exposes both async variants as coroutines."""

    @pytest.mark.parametrize("cls", [Classifier, Detector, Segmenter])
    def test_async_predict_is_coroutine_function(self, cls: type) -> None:
        assert inspect.iscoroutinefunction(cls.async_predict)

    @pytest.mark.parametrize("cls", [Classifier, Detector, Segmenter])
    def test_ort_async_predict_is_coroutine_function(self, cls: type) -> None:
        assert inspect.iscoroutinefunction(cls.ort_async_predict)

    @pytest.mark.parametrize("cls", [Classifier, Detector, Segmenter])
    def test_predict_signature_matches_async_predict(self, cls: type) -> None:
        """The async variants must accept the exact same kwargs as ``predict``."""
        sync_sig = inspect.signature(cls.predict)
        async_sig = inspect.signature(cls.async_predict)
        ort_async_sig = inspect.signature(cls.ort_async_predict)
        assert sync_sig.parameters == async_sig.parameters
        assert sync_sig.parameters == ort_async_sig.parameters
