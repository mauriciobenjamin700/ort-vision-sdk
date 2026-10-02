"""Tests for :func:`ort_vision_sdk.optimize_model`.

The optimized file has to be a drop-in replacement for its source: same
outputs, same baked-in metadata, plus the mark that tells the web SDK it can
skip ONNX Runtime's optimizer. The mark is appended to the serialized model by
hand rather than through ``onnx``, so these tests read it back through ORT
itself — the reader that has to agree with the writer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

from ort_vision_sdk import GRAPH_OPTIMIZATION_KEY, optimize_model

MODELS = Path(__file__).parent / "fixtures" / "models"
DETECTOR = MODELS / "tiny_detector.onnx"


def _metadata(path: Path) -> dict[str, str]:
    """Read a model's custom metadata through ONNX Runtime.

    Args:
        path: Model file.

    Returns:
        The custom metadata map.
    """
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return dict(session.get_modelmeta().custom_metadata_map)


def _run(path: Path, feed: np.ndarray) -> list[np.ndarray]:
    """Run a model on one feed.

    Args:
        path: Model file.
        feed: Input tensor.

    Returns:
        Every output.
    """
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    outputs: list[np.ndarray] = session.run(None, {session.get_inputs()[0].name: feed})
    return outputs


def test_pins_the_metadata_key_the_web_sdk_reads() -> None:
    assert GRAPH_OPTIMIZATION_KEY == "ort_vision_sdk.graph_optimization"


@pytest.mark.parametrize("level", ["basic", "extended"])
def test_marks_the_level_and_keeps_the_source_metadata(tmp_path: Path, level: str) -> None:
    output = optimize_model(DETECTOR, tmp_path / "opt.onnx", level=level)  # type: ignore[arg-type]

    metadata = _metadata(output)
    assert metadata[GRAPH_OPTIMIZATION_KEY] == level
    assert metadata["names"] == _metadata(DETECTOR)["names"]


def test_optimized_model_computes_the_same_outputs(tmp_path: Path) -> None:
    output = optimize_model(DETECTOR, tmp_path / "opt.onnx")
    feed = np.random.default_rng(0).random((1, 3, 64, 64), dtype=np.float32)

    for before, after in zip(_run(DETECTOR, feed), _run(output, feed), strict=True):
        np.testing.assert_allclose(after, before, rtol=1e-5, atol=1e-6)


def test_returns_the_output_path(tmp_path: Path) -> None:
    target = tmp_path / "opt.onnx"

    assert optimize_model(str(DETECTOR), str(target)) == target


def test_refuses_an_unknown_level(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="'basic' or 'extended'"):
        optimize_model(DETECTOR, tmp_path / "opt.onnx", level="all")  # type: ignore[arg-type]


def test_refuses_a_missing_model(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        optimize_model(tmp_path / "missing.onnx", tmp_path / "opt.onnx")
