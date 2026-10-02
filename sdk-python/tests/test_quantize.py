"""Tests for :func:`ort_vision_sdk.quantize_model`.

The property that matters is that calibration sees what inference feeds. The
tests below pin the observable consequences: the task is read off the model
when not given, each task's quantized model still loads and runs through that
task, and the inputs quantization cannot start from are refused up front
rather than failing deep inside ONNX Runtime's tooling.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

from ort_vision_sdk import (
    QUANTIZATION_KEY,
    Classifier,
    Detector,
    Segmenter,
    optimize_model,
    quantize_model,
)

MODELS = Path(__file__).parent / "fixtures" / "models"


def _images(count: int = 4) -> list[np.ndarray]:
    """Random RGB images at a size no model here uses natively.

    Args:
        count: How many images.

    Returns:
        HWC uint8 arrays, so letterbox and resize both have work to do.
    """
    generator = np.random.default_rng(7)
    return [generator.integers(0, 256, (90, 120, 3), dtype=np.uint8) for _ in range(count)]


def _conv_classifier(path: Path) -> Path:
    """Write a minimal classifier with one convolution, which is quantizable.

    The shared fixtures are built from constants and element-wise ops only, so
    quantization finds nothing to quantize in them: they cover the
    preprocessing path, and this model covers the quantization itself.

    Args:
        path: Where to write the model.

    Returns:
        ``path``.
    """
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    weights = np.random.default_rng(0).normal(0, 0.1, (4, 3, 3, 3)).astype(np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("Conv", ["images", "w"], ["conv"], pads=[1, 1, 1, 1]),
            helper.make_node("GlobalAveragePool", ["conv"], ["pooled"]),
            helper.make_node("Flatten", ["pooled"], ["output0"]),
        ],
        "conv_classifier",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 32, 32])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 4])],
        [numpy_helper.from_array(weights, "w")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    for key, value in {"task": "classify", "names": "{0: 'a', 1: 'b', 2: 'c', 3: 'd'}"}.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.save(model, str(path))
    return path


def test_quantizes_a_convolution_to_qdq(tmp_path: Path) -> None:
    source = _conv_classifier(tmp_path / "conv.onnx")

    output = quantize_model(source, tmp_path / "q.onnx", _images())

    assert b"QuantizeLinear" in output.read_bytes()
    probs = Classifier(output)(_images(1)[0])[0].probs.data
    assert probs.shape == (4,)


@pytest.mark.parametrize(
    ("model", "task_class"),
    [
        ("tiny_detector.onnx", Detector),
        ("tiny_segmenter.onnx", Segmenter),
        ("tiny_classifier.onnx", Classifier),
    ],
)
def test_quantized_model_runs_through_its_task(
    tmp_path: Path, model: str, task_class: type[Detector | Segmenter | Classifier]
) -> None:
    output = quantize_model(MODELS / model, tmp_path / "q.onnx", _images())

    results = task_class(output)(_images(1)[0])
    assert len(results) == 1


def test_pins_the_metadata_key_the_web_sdk_reads() -> None:
    assert QUANTIZATION_KEY == "ort_vision_sdk.quantization"


@pytest.mark.parametrize(
    ("per_channel", "scheme"),
    [(True, "qdq-u8s8-per-channel"), (False, "qdq-u8s8-per-tensor")],
)
def test_marks_the_scheme_in_the_metadata(
    tmp_path: Path, per_channel: bool, scheme: str
) -> None:
    output = quantize_model(
        MODELS / "tiny_detector.onnx", tmp_path / "q.onnx", _images(), per_channel=per_channel
    )

    metadata = ort.InferenceSession(str(output)).get_modelmeta().custom_metadata_map
    assert metadata[QUANTIZATION_KEY] == scheme


def test_keeps_the_source_metadata(tmp_path: Path) -> None:
    output = quantize_model(MODELS / "tiny_detector.onnx", tmp_path / "q.onnx", _images())

    names = ort.InferenceSession(str(output)).get_modelmeta().custom_metadata_map["names"]
    assert names == ort.InferenceSession(
        str(MODELS / "tiny_detector.onnx")
    ).get_modelmeta().custom_metadata_map["names"]


def test_an_explicit_task_covers_a_model_without_metadata(tmp_path: Path) -> None:
    output = quantize_model(
        MODELS / "tiny_detector_no_metadata.onnx", tmp_path / "q.onnx", _images(), task="detect"
    )

    assert output.is_file()


def test_refuses_a_model_whose_task_it_cannot_tell(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Pass task="):
        quantize_model(MODELS / "tiny_detector_no_metadata.onnx", tmp_path / "q.onnx", _images())


def test_refuses_an_unknown_task(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="task must be"):
        quantize_model(
            MODELS / "tiny_detector.onnx",
            tmp_path / "q.onnx",
            _images(),
            task="pose",  # type: ignore[arg-type]
        )


def test_refuses_a_half_precision_model(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="float32 export"):
        quantize_model(MODELS / "tiny_classifier_fp16.onnx", tmp_path / "q.onnx", _images())


def test_refuses_an_empty_calibration_set(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no images"):
        quantize_model(MODELS / "tiny_detector.onnx", tmp_path / "q.onnx", [])


def test_refuses_a_missing_model(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        quantize_model(tmp_path / "missing.onnx", tmp_path / "q.onnx", _images())


def test_optimize_model_warns_on_a_quantized_model(tmp_path: Path) -> None:
    source = _conv_classifier(tmp_path / "conv.onnx")
    quantized = quantize_model(source, tmp_path / "q.onnx", _images())

    with pytest.warns(UserWarning, match="quantized"):
        optimize_model(quantized, tmp_path / "opt.onnx")
