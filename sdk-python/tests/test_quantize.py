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


def _yolo_head_detector(path: Path) -> Path:
    """Write a detector whose output concatenates pixel boxes with sigmoid scores.

    That is the shape of an Ultralytics detection head, ``[1, 4 + nc, N]``, and
    the shape that broke quantization (#65): one ``Concat`` joins boxes in
    ``[0, 640]`` with scores in ``[0, 1]``, so a single int8 scale for the
    whole tensor is ~2.5 per step and every score rounds to 0. A convolution
    feeds both branches so the quantizer has real activations to calibrate.

    Args:
        path: Where to write the model.

    Returns:
        ``path``.
    """
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    weights = np.random.default_rng(1).normal(0, 0.5, (5, 3, 1, 1)).astype(np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("Conv", ["images", "w"], ["conv"]),
            helper.make_node("Reshape", ["conv", "flat_shape"], ["flat"]),
            helper.make_node("Split", ["flat", "split"], ["raw_boxes", "logits"], axis=1),
            helper.make_node("Sigmoid", ["raw_boxes"], ["unit_boxes"]),
            helper.make_node("Mul", ["unit_boxes", "pixels"], ["boxes"]),
            helper.make_node("Sigmoid", ["logits"], ["scores"]),
            helper.make_node("Concat", ["boxes", "scores"], ["output0"], axis=1),
        ],
        "yolo_head_detector",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 32, 32])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 5, 1024])],
        [
            numpy_helper.from_array(weights, "w"),
            numpy_helper.from_array(np.array([1, 5, 1024], dtype=np.int64), "flat_shape"),
            numpy_helper.from_array(np.array([4, 1], dtype=np.int64), "split"),
            numpy_helper.from_array(np.array(640.0, dtype=np.float32), "pixels"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    for key, value in {"task": "detect", "names": "{0: 'a'}", "imgsz": "[32, 32]"}.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.save(model, str(path))
    return path


def test_keeps_detection_scores_next_to_pixel_boxes(tmp_path: Path) -> None:
    source = _yolo_head_detector(tmp_path / "det.onnx")
    feed = {"images": np.random.default_rng(3).random((1, 3, 32, 32), dtype=np.float32)}

    output = quantize_model(source, tmp_path / "q.onnx", _images(8))

    def scores(model: Path) -> np.ndarray:
        session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        return np.asarray(session.run(None, feed)[0])[0, 4]

    expected = scores(source)
    quantized = scores(output)
    assert expected.max() > 0.2
    assert quantized.max() > 0.2
    assert float(np.abs(quantized - expected).mean()) < 0.02
    np.testing.assert_allclose(quantized, expected, atol=0.1)


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
