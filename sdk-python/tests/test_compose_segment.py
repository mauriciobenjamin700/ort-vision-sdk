"""Tests for fusing a detector, a per-crop segmenter and a classifier.

Same principle as ``test_compose.py``: every stage is synthetic so the fused
graph's output is predictable by hand. The detector emits a constant head with
boxes at known coordinates, the segmenter marks a known half of each crop as
foreground, and the classifier reduces each crop to its per-channel mean.

That last choice is what makes the mask observable from outside. If the mask
gates the crop, the classifier's output says *which pixels survived* — masking
the bottom half of a uniformly red crop halves the red mean. No shape assertion
could catch a mask that was computed and then quietly ignored; this one does.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from ort_vision_sdk.compose import fuse_detect_segment_classify
from ort_vision_sdk.core.exceptions import FusionError
from ort_vision_sdk.fusion import FUSION_KIND_DETECT_SEGMENT_CLASSIFY, FusionSpec

_IMAGE_SIZE = 64
_CROP_SIZE = 8


def _write_detector(path: Path) -> Path:
    """Write a detector whose head is a constant with two confident boxes.

    One box at ``(8, 8, 24, 24)`` for class 0, one at ``(40, 40, 56, 56)`` for
    class 1, plus two anchors below any sane threshold.

    Args:
        path: Where to write the ``.onnx``.

    Returns:
        Path: ``path``, for chaining.
    """
    head = np.zeros((1, 6, 4), dtype=np.float32)
    head[0, :4, 0] = [16, 16, 16, 16]
    head[0, 4, 0], head[0, 5, 0] = 0.9, 0.1
    head[0, :4, 1] = [48, 48, 16, 16]
    head[0, 4, 1], head[0, 5, 1] = 0.1, 0.8
    head[0, :4, 2] = head[0, :4, 3] = [5, 5, 4, 4]
    head[0, 4, 2] = head[0, 5, 2] = head[0, 4, 3] = head[0, 5, 3] = 0.01

    node = helper.make_node(
        "Constant", [], ["head"], value=numpy_helper.from_array(head, name="head_value")
    )
    graph = helper.make_graph(
        [node],
        "synthetic_detector",
        inputs=[
            helper.make_tensor_value_info(
                "images", TensorProto.FLOAT, [1, 3, _IMAGE_SIZE, _IMAGE_SIZE]
            )
        ],
        outputs=[helper.make_tensor_value_info("head", TensorProto.FLOAT, [1, 6, 4])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    model.metadata_props.add(key="names", value="{0: 'cat', 1: 'dog'}")
    onnx.save(model, str(path))
    return path


def _write_segmenter(path: Path, *, channels: int = 1, top_rows: int = 4) -> Path:
    """Write a segmenter that marks the first ``top_rows`` rows of every crop as foreground.

    The logits are a constant, so the mask does not depend on the crop's
    contents — which is what makes "the mask gated the classifier" separable
    from "the crop happened to be dark there".

    Args:
        path: Where to write the ``.onnx``.
        channels: Output channel count. ``1`` exercises the sigmoid head, more
            than one the softmax head.
        top_rows: How many rows from the top are marked foreground.

    Returns:
        Path: ``path``, for chaining.
    """
    logits = np.full((1, channels, _CROP_SIZE, _CROP_SIZE), -10.0, dtype=np.float32)
    logits[0, 0, :top_rows, :] = 10.0
    if channels > 1:
        logits[0, 1:, :top_rows, :] = -10.0
        logits[0, 1:, top_rows:, :] = 10.0

    nodes = [
        helper.make_node(
            "Constant",
            [],
            ["logits"],
            value=numpy_helper.from_array(logits, name="logits_value"),
        ),
        helper.make_node("Shape", ["input"], ["in_shape"]),
        helper.make_node(
            "Constant",
            [],
            ["batch_start"],
            value=numpy_helper.from_array(np.array([0], dtype=np.int64), name="batch_start_v"),
        ),
        helper.make_node(
            "Constant",
            [],
            ["batch_end"],
            value=numpy_helper.from_array(np.array([1], dtype=np.int64), name="batch_end_v"),
        ),
        helper.make_node("Slice", ["in_shape", "batch_start", "batch_end"], ["batch"]),
        helper.make_node(
            "Constant",
            [],
            ["tail"],
            value=numpy_helper.from_array(
                np.array([channels, _CROP_SIZE, _CROP_SIZE], dtype=np.int64), name="tail_v"
            ),
        ),
        helper.make_node("Concat", ["batch", "tail"], ["target_shape"], axis=0),
        helper.make_node("Expand", ["logits", "target_shape"], ["out"]),
    ]
    graph = helper.make_graph(
        nodes,
        "synthetic_segmenter",
        inputs=[
            helper.make_tensor_value_info(
                "input", TensorProto.FLOAT, ["k", 3, _CROP_SIZE, _CROP_SIZE]
            )
        ],
        outputs=[
            helper.make_tensor_value_info(
                "out", TensorProto.FLOAT, ["k", channels, _CROP_SIZE, _CROP_SIZE]
            )
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.save(model, str(path))
    return path


def _write_classifier(path: Path) -> Path:
    """Write a classifier that reports each crop's per-channel mean.

    Args:
        path: Where to write the ``.onnx``.

    Returns:
        Path: ``path``, for chaining.
    """
    nodes = [
        helper.make_node("GlobalAveragePool", ["input"], ["pooled"]),
        helper.make_node("Flatten", ["pooled"], ["flat"], axis=1),
    ]
    graph = helper.make_graph(
        nodes,
        "synthetic_classifier",
        inputs=[
            helper.make_tensor_value_info(
                "input", TensorProto.FLOAT, [1, 3, _CROP_SIZE, _CROP_SIZE]
            )
        ],
        outputs=[helper.make_tensor_value_info("flat", TensorProto.FLOAT, [1, 3])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    model.metadata_props.add(key="names", value="{0: 'red', 1: 'green', 2: 'blue'}")
    onnx.save(model, str(path))
    return path


def _marked_tensor() -> np.ndarray:
    """An NCHW image whose two box regions are pure red and pure green."""
    image = np.zeros((1, 3, _IMAGE_SIZE, _IMAGE_SIZE), dtype=np.float32)
    image[0, 0, 8:24, 8:24] = 1.0
    image[0, 1, 40:56, 40:56] = 1.0
    return image


@pytest.fixture
def stages(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A synthetic detector/segmenter/classifier trio on disk."""
    return (
        _write_detector(tmp_path / "det.onnx"),
        _write_segmenter(tmp_path / "seg.onnx"),
        _write_classifier(tmp_path / "clf.onnx"),
    )


def _fuse(stages: tuple[Path, Path, Path], tmp_path: Path, **kwargs: object) -> Path:
    """Fuse the trio with identity normalization, so crop means survive unchanged."""
    detector, segmenter, classifier = stages
    output = tmp_path / "fused.onnx"
    defaults: dict[str, object] = {
        "mean": (0.0, 0.0, 0.0),
        "std": (1.0, 1.0, 1.0),
        "segmenter_mean": (0.0, 0.0, 0.0),
        "segmenter_std": (1.0, 1.0, 1.0),
        "sampling_ratio": 1,
        "max_detections": 2,
    }
    defaults.update(kwargs)
    fuse_detect_segment_classify(detector, segmenter, classifier, output, **defaults)  # type: ignore[arg-type]
    return output


def _session(path: Path) -> ort.InferenceSession:
    """Open a fused model on the CPU provider."""
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


class TestFusedGraph:
    def test_declares_the_pipeline_io(self, stages: tuple[Path, Path, Path], tmp_path: Path) -> None:
        session = _session(_fuse(stages, tmp_path))

        assert [i.name for i in session.get_inputs()] == ["images"]
        assert [o.name for o in session.get_outputs()] == [
            "boxes",
            "scores",
            "classes",
            "num_detections",
            "masks",
            "probs",
        ]

    def test_records_the_pipeline_kind_and_mask_settings(
        self, stages: tuple[Path, Path, Path], tmp_path: Path
    ) -> None:
        model = onnx.load(str(_fuse(stages, tmp_path, mask_threshold=0.7, apply_mask=False)))

        spec = FusionSpec.from_metadata({e.key: e.value for e in model.metadata_props})

        assert spec is not None
        assert spec.kind == FUSION_KIND_DETECT_SEGMENT_CLASSIFY
        assert spec.has_masks is True
        assert spec.mask_threshold == pytest.approx(0.7)
        assert spec.mask_applied is False

    def test_masks_carry_the_segmenters_shape(
        self, stages: tuple[Path, Path, Path], tmp_path: Path
    ) -> None:
        session = _session(_fuse(stages, tmp_path))

        masks = session.run(None, {"images": _marked_tensor()})[4]

        assert masks.shape == (2, 1, _CROP_SIZE, _CROP_SIZE)

    def test_the_mask_is_binary_and_marks_the_rows_the_segmenter_chose(
        self, stages: tuple[Path, Path, Path], tmp_path: Path
    ) -> None:
        """The synthetic segmenter marks the top four rows of every crop."""
        session = _session(_fuse(stages, tmp_path))

        masks = session.run(None, {"images": _marked_tensor()})[4]

        assert set(np.unique(masks)) <= {0.0, 1.0}
        assert (masks[0, 0, :4, :] == 1.0).all()
        assert (masks[0, 0, 4:, :] == 0.0).all()


class TestMaskReachesTheClassifier:
    """What the classifier sees is the observable that matters."""

    def test_masking_halves_the_channel_mean_of_a_uniform_crop(
        self, stages: tuple[Path, Path, Path], tmp_path: Path
    ) -> None:
        """The first box is pure red; masking its bottom half halves the red mean.

        Without the mask the classifier reports ~1.0 for red. With the top four
        of eight rows kept, it must report ~0.5 — a number that can only come
        from the mask having actually multiplied the crop.
        """
        on = tmp_path / "on"
        off = tmp_path / "off"
        on.mkdir()
        off.mkdir()
        masked = _session(_fuse(stages, on, apply_mask=True))
        unmasked = _session(_fuse(stages, off, apply_mask=False))
        image = _marked_tensor()

        masked_probs = masked.run(None, {"images": image})[5]
        unmasked_probs = unmasked.run(None, {"images": image})[5]

        assert unmasked_probs[0, 0] == pytest.approx(1.0, abs=1e-3)
        assert masked_probs[0, 0] == pytest.approx(0.5, abs=1e-3)

    def test_the_untouched_channels_stay_zero(
        self, stages: tuple[Path, Path, Path], tmp_path: Path
    ) -> None:
        session = _session(_fuse(stages, tmp_path))

        probs = session.run(None, {"images": _marked_tensor()})[5]

        assert probs[0, 1] == pytest.approx(0.0, abs=1e-6)
        assert probs[0, 2] == pytest.approx(0.0, abs=1e-6)


class TestSegmenterHeads:
    def test_a_multi_channel_head_uses_softmax_and_the_chosen_channel(
        self, tmp_path: Path
    ) -> None:
        """Channel 0 is foreground on the top rows; channel 1 competes below."""
        stages = (
            _write_detector(tmp_path / "det.onnx"),
            _write_segmenter(tmp_path / "seg.onnx", channels=2),
            _write_classifier(tmp_path / "clf.onnx"),
        )
        session = _session(_fuse(stages, tmp_path))

        masks = session.run(None, {"images": _marked_tensor()})[4]

        assert (masks[0, 0, :4, :] == 1.0).all()
        assert (masks[0, 0, 4:, :] == 0.0).all()

    def test_selecting_the_other_channel_inverts_the_mask(self, tmp_path: Path) -> None:
        stages = (
            _write_detector(tmp_path / "det.onnx"),
            _write_segmenter(tmp_path / "seg.onnx", channels=2),
            _write_classifier(tmp_path / "clf.onnx"),
        )
        session = _session(_fuse(stages, tmp_path, mask_channel=1))

        masks = session.run(None, {"images": _marked_tensor()})[4]

        assert (masks[0, 0, :4, :] == 0.0).all()
        assert (masks[0, 0, 4:, :] == 1.0).all()


class TestRejection:
    def test_a_crop_size_disagreement_is_refused(self, tmp_path: Path) -> None:
        """A mask computed at one resolution cannot gate a crop at another."""
        detector = _write_detector(tmp_path / "det.onnx")
        segmenter = _write_segmenter(tmp_path / "seg.onnx")
        classifier = tmp_path / "clf.onnx"
        _write_classifier(classifier)
        model = onnx.load(str(classifier))
        model.graph.input[0].type.tensor_type.shape.dim[2].dim_value = 16
        model.graph.input[0].type.tensor_type.shape.dim[3].dim_value = 16
        onnx.save(model, str(classifier))

        with pytest.raises(FusionError, match="disagree on crop size"):
            fuse_detect_segment_classify(
                detector, segmenter, classifier, tmp_path / "fused.onnx"
            )

    def test_an_impossible_mask_threshold_is_refused(
        self, stages: tuple[Path, Path, Path], tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="mask_threshold"):
            _fuse(stages, tmp_path, mask_threshold=1.0)
