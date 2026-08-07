"""Generate the tiny ONNX models the Python end-to-end tests run against.

The models are committed under ``sdk-python/tests/fixtures/models`` so the test
suite exercises a **real** ``onnxruntime`` session without downloading anything
and without adding ``onnx`` to the package's dependencies. Only this script
needs ``onnx``, and it is meant to run in a throwaway environment:

.. code-block:: bash

    uv run --with onnx --with numpy python scripts/gen_test_models.py

Every model emits its outputs from a ``Constant`` node, so the values are fixed
at export time and the expected predictions can be written down exactly in the
tests. The image input is declared but unused: preprocessing correctness is
covered separately, by asserting that boxes decoded from a non-square image land
where the letterbox geometry says they must.

Re-running the script must be a no-op unless the layout below changes — the
tests hard-code the expected predictions, so any edit here is a deliberate
fixture change that has to be reflected in ``test_e2e_onnx.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "sdk-python" / "tests" / "fixtures" / "models"

OPSET = 13
"""ONNX opset the models target.

``Constant`` and ``Identity`` are the only operators used, and both are far
older than this. Opset 13 pairs with IR version 7, which every
``onnxruntime >= 1.17`` (the package's declared floor) accepts.
"""

IR_VERSION = 7
"""ONNX IR version written into the models.

Pinned rather than inherited from the installed ``onnx`` release, which tracks
the newest IR and would emit files that an older ONNX Runtime refuses to load.
"""


def build_constant_model(
    graph_name: str,
    input_shape: tuple[int, ...],
    outputs: list[tuple[str, np.ndarray]],
    metadata: dict[str, str],
) -> onnx.ModelProto:
    """Build a model whose outputs are fixed tensors, ignoring its input.

    Each output is produced by a ``Constant`` node followed by an ``Identity``
    node. The ``Identity`` is not redundant: an ONNX graph output has to be
    produced by a node, and routing a folded constant straight to a graph output
    is rejected by some runtimes.

    Args:
        graph_name: Name recorded in the ONNX graph.
        input_shape: Declared NCHW shape of the (unused) ``images`` input.
        outputs: ``(name, values)`` pairs. Values are cast to ``float32``.
        metadata: Custom metadata map to bake in — ``names``, ``task``, ``imgsz``
            for an Ultralytics-style export.

    Returns:
        The assembled model, already validated by ``onnx.checker``.
    """
    nodes = []
    graph_outputs = []
    for name, values in outputs:
        array = np.ascontiguousarray(values, dtype=np.float32)
        nodes.append(
            helper.make_node(
                "Constant",
                inputs=[],
                outputs=[f"{name}_const"],
                value=numpy_helper.from_array(array, name=f"{name}_value"),
            )
        )
        nodes.append(helper.make_node("Identity", inputs=[f"{name}_const"], outputs=[name]))
        graph_outputs.append(
            helper.make_tensor_value_info(name, TensorProto.FLOAT, list(array.shape))
        )

    graph = helper.make_graph(
        nodes,
        graph_name,
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, list(input_shape))],
        graph_outputs,
    )
    model = helper.make_model(
        graph,
        producer_name="ort-vision-sdk-tests",
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.ir_version = IR_VERSION
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.checker.check_model(model)
    return model


def build_identity_model() -> onnx.ModelProto:
    """Build a model that echoes its input, for session-level round-trip tests.

    Used to prove that ``OrtSession.run``, ``async_run`` and ``ort_async_run``
    all reach real ONNX Runtime and return the tensor they were fed — the
    async paths are otherwise only ever exercised against fakes.

    Returns:
        The assembled model, already validated by ``onnx.checker``.
    """
    shape = [1, 3, 4, 4]
    graph = helper.make_graph(
        [helper.make_node("Identity", inputs=["images"], outputs=["echo"])],
        "tiny_identity",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("echo", TensorProto.FLOAT, shape)],
    )
    model = helper.make_model(
        graph,
        producer_name="ort-vision-sdk-tests",
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.ir_version = IR_VERSION
    onnx.checker.check_model(model)
    return model


def detector_output() -> np.ndarray:
    """Per-anchor detection tensor with four hand-placed anchors.

    Layout is the anchor-free YOLO one — ``(1, 4 + num_classes, num_anchors)``
    with 3 classes and 20 anchors. Boxes are ``(cx, cy, w, h)`` in the 64x64
    input-tensor space:

    - anchor 0: ``(16, 16, 48, 48)`` xyxy, class 0, score 0.90 — survives.
    - anchor 1: ``(17, 17, 49, 49)`` xyxy, class 0, score 0.85 — IoU 0.88 with
      anchor 0, so per-class NMS drops it.
    - anchor 2: ``(6, 28, 14, 36)`` xyxy, class 1, score 0.70 — survives, and
      sits inside the content band of a letterboxed 128x64 image.
    - anchor 3: same box as anchor 0 but class 2, score 0.80 — survives, since
      NMS is per class.

    Remaining anchors are all-zero, so their class scores fall below any usable
    confidence threshold.

    Returns:
        A ``(1, 7, 20)`` ``float32`` array.
    """
    out = np.zeros((1, 7, 20), dtype=np.float32)
    for anchor, (cx, cy, w, h, cls, score) in enumerate(
        [
            (32.0, 32.0, 32.0, 32.0, 0, 0.90),
            (33.0, 33.0, 32.0, 32.0, 0, 0.85),
            (10.0, 32.0, 8.0, 8.0, 1, 0.70),
            (32.0, 32.0, 32.0, 32.0, 2, 0.80),
        ]
    ):
        out[0, 0, anchor] = cx
        out[0, 1, anchor] = cy
        out[0, 2, anchor] = w
        out[0, 3, anchor] = h
        out[0, 4 + cls, anchor] = score
    return out


def segmenter_outputs() -> tuple[np.ndarray, np.ndarray]:
    """Per-anchor and prototype tensors for a single full-image instance.

    Channel count is ``4 + 3 classes + 32 mask coefficients = 39``, with 40
    anchors — the anchor axis has to stay above the channel axis for the
    segmenter's ``num_classes`` inference, which picks the smallest static
    dimension as the channel count.

    Anchor 0 covers the whole 64x64 image with class 1 at score 0.90 and mask
    coefficient 0 set to 1.0. Prototype 0 is ``+5`` over its top half and ``-5``
    over its bottom half, so the instance mask is white on top and black on
    bottom regardless of how the soft mask is resized.

    Returns:
        ``(per_anchor, prototypes)`` — a ``(1, 39, 40)`` and a ``(1, 32, 8, 8)``
        ``float32`` array.
    """
    num_classes = 3
    num_mask_coefs = 32
    mask_h = mask_w = 8

    per_anchor = np.zeros((1, 4 + num_classes + num_mask_coefs, 40), dtype=np.float32)
    per_anchor[0, 0, 0] = 32.0
    per_anchor[0, 1, 0] = 32.0
    per_anchor[0, 2, 0] = 64.0
    per_anchor[0, 3, 0] = 64.0
    per_anchor[0, 4 + 1, 0] = 0.90
    per_anchor[0, 4 + num_classes + 0, 0] = 1.0

    prototypes = np.zeros((1, num_mask_coefs, mask_h, mask_w), dtype=np.float32)
    prototypes[0, 0, : mask_h // 2, :] = 5.0
    prototypes[0, 0, mask_h // 2 :, :] = -5.0
    return per_anchor, prototypes


def main() -> None:
    """Write every fixture model to :data:`OUTPUT_DIR`."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    per_anchor, prototypes = segmenter_outputs()
    models: dict[str, onnx.ModelProto] = {
        "tiny_detector.onnx": build_constant_model(
            "tiny_detector",
            (1, 3, 64, 64),
            [("output0", detector_output())],
            {
                "names": repr({0: "cat", 1: "dog", 2: "bird"}),
                "task": "detect",
                "imgsz": "[64, 64]",
            },
        ),
        "tiny_detector_no_metadata.onnx": build_constant_model(
            "tiny_detector_no_metadata",
            (1, 3, 64, 64),
            [("output0", detector_output())],
            {},
        ),
        "tiny_classifier.onnx": build_constant_model(
            "tiny_classifier",
            (1, 3, 32, 32),
            [("logits", np.array([[1.0, 3.0, 0.5, 2.0]], dtype=np.float32))],
            {
                "names": repr({0: "ant", 1: "bee", 2: "cow", 3: "doe"}),
                "task": "classify",
            },
        ),
        "tiny_segmenter.onnx": build_constant_model(
            "tiny_segmenter",
            (1, 3, 64, 64),
            [("output0", per_anchor), ("output1", prototypes)],
            {"names": repr({0: "leaf", 1: "stem", 2: "root"}), "task": "segment"},
        ),
        "tiny_identity.onnx": build_identity_model(),
    }

    for filename, model in models.items():
        destination = OUTPUT_DIR / filename
        destination.write_bytes(model.SerializeToString())
        print(f"wrote {filename} ({destination.stat().st_size} B)")
    print(f"into {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
