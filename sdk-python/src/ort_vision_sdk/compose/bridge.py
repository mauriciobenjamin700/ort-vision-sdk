"""The ONNX subgraph that turns detections into classifier input.

Everything the SDK normally does in Python between a detector and a classifier
— decode the head, run NMS, pick the surviving boxes, crop them out of the
image, resize each crop, normalize it — is expressed here as ONNX operators, so
it can live *inside* the model file instead of around it. That is what makes a
detector and a classifier collapse into one graph, one session and one load.

The op mapping is the interesting part:

- **NMS** is ``NonMaxSuppression`` with ``center_point_box=1``, which consumes
  the YOLO head's ``(cx, cy, w, h)`` directly — no conversion node. It runs on
  letterboxed coordinates, and that is exact rather than approximate: IoU is
  invariant under the uniform scale and translation a letterbox applies, so
  suppressing in letterbox space and in original-image space keep the same
  boxes.
- **Crop + resize** is ``RoiAlign``. Nothing in ONNX crops a batch of regions
  and resamples each to a fixed size except RoiAlign, which is exactly that
  operation once ``spatial_scale`` is 1.0 and the "feature map" it samples is
  the image tensor itself.
- **Padding to a fixed row count** is ``TopK`` (rank by confidence, cap at
  ``K``) followed by ``Pad``. This is what keeps every shape in the graph
  static, which in turn is what lets TensorRT/NNAPI/WebGPU compile it, and what
  removes the zero-detection edge case — an empty batch never reaches the
  classifier because the batch is always ``K``.

One deliberate behavioural difference from
:func:`~ort_vision_sdk.postprocess.detection.decode_yolo`: the Python decoder
collapses each anchor to its ``argmax`` class before suppressing, while ONNX's
NMS scores every class independently. An anchor whose scores clear the
threshold for two classes therefore yields two rows here and one row there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ort_vision_sdk.fusion import (
    INPUT_PAD,
    INPUT_SCALE,
    INPUT_SOURCE,
    OUTPUT_BOXES,
    OUTPUT_CLASSES,
    OUTPUT_NUM_DETECTIONS,
    OUTPUT_SCORES,
    CropSource,
)

__all__ = ["MIN_OPSET", "BridgeGraph", "build_bridge"]

MIN_OPSET = 16
"""Lowest ONNX opset the bridge can be emitted at.

Set by ``RoiAlign``, which only gained ``coordinate_transformation_mode`` in
opset 16. Below that the operator silently uses the ``output_half_pixel``
convention, which offsets every crop by half a pixel relative to the crop a
caller would take in NumPy — a misalignment that shows up as slightly worse
classifier accuracy rather than as an error, so it is refused instead of
worked around.
"""

_INT64_MAX = 9223372036854775807


@dataclass
class BridgeGraph:
    """The pieces a bridge contributes to the fused graph.

    Attributes:
        nodes: Bridge nodes, already in topological order.
        initializers: Constant tensors the nodes reference.
        inputs: Extra graph inputs the bridge needs — empty for
            ``crop_source="detector_input"``, and the full-resolution image plus
            its letterbox parameters for ``crop_source="original"``.
        outputs: The detection outputs the fused model exposes (boxes, scores,
            classes, num_detections). The classifier's own output is added by
            the caller, since it comes from the classifier graph.
    """

    nodes: list[onnx.NodeProto] = field(default_factory=list)
    initializers: list[onnx.TensorProto] = field(default_factory=list)
    inputs: list[onnx.ValueInfoProto] = field(default_factory=list)
    outputs: list[onnx.ValueInfoProto] = field(default_factory=list)


class _Builder:
    """Accumulates nodes and initializers under a collision-free name prefix.

    The bridge is spliced between two graphs whose tensor names it does not
    control, so every name it mints is prefixed. Names are also numbered per
    operator type, which keeps a dumped graph readable (``bridge_Slice_0``,
    ``bridge_Slice_1``) without the caller having to name intermediates.
    """

    def __init__(self, prefix: str) -> None:
        """Initialize an empty builder.

        Args:
            prefix: String prepended to every node, tensor and initializer name.
        """
        self.prefix: str = prefix
        self.nodes: list[onnx.NodeProto] = []
        self.initializers: list[onnx.TensorProto] = []
        self._counters: dict[str, int] = {}

    def mint(self, base: str) -> str:
        """Return a fresh prefixed name for ``base``.

        Args:
            base: Human-readable stem.

        Returns:
            str: A name unique within this builder.
        """
        index = self._counters.get(base, 0)
        self._counters[base] = index + 1
        return f"{self.prefix}{base}_{index}"

    def const(self, base: str, value: np.ndarray) -> str:
        """Register a constant tensor and return the name it is bound to.

        Args:
            base: Human-readable stem for the generated name.
            value: The constant's value; its dtype and shape are used as-is.

        Returns:
            str: Name of the initializer.
        """
        name = self.mint(f"const_{base}")
        self.initializers.append(numpy_helper.from_array(value, name=name))
        return name

    def op(
        self,
        op_type: str,
        inputs: list[str],
        *,
        outputs: list[str] | None = None,
        **attributes: Any,  # noqa: ANN401
    ) -> str:
        """Append a single-output node and return its output name.

        Args:
            op_type: ONNX operator type.
            inputs: Input tensor names, in operator order.
            outputs: Explicit output names. ``None`` (default) mints one.
            **attributes: Operator attributes, passed to
                :func:`onnx.helper.make_node`, which types them as ``Any``.

        Returns:
            str: Name of the node's first output.
        """
        names = outputs if outputs is not None else [self.mint(op_type)]
        self.nodes.append(
            helper.make_node(
                op_type,
                inputs=inputs,
                outputs=names,
                name=self.mint(f"node_{op_type}"),
                **attributes,
            )
        )
        return names[0]

    def op_multi(
        self,
        op_type: str,
        inputs: list[str],
        *,
        count: int,
        **attributes: Any,  # noqa: ANN401
    ) -> list[str]:
        """Append a multi-output node and return every output name.

        Args:
            op_type: ONNX operator type.
            inputs: Input tensor names, in operator order.
            count: How many outputs the node produces.
            **attributes: Operator attributes.

        Returns:
            list[str]: The node's output names, in order.
        """
        names = [self.mint(f"{op_type}_out") for _ in range(count)]
        self.nodes.append(
            helper.make_node(
                op_type,
                inputs=inputs,
                outputs=names,
                name=self.mint(f"node_{op_type}"),
                **attributes,
            )
        )
        return names


def _i64(*values: int) -> np.ndarray:
    """Build a 1-D int64 array — the type every shape and index input wants."""
    return np.asarray(values, dtype=np.int64)


def _f32(value: float) -> np.ndarray:
    """Build a 1-element float32 array, the shape ORT accepts for scalar inputs."""
    return np.asarray([value], dtype=np.float32)


def build_bridge(
    *,
    detector_output: str,
    detector_input: str,
    classifier_input: str,
    crop_size: tuple[int, int],
    channels: int,
    crop_source: CropSource,
    max_detections: int | None,
    conf_threshold: float,
    iou_threshold: float,
    max_boxes_per_class: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    input_scale: float,
    sampling_ratio: int,
    prefix: str = "ovs_bridge_",
) -> BridgeGraph:
    """Build the subgraph that joins a detector's output to a classifier's input.

    The resulting nodes read the detector's raw head, suppress overlapping
    boxes, rank what survives, crop each box out of an image tensor, resample it
    to the classifier's resolution and normalize it — emitting a tensor named
    ``classifier_input`` plus the four detection outputs the runtime needs to
    interpret the classifier's rows.

    Args:
        detector_output: Name of the detector's raw output tensor, expected to
            carry the anchor-free YOLO layout ``(1, 4 + nc, N)``.
        detector_input: Name of the detector's image input tensor. Cropped from
            directly when ``crop_source`` is ``"detector_input"``.
        classifier_input: Name the classifier graph expects its input under —
            the bridge binds its final tensor to it.
        crop_size: ``(width, height)`` every crop is resampled to.
        channels: Channel count of the image tensor (3 for RGB). Needed to
            broadcast the normalization constants and to pin the crop shape.
        crop_source: Which tensor to crop from — see
            :data:`~ort_vision_sdk.fusion.CropSource`.
        max_detections: Fixed row count ``K`` for every output, surplus rows
            zero-padded. ``None`` emits exactly the surviving rows, leaving
            every downstream shape dynamic.
        conf_threshold: Score threshold handed to NMS.
        iou_threshold: IoU threshold handed to NMS.
        max_boxes_per_class: Cap NMS applies **per class** before the global
            ranking. Kept separate from ``max_detections`` because a per-class
            cap equal to the global one would let a crowded class starve the
            others out of the ranking.
        mean: Per-channel mean subtracted from each crop, in the crop's own
            scale (i.e. after ``input_scale``).
        std: Per-channel standard deviation each crop is divided by.
        input_scale: Multiplier applied to the crop before ``mean``/``std``. The
            SDK feeds images as ``float32`` in ``[0, 1]``, so this is 1.0 for a
            torchvision-style classifier and 255.0 for one that expects raw
            ``0..255`` values.
        sampling_ratio: RoiAlign samples per output bin. ``0`` adapts the count
            to the box size, which anti-aliases when a large box is downscaled
            into a small crop; ``1`` is a plain bilinear resample.
        prefix: Name prefix for every tensor and node the bridge mints.

    Returns:
        BridgeGraph: Nodes, initializers, extra inputs and detection outputs.

    Raises:
        ValueError: If ``max_detections`` or ``max_boxes_per_class`` is not
            positive, if ``crop_size`` or ``channels`` are not positive, or if
            any entry of ``std`` is zero.
    """
    if max_detections is not None and max_detections < 1:
        raise ValueError(f"max_detections must be >= 1 or None, got {max_detections}.")
    if max_boxes_per_class < 1:
        raise ValueError(f"max_boxes_per_class must be >= 1, got {max_boxes_per_class}.")
    if crop_size[0] < 1 or crop_size[1] < 1:
        raise ValueError(f"crop_size must be positive, got {crop_size}.")
    if channels < 1:
        raise ValueError(f"channels must be >= 1, got {channels}.")
    if any(value == 0.0 for value in std):
        raise ValueError(f"std entries must be non-zero, got {std}.")

    b = _Builder(prefix)
    bridge = BridgeGraph()
    crop_width, crop_height = crop_size

    boxes_xyxy, scores, classes, num_detections = _select_boxes(
        b,
        detector_output=detector_output,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        max_boxes_per_class=max_boxes_per_class,
        max_detections=max_detections,
    )

    if crop_source == "original":
        bridge.inputs.extend(_source_inputs(channels))
        roi_xyxy = _undo_letterbox(b, boxes_xyxy, scale=INPUT_SCALE, pad=INPUT_PAD)
        source = INPUT_SOURCE
    else:
        source = detector_input
        roi_xyxy = boxes_xyxy

    crops = _crop_and_resize(
        b,
        source=source,
        rois=_clamp_to_tensor(b, roi_xyxy, tensor=source),
        crop_size=crop_size,
        sampling_ratio=sampling_ratio,
    )
    if max_detections is not None:
        crops = b.op(
            "Reshape",
            [crops, b.const("crop_shape", _i64(max_detections, channels, crop_height, crop_width))],
        )
    _normalize(
        b,
        crops,
        output=classifier_input,
        channels=channels,
        mean=mean,
        std=std,
        input_scale=input_scale,
    )

    b.op("Identity", [boxes_xyxy], outputs=[OUTPUT_BOXES])
    b.op("Identity", [scores], outputs=[OUTPUT_SCORES])
    b.op("Identity", [classes], outputs=[OUTPUT_CLASSES])
    b.op("Identity", [num_detections], outputs=[OUTPUT_NUM_DETECTIONS])

    rows: int | str = max_detections if max_detections is not None else "num_detections"
    bridge.outputs.extend(
        [
            helper.make_tensor_value_info(OUTPUT_BOXES, TensorProto.FLOAT, [rows, 4]),
            helper.make_tensor_value_info(OUTPUT_SCORES, TensorProto.FLOAT, [rows]),
            helper.make_tensor_value_info(OUTPUT_CLASSES, TensorProto.INT64, [rows]),
            helper.make_tensor_value_info(OUTPUT_NUM_DETECTIONS, TensorProto.INT64, [1]),
        ]
    )
    bridge.nodes.extend(b.nodes)
    bridge.initializers.extend(b.initializers)
    return bridge


def _select_boxes(
    b: _Builder,
    *,
    detector_output: str,
    conf_threshold: float,
    iou_threshold: float,
    max_boxes_per_class: int,
    max_detections: int | None,
) -> tuple[str, str, str, str]:
    """Decode the YOLO head, suppress, rank by confidence and pad to ``K``.

    ``NonMaxSuppression`` emits an ``(S, 3)`` selection matrix of
    ``(batch, class, box)`` triples grouped by class, not by score, so the rows
    are re-ranked here before being capped — otherwise capping at ``K`` would
    keep whichever classes happen to come first rather than the ``K`` most
    confident detections.

    Args:
        b: The node accumulator.
        detector_output: Name of the ``(1, 4 + nc, N)`` head tensor.
        conf_threshold: Score threshold for NMS.
        iou_threshold: IoU threshold for NMS.
        max_boxes_per_class: NMS per-class cap.
        max_detections: Fixed row count, or ``None`` to keep rows dynamic.

    Returns:
        tuple[str, str, str, str]: Names of ``(boxes_xyxy, scores, classes,
        num_detections)``. Boxes are corner-form in the detector's own
        letterboxed pixel space — the space the SDK's Python and TypeScript
        runtimes already know how to map back — so both crop sources report
        boxes identically.
    """
    zero, four = b.const("zero", _i64(0)), b.const("four", _i64(4))
    end, axis1 = b.const("end", _i64(_INT64_MAX)), b.const("axis1", _i64(1))

    boxes_cxcywh = b.op("Slice", [detector_output, zero, four, axis1])
    scores_bcn = b.op("Slice", [detector_output, four, end, axis1])
    boxes_bn4 = b.op("Transpose", [boxes_cxcywh], perm=[0, 2, 1])

    selected = b.op(
        "NonMaxSuppression",
        [
            boxes_bn4,
            scores_bcn,
            b.const("max_per_class", _i64(max_boxes_per_class)),
            b.const("iou", _f32(iou_threshold)),
            b.const("score", _f32(conf_threshold)),
        ],
        center_point_box=1,
    )

    one, two, three = b.const("one", _i64(1)), b.const("two", _i64(2)), b.const("three", _i64(3))
    class_column = b.op("Slice", [selected, one, two, axis1])
    box_column = b.op("Slice", [selected, two, three, axis1])

    scores_cn = b.op("Squeeze", [scores_bcn, b.const("axis0", _i64(0))])
    lookup = b.op("Concat", [class_column, box_column], axis=1)
    confidences = b.op("GatherND", [scores_cn, lookup])

    boxes_n4 = b.op("Squeeze", [boxes_bn4, b.const("axis0", _i64(0))])
    box_index = b.op("Squeeze", [box_column, b.const("axis1", _i64(1))])
    class_index = b.op("Squeeze", [class_column, b.const("axis1", _i64(1))])
    selected_boxes = b.op("Gather", [boxes_n4, box_index], axis=0)

    survivors = b.op("Shape", [confidences])
    keep = (
        survivors
        if max_detections is None
        else b.op("Min", [survivors, b.const("cap", _i64(max_detections))])
    )

    ranked_scores, ranking = b.op_multi(
        "TopK", [confidences, keep], count=2, axis=0, largest=1, sorted=1
    )
    boxes_xyxy = _to_xyxy(b, b.op("Gather", [selected_boxes, ranking], axis=0))
    ranked_classes = b.op("Gather", [class_index, ranking], axis=0)

    if max_detections is None:
        return boxes_xyxy, ranked_scores, ranked_classes, keep

    return _pad_to_fixed(
        b,
        boxes_xyxy=boxes_xyxy,
        scores=ranked_scores,
        classes=ranked_classes,
        keep=keep,
        max_detections=max_detections,
    )


def _to_xyxy(b: _Builder, boxes_cxcywh: str) -> str:
    """Convert ``(K, 4)`` centre-form boxes to corner form.

    Args:
        b: The node accumulator.
        boxes_cxcywh: Name of the ``(K, 4)`` ``(cx, cy, w, h)`` tensor.

    Returns:
        str: Name of the ``(K, 4)`` ``(x1, y1, x2, y2)`` tensor.
    """
    cx, cy, width, height = _split_columns(b, boxes_cxcywh)
    half = b.const("half", _f32(0.5))
    half_w = b.op("Mul", [width, half])
    half_h = b.op("Mul", [height, half])
    return b.op(
        "Concat",
        [
            b.op("Sub", [cx, half_w]),
            b.op("Sub", [cy, half_h]),
            b.op("Add", [cx, half_w]),
            b.op("Add", [cy, half_h]),
        ],
        axis=1,
    )


def _split_columns(b: _Builder, boxes: str) -> tuple[str, str, str, str]:
    """Split a ``(K, 4)`` tensor into its four ``(K, 1)`` columns.

    Args:
        b: The node accumulator.
        boxes: Name of the ``(K, 4)`` tensor.

    Returns:
        tuple[str, str, str, str]: The four column tensor names, in order.
    """
    columns = b.op_multi(
        "Split",
        [boxes, b.const("split4", _i64(1, 1, 1, 1))],
        count=4,
        axis=1,
    )
    return columns[0], columns[1], columns[2], columns[3]


def _pad_to_fixed(
    b: _Builder,
    *,
    boxes_xyxy: str,
    scores: str,
    classes: str,
    keep: str,
    max_detections: int,
) -> tuple[str, str, str, str]:
    """Zero-pad every per-detection tensor up to exactly ``max_detections`` rows.

    Padding is what makes the classifier's batch static, and it is also what
    removes the empty-batch case: with nothing detected the graph still hands
    the classifier ``K`` (degenerate) crops rather than a zero-row tensor some
    execution providers refuse to run. The trailing ``Reshape`` is not cosmetic
    — the padded length is computed from a runtime tensor, so without it shape
    inference reports a dynamic first axis and every downstream compiler loses
    the static shape the padding exists to provide.

    Args:
        b: The node accumulator.
        boxes_xyxy: Name of the ``(k, 4)`` box tensor.
        scores: Name of the ``(k,)`` confidence tensor.
        classes: Name of the ``(k,)`` class tensor.
        keep: Name of the ``(1,)`` int64 tensor holding ``k``.
        max_detections: The fixed row count ``K``.

    Returns:
        tuple[str, str, str, str]: Names of the padded ``(boxes, scores,
        classes, num_detections)`` tensors.
    """
    zero = b.const("pad_zero", _i64(0))
    missing = b.op("Sub", [b.const("target", _i64(max_detections)), keep])
    pads_2d = b.op("Concat", [zero, zero, missing, zero], axis=0)
    pads_1d = b.op("Concat", [zero, missing], axis=0)
    float_zero = b.const("pad_f32", np.asarray(0.0, dtype=np.float32))
    int_zero = b.const("pad_i64", np.asarray(0, dtype=np.int64))

    padded_boxes = b.op(
        "Reshape",
        [
            b.op("Pad", [boxes_xyxy, pads_2d, float_zero]),
            b.const("boxes_shape", _i64(max_detections, 4)),
        ],
    )
    padded_scores = b.op(
        "Reshape",
        [
            b.op("Pad", [scores, pads_1d, float_zero]),
            b.const("scores_shape", _i64(max_detections)),
        ],
    )
    padded_classes = b.op(
        "Reshape",
        [
            b.op("Pad", [classes, pads_1d, int_zero]),
            b.const("classes_shape", _i64(max_detections)),
        ],
    )
    return padded_boxes, padded_scores, padded_classes, keep


def _source_inputs(channels: int) -> list[onnx.ValueInfoProto]:
    """Declare the extra graph inputs the ``"original"`` crop source needs.

    The image's spatial axes are left dynamic on purpose: cropping at native
    resolution is the entire reason this mode exists, so pinning a size here
    would defeat it.

    Args:
        channels: Channel count of the full-resolution image tensor.

    Returns:
        list[onnx.ValueInfoProto]: The image input followed by the letterbox
        ``scale`` and ``(pad_left, pad_top)`` that map detector coordinates back
        onto it.
    """
    return [
        helper.make_tensor_value_info(
            INPUT_SOURCE,
            TensorProto.FLOAT,
            [1, channels, "source_height", "source_width"],
        ),
        helper.make_tensor_value_info(INPUT_SCALE, TensorProto.FLOAT, [1]),
        helper.make_tensor_value_info(INPUT_PAD, TensorProto.FLOAT, [2]),
    ]


def _undo_letterbox(b: _Builder, boxes_xyxy: str, *, scale: str, pad: str) -> str:
    """Map letterboxed box coordinates back onto the full-resolution image.

    Inverts exactly what :func:`~ort_vision_sdk.preprocess.image.letterbox`
    applied — subtract the padding, divide by the scale — so a box the detector
    found at 640x640 addresses the same object in the original photo.

    Args:
        b: The node accumulator.
        boxes_xyxy: Name of the ``(K, 4)`` corner-form tensor in letterbox space.
        scale: Name of the ``(1,)`` letterbox scale input.
        pad: Name of the ``(2,)`` ``(pad_left, pad_top)`` input.

    Returns:
        str: Name of the ``(K, 4)`` tensor in full-resolution pixel coordinates.
    """
    pad_x = b.op("Gather", [pad, b.const("pad_x_idx", _i64(0))], axis=0)
    pad_y = b.op("Gather", [pad, b.const("pad_y_idx", _i64(1))], axis=0)
    x1, y1, x2, y2 = _split_columns(b, boxes_xyxy)
    return b.op(
        "Concat",
        [
            b.op("Div", [b.op("Sub", [x1, pad_x]), scale]),
            b.op("Div", [b.op("Sub", [y1, pad_y]), scale]),
            b.op("Div", [b.op("Sub", [x2, pad_x]), scale]),
            b.op("Div", [b.op("Sub", [y2, pad_y]), scale]),
        ],
        axis=1,
    )


def _clamp_to_tensor(b: _Builder, boxes_xyxy: str, *, tensor: str) -> str:
    """Clamp boxes into ``tensor``, keeping every ROI at least one pixel wide.

    Two things are enforced. Coordinates are clipped to the tensor's spatial
    extent, because a detector can place a box partly outside the image and
    RoiAlign would otherwise sample undefined territory. And the far corner is
    pushed to at least one pixel past the near corner, so the padded rows —
    whose boxes are all-zero — and any genuinely degenerate detection still
    produce a resamplable region instead of a zero-area one. The near corner is
    clipped two pixels short of the edge precisely so that pushing the far
    corner out cannot leave the tensor.

    Args:
        b: The node accumulator.
        boxes_xyxy: Name of the ``(K, 4)`` corner-form tensor.
        tensor: Name of the NCHW image tensor the boxes index into.

    Returns:
        str: Name of the clamped ``(K, 4)`` tensor, ready for RoiAlign.
    """
    shape = b.op("Shape", [tensor])
    height = b.op(
        "Cast", [b.op("Gather", [shape, b.const("h_idx", _i64(2))], axis=0)], to=TensorProto.FLOAT
    )
    width = b.op(
        "Cast", [b.op("Gather", [shape, b.const("w_idx", _i64(3))], axis=0)], to=TensorProto.FLOAT
    )

    zero = b.const("clamp_zero", _f32(0.0))
    one = b.const("clamp_one", _f32(1.0))
    two = b.const("clamp_two", _f32(2.0))
    far_limit_x = b.op("Sub", [width, one])
    far_limit_y = b.op("Sub", [height, one])
    near_limit_x = b.op("Sub", [width, two])
    near_limit_y = b.op("Sub", [height, two])

    x1, y1, x2, y2 = _split_columns(b, boxes_xyxy)
    near_x = b.op("Min", [b.op("Max", [x1, zero]), near_limit_x])
    near_y = b.op("Min", [b.op("Max", [y1, zero]), near_limit_y])
    far_x = b.op("Max", [b.op("Min", [x2, far_limit_x]), b.op("Add", [near_x, one])])
    far_y = b.op("Max", [b.op("Min", [y2, far_limit_y]), b.op("Add", [near_y, one])])
    return b.op("Concat", [near_x, near_y, far_x, far_y], axis=1)


def _crop_and_resize(
    b: _Builder,
    *,
    source: str,
    rois: str,
    crop_size: tuple[int, int],
    sampling_ratio: int,
) -> str:
    """Crop every ROI out of ``source`` and resample it to ``crop_size``.

    Every ROI addresses batch 0 — the fused pipeline processes one image per
    run — so the ``batch_indices`` RoiAlign requires are a run-length zero
    vector built from the ROI count rather than a constant, which is what keeps
    this correct in the dynamic-row mode too.

    Args:
        b: The node accumulator.
        source: Name of the NCHW image tensor to crop from.
        rois: Name of the ``(K, 4)`` corner-form ROI tensor, in ``source``'s own
            pixel coordinates.
        crop_size: ``(width, height)`` every crop is resampled to.
        sampling_ratio: RoiAlign samples per output bin.

    Returns:
        str: Name of the ``(K, channels, height, width)`` crop batch.
    """
    rows = b.op(
        "Slice",
        [
            b.op("Shape", [rois]),
            b.const("k_start", _i64(0)),
            b.const("k_end", _i64(1)),
            b.const("k_axis", _i64(0)),
        ],
    )
    batch_indices = b.op(
        "ConstantOfShape",
        [rows],
        value=numpy_helper.from_array(np.asarray([0], dtype=np.int64)),
    )
    return b.op(
        "RoiAlign",
        [source, rois, batch_indices],
        coordinate_transformation_mode="half_pixel",
        mode="avg",
        output_height=crop_size[1],
        output_width=crop_size[0],
        sampling_ratio=sampling_ratio,
        spatial_scale=1.0,
    )


def _normalize(
    b: _Builder,
    crops: str,
    *,
    output: str,
    channels: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    input_scale: float,
) -> None:
    """Apply the classifier's own normalization and bind the result to its input.

    Mirrors :func:`~ort_vision_sdk.preprocess.image.normalize`: the crop is
    scaled, then centred, then divided. Steps that would be no-ops (unit scale,
    zero mean, unit deviation) are skipped rather than emitted, so a classifier
    that wants raw ``[0, 1]`` crops adds no arithmetic at all.

    Args:
        b: The node accumulator.
        crops: Name of the ``(K, C, H, W)`` crop batch.
        output: Name to bind the normalized tensor to — the classifier's input.
        channels: Channel count, used to shape the broadcast constants.
        mean: Per-channel mean.
        std: Per-channel standard deviation.
        input_scale: Multiplier applied before centring.
    """
    current = crops
    if input_scale != 1.0:
        scale_constant = np.asarray(input_scale, dtype=np.float32)
        current = b.op("Mul", [current, b.const("input_scale", scale_constant)])
    if any(value != 0.0 for value in mean):
        constant = np.asarray(mean[:channels], dtype=np.float32).reshape(1, channels, 1, 1)
        current = b.op("Sub", [current, b.const("mean", constant)])
    if any(value != 1.0 for value in std):
        constant = np.asarray(std[:channels], dtype=np.float32).reshape(1, channels, 1, 1)
        current = b.op("Div", [current, b.const("std", constant)])
    b.op("Identity", [current], outputs=[output])
