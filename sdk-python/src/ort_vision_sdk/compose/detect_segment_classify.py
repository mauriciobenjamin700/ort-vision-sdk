"""Fuse a detector, a segmenter and a classifier into one ONNX pipeline.

The three stages compose like this: the detector finds boxes, ``RoiAlign`` cuts
each one out of the image, the segmenter turns each crop into a foreground mask,
and the classifier reads the crops that mask gates. One file, one session, one
model load.

This is not the arrangement to reach for when the second stage is a YOLO-seg
export — that model already detects and segments in a single pass, so a separate
detector duplicates its work. It is for the case where segmentation is a plain
image-to-mask network (U-Net and relatives) that has no detector of its own and
runs best on a tight crop rather than a full frame.

Two constraints are structural rather than incidental:

- **The segmenter and the classifier share a crop size.** The mask is computed
  in the crop's coordinate space and multiplies the crop directly, so a second
  resolution between them would mean resampling one of the two per instance. A
  mismatch is refused with a message naming both sizes rather than papered over.
- **The bridge stays float32.** Same reason as the two-stage pipeline: ONNX's
  ``NonMaxSuppression`` is defined for ``tensor(float)`` only. Half-precision
  stages are cast at the seams — see
  :func:`~ort_vision_sdk.compose.detect_classify.fuse_detect_classify`.

The stage helpers (loading, opset reconciliation, shape resolution, metadata,
post-build validation) are shared with the two-stage fusion and live in
:mod:`ort_vision_sdk.compose.detect_classify`, which is their only other caller.
"""

from __future__ import annotations

from pathlib import Path

import onnx
from onnx import TensorProto, compose, helper

from ort_vision_sdk import __version__
from ort_vision_sdk.compose.bridge import (
    MIN_OPSET,
    MaskActivation,
    build_bridge,
    build_mask_bridge,
)
from ort_vision_sdk.compose.detect_classify import (
    _align_opset,
    _bind,
    _channels,
    _check_detector_head,
    _classifier_classes,
    _crop_value_infos,
    _default_opset,
    _ends_in_softmax,
    _head_classes,
    _image_input_shape,
    _input_elem_type,
    _load,
    _merged_opsets,
    _output_elem_type,
    _probs_value_info,
    _resolve_per_class_cap,
    _resolve_size,
    _stage_names,
    _validate,
    _write_metadata,
)
from ort_vision_sdk.core.exceptions import FusionError
from ort_vision_sdk.fusion import (
    FUSION_KIND_DETECT_SEGMENT_CLASSIFY,
    INPUT_IMAGE,
    OUTPUT_PROBS,
    CropSource,
    FusionSpec,
)
from ort_vision_sdk.labels import LabelSpec
from ort_vision_sdk.normalization import Normalization, resolve_normalization

__all__ = ["fuse_detect_segment_classify"]

_DETECTOR_PREFIX = "det_"
_SEGMENTER_PREFIX = "seg_"
_CLASSIFIER_PREFIX = "clf_"

_GRAPH_NAME = "ort_vision_detect_segment_classify"

_RAW_CROPS = "ovs_crops_raw"
"""Internal name for the un-normalized crop batch both later stages read."""


def fuse_detect_segment_classify(
    detector: str | Path,
    segmenter: str | Path,
    classifier: str | Path,
    output: str | Path | None = None,
    *,
    crop_source: CropSource = "detector_input",
    max_detections: int | None = 20,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_boxes_per_class: int | None = None,
    input_size: tuple[int, int] | None = None,
    crop_size: tuple[int, int] | None = None,
    mask_channel: int = 0,
    mask_activation: MaskActivation | None = None,
    mask_threshold: float = 0.5,
    apply_mask: bool = True,
    segmenter_normalization: Normalization = "auto",
    segmenter_mean: tuple[float, float, float] | None = None,
    segmenter_std: tuple[float, float, float] | None = None,
    segmenter_input_scale: float = 1.0,
    normalization: Normalization = "auto",
    mean: tuple[float, float, float] | None = None,
    std: tuple[float, float, float] | None = None,
    input_scale: float = 1.0,
    sampling_ratio: int = 0,
    apply_softmax: bool | None = None,
    detector_labels: LabelSpec = None,
    segmenter_labels: LabelSpec = None,
    classifier_labels: LabelSpec = None,
    validate: bool = True,
) -> onnx.ModelProto:
    """Fuse a detector, a per-crop segmenter and a classifier into one ONNX graph.

    The fused graph takes a letterboxed image and emits ``boxes``, ``scores``,
    ``classes``, ``num_detections``, ``masks`` and ``probs``. Load it with
    :class:`~ort_vision_sdk.tasks.pipeline.DetectClassify`, which reads the
    configuration out of the file and exposes the masks alongside each detection.

    Example:
        >>> from ort_vision_sdk.compose import fuse_detect_segment_classify
        >>> fuse_detect_segment_classify(
        ...     "yolov8n.onnx",
        ...     "unet.onnx",
        ...     "resnet18.onnx",
        ...     "pipeline.onnx",
        ...     max_detections=10,
        ... )

    Args:
        detector: Path to the YOLO detector export.
        segmenter: Path to the segmentation model. Takes a ``(K, C, H, W)`` crop
            batch and returns ``(K, S, H, W)`` logits at the same resolution.
        classifier: Path to the classifier export.
        output: Where to write the fused ``.onnx``. ``None`` returns the model
            without writing it.
        crop_source: Which tensor the boxes are cropped from — see
            :data:`~ort_vision_sdk.fusion.CropSource`.
        max_detections: Fixed row count ``K`` for every output, surplus rows
            zero-padded. ``None`` leaves the row count dynamic.
        conf_threshold: Score threshold baked into the graph's NMS node.
        iou_threshold: IoU threshold baked into the graph's NMS node.
        max_boxes_per_class: Per-class NMS cap before the global ranking.
        input_size: ``(width, height)`` the detector expects. ``None`` reads it
            off the detector's graph.
        crop_size: ``(width, height)`` every crop is resampled to. ``None``
            reads it off the segmenter's graph, which must agree with the
            classifier's.
        mask_channel: Which channel of the segmenter output carries the
            foreground. ``0`` for a single-channel head.
        mask_activation: How segmenter logits become probabilities. ``None``
            (default) picks ``"sigmoid"`` for a single-channel head and
            ``"softmax"`` for a multi-channel one.
        mask_threshold: Probability above which a pixel counts as foreground.
        apply_mask: Whether the mask multiplies the crop before the classifier
            sees it. Set ``False`` when the classifier was trained on un-masked
            crops but the masks are still wanted as an output.
        segmenter_normalization: Which preprocessing the segmenter expects —
            see :data:`~ort_vision_sdk.normalization.Normalization`.
        segmenter_mean: Per-channel mean for the segmenter. ``None`` takes it
            from ``segmenter_normalization``.
        segmenter_std: Per-channel standard deviation for the segmenter.
        segmenter_input_scale: Multiplier applied to the crop before the
            segmenter's mean/std.
        normalization: Which preprocessing the classifier expects.
        mean: Per-channel mean for the classifier.
        std: Per-channel standard deviation for the classifier.
        input_scale: Multiplier applied to the masked crop before the
            classifier's mean/std.
        sampling_ratio: RoiAlign samples per output bin.
        apply_softmax: Whether the runtime should softmax the classifier's rows.
            ``None`` inspects the classifier's graph.
        detector_labels: Class names for the detection stage.
        segmenter_labels: Class names for the segmenter's channels, recorded in
            the file's metadata for a caller that wants to name them.
        classifier_labels: Class names for the classification stage.
        validate: Run the fused graph once in ONNX Runtime before returning.

    Returns:
        onnx.ModelProto: The fused model, also written to ``output`` when given.

    Raises:
        FusionError: If a file cannot be read, the detector's head is not the
            anchor-free YOLO layout, the segmenter and the classifier disagree
            on crop size, a required resolution is neither declared nor
            supplied, the opsets cannot be reconciled, or ``validate`` is on and
            the fused graph fails to run.
        ValueError: If a numeric argument is out of range.
    """
    detector_model = _load(detector, role="detector")
    segmenter_model = _load(segmenter, role="segmenter")
    classifier_model = _load(classifier, role="classifier")

    detector_input_shape = _image_input_shape(detector_model, role="detector")
    segmenter_input_shape = _image_input_shape(segmenter_model, role="segmenter")
    classifier_input_shape = _image_input_shape(classifier_model, role="classifier")
    channels = _channels(detector_input_shape, classifier_input_shape)

    resolved_input_size = _resolve_size(
        declared=detector_input_shape,
        requested=input_size,
        role="detector",
        argument="input_size",
    )
    resolved_crop_size = _resolve_size(
        declared=segmenter_input_shape,
        requested=crop_size,
        role="segmenter",
        argument="crop_size",
    )
    _check_crop_sizes(
        segmenter_size=resolved_crop_size,
        classifier_shape=classifier_input_shape,
        requested=crop_size,
    )
    _check_detector_head(detector_model)

    target_opset = max(
        _default_opset(detector_model),
        _default_opset(segmenter_model),
        _default_opset(classifier_model),
        MIN_OPSET,
    )
    detector_model = _align_opset(detector_model, target_opset, role="detector")
    segmenter_model = _align_opset(segmenter_model, target_opset, role="segmenter")
    classifier_model = _align_opset(classifier_model, target_opset, role="classifier")

    detector_names = _stage_names(detector_model, detector_labels, _head_classes(detector_model))
    segmenter_names = _stage_names(
        segmenter_model, segmenter_labels, _segmenter_channels(segmenter_model)
    )
    classifier_names = _stage_names(
        classifier_model, classifier_labels, _classifier_classes(classifier_model)
    )
    softmax_needed = (
        apply_softmax if apply_softmax is not None else not _ends_in_softmax(classifier_model)
    )
    activation = mask_activation or _default_activation(segmenter_model)

    _, segmenter_resolved_mean, segmenter_resolved_std = resolve_normalization(
        {entry.key: entry.value for entry in segmenter_model.metadata_props},
        normalization=segmenter_normalization,
        mean=segmenter_mean,
        std=segmenter_std,
    )
    normalization_name, resolved_mean, resolved_std = resolve_normalization(
        {entry.key: entry.value for entry in classifier_model.metadata_props},
        normalization=normalization,
        mean=mean,
        std=std,
    )

    spec = FusionSpec(
        input_size=resolved_input_size,
        crop_size=resolved_crop_size,
        crop_source=crop_source,
        max_detections=max_detections,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        apply_softmax=softmax_needed,
        detector_names=detector_names,
        classifier_names=classifier_names,
        sdk_version=__version__,
        kind=FUSION_KIND_DETECT_SEGMENT_CLASSIFY,
        segmenter_names=segmenter_names,
        mask_threshold=mask_threshold,
        mask_applied=apply_mask,
    )

    model = _assemble(
        detector_model=detector_model,
        segmenter_model=segmenter_model,
        classifier_model=classifier_model,
        spec=spec,
        channels=channels,
        max_boxes_per_class=_resolve_per_class_cap(max_boxes_per_class, max_detections),
        segmenter_mean=segmenter_resolved_mean,
        segmenter_std=segmenter_resolved_std,
        segmenter_input_scale=segmenter_input_scale,
        mean=resolved_mean,
        std=resolved_std,
        normalization_name=normalization_name,
        input_scale=input_scale,
        sampling_ratio=sampling_ratio,
        mask_channel=mask_channel,
        mask_activation=activation,
        target_opset=target_opset,
    )

    saved_to: str | Path | None = None
    if output is not None:
        onnx.save(model, str(output))
        saved_to = output
    if validate:
        _validate(model, spec=spec, channels=channels, saved_to=saved_to)
    return model


def _check_crop_sizes(
    *,
    segmenter_size: tuple[int, int],
    classifier_shape: list[int | str | None],
    requested: tuple[int, int] | None,
) -> None:
    """Refuse a segmenter and a classifier that want different crop resolutions.

    The mask multiplies the crop in place, so both stages have to read the same
    tensor. Resampling between them would mean an extra resize per instance and
    a mask that no longer lines up with the crop it describes — worth refusing
    rather than hiding.

    Args:
        segmenter_size: ``(width, height)`` resolved for the segmenter.
        classifier_shape: The classifier's declared input shape.
        requested: The ``crop_size`` the caller passed, if any.

    Raises:
        FusionError: If the classifier declares a static resolution that differs
            from the segmenter's.
    """
    height, width = classifier_shape[2], classifier_shape[3]
    if not isinstance(height, int) or not isinstance(width, int):
        return
    if (width, height) == segmenter_size:
        return
    chosen = "requested" if requested is not None else "read off the segmenter"
    raise FusionError(
        f"The segmenter and the classifier disagree on crop size: {segmenter_size} "
        f"({chosen}) against {(width, height)} declared by the classifier. The mask is "
        "computed in the crop's own space and multiplies it directly, so both stages must "
        "take the same resolution. Re-export one of them, or pass crop_size explicitly if "
        "one of the two graphs leaves the axes dynamic."
    )


def _segmenter_channels(model: onnx.ModelProto) -> int | None:
    """Read the segmenter's output channel count, if the graph states it.

    Args:
        model: The segmenter export.

    Returns:
        int | None: The channel count, or ``None`` when the axis is dynamic.
    """
    output = model.graph.output[0].type.tensor_type
    if len(output.shape.dim) < 2:
        return None
    channel = output.shape.dim[1]
    return int(channel.dim_value) if channel.dim_value else None


def _default_activation(model: onnx.ModelProto) -> MaskActivation:
    """Pick the activation a segmenter's head implies.

    Args:
        model: The segmenter export.

    Returns:
        MaskActivation: ``"sigmoid"`` for a one-channel head — where each pixel
        carries a single foreground logit — and ``"softmax"`` when the channels
        compete. A dynamic channel axis falls back to ``"sigmoid"``, the
        single-channel case being overwhelmingly the common one for a binary
        mask; pass ``mask_activation`` to override.
    """
    channels = _segmenter_channels(model)
    return "softmax" if channels is not None and channels > 1 else "sigmoid"


def _assemble(
    *,
    detector_model: onnx.ModelProto,
    segmenter_model: onnx.ModelProto,
    classifier_model: onnx.ModelProto,
    spec: FusionSpec,
    channels: int,
    max_boxes_per_class: int,
    segmenter_mean: tuple[float, float, float],
    segmenter_std: tuple[float, float, float],
    segmenter_input_scale: float,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    normalization_name: str,
    input_scale: float,
    sampling_ratio: int,
    mask_channel: int,
    mask_activation: MaskActivation,
    target_opset: int,
) -> onnx.ModelProto:
    """Splice the detector, both bridges, the segmenter and the classifier into one graph.

    Args:
        detector_model: The detector, already at ``target_opset``.
        segmenter_model: The segmenter, already at ``target_opset``.
        classifier_model: The classifier, already at ``target_opset``.
        spec: The pipeline configuration, recorded into the result's metadata.
        channels: Image channel count.
        max_boxes_per_class: NMS per-class cap.
        segmenter_mean: Per-channel mean for the segmenter's normalization.
        segmenter_std: Per-channel standard deviation for the segmenter.
        segmenter_input_scale: Multiplier applied before the segmenter's mean.
        mean: Per-channel mean for the classifier's normalization.
        std: Per-channel standard deviation for the classifier.
        normalization_name: Preset name recorded in the fused file's metadata.
        input_scale: Multiplier applied before the classifier's mean.
        sampling_ratio: RoiAlign samples per output bin.
        mask_channel: Segmenter channel carrying the foreground.
        mask_activation: How segmenter logits become probabilities.
        target_opset: Opset the fused graph declares.

    Returns:
        onnx.ModelProto: The fused, checked model.

    Raises:
        FusionError: If the assembled graph fails ONNX's structural checker.
    """
    detector_model = compose.add_prefix(detector_model, _DETECTOR_PREFIX)
    segmenter_model = compose.add_prefix(segmenter_model, _SEGMENTER_PREFIX)
    classifier_model = compose.add_prefix(classifier_model, _CLASSIFIER_PREFIX)

    detector_input = detector_model.graph.input[0].name
    detector_output = detector_model.graph.output[0].name
    segmenter_input = segmenter_model.graph.input[0].name
    segmenter_output = segmenter_model.graph.output[0].name
    classifier_input = classifier_model.graph.input[0].name
    classifier_output = classifier_model.graph.output[0].name

    detector_in_type = _input_elem_type(detector_model)
    detector_out_type = _output_elem_type(detector_model)
    segmenter_in_type = _input_elem_type(segmenter_model)
    segmenter_out_type = _output_elem_type(segmenter_model)
    classifier_in_type = _input_elem_type(classifier_model)
    classifier_out_type = _output_elem_type(classifier_model)

    bridge_source = INPUT_IMAGE if detector_in_type != TensorProto.FLOAT else detector_input
    bridge_detector_output = _float32_alias(detector_output, detector_out_type)
    bridge_segmenter_input = _float32_alias(segmenter_input, segmenter_in_type)
    bridge_segmenter_output = _float32_alias(segmenter_output, segmenter_out_type)
    bridge_classifier_input = _float32_alias(classifier_input, classifier_in_type)

    bridge = build_bridge(
        detector_output=bridge_detector_output,
        detector_input=bridge_source,
        classifier_input=bridge_segmenter_input,
        crop_size=spec.crop_size,
        channels=channels,
        crop_source=spec.crop_source,
        max_detections=spec.max_detections,
        conf_threshold=spec.conf_threshold,
        iou_threshold=spec.iou_threshold,
        max_boxes_per_class=max_boxes_per_class,
        mean=segmenter_mean,
        std=segmenter_std,
        input_scale=segmenter_input_scale,
        sampling_ratio=sampling_ratio,
        crops_output=_RAW_CROPS,
    )
    mask_bridge = build_mask_bridge(
        segmenter_output=bridge_segmenter_output,
        crops=_RAW_CROPS,
        classifier_input=bridge_classifier_input,
        crop_size=spec.crop_size,
        channels=channels,
        max_detections=spec.max_detections,
        mask_channel=mask_channel,
        mask_activation=mask_activation,
        mask_threshold=spec.mask_threshold,
        apply_mask=spec.mask_applied,
        mean=mean,
        std=std,
        input_scale=input_scale,
    )

    width, height = spec.input_size
    image_input = helper.make_tensor_value_info(
        INPUT_IMAGE, TensorProto.FLOAT, [1, channels, height, width]
    )
    probs_output = _probs_value_info(classifier_model, rows=spec.max_detections)

    nodes = [
        *_bind(
            "ovs_bind_input",
            INPUT_IMAGE,
            detector_input,
            source_type=int(TensorProto.FLOAT),
            target_type=detector_in_type,
        ),
        *detector_model.graph.node,
        *_bind(
            "ovs_cast_detector_output",
            detector_output,
            bridge_detector_output,
            source_type=detector_out_type,
            target_type=int(TensorProto.FLOAT),
        ),
        *bridge.nodes,
        *_bind(
            "ovs_cast_segmenter_input",
            bridge_segmenter_input,
            segmenter_input,
            source_type=int(TensorProto.FLOAT),
            target_type=segmenter_in_type,
        ),
        *segmenter_model.graph.node,
        *_bind(
            "ovs_cast_segmenter_output",
            segmenter_output,
            bridge_segmenter_output,
            source_type=segmenter_out_type,
            target_type=int(TensorProto.FLOAT),
        ),
        *mask_bridge.nodes,
        *_bind(
            "ovs_cast_crops",
            bridge_classifier_input,
            classifier_input,
            source_type=int(TensorProto.FLOAT),
            target_type=classifier_in_type,
        ),
        *classifier_model.graph.node,
        *_bind(
            "ovs_bind_probs",
            classifier_output,
            OUTPUT_PROBS,
            source_type=classifier_out_type,
            target_type=int(TensorProto.FLOAT),
        ),
    ]
    graph = helper.make_graph(
        nodes=nodes,
        name=_GRAPH_NAME,
        inputs=[image_input, *bridge.inputs],
        outputs=[*bridge.outputs, *mask_bridge.outputs, probs_output],
        initializer=[
            *detector_model.graph.initializer,
            *bridge.initializers,
            *segmenter_model.graph.initializer,
            *mask_bridge.initializers,
            *classifier_model.graph.initializer,
        ],
        value_info=[
            *detector_model.graph.value_info,
            *segmenter_model.graph.value_info,
            *classifier_model.graph.value_info,
            *_crop_value_infos(
                bridge_segmenter_input,
                segmenter_input,
                spec=spec,
                channels=channels,
                elem_type=segmenter_in_type,
            ),
            *_crop_value_infos(
                bridge_classifier_input,
                classifier_input,
                spec=spec,
                channels=channels,
                elem_type=classifier_in_type,
            ),
        ],
    )

    model = helper.make_model(
        graph,
        opset_imports=_merged_opsets(detector_model, classifier_model, target_opset),
        functions=[
            *detector_model.functions,
            *segmenter_model.functions,
            *classifier_model.functions,
        ],
        producer_name="ort-vision-sdk",
        producer_version=spec.sdk_version,
    )
    model.ir_version = max(
        detector_model.ir_version, segmenter_model.ir_version, classifier_model.ir_version
    )
    _write_metadata(model, detector_model, spec, normalization_name=normalization_name)

    try:
        onnx.checker.check_model(model)
    except Exception as exc:
        raise FusionError(f"The fused graph is not a valid ONNX model: {exc}") from exc
    return model


def _float32_alias(name: str, elem_type: int) -> str:
    """Name the float32 tensor that sits beside a half-precision one at a seam.

    Args:
        name: The stage's own tensor name.
        elem_type: The element type that stage declares.

    Returns:
        str: A distinct name when a cast is needed, or ``name`` itself when the
        stage is already float32 and the seam is a plain bind.
    """
    return name if elem_type == TensorProto.FLOAT else f"{name}_ovs_fp32"
