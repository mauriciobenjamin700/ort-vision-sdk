"""Fuse a detector and a classifier into one ``.onnx`` file.

Running detection and classification as two models means two sessions, two
model loads, two sets of weights resident in memory, and a round trip through
Python (or JavaScript) for every crop — decode the boxes, slice the image,
resize each region, restack a batch, call the second runtime. On a phone or in
a browser tab that round trip is frequently the dominant cost, and the second
load is what makes a page feel slow to become interactive.

:func:`fuse_detect_classify` removes both. It rewrites the two protobufs into a
single graph: the detector's nodes, then the bridge from
:mod:`ort_vision_sdk.compose.bridge`, then the classifier's nodes — one file,
one session, one load, and the crops never leave the runtime.

What the caller still does is unchanged: letterbox the image and hand it over.
What they no longer do is anything between the two models.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import onnx
import onnx.version_converter
from onnx import TensorProto, compose, helper

from ort_vision_sdk import __version__
from ort_vision_sdk.compose.bridge import MIN_OPSET, build_bridge
from ort_vision_sdk.core.exceptions import FusionError
from ort_vision_sdk.fusion import (
    INPUT_IMAGE,
    INPUT_PAD,
    INPUT_SCALE,
    INPUT_SOURCE,
    METADATA_PREFIX,
    OUTPUT_NUM_DETECTIONS,
    OUTPUT_PROBS,
    CropSource,
    FusionSpec,
)
from ort_vision_sdk.graph import model_names
from ort_vision_sdk.labels import LabelSpec, resolve_labels
from ort_vision_sdk.preprocess.image import IMAGENET_MEAN, IMAGENET_STD

__all__ = ["fuse_detect_classify"]

_DETECTOR_PREFIX = "det_"
_CLASSIFIER_PREFIX = "clf_"
_CROP_BATCH_DIM = OUTPUT_NUM_DETECTIONS
"""Symbol the dynamic-row mode gives the crop-batch axis.

Deliberately the same symbol the bridge gives the detection outputs: the two
axes are always the same length, and naming them identically is what tells a
shape-inference pass — and any compiler downstream of it — that they are tied
rather than merely both unknown.
"""
_GRAPH_NAME = "ort_vision_detect_classify"


def fuse_detect_classify(
    detector: str | Path,
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
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
    input_scale: float = 1.0,
    sampling_ratio: int = 0,
    apply_softmax: bool | None = None,
    detector_labels: LabelSpec = None,
    classifier_labels: LabelSpec = None,
    validate: bool = True,
) -> onnx.ModelProto:
    """Fuse a YOLO detector and an image classifier into a single ONNX pipeline.

    The fused graph takes a letterboxed image, detects objects, crops each
    surviving box, resamples it to the classifier's resolution, normalizes it,
    and classifies the whole batch — emitting ``boxes``, ``scores``,
    ``classes``, ``num_detections`` and ``probs``. Load it with
    :class:`~ort_vision_sdk.tasks.pipeline.DetectClassify`, which reads the
    configuration recorded here straight out of the file.

    Example:
        >>> from ort_vision_sdk.compose import fuse_detect_classify
        >>> fuse_detect_classify(
        ...     "yolov8n.onnx",
        ...     "resnet18.onnx",
        ...     "pipeline.onnx",
        ...     max_detections=20,
        ... )

    Args:
        detector: Path to the detector ``.onnx``. Must be an anchor-free YOLO
            export — output shape ``(1, 4 + num_classes, num_anchors)``, the
            same family :class:`~ort_vision_sdk.tasks.detector.Detector`
            handles.
        classifier: Path to the classifier ``.onnx``. Must take one NCHW image
            input and emit one ``(batch, num_classes)`` output.
        output: Where to write the fused model. ``None`` builds it in memory
            and returns it without touching disk.
        crop_source: Which tensor the crops come from — see
            :data:`~ort_vision_sdk.fusion.CropSource`. ``"detector_input"``
            (default) keeps the pipeline at a single input; ``"original"`` adds
            a full-resolution input so small objects are not classified from
            their downscaled copy.
        max_detections: Fixed number of boxes the pipeline reports, surplus rows
            zero-padded and counted by ``num_detections``. ``None`` makes every
            shape dynamic, which is cheaper per run but requires a classifier
            that accepts a variable — and possibly zero-row — batch.
        conf_threshold: Score threshold baked into the graph's NMS. Fixed at
            fusion time; changing it means re-fusing.
        iou_threshold: IoU threshold baked into the graph's NMS.
        max_boxes_per_class: NMS per-class cap. ``None`` (default) uses four
            times ``max_detections`` — enough headroom that one crowded class
            cannot fill the ranking — or 300 in dynamic mode.
        input_size: ``(width, height)`` the detector stage runs at. ``None``
            reads it from the detector's graph, which is required unless the
            export left its spatial axes dynamic.
        crop_size: ``(width, height)`` each crop is resampled to. ``None`` reads
            it from the classifier's graph.
        mean: Per-channel mean subtracted from each crop. Defaults to the
            ImageNet values, matching
            :class:`~ort_vision_sdk.tasks.classifier.Classifier`.
        std: Per-channel standard deviation each crop is divided by.
        input_scale: Multiplier applied to a crop before ``mean``/``std``. The
            pipeline feeds images as ``float32`` in ``[0, 1]``, so leave it at
            1.0 for a torchvision-style classifier and set 255.0 for one that
            expects raw ``0..255`` values.
        sampling_ratio: RoiAlign samples per output bin. ``0`` (default) adapts
            to the box size and anti-aliases large boxes; ``1`` is a plain
            bilinear resample.
        apply_softmax: Whether the classifier's output still needs a softmax at
            runtime. ``None`` (default) inspects the classifier graph and
            answers ``False`` when it already ends in one.
        detector_labels: Class names for the detection stage — see
            :func:`~ort_vision_sdk.labels.resolve_labels`. ``None`` (default)
            carries over the names the detector export baked into its metadata.
        classifier_labels: Class names for the classification stage. ``None``
            carries over the classifier's own metadata names.
        validate: Run the fused graph once in ONNX Runtime before returning.
            This is what catches a classifier whose graph cannot actually take a
            batch of crops — a hardcoded batch size inside a ``Reshape``, for
            instance, which no amount of input-shape rewriting fixes.

    Returns:
        onnx.ModelProto: The fused model, also written to ``output`` when given.

    Raises:
        FusionError: If either file cannot be read, the detector's head is not
            the anchor-free YOLO layout, a required resolution is neither
            declared by a graph nor supplied, the opsets cannot be reconciled,
            or ``validate`` is on and the fused graph fails to run.
        ValueError: If a numeric argument is out of range — propagated from
            :func:`~ort_vision_sdk.compose.bridge.build_bridge`.
    """
    detector_model = _load(detector, role="detector")
    classifier_model = _load(classifier, role="classifier")

    detector_input_shape = _image_input_shape(detector_model, role="detector")
    classifier_input_shape = _image_input_shape(classifier_model, role="classifier")
    channels = _channels(detector_input_shape, classifier_input_shape)

    resolved_input_size = _resolve_size(
        declared=detector_input_shape,
        requested=input_size,
        role="detector",
        argument="input_size",
    )
    resolved_crop_size = _resolve_size(
        declared=classifier_input_shape,
        requested=crop_size,
        role="classifier",
        argument="crop_size",
    )
    _check_detector_head(detector_model)

    target_opset = max(_default_opset(detector_model), _default_opset(classifier_model), MIN_OPSET)
    detector_model = _align_opset(detector_model, target_opset, role="detector")
    classifier_model = _align_opset(classifier_model, target_opset, role="classifier")

    detector_names = _stage_names(detector_model, detector_labels, _head_classes(detector_model))
    classifier_names = _stage_names(
        classifier_model, classifier_labels, _classifier_classes(classifier_model)
    )
    softmax_needed = (
        apply_softmax if apply_softmax is not None else not _ends_in_softmax(classifier_model)
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
    )

    model = _assemble(
        detector_model=detector_model,
        classifier_model=classifier_model,
        spec=spec,
        channels=channels,
        max_boxes_per_class=_resolve_per_class_cap(max_boxes_per_class, max_detections),
        mean=mean,
        std=std,
        input_scale=input_scale,
        sampling_ratio=sampling_ratio,
        target_opset=target_opset,
    )

    if output is not None:
        onnx.save(model, str(output))
    if validate:
        _validate(model, spec=spec, channels=channels, saved_to=output)
    return model


def _load(path: str | Path, *, role: str) -> onnx.ModelProto:
    """Read an ONNX model off disk, reporting which of the two stages failed.

    Args:
        path: Filesystem path to the ``.onnx`` file.
        role: ``"detector"`` or ``"classifier"``, used in the error message.

    Returns:
        onnx.ModelProto: The parsed model.

    Raises:
        FusionError: If the file is missing or is not a readable ONNX protobuf.
    """
    resolved = Path(path)
    if not resolved.is_file():
        raise FusionError(f"{role} model not found: {resolved}")
    try:
        return onnx.load(str(resolved))
    except Exception as exc:
        raise FusionError(f"Failed to read the {role} model at {resolved}: {exc}") from exc


def _image_input_shape(model: onnx.ModelProto, *, role: str) -> list[int | str | None]:
    """Read the declared shape of a model's single image input.

    Args:
        model: The model to inspect.
        role: ``"detector"`` or ``"classifier"``, used in the error message.

    Returns:
        list[int | str | None]: One entry per axis — an ``int`` for a static
        dimension, the symbol name for a named dynamic one, ``None`` for an
        unnamed dynamic one.

    Raises:
        FusionError: If the model does not declare exactly one input, or that
            input is not a 4-D NCHW tensor. A model with more than one input
            has no unambiguous place for the bridge to attach.
    """
    initializers = {tensor.name for tensor in model.graph.initializer}
    inputs = [entry for entry in model.graph.input if entry.name not in initializers]
    if len(inputs) != 1:
        names = ", ".join(entry.name for entry in inputs) or "none"
        raise FusionError(
            f"The {role} model must declare exactly one input, found {len(inputs)} ({names})."
        )
    dims = inputs[0].type.tensor_type.shape.dim
    if len(dims) != 4:
        raise FusionError(
            f"The {role} model's input {inputs[0].name!r} must be a 4-D NCHW tensor, "
            f"got {len(dims)} dimensions."
        )
    shape: list[int | str | None] = []
    for dim in dims:
        if dim.HasField("dim_value"):
            shape.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            shape.append(str(dim.dim_param))
        else:
            shape.append(None)
    return shape


def _channels(
    detector_shape: list[int | str | None],
    classifier_shape: list[int | str | None],
) -> int:
    """Agree on the channel count both stages use.

    Args:
        detector_shape: Declared shape of the detector's input.
        classifier_shape: Declared shape of the classifier's input.

    Returns:
        int: The shared channel count.

    Raises:
        FusionError: If the two disagree, or if neither declares one statically.
            The bridge crops from a tensor with the detector's channel count and
            feeds it to the classifier unchanged, so a mismatch cannot be
            bridged — it means the two models were trained on different image
            formats.
    """
    detector_channels = detector_shape[1] if isinstance(detector_shape[1], int) else None
    classifier_channels = classifier_shape[1] if isinstance(classifier_shape[1], int) else None
    if (
        detector_channels is not None
        and classifier_channels is not None
        and detector_channels != classifier_channels
    ):
        raise FusionError(
            f"The detector takes {detector_channels}-channel images and the classifier takes "
            f"{classifier_channels}-channel ones; the crops cannot be handed over unchanged."
        )
    channels = detector_channels if detector_channels is not None else classifier_channels
    if channels is None:
        raise FusionError(
            "Neither model declares a static channel count, so the crop tensor cannot be shaped. "
            "Re-export one of them with a fixed channel dimension."
        )
    return channels


def _resolve_size(
    *,
    declared: list[int | str | None],
    requested: tuple[int, int] | None,
    role: str,
    argument: str,
) -> tuple[int, int]:
    """Settle the ``(width, height)`` a stage runs at.

    Precedence mirrors :func:`~ort_vision_sdk.graph.resolve_input_size`: what
    the graph declares statically wins, because that is the only shape ONNX
    Runtime will accept for it. The argument is there for exports that left
    their spatial axes dynamic, where the fusion has to pin a resolution — a
    fused pipeline records one input size in its metadata, so it cannot stay
    open-ended.

    Args:
        declared: Declared NCHW shape of the stage's input.
        requested: Size the caller asked for, if any.
        role: ``"detector"`` or ``"classifier"``, used in messages.
        argument: Name of the keyword argument that supplies it.

    Returns:
        tuple[int, int]: The ``(width, height)`` to use.

    Raises:
        FusionError: If the graph is dynamic and the caller supplied nothing.
    """
    height, width = declared[2], declared[3]
    if isinstance(height, int) and isinstance(width, int):
        if requested is not None and requested != (width, height):
            warnings.warn(
                f"The {role} model declares a {width}x{height} input; ignoring the requested "
                f"{argument}={requested}, which ONNX Runtime would reject.",
                UserWarning,
                stacklevel=3,
            )
        return width, height
    if requested is None:
        raise FusionError(
            f"The {role} model leaves its spatial axes dynamic, so the fused pipeline has no "
            f"resolution to record. Pass {argument}=(width, height)."
        )
    return requested


def _check_detector_head(model: onnx.ModelProto) -> None:
    """Verify the detector emits the anchor-free YOLO layout the bridge decodes.

    Args:
        model: The detector model.

    Raises:
        FusionError: If the model has no outputs, or its first output is not a
            3-D tensor. A different rank means a different head family (YOLOv5's
            ``(1, N, 5 + nc)``, YOLOv10's ``(1, 300, 6)``), and the bridge's
            slicing would silently read the wrong channels rather than fail.
    """
    if not model.graph.output:
        raise FusionError("The detector model declares no outputs.")
    dims = model.graph.output[0].type.tensor_type.shape.dim
    if len(dims) != 3:
        raise FusionError(
            f"The detector's output {model.graph.output[0].name!r} must be the anchor-free YOLO "
            f"layout (1, 4 + num_classes, num_anchors), got {len(dims)} dimensions. Heads with "
            "an explicit objectness channel (YOLOv5/v6/v7) or built-in NMS (YOLOv10 end2end) "
            "are not supported."
        )
    channels = dims[1].dim_value if dims[1].HasField("dim_value") else 0
    if channels and channels < 5:
        raise FusionError(
            f"The detector's output declares {channels} channels; the anchor-free YOLO layout "
            "needs at least 5 (4 box coordinates plus one class score)."
        )


def _head_classes(model: onnx.ModelProto) -> int | None:
    """Read ``num_classes`` off the detector head's channel count, if it is static."""
    dims = model.graph.output[0].type.tensor_type.shape.dim
    if not dims[1].HasField("dim_value"):
        return None
    return int(dims[1].dim_value) - 4


def _classifier_classes(model: onnx.ModelProto) -> int | None:
    """Read ``num_classes`` off the classifier output's last dimension, if it is static."""
    if not model.graph.output:
        return None
    dims = model.graph.output[0].type.tensor_type.shape.dim
    if not dims or not dims[-1].HasField("dim_value"):
        return None
    return int(dims[-1].dim_value)


def _stage_names(
    model: onnx.ModelProto,
    spec: LabelSpec,
    num_classes: int | None,
) -> dict[int, str] | None:
    """Decide the class map to record for one stage of the pipeline.

    Args:
        model: The stage's source model, read for the ``names`` its exporter
            baked in.
        spec: An explicit label spec from the caller, or ``None``.
        num_classes: Class count read off the graph, used to size a generated
            or file-loaded map.

    Returns:
        dict[int, str] | None: Class id → name, or ``None`` when the caller
        passed nothing and the model carries no usable ``names``. ``None`` is
        not a failure: the runtime falls back to the same defaults a standalone
        task would (COCO for detection, generated ``class_N`` otherwise).
    """
    if spec is None:
        return model_names({entry.key: entry.value for entry in model.metadata_props})
    labels = resolve_labels(spec, num_classes=num_classes)
    return dict(enumerate(labels))


def _ends_in_softmax(model: onnx.ModelProto) -> bool:
    """Whether the classifier's final output is produced by a ``Softmax`` node.

    Args:
        model: The classifier model.

    Returns:
        bool: ``True`` when the graph already emits probabilities, so the
        runtime must not apply softmax a second time.
    """
    if not model.graph.output:
        return False
    target = model.graph.output[0].name
    return any(target in node.output and node.op_type == "Softmax" for node in model.graph.node)


def _default_opset(model: onnx.ModelProto) -> int:
    """Read a model's opset version for the default (``ai.onnx``) domain."""
    for entry in model.opset_import:
        if entry.domain in ("", "ai.onnx"):
            return int(entry.version)
    return MIN_OPSET


def _align_opset(model: onnx.ModelProto, target: int, *, role: str) -> onnx.ModelProto:
    """Raise a model's default-domain opset to ``target``.

    A single graph carries a single opset per domain, so two models exported at
    different versions cannot simply be concatenated. Only upgrades are
    attempted — downgrading a model to meet an older peer would silently drop
    whatever newer semantics it relied on.

    Args:
        model: The model to convert.
        target: Opset version to reach.
        role: ``"detector"`` or ``"classifier"``, used in the error message.

    Returns:
        onnx.ModelProto: The converted model, or ``model`` unchanged when it is
        already at ``target``.

    Raises:
        FusionError: If the converter cannot express the model at ``target``.
    """
    current = _default_opset(model)
    if current >= target:
        return model
    try:
        return onnx.version_converter.convert_version(model, target)
    except Exception as exc:
        raise FusionError(
            f"The {role} model is exported at opset {current} and cannot be converted to opset "
            f"{target}, which the other stage (or the bridge's RoiAlign) requires: {exc}. "
            f"Re-export it at opset {target} or later."
        ) from exc


def _resolve_per_class_cap(requested: int | None, max_detections: int | None) -> int:
    """Pick the NMS per-class cap.

    Args:
        requested: An explicit cap from the caller, or ``None``.
        max_detections: The pipeline's global row count, or ``None`` in dynamic
            mode.

    Returns:
        int: The cap to bake into the NMS node. The default leaves four times
        the global budget per class so that one crowded class cannot consume
        the whole ranking before the others are considered.
    """
    if requested is not None:
        return requested
    return 300 if max_detections is None else max_detections * 4


def _assemble(
    *,
    detector_model: onnx.ModelProto,
    classifier_model: onnx.ModelProto,
    spec: FusionSpec,
    channels: int,
    max_boxes_per_class: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    input_scale: float,
    sampling_ratio: int,
    target_opset: int,
) -> onnx.ModelProto:
    """Splice the detector, the bridge and the classifier into one graph.

    Both source graphs are renamed under a prefix first: they were exported
    independently and are free to use the same tensor names, which in a single
    graph would silently cross-wire them. The pipeline's own public names
    (``images``, ``probs``) are then bound with ``Identity`` nodes rather than
    by renaming, so neither source graph has to be rewritten a second time.

    Args:
        detector_model: The detector, already at ``target_opset``.
        classifier_model: The classifier, already at ``target_opset``.
        spec: The pipeline configuration, recorded into the result's metadata.
        channels: Image channel count.
        max_boxes_per_class: NMS per-class cap.
        mean: Per-channel normalization mean.
        std: Per-channel normalization standard deviation.
        input_scale: Multiplier applied before normalization.
        sampling_ratio: RoiAlign samples per output bin.
        target_opset: Opset the fused graph declares.

    Returns:
        onnx.ModelProto: The fused, checked model.

    Raises:
        FusionError: If the assembled graph fails ONNX's structural checker.
    """
    detector_model = compose.add_prefix(detector_model, _DETECTOR_PREFIX)
    classifier_model = compose.add_prefix(classifier_model, _CLASSIFIER_PREFIX)

    detector_input = detector_model.graph.input[0].name
    detector_output = detector_model.graph.output[0].name
    classifier_input = classifier_model.graph.input[0].name
    classifier_output = classifier_model.graph.output[0].name

    bridge = build_bridge(
        detector_output=detector_output,
        detector_input=detector_input,
        classifier_input=classifier_input,
        crop_size=spec.crop_size,
        channels=channels,
        crop_source=spec.crop_source,
        max_detections=spec.max_detections,
        conf_threshold=spec.conf_threshold,
        iou_threshold=spec.iou_threshold,
        max_boxes_per_class=max_boxes_per_class,
        mean=mean,
        std=std,
        input_scale=input_scale,
        sampling_ratio=sampling_ratio,
    )

    width, height = spec.input_size
    image_input = helper.make_tensor_value_info(
        INPUT_IMAGE, TensorProto.FLOAT, [1, channels, height, width]
    )
    probs_output = _probs_value_info(classifier_model, rows=spec.max_detections)

    nodes = [
        helper.make_node("Identity", [INPUT_IMAGE], [detector_input], name="ovs_bind_input"),
        *detector_model.graph.node,
        *bridge.nodes,
        *classifier_model.graph.node,
        helper.make_node("Identity", [classifier_output], [OUTPUT_PROBS], name="ovs_bind_probs"),
    ]
    graph = helper.make_graph(
        nodes=nodes,
        name=_GRAPH_NAME,
        inputs=[image_input, *bridge.inputs],
        outputs=[*bridge.outputs, probs_output],
        initializer=[
            *detector_model.graph.initializer,
            *bridge.initializers,
            *classifier_model.graph.initializer,
        ],
        value_info=[
            *detector_model.graph.value_info,
            *classifier_model.graph.value_info,
            _crop_value_info(classifier_input, spec=spec, channels=channels),
        ],
    )

    model = helper.make_model(
        graph,
        opset_imports=_merged_opsets(detector_model, classifier_model, target_opset),
        functions=[*detector_model.functions, *classifier_model.functions],
        producer_name="ort-vision-sdk",
        producer_version=spec.sdk_version,
    )
    model.ir_version = max(detector_model.ir_version, classifier_model.ir_version)
    _write_metadata(model, detector_model, spec)

    try:
        onnx.checker.check_model(model)
    except Exception as exc:
        raise FusionError(f"The fused graph is not a valid ONNX model: {exc}") from exc
    return model


def _probs_value_info(
    classifier_model: onnx.ModelProto, *, rows: int | None
) -> onnx.ValueInfoProto:
    """Declare the classifier output under the pipeline's public name.

    Args:
        classifier_model: The prefixed classifier, read for its output's class
            count.
        rows: The pipeline's fixed row count, or ``None`` in dynamic mode.

    Returns:
        onnx.ValueInfoProto: A ``(rows, num_classes)`` float32 output named
        ``probs``, with either axis left symbolic when it is not known.
    """
    num_classes = _classifier_classes(classifier_model)
    first: int | str = rows if rows is not None else _CROP_BATCH_DIM
    second: int | str = num_classes if num_classes is not None else "num_classifier_classes"
    return helper.make_tensor_value_info(OUTPUT_PROBS, TensorProto.FLOAT, [first, second])


def _crop_value_info(
    classifier_input: str, *, spec: FusionSpec, channels: int
) -> onnx.ValueInfoProto:
    """Declare the crop batch the bridge hands to the classifier.

    The classifier's input stops being a graph input once it is spliced, so its
    shape has to be restated as an internal value — including the batch size,
    which is now the pipeline's detection count rather than whatever the
    standalone export declared.

    Args:
        classifier_input: Prefixed name of the classifier's input tensor.
        spec: The pipeline configuration.
        channels: Image channel count.

    Returns:
        onnx.ValueInfoProto: The crop batch's shape and type.
    """
    crop_width, crop_height = spec.crop_size
    batch: int | str = spec.max_detections if spec.max_detections is not None else _CROP_BATCH_DIM
    return helper.make_tensor_value_info(
        classifier_input, TensorProto.FLOAT, [batch, channels, crop_height, crop_width]
    )


def _merged_opsets(
    detector_model: onnx.ModelProto,
    classifier_model: onnx.ModelProto,
    target_opset: int,
) -> list[onnx.OperatorSetIdProto]:
    """Union both models' operator-set imports, pinning the default domain.

    Args:
        detector_model: The detector.
        classifier_model: The classifier.
        target_opset: Version to declare for the default ``ai.onnx`` domain.

    Returns:
        list[onnx.OperatorSetIdProto]: One entry per domain, custom domains
        taking the higher of the two versions.
    """
    versions: dict[str, int] = {"": target_opset}
    for model in (detector_model, classifier_model):
        for entry in model.opset_import:
            if entry.domain in ("", "ai.onnx"):
                continue
            versions[entry.domain] = max(versions.get(entry.domain, 0), int(entry.version))
    return [helper.make_opsetid(domain, version) for domain, version in versions.items()]


def _write_metadata(
    model: onnx.ModelProto,
    detector_model: onnx.ModelProto,
    spec: FusionSpec,
) -> None:
    """Record the pipeline spec — and the detector's own metadata — on the model.

    The detector's entries (``names``, ``task``, ``imgsz`` for an Ultralytics
    export) are carried over so tools that inspect the file still see what the
    detection stage is, and the ``ovs.*`` entries are written on top so the
    runtime never has to guess.

    Args:
        model: The fused model to annotate.
        detector_model: The detector, read for its metadata entries.
        spec: The pipeline configuration.
    """
    entries = {
        entry.key: entry.value
        for entry in detector_model.metadata_props
        if not entry.key.startswith(METADATA_PREFIX)
    }
    entries.update(spec.to_metadata())
    del model.metadata_props[:]
    for key, value in entries.items():
        model.metadata_props.add(key=key, value=value)


def _validate(
    model: onnx.ModelProto,
    *,
    spec: FusionSpec,
    channels: int,
    saved_to: str | Path | None,
) -> None:
    """Run the fused graph once, so a broken splice fails here rather than in production.

    The probe feeds an all-zero image, which is the harshest case for the
    padding logic: nothing clears the confidence threshold, so NMS selects no
    boxes and every row the pipeline reports is a padded one. A classifier that
    cannot take the crop batch — the usual cause being a hardcoded batch size
    baked into its own ``Reshape`` nodes, which rewriting the declared input
    shape does not touch — fails right here with ORT's own message.

    Args:
        model: The fused model.
        spec: The pipeline configuration, used to shape the probe inputs.
        channels: Image channel count.
        saved_to: Path the model was written to, if any. Loading from disk
            avoids re-serializing a model that may exceed protobuf's 2 GB limit.

    Raises:
        FusionError: If ONNX Runtime refuses to load or run the fused graph.
    """
    import onnxruntime as ort

    width, height = spec.input_size
    feeds: dict[str, np.ndarray] = {
        INPUT_IMAGE: np.zeros((1, channels, height, width), dtype=np.float32)
    }
    if spec.needs_source_image:
        feeds[INPUT_SOURCE] = np.zeros((1, channels, height * 2, width * 2), dtype=np.float32)
        feeds[INPUT_SCALE] = np.asarray([0.5], dtype=np.float32)
        feeds[INPUT_PAD] = np.asarray([0.0, 0.0], dtype=np.float32)

    try:
        source = str(saved_to) if saved_to is not None else model.SerializeToString()
        session = ort.InferenceSession(source, providers=["CPUExecutionProvider"])
        session.run(None, feeds)
    except Exception as exc:
        raise FusionError(
            f"The fused pipeline was built but ONNX Runtime could not run it: {exc}. "
            "The usual cause is a classifier whose graph only accepts the batch size it was "
            "exported with; re-export it with a dynamic batch axis, or set "
            f"max_detections to match it."
        ) from exc
