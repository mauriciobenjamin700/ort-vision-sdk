"""Class label resolution: presets, lists, dicts, file paths, or auto-generated.

Tasks call :func:`resolve_labels` once at construction time to turn whatever
the caller passed (preset name, list, dict, file path, or ``None``) into an
ordered tuple of class names indexed by class id.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

from ort_vision_sdk.core.exceptions import LabelMapError

LabelSpec: TypeAlias = (
    list[str] | tuple[str, ...] | dict[int, str] | str | Path | None
)
"""Anything accepted by :func:`resolve_labels`.

- ``list[str]`` / ``tuple[str, ...]``: explicit names indexed by class id.
- ``dict[int, str]``: sparse mapping (gaps filled with ``"class_<id>"``).
- ``str``: either a preset name (e.g. ``"coco"``) or a file path.
- ``Path``: filesystem path to a labels file (one class per line).
- ``None``: auto-generate ``"class_0"`` ... ``"class_{num_classes-1}"``.
"""


COCO_CLASSES: tuple[str, ...] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
)
"""COCO 2017 80-class labels in canonical class-id order."""


_PRESETS: dict[str, tuple[str, ...]] = {
    "coco": COCO_CLASSES,
}


def resolve_labels(
    spec: LabelSpec,
    *,
    num_classes: int | None = None,
) -> tuple[str, ...]:
    """Resolve a labels specification into an ordered tuple of class names.

    Args:
        spec: A :data:`LabelSpec`.
        num_classes: Expected number of classes. Required when ``spec`` is
            ``None``. When provided alongside an explicit spec, the resolved
            length is validated against this value.

    Returns:
        Tuple of class names with index = class id.

    Raises:
        LabelMapError: If the spec is invalid, the file is missing or empty,
            the preset is unknown, or the resolved length disagrees with
            ``num_classes``.
    """
    labels = _resolve(spec, num_classes)
    if num_classes is not None and len(labels) != num_classes:
        raise LabelMapError(
            f"Resolved {len(labels)} labels but the model has {num_classes} classes."
        )
    return labels


def _resolve(spec: LabelSpec, num_classes: int | None) -> tuple[str, ...]:
    """Dispatch the spec by type to the appropriate resolver.

    Args:
        spec: The :data:`LabelSpec` to resolve.
        num_classes: Expected number of classes; only consulted when ``spec``
            is ``None`` to drive auto-generation.

    Returns:
        Tuple of class names indexed by class id.

    Raises:
        LabelMapError: If ``spec`` is ``None`` without ``num_classes``, or if
            the spec is unsupported, the file is missing/empty, or the preset
            name is unknown.
    """
    if spec is None:
        if num_classes is None:
            raise LabelMapError(
                "Cannot auto-generate labels without num_classes. "
                "Pass an explicit labels spec or a model whose output shape is statically known."
            )
        return tuple(f"class_{i}" for i in range(num_classes))

    if isinstance(spec, (list, tuple)):
        return tuple(spec)

    if isinstance(spec, dict):
        if not spec:
            return ()
        max_id = max(spec.keys())
        return tuple(spec.get(i, f"class_{i}") for i in range(max_id + 1))

    if isinstance(spec, Path):
        return _load_file(spec)

    if isinstance(spec, str):
        if spec in _PRESETS:
            return _PRESETS[spec]
        path = Path(spec)
        if path.is_file():
            return _load_file(path)
        raise LabelMapError(
            f"Unknown labels preset or missing file: {spec!r}. "
            f"Known presets: {sorted(_PRESETS)}."
        )

    raise LabelMapError(f"Unsupported labels spec type: {type(spec).__name__}.")


def _load_file(path: Path) -> tuple[str, ...]:
    """Read a labels file with one class name per line."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise LabelMapError(f"Failed to read labels file {path}: {exc}") from exc
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise LabelMapError(f"Labels file {path} is empty.")
    return tuple(lines)
