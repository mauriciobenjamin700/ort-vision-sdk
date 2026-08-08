"""What the ONNX graph itself says about the model.

Two properties of an export keep being restated in configuration, and any
restatement is free to drift from the file it describes:

- **The input resolution.** Feeding a 640x640 tensor to a graph exported at
  224x224 makes ORT abort the run with ``Got invalid dimensions for input:
  images ... Got: 640 Expected: 224``. The number lives in the graph, so the SDK
  reads it there and treats a configured ``input_size`` as a fallback for when
  the graph leaves the axes dynamic.
- **The class names.** Ultralytics bakes ``names`` into the model's metadata as
  the ``repr`` of a ``dict[int, str]``. A hand-maintained label list next to it
  can be reordered by accident, which silently swaps predictions between classes
  instead of failing. When the caller passes no labels, the SDK reads the ones
  the model was exported with.
"""

from __future__ import annotations

import ast
import warnings

__all__ = [
    "model_names",
    "parse_names",
    "resolve_input_size",
    "spatial_input_size",
]


def spatial_input_size(
    shape: tuple[int | str, ...] | None,
) -> tuple[int, int] | None:
    """Read the spatial input size out of a declared NCHW shape.

    Args:
        shape (tuple[int | str, ...] | None): Declared shape of the model's
            image input, dynamic axes appearing as strings (as ONNX Runtime
            reports them).

    Returns:
        tuple[int, int] | None: ``(width, height)`` in pixels, or ``None`` when
        the shape is not 4D or leaves either spatial axis dynamic — in which
        case the model accepts more than one resolution and there is nothing to
        correct.
    """
    if shape is None or len(shape) != 4:
        return None
    height, width = shape[2], shape[3]
    if not isinstance(height, int) or not isinstance(width, int):
        return None
    if height < 1 or width < 1:
        return None
    return width, height


def resolve_input_size(
    *,
    graph_shape: tuple[int | str, ...] | None,
    requested: tuple[int, int] | None,
    fallback: tuple[int, int],
) -> tuple[int, int]:
    """Decide the input size a task will preprocess to.

    Precedence is graph → caller → fallback. The graph wins over an explicit
    ``input_size`` because a static shape is not a preference, it is what ORT
    will accept: honoring the caller there would only turn a fixable mismatch
    into a failed run. A disagreement is a configuration bug in the caller, so
    it is surfaced as a :class:`UserWarning` instead of being swallowed.

    Args:
        graph_shape (tuple[int | str, ...] | None): Declared shape of the
            model's image input.
        requested (tuple[int, int] | None): Size the caller asked for, if any.
        fallback (tuple[int, int]): Size to use when neither the graph nor the
            caller pins one.

    Returns:
        tuple[int, int]: The ``(width, height)`` to preprocess to.
    """
    graph = spatial_input_size(graph_shape)
    if graph is None:
        return requested if requested is not None else fallback
    if requested is not None and requested != graph:
        warnings.warn(
            f"The model declares a {graph[0]}x{graph[1]} input; ignoring the "
            f"requested {requested[0]}x{requested[1]}, which ONNX Runtime "
            f"would reject.",
            UserWarning,
            stacklevel=3,
        )
    return graph


def model_names(metadata: dict[str, str] | None) -> dict[int, str] | None:
    """Read the class names an export baked into the model metadata.

    Ultralytics writes ``names`` as the ``repr`` of a ``dict[int, str]`` — e.g.
    ``"{0: 'deworm', 1: 'not_deworm'}"``. Parsing is done with
    :func:`ast.literal_eval`, so a malformed or hostile value cannot execute
    anything; anything unparseable, non-dict, or not keyed by contiguous
    integers from zero is rejected as unusable rather than half-applied.

    Args:
        metadata (dict[str, str] | None): The model's custom metadata map, or
            ``None`` when the backend does not expose one.

    Returns:
        dict[int, str] | None: Class id → name, or ``None`` when the model
        carries no usable ``names`` entry.
    """
    if not metadata:
        return None
    return parse_names(metadata.get("names"))


def parse_names(raw: str | None) -> dict[int, str] | None:
    """Parse a ``repr``-encoded ``dict[int, str]`` class map.

    Split out of :func:`model_names` because the same encoding is reused by a
    fused pipeline, which carries one class map per stage and therefore cannot
    store both under the single ``names`` key Ultralytics uses.

    Args:
        raw (str | None): The encoded map — e.g. ``"{0: 'deworm', 1: 'not_deworm'}"``.

    Returns:
        dict[int, str] | None: Class id → name, or ``None`` when ``raw`` is
        empty, unparseable, not a dict, or not keyed by contiguous integers
        from zero. Half-valid maps are rejected outright: a partially applied
        label map silently renames the wrong classes.
    """
    if not raw:
        return None
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return None
    if not isinstance(parsed, dict) or not parsed:
        return None
    names: dict[int, str] = {}
    for key, value in parsed.items():
        if not isinstance(key, int) or not isinstance(value, str):
            return None
        names[key] = value
    if sorted(names) != list(range(len(names))):
        return None
    return names
