"""Which preprocessing a classifier expects its input to have had.

A classifier is trained on a specific tensor, and feeding it a differently
prepared one degrades it silently — no exception, no warning, just worse
predictions. The two families this SDK sees most disagree completely:
torchvision-style models want the ImageNet mean and deviation subtracted and
divided out, while an Ultralytics classification head consumes raw ``[0, 1]``.

Guessing wrong is not detectable from the outside, but it is not a guess: an
Ultralytics export stamps ``author`` and ``task`` into its own metadata, and
every task in this SDK already reads that map for the class names. This module
turns that into the decision, so a caller who has nothing to say about
normalization gets the right one.

It deliberately imports nothing beyond the standard library: the fusion
(:mod:`ort_vision_sdk.compose`) needs it at build time and
:class:`~ort_vision_sdk.tasks.classifier.Classifier` needs it at run time, and
those two live behind different install extras.
"""

from __future__ import annotations

import warnings
from typing import Literal

__all__ = [
    "IDENTITY_MEAN",
    "IDENTITY_STD",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "NORMALIZATION_PRESETS",
    "Normalization",
    "is_ultralytics_classifier",
    "resolve_normalization",
]

Normalization = Literal["auto", "imagenet", "ultralytics", "none"]
"""Which preprocessing the classifier expects its input to have had.

- ``"auto"`` (default everywhere it is accepted) reads the model's own export
  metadata and picks. An Ultralytics classification head gets
  ``"ultralytics"``; anything else gets ``"imagenet"``.
- ``"imagenet"`` subtracts the ImageNet mean and divides by the ImageNet
  deviation — the torchvision convention.
- ``"ultralytics"`` leaves the image in ``[0, 1]``. Ultralytics' own classifier
  applies no mean/std at all, so anything else feeds it images it never saw in
  training.
- ``"none"`` is the same arithmetic as ``"ultralytics"`` — identity — under a
  name that says "this model wants raw ``[0, 1]``" rather than naming a vendor.
"""

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
"""Per-channel RGB mean of ImageNet, the torchvision preprocessing convention."""

IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
"""Per-channel RGB standard deviation of ImageNet."""

IDENTITY_MEAN: tuple[float, float, float] = (0.0, 0.0, 0.0)
"""Mean that leaves an image untouched — what an Ultralytics classifier expects."""

IDENTITY_STD: tuple[float, float, float] = (1.0, 1.0, 1.0)
"""Deviation that leaves an image untouched — what an Ultralytics classifier expects."""

NORMALIZATION_PRESETS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "imagenet": (IMAGENET_MEAN, IMAGENET_STD),
    "ultralytics": (IDENTITY_MEAN, IDENTITY_STD),
    "none": (IDENTITY_MEAN, IDENTITY_STD),
}
"""Named presets → their ``(mean, std)``."""

CUSTOM_NORMALIZATION = "custom"
"""Name reported when the caller supplied ``mean``/``std`` directly."""


def is_ultralytics_classifier(metadata: dict[str, str]) -> bool:
    """Whether a metadata map came out of ``YOLO(...).export(format="onnx")``.

    Every Ultralytics export stamps ``author`` and ``task`` into
    ``metadata_props``, and the pair is unambiguous: ``"Ultralytics"`` plus
    ``"classify"`` is a classification head from that codebase and nothing else.

    Args:
        metadata: The model's custom metadata map.

    Returns:
        bool: ``True`` for an Ultralytics classification export.
    """
    return (
        metadata.get("author", "").strip().lower() == "ultralytics"
        and metadata.get("task", "").strip().lower() == "classify"
    )


def resolve_normalization(
    metadata: dict[str, str],
    *,
    normalization: Normalization,
    mean: tuple[float, float, float] | None,
    std: tuple[float, float, float] | None,
    stacklevel: int = 3,
) -> tuple[str, tuple[float, float, float], tuple[float, float, float]]:
    """Settle which ``(mean, std)`` to apply, and what to call the choice.

    Explicit ``mean``/``std`` always win — they are the escape hatch for a model
    whose preprocessing neither preset describes. Anything they leave open falls
    back to the preset, so passing only a ``mean`` does not silently reset the
    deviation to 1.

    Args:
        metadata: The model's custom metadata map, read to detect the family.
        normalization: The preset the caller asked for, or ``"auto"``.
        mean: Explicit per-channel mean, or ``None``.
        std: Explicit per-channel deviation, or ``None``.
        stacklevel: Stack level for the warning, so it points at the caller's
            own line rather than at this module.

    Returns:
        tuple[str, tuple[float, float, float], tuple[float, float, float]]: The
        name of the choice, followed by the mean and deviation to apply.

    Raises:
        ValueError: If ``normalization`` is not a known preset, or names one
            while ``mean``/``std`` are also supplied — two answers to the same
            question, and guessing which one the caller meant is how a model
            ends up preprocessed differently from what it reports.

    Warns:
        UserWarning: If the model is an Ultralytics export and the supplied
            ``mean``/``std`` are not the identity it was trained with. Nothing
            fails in that case: the prediction has the right shape and is simply
            less accurate, which is exactly why it is worth saying out loud.
    """
    explicit = mean is not None or std is not None
    if normalization not in ("auto", *NORMALIZATION_PRESETS):
        raise ValueError(
            f"normalization must be one of 'auto', 'imagenet', 'ultralytics', 'none'; "
            f"got {normalization!r}."
        )
    if explicit and normalization != "auto":
        raise ValueError(
            f"Pass either normalization={normalization!r} or explicit mean/std, not both."
        )

    ultralytics = is_ultralytics_classifier(metadata)
    preset = normalization
    if preset == "auto":
        preset = "ultralytics" if ultralytics else "imagenet"
    preset_mean, preset_std = NORMALIZATION_PRESETS[preset]

    if not explicit:
        return preset, preset_mean, preset_std

    resolved_mean = mean if mean is not None else preset_mean
    resolved_std = std if std is not None else preset_std
    if ultralytics and (resolved_mean, resolved_std) != (IDENTITY_MEAN, IDENTITY_STD):
        warnings.warn(
            "The classifier is an Ultralytics export, whose classification head is trained on "
            f"raw [0, 1] images, but mean={resolved_mean} / std={resolved_std} was requested. "
            "It will be fed images normalized in a way it never saw in training, which degrades "
            "accuracy without raising anything. Drop mean/std to let normalization='auto' pick "
            "the identity.",
            UserWarning,
            stacklevel=stacklevel,
        )
    return CUSTOM_NORMALIZATION, resolved_mean, resolved_std
