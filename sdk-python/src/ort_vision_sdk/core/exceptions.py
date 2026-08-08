"""Exceptions raised by the SDK.

All exceptions inherit from :class:`OrtVisionError`, so callers can catch the
base class to handle any SDK-originated failure uniformly.
"""

from __future__ import annotations


class OrtVisionError(Exception):
    """Base class for all ort-vision-sdk errors."""


class ModelLoadError(OrtVisionError):
    """Raised when an ONNX model cannot be loaded into an inference session."""


class InferenceError(OrtVisionError):
    """Raised when ONNX Runtime fails while executing a model."""


class ProviderNotAvailableError(OrtVisionError):
    """Raised when a requested execution provider is not available in this ORT build."""


class ImageLoadError(OrtVisionError):
    """Raised when an input image cannot be decoded into the canonical array format."""


class LabelMapError(OrtVisionError):
    """Raised when class labels cannot be resolved from the supplied spec."""


class NoDetectionsError(OrtVisionError):
    """Raised when a detection task finds nothing and was asked to treat that as an error.

    Only raised when the caller opts in with ``raise_on_empty=True``. The
    default stays an empty result, because "the model looked and found nothing"
    is a successful inference, not a failure — a photo of an empty field is a
    valid photo. What the flag is for is the opposite situation: a pipeline step
    whose *precondition* is that something is there, where an empty result means
    the caller should stop rather than quietly carry on with zero rows.

    "Nothing was detected" and "nothing was confident enough" are the same
    condition here, since the confidence threshold is what decides what counts
    as a detection in the first place. Raise the threshold and the error covers
    the stricter bar.
    """


class FusionError(OrtVisionError):
    """Raised when two models cannot be fused into a single pipeline graph.

    Covers every way :mod:`ort_vision_sdk.compose` can refuse a pair of models:
    an unreadable file, a head layout the bridge cannot decode, opsets that
    cannot be reconciled, a classifier whose graph will not accept a batch of
    crops, or a fused graph that fails its post-build validation run.
    """
