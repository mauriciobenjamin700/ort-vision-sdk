"""Classification head postprocessing — softmax + top-k."""

from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Apply numerically-stable softmax along an axis.

    Args:
        logits: Input array.
        axis: Axis along which to normalize (default: last).

    Returns:
        A ``float32`` array of the same shape as ``logits``, summing to 1
        along ``axis``.
    """
    x = logits.astype(np.float32, copy=True)
    x -= np.max(x, axis=axis, keepdims=True)
    np.exp(x, out=x)
    x /= np.sum(x, axis=axis, keepdims=True)
    return x


def topk(probabilities: np.ndarray, k: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Return the top-k entries of a 1-D probability vector, sorted descending.

    Args:
        probabilities: 1-D probability vector of shape ``(num_classes,)``.
        k: Number of entries to return. ``None`` returns all entries.

    Returns:
        Tuple ``(indices, values)``, both of shape ``(min(k, num_classes),)``.

    Raises:
        ValueError: If ``probabilities`` is not 1-D.
    """
    if probabilities.ndim != 1:
        raise ValueError(f"Expected a 1-D probability vector, got shape {probabilities.shape}.")
    n = probabilities.shape[0]
    k_eff = n if k is None else min(k, n)
    indices = np.argsort(-probabilities, kind="stable")[:k_eff]
    values = probabilities[indices]
    return indices, values
