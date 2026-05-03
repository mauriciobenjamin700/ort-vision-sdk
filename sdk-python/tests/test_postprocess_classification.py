"""Tests for ``ort_vision_sdk.postprocess.classification`` (softmax + topk)."""

from __future__ import annotations

import numpy as np
import pytest

from ort_vision_sdk.postprocess.classification import softmax, topk


class TestSoftmax:
    def test_sums_to_one(self) -> None:
        logits = np.array([2.0, 1.0, 0.1, 5.0, -1.0])
        probs = softmax(logits)
        assert probs.sum() == pytest.approx(1.0, abs=1e-6)

    def test_all_in_unit_interval(self) -> None:
        logits = np.array([100.0, -100.0, 0.0, 50.0])
        probs = softmax(logits)
        assert (probs >= 0).all()
        assert (probs <= 1).all()
        # Numerical stability: doesn't overflow on large logits.
        assert not np.isnan(probs).any()
        assert not np.isinf(probs).any()

    def test_argmax_preserved(self) -> None:
        logits = np.array([0.1, 0.5, 5.0, 0.2, 0.3])
        probs = softmax(logits)
        assert int(np.argmax(probs)) == int(np.argmax(logits))

    def test_dtype_is_float32(self) -> None:
        logits = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        probs = softmax(logits)
        assert probs.dtype == np.float32

    def test_does_not_mutate_input(self) -> None:
        logits = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        original = logits.copy()
        softmax(logits)
        np.testing.assert_array_equal(logits, original)


class TestTopK:
    def test_returns_top_k_descending(self) -> None:
        probs = np.array([0.05, 0.10, 0.50, 0.25, 0.10])
        idx, val = topk(probs, k=3)
        assert idx.tolist() == [2, 3, 1]  # 0.50, 0.25, 0.10 (first 0.10 wins via stable sort)
        np.testing.assert_array_almost_equal(val, [0.50, 0.25, 0.10])

    def test_k_none_returns_all_sorted(self) -> None:
        probs = np.array([0.3, 0.5, 0.2])
        idx, val = topk(probs, k=None)
        assert idx.tolist() == [1, 0, 2]
        assert val.tolist() == [0.5, 0.3, 0.2]

    def test_k_larger_than_n_clamps(self) -> None:
        probs = np.array([0.6, 0.4])
        idx, val = topk(probs, k=10)
        assert idx.shape == (2,)
        assert val.shape == (2,)

    def test_rejects_non_1d(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            topk(np.zeros((3, 4)), k=2)
