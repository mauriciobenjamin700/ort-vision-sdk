"""Tests for ``ort_vision_sdk.preprocess.image``.

The preprocessing primitives had no direct tests: they were only exercised
through the tasks, which meant a change to ``resize`` or ``letterbox`` could
only be caught by a downstream assertion about boxes. Since ``resize`` now
takes a two-step path on a downscale, the guard deciding *when* that applies
needs tests of its own — the interesting cases are the ones where it must
**not** engage.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from ort_vision_sdk.preprocess.image import (
    add_batch_dim,
    from_cv2,
    letterbox,
    normalize,
    reduction_factor,
    resize,
    to_chw,
    to_cv2,
    to_tensor,
)


def gradient_image(width: int, height: int) -> np.ndarray:
    """Build a smooth HWC uint8 RGB image, deterministic in its dimensions."""
    rows, cols = np.mgrid[0:height, 0:width].astype(np.float32)
    red = (cols / max(width - 1, 1)) * 255.0
    green = (rows / max(height - 1, 1)) * 255.0
    blue = ((cols + rows) / max(width + height - 2, 1)) * 255.0
    return np.stack([red, green, blue], axis=-1).astype(np.uint8)


def detailed_image(width: int, height: int) -> np.ndarray:
    """Build an image carrying detail a downscale can actually destroy.

    :func:`gradient_image` is a linear ramp, which every linear filter — down
    to nearest-neighbour — reproduces exactly. Comparing resamplers on it
    yields zero error for all of them, so any test about resampling quality
    needs edges and curvature instead.
    """
    rows, cols = np.mgrid[0:height, 0:width].astype(np.float32)
    base = 128.0 + 90.0 * np.sin(cols / 21.0) * np.cos(rows / 17.0)
    array = np.stack([base, base * 0.75 + 40.0, base * 0.5 + 80.0], axis=-1)
    image = Image.fromarray(array.clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    step = max(width // 14, 1)
    for index in range(14):
        left = 20 + index * step
        top = 20 + (index % 5) * max(height // 8, 1)
        draw.ellipse(
            [left, top, left + step // 2, top + step // 2], outline=(255, 250, 240), width=3
        )
        draw.rectangle([left, top + height // 2, left + step // 3, top + height // 2 + 40], width=2)
    return np.asarray(image, dtype=np.uint8)


class TestReductionFactor:
    @pytest.mark.parametrize(
        ("source", "target", "expected"),
        [
            ((1920, 1080), (640, 360), 3),
            ((3840, 2160), (640, 360), 6),
            ((1600, 900), (640, 360), 2),
            ((1280, 720), (640, 360), 2),
            ((800, 600), (640, 360), 1),
            ((640, 360), (640, 360), 1),
            ((320, 240), (640, 640), 1),
            ((1920, 1080), (0, 0), 1),
        ],
    )
    def test_factor(self, source: tuple[int, int], target: tuple[int, int], expected: int) -> None:
        assert reduction_factor(source, target) == expected

    @pytest.mark.parametrize(
        ("source", "target"),
        [
            ((1920, 1080), (640, 360)),
            ((3840, 2160), (640, 360)),
            ((1600, 900), (640, 360)),
            ((1000, 700), (333, 219)),
            ((641, 361), (200, 100)),
        ],
    )
    def test_reduced_image_never_falls_below_the_target(
        self, source: tuple[int, int], target: tuple[int, int]
    ) -> None:
        """The final resample must stay a downscale, so it never invents detail.

        ``Image.reduce`` divides with a ceiling, so this holds by construction —
        but the whole two-step path depends on it, so it is pinned rather than
        argued.
        """
        factor = reduction_factor(source, target)
        reduced = (
            Image.new("RGB", source).reduce(factor) if factor > 1 else Image.new("RGB", source)
        )

        assert reduced.size[0] >= target[0]
        assert reduced.size[1] >= target[1]


class TestResize:
    def test_produces_the_requested_size(self) -> None:
        resized = resize(gradient_image(1920, 1080), (640, 360))

        assert resized.shape == (360, 640, 3)
        assert resized.dtype == np.uint8

    @pytest.mark.parametrize(
        ("source", "target"),
        [
            ((800, 600), (640, 360)),
            ((640, 360), (640, 360)),
            ((320, 240), (640, 640)),
        ],
    )
    def test_matches_single_pass_pil_when_no_reduction_applies(
        self, source: tuple[int, int], target: tuple[int, int]
    ) -> None:
        """Below a 2x downscale the output must be byte-identical to plain PIL.

        This is what keeps the optimization from silently changing results for
        callers whose images were never large enough to reduce.
        """
        image = gradient_image(*source)
        expected = np.asarray(
            Image.fromarray(image).resize(target, resample=Image.Resampling.BILINEAR),
            dtype=np.uint8,
        )

        np.testing.assert_array_equal(resize(image, target), expected)

    def test_nearest_is_exempt_from_the_box_reduction(self) -> None:
        """A caller asking for NEAREST wants unblended pixels; reducing blends them."""
        image = gradient_image(1920, 1080)
        expected = np.asarray(
            Image.fromarray(image).resize((640, 360), resample=Image.Resampling.NEAREST),
            dtype=np.uint8,
        )

        actual = resize(image, (640, 360), resample=Image.Resampling.NEAREST)

        np.testing.assert_array_equal(actual, expected)

    def test_downscale_stays_close_to_a_lanczos_reference(self) -> None:
        """The two-step path is a legitimate resampling, not a cheap one.

        Deliberately **not** asserting it beats the single pass. It does on
        photographic content and loses badly when the content's period resonates
        with the reduction factor, so a superiority assertion would only pin
        whichever image the test happened to use. What must hold is that the
        result tracks a good reference far more closely than a cheap filter
        does — on content with enough detail for the difference to exist.
        """
        image = detailed_image(1920, 1080)
        source = Image.fromarray(image)
        reference = np.asarray(source.resize((640, 360), resample=Image.Resampling.LANCZOS)).astype(
            np.float64
        )
        nearest = np.asarray(source.resize((640, 360), resample=Image.Resampling.NEAREST)).astype(
            np.float64
        )
        two_step = resize(image, (640, 360)).astype(np.float64)

        two_step_error = ((two_step - reference) ** 2).mean()
        nearest_error = ((nearest - reference) ** 2).mean()

        assert two_step_error < nearest_error / 4

    def test_resonant_content_is_the_known_weak_case(self) -> None:
        """Pins the regression this optimization accepts, so it stays visible.

        Two-pixel stripes every six rows, reduced by three, land the box filter
        on the pattern's own period. The single pass wins there by a wide margin.
        Asserting it keeps the trade-off in the test suite rather than only in a
        changelog entry nobody re-reads.
        """
        image = gradient_image(1920, 1080)
        for start in range(0, 1080, 6):
            image[start : start + 2, :] = 255

        source = Image.fromarray(image)
        reference = np.asarray(source.resize((640, 360), resample=Image.Resampling.LANCZOS)).astype(
            np.float64
        )
        single_pass = np.asarray(
            source.resize((640, 360), resample=Image.Resampling.BILINEAR)
        ).astype(np.float64)
        two_step = resize(image, (640, 360)).astype(np.float64)

        assert ((two_step - reference) ** 2).mean() > ((single_pass - reference) ** 2).mean()


class TestLetterbox:
    @pytest.mark.parametrize(
        ("source", "expected_scale", "expected_pad"),
        [
            ((1920, 1080), 640 / 1920, (0, 140)),
            ((1080, 1920), 640 / 1920, (140, 0)),
            ((640, 640), 1.0, (0, 0)),
            ((320, 240), 2.0, (0, 80)),
        ],
    )
    def test_geometry(
        self,
        source: tuple[int, int],
        expected_scale: float,
        expected_pad: tuple[int, int],
    ) -> None:
        boxed, scale, pad = letterbox(gradient_image(*source), (640, 640))

        assert boxed.shape == (640, 640, 3)
        assert scale == pytest.approx(expected_scale)
        assert pad == expected_pad

    def test_pads_with_the_fill_colour(self) -> None:
        boxed, _, (pad_left, pad_top) = letterbox(gradient_image(1920, 1080), (640, 640))

        assert (boxed[:pad_top] == 114).all()
        assert (boxed[640 - pad_top :] == 114).all()
        assert pad_left == 0

    def test_custom_fill_colour(self) -> None:
        boxed, _, (_, pad_top) = letterbox(gradient_image(1920, 1080), (640, 640), fill=(1, 2, 3))

        assert (boxed[:pad_top] == (1, 2, 3)).all()

    def test_content_band_is_the_resized_image(self) -> None:
        image = gradient_image(1280, 720)

        boxed, _, (pad_left, pad_top) = letterbox(image, (640, 640))

        expected = resize(image, (640, 360))
        np.testing.assert_array_equal(
            boxed[pad_top : pad_top + 360, pad_left : pad_left + 640], expected
        )


class TestLayoutHelpers:
    def test_to_chw_transposes_and_stays_contiguous(self) -> None:
        image = gradient_image(8, 4)

        chw = to_chw(image)

        assert chw.shape == (3, 4, 8)
        assert chw.flags["C_CONTIGUOUS"]
        np.testing.assert_array_equal(chw[0], image[..., 0])

    def test_to_tensor_scales_to_unit_range(self) -> None:
        image = np.full((4, 4, 3), 255, dtype=np.uint8)
        image[0, 0] = 0

        tensor = to_tensor(image)

        assert tensor.shape == (3, 4, 4)
        assert tensor.dtype == np.float32
        assert tensor[:, 0, 0].tolist() == [0.0, 0.0, 0.0]
        assert tensor[0, 1, 1] == pytest.approx(1.0)

    def test_add_batch_dim(self) -> None:
        assert add_batch_dim(np.zeros((3, 4, 4))).shape == (1, 3, 4, 4)

    def test_normalize_applies_mean_and_std_per_channel(self) -> None:
        image = np.full((2, 2, 3), 255, dtype=np.uint8)

        normalized = normalize(image, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

        assert normalized.dtype == np.float32
        np.testing.assert_allclose(normalized, 1.0)


class TestCv2Interop:
    def test_round_trip_is_lossless(self) -> None:
        image = gradient_image(6, 4)

        np.testing.assert_array_equal(from_cv2(to_cv2(image)), image)

    def test_from_cv2_swaps_channels(self) -> None:
        bgr = np.zeros((1, 1, 3), dtype=np.uint8)
        bgr[0, 0] = (10, 20, 30)

        assert from_cv2(bgr)[0, 0].tolist() == [30, 20, 10]

    @pytest.mark.parametrize(
        "bad", [np.zeros((4, 4), dtype=np.uint8), np.zeros((4, 4, 4), np.uint8)]
    )
    def test_rejects_non_hwc_three_channel_input(self, bad: np.ndarray) -> None:
        with pytest.raises(ValueError, match="HWC 3-channel"):
            from_cv2(bad)
        with pytest.raises(ValueError, match="HWC 3-channel"):
            to_cv2(bad)
