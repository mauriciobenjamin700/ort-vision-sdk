"""Tests for the metadata contract a fused pipeline carries inside its own file.

This is the seam between build time and run time: :mod:`ort_vision_sdk.compose`
writes it with the ``onnx`` library, and
:class:`~ort_vision_sdk.tasks.pipeline.DetectClassify` reads it back with
nothing but ``onnxruntime``. These tests exercise the codec directly, without
either — a decoding bug here would otherwise only surface as a pipeline that
runs at the wrong resolution.
"""

from __future__ import annotations

import pytest

from ort_vision_sdk.fusion import METADATA_PREFIX, FusionSpec


def _spec(**overrides: object) -> FusionSpec:
    """Build a spec with sensible defaults, overriding named fields."""
    defaults: dict[str, object] = {
        "input_size": (640, 640),
        "crop_size": (224, 224),
        "crop_source": "detector_input",
        "max_detections": 20,
        "conf_threshold": 0.25,
        "iou_threshold": 0.45,
        "apply_softmax": True,
        "detector_names": {0: "sheep"},
        "classifier_names": {0: "healthy", 1: "anaemic"},
        "sdk_version": "9.9.9",
    }
    defaults.update(overrides)
    return FusionSpec(**defaults)  # type: ignore[arg-type]


class TestRoundTrip:
    """Everything written must come back unchanged."""

    def test_preserves_every_field(self) -> None:
        original = _spec()
        assert FusionSpec.from_metadata(original.to_metadata()) == original

    def test_preserves_the_dynamic_row_mode(self) -> None:
        original = _spec(max_detections=None)
        decoded = FusionSpec.from_metadata(original.to_metadata())

        assert decoded is not None
        assert decoded.max_detections is None

    def test_preserves_the_original_crop_source(self) -> None:
        decoded = FusionSpec.from_metadata(_spec(crop_source="original").to_metadata())

        assert decoded is not None
        assert decoded.crop_source == "original"
        assert decoded.needs_source_image is True

    def test_omits_absent_class_maps(self) -> None:
        """A stage with no names must not write an entry that decodes to an empty map."""
        entries = _spec(detector_names=None, classifier_names=None).to_metadata()

        assert f"{METADATA_PREFIX}detector_names" not in entries
        decoded = FusionSpec.from_metadata(entries)
        assert decoded is not None
        assert decoded.detector_names is None
        assert decoded.classifier_names is None


class TestNamespacing:
    """The ``ovs.`` prefix keeps pipeline keys apart from the exporter's own."""

    def test_every_key_is_prefixed(self) -> None:
        assert all(key.startswith(METADATA_PREFIX) for key in _spec().to_metadata())

    def test_ignores_foreign_metadata(self) -> None:
        """An Ultralytics ``names`` entry sits alongside ours without being read as ours."""
        entries = dict(_spec().to_metadata())
        entries["names"] = "{0: 'something else'}"
        entries["task"] = "detect"

        decoded = FusionSpec.from_metadata(entries)
        assert decoded is not None
        assert decoded.detector_names == {0: "sheep"}


class TestRejection:
    """Metadata that does not describe a pipeline this version can drive."""

    def test_no_metadata_at_all(self) -> None:
        assert FusionSpec.from_metadata(None) is None
        assert FusionSpec.from_metadata({}) is None

    def test_a_plain_model(self) -> None:
        assert FusionSpec.from_metadata({"names": "{0: 'cat'}", "task": "detect"}) is None

    def test_an_unknown_pipeline_kind(self) -> None:
        entries = dict(_spec().to_metadata())
        entries[f"{METADATA_PREFIX}kind"] = "detect_segment_classify"

        assert FusionSpec.from_metadata(entries) is None

    @pytest.mark.parametrize("key", ["input_size", "crop_size"])
    def test_an_unusable_resolution(self, key: str) -> None:
        """There is no safe default for a resolution, so a malformed one is fatal."""
        entries = dict(_spec().to_metadata())
        entries[f"{METADATA_PREFIX}{key}"] = "not-a-size"

        assert FusionSpec.from_metadata(entries) is None

    @pytest.mark.parametrize("value", ["640", "640,0", "640,640,640", ""])
    def test_malformed_size_encodings(self, value: str) -> None:
        entries = dict(_spec().to_metadata())
        entries[f"{METADATA_PREFIX}input_size"] = value

        assert FusionSpec.from_metadata(entries) is None


class TestTolerance:
    """A single bad entry must not throw away an otherwise loadable pipeline."""

    def test_a_malformed_threshold_falls_back(self) -> None:
        entries = dict(_spec().to_metadata())
        entries[f"{METADATA_PREFIX}conf_threshold"] = "quite high"

        decoded = FusionSpec.from_metadata(entries)
        assert decoded is not None
        assert decoded.conf_threshold == pytest.approx(0.25)

    def test_a_malformed_row_count_falls_back_to_dynamic(self) -> None:
        entries = dict(_spec().to_metadata())
        entries[f"{METADATA_PREFIX}max_detections"] = "several"

        decoded = FusionSpec.from_metadata(entries)
        assert decoded is not None
        assert decoded.max_detections is None

    def test_an_unrecognised_crop_source_falls_back_to_the_single_input_one(self) -> None:
        """Guessing ``"original"`` would demand inputs the caller has no reason to send."""
        entries = dict(_spec().to_metadata())
        entries[f"{METADATA_PREFIX}crop_source"] = "somewhere_else"

        decoded = FusionSpec.from_metadata(entries)
        assert decoded is not None
        assert decoded.crop_source == "detector_input"
        assert decoded.needs_source_image is False

    def test_a_broken_class_map_is_dropped_rather_than_half_applied(self) -> None:
        entries = dict(_spec().to_metadata())
        entries[f"{METADATA_PREFIX}classifier_names"] = "{0: 'healthy', 7: 'anaemic'}"

        decoded = FusionSpec.from_metadata(entries)
        assert decoded is not None
        assert decoded.classifier_names is None
