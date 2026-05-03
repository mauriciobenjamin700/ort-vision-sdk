"""Tests for ``ort_vision_sdk.labels.resolve_labels`` and presets."""

from __future__ import annotations

from pathlib import Path

import pytest

from ort_vision_sdk.core.exceptions import LabelMapError
from ort_vision_sdk.labels import COCO_CLASSES, resolve_labels


class TestPresets:
    def test_coco_has_80_classes(self) -> None:
        assert len(COCO_CLASSES) == 80
        # Spot-check a few canonical positions (Ultralytics order).
        assert COCO_CLASSES[0] == "person"
        assert COCO_CLASSES[16] == "dog"
        assert COCO_CLASSES[79] == "toothbrush"

    def test_resolve_coco_returns_preset(self) -> None:
        assert resolve_labels("coco") == COCO_CLASSES

    def test_unknown_preset_raises(self) -> None:
        with pytest.raises(LabelMapError, match="Unknown labels preset"):
            resolve_labels("imagenet")  # not a registered preset


class TestList:
    def test_explicit_list(self) -> None:
        labels = ["a", "b", "c"]
        assert resolve_labels(labels) == ("a", "b", "c")

    def test_tuple_input(self) -> None:
        labels = ("a", "b", "c")
        assert resolve_labels(labels) == ("a", "b", "c")

    def test_validates_length_against_num_classes(self) -> None:
        with pytest.raises(LabelMapError, match="Resolved 3 labels but the model has 5"):
            resolve_labels(["a", "b", "c"], num_classes=5)


class TestDict:
    def test_full_dict(self) -> None:
        labels = {0: "a", 1: "b", 2: "c"}
        assert resolve_labels(labels) == ("a", "b", "c")

    def test_sparse_dict_fills_gaps(self) -> None:
        labels = {0: "first", 2: "third"}
        result = resolve_labels(labels)
        assert result == ("first", "class_1", "third")


class TestNone:
    def test_none_with_num_classes(self) -> None:
        assert resolve_labels(None, num_classes=3) == ("class_0", "class_1", "class_2")

    def test_none_without_num_classes_raises(self) -> None:
        with pytest.raises(LabelMapError, match="Cannot auto-generate labels"):
            resolve_labels(None)


class TestFile:
    def test_load_from_path(self, tmp_path: Path) -> None:
        labels_file = tmp_path / "labels.txt"
        labels_file.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
        assert resolve_labels(labels_file) == ("alpha", "beta", "gamma")

    def test_load_from_string_path(self, tmp_path: Path) -> None:
        labels_file = tmp_path / "labels.txt"
        labels_file.write_text("one\ntwo\n", encoding="utf-8")
        assert resolve_labels(str(labels_file)) == ("one", "two")

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        labels_file = tmp_path / "empty.txt"
        labels_file.write_text("\n\n  \n", encoding="utf-8")
        with pytest.raises(LabelMapError, match="empty"):
            resolve_labels(labels_file)

    def test_blank_lines_stripped(self, tmp_path: Path) -> None:
        labels_file = tmp_path / "labels.txt"
        labels_file.write_text("a\n\n  b  \n\nc\n", encoding="utf-8")
        assert resolve_labels(labels_file) == ("a", "b", "c")
