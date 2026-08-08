"""Guards that the removed ``decode_yolov8*`` aliases stay removed.

They were deprecated in 0.2.0 with "will be removed in 0.3.0" and were still
shipping at 0.6.0, which is how a deprecation stops meaning anything. Now that
they are gone, this file exists so a well-meaning re-add — or a merge that
resurrects them — fails loudly instead of quietly restoring three names the
changelog says do not exist.

The replacements are the same functions under honest names: the decoder was
never v8-specific, it covers every anchor-free YOLO head from v8 to v12.
"""

from __future__ import annotations

import pytest

import ort_vision_sdk.postprocess as postprocess
from ort_vision_sdk.postprocess import detection, segmentation

REMOVED: dict[str, str] = {
    "decode_yolov8": "decode_yolo",
    "decode_yolov8_anchors": "decode_yolo_anchors",
    "decode_yolov8_seg": "decode_yolo_seg",
}
"""Removed alias → the name that replaced it."""


class TestRemovedAliases:
    @pytest.mark.parametrize("name", sorted(REMOVED))
    def test_is_not_importable_from_the_package(self, name: str) -> None:
        assert not hasattr(postprocess, name)
        assert name not in postprocess.__all__

    @pytest.mark.parametrize("name", sorted(REMOVED))
    def test_is_not_importable_from_its_module(self, name: str) -> None:
        assert not hasattr(detection, name)
        assert not hasattr(segmentation, name)

    @pytest.mark.parametrize(("removed", "replacement"), sorted(REMOVED.items()))
    def test_its_replacement_is_exported(self, removed: str, replacement: str) -> None:
        assert hasattr(postprocess, replacement), (
            f"{removed} was removed but {replacement} is gone too"
        )
        assert replacement in postprocess.__all__
