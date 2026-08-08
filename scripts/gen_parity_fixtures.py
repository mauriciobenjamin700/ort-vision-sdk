"""Generate the shared fixtures that pin Python/Web postprocessing parity.

The two artifacts published from this repository promise the same API and the
same numbers. Nothing checked that until these fixtures existed: each suite
tested its own implementation against its own expectations, so the two could
drift apart indefinitely — and did (see ``CHANGELOG`` for the mask resampling
divergence this file's first run surfaced).

Each fixture records **inputs** and the **expected outputs**, so both suites
compare against the same committed numbers:

- ``sdk-python/tests/test_parity.py`` reads them as a regression test — any
  change in the Python maths shows up here first.
- ``sdk-js-web/test/parity.test.ts`` reads them as a parity test — the web
  implementation has to reproduce what Python produced.

Expected values are computed by the Python implementation, which makes Python
the reference. Regenerating therefore *moves the reference*, and the diff has
to be reviewed rather than accepted: a changed expectation is either a
deliberate fix or a regression, and this script cannot tell them apart.

Run from the repository root:

.. code-block:: bash

    PYTHONPATH=sdk-python/src python scripts/gen_parity_fixtures.py

Only the numeric core is covered. Preprocessing is deliberately absent:
``letterbox`` resizes through PIL on Python and through a canvas in the
browser, and those resamplers do not agree pixel for pixel. Geometry
(``scale``, ``pad``) is covered indirectly, since every detection case below
carries a letterbox mapping that both sides must invert identically.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, SupportsFloat

import numpy as np

from ort_vision_sdk.graph import model_names
from ort_vision_sdk.postprocess.classification import softmax, topk
from ort_vision_sdk.postprocess.detection import batched_nms, decode_yolo, nms
from ort_vision_sdk.postprocess.segmentation import decode_yolo_seg

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "parity" / "postprocess.json"

FLOAT_DECIMALS = 9
"""Decimals kept when serializing a float.

Enough to round-trip a ``float32`` exactly through a JSON double, while
keeping the committed file readable and diffable.
"""


def rounded(value: SupportsFloat) -> float:
    """Round a scalar to :data:`FLOAT_DECIMALS` for stable serialization."""
    return round(float(value), FLOAT_DECIMALS)


def flat(array: np.ndarray) -> list[float]:
    """Flatten an array into a list of rounded floats."""
    return [rounded(v) for v in np.asarray(array, dtype=np.float32).reshape(-1)]


def tensor(array: np.ndarray) -> dict[str, Any]:
    """Serialize an array as ``{dims, data}`` with data flattened row-major."""
    return {"dims": [int(d) for d in array.shape], "data": flat(array)}


def mask_bits(mask: np.ndarray) -> str:
    """Serialize a binary 0/255 mask as a row-major string of ``0``/``1``.

    A 64x64 mask is 4096 values; as a JSON array under ``indent=2`` that is
    4096 lines. As a single string it is one line, still diffable, and the
    decode is a character comparison on either side.

    Args:
        mask: Binary mask with values in ``{0, 255}``.

    Returns:
        One character per pixel, row-major: ``"1"`` where the mask is set.

    Raises:
        ValueError: If the mask holds a value other than 0 or 255, which would
            mean the decoder stopped producing strictly binary masks.
    """
    unique = set(np.unique(mask).tolist())
    if not unique <= {0, 255}:
        raise ValueError(f"Expected a binary 0/255 mask, found values {sorted(unique)}.")
    return "".join("1" if v else "0" for v in mask.reshape(-1))


def nms_cases() -> list[dict[str, Any]]:
    """Cases for greedy single-class NMS.

    Returns:
        A list of ``{name, boxes, scores, iouThreshold, expectedKeep}`` entries.
        ``boxes`` is flat row-major xyxy, as the web signature takes it.
    """
    cases: list[dict[str, Any]] = []
    for name, boxes, scores, iou in [
        (
            "suppresses-high-overlap",
            [[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30]],
            [0.9, 0.8, 0.7],
            0.45,
        ),
        (
            "keeps-low-overlap",
            [[0, 0, 10, 10], [8, 8, 18, 18]],
            [0.9, 0.8],
            0.45,
        ),
        (
            "tied-scores-break-by-index",
            [[0, 0, 10, 10], [1, 1, 11, 11], [2, 2, 12, 12]],
            [0.5, 0.5, 0.5],
            0.45,
        ),
        ("empty", [], [], 0.45),
        (
            "degenerate-zero-area-boxes",
            [[5, 5, 5, 5], [0, 0, 10, 10]],
            [0.9, 0.8],
            0.45,
        ),
        (
            "two-degenerate-boxes-have-zero-union",
            [[5, 5, 5, 5], [5, 5, 5, 5], [0, 0, 10, 10]],
            [0.9, 0.8, 0.7],
            0.45,
        ),
    ]:
        box_arr = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        score_arr = np.asarray(scores, dtype=np.float32)
        keep = nms(box_arr, score_arr, iou)
        cases.append(
            {
                "name": name,
                "boxes": flat(box_arr),
                "scores": flat(score_arr),
                "iouThreshold": iou,
                "expectedKeep": [int(i) for i in keep],
            }
        )
    return cases


def batched_nms_cases() -> list[dict[str, Any]]:
    """Cases for per-class NMS.

    Returns:
        A list of ``{name, boxes, scores, classIds, iouThreshold, expectedKeep}``
        entries.
    """
    cases: list[dict[str, Any]] = []
    for name, boxes, scores, idxs, iou in [
        (
            "same-box-different-classes-both-survive",
            [[0, 0, 10, 10], [0, 0, 10, 10]],
            [0.9, 0.8],
            [0, 1],
            0.45,
        ),
        (
            "same-class-overlap-suppressed",
            [[0, 0, 10, 10], [1, 1, 11, 11]],
            [0.9, 0.8],
            [3, 3],
            0.45,
        ),
        (
            "mixed-classes-sorted-across-classes",
            [[0, 0, 10, 10], [40, 40, 50, 50], [1, 1, 11, 11], [41, 41, 51, 51]],
            [0.6, 0.95, 0.55, 0.7],
            [0, 1, 0, 1],
            0.45,
        ),
        (
            "cross-class-score-ties-break-by-index",
            [[0, 0, 10, 10], [40, 40, 50, 50], [80, 80, 90, 90]],
            [0.5, 0.5, 0.5],
            [2, 0, 1],
            0.45,
        ),
        ("empty", [], [], [], 0.45),
    ]:
        box_arr = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        score_arr = np.asarray(scores, dtype=np.float32)
        idx_arr = np.asarray(idxs, dtype=np.int64)
        keep = batched_nms(box_arr, score_arr, idx_arr, iou)
        cases.append(
            {
                "name": name,
                "boxes": flat(box_arr),
                "scores": flat(score_arr),
                "classIds": [int(i) for i in idx_arr],
                "iouThreshold": iou,
                "expectedKeep": [int(i) for i in keep],
            }
        )
    return cases


def _detection_output() -> np.ndarray:
    """Per-anchor detection tensor: 3 classes, 12 anchors, hand-placed boxes."""
    out = np.zeros((1, 7, 12), dtype=np.float32)
    for anchor, (cx, cy, w, h, cls, score) in enumerate(
        [
            (32.0, 32.0, 32.0, 32.0, 0, 0.90),
            (33.0, 33.0, 32.0, 32.0, 0, 0.85),
            (10.0, 32.0, 8.0, 8.0, 1, 0.70),
            (32.0, 32.0, 32.0, 32.0, 2, 0.80),
            (50.0, 20.0, 12.0, 9.0, 2, 0.31),
        ]
    ):
        out[0, 0, anchor] = cx
        out[0, 1, anchor] = cy
        out[0, 2, anchor] = w
        out[0, 3, anchor] = h
        out[0, 4 + cls, anchor] = score
    return out


def decode_yolo_cases() -> list[dict[str, Any]]:
    """Cases for the anchor-free YOLO detect decoder.

    Covers the identity letterbox, a real letterbox mapping that has to be
    inverted, a confidence threshold that empties the result, and a
    ``maxDetections`` cap.

    Returns:
        A list of ``{name, output, dims, options, expected}`` entries.
    """
    output = _detection_output()
    cases: list[dict[str, Any]] = []
    for name, geometry, conf, max_det in [
        ("square-identity-letterbox", ((64, 64), (0, 0), 1.0), 0.25, 300),
        ("wide-letterbox-scale-half", ((128, 64), (0, 16), 0.5), 0.25, 300),
        ("threshold-above-every-score", ((64, 64), (0, 0), 1.0), 0.95, 300),
        ("max-detections-caps-result", ((64, 64), (0, 0), 1.0), 0.25, 2),
    ]:
        original_size, pad, scale = geometry
        decoded = decode_yolo(
            output,
            original_size=original_size,
            pad=pad,
            scale=scale,
            conf_threshold=conf,
            iou_threshold=0.45,
            max_detections=max_det,
        )
        cases.append(
            {
                "name": name,
                "tensor": "detectionOutput",
                "options": {
                    "originalWidth": original_size[0],
                    "originalHeight": original_size[1],
                    "padLeft": pad[0],
                    "padTop": pad[1],
                    "scale": scale,
                    "confThreshold": conf,
                    "iouThreshold": 0.45,
                    "maxDetections": max_det,
                },
                "expected": [
                    {
                        "bbox": [rounded(v) for v in bbox.xyxy],
                        "classId": int(class_id),
                        "confidence": rounded(confidence),
                    }
                    for bbox, class_id, confidence in decoded
                ],
            }
        )
    return cases


def _segmentation_outputs() -> tuple[np.ndarray, np.ndarray]:
    """Per-anchor and prototype tensors: 2 classes, 4 coefficients, 8x8 protos.

    Prototype 0 splits top/bottom and prototype 1 splits left/right, both with
    logits far from zero, so the resulting soft masks sit near 0 or 1 almost
    everywhere. That matters: a soft mask hovering at the 0.5 cutoff would make
    the binary result depend on whether the resampler ran in ``float32``
    (Python) or ``float64`` (JavaScript), and the fixture would flap.
    """
    num_classes = 2
    num_coefs = 4
    mask = 8

    per_anchor = np.zeros((1, 4 + num_classes + num_coefs, 8), dtype=np.float32)
    # Anchor 0: whole image, class 1, uses prototype 0 (top/bottom split).
    per_anchor[0, 0, 0] = 32.0
    per_anchor[0, 1, 0] = 32.0
    per_anchor[0, 2, 0] = 64.0
    per_anchor[0, 3, 0] = 64.0
    per_anchor[0, 4 + 1, 0] = 0.90
    per_anchor[0, 4 + num_classes + 0, 0] = 1.0
    # Anchor 3: smaller box, class 0, uses prototype 1 (left/right split). Its
    # box straddles the split, so the mask has a vertical boundary inside it —
    # a uniformly-white mask would not test the resampler at all.
    per_anchor[0, 0, 3] = 32.0
    per_anchor[0, 1, 3] = 24.0
    per_anchor[0, 2, 3] = 21.0
    per_anchor[0, 3, 3] = 17.0
    per_anchor[0, 4 + 0, 3] = 0.75
    per_anchor[0, 4 + num_classes + 1, 3] = 1.0

    prototypes = np.zeros((1, num_coefs, mask, mask), dtype=np.float32)
    prototypes[0, 0, : mask // 2, :] = 6.0
    prototypes[0, 0, mask // 2 :, :] = -6.0
    prototypes[0, 1, :, : mask // 2] = 6.0
    prototypes[0, 1, :, mask // 2 :] = -6.0
    return per_anchor, prototypes


def decode_yolo_seg_cases() -> list[dict[str, Any]]:
    """Cases for the YOLO instance-segmentation decoder.

    Returns:
        A list of ``{name, perAnchor, perAnchorDims, prototypes,
        prototypeDims, options, expected}`` entries. Each expected instance
        carries its binary mask flattened row-major plus its width and height.
    """
    per_anchor, prototypes = _segmentation_outputs()
    cases: list[dict[str, Any]] = []
    for name, conf, mask_threshold in [
        ("two-instances", 0.25, 0.5),
        ("only-strongest-instance", 0.8, 0.5),
        ("high-mask-threshold", 0.25, 0.9),
    ]:
        decoded = decode_yolo_seg(
            per_anchor,
            prototypes,
            num_classes=2,
            input_size=(64, 64),
            original_size=(64, 64),
            pad=(0, 0),
            scale=1.0,
            conf_threshold=conf,
            iou_threshold=0.45,
            max_detections=300,
            mask_threshold=mask_threshold,
        )
        cases.append(
            {
                "name": name,
                "perAnchorTensor": "segPerAnchor",
                "prototypeTensor": "segPrototypes",
                "options": {
                    "numClasses": 2,
                    "inputWidth": 64,
                    "inputHeight": 64,
                    "originalWidth": 64,
                    "originalHeight": 64,
                    "padLeft": 0,
                    "padTop": 0,
                    "scale": 1.0,
                    "confThreshold": conf,
                    "iouThreshold": 0.45,
                    "maxDetections": 300,
                    "maskThreshold": mask_threshold,
                },
                "expected": [
                    {
                        "bbox": [rounded(v) for v in bbox.xyxy],
                        "classId": int(class_id),
                        "confidence": rounded(confidence),
                        "maskWidth": int(mask.shape[1]),
                        "maskHeight": int(mask.shape[0]),
                        "maskBits": mask_bits(mask),
                    }
                    for bbox, class_id, confidence, mask in decoded
                ],
            }
        )
    return cases


def softmax_cases() -> list[dict[str, Any]]:
    """Cases for softmax, including magnitudes that would overflow naively."""
    cases: list[dict[str, Any]] = []
    for name, logits in [
        ("small-vector", [1.0, 3.0, 0.5, 2.0]),
        ("large-magnitudes", [1000.0, 1001.0, 999.0]),
        ("all-equal", [2.0, 2.0, 2.0, 2.0]),
        ("negative", [-5.0, -1.0, -3.0]),
        ("single-entry", [7.0]),
    ]:
        logit_arr = np.asarray(logits, dtype=np.float32)
        cases.append(
            {
                "name": name,
                "logits": flat(logit_arr),
                "expected": flat(softmax(logit_arr)),
            }
        )
    return cases


def topk_cases() -> list[dict[str, Any]]:
    """Cases for top-k, including a tie whose ordering both sides must share."""
    cases: list[dict[str, Any]] = []
    for name, probabilities, k in [
        ("top-2", [0.1, 0.6, 0.05, 0.25], 2),
        ("k-none-returns-all", [0.1, 0.6, 0.05, 0.25], None),
        ("k-above-length-is-clamped", [0.4, 0.6], 5),
        ("ties-keep-lowest-index-first", [0.25, 0.25, 0.25, 0.25], 3),
    ]:
        prob_arr = np.asarray(probabilities, dtype=np.float32)
        indices, values = topk(prob_arr, k)
        cases.append(
            {
                "name": name,
                "probabilities": flat(prob_arr),
                "k": k,
                "expectedIndices": [int(i) for i in indices],
                "expectedValues": flat(values),
            }
        )
    return cases


def model_names_cases() -> list[dict[str, Any]]:
    """Cases for parsing the ``names`` map an export bakes into its metadata."""
    cases: list[dict[str, Any]] = []
    for name, raw in [
        ("ultralytics-repr", "{0: 'cat', 1: 'dog', 2: 'bird'}"),
        ("double-quoted", '{0: "cat", 1: "dog"}'),
        ("non-contiguous-keys-rejected", "{0: 'cat', 2: 'bird'}"),
        ("not-starting-at-zero-rejected", "{1: 'cat', 2: 'bird'}"),
        ("malformed-rejected", "{0: 'cat',"),
        ("not-a-dict-rejected", "['cat', 'dog']"),
        ("empty-rejected", ""),
        ("non-string-values-rejected", "{0: 1, 1: 2}"),
    ]:
        parsed = model_names({"names": raw} if raw else {})
        cases.append(
            {
                "name": name,
                "raw": raw,
                "expected": (
                    None if parsed is None else {str(k): v for k, v in sorted(parsed.items())}
                ),
            }
        )
    return cases


def main() -> None:
    """Write the parity fixture to :data:`FIXTURE_PATH`."""
    seg_per_anchor, seg_prototypes = _segmentation_outputs()
    fixture = {
        "_generator": "scripts/gen_parity_fixtures.py",
        "_reference": "sdk-python — expectations are what the Python implementation produces",
        "tensors": {
            "detectionOutput": tensor(_detection_output()),
            "segPerAnchor": tensor(seg_per_anchor),
            "segPrototypes": tensor(seg_prototypes),
        },
        "nms": nms_cases(),
        "batchedNms": batched_nms_cases(),
        "decodeYolo": decode_yolo_cases(),
        "decodeYoloSeg": decode_yolo_seg_cases(),
        "softmax": softmax_cases(),
        "topk": topk_cases(),
        "modelNames": model_names_cases(),
    }
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
    counts = {key: len(value) for key, value in fixture.items() if isinstance(value, list)}
    print(f"wrote {FIXTURE_PATH} ({FIXTURE_PATH.stat().st_size} B)")
    print(f"cases: {counts}")


if __name__ == "__main__":
    main()
