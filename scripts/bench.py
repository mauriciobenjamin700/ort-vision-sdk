"""Microbenchmarks for the Python SDK's hot pure functions.

Purpose is to make the next round of optimization measurable: preprocessing and
decode/NMS are where a real workload spends its non-inference time, and none of
it had a number attached before this script existed. Every case below runs on
shapes a 640x640 YOLO export actually produces, with deterministic inputs, so
two runs on the same machine are comparable.

Usage:

.. code-block:: bash

    PYTHONPATH=sdk-python/src python scripts/bench.py
    PYTHONPATH=sdk-python/src python scripts/bench.py --json bench/baseline-python.json
    PYTHONPATH=sdk-python/src python scripts/bench.py --compare bench/baseline-python.json

The committed baseline is a **local reference**, not a CI gate. Shared runners
vary by more than the regressions worth catching, so wiring ``--compare`` into
CI would produce flakes rather than signal; run it on the machine that produced
the baseline, before and after a change.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from ort_vision_sdk.postprocess.classification import softmax, topk
from ort_vision_sdk.postprocess.detection import batched_nms, decode_yolo, nms
from ort_vision_sdk.postprocess.segmentation import decode_yolo_seg
from ort_vision_sdk.preprocess.image import add_batch_dim, letterbox, to_tensor

INPUT_SIZE = (640, 640)
NUM_CLASSES = 80
NUM_ANCHORS = 8400
NUM_MASK_COEFS = 32
MASK_SIZE = 160

DEFAULT_TOLERANCE = 0.25
"""Fractional slowdown accepted by ``--compare`` before a case counts as a regression.

Loose on purpose: a microbenchmark on a desktop under normal load swings by
more than 10% between runs, so a tighter bound would report noise as a finding.
"""

NOISE_FLOOR_MS = 0.1
"""Median below which a case is reported but never counted as a regression.

``softmax_1000_classes`` runs in roughly 5 microseconds. At that scale a single
context switch doubles the number, and this harness's first comparison duly
reported a "+253% REGRESSION" on a function nobody had touched. A case has to be
slow enough to measure before a percentage about it means anything.
"""


def rng() -> np.random.Generator:
    """A seeded generator, so every case sees byte-identical inputs across runs."""
    return np.random.default_rng(20260806)


def make_image(width: int, height: int) -> np.ndarray:
    """Build an HWC uint8 RGB image of the given size."""
    return rng().integers(0, 256, (height, width, 3), dtype=np.uint8)


def make_detection_output(num_candidates: int) -> np.ndarray:
    """Build a per-anchor detect tensor with a chosen number of above-threshold anchors.

    Args:
        num_candidates: How many anchors carry a class score above 0.25. The
            rest are filled with low scores, which is what a real frame looks
            like — a few hundred candidates out of 8400.

    Returns:
        A ``(1, 4 + NUM_CLASSES, NUM_ANCHORS)`` ``float32`` array.
    """
    generator = rng()
    out = np.zeros((1, 4 + NUM_CLASSES, NUM_ANCHORS), dtype=np.float32)
    out[0, 4:, :] = generator.uniform(0.0, 0.2, (NUM_CLASSES, NUM_ANCHORS))
    out[0, 0, :] = generator.uniform(0, 640, NUM_ANCHORS)
    out[0, 1, :] = generator.uniform(0, 640, NUM_ANCHORS)
    out[0, 2, :] = generator.uniform(10, 120, NUM_ANCHORS)
    out[0, 3, :] = generator.uniform(10, 120, NUM_ANCHORS)
    hot = generator.choice(NUM_ANCHORS, size=num_candidates, replace=False)
    for i, anchor in enumerate(hot):
        out[0, 4 + (i % NUM_CLASSES), anchor] = 0.9
    return out


def make_segmentation_outputs(num_instances: int) -> tuple[np.ndarray, np.ndarray]:
    """Build per-anchor and prototype tensors for a seg head.

    Args:
        num_instances: How many anchors carry an above-threshold score.

    Returns:
        ``(per_anchor, prototypes)`` with the standard YOLO-seg shapes.
    """
    generator = rng()
    channels = 4 + NUM_CLASSES + NUM_MASK_COEFS
    out = np.zeros((1, channels, NUM_ANCHORS), dtype=np.float32)
    out[0, 4 : 4 + NUM_CLASSES, :] = generator.uniform(0.0, 0.2, (NUM_CLASSES, NUM_ANCHORS))
    for i in range(num_instances):
        anchor = i * (NUM_ANCHORS // num_instances)
        out[0, 0, anchor] = generator.uniform(100, 540)
        out[0, 1, anchor] = generator.uniform(100, 540)
        out[0, 2, anchor] = 80.0
        out[0, 3, anchor] = 80.0
        out[0, 4 + (i % NUM_CLASSES), anchor] = 0.9
        out[0, 4 + NUM_CLASSES :, anchor] = generator.normal(0.0, 0.5, NUM_MASK_COEFS)
    prototypes = generator.normal(0.0, 1.0, (1, NUM_MASK_COEFS, MASK_SIZE, MASK_SIZE)).astype(
        np.float32
    )
    return out, prototypes


def make_boxes(count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build overlapping boxes, scores and class ids for the NMS cases."""
    generator = rng()
    centers = generator.uniform(50, 590, (count, 2)).astype(np.float32)
    sides = generator.uniform(20, 90, (count, 1)).astype(np.float32)
    boxes = np.hstack([centers - sides / 2, centers + sides / 2]).astype(np.float32)
    scores = generator.uniform(0.25, 1.0, count).astype(np.float32)
    class_ids = generator.integers(0, 20, count, dtype=np.int64)
    return boxes, scores, class_ids


def build_cases() -> dict[str, Callable[[], object]]:
    """Assemble every benchmark case as a zero-argument callable.

    Inputs are built once, outside the timed callable, so the numbers measure
    the function under test rather than the fixture construction.

    Returns:
        Case name → callable to time.
    """
    hd_image = make_image(1920, 1080)
    uhd_image = make_image(3840, 2160)
    hd_ready_image = make_image(1280, 720)
    small_image = make_image(800, 600)
    square_image = make_image(640, 640)
    letterboxed, _, _ = letterbox(hd_image, INPUT_SIZE)

    sparse_output = make_detection_output(50)
    dense_output = make_detection_output(2000)
    seg_output, seg_prototypes = make_segmentation_outputs(30)

    boxes_300, scores_300, classes_300 = make_boxes(300)
    boxes_2000, scores_2000, classes_2000 = make_boxes(2000)

    logits = rng().normal(0.0, 4.0, 1000).astype(np.float32)
    probabilities = softmax(logits)

    decode_kwargs: dict[str, Any] = {
        "original_size": (1920, 1080),
        "pad": (0, 140),
        "scale": 640 / 1920,
        "conf_threshold": 0.25,
        "iou_threshold": 0.45,
        "max_detections": 300,
    }

    return {
        "letterbox_1080p_to_640": lambda: letterbox(hd_image, INPUT_SIZE),
        "letterbox_4k_to_640": lambda: letterbox(uhd_image, INPUT_SIZE),
        "letterbox_720p_to_640": lambda: letterbox(hd_ready_image, INPUT_SIZE),
        "letterbox_800x600_to_640_no_reduction": lambda: letterbox(small_image, INPUT_SIZE),
        "to_tensor_640": lambda: to_tensor(letterboxed),
        "preprocess_detector_1080p": lambda: np.ascontiguousarray(
            add_batch_dim(to_tensor(letterbox(hd_image, INPUT_SIZE)[0]))
        ),
        "preprocess_detector_640": lambda: np.ascontiguousarray(
            add_batch_dim(to_tensor(letterbox(square_image, INPUT_SIZE)[0]))
        ),
        "decode_yolo_50_candidates": lambda: decode_yolo(sparse_output, **decode_kwargs),
        "decode_yolo_2000_candidates": lambda: decode_yolo(dense_output, **decode_kwargs),
        "nms_300_boxes": lambda: nms(boxes_300, scores_300, 0.45),
        "nms_2000_boxes": lambda: nms(boxes_2000, scores_2000, 0.45),
        "batched_nms_300_boxes_20_classes": lambda: batched_nms(
            boxes_300, scores_300, classes_300, 0.45
        ),
        "batched_nms_2000_boxes_20_classes": lambda: batched_nms(
            boxes_2000, scores_2000, classes_2000, 0.45
        ),
        "decode_yolo_seg_30_instances": lambda: decode_yolo_seg(
            seg_output,
            seg_prototypes,
            num_classes=NUM_CLASSES,
            input_size=INPUT_SIZE,
            **decode_kwargs,
        ),
        "softmax_1000_classes": lambda: softmax(logits),
        "topk_1000_classes": lambda: topk(probabilities, 5),
    }


def measure(case: Callable[[], object], *, reps: int, warmup: int) -> dict[str, float]:
    """Time a case and summarize the samples.

    Args:
        case: Zero-argument callable to time.
        reps: Number of timed repetitions.
        warmup: Number of untimed repetitions run first, to let NumPy allocate
            its scratch buffers and warm the caches.

    Returns:
        ``{"median_ms", "min_ms", "max_ms"}`` in milliseconds. The median is the
        number to compare; the min shows the floor and the max exposes how noisy
        the machine was.
    """
    for _ in range(warmup):
        case()
    samples: list[float] = []
    for _ in range(reps):
        start = time.perf_counter()
        case()
        samples.append((time.perf_counter() - start) * 1000.0)
    return {
        "median_ms": round(float(np.median(samples)), 4),
        "min_ms": round(float(np.min(samples)), 4),
        "max_ms": round(float(np.max(samples)), 4),
    }


def run(reps: int, warmup: int) -> dict[str, Any]:
    """Run every case and return the full result document."""
    cases = build_cases()
    results = {name: measure(case, reps=reps, warmup=warmup) for name, case in cases.items()}
    return {
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "settings": {"reps": reps, "warmup": warmup},
        "results": results,
    }


def report(document: dict[str, Any]) -> None:
    """Print a result document as an aligned table."""
    results: dict[str, dict[str, float]] = document["results"]
    width = max(len(name) for name in results)
    print(f"{'case':<{width}}  {'median':>10}  {'min':>10}  {'max':>10}")
    for name, values in results.items():
        print(
            f"{name:<{width}}  {values['median_ms']:>9.3f}ms"
            f"  {values['min_ms']:>9.3f}ms  {values['max_ms']:>9.3f}ms"
        )


def compare(document: dict[str, Any], baseline_path: Path, tolerance: float) -> int:
    """Compare a run against a baseline and report regressions.

    Args:
        document: The fresh result document.
        baseline_path: Path to a previously saved document.
        tolerance: Fractional slowdown accepted before a case is a regression.

    Returns:
        Process exit status — ``1`` when any case regressed beyond ``tolerance``
        or a baseline case is missing from the run, ``0`` otherwise.
    """
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    old: dict[str, dict[str, float]] = baseline["results"]
    new: dict[str, dict[str, float]] = document["results"]

    if baseline["environment"] != document["environment"]:
        print("! baseline was recorded on a different environment — numbers are not comparable")
        print(f"  baseline: {baseline['environment']}")
        print(f"  current:  {document['environment']}")

    width = max(len(name) for name in set(old) | set(new))
    regressions: list[str] = []
    missing = sorted(set(old) - set(new))

    print(f"{'case':<{width}}  {'baseline':>10}  {'current':>10}  {'delta':>9}")
    for name in new:
        if name not in old:
            print(f"{name:<{width}}  {'—':>10}  {new[name]['median_ms']:>9.3f}ms  {'new':>9}")
            continue
        before = old[name]["median_ms"]
        after = new[name]["median_ms"]
        delta = (after - before) / before if before > 0 else 0.0
        too_fast_to_judge = max(before, after) < NOISE_FLOOR_MS
        regressed = delta > tolerance and not too_fast_to_judge
        flag = "  REGRESSION" if regressed else ("  below noise floor" if too_fast_to_judge else "")
        print(f"{name:<{width}}  {before:>9.3f}ms  {after:>9.3f}ms  {delta * 100:>+8.1f}%{flag}")
        if regressed:
            regressions.append(name)

    for name in missing:
        print(f"{name:<{width}}  {old[name]['median_ms']:>9.3f}ms  {'—':>10}  {'MISSING':>9}")

    if regressions or missing:
        print(
            f"\nfailed: {len(regressions)} regression(s) beyond {tolerance:.0%}"
            f", {len(missing)} case(s) missing"
        )
        return 1
    print(f"\nok: no case regressed beyond {tolerance:.0%}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the benchmarks and report.

    Args:
        argv: Argument list; ``None`` reads ``sys.argv``.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=25, help="timed repetitions per case")
    parser.add_argument("--warmup", type=int, default=3, help="untimed repetitions per case")
    parser.add_argument("--json", type=Path, default=None, help="write results to this path")
    parser.add_argument(
        "--compare", type=Path, default=None, help="compare against a saved baseline"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help="fractional slowdown accepted by --compare",
    )
    args = parser.parse_args(argv)

    document = run(args.reps, args.warmup)

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")

    if args.compare is not None:
        return compare(document, args.compare, args.tolerance)

    report(document)
    return 0


if __name__ == "__main__":
    sys.exit(main())
