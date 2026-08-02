"""Tests for the per-stage inference timer."""

from __future__ import annotations

import numpy as np
import pytest

from ort_vision_sdk.core.timing import STAGES, SpeedTimer
from ort_vision_sdk.results import ClassificationResults, DetectionResults, Boxes, Probs
from ort_vision_sdk.types import ClassificationResult


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch):
    """Drive ``perf_counter`` from a script so durations are exact.

    Returns a factory that installs a fake clock reading the given sequence of
    seconds, one value per call.
    """

    def install(sequence: list[float]) -> None:
        values = iter(sequence)
        last = sequence[-1]

        def fake() -> float:
            nonlocal last
            try:
                last = next(values)
            except StopIteration:
                pass
            return last

        monkeypatch.setattr("ort_vision_sdk.core.timing.time.perf_counter", fake)

    return install


def test_stage_attributes_each_interval_to_the_stage_that_closed_it(clock) -> None:
    """Every boundary charges the elapsed time to the stage that just ended."""
    clock([0.0, 0.010, 0.025, 0.125, 0.150])

    timer = SpeedTimer()
    for stage in STAGES:
        timer.stage(stage)

    assert timer.speed() == pytest.approx(
        {"load": 10.0, "preprocess": 15.0, "inference": 100.0, "postprocess": 25.0}
    )


def test_stage_accumulates_when_closed_more_than_once(clock) -> None:
    """A stage entered twice sums both intervals instead of overwriting."""
    clock([0.0, 0.005, 0.008, 0.020])

    timer = SpeedTimer()
    timer.stage("inference")
    timer.stage("postprocess")
    timer.stage("inference")

    assert timer.speed()["inference"] == pytest.approx(17.0)
    assert timer.speed()["postprocess"] == pytest.approx(3.0)


def test_speed_starts_at_zero_for_every_stage(clock) -> None:
    """A fresh timer reports all four stages, all zeroed."""
    clock([0.0])

    assert SpeedTimer().speed() == {stage: 0.0 for stage in STAGES}


def test_speed_returns_a_copy(clock) -> None:
    """A captured snapshot is not mutated by later stages."""
    clock([0.0, 0.004, 0.009])

    timer = SpeedTimer()
    timer.stage("load")
    snapshot = timer.speed()
    timer.stage("load")

    assert snapshot["load"] == pytest.approx(4.0)
    assert timer.speed()["load"] == pytest.approx(9.0)


def test_results_default_to_an_empty_speed_mapping() -> None:
    """Envelopes built outside ``predict()`` carry no timings."""
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    envelope = DetectionResults(
        boxes=Boxes(
            xyxy=np.zeros((0, 4), dtype=np.float64),
            cls=np.zeros((0,), dtype=np.int64),
            conf=np.zeros((0,), dtype=np.float64),
            orig_shape=(1, 1),
        ),
        detections=(),
        names={},
        orig_img=image,
        orig_shape=(1, 1),
    )

    assert envelope.speed == {}


def test_results_carry_the_speed_they_are_given() -> None:
    """The envelope stores the mapping the task hands it, verbatim."""
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    speed = {"load": 1.0, "preprocess": 2.0, "inference": 3.0, "postprocess": 4.0}
    envelope = ClassificationResults(
        probs=Probs(data=np.array([1.0])),
        result=ClassificationResult(
            class_id=0,
            class_name="a",
            confidence=1.0,
            image=image,
            probabilities=(),
        ),
        names={0: "a"},
        orig_img=image,
        orig_shape=(1, 1),
        speed=speed,
    )

    assert envelope.speed == speed
