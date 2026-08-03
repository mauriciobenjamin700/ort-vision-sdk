"""Per-stage timing for a single ``predict()`` call.

Populates the ``speed`` field every ``Results`` envelope carries, mirroring
Ultralytics' ``results[0].speed``. All values are milliseconds measured with
:func:`time.perf_counter`.
"""

from __future__ import annotations

import time
from typing import Literal

Stage = Literal["load", "preprocess", "inference", "postprocess"]

STAGES: tuple[Stage, ...] = ("load", "preprocess", "inference", "postprocess")
"""Stage names, in the order a ``predict()`` call goes through them.

``preprocess``, ``inference`` and ``postprocess`` are the three keys
Ultralytics reports, measured over the same boundaries. ``load`` is specific
to this SDK: ``predict()`` accepts a path, bytes or array and decodes it
internally, so the read/decode cost would otherwise be invisible — and on a
cold page cache it dominates everything else.
"""


class SpeedTimer:
    """Accumulate stage durations while a ``predict()`` call runs.

    Each :meth:`stage` call closes the previous stage: the elapsed time since
    the last boundary is attributed to the name given. This keeps the call
    sites free of paired start/stop bookkeeping and guarantees the four stages
    tile the whole call without gaps.
    """

    def __init__(self) -> None:
        """Start the timer at the current monotonic clock reading."""
        self._last: float = time.perf_counter()
        self._speed: dict[str, float] = {stage: 0.0 for stage in STAGES}

    def stage(self, stage: Stage) -> None:
        """Attribute the time elapsed since the previous boundary to ``stage``.

        Args:
            stage: Which stage just finished.
        """
        now = time.perf_counter()
        self._speed[stage] += (now - self._last) * 1000.0
        self._last = now

    def speed(self) -> dict[str, float]:
        """Return the accumulated durations.

        Returns:
            A copy of the ``speed`` mapping to hand to the ``Results``
            envelope, in milliseconds.
        """
        return dict(self._speed)
