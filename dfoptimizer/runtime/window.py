"""Application-agnostic analysis window for streaming I/O optimization.

The Window manages temporal boundaries for the DFSuite analysis pipeline.
Instead of hardcoded epoch/step trigger logic, the application calls
``window.start()`` and ``window.stop()`` at each logical iteration.
The Window emits analysis boundary events to DFTracer at a configurable
cadence (``every_n``), which is self-tuning via the optimizer.

The cadence starts at 1 (analyze every step) and doubles on each
``increase_cadence()`` call until overhead is acceptable. This
converges to the right window size within a few optimizer cycles.

Example::

    window = Window(dftracer.get_instance())

    for epoch in range(20):
        for step in range(steps_per_epoch):
            window.start()
            batch = next(loader)
            compute(batch)
            window.stop()
        window.flush()  # force boundary at epoch end
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# DFTracer tag helpers — match TagValue.value() format:
# int_args: tuple = (tag_type, value)
# C++ reads std::get<0> as tag_type, std::get<1> as value
# tag_type 0 = KEY (included in output)
_INT_TAG = lambda v: (0, v)   # (tag_type=KEY, value)


class Window:
    """Manages analysis window boundaries with self-tuning cadence.

    Parameters
    ----------
    dftracer_instance:
        The DFTracer profiler instance (from ``dftracer.get_instance()``).
        Used to emit window events and get timestamps.
    max_every_n:
        Upper bound for cadence (e.g., ``steps_per_epoch``).
        If None, no upper bound.
    """

    def __init__(
        self,
        dftracer_instance: Any,
        max_every_n: Optional[int] = None,
    ):
        self._dft = dftracer_instance
        self._every_n = 1
        self._max_every_n = max_every_n
        self._counter = 0
        self._window_index = 0
        self._active = False
        self._start_time = 0

    @property
    def every_n(self) -> int:
        return self._every_n

    @property
    def window_index(self) -> int:
        return self._window_index

    @property
    def counter(self) -> int:
        return self._counter

    def start(self) -> bool:
        """Called at every logical iteration start.

        Increments the internal counter. Emits a ``window.start`` event
        only when the counter hits the cadence boundary (``counter % every_n == 1``).

        Returns True if a window boundary was emitted, False otherwise.
        """
        self._counter += 1

        if self._every_n > 1 and self._counter % self._every_n != 1:
            return False

        # This is a boundary — emit window.start
        self._window_index += 1
        self._active = True
        if self._dft is not None:
            self._start_time = self._dft.get_time()
            self._dft.enter_event()
            self._dft.log_event(
                name="window.start",
                cat="window",
                start_time=self._start_time,
                duration=0,
                int_args={
                    "window_index": _INT_TAG(self._window_index),
                    "every_n": _INT_TAG(self._every_n),
                    "counter": _INT_TAG(self._counter),
                },
            )
            self._dft.exit_event()
        return True

    def stop(self) -> bool:
        """Called at every logical iteration end.

        Emits a ``window.stop`` event only at cadence boundaries
        (when the counter is a multiple of ``every_n``).

        Returns True if a window boundary was emitted, False otherwise.
        """
        if not self._active:
            return False

        if self._every_n > 1 and self._counter % self._every_n != 0:
            return False

        # This is a boundary — emit window.stop
        self._active = False
        if self._dft is not None:
            t = self._dft.get_time()
            self._dft.enter_event()
            self._dft.log_event(
                name="window.stop",
                cat="window",
                start_time=self._start_time,
                duration=t - self._start_time,
                int_args={
                    "window_index": _INT_TAG(self._window_index),
                    "every_n": _INT_TAG(self._every_n),
                    "counter": _INT_TAG(self._counter),
                },
            )
            self._dft.exit_event()
        return True

    def flush(self) -> bool:
        """Force a window boundary and reset counter.

        Called at application-level boundaries (epoch end, phase change).
        Ensures analysis happens even if counter hasn't reached ``every_n``.
        Resets the counter so the next phase starts fresh.

        Returns True if a window boundary was emitted, False otherwise.
        """
        emitted = False

        if self._active:
            # Active window — close it
            self._active = False
            if self._dft is not None:
                t = self._dft.get_time()
                self._dft.enter_event()
                self._dft.log_event(
                    name="window.stop",
                    cat="window",
                    start_time=self._start_time,
                    duration=t - self._start_time,
                    int_args={
                        "window_index": _INT_TAG(self._window_index),
                        "every_n": _INT_TAG(self._every_n),
                        "counter": _INT_TAG(self._counter),
                        "flush": _INT_TAG(1),
                    },
                )
                self._dft.exit_event()
            emitted = True
        elif self._counter > 0 and self._counter % self._every_n != 0:
            # Partial window — events accumulated but no boundary yet
            self._window_index += 1
            if self._dft is not None:
                t = self._dft.get_time()
                self._dft.enter_event()
                self._dft.log_event(
                    name="window.stop",
                    cat="window",
                    start_time=t,
                    duration=0,
                    int_args={
                        "window_index": _INT_TAG(self._window_index),
                        "every_n": _INT_TAG(self._every_n),
                        "counter": _INT_TAG(self._counter),
                        "flush": _INT_TAG(1),
                    },
                )
                self._dft.exit_event()
            emitted = True

        self._counter = 0
        return emitted

    def increase_cadence(self) -> None:
        """Double the cadence (called by optimizer when overhead is too high)."""
        new = self._every_n * 2
        if self._max_every_n is not None:
            new = min(new, self._max_every_n)
        if new != self._every_n:
            logger.info(
                "window.cadence_increase",
                extra={"old": self._every_n, "new": new},
            )
            self._every_n = new

    def decrease_cadence(self) -> None:
        """Halve the cadence (called by optimizer when analysis is stale)."""
        new = max(1, self._every_n // 2)
        if new != self._every_n:
            logger.info(
                "window.cadence_decrease",
                extra={"old": self._every_n, "new": new},
            )
            self._every_n = new

    def update_cadence(self, new_every_n: int) -> None:
        """Set cadence to an explicit value (called by optimizer)."""
        new_every_n = max(1, new_every_n)
        if self._max_every_n is not None:
            new_every_n = min(new_every_n, self._max_every_n)
        if new_every_n != self._every_n:
            logger.info(
                "window.cadence_update",
                extra={"old": self._every_n, "new": new_every_n},
            )
            self._every_n = new_every_n
