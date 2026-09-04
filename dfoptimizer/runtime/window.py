"""Application-agnostic analysis window for streaming I/O optimization.

The Window manages temporal boundaries for the LiveFlow analysis pipeline.
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
        epochs_total: Optional[int] = None,
        steps_per_epoch: Optional[int] = None,
    ):
        self._dft = dftracer_instance
        self._every_n = 1
        self._max_every_n = max_every_n
        self._counter = 0
        self._window_index = 0
        self._active = False
        self._start_time = 0
        # HORIZON CONTRACT (liveflow_shm/HORIZON-CONTRACT.md). The application supplies the two
        # facts only it knows -- how long the run is, and how long one epoch is -- and the window
        # carries them onto every boundary event so the analyzer can publish epochs_remaining and
        # windows_remaining. Without them a cost-aware planner cannot price an actuator over the
        # run that is actually left, and the correct behaviour is then to decline: a 465 s stage
        # that pays back over three epochs and one that pays back over fifty are the same request
        # with different answers, and the difference is not visible from inside a window.
        self._epochs_total = int(epochs_total) if epochs_total else None
        self._steps_per_epoch = int(steps_per_epoch) if steps_per_epoch else None
        self._epoch_index = 0

    def set_horizon(self, epochs_total: Optional[int], steps_per_epoch: Optional[int]) -> None:
        """Supply the two application facts after construction.

        DLIO knows both before it builds the Window and passes them to __init__. halo3d does not:
        its steps-per-epoch is the length of the reader's shard list, which is only known further
        down, after the loader is built. Without this setter halo3d could not satisfy the contract
        at all, and it did not: every window of the 2-node loop cell on 2026-09-03 logged
        `window.horizon.missing`. Same all-or-none discipline as _horizon_args: a falsy value
        leaves the field unset rather than storing a zero that would read as instrumented.
        """
        if epochs_total:
            self._epochs_total = int(epochs_total)
        if steps_per_epoch:
            self._steps_per_epoch = int(steps_per_epoch)

    def set_epoch(self, epoch_index: int) -> None:
        """Tell the window which epoch it is in. One based, per the contract."""
        self._epoch_index = int(epoch_index)

    @property
    def windows_per_epoch(self) -> Optional[int]:
        """Analysis windows expected in one full epoch AT THE CURRENT CADENCE.

        Derived rather than stored, because `every_n` is self-tuning and a value captured at
        construction would silently describe a cadence the run is no longer using.

        CAVEAT worth carrying: if the cadence changes mid-run, windows already emitted were counted
        at the old cadence, so `epochs_total * windows_per_epoch` is no longer a count of the
        windows this run will actually produce. The epoch-based horizon does not have that problem,
        which is why it is the one to trust when the two disagree.
        """
        if not self._steps_per_epoch or self._every_n <= 0:
            return None
        return max(1, -(-self._steps_per_epoch // self._every_n))   # ceil

    def _horizon_args(self) -> Dict[str, Any]:
        """The four contract fields, or NOTHING.

        All four or none, deliberately. The analyzer publishes a derived remaining value only when
        every source field is present, so emitting a partial set would produce an event that looks
        instrumented and yields no metric -- the failure mode this whole contract exists to avoid.
        """
        wpe = self.windows_per_epoch
        if not (self._epoch_index and self._epochs_total and wpe):
            return {}
        return {
            "epoch_index": _INT_TAG(self._epoch_index),
            "epochs_total": _INT_TAG(self._epochs_total),
            "windows_per_epoch": _INT_TAG(wpe),
        }

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
                    **self._horizon_args(),
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
                    **self._horizon_args(),
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
                        **self._horizon_args(),
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
                        **self._horizon_args(),
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
