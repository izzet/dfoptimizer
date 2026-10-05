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
import math
import os
import statistics
import time
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
        self._liveflow_v1 = os.environ.get("LIVEFLOW_PRESET") == "liveflow-v1"
        self._every_n = 20 if self._liveflow_v1 else 1
        self._steady_step_times = []
        self._timed_steps = 0
        self._step_started = None
        self._sized = False
        self._max_every_n = max_every_n
        self._counter = 0
        self._completed_steps = 0
        self._configured_total_steps = None
        self._physical_current = {}
        self._physical_window = {}
        # Wall seconds of each completed step of the OPEN window (liveflow-v1 only). The tail of
        # the per-step distribution is what a mean hides: StorMer's mean step is dominated by a few
        # input-stall steps, so a worker change can halve the steps over 1 s and leave the window
        # mean unchanged (stormer-v1-smoke attempt2).
        self._window_step_walls = []
        # Stall an apply caused, as training saw it: the wall of the first step after the apply
        # against the median step of the window before (see begin_apply_probe).
        self._apply_probe = None
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

    def set_physical_config(self, vector, applied_step, apply_cost_s=None):
        """All-rank reconciled state, snapshotted when the measured window opens.

        `vector` maps each registered knob's short name to its applied integer value (halo3d:
        read_threads/decode_threads; StorMer: num_workers/prefetch_factor/bucket_cap_mb); each
        becomes `physical_<name>`, next to the cumulative step it took effect at.
        """
        current = {"physical_" + str(name): int(value) for name, value in vector.items()}
        current["physical_applied_step"] = int(applied_step)
        if apply_cost_s is not None:
            # Wall time (microseconds, worst rank) the apply that produced this vector held
            # training. It rides with the vector so every node's planner prices from the SAME
            # measurement instead of only the node whose plan happened to be applied.
            current["physical_apply_cost_us"] = int(round(float(apply_cost_s) * 1e6))
        self._physical_current = current

    def begin_apply_probe(self, apply_latency_s):
        """Call right after an apply that changed the loader: measure what it cost the NEXT step.

        The apply latency alone understates the stall (a rebuilt or grown pool made the following
        step 5-9 s slower than the apply itself, stormer-eng small job), so the cost is the wall of
        the first step after the apply less the median step wall of the window it follows.
        """
        walls = self._window_step_walls
        self._apply_probe = {"latency": float(apply_latency_s),
                             "baseline": statistics.median(walls) if walls else 0.0, "wall": None}

    def apply_probe_result(self):
        """Seconds the last probed apply held training (rank-local), or None until measured."""
        probe = self._apply_probe
        if probe is None or probe["wall"] is None:
            return None
        self._apply_probe = None
        return max(probe["wall"] - probe["baseline"], probe["latency"], 0.0)

    @staticmethod
    def _p90(values):
        """Nearest-rank 90th percentile of a non-empty list."""
        ordered = sorted(values)
        return ordered[max(math.ceil(0.9 * len(ordered)), 1) - 1]

    def _tail_args(self):
        """The window's tail metric, or nothing: p90 of the per-step wall in microseconds."""
        if not (self._liveflow_v1 and self._window_step_walls):
            return {}
        return {"step_wall_p90_us": _INT_TAG(int(round(self._p90(self._window_step_walls) * 1e6)))}

    def set_total_steps(self, total_steps: int) -> None:
        """Exact configured step limit, including a partial final epoch."""
        self._configured_total_steps = int(total_steps)

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

    def _epoch_args(self) -> Dict[str, Any]:
        """`epoch` on every window event, whenever the epoch is known.

        The window events carried `epoch_index` and no `epoch`. The analyzer's high-level views are
        keyed on `args.epoch`, so an event without it is dropped from the epoch-grouped frame, and
        the DLIO rule set reads its objective (`window_wall_time_proc_max`) out of exactly that
        frame. See ANALYZER-FASTPATH.md section 4. It is one tag, it costs nothing, and its absence
        is the kind of thing that shows up as a metric that is silently NA rather than as an error.

        Deliberately NOT part of the all-or-none horizon set below. The horizon trio is all-or-none
        because a partial set yields an event that looks instrumented and produces no metric; the
        grouping key has no such property and is worth emitting whenever it is known, including in
        a run with no horizon at all.
        """
        if not self._epoch_index:
            return {}
        return {"epoch": _INT_TAG(self._epoch_index)}

    def _horizon_args(self) -> Dict[str, Any]:
        """The four contract fields, or NOTHING.

        All four or none, deliberately. The analyzer publishes a derived remaining value only when
        every source field is present, so emitting a partial set would produce an event that looks
        instrumented and yields no metric -- the failure mode this whole contract exists to avoid.
        """
        wpe = self.windows_per_epoch
        if not (self._epoch_index and self._epochs_total and wpe):
            return {}
        result = {
            "epoch_index": _INT_TAG(self._epoch_index),
            "epochs_total": _INT_TAG(self._epochs_total),
            "windows_per_epoch": _INT_TAG(wpe),
        }
        if self._liveflow_v1:
            total = self._configured_total_steps or self._epochs_total * self._steps_per_epoch
            result.update(configured_total_steps=_INT_TAG(total),
                          completed_steps=_INT_TAG(self._completed_steps),
                          configured_window_steps=_INT_TAG(self._every_n),
                          planner_history_ready=_INT_TAG(int(self._sized)))
        result.update({k: _INT_TAG(v) for k, v in self._physical_window.items()})
        return result

    @property
    def every_n(self) -> int:
        return self._every_n

    @property
    def window_index(self) -> int:
        return self._window_index

    @property
    def counter(self) -> int:
        return self._counter

    def now(self) -> Optional[int]:
        """The tracer's clock, for a caller that must open a window at a moment it has passed.

        The window's wall has to be on the SAME clock as `start()` would have read, so a caller
        cannot substitute `time.time()` here even where the two happen to agree today.
        """
        return None if self._dft is None else self._dft.get_time()

    def start(self, at_time: Optional[int] = None) -> bool:
        """Called at every logical iteration start.

        Increments the internal counter. Emits a ``window.start`` event
        only when the counter hits the cadence boundary (``counter % every_n == 1``).

        `at_time`, when given, is when the window LOGICALLY began, which is not always when this
        call happens. A loop shaped `for batch in loader:` has already blocked on the fetch by the
        time the body runs, so a window opened here excludes the fetch that opened it while the
        stall accumulator drained at its close includes it. That is not a rounding difference: on
        `stgi_capacity_gate1` the run's first window carried 11.96 s of stall against a wall of
        36.42 s that did not contain it, and the four domains summed to 131 percent of the window
        as a result (STATUS-ATTRIBUTION-FIX-2026-09-08.md section 3, defect C2). Callers pass the
        instant the previous step ended, which is where the fetch began.

        Returns True if a window boundary was emitted, False otherwise.
        """
        if self._liveflow_v1:
            self._step_started = time.perf_counter()
            if at_time is not None and self._dft is not None:
                self._step_started -= max(0, self._dft.get_time() - at_time) / 1_000_000.0
        self._counter += 1

        if self._every_n > 1 and self._counter % self._every_n != 1:
            return False

        self._physical_window = dict(self._physical_current)
        self._window_step_walls = []
        # This is a boundary — emit window.start
        self._window_index += 1
        self._active = True
        if self._dft is not None:
            self._start_time = at_time if at_time is not None else self._dft.get_time()
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
                    **self._epoch_args(),
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
        self._completed_steps += 1
        if self._liveflow_v1 and self._step_started is not None:
            self._timed_steps += 1
            if 5 < self._timed_steps <= 25:
                self._steady_step_times.append(time.perf_counter() - self._step_started)
            _wall = time.perf_counter() - self._step_started
            if self._active:
                self._window_step_walls.append(_wall)
            if self._apply_probe is not None and self._apply_probe["wall"] is None:
                self._apply_probe["wall"] = _wall
            self._step_started = None
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
                    **self._epoch_args(),
                    **self._horizon_args(),
                    **self._tail_args(),
                },
            )
            self._dft.exit_event()
        if self._liveflow_v1 and not self._sized and len(self._steady_step_times) == 20:
            seconds = self._agreed_step_time(statistics.median(self._steady_step_times))
            self._every_n = max(20, math.ceil(30.0 / seconds))
            self._counter = 0  # change cadence only after closing a whole window
            self._sized = True
            logger.info("window.liveflow_v1_sized", extra={"target_seconds": 30.0,
                        "steady_step_seconds": seconds, "every_n": self._every_n})
        return True

    @staticmethod
    def _agreed_step_time(seconds):
        """One MAX reduction over training ranks; all ranks adopt the same K."""
        import sys
        torch = sys.modules.get('torch')
        if torch is not None and torch.distributed.is_available() and torch.distributed.is_initialized():
            dist = torch.distributed
            device = torch.device('cuda', torch.cuda.current_device()) if dist.get_backend() == 'nccl' else torch.device('cpu')
            value = torch.tensor(seconds, dtype=torch.float64, device=device)
            dist.all_reduce(value, op=dist.ReduceOp.MAX)
            return float(value.item())
        from mpi4py import MPI
        return float(MPI.COMM_WORLD.allreduce(seconds, op=MPI.MAX))

    @property
    def will_close(self):
        return self._active and self._counter % self._every_n == 0

    @property
    def pending_steps(self):
        return (self._counter - 1) % self._every_n + 1 if self._active else 0

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
                        **self._epoch_args(),
                        **self._horizon_args(),
                        **self._tail_args(),
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
                        **self._epoch_args(),
                        **self._horizon_args(),
                    },
                )
                self._dft.exit_event()
            emitted = True

        self._counter = 0
        return emitted

    def discard(self) -> bool:
        """Close an already-announced window that contained no completed step.

        A loop may open a window immediately before ``next(loader)`` discovers
        end-of-input.  Its start boundary is already visible to the streaming
        tracker, so silently removing the close would corrupt the next window.
        Emit a paired, explicitly non-actionable close instead.  This is not a
        partial epoch flush: callers use it only after an unsuccessful fetch.
        """
        if not self._active:
            return False
        self._active = False
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
                    "discard": _INT_TAG(1),
                    **self._epoch_args(),
                    **self._horizon_args(),
                },
            )
            self._dft.exit_event()
        self._counter = 0
        return True

    def increase_cadence(self) -> None:
        """Double the cadence (called by optimizer when overhead is too high)."""
        if self._liveflow_v1:
            return
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
        if self._liveflow_v1:
            return
        new = max(1, self._every_n // 2)
        if new != self._every_n:
            logger.info(
                "window.cadence_decrease",
                extra={"old": self._every_n, "new": new},
            )
            self._every_n = new

    def update_cadence(self, new_every_n: int) -> None:
        """Set cadence to an explicit value (called by optimizer)."""
        if self._liveflow_v1:
            return  # v1 uses the one steady wall-time sizing rule
        new_every_n = max(1, new_every_n)
        if self._max_every_n is not None:
            new_every_n = min(new_every_n, self._max_every_n)
        if new_every_n != self._every_n:
            logger.info(
                "window.cadence_update",
                extra={"old": self._every_n, "new": new_every_n},
            )
            self._every_n = new_every_n
