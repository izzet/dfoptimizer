"""A neighbour below the shipped value is never explored under strict_explore, so it must not block escalation.

Job-level TF run (2026-10-06): prefetch_size 2 -> 4 was tried and reverted, then every later window logged
`untried_neighbour:1` for prefetch_size and the planner never handed over to read_threads (reached in 1 of 3 runs)."""
import os
import sys
from collections import deque
from pathlib import Path

os.environ.update(LIVEFLOW_WORKLOAD="dlio-tf", LIVEFLOW_PRESET="liveflow-v1", DFOPTIMIZER_ALGO="cbandit",
                  DFOPTIMIZER_CBANDIT_STRICT_EXPLORE="1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dfoptimizer.planner.cbandit_planner import CoordinateBanditPlanner  # noqa: E402
from dfoptimizer.runtime.knob import knob, knob_def_from_dict  # noqa: E402


def _planner(strict):
    os.environ["DFOPTIMIZER_CBANDIT_STRICT_EXPLORE"] = "1" if strict else "0"
    pl = CoordinateBanditPlanner()
    resp = lambda tag: {tag: {"direction": "increase", "step_mode": "add", "step": 2, "min_persistence": 1,
                              "cooldown_windows": 1, "apply_when": "window_boundary"}}
    pl.register_knobs(
        {"dlio.prefetch_size": knob_def_from_dict("dlio.prefetch_size", knob(default=2, range=(1, 16), type=int,
                                                 responds_to=resp("dataloader_prefetch")), "dlio.make_loader"),
         "dlio.read_threads": knob_def_from_dict("dlio.read_threads", knob(default=4, range=(0, 192), type=int,
                                                 responds_to=resp("reader_parallelism")), "dlio.make_loader")},
        {"dlio.prefetch_size": 2, "dlio.read_threads": 4})
    return pl


def _sample(pl, level, walls):
    pl._samples[("prefetch_size", level)] = deque(walls, maxlen=64)


def test_below_shipped_neighbour_does_not_block_escalation():
    pl = _planner(strict=True)
    _sample(pl, 2, [0.0100, 0.0101, 0.0099, 0.0100])      # incumbent
    _sample(pl, 4, [0.0101, 0.0102, 0.0100, 0.0101])      # tried, not better (reverted)
    may, why = pl._escalation_state("prefetch_size", 2)
    assert may is True and why == "neighbourhood_exhausted_no_help", (may, why)


def test_an_untried_higher_neighbour_still_holds_escalation_back():
    pl = _planner(strict=True)
    _sample(pl, 2, [0.0100, 0.0101, 0.0099, 0.0100])
    may, why = pl._escalation_state("prefetch_size", 2)
    assert may is False and why.startswith("untried_neighbour:4"), (may, why)


def test_without_strict_explore_the_lower_neighbour_is_still_required():
    pl = _planner(strict=False)
    _sample(pl, 2, [0.0100, 0.0101, 0.0099, 0.0100])
    _sample(pl, 4, [0.0101, 0.0102, 0.0100, 0.0101])
    may, why = pl._escalation_state("prefetch_size", 2)
    assert may is False and why == "untried_neighbour:1", (may, why)
