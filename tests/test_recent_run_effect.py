"""Recent-run (Amdahl) predicted effect: used only when the epoch-to-date bound is unusable and no higher level was measured."""
from collections import deque

from dfoptimizer.planner.cbandit_planner import CoordinateBanditPlanner
from dfoptimizer.types import KnobDef


def _planner():
    p = CoordinateBanditPlanner()
    p.register_knobs({"halo3d.read_threads": KnobDef(id="halo3d.read_threads", default=1, type=int, range=(1, 8),
                                                     cost_seconds=0.001, target_function="halo3d.make_loader")})
    p._rate_mode = True
    p._min_samples = 3          # other tests leak DFOPTIMIZER_CBANDIT_MIN_SAMPLES into the environment
    return p


def test_prediction_from_sustained_exposure_scaled_by_next_level():
    p = _planner()
    p._epoch_windows = 20
    p._epoch_dom["input"] = deque([0.0] * 15 + [0.01, 0.04, 0.08, 0.10, 0.12])   # past epoch unexposed, now exposed
    assert p._epoch_domain_exposed("input") is None                                # the epoch bound alone is unusable
    delta, src = p.predicted_effect("read_threads", "input")
    assert abs(delta - 0.08 * (1 - 1 / 2)) < 1e-12 and "amdahl:1.0->2" in src     # min of last 3, share 1-cur/next


def test_no_prediction_when_run_shorter_than_min_samples_or_higher_level_measured():
    p = _planner()
    p._epoch_windows = 20
    p._epoch_dom["input"] = deque([0.0] * 18 + [0.1, 0.1])
    assert p.predicted_effect("read_threads", "input")[0] is None
    p._epoch_dom["input"] = deque([0.0] * 15 + [0.1] * 5)
    p._samples[("read_threads", 2)] = deque([0.2, 0.2, 0.2])
    assert p.predicted_effect("read_threads", "input")[0] is None                  # a measured level decides instead


def test_recent_run_move_is_verified_against_its_own_premove_walls():
    p = _planner()
    p._move_base_walls["read_threads"] = [0.27, 0.28, 0.275]
    p._samples[("read_threads", 1)] = deque([0.09] * 15 + [0.27, 0.28, 0.275])      # phase-1 windows contaminate the arm
    p._samples[("read_threads", 2)] = deque([0.22, 0.23, 0.22])
    p._move_from["read_threads"] = (1, 0.04, 20, 0.2, 2)
    p._rate_mode = False
    p._physical_latest_cfg = None
    p.current_values["halo3d.read_threads"] = 2
    p._planned_cfg[21] = {"read_threads": 2}
    assert p._unconfirmed_move(24, 3000.0) is None


def test_pool_knob_moves_geometrically_capped_then_refines_after_a_revert():
    p = _planner()
    assert [p._next_up("read_threads", c) for c in (1, 2, 3, 4, 5, 8)] == [2, 4, 6, 8, 8, None]
    p._refine["read_threads"] = True                       # a revert was observed
    assert p._next_up("read_threads", 2) == 3              # one refinement step, not x2
    p._refine["read_threads"] = False
    assert p._next_up("read_threads", 2) == 4
    p._rate_mode = False
    assert p._next_up("read_threads", 2) == 3              # outside rate mode the grid neighbour is unchanged


def test_observed_decrease_sets_refinement_and_observed_raise_clears_it():
    p = _planner()
    p._physical_latest_cfg = {"read_threads": 2}
    p._refine["read_threads"] = False
    metrics = {"liveflow_v1_attribution": 1.0, "physical_read_threads": 1, "physical_applied_step": 10,
               "step_min": 11, "step_nunique": 5}
    assert p._observed_cfg(5, metrics) == {"read_threads": 1}
    assert p._refine["read_threads"] is True
    metrics = dict(metrics, physical_read_threads=2, physical_applied_step=30, step_min=31)
    p._observed_cfg(6, metrics)
    assert p._refine["read_threads"] is False


def _bucket_planner(**kw):
    p = CoordinateBanditPlanner()
    p.register_knobs({"stormer.bucket_cap_mb": KnobDef(id="stormer.bucket_cap_mb", default=25, type=int,
                                                       values=[25, 50, 100, 200, 400], cost_seconds=15.0,
                                                       target_function="stormer.train", **kw)})
    p._rate_mode = True
    p._min_samples = 3
    return p


def test_bucket_cap_moves_x4_capped_at_declared_max_then_refines_after_a_revert():
    p = _bucket_planner()
    assert [p._next_up("bucket_cap_mb", c) for c in (25, 50, 100, 200, 400)] == [100, 200, 400, 400, None]
    p._refine["bucket_cap_mb"] = True                          # the 25 -> 100 move was reverted
    assert p._next_up("bucket_cap_mb", 25) == 50               # one refinement step
    p._refine["bucket_cap_mb"] = False
    assert p._next_up("bucket_cap_mb", 50) == 200              # then geometric again
    p._rate_mode = False
    assert p._next_up("bucket_cap_mb", 25) == 50               # outside rate mode: grid neighbour


def test_bucket_cap_neighbourhood_is_stay_previous_level_and_geometric_up():
    p = _bucket_planner()
    assert p._neighbourhood("bucket_cap_mb", 25) == [25, 100]
    assert p._neighbourhood("bucket_cap_mb", 100) == [100, 25, 400]
    assert p._neighbourhood("bucket_cap_mb", 400) == [400, 100]
    assert p._neighbourhood("bucket_cap_mb", 50) == [50, 25, 200]    # refinement level steps back one grid level
    p._rate_mode = False
    assert p._neighbourhood("bucket_cap_mb", 100) == [100, 50, 200]


def test_declared_step_factor_overrides_the_default_and_round_trips_on_the_wire():
    from dfoptimizer.runtime.knob import knob_def_from_wire, knob_def_to_wire
    p = _bucket_planner(step_factor=2.0)
    assert [p._next_up("bucket_cap_mb", c) for c in (25, 50, 100)] == [50, 100, 200]
    assert knob_def_from_wire(knob_def_to_wire(p.knobs["stormer.bucket_cap_mb"])).step_factor == 2.0
    assert _bucket_planner().knobs["stormer.bucket_cap_mb"].step_factor is None


def test_predicted_effect_for_bucket_cap_uses_the_geometric_next_level():
    p = _bucket_planner()
    p._epoch_windows = 20
    p._epoch_dom["comm"] = deque([0.0] * 15 + [0.15, 0.16, 0.15, 0.15, 0.15])
    delta, src = p.predicted_effect("bucket_cap_mb", "comm")
    assert delta is not None and abs(delta - 0.15 * (1 - 25 / 100)) < 1e-12 and "25.0->100" in src
