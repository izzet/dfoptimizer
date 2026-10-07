"""Little's-law sizing (LIVEFLOW_SIZING=littles_law|geometric): needed_s = ceil(lambda * S_s).

Pure arithmetic (planner/sizing.py), then the planner path: one move to the model's level, joint plans for several
under-provisioned stages of the input side, revert of the whole plan, fallback to the geometric/refinement step.
"""
import math

import numpy as np
import pytest

from dfoptimizer import step_series
from dfoptimizer.planner import sizing
from dfoptimizer.planner.cbandit_planner import CoordinateBanditPlanner
from dfoptimizer.types import DiagnosisFindingMsg, KnobDef


# ------------------------------------------------------------------ arithmetic
def test_needed_parallelism_is_the_ceiling_of_rate_times_service():
    assert sizing.needed_parallelism(0.0025, 0.0097) == 1            # a mean service well below one worker
    assert sizing.needed_parallelism(0.092, 0.017) == 6              # ceil(5.41)
    assert sizing.needed_parallelism(0.040, 0.010) == 4              # exactly 4.0 stays 4 (float guard)
    assert sizing.needed_parallelism(0.0400001, 0.010) == 5


def test_free_step_time_removes_every_attributed_input_wait_and_nothing_else():
    d = {"input": 0.004, "decode": 0.002, "delivery": 0.001, "comm": 0.03, "compute": 0.05}
    assert math.isclose(sizing.free_step_time(0.09, d), 0.083)
    assert sizing.free_step_time(0.005, d) is None                    # wait >= wall: no free rate is defined


def _m(**kw):
    m = {"lf_v1_svc_storage": 0.012, "lf_v1_svc_storage_critical": 1.0, "lf_v1_svc_decode": 0.030, "lf_v1_svc_decode_critical": 1.0}
    m.update(kw)
    return m


def test_size_knobs_uses_each_knobs_own_stage_and_caps_at_the_registry_maximum():
    knobs = {"read_threads": (1, 0, 192), "decode_threads": (1, 1, 8), "computation_threads": (1, 1, 4), "prefetch_size": (2, 1, 16)}
    got = sizing.size_knobs(_m(), 0.05, {"input": 0.01, "decode": 0.01}, knobs)      # T_free = 0.03
    assert got["read_threads"]["needed"] == 1 and got["read_threads"]["target"] == 1  # 0.012/0.03 = 0.4
    assert got["decode_threads"]["needed"] == 1 and got["decode_threads"]["target"] == 1
    got = sizing.size_knobs(_m(lf_v1_svc_decode=0.500), 0.05, {"input": 0.01, "decode": 0.01}, knobs)
    assert got["decode_threads"]["needed"] == 17 and got["decode_threads"]["target"] == 8 and got["decode_threads"]["capped"]
    assert got["computation_threads"]["target"] == 4                                  # its own registry maximum
    assert "prefetch_size" not in got                                                 # not a parallelism pool
    assert math.isclose(got["decode_threads"]["lambda_per_s"], 1 / 0.03, rel_tol=1e-3)


def test_missing_or_nonfinite_service_and_zero_normalisation_suppress():
    knobs = {"read_threads": (1, 0, 192)}
    assert sizing.size_knobs({}, 0.05, {"input": 0.01}, knobs) == {}
    assert sizing.size_knobs(_m(lf_v1_svc_storage=float("nan")), 0.05, {"input": 0.01}, knobs) == {}
    assert sizing.size_knobs(_m(lf_v1_svc_storage=0.0), 0.05, {"input": 0.01}, knobs) == {}
    assert sizing.size_knobs(_m(), 0.01, {"input": 0.02}, knobs) == {}               # negative free time


def test_num_workers_carries_every_measured_worker_side_stage():
    got = sizing.size_knobs(_m(), 0.05, {"input": 0.01}, {"num_workers": (1, 1, 32)})   # (0.012 + 0.030)/0.04
    assert got["num_workers"]["needed"] == 2 and got["num_workers"]["stages"] == "storage+decode"
    got = sizing.size_knobs({"lf_v1_svc_storage": 0.09}, 0.05, {"input": 0.01}, {"num_workers": (1, 1, 32)})
    assert got["num_workers"]["needed"] == 3 and got["num_workers"]["stages"] == "storage"


def test_mode_selection(monkeypatch):
    monkeypatch.delenv("LIVEFLOW_SIZING", raising=False)
    assert sizing.sizing_mode() == "littles_law"
    monkeypatch.setenv("LIVEFLOW_SIZING", "geometric")
    assert sizing.sizing_mode() == "geometric"
    monkeypatch.setenv("LIVEFLOW_SIZING", "bogus")
    assert sizing.sizing_mode() == "littles_law"


# ------------------------------------------------------------------ planner path
STEPS_PER_WINDOW = 120


def _planner(monkeypatch, mode):
    monkeypatch.setenv("LIVEFLOW_SIZING", mode)
    monkeypatch.setenv("LIVEFLOW_VERIFY", "per_step")
    monkeypatch.setenv("LIVEFLOW_PRESET", "liveflow-v1")
    monkeypatch.setenv("DFOPTIMIZER_CBANDIT_STRICT_EXPLORE", "0")
    p = CoordinateBanditPlanner()
    defs = {f"halo3d.{k}": KnobDef(id=f"halo3d.{k}", default=1, type=int, range=(1, 8), cost_seconds=0.001,
                                   target_function="halo3d.make_loader") for k in ("read_threads", "decode_threads")}
    p.register_knobs(defs, current_values={k: 1 for k in defs})
    p._min_samples = 3
    return p


def _finding(w, cfg, first, wall=0.1, wait_in=0.0, wait_dec=0.05, svc_in=0.004, svc_dec=0.2, applied=0, rng=None):
    """One window of a halo-like loop: step wall `wall`, exposed storage / decode wait, per-stage service (critical rank)."""
    rng = rng or np.random.default_rng(w)
    n = STEPS_PER_WINDOW
    wall_s = wall + rng.normal(0, 0.002, n)
    sto = np.full(n, wait_in) + rng.normal(0, 1e-5, n)
    dec = np.full(n, wait_dec) + rng.normal(0, 1e-4, n)
    comm = np.full(n, 0.01)
    series = step_series.encode(first, {"wall": wall_s.tolist(), "storage": np.maximum(sto, 0).tolist(),
                                        "decode": np.maximum(dec, 0).tolist(), "comm": comm.tolist()})
    compute = max(wall - wait_in - wait_dec - 0.01, 0.0)
    m = {"liveflow_v1_attribution": 1.0, "lf_v1_per_step": 1.0, "planner_history_ready": 1.0, "lf_v1_valid": 1.0,
         "lf_v1_storage_wait": wait_in, "lf_v1_decode_wait": wait_dec, "lf_v1_delivery_wait": 0.0, "lf_v1_delivery_declared": 0.0,
         "lf_v1_comm_wait": 0.01, "lf_v1_compute_time": compute, "lf_v1_active_time": wall, "active_time": wall,
         "step_nunique": float(n), "completed_steps": float(first + n - 1), "configured_window_steps": float(n),
         "remaining_steps": 5000.0, "step_wall_p90_us": float(np.quantile(wall_s, 0.9) * 1e6),
         "physical_read_threads": float(cfg["read_threads"]), "physical_decode_threads": float(cfg["decode_threads"]),
         "physical_applied_step": float(applied), "step_series": series,
         "lf_v1_svc_storage": svc_in, "lf_v1_svc_storage_critical": 1.0, "lf_v1_svc_decode": svc_dec,
         "lf_v1_svc_decode_critical": 1.0}
    return DiagnosisFindingMsg(finding_type="fetch_pressure", motif="", severity="high", severity_score=1.0, confidence=1.0,
                               prevalence=1.0, persistence=4, trend_direction="stable", contributing_facts=[],
                               recommendation_bundle="", opportunity_tags=["reader_parallelism"], summary="", window_index=w,
                               key_metrics=m)


def _run_until_plan(p, **kw):
    cfg = {"read_threads": 1, "decode_threads": 1}
    first = 6
    for w in range(1, 8):
        plans = p.process_finding(_finding(w, cfg, first, **kw))
        first += STEPS_PER_WINDOW
        if plans:
            return plans
    return []


def test_decode_exposure_is_sized_to_the_model_level_in_one_move(monkeypatch):
    p = _planner(monkeypatch, "littles_law")
    # wall 0.10, exposed decode 0.05 -> T_free 0.05; decode service 0.2 s/step -> ceil(4.0) = 4: one move 1 -> 4 (geometric would be 2)
    plans = _run_until_plan(p)
    assert [(x.knob_id, x.new_value) for x in plans] == [("halo3d.decode_threads", 4)]


def test_geometric_mode_takes_the_next_x2_step(monkeypatch):
    p = _planner(monkeypatch, "geometric")
    plans = _run_until_plan(p)
    assert [(x.knob_id, x.new_value) for x in plans] == [("halo3d.decode_threads", 2)]


def test_the_model_level_equal_to_the_next_geometric_step_changes_nothing(monkeypatch):
    p = _planner(monkeypatch, "littles_law")
    plans = _run_until_plan(p, svc_dec=0.09)                         # ceil(0.09/0.05) = 2 = the next x2 step
    assert [(x.knob_id, x.new_value) for x in plans] == [("halo3d.decode_threads", 2)]


def test_a_stage_the_model_finds_adequately_provisioned_is_not_moved_ahead_of_one_it_sizes_above_current(monkeypatch):
    p = _planner(monkeypatch, "littles_law")
    plans = _run_until_plan(p, wait_in=0.02, wait_dec=0.03, svc_in=0.004, svc_dec=0.3)   # T_free 0.05: read need 1, decode need 6
    assert [(x.knob_id, x.new_value) for x in plans] == [("halo3d.decode_threads", 6)]


def test_several_under_provisioned_stages_are_sized_jointly_in_one_plan(monkeypatch):
    p = _planner(monkeypatch, "littles_law")
    plans = _run_until_plan(p, wait_in=0.025, wait_dec=0.025, svc_in=0.15, svc_dec=0.2)  # T_free 0.05: read 3, decode 4
    assert {x.knob_id: x.new_value for x in plans} == {"halo3d.read_threads": 3, "halo3d.decode_threads": 4}
    assert len({x.window_index for x in plans}) == 1                                    # one apply
    lead = next(iter(p._ll_partners))
    assert p._ll_partners[lead][1]                                                      # remembered for a joint revert


def test_a_partner_without_exposed_wait_is_not_moved(monkeypatch):
    p = _planner(monkeypatch, "littles_law")
    plans = _run_until_plan(p, wait_in=0.0, wait_dec=0.05, svc_in=0.4, svc_dec=0.2)     # read would need 8 but nobody waits on storage
    assert {x.knob_id: x.new_value for x in plans} == {"halo3d.decode_threads": 4}


def test_missing_service_falls_back_to_geometric_unchanged(monkeypatch):
    p = _planner(monkeypatch, "littles_law")
    cfg = {"read_threads": 1, "decode_threads": 1}
    first, plans = 6, []
    for w in range(1, 8):
        f = _finding(w, cfg, first)
        for k in ("lf_v1_svc_storage", "lf_v1_svc_decode"):
            f.key_metrics.pop(k)
        plans = p.process_finding(f)
        first += STEPS_PER_WINDOW
        if plans:
            break
    assert [(x.knob_id, x.new_value) for x in plans] == [("halo3d.decode_threads", 2)]


def test_a_revert_restores_the_whole_joint_plan_and_the_next_raise_is_a_refinement(monkeypatch):
    p = _planner(monkeypatch, "littles_law")
    kw = dict(wait_in=0.025, wait_dec=0.025, svc_in=0.15, svc_dec=0.2)
    plans = _run_until_plan(p, **kw)
    assert {x.knob_id: x.new_value for x in plans} == {"halo3d.read_threads": 3, "halo3d.decode_threads": 4}
    w0 = max(x.window_index for x in plans)
    first = 6 + STEPS_PER_WINDOW * w0
    applied = first - 1                                    # the plan was applied at the boundary before the next window
    new = {"read_threads": 3, "decode_threads": 4}
    # the verdict path says revert (a worse step time after the apply): both knobs go back in ONE plan
    p._sv_verdict = lambda *a, **k: "revert"
    out = []
    for w in range(w0 + 1, w0 + 4):
        out = p.process_finding(_finding(w, new, first, applied=applied, wall=0.12, **kw))
        first += STEPS_PER_WINDOW
        if out:
            break
    assert {x.knob_id: x.new_value for x in out} == {"halo3d.read_threads": 1, "halo3d.decode_threads": 1}
    assert not p._ll_partners


def test_after_a_revert_the_model_level_is_not_proposed_again_from_the_restored_value(monkeypatch):
    p = _planner(monkeypatch, "littles_law")
    _run_until_plan(p)                                      # a window's sizing is now present
    assert p._ll_level("decode_threads", 1) == 4
    p._refine["decode_threads"] = True                      # set by the planner when it observes a decrease
    assert p._ll_level("decode_threads", 1) is None
