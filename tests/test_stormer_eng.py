"""stormer-eng: apply-cost-aware gate, tail-aware verification, applied-vector credit, Window tail/physical fields.

Window metrics follow the schema of a REAL retained StorMer v1 window (stormer_eng_window_fixture.json, 8N attempt2)."""
import json
import os
import sys
from collections import deque
from pathlib import Path

os.environ.update(LIVEFLOW_WORKLOAD="stormer", LIVEFLOW_PRESET="liveflow-v1", DFOPTIMIZER_ALGO="cbandit",
                  DFOPTIMIZER_CBANDIT_STRICT_EXPLORE="1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dfoptimizer.planner.cbandit_planner import CoordinateBanditPlanner  # noqa: E402
from dfoptimizer.runtime.knob import knob, knob_def_from_dict  # noqa: E402
from dfoptimizer.runtime.window import Window  # noqa: E402
from dfoptimizer.types import DiagnosisFindingMsg  # noqa: E402

FIX = json.loads((Path(__file__).parent / "stormer_eng_window_fixture.json").read_text())["3"]


def _resp(tag):
    return {tag: {"direction": "increase", "step_mode": "evidence", "min_persistence": 2,
                  "cooldown_windows": 3, "apply_when": "window_boundary"}}


def _planner():
    pl = CoordinateBanditPlanner()
    pl.register_knobs(
        {"stormer.num_workers": knob_def_from_dict(
            "stormer.num_workers", knob(default=8, range=(0, 40), type=int, responds_to=_resp("reader_parallelism")),
            "stormer.make_loader"),
         "stormer.prefetch_factor": knob_def_from_dict(
            "stormer.prefetch_factor", knob(default=2, range=(1, 16), type=int, responds_to=_resp("dataloader_prefetch")),
            "stormer.make_loader")},
        {"stormer.num_workers": 8, "stormer.prefetch_factor": 2})
    return pl


def _metrics(w, wall, p90=None, vec=(8, 2), applied=0, cost_us=None, step0=None):
    m = dict(FIX)
    n = 102
    first = step0 if step0 is not None else 41 + (w - 3) * n
    m.update(active_time=wall, lf_v1_active_time=wall, lf_v1_storage_wait=wall - 0.27, step_min=float(first),
             step_max=float(first + n - 1), completed_steps=float(first + n - 1),
             remaining_steps=float(1802 - (first + n - 1)))
    m.update(physical_num_workers=vec[0], physical_prefetch_factor=vec[1], physical_bucket_cap_mb=25,
             physical_applied_step=applied)
    if cost_us is not None:
        m["physical_apply_cost_us"] = cost_us
    if p90 is not None:
        m["step_wall_p90_us"] = int(p90 * 1e6)
    return m


def _finding(w, m):
    return DiagnosisFindingMsg(
        finding_type="xlayer_contrast_attribution_v2", motif="", severity="", severity_score=1.0, confidence=1.0,
        prevalence=1.0, persistence=1, trend_direction="", contributing_facts=[], recommendation_bundle="",
        opportunity_tags=["reader_parallelism"], summary="", scope="", layer="reader_posix", window_index=w,
        key_metrics=m)


def test_every_node_learns_the_apply_cost_and_state_from_the_shared_vector():
    pls = [_planner(), _planner()]
    seq = [(3, 0.50, 1.2, (8, 2), 0, None), (4, 0.51, 1.2, (8, 2), 0, None), (5, 0.52, 1.1, (8, 2), 0, None),
           (6, 0.51, 1.0, (8, 2), 0, None),
           # applied 8 -> 16 at boundary 550: the window starting at 551 has applied_step 551 (excluded), the next is clean
           (7, 0.66, 1.1, (16, 2), 551, 9_400_000), (8, 0.50, 0.7, (16, 2), 551, 9_400_000),
           (9, 0.50, 0.7, (16, 2), 551, 9_400_000)]
    for pl in pls:
        for w, wall, p90, vec, app, cost in seq + [(10, 0.5, 0.7, (16, 2), 551, 9_400_000)]:
            pl.process_finding(_finding(w, _metrics(w, wall, p90, vec, app, cost, step0=449 + (w - 7) * 102 + 102 * 0
                                                    if w >= 7 else None)))
    for pl in pls:
        assert pl.current_values["stormer.num_workers"] in (16, 24) or pl.current_values["stormer.num_workers"] == 16
        assert pl._apply_costs.get("num_workers") == [(8.0, 9.4)], pl._apply_costs
        # the apply window (first step <= applied_step) is not an after-sample of either arm
        assert all(len(v) <= 4 for (k, lv), v in pl._samples.items() if k == "num_workers" and lv == 16)
    assert pls[0]._move_cost("num_workers", 32) == pls[1]._move_cost("num_workers", 32)


def test_move_cost_is_the_measured_apply_scaled_by_change_size():
    pl = _planner()
    pl._apply_costs["num_workers"] = [(8.0, 10.0)]
    pl.current_values["stormer.num_workers"] = 16
    assert abs(pl._move_cost("num_workers", 24) - 10.0) < 1e-9
    assert abs(pl._move_cost("num_workers", 32) - 20.0) < 1e-9
    assert pl._move_cost("prefetch_factor", 4) == 0.0          # nothing measured for that knob: no invented cost


def test_apply_cost_veto_uses_predicted_gain_over_remaining_steps():
    pl = _planner()
    pl._apply_costs["num_workers"] = [(8.0, 70.0)]
    assert pl._apply_cost_veto("num_workers", 32, 40, delta=0.2, remaining=300.0, window=1) is not None   # 60 s < 70 s
    assert pl._apply_cost_veto("num_workers", 32, 40, delta=0.2, remaining=400.0, window=1) is None       # 80 s > 70 s
    assert pl._apply_cost_veto("num_workers", 32, 40, delta=None, remaining=400.0, window=1) is None
    assert pl._apply_cost_veto("prefetch_factor", 2, 4, delta=0.2, remaining=1.0, window=1) is None       # unmeasured knob


def _arm(pl, level, walls, tails):
    pl._samples[("num_workers", level)] = deque(walls, maxlen=32)
    pl._tail_samples[("num_workers", level)] = deque(tails, maxlen=32)


def test_tail_confirms_what_the_mean_cannot_and_reverts_when_neither_improves():
    pl = _planner()
    pl._rate_mode = False
    pl.current_values["stormer.num_workers"] = 24
    _arm(pl, 8, [0.500, 0.537, 0.547, 0.491, 0.572], [0.84, 1.252, 1.205, 0.926, 1.368])
    _arm(pl, 24, [0.724, 0.438, 0.373], [0.678, 0.476, 0.403])                  # real attempt2 W8 / W24 windows
    pl._move_from["num_workers"] = (8, 0.2277, 5, 0.529, 24)
    assert pl._unconfirmed_move(14, 800.0) is None, "tail improvement should confirm the move"
    assert "num_workers" not in pl._move_from
    # same mean, no tail improvement: reverted
    pl2 = _planner()
    pl2._rate_mode = False
    pl2.current_values["stormer.num_workers"] = 24
    _arm(pl2, 8, [0.500, 0.537, 0.547, 0.491, 0.572], [0.84, 1.252, 1.205, 0.926, 1.368])
    _arm(pl2, 24, [0.52, 0.51, 0.55], [1.1, 1.2, 1.3])
    pl2._move_from["num_workers"] = (8, 0.2277, 5, 0.529, 24)
    assert pl2._unconfirmed_move(14, 800.0) == ("num_workers", 8)
    # a lower tail never excuses a significantly worse mean
    pl3 = _planner()
    pl3._rate_mode = False
    pl3.current_values["stormer.num_workers"] = 24
    _arm(pl3, 8, [0.50, 0.51, 0.52, 0.50, 0.51], [1.2, 1.3, 1.25, 1.3, 1.2])
    _arm(pl3, 24, [0.90, 0.95, 0.92], [0.5, 0.6, 0.55])
    pl3._move_from["num_workers"] = (8, 0.2277, 5, 0.51, 24)
    assert pl3._unconfirmed_move(14, 800.0) == ("num_workers", 8)


def test_tail_needs_min_samples_in_both_arms():
    pl = _planner()
    assert not pl._tail_accepts([1.2, 1.2], [0.5, 0.5, 0.5], [0.5, 0.5], [0.5, 0.5, 0.5])
    assert not pl._tail_accepts([1.2, 1.2, 1.2], [0.5, 0.5], [0.5] * 3, [0.5] * 2)


class _Dft:
    def __init__(self):
        self.events = []
        self.t = 0

    def get_time(self):
        self.t += 1000
        return self.t

    def enter_event(self):
        pass

    def exit_event(self):
        pass

    def log_event(self, **kw):
        self.events.append(kw)


def test_window_emits_p90_step_wall_and_generic_physical_vector():
    dft = _Dft()
    w = Window(dft, max_every_n=50, epochs_total=2, steps_per_epoch=100)
    w.set_epoch(1)
    w.set_physical_config({"num_workers": 16, "prefetch_factor": 2, "bucket_cap_mb": 25}, 551, apply_cost_s=9.4)
    for step in range(1, 21):
        w.start()
        w._step_started -= (0.1 if step != 7 else 3.0)      # one 3 s step, nineteen 0.1 s steps
        w.stop()
    stop = [e for e in dft.events if e["name"] == "window.stop"][-1]["int_args"]
    start = [e for e in dft.events if e["name"] == "window.start"][-1]["int_args"]
    assert "step_wall_p90_us" in stop and 90_000 < stop["step_wall_p90_us"][1] < 130_000, stop["step_wall_p90_us"]
    assert start["physical_num_workers"][1] == 16 and start["physical_prefetch_factor"][1] == 2
    assert start["physical_applied_step"][1] == 551 and start["physical_apply_cost_us"][1] == 9_400_000
    assert Window._p90([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) == 9           # nearest-rank


def test_apply_probe_measures_the_stall_of_the_step_after_the_apply():
    dft = _Dft()
    w = Window(dft, max_every_n=50, epochs_total=2, steps_per_epoch=100)
    w.set_epoch(1)
    for step in range(1, 8):
        w.start()
        w._step_started -= 0.1
        w.stop()
    w.begin_apply_probe(1.0)                    # apply latency 1 s
    assert w.apply_probe_result() is None       # next step not completed yet
    w.start()
    w._step_started -= 9.0                      # the step after the apply took ~9.1 s
    w.stop()
    cost = w.apply_probe_result()
    assert 8.9 < cost < 9.2, cost               # wall of that step less the 0.1 s median
    assert w.apply_probe_result() is None       # one measurement per apply


def test_refined_apply_cost_replaces_the_first_measurement_of_the_same_apply():
    pl = _planner()
    for w, wall, p90, vec, app, cost, first in [(3, 0.5, 1.2, (8, 2), 0, None, 41), (4, 0.5, 1.2, (8, 2), 0, None, 143),
                                                (5, 0.5, 1.2, (16, 2), 551, 1_000_000, 551),     # latency only (apply window)
                                                (6, 0.5, 0.8, (16, 2), 551, 9_800_000, 653),     # refined: step after the apply
                                                (7, 0.5, 0.8, (16, 2), 551, 9_800_000, 755)]:
        pl.process_finding(_finding(w, _metrics(w, wall, p90, vec, app, cost, step0=first)))
    pl.process_finding(_finding(8, _metrics(8, 0.5, 0.8, (16, 2), 551, 9_800_000, step0=857)))
    assert pl._apply_costs["num_workers"] == [(8.0, 9.8)], pl._apply_costs


def test_one_change_in_flight_blocks_every_other_move_until_a_verdict():
    pl = _planner()
    plans = []
    # the applied vector never changes (the first plan is not applied): nothing else may be proposed
    for w in range(3, 14):
        plans += pl.process_finding(_finding(w, _metrics(w, 0.5, 1.2, (8, 2), 0, None)))
    assert [(p.knob_id, p.new_value) for p in plans] == [("stormer.num_workers", 16)], plans
    assert pl._inflight_move() == "num_workers"


def test_refused_move_is_no_longer_in_flight():
    pl = _planner()
    for w in range(3, 8):
        pl.process_finding(_finding(w, _metrics(w, 0.5, 1.2, (8, 2), 0, None)))
    assert pl._inflight_move() == "num_workers"
    pl.apply_ack(plan_id="x", knob_id="stormer.num_workers", status="rejected", old_value=8, new_value=16)
    assert pl._inflight_move() is None
