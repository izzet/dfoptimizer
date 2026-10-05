"""Safety boundary tests for coherent per-rank domain metrics.

These tests use the public register_knobs/process_finding interfaces only.  No
pipeline or transport is involved.
"""

from copy import deepcopy
import math

import pytest

from dfoptimizer.planner.cbandit_planner import CoordinateBanditPlanner
from dfoptimizer.planner.clbo_planner import CLBOPlanner, domains_from_metrics
from dfoptimizer.optimizer import Optimizer
from dfoptimizer.types import ActionPlan, DiagnosisFindingMsg, KnobDef


def _knobs():
    knob = KnobDef(
        id="dlio.read_threads",
        default=1,
        type=int,
        range=(1, 4),
        target_function="make_loader",
    )
    return {knob.id: knob}


def _finding(window_index, key_metrics):
    return DiagnosisFindingMsg(
        finding_type="fetch_pressure",
        motif="",
        severity="high",
        severity_score=1.0,
        confidence=1.0,
        prevalence=1.0,
        persistence=4,
        trend_direction="stable",
        contributing_facts=[],
        recommendation_bundle="",
        opportunity_tags=["reader_parallelism"],
        summary="",
        window_index=window_index,
        key_metrics=key_metrics,
    )


def _invalid_coherent_metrics():
    # The legacy maxima would make input/comm highly actionable, but the claimed
    # coherent vector has no active rank and therefore cannot be trusted.
    return {
        "active_time": 100.0,
        "domain_coherent": 1.0,
        "domain_procs_active": 0.0,
        "domain_input_time_proc_crit": 80.0,
        "domain_comm_time_proc_crit": 70.0,
        "domain_compute_time_proc_crit": 20.0,
        "fetch_iter_time_proc_max": 90.0,
        "reader_decode_time_proc_max": 0.0,
        "comm_time_proc_max": 80.0,
        "compute_time_proc_max": 100.0,
    }


def _valid_legacy_metrics():
    return {
        "active_time": 10.0,
        "fetch_iter_time_proc_max": 8.0,
        "reader_decode_time_proc_max": 1.0,
        "comm_time_proc_max": 1.0,
        "compute_time_proc_max": 2.0,
    }


def _valid_coherent_metrics(residual=0.0):
    return {
        "active_time": 10.0,
        "domain_coherent": 1.0,
        "domain_procs_active": 1.0,
        "window_wall_time_proc_crit": 10.0,
        "domain_input_time_proc_crit": 7.0,
        "domain_decode_time_proc_crit": 1.0,
        "domain_comm_time_proc_crit": 1.0,
        "domain_compute_time_proc_crit": 1.0,
        "residual_time_proc_crit": residual,
        # Attractive legacy values ensure invalid provenance cannot fall through.
        "fetch_iter_time_proc_max": 90.0,
        "reader_decode_time_proc_max": 0.0,
        "comm_time_proc_max": 80.0,
        "compute_time_proc_max": 100.0,
    }


def _learning_state(planner):
    """State which must not change when a completed window is rejected."""
    state = {
        "current_values": dict(planner.current_values),
        "planned_cfg": deepcopy(planner._planned_cfg),
    }
    if isinstance(planner, CLBOPlanner):
        state.update(
            cfgs=deepcopy(planner._cfgs),
            domain_obs=deepcopy(planner._domain_obs),
            walls=list(planner._walls),
            incumbent_wall=planner._inc_wall,
            incumbent_cfg=deepcopy(planner._inc_cfg),
            decisions=planner._decisions,
        )
    else:
        state.update(
            stats=dict(planner._stats),
            samples={key: list(value) for key, value in planner._samples.items()},
            total_pulls=planner._total_pulls,
            epoch_dom=deepcopy(planner._epoch_dom),
            epoch_windows=planner._epoch_windows,
        )
    return state


@pytest.mark.parametrize("planner_type", [CLBOPlanner, CoordinateBanditPlanner])
def test_invalid_coherent_window_is_rejected_without_learning_and_valid_legacy_recovers(planner_type):
    planner = planner_type()
    planner.register_knobs(_knobs())

    # The next window crosses the process_finding boundary and finalizes window 0.
    assert planner.process_finding(_finding(0, _invalid_coherent_metrics())) == []
    before_invalid_boundary = _learning_state(planner)
    assert planner.process_finding(_finding(1, {})) == []
    assert _learning_state(planner) == before_invalid_boundary

    # A subsequent normal, legacy window is still admitted through the same boundary.
    assert planner.process_finding(_finding(2, _valid_legacy_metrics())) == []
    planner.process_finding(_finding(3, {}))

    if isinstance(planner, CLBOPlanner):
        assert planner._walls == [10.0]
        assert planner._inc_wall == 10.0
        assert len(planner._cfgs) == 1
    else:
        assert planner._total_pulls == 1
        assert sum(pulls for pulls, _mean in planner._stats.values()) == 1


def test_legacy_metrics_without_domain_coherent_keep_exposure_fallback():
    metrics = _valid_legacy_metrics()

    domains = domains_from_metrics(metrics)

    assert domains == {"input": 7.0, "decode": 1.0, "comm": 1.0, "compute": 1.0}
    assert metrics["domain_exposure_present"] == 0.0
    assert metrics["domain_exposure_fallback"] == 1.0


def test_cbandit_uses_the_authoritative_coherent_wall_when_legacy_wall_is_absent():
    planner = CoordinateBanditPlanner()
    planner.register_knobs(_knobs())
    metrics = _valid_coherent_metrics()
    metrics.pop("active_time")
    planner._win_metrics = metrics

    observation = planner._finalize_window(4)

    assert observation is not None
    assert observation["wall"] == metrics["window_wall_time_proc_crit"]


@pytest.mark.parametrize("field,value", [
    ("domain_coherent", 0.0), ("domain_coherent", 2.0),
    ("domain_coherent", math.nan),
    ("domain_procs_active", 0.0),
    ("window_wall_time_proc_crit", None), ("window_wall_time_proc_crit", 0.0),
    ("window_wall_time_proc_crit", -1.0), ("window_wall_time_proc_crit", math.nan),
    ("window_wall_time_proc_crit", math.inf),
    ("domain_input_time_proc_crit", None), ("domain_input_time_proc_crit", -1.0),
    ("domain_input_time_proc_crit", math.nan), ("domain_input_time_proc_crit", math.inf),
    ("domain_decode_time_proc_crit", None), ("domain_decode_time_proc_crit", -1.0),
    ("domain_decode_time_proc_crit", math.nan), ("domain_decode_time_proc_crit", math.inf),
    ("domain_comm_time_proc_crit", None), ("domain_comm_time_proc_crit", -1.0),
    ("domain_comm_time_proc_crit", math.nan), ("domain_comm_time_proc_crit", math.inf),
    ("domain_compute_time_proc_crit", None), ("domain_compute_time_proc_crit", -1.0),
    ("domain_compute_time_proc_crit", math.nan), ("domain_compute_time_proc_crit", math.inf),
])
def test_each_invalid_explicit_coherent_field_fails_closed(field, value):
    metrics = _valid_coherent_metrics()
    metrics[field] = value

    assert domains_from_metrics(metrics) is None
    assert metrics["domain_exposure_invalid"] == 1.0
    assert metrics["domain_exposure_fallback"] == 0.0


@pytest.mark.parametrize("residual", [2.0, 0.0, -2.0])
def test_valid_coherent_vector_accepts_signed_residual(residual):
    metrics = _valid_coherent_metrics(residual)

    assert domains_from_metrics(metrics) == {
        "input": 7.0, "decode": 1.0, "comm": 1.0, "compute": 1.0,
    }
    assert metrics["domain_exposure_invalid"] == 0.0


@pytest.mark.parametrize("planner_type", [CLBOPlanner, CoordinateBanditPlanner])
def test_valid_invalid_valid_sequence_does_not_carry_invalid_learning(planner_type):
    planner = planner_type()
    planner.register_knobs(_knobs())

    assert planner.process_finding(_finding(0, _valid_coherent_metrics())) == []
    planner.process_finding(_finding(1, _invalid_coherent_metrics()) )
    after_valid = _learning_state(planner)

    # This boundary closes the invalid window.  It must neither act nor mutate
    # learning/epoch state, even though the fixture carries attractive legacy data.
    assert planner.process_finding(_finding(2, _valid_legacy_metrics())) == []
    assert _learning_state(planner) == after_valid

    # The following boundary closes the later valid window and proves recovery.
    planner.process_finding(_finding(3, {}))
    recovered = _learning_state(planner)
    if isinstance(planner, CLBOPlanner):
        assert len(recovered["walls"]) == len(after_valid["walls"]) + 1
    else:
        assert recovered["total_pulls"] == after_valid["total_pulls"] + 1


def test_experiment_allowlist_and_ack_cap_are_opt_in_and_fail_closed(monkeypatch):
    monkeypatch.setenv("DFOPTIMIZER_ALLOWED_KNOBS", "dlio.read_threads")
    monkeypatch.setenv("DFOPTIMIZER_ACTION_ACK_CAP", "1")
    optimizer = Optimizer()
    read = KnobDef(id="dlio.read_threads", default=1, type=int, range=(1, 4))
    prefetch = KnobDef(id="dlio.prefetch_size", default=2, type=int, range=(1, 8))

    kept, values = optimizer._filter_experiment_knobs(
        {read.id: read, prefetch.id: prefetch},
        {read.id: 1, prefetch.id: 2},
        source="test",
    )
    assert set(kept) == {"dlio.read_threads"}
    assert values == {"dlio.read_threads": 1}
    assert not optimizer._action_capacity_exhausted()
    first = ActionPlan(
        plan_id="plan-1", knob_id="dlio.read_threads", old_value=1, new_value=2,
        target_function="make_loader", apply_when="window_boundary", rationale="test",
        finding_type="test", severity=1.0, opportunity_tag="reader_parallelism", window_index=1,
    )
    second = ActionPlan(
        plan_id="plan-2", knob_id="dlio.read_threads", old_value=2, new_value=3,
        target_function="make_loader", apply_when="window_boundary", rationale="test",
        finding_type="test", severity=1.0, opportunity_tag="reader_parallelism", window_index=2,
    )
    admitted, suppressed = optimizer._reserve_experiment_plans([first])
    assert admitted == [first] and suppressed == []
    # A second eligible control window before the first ack cannot publish a
    # second change, which is the experiment's one-action safety boundary.
    admitted, suppressed = optimizer._reserve_experiment_plans([second])
    assert admitted == [] and suppressed == [second]
    assert optimizer._action_capacity_exhausted()
    assert optimizer._record_acknowledged_action(
        plan_id="plan-1", status="applied", old_value=1, new_value=2
    )
    assert optimizer._action_capacity_exhausted()
    # Every rank may acknowledge the same plan, but it is only one change.
    assert not optimizer._record_acknowledged_action(
        plan_id="plan-1", status="applied", old_value=1, new_value=2
    )

    # A declined pending plan releases its reservation rather than consuming a
    # change that the workload never accepted.
    monkeypatch.setenv("DFOPTIMIZER_ACTION_ACK_CAP", "1")
    released = Optimizer()
    admitted, _ = released._reserve_experiment_plans([first])
    assert admitted == [first]
    assert not released._record_acknowledged_action(
        plan_id="plan-1", status="rejected", old_value=1, new_value=2
    )
    admitted, suppressed = released._reserve_experiment_plans([second])
    assert admitted == [second] and suppressed == []
