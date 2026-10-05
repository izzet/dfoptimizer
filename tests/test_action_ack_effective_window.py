"""Focused compatibility tests for the optional native ACK window identity."""

import dataclasses

from dfoptimizer.optimizer import Optimizer
from dfoptimizer.planner.cbandit_planner import CoordinateBanditPlanner
from dfoptimizer.planner.planner import Planner
from dfoptimizer.runtime.context import OptimizerContext
from dfoptimizer.types import ActionPlan, DiagnosisFindingMsg, KnobDef, KnobResponse


def _plan(window_index=4):
    return ActionPlan(
        plan_id="plan-native-window", knob_id="dlio.read_threads",
        target_function="make_loader", old_value=1, new_value=3,
        apply_when="window_boundary", rationale="test", finding_type="fetch_pressure",
        severity=0.8, opportunity_tag="reader_parallelism", window_index=window_index,
    )


def _finding(window_index=4):
    return DiagnosisFindingMsg(
        finding_type="fetch_pressure", motif="test", severity="high", severity_score=0.8,
        confidence=1.0, prevalence=1.0, persistence=2, trend_direction="stable",
        contributing_facts=[], recommendation_bundle="", opportunity_tags=["reader_parallelism"],
        summary="", scope="reader", layer="reader", support_windows=2,
        last_seen_window=window_index, window_index=window_index,
    )


def _planner_with_pending_plan():
    planner = Planner()
    planner.register_knobs({
        "dlio.read_threads": KnobDef(
            id="dlio.read_threads", default=1, type=int, range=(0, 8),
            target_function="make_loader",
            responds_to={"reader_parallelism": KnobResponse(
                direction="increase", step=1, min_severity=0.5, min_persistence=1,
                cooldown_windows=3,
            )},
        )
    })
    plans = planner.process_finding(_finding())
    assert len(plans) == 1
    return planner, plans[0]


def test_context_serializes_optional_effective_window_id(monkeypatch):
    context = OptimizerContext()
    captured = []
    monkeypatch.setattr(context, "_publish_ack", captured.append)

    context.ack_action(_plan(), 1, 3, effective_window_id=9)

    assert len(captured) == 1
    assert dataclasses.asdict(captured[0])["effective_window_id"] == 9


def test_optimizer_forwards_native_window_to_planner():
    class Recorder:
        def __init__(self):
            self.kwargs = None

        def apply_ack(self, **kwargs):
            self.kwargs = kwargs

    optimizer = Optimizer()
    optimizer.planner = Recorder()
    optimizer._forward_ack_to_planner({
        "plan_id": "p", "knob_id": "dlio.read_threads", "status": "applied",
        "old_value": 1, "new_value": 3, "effective_window_id": 11,
    })

    assert optimizer.planner.kwargs["effective_window_id"] == 11


def test_collective_ack_rejects_mismatched_effective_windows(monkeypatch):
    class Recorder:
        def __init__(self):
            self.calls = []

        def apply_ack(self, **kwargs):
            self.calls.append(kwargs)

    monkeypatch.setenv("DFOPTIMIZER_EXPECTED_LOCAL_RANKS", "2")
    optimizer = Optimizer()
    optimizer.planner = Recorder()
    base = {
        "host": "node0", "plan_id": "p", "knob_id": "dlio.read_threads",
        "status": "applied", "old_value": 1, "new_value": 3,
    }
    optimizer._handle_ack_message({**base, "local_rank": 0, "global_rank": 0,
                                   "effective_window_id": 9})
    optimizer._handle_ack_message({**base, "local_rank": 1, "global_rank": 1,
                                   "effective_window_id": 10})

    assert optimizer.planner.calls == []


def test_effective_window_is_the_effectiveness_baseline():
    planner, plan = _planner_with_pending_plan()

    planner.apply_ack(
        plan.plan_id, plan.knob_id, "applied", old_value=1, new_value=3,
        window_index=-1, effective_window_id=9,
    )

    record = planner.cooldown._last_action[(plan.knob_id, "")]
    assert record.window_index == 9


def test_old_ack_without_native_window_keeps_plan_window_fallback():
    planner, plan = _planner_with_pending_plan()

    planner.apply_ack(
        plan.plan_id, plan.knob_id, "applied", old_value=1, new_value=3,
        window_index=-1,
    )

    record = planner.cooldown._last_action[(plan.knob_id, "")]
    assert record.window_index == plan.window_index


def _cbandit_knob(*, changes_regime=False, relocates_data=False):
    return KnobDef(
        id="test.data_source", default="old", type=str, values=["old", "new"],
        target_function="make_loader", changes_regime=changes_regime,
        relocates_data=relocates_data,
    )


def test_cbandit_regime_reset_prefers_native_effective_window(monkeypatch):
    planner = CoordinateBanditPlanner()
    planner.register_knobs({"test.data_source": _cbandit_knob(changes_regime=True)})
    reset_calls = []
    monkeypatch.setattr(planner, "_reset_other_arms",
                        lambda short, window: reset_calls.append((short, window)))

    planner.apply_ack(
        knob_id="test.data_source", status="applied", old_value="old", new_value="new",
        window_index=3, effective_window_id=9,
    )

    assert reset_calls == [("data_source", 9)]


def test_cbandit_regime_reset_without_native_window_keeps_receipt_window(monkeypatch):
    planner = CoordinateBanditPlanner()
    planner.register_knobs({"test.data_source": _cbandit_knob(changes_regime=True)})
    reset_calls = []
    monkeypatch.setattr(planner, "_reset_other_arms",
                        lambda short, window: reset_calls.append((short, window)))

    planner.apply_ack(
        knob_id="test.data_source", status="applied", old_value="old", new_value="new",
        window_index=3,
    )

    assert reset_calls == [("data_source", 3)]


def test_cbandit_relocation_live_from_prefers_native_effective_window():
    planner = CoordinateBanditPlanner()
    planner.register_knobs({"test.data_source": _cbandit_knob(relocates_data=True)})

    planner.apply_ack(
        knob_id="test.data_source", status="applied", old_value="old", new_value="new",
        reason="relocated", completed_at=1.0, window_index=3, effective_window_id=9,
    )

    assert planner._reloc_live_from["data_source"] == 9


def test_cbandit_relocation_without_native_window_keeps_receipt_window():
    planner = CoordinateBanditPlanner()
    planner.register_knobs({"test.data_source": _cbandit_knob(relocates_data=True)})

    planner.apply_ack(
        knob_id="test.data_source", status="applied", old_value="old", new_value="new",
        reason="relocated", completed_at=1.0, window_index=3,
    )

    assert planner._reloc_live_from["data_source"] == 3
