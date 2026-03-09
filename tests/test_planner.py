from dfoptimizer.planner.planner import Planner
from dfoptimizer.types import DiagnosisFindingMsg, KnobDef, KnobResponse


def _make_finding(**overrides):
    payload = {
        "finding_type": "fetch_pressure",
        "motif": "persistent_pressure",
        "severity": "critical",
        "confidence": 0.9,
        "prevalence": 1.0,
        "persistence": 4,
        "trend_direction": "stable",
        "contributing_facts": [("fetch_pressure", "reader_posix:epoch")],
        "recommendation_bundle": "input_pipeline_tuning",
        "opportunity_tags": ["input_pipeline_tuning"],
        "summary": "fetch_pressure(reader_posix:epoch)",
        "scope": "reader_posix:epoch",
        "layer": "reader_posix",
        "support_windows": 4,
        "last_seen_window": 3,
        "window_index": 3,
        "publish_mode": "control",
    }
    payload.update(overrides)
    return DiagnosisFindingMsg(**payload)


def test_planner_skips_non_control_publish_modes():
    planner = Planner()
    planner.register_knobs(
        {
            "dlio.prefetch_size": KnobDef(
                id="dlio.prefetch_size",
                default=2,
                type=int,
                range=(1, 32),
                target_function="make_loader",
                responds_to={
                    "input_pipeline_tuning": KnobResponse(
                        direction="increase",
                        step=2,
                        min_severity=0.5,
                        min_persistence=2,
                        cooldown_windows=0,
                    )
                },
            )
        }
    )

    finding = _make_finding(publish_mode="summary")

    assert planner.process_finding(finding) == []


def test_planner_applies_minimum_integer_step_for_stable_findings():
    planner = Planner()
    planner.register_knobs(
        {
            "dlio.read_threads": KnobDef(
                id="dlio.read_threads",
                default=0,
                type=int,
                range=(0, 4),
                target_function="make_loader",
                responds_to={
                    "input_pipeline_tuning": KnobResponse(
                        direction="increase",
                        step=1,
                        min_severity=0.5,
                        min_persistence=2,
                        cooldown_windows=0,
                    )
                },
            )
        }
    )

    finding = _make_finding()
    plans = planner.process_finding(finding)

    assert len(plans) == 1
    assert plans[0].knob_id == "dlio.read_threads"
    assert plans[0].old_value == 0
    assert plans[0].new_value == 1
