"""Regression guard for the sec 17 predicted_effect call site (orchestrator fix, 2026-09-27):
`window_exposure` must be passed ONLY when DFOPTIMIZER_CBANDIT_COMM_EFFECT_CAP_WINDOW is on, so a
caller stubbing `predicted_effect` with the pre-sec-17 signature (short, domain, reloc_gain=None)
keeps working when the flag is off (the default, unchanged case) -- this is exactly what
liveflow_shm/tests/test_cbandit_no_predicted_effect.py's `missing_effect` stub does, outside this
sandbox's own test scope, which is why the regression was caught only when integrator built v4.5.
"""
from dfoptimizer.planner.cbandit_planner import CoordinateBanditPlanner
from dfoptimizer.types import DiagnosisFindingMsg, KnobDef


def _knobs():
    return {
        "stormer.num_workers": KnobDef(
            id="stormer.num_workers", default=8, type=int, range=(0, 48),
            target_function="make_loader"),
    }


def _finding(window_index, key_metrics):
    return DiagnosisFindingMsg(
        finding_type="fetch_pressure", motif="", severity="high", severity_score=1.0,
        confidence=1.0, prevalence=1.0, persistence=4, trend_direction="stable",
        contributing_facts=[], recommendation_bundle="", opportunity_tags=["reader_parallelism"],
        summary="", window_index=window_index, key_metrics=key_metrics,
    )


def _coherent_metrics(**overrides):
    m = {
        "active_time": 10.0, "domain_coherent": 1.0, "domain_procs_active": 16.0,
        "window_wall_time_proc_crit": 10.0, "domain_input_time_proc_crit": 3.0,
        "domain_decode_time_proc_crit": 0.0, "domain_comm_time_proc_crit": 1.0,
        "domain_compute_time_proc_crit": 1.0, "residual_time_proc_crit": 0.0,
    }
    m.update(overrides)
    return m


def test_flag_off_calls_predicted_effect_with_the_pre_sec17_signature(monkeypatch):
    """The exact regression: a stub with NO window_exposure parameter must not raise when the flag
    is off (default) -- a keyword arg present with value None still breaks such a stub, so the fix
    must omit the kwarg entirely in this path, not merely default it to None."""
    p = CoordinateBanditPlanner()
    p.register_knobs(_knobs())
    assert p._comm_effect_cap_window is False   # default off

    def old_signature_stub(short, domain, reloc_gain=None):
        return None, "stubbed_no_effect"

    monkeypatch.setattr(p, "predicted_effect", old_signature_stub)
    m = _coherent_metrics(domain_input_time_proc_crit=5.0, domain_compute_time_proc_crit=0.5,
                          domain_comm_time_proc_crit=0.2)
    p.process_finding(_finding(0, m))
    p.process_finding(_finding(1, m))   # must not raise TypeError


def test_flag_on_passes_window_exposure_to_predicted_effect(monkeypatch):
    p = CoordinateBanditPlanner()
    p.register_knobs(_knobs())
    monkeypatch.setenv("DFOPTIMIZER_CBANDIT_COMM_EFFECT_CAP_WINDOW", "1")
    p2 = CoordinateBanditPlanner()
    p2.register_knobs(_knobs())
    assert p2._comm_effect_cap_window is True

    seen = {}

    def spy(short, domain, reloc_gain=None, window_exposure=None):
        seen["window_exposure"] = window_exposure
        seen["called"] = True
        return None, "stubbed_no_effect"

    monkeypatch.setattr(p2, "predicted_effect", spy)
    m = _coherent_metrics(domain_input_time_proc_crit=5.0, domain_compute_time_proc_crit=0.5,
                          domain_comm_time_proc_crit=0.2)
    p2.process_finding(_finding(0, m))
    p2.process_finding(_finding(1, m))
    assert seen.get("called"), "predicted_effect was never reached"
    assert "window_exposure" in seen
