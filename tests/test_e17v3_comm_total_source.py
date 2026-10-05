"""Tests for DFOPTIMIZER_COMM_EXPOSURE_SOURCE (sec 17 v3, orchestrator decision 2026-09-27):
comm_total is a PROPOSAL bound (upper bound on what a bucket_cap move could be worth), selected only
when the flag is set to "total"; default ("exposed", unset) leaves sec 17 v2's behavior unchanged.
"""
import math

from dfoptimizer.planner.clbo_planner import domains_from_metrics


def _coherent_metrics(**overrides):
    m = {
        "active_time": 10.0, "domain_coherent": 1.0, "domain_procs_active": 16.0,
        "window_wall_time_proc_crit": 10.0, "domain_input_time_proc_crit": 0.2,
        "domain_decode_time_proc_crit": 0.0, "domain_comm_time_proc_crit": 1.0,
        "domain_compute_time_proc_crit": 1.0, "residual_time_proc_crit": 0.0,
    }
    m.update(overrides)
    return m


def test_default_source_ignores_comm_total_even_when_present(monkeypatch):
    monkeypatch.delenv("DFOPTIMIZER_COMM_EXPOSURE_SOURCE", raising=False)
    m = _coherent_metrics(
        domain_comm_pure_exposure_time_v2=1.8, domain_comm_exposure_step_nunique=9,
        domain_comm_total_median_time_v3=173.0, domain_comm_total_step_nunique=9,
    )
    assert domains_from_metrics(m)["comm"] == 1.8


def test_source_total_prefers_comm_total_over_exposed(monkeypatch):
    monkeypatch.setenv("DFOPTIMIZER_COMM_EXPOSURE_SOURCE", "total")
    m = _coherent_metrics(
        domain_comm_pure_exposure_time_v2=1.8, domain_comm_exposure_step_nunique=9,
        domain_comm_total_median_time_v3=173.0, domain_comm_total_step_nunique=9,
    )
    assert domains_from_metrics(m)["comm"] == 173.0


def test_source_total_falls_back_to_exposed_when_total_absent(monkeypatch):
    monkeypatch.setenv("DFOPTIMIZER_COMM_EXPOSURE_SOURCE", "total")
    m = _coherent_metrics(
        domain_comm_pure_exposure_time_v2=1.8, domain_comm_exposure_step_nunique=9,
    )
    assert domains_from_metrics(m)["comm"] == 1.8


def test_source_total_falls_back_to_crit_when_neither_present(monkeypatch):
    monkeypatch.setenv("DFOPTIMIZER_COMM_EXPOSURE_SOURCE", "total")
    m = _coherent_metrics()
    assert domains_from_metrics(m)["comm"] == 1.0


def test_source_total_with_single_step_window_falls_back(monkeypatch):
    monkeypatch.setenv("DFOPTIMIZER_COMM_EXPOSURE_SOURCE", "total")
    m = _coherent_metrics(
        domain_comm_pure_exposure_time_v2=1.8, domain_comm_exposure_step_nunique=9,
        domain_comm_total_median_time_v3=173.0, domain_comm_total_step_nunique=1,  # < 2, suppressed
    )
    assert domains_from_metrics(m)["comm"] == 1.8


def test_non_finite_comm_total_falls_back(monkeypatch):
    monkeypatch.setenv("DFOPTIMIZER_COMM_EXPOSURE_SOURCE", "total")
    m = _coherent_metrics(
        domain_comm_pure_exposure_time_v2=1.8, domain_comm_exposure_step_nunique=9,
        domain_comm_total_median_time_v3=math.nan, domain_comm_total_step_nunique=9,
    )
    assert domains_from_metrics(m)["comm"] == 1.8
