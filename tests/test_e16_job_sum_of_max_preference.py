"""Sec 16 (judge batch 5): the 3-tier input source preference in domains_from_metrics --
job sum-of-max (cross-node) > per-node sum-of-max > crit fallback."""
from dfoptimizer.planner.clbo_planner import domains_from_metrics


def _coherent_metrics(**overrides):
    m = {
        "active_time": 10.0,
        "domain_coherent": 1.0,
        "domain_procs_active": 16.0,
        "window_wall_time_proc_crit": 10.0,
        "domain_input_time_proc_crit": 0.2,
        "domain_decode_time_proc_crit": 0.0,
        "domain_comm_time_proc_crit": 1.0,
        "domain_compute_time_proc_crit": 1.0,
        "residual_time_proc_crit": 0.0,
    }
    m.update(overrides)
    return m


def test_job_sum_of_max_wins_over_per_node_and_crit_when_all_three_present():
    m = _coherent_metrics(
        domain_input_job_step_max_sum_time=9.0,
        domain_input_step_max_sum_time=4.3, domain_input_step_nunique=9,
    )
    assert domains_from_metrics(m)["input"] == 9.0


def test_falls_back_to_per_node_when_job_value_absent():
    m = _coherent_metrics(domain_input_step_max_sum_time=4.3, domain_input_step_nunique=9)
    assert domains_from_metrics(m)["input"] == 4.3


def test_falls_back_to_crit_when_neither_sum_of_max_present():
    m = _coherent_metrics()
    assert domains_from_metrics(m)["input"] == 0.2


def test_non_finite_job_value_falls_back_to_per_node():
    import math
    m = _coherent_metrics(
        domain_input_job_step_max_sum_time=math.nan,
        domain_input_step_max_sum_time=4.3, domain_input_step_nunique=9,
    )
    assert domains_from_metrics(m)["input"] == 4.3


def test_job_value_of_zero_is_used_not_treated_as_absent():
    # A genuinely zero job-wide exposure (every rank's fetch wait was 0 that window) is real
    # evidence, not a missing value -- it must still win over a nonzero per-node fallback.
    m = _coherent_metrics(
        domain_input_job_step_max_sum_time=0.0,
        domain_input_step_max_sum_time=4.3, domain_input_step_nunique=9,
    )
    assert domains_from_metrics(m)["input"] == 0.0
