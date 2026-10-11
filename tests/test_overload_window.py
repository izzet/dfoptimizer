import pytest
from dfoptimizer.planner.cbandit_planner import CoordinateBanditPlanner
from test_coherent_window_safety import _finding, _knobs, _valid_coherent_metrics

@pytest.mark.parametrize('marker',['lf_v1_valid','lf_v1_trace_complete'])
def test_invalid_window_never_ingests_or_updates(marker):
    p=CoordinateBanditPlanner();p.register_knobs(_knobs())
    m=_valid_coherent_metrics();m[marker]=0.0
    p._sv_ingest=lambda *args: pytest.fail('invalid evidence ingested')
    assert p.process_finding(_finding(5,m)) == []
    assert p.process_window_completion(5) == []
    assert p._total_pulls == 0
