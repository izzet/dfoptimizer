"""Same native-window finalization: identity, exactly-once and empty/EOS safety."""
from dfoptimizer.planner.cbandit_planner import CoordinateBanditPlanner
from test_coherent_window_safety import _finding, _knobs, _valid_coherent_metrics


def planner():
    p = CoordinateBanditPlanner(); p.register_knobs(_knobs()); return p


def test_same_window_records_once_and_preserves_clock():
    p = planner()
    assert p.process_finding(_finding(5, _valid_coherent_metrics())) == []
    assert p._total_pulls == 0
    p.process_window_completion(5)
    assert p._total_pulls == 1
    assert p._cur_window == 5 and p._pending_finding is None
    assert p.process_window_completion(5) == []
    assert p.process_finding(_finding(5, _valid_coherent_metrics())) == []
    assert p.process_finding(_finding(4, _valid_coherent_metrics())) == []
    assert p._total_pulls == 1
    p.process_finding(_finding(6, _valid_coherent_metrics()))
    p.process_window_completion(6)
    assert p._total_pulls == 2


def test_empty_terminal_and_wrong_marker_never_invent_evidence():
    p = planner()
    assert p.process_window_completion(1) == []
    p.process_finding(_finding(5, _valid_coherent_metrics()))
    assert p.process_window_completion(4) == []
    assert p.process_window_completion(6) == []
    assert p.process_window_completion(8) == []
    assert p.process_window_completion(5,terminal=True) == []
    assert p._total_pulls == 0
    p.process_window_completion(5)
    assert p._total_pulls == 1
    assert p.process_window_completion(6) == []
    assert p._total_pulls == 1


def test_successor_fallback_and_marker_do_not_double_record():
    p = planner()
    p.process_finding(_finding(5, _valid_coherent_metrics()))
    p.process_finding(_finding(6, _valid_coherent_metrics()))
    assert p._total_pulls == 1
    assert p.process_window_completion(5) == []
    p.process_window_completion(6)
    assert p._total_pulls == 2


def test_late_finding_cannot_finalize_or_replace_a_pending_newer_window():
    p = planner()
    p.process_finding(_finding(5, _valid_coherent_metrics()))
    assert p.process_finding(_finding(4, _valid_coherent_metrics())) == []
    assert p._cur_window == 5 and p._total_pulls == 0
    p.process_window_completion(5)
    assert p._total_pulls == 1


def test_all_same_window_findings_accumulate_before_marker(monkeypatch):
    p = planner(); seen = []
    p.process_finding(_finding(5, {'active_time':10.,'first_unique':1.}))
    p.process_finding(_finding(5, {'second_unique':2.}))
    original = p._finalize_window
    def observe(w):
        seen.append((w,dict(p._win_metrics))); return original(w)
    monkeypatch.setattr(p,'_finalize_window',observe)
    p.process_window_completion(5)
    assert len(seen) == 1 and seen[0][0] == 5
    assert seen[0][1]['first_unique'] == 1. and seen[0][1]['second_unique'] == 2.
