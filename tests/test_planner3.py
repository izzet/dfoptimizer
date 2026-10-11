"""Verification and first-window admission without exposure-specific samples."""
import numpy as np
import pytest
from dfoptimizer import step_series
from dfoptimizer.planner.cbandit_planner import CoordinateBanditPlanner
from dfoptimizer.types import KnobDef


def planner(monkeypatch, mode='per_step'):
    monkeypatch.setenv('LIVEFLOW_VERIFY', mode)
    p = CoordinateBanditPlanner()
    p.register_knobs({'halo3d.decode_threads': KnobDef(id='halo3d.decode_threads', default=1,
                     type=int, values=[1, 2, 4], cost_seconds=.001, target_function='halo3d.make_loader')})
    p._rate_mode = True
    p._physical_latest_cfg = {'decode_threads': 2}
    p._physical_changes = [('decode_threads', 2, 205)]
    p._move_from['decode_threads'] = (1, .02, 1, .1, 2)
    p.current_values['halo3d.decode_threads'] = 2
    return p


@pytest.mark.parametrize('shift,expected', [(-.02, 'confirm'), (.02, 'revert'), (0, 'collect')])
def test_wall_only_verification_uses_bootstrap(monkeypatch, shift, expected):
    p = planner(monkeypatch)
    pre = .1 + np.random.default_rng(0).normal(0, .002, 200)
    p._sv_ingest(1, {'step_series': step_series.encode(6, {'wall': pre})})
    p._sv_ingest(2, {'step_series': step_series.encode(206, {'wall': pre + shift})})
    assert p._sv_verdict('decode_threads', 1, 2, 205, 2, .02) == expected
    rec = p._sv_last['decode_threads']
    assert rec['target_series'] is None and rec['exposure_ci'] is None
    if expected == 'confirm':
        assert p._unconfirmed_move(2, 10000) is None
        assert 'decode_threads' not in p._move_from


def test_one_post_step_collects_instead_of_welch_rejection(monkeypatch):
    p = planner(monkeypatch)
    p._sv_ingest(1, {'step_series': step_series.encode(6, {'wall': [.1] * 200})})
    p._sv_ingest(2, {'step_series': step_series.encode(206, {'wall': [.05]})})
    assert p._unconfirmed_move(2, 10000) is None
    assert 'decode_threads' in p._move_from
    assert p._sv_last['decode_threads']['verdict'] == 'collect'


def test_window_fallback_does_not_test_one_candidate(monkeypatch):
    p = planner(monkeypatch)
    p._samples[('decode_threads', 1)] = [.1] * 3
    p._credit_observations = [(2, {'decode_threads': 2}, .05)]
    p._observation_first_steps[2] = 206
    monkeypatch.setattr(p, 'required_dwell', lambda *a, **k: (1, {}))
    assert p._unconfirmed_move(2, 10000) is None
    assert 'decode_threads' in p._move_from


def test_combined_input_is_preserved_and_labelled(monkeypatch):
    p = planner(monkeypatch)
    ss = step_series.decode(step_series.encode(41, {'wall': [.1] * 30, 'input': [.03] * 30}))
    assert p._sv_key('decode_threads', ss) == 'input'
    ss.series['decode'] = np.full(30, .01)
    assert p._sv_key('decode_threads', ss) == 'decode'


def test_first_clean_window_requires_precision_and_history(monkeypatch):
    p = planner(monkeypatch)
    p._physical_changes.clear()
    p._epoch_windows = 1
    p._epoch_dom['decode'] = [.01]
    p._win_metrics = {'planner_history_ready': 1}
    p._sv_ingest(3, {'step_series': step_series.encode(41, {'wall': [.1] * 100})})
    delta, source = p.predicted_effect('decode_threads', 'decode')
    assert delta == .01 and 'wall_precision95' in source
    p._win_metrics['planner_history_ready'] = 0
    assert p.predicted_effect('decode_threads', 'decode')[0] is None
    p._win_metrics['planner_history_ready'] = 1
    p._tl.clear()
    p._sv_ingest(3, {'step_series': step_series.encode(41, {'wall': np.random.default_rng(7).uniform(0, 1, 100)})})
    assert p.predicted_effect('decode_threads', 'decode')[0] is None


def test_window_mean_first_window_keeps_minimum(monkeypatch):
    p = planner(monkeypatch, 'window_mean')
    p._epoch_windows = 1
    p._epoch_dom['decode'] = [.01]
    assert p.predicted_effect('decode_threads', 'decode')[0] is None
