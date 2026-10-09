"""Power and verdict use the same unit: a block mean, including recurring stalls."""
import numpy as np
import pytest
from dfoptimizer.planner import step_verify as sv
from dfoptimizer.planner.cbandit_planner import CoordinateBanditPlanner
from dfoptimizer import step_series
from dfoptimizer.types import KnobDef


def test_periodic_loader_cost_reverts_and_blocks_regrowth(monkeypatch):
    monkeypatch.setenv('LIVEFLOW_VERIFY', 'per_step')
    monkeypatch.setenv('LIVEFLOW_PRESET', 'liveflow-v1')
    monkeypatch.setenv('LIVEFLOW_WORKLOAD', 'stormer')
    p = CoordinateBanditPlanner()
    p.register_knobs({'stormer.num_workers': KnobDef(id='stormer.num_workers', default=8,
                           type=int, values=[8,16,32], target_function='stormer.make_loader')})
    p._rate_mode = True
    p._control_steps = 47
    # Post body is faster but each epoch reset is more expensive: net move loses.
    pre = np.tile(np.r_[1.2, np.full(46,.2)], 5)
    post = np.tile(np.r_[4.2, np.full(46,.19)], 5)
    fetch = np.tile(np.r_[1., np.full(46,.02)], 5)
    p._sv_ingest(1, {'step_series':step_series.encode(1, {'wall':pre, 'storage':fetch})})
    p._sv_ingest(2, {'step_series':step_series.encode(236, {'wall':post, 'storage':fetch})})
    p._physical_latest_cfg = {'num_workers':16}
    p._physical_changes = [('num_workers',16,235)]
    assert p._sv_verdict('num_workers',8,16,235,2,.02) == 'revert'
    assert p._sv_last['num_workers']['step_ci']['lo'] > 0
    # Memory is attached to other knobs, so restoring W8 preserves the refutation.
    p._physical_latest_cfg = {'num_workers':8}
    assert p._sv_refuted('num_workers',8,16,.01).startswith('refuted:')


def test_recurring_epoch_first_steps_survive_warmup(monkeypatch):
    monkeypatch.setenv('LIVEFLOW_VERIFY', 'per_step')
    p = CoordinateBanditPlanner()
    wall = np.tile(np.r_[4., np.full(46, .2)], 3)
    p._sv_ingest(1, {'step_series': step_series.encode(1, {'wall': wall})})
    x, w = p._sv_segment('wall', 0, float('inf'))
    assert int(w.sum()) == 141 - 5
    assert (x == 4.).sum() == 2
    assert np.average(x, weights=w) == pytest.approx(wall[5:].mean())
