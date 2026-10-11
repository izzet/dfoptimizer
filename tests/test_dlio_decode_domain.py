"""Only the explicitly enabled DLIO HDF5 worker pool owns native decode."""
from dfoptimizer.planner.clbo_planner import _domain_of


def test_dlio_decode_pool_mapping_is_workload_scoped(monkeypatch):
    monkeypatch.delenv('LIVEFLOW_WORKLOAD', raising=False)
    monkeypatch.delenv('LIVEFLOW_DLIO_HDF5_DECODE', raising=False)
    assert _domain_of('read_threads') == 'input'
    monkeypatch.setenv('LIVEFLOW_DLIO_HDF5_DECODE', '1')
    monkeypatch.setenv('LIVEFLOW_WORKLOAD', 'stormer')
    assert _domain_of('read_threads') == 'input'
    monkeypatch.setenv('LIVEFLOW_WORKLOAD', 'dlio-pytorch')
    assert _domain_of('read_threads') == 'decode'
    assert _domain_of('prefetch_size') == 'input'
    monkeypatch.setenv('LIVEFLOW_DLIO_HDF5_DECODE', '0')
    assert _domain_of('read_threads') == 'input'
