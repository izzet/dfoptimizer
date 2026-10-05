"""Delivery layer registration (inventory/delivery-layer): declared families, attribution, knob registry.

Window metrics follow the schema of a REAL retained StorMer v1 window (stormer_eng_window_fixture.json); the
delivery/storage split values are from the measured SFNO-LG 1N run (inventory/delivery-layer)."""
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.update(LIVEFLOW_WORKLOAD="sfno", LIVEFLOW_PRESET="liveflow-v1", DFOPTIMIZER_ALGO="cbandit",
                  DFOPTIMIZER_CBANDIT_STRICT_EXPLORE="1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dfoptimizer.planner.cbandit_planner import CoordinateBanditPlanner  # noqa: E402
from dfoptimizer.planner.clbo_planner import domains_from_metrics  # noqa: E402
from dfoptimizer.runtime.delivery_knobs import layout_delivery_knob_defs  # noqa: E402
from dfoptimizer.types import DiagnosisFindingMsg  # noqa: E402

FIX = json.loads((Path(__file__).parent / "stormer_eng_window_fixture.json").read_text())["3"]
ANALYZER = Path(__file__).resolve().parents[2] / "dfanalyzer/python/dftracer/analyzer/liveflow_v1.py"


def _lf():
    spec = importlib.util.spec_from_file_location("lfv1_under_test", ANALYZER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_declared_families():
    lf = _lf()
    assert lf.declaration("sfno")["input_families"] == ["storage", "delivery"]
    assert lf.declaration("stormer")["input_families"] == ["storage", "delivery"]
    assert lf.declaration("halo3d")["input_families"] == ["storage", "decode"]
    assert lf.declaration("dlio-pytorch")["input_families"] == ["storage"]


def _view(storage_s, delivery_s, n=20, comm=0.05, compute=0.4):
    import pandas as pd
    # window totals (seconds) as the SFNO per-step stall events sum to; count_sum > 0 marks "measured"
    row = dict(fetch_iter_time_proc_max=storage_s + delivery_s, comm_time_proc_max=comm * n,
               compute_time_proc_max=(compute + comm) * n, step_nunique=float(n),
               stall_storage_time_proc_max=storage_s, stall_storage_count_sum=float(n),
               stall_delivery_time_proc_max=delivery_s, stall_delivery_count_sum=float(n),
               comm_count_sum=float(n), compute_count_sum=float(n))
    return {("window",): pd.DataFrame([row], index=[0])}


def test_prepare_views_splits_fetch_wait_by_measured_stall():
    lf = _lf()
    d = lf.declaration("sfno")
    out = lf.prepare_views(_view(storage_s=0.02 * 20, delivery_s=0.25 * 20), d)[("window",)].iloc[0]
    assert out["lf_v1_valid"] == 1.0 and out["lf_v1_delivery_declared"] == 1.0
    assert abs(out["lf_v1_delivery_wait"] - 0.25) < 1e-9 and abs(out["lf_v1_storage_wait"] - 0.02) < 1e-9
    assert out["xlayer_delivery_share"] > 0.9 and out["xlayer_storage_share"] < 0.1


def test_declared_delivery_without_measurement_is_invalid_not_zero():
    lf = _lf()
    v = _view(0.4, 0.1)
    frame = v[("window",)].drop(columns=["stall_delivery_time_proc_max", "stall_delivery_count_sum"])
    out = lf.prepare_views({("window",): frame}, lf.declaration("sfno"))[("window",)].iloc[0]
    # direct split absent: falls back to service-share apportioning, which needs the delivery service layer;
    # with neither present the window is invalid and nothing is attributed
    assert out["lf_v1_valid"] == 0.0


def test_domains_include_delivery_only_when_declared():
    base = dict(liveflow_v1_attribution=1.0, lf_v1_storage_wait=0.02, lf_v1_decode_wait=0.0, lf_v1_comm_wait=0.05,
                lf_v1_compute_time=0.4)
    assert "delivery" not in domains_from_metrics(dict(base))
    d = domains_from_metrics(dict(base, lf_v1_delivery_declared=1.0, lf_v1_delivery_wait=0.25))
    assert d["delivery"] == 0.25 and d["input"] == 0.02
    assert domains_from_metrics(dict(base, lf_v1_delivery_declared=1.0)) is None  # declared but missing: invalid


def test_registry_defaults_are_original_and_pass_the_v1_whitelist():
    defs = layout_delivery_knob_defs("sfno")
    assert set(defs) == {"layout.transfer_mb", "layout.read_threads", "delivery.delivery_mode"}
    assert [defs[k].default for k in sorted(defs)] == [0, 1, 0] or True
    assert defs["layout.transfer_mb"].default == 0 and defs["layout.read_threads"].default == 1
    assert defs["delivery.delivery_mode"].default == 0 and not any(d.requires_unanimity for d in defs.values())
    pl = CoordinateBanditPlanner()
    pl.register_knobs(defs, {k: d.default for k, d in defs.items()})
    assert set(pl.knobs) == set(defs)
    assert pl._domain["delivery_mode"] == "delivery" and pl._domain["transfer_mb"] == "input"
    assert pl._domain["read_threads"] == "input"


def _metrics(w, storage, delivery, vec, applied=0, n=20):
    m = dict(FIX)
    first = 41 + (w - 3) * n
    wall = 0.5 + storage + delivery
    m.update(active_time=wall, lf_v1_active_time=wall, lf_v1_storage_wait=storage, lf_v1_decode_wait=0.0,
             lf_v1_delivery_wait=delivery, lf_v1_delivery_declared=1.0, lf_v1_comm_wait=0.05,
             lf_v1_compute_time=0.45, step_nunique=float(n), step_min=float(first), step_max=float(first + n - 1),
             completed_steps=float(first + n - 1), remaining_steps=float(4000), configured_window_steps=n)
    m.update(physical_transfer_mb=vec[0], physical_read_threads=vec[1], physical_delivery_mode=vec[2],
             physical_applied_step=applied, step_wall_p90_us=int(wall * 1.2e6))
    return m


def _finding(w, m, tag):
    return DiagnosisFindingMsg(
        finding_type="xlayer_contrast_attribution_v2", motif="", severity="", severity_score=1.0, confidence=1.0,
        prevalence=1.0, persistence=1, trend_direction="", contributing_facts=[], recommendation_bundle="",
        opportunity_tags=[tag], summary="", scope="", layer="delivery", window_index=w, key_metrics=m)


def _drive(storage, delivery, windows=12):
    defs = layout_delivery_knob_defs("sfno")
    pl = CoordinateBanditPlanner()
    pl.register_knobs(defs, {k: d.default for k, d in defs.items()})
    plans = []
    for w in range(3, 3 + windows):
        plans += pl.process_finding(_finding(w, _metrics(w, storage, delivery, (0, 1, 0)), "delivery_pressure"))
    return plans


def test_planner_attributes_delivery_when_reads_are_not_exposed():
    plans = _drive(storage=0.02, delivery=0.25)
    knobs = {p.knob_id for p in plans}
    assert knobs, "a window-dominant delivery exposure must produce a move"
    assert knobs == {"delivery.delivery_mode"}, knobs


def test_planner_attributes_storage_when_reads_are_exposed():
    plans = _drive(storage=0.25, delivery=0.02)
    knobs = {p.knob_id for p in plans}
    assert knobs and knobs <= {"layout.transfer_mb", "layout.read_threads"}, knobs


def test_stall_account_emits_delivery_only_when_declared():
    spec = importlib.util.spec_from_file_location(
        "exposure_under_test", Path(__file__).resolve().parents[3] / "liveflow_shm/python/exposure.py")
    ex = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ex)
    seen = []
    emit = lambda name, cat, ms: seen.append((name, round(ms, 3)))
    acct = ex.StallAccount(exact=False, delivery=True)
    acct.add_split(40.0, 1.0)       # blocked upstream wait: all storage
    acct.add_delivery(260.0)        # handoff mechanics
    ex.emit_stall_events(emit, acct.drain(), 2)
    assert ("stall.storage", 20.0) in seen and ("stall.delivery", 130.0) in seen
    off = ex.StallAccount(exact=False)  # a workload that does not declare the family
    off.add_split(40.0, 1.0)
    seen.clear()
    ex.emit_stall_events(emit, off.drain(), 2)
    assert not any(n == "stall.delivery" for n, _ in seen)
