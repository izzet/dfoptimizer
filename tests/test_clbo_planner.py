"""Interface test for CLBOPlanner: the model-based (factored cross-layer surrogate) planner
driven through the live Planner interface (register_knobs + process_finding). Verifies it
registers knobs into a domain-factored knob space, consumes DiagnosisFindingMsg key_metrics
window by window, emits ActionPlans, and drives the deployed incumbent toward the optimum
without the incumbent ever regressing. No pipeline/Mofka needed. Run:
  PYTHONPATH=. <venv>/bin/python tests/test_clbo_planner.py
"""
from dfoptimizer.planner.clbo_planner import CLBOPlanner, domains_from_metrics
from dfoptimizer.types import DiagnosisFindingMsg, KnobDef


def _knobs():
    # Dict[str knob_id, KnobDef] exactly as the optimizer's _handle_registration passes it.
    kw = KnobDef(id="stormer.num_workers", default=1, type=int, range=(1, 8),
                 target_function="stormer.make_loader")
    kb = KnobDef(id="stormer.bucket_cap", default=25, type=int, range=(25, 400),
                 target_function="stormer.setup_ddp")
    return {kw.id: kw, kb.id: kb}


def _finding(window_index, key_metrics, ftype="fetch_pressure"):
    return DiagnosisFindingMsg(
        finding_type=ftype, motif="", severity="high", severity_score=1.0, confidence=0.9,
        prevalence=1.0, persistence=1, trend_direction="steady", contributing_facts=[],
        recommendation_bundle="", opportunity_tags=["reader_parallelism"], summary="",
        window_index=window_index, key_metrics=key_metrics,
    )


def _surface(cfg):
    """Synthetic cross-layer surface. Optimum: many workers (input relieved) + mid bucket
    (comm minimized). input hides as workers rise; comm minimized at bucket=175; compute flat."""
    inp = 240.0 / max(cfg["num_workers"], 1)          # 240 @ w=1 -> 30 @ w=8
    com = abs(cfg["bucket_cap"] - 175) / 3.0 + 20.0   # min 20 @ bucket=175
    comp = 100.0
    return {"fetch_iter_time_proc_max": inp, "comm_time_proc_max": com,
            "compute_time_proc_max": comp + com, "active_time": inp + comp + com}


def _run():
    passed = 0

    # --- test 1: registration builds a domain-factored knob space ---
    p = CLBOPlanner()
    p.register_knobs(_knobs())
    assert p._clbo is not None, "clbo not initialized"
    assert p._domain["num_workers"] == "input" and p._domain["bucket_cap"] == "comm", p._domain
    assert 8 in p._levels["num_workers"] and 175 in p._levels["bucket_cap"], p._levels
    print("PASS 1 register builds factored knob space (num_workers=input, bucket_cap=comm)")
    passed += 1

    # --- test 2: online loop converges, incumbent never regresses, plans are emitted ---
    default_short = dict(p._default_short)
    inc_hist, n_plans = [], 0
    for i in range(30):
        cfg_i = p._planned_cfg.get(i, default_short)         # config that RAN window i (latency 1)
        f = _finding(i, _surface(cfg_i))
        plans = p.process_finding(f)
        n_plans += len(plans)
        inc_hist.append(p._inc_wall)

    # incumbent wall is non-increasing (accept-guard safety)
    for a, b in zip(inc_hist, inc_hist[1:]):
        assert b <= a + 1e-9, f"incumbent regressed: {a} -> {b}"
    assert n_plans > 0, "no ActionPlans were ever emitted"
    print(f"PASS 2 online loop emitted {n_plans} plans; incumbent never regressed")
    passed += 1

    # --- test 3: converges near the optimum (150 = w=8 inp30 + comp100 + comm20) ---
    opt = _surface({"num_workers": 8, "bucket_cap": 175})["active_time"]
    assert p._inc_wall <= opt * 1.15, f"did not converge: incumbent {p._inc_wall:.1f} vs opt {opt:.1f}"
    assert p._inc_cfg["num_workers"] >= 4, f"did not relieve input: {p._inc_cfg}"
    print(f"PASS 3 converged near optimum (incumbent {p._inc_wall:.1f} <= opt {opt:.1f}*1.15, "
          f"cfg={p._inc_cfg})")
    passed += 1

    # --- test 4: emitted plans carry the right knob + target + apply_when ---
    p2 = CLBOPlanner()
    p2.register_knobs(_knobs())
    got_plan = None
    for i in range(6):
        cfg_i = p2._planned_cfg.get(i, dict(p2._default_short))
        pl = p2.process_finding(_finding(i, _surface(cfg_i)))
        if pl:
            got_plan = pl[0]
            break
    assert got_plan is not None, "no plan produced in first 6 windows"
    assert got_plan.knob_id in ("stormer.num_workers", "stormer.bucket_cap")
    assert got_plan.apply_when == "window_boundary"
    assert got_plan.target_function in ("stormer.make_loader", "stormer.setup_ddp")
    print(f"PASS 4 ActionPlan well-formed (knob={got_plan.knob_id}, "
          f"{got_plan.old_value}->{got_plan.new_value}, apply={got_plan.apply_when})")
    passed += 1

    print(f"\nALL {passed} CLBOPlanner interface tests PASSED")


def test_domains_record_absent_exposure_fallback(capsys):
    metrics = {
        "fetch_iter_time_proc_max": 0.012,
        "reader_decode_time_proc_max": 0.002,
        "compute_time_proc_max": 0.020,
    }
    domains = domains_from_metrics(metrics)

    assert domains["input"] == 0.010
    assert domains["decode"] == 0.002
    assert metrics["domain_exposure_present"] == 0.0
    assert metrics["domain_exposure_fallback"] == 1.0
    assert "exposure_fallback" in capsys.readouterr().out


def test_domains_distinguish_present_zero_exposure_from_absent(capsys):
    metrics = {
        "fetch_iter_time_proc_max": 0.012,
        "stall_storage_time_proc_max": 0.0,
        "stall_decode_time_proc_max": 0.0,
        "compute_time_proc_max": 0.020,
    }
    domains = domains_from_metrics(metrics)

    assert domains["input"] == 0.0
    assert domains["decode"] == 0.0
    assert metrics["domain_exposure_present"] == 1.0
    assert metrics["domain_exposure_fallback"] == 0.0
    assert "exposure_fallback" not in capsys.readouterr().out


if __name__ == "__main__":
    _run()
