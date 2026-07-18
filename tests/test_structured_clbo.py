"""Unit tests for the HARDENED LiveFlow controller: the factored cross-layer surrogate
(the non-subsumable win), the statistical accept-guard (safety), and the clean-probe
requirement. Reproduces the ablation verdict as asserts. Run:
  <venv>/bin/python test_structured_clbo.py
"""
import numpy as np
from dfoptimizer.planner.clbo import GP, FactoredSurrogate, StructuredCLBO, accept_decision
from ablation import (Surface, DIM_KNOBS, KNOB_LEVELS, KNOB_DOMAIN, STRAGGLER_K,
                      optimization_ablation)


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    assert cond, name


def _measure_factory(surf, rng, clean=True, guard_from_input=False):
    """measure(cfg) -> the dict StructuredCLBO expects. 'domains' clean (synchronized probe)
    or contaminated (comm carries straggler inflation). Optional guard = read-pressure proxy."""
    def measure(cfg):
        d = surf.domains_true(cfg)
        samples = surf.wall_true(cfg) + rng.normal(0, 10, size=12)
        dom = dict(d)
        if not clean:
            dom["comm"] = d["comm"] + STRAGGLER_K * d["input"]
        out = {"wall_mean": float(np.mean(samples)), "wall_samples": samples, "domains": dom}
        if guard_from_input:
            out["guard_samples"] = d["input"] + rng.normal(0, 1, size=12)
        return out
    return measure


def test_factored_beats_joint_prediction():
    surf = Surface(DIM_KNOBS[4]); rng = np.random.default_rng(0); grid = surf.grid
    tr_idx = rng.choice(len(grid), 20, replace=False)
    te_idx = rng.choice([i for i in range(len(grid)) if i not in set(tr_idx)], 40, replace=False)
    tr, te = [grid[i] for i in tr_idx], [grid[i] for i in te_idx]
    dom_tr = [surf.domains_true(c) for c in tr]
    walls_tr = [surf.wall_true(c) + rng.normal(0, 10) for c in tr]
    fac = FactoredSurrogate(KNOB_DOMAIN, KNOB_LEVELS, overlap=0.0).fit(tr, dom_tr)
    mu_f, _ = fac.predict(te)
    knobs = surf.knobs
    lo = {k: min(KNOB_LEVELS[k]) for k in knobs}; sp = {k: (max(KNOB_LEVELS[k]) - min(KNOB_LEVELS[k])) or 1.0 for k in knobs}
    enc = lambda c: [(c[k] - lo[k]) / sp[k] for k in knobs]
    gj = GP(0.5, 1e-2).fit([enc(c) for c in tr], walls_tr)
    mu_j, _ = gj.predict([enc(c) for c in te])
    truth = np.array([surf.wall_true(c) for c in te])
    mae_f, mae_j = np.mean(np.abs(mu_f - truth)), np.mean(np.abs(mu_j - truth))
    check(f"1 factored surrogate predicts step better than joint GP (MAE {mae_f:.1f} < {mae_j:.1f})",
          mae_f < mae_j)


def test_accept_guard():
    rng = np.random.default_rng(1)
    base = rng.normal(300, 8, 50); better = rng.normal(250, 8, 50); noise = rng.normal(299, 8, 50)
    check("2 accept-guard ACCEPTS a real improvement", accept_decision(base, better) == "accept")
    check("3 accept-guard REJECTS a noise-only change", accept_decision(base, noise) == "reject")
    gb, gc = rng.normal(5, 1, 50), rng.normal(30, 3, 50)
    check("4 accept-guard REJECTS a faster-but-HARMFUL change (guard trips)",
          accept_decision(base, better, guard_base=gb, guard_cand=gc) == "reject")


def test_structured_optimize_converges_and_safe():
    # Separate two things the hardened controller must do: the OPTIMIZER must find a good
    # config, and the deployed INCUMBENT must be safe (the guard makes it lag, on purpose).
    surf = Surface(DIM_KNOBS[2]); opt = surf.opt()
    ctrl = StructuredCLBO(surf.knobs, KNOB_DOMAIN, KNOB_LEVELS, overlap=0.0)
    _, _, trace = ctrl.optimize(_measure_factory(surf, np.random.default_rng(2)), n_init=4, n_iter=16, seed=2)
    best_eval = min(t[2] for t in trace)
    check(f"5a optimizer FINDS a near-optimal config (best_eval {best_eval:.0f} <= opt*1.10 {opt * 1.10:.0f})",
          best_eval <= opt * 1.10)
    init_inc = min(t[2] for t in trace if t[0] == "init")
    inc_seq = [init_inc] + [t[2] for t in trace if t[0] == "accept"]
    check("5b deployed INCUMBENT never regresses (accept-guard safety)",
          all(inc_seq[i + 1] <= inc_seq[i] + 1e-6 for i in range(len(inc_seq) - 1)))


def test_clean_probe_beats_contaminated():
    surf = Surface(DIM_KNOBS[5]); opt = surf.opt()
    def run(clean):
        vals = []
        for s in range(8):
            ctrl = StructuredCLBO(surf.knobs, KNOB_DOMAIN, KNOB_LEVELS, overlap=0.0)
            _, _, trace = ctrl.optimize(_measure_factory(surf, np.random.default_rng(100 + s), clean=clean),
                                        n_init=4, n_iter=10, seed=s)
            vals.append(min(t[2] for t in trace) - opt)   # best config the surrogate found
        return float(np.mean(vals))
    clean_reg, contam_reg = run(True), run(False)
    check(f"6 clean synchronized-rank probe beats contaminated comm (regret {clean_reg:.0f} < {contam_reg:.0f})",
          clean_reg < contam_reg)


def test_verdict_gap_grows_with_dim():
    reg = optimization_ablation(overlap=0.0, seeds=20)
    red = lambda dim: (reg[dim]["plain"][4] - reg[dim]["structured_clean"][4]) / reg[dim]["plain"][4] * 100
    g2, g5 = red(2), red(5)
    check(f"7a FACTORED cross-layer surrogate beats layer-blind BO at dim5 ({g5:+.0f}%)", g5 > 10)
    check(f"7b the cross-layer advantage GROWS with knob count (dim5 {g5:+.0f}% > dim2 {g2:+.0f}%)", g5 > g2)


if __name__ == "__main__":
    tests = [test_factored_beats_joint_prediction, test_accept_guard,
             test_structured_optimize_converges_and_safe, test_clean_probe_beats_contaminated,
             test_verdict_gap_grows_with_dim]
    for t in tests:
        t()
    print(f"\nALL structured-CLBO tests PASSED ({len(tests)} groups)")
