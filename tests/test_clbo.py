"""Unit tests for CLBO (Cross-Layer-Directed Bayesian Optimization) — first
principles. Synthetic cross-layer objective: read_threads relieves the INPUT
(fetch) domain, bucket relieves the COMM domain; they overlap with compute. The
joint optimum needs BOTH tuned. Run: <venv>/bin/python test_clbo.py
"""
import numpy as np
from dfoptimizer.planner.clbo import GP, expected_improvement, CLBO

KNOBS = {"read_threads": [1, 2, 4, 8], "bucket": [25, 50, 100, 200]}
KNOB_DOMAIN = {"read_threads": "input", "bucket": "comm"}


def measure_factory(noise=0.0, seed=0):
    rng = np.random.default_rng(seed)

    def measure(cfg):
        rt, bk = cfg["read_threads"], cfg["bucket"]
        fetch = 120.0 / rt                              # input relieved by read parallelism
        comm = 20.0 + 80.0 * abs(bk - 100) / 100.0      # comm min at bucket=100
        compute = 30.0
        step = max(fetch, compute) + comm               # fetch/compute overlap; comm serial
        if noise:
            step += rng.normal(0, noise)
        return step, {"input": fetch, "comm": comm, "compute": compute}
    return measure
# true optimum: read_threads=8 (fetch=15), bucket=100 (comm=20) -> step = max(15,30)+20 = 50


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    assert cond, name


def test_gp_regresses():
    X = np.linspace(0, 1, 8).reshape(-1, 1)
    y = np.sin(3 * X).ravel()
    gp = GP(ls=0.25, noise=1e-4).fit(X, y)
    mu, sig = gp.predict(np.array([[0.5]]))
    check("1 GP regresses a smooth function", abs(mu[0] - np.sin(1.5)) < 0.1 and sig[0] < 0.5)


def test_ei_prefers_improvement():
    # candidate A likely-better (low mu), B worse (high mu); EI should prefer A
    ei = expected_improvement(np.array([10.0, 40.0]), np.array([3.0, 3.0]), y_best=20.0)
    check("2 EI prefers the improving candidate", ei[0] > ei[1])


def test_clbo_finds_joint_optimum():
    # optimum step=50 requires bucket=100 (comm min) AND read_threads>=4 (fetch<=compute);
    # (4,100) and (8,100) are both optimal.
    ctrl = CLBO(KNOBS, KNOB_DOMAIN)
    best, val, trace = ctrl.optimize(measure_factory(), n_init=3, n_iter=10, seed=1)
    check("3 CLBO finds joint optimum (bucket=100 & read_threads>=4, step=50)",
          abs(val - 50) < 1e-6 and best["bucket"] == 100 and best["read_threads"] >= 4)


def test_cross_layer_direction_input():
    # state: INPUT dominates -> next suggestion must change the INPUT knob (read_threads)
    ctrl = CLBO(KNOBS, KNOB_DOMAIN)
    X = [{"read_threads": 1, "bucket": 100}, {"read_threads": 1, "bucket": 25}]
    y = [70.0, 130.0]                      # bucket=100 better; best has fetch dominating
    dom = {"input": 120.0, "comm": 20.0, "compute": 30.0}   # INPUT dominant
    nxt = ctrl.suggest(X, y, dom)
    check("4 direction: input-bound -> suggests changing read_threads",
          nxt["read_threads"] != 1)


def test_cross_layer_direction_comm():
    # state: COMM dominates -> next suggestion must change the COMM knob (bucket)
    ctrl = CLBO(KNOBS, KNOB_DOMAIN)
    X = [{"read_threads": 8, "bucket": 25}, {"read_threads": 4, "bucket": 25}]
    y = [95.0, 100.0]
    dom = {"input": 15.0, "comm": 80.0, "compute": 30.0}    # COMM dominant
    nxt = ctrl.suggest(X, y, dom)
    check("5 direction: comm-bound -> suggests changing bucket", nxt["bucket"] != 25)


def test_beats_random_search():
    # CLBO should reach the optimum in fewer evals than random search on average
    meas = measure_factory()
    clbo_evals, rand_evals = [], []
    for s in range(8):
        _, _, tr = CLBO(KNOBS, KNOB_DOMAIN).optimize(meas, n_init=3, n_iter=13, seed=s)
        clbo_evals.append(next((i + 1 for i, (_, v) in enumerate(tr) if abs(v - 50) < 1e-6), 16))
        rng = np.random.default_rng(100 + s)
        grid = [{"read_threads": r, "bucket": b} for r in KNOBS["read_threads"] for b in KNOBS["bucket"]]
        order = rng.permutation(len(grid))
        rand_evals.append(next((i + 1 for i, gi in enumerate(order) if abs(meas(grid[gi])[0] - 50) < 1e-6), 16))
    check(f"6 CLBO reaches optimum faster than random (CLBO {np.mean(clbo_evals):.1f} vs random {np.mean(rand_evals):.1f} evals)",
          np.mean(clbo_evals) < np.mean(rand_evals))


def test_robust_to_noise():
    # with measurement noise, CLBO still lands at or near the optimum
    best, val, _ = CLBO(KNOBS, KNOB_DOMAIN).optimize(measure_factory(noise=1.5, seed=3),
                                                     n_init=4, n_iter=12, seed=3)
    check("7 CLBO robust to measurement noise (near-optimal config)",
          best["read_threads"] >= 4 and best["bucket"] in (50, 100))


if __name__ == "__main__":
    tests = [test_gp_regresses, test_ei_prefers_improvement, test_clbo_finds_joint_optimum,
             test_cross_layer_direction_input, test_cross_layer_direction_comm,
             test_beats_random_search, test_robust_to_noise]
    for t in tests:
        t()
    print(f"\nALL {len(tests)} CLBO first-principles tests PASSED")
