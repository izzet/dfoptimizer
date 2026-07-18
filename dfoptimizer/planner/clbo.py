"""LiveFlow controller PROTOTYPE — Cross-Layer-Directed Bayesian Optimization (CLBO).

DISTINCT from LiveIO's SafeIO (which is model-FREE statistical coordinate search).
CLBO is MODEL-BASED: a Gaussian-Process surrogate over the joint knob space picks
configs by Expected Improvement, but the acquisition is DIRECTED by the cross-layer
attribution — it boosts candidates that change the knob affecting the currently
DOMINANT domain (argmax of the per-domain fetch/comm/compute times; a COMPARISON,
not a threshold). This exploits LiveFlow's cross-layer observation to converge in
few evals and to FOLLOW the bottleneck when it moves input<->comm.

Robust (GP posterior + noise model), ML (GP), threshold-free (direction from
argmax; accept from posterior improvement, no magic gates), cross-layer-native.

Knobs (LiveFlow): read_threads/num_workers, prefetch (input domain); DDP
bucket_cap, all_to_all fusion, shard strategy (comm domain).
"""
from __future__ import annotations
import itertools
import math
import numpy as np


class _Norm:  # scipy-free standard normal cdf/pdf (vectorized)
    @staticmethod
    def cdf(x):
        from math import sqrt
        return 0.5 * (1.0 + np.vectorize(__import__("math").erf)(np.asarray(x, float) / sqrt(2.0)))

    @staticmethod
    def pdf(x):
        x = np.asarray(x, float)
        return np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)


norm = _Norm()


def _rbf(A, B, ls):
    A, B = np.atleast_2d(A), np.atleast_2d(B)
    d = (A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2 * A @ B.T
    return np.exp(-0.5 * np.maximum(d, 0) / ls**2)


class GP:
    """Minimal zero-mean-ish GP regressor (RBF kernel, homoscedastic noise)."""
    def __init__(self, ls=0.5, noise=1e-3):
        self.ls, self.noise = ls, noise

    def fit(self, X, y):
        self.X = np.atleast_2d(np.asarray(X, float))
        self.y = np.asarray(y, float)
        self.ymean = self.y.mean()
        K = _rbf(self.X, self.X, self.ls) + self.noise * np.eye(len(self.X))
        self.L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y - self.ymean))
        return self

    def predict(self, Xs):
        Xs = np.atleast_2d(np.asarray(Xs, float))
        Ks = _rbf(self.X, Xs, self.ls)
        mu = self.ymean + Ks.T @ self.alpha
        v = np.linalg.solve(self.L, Ks)
        var = np.maximum(1e-9, np.diag(_rbf(Xs, Xs, self.ls)) - (v**2).sum(0))
        return mu, np.sqrt(var)


def expected_improvement(mu, sigma, y_best):
    """EI for MINIMIZATION (lower step-time is better)."""
    imp = y_best - mu
    z = imp / sigma
    return imp * norm.cdf(z) + sigma * norm.pdf(z)


class CLBO:
    def __init__(self, knobs: dict, knob_domain: dict, ls=0.5, noise=1e-3, direct_boost=4.0):
        self.knobs = knobs                     # name -> sorted list of settings
        self.knob_domain = knob_domain         # name -> 'input' | 'comm' | 'compute'
        self.names = list(knobs)
        self.grid = [dict(zip(self.names, c)) for c in itertools.product(*[knobs[n] for n in self.names])]
        self.ls, self.noise, self.boost = ls, noise, direct_boost
        self._lo = {n: min(knobs[n]) for n in self.names}
        self._sp = {n: (max(knobs[n]) - min(knobs[n])) or 1.0 for n in self.names}

    def _enc(self, cfg):
        return [(cfg[n] - self._lo[n]) / self._sp[n] for n in self.names]

    def suggest(self, X_cfgs, y_steptime, domain_times_best):
        """Next config to try. domain_times_best = {domain: time} at the current best."""
        gp = GP(self.ls, self.noise).fit([self._enc(c) for c in X_cfgs], y_steptime)
        cand = [c for c in self.grid if c not in X_cfgs]
        if not cand:
            return None
        mu, sig = gp.predict([self._enc(c) for c in cand])
        ei = expected_improvement(mu, sig, min(y_steptime))
        # cross-layer DIRECTION: boost candidates that move the DOMINANT domain's knob
        dom = max(domain_times_best, key=domain_times_best.get)
        best = X_cfgs[int(np.argmin(y_steptime))]
        for i, c in enumerate(cand):
            if any(self.knob_domain.get(n) == dom and c[n] != best[n] for n in self.names):
                ei[i] *= self.boost
        return cand[int(np.argmax(ei))]

    def optimize(self, measure, n_init=3, n_iter=10, seed=0):
        """measure(cfg) -> (step_time, {domain: time}). Returns (best_cfg, best_time, trace)."""
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(self.grid), size=min(n_init, len(self.grid)), replace=False)
        X = [self.grid[i] for i in idx]
        res = [measure(c) for c in X]
        y = [r[0] for r in res]
        dom_best = res[int(np.argmin(y))][1]
        trace = [(dict(c), t) for c, t in zip(X, y)]
        for _ in range(n_iter):
            nxt = self.suggest(X, y, dom_best)
            if nxt is None:
                break
            st, dt = measure(nxt)
            X.append(nxt); y.append(st); trace.append((dict(nxt), st))
            if st <= min(y):
                dom_best = dt
        b = int(np.argmin(y))
        return X[b], y[b], trace


# =============================================================================
# HARDENED controller (proven algorithm + safety). See CLBO-vs-SafeIO.md sec 6-7.
#   - FACTORED cross-layer surrogate = the non-subsumable use of the decomposition
#     (beats a layer-blind joint GP, gap grows with knob count; ablation.py proves it).
#   - Statistical accept-guard = SafeIO-style "whether to keep" (generic statistics).
#   - Clean synchronized-rank probe = trustworthy decomposition WITHOUT any threshold.
# Constants: GP hypers (learnable) + alpha (universal) + effect_floor (utility). No
# domain-attribution cutoffs.
# =============================================================================
def welch_p(a, b):
    """Two-sided Welch t-test p-value, scipy-free (normal tail approx)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return 1.0
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) or 1e-12
    t = (a.mean() - b.mean()) / se
    return float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t) / math.sqrt(2.0)))))


def accept_decision(base, cand, alpha=0.05, effect_floor=0.02, guard_base=None, guard_cand=None):
    """Keep a candidate as the new incumbent ONLY if its per-step wall samples are
    significantly (Welch p<alpha) AND meaningfully (relative gain>=effect_floor) faster than
    the incumbent's, AND a guarded metric (e.g. read p99 / loss-delta) is not significantly
    worse. Returns 'accept' or 'reject'. This is the no-harm safety layer on top of the
    surrogate's pick; the surrogate still LEARNS from every measurement either way."""
    base, cand = np.asarray(base, float), np.asarray(cand, float)
    gain = (base.mean() - cand.mean()) / base.mean() if base.mean() else 0.0
    significant = welch_p(base, cand) < alpha
    guard_bad = False
    if guard_base is not None and guard_cand is not None and len(np.atleast_1d(guard_base)) > 1:
        gb, gc = np.asarray(guard_base, float), np.asarray(guard_cand, float)
        guard_bad = gc.mean() > gb.mean() and welch_p(gb, gc) < alpha
    return "accept" if (significant and gain >= effect_floor and not guard_bad) else "reject"


class FactoredSurrogate:
    """Cross-layer FACTORED surrogate: one low-dim GP per domain on that domain's OWN knobs,
    from the per-domain observations, composed as step = compute + sum_d relu(dom_d - overlap).
    A layer-blind joint GP must learn the full K-dim surface; this learns D small surfaces and
    composes them -> the advantage grows with knob count (ablation.py)."""
    def __init__(self, knob_domain, knob_levels, overlap=0.0, ls=0.6, noise=1e-2):
        self.kd, self.kl, self.ov, self.ls, self.noise = knob_domain, knob_levels, overlap, ls, noise

    def _enc(self, sub):
        lo = {k: min(self.kl[k]) for k in sub}
        sp = {k: (max(self.kl[k]) - min(self.kl[k])) or 1.0 for k in sub}
        return lambda c: [(c[k] - lo[k]) / sp[k] for k in sub]

    def fit(self, cfgs, domain_obs):
        knobs = list(cfgs[0].keys())
        self.subs = {}      # domain -> (GP, encoder) for domains with knobs
        self.const = {}     # domain -> mean time for domains without knobs
        for dom in ("input", "comm", "compute"):
            sub = [k for k in knobs if self.kd.get(k) == dom]
            y = [d.get(dom, 0.0) for d in domain_obs]
            if sub:
                e = self._enc(sub)
                self.subs[dom] = (GP(self.ls, self.noise).fit([e(c) for c in cfgs], y), e)
            else:
                self.const[dom] = float(np.mean(y)) if y else 0.0
        return self

    def predict(self, cfgs):
        mu = np.zeros(len(cfgs))
        var = np.zeros(len(cfgs))
        for dom in ("input", "comm", "compute"):
            if dom in self.subs:
                gp, e = self.subs[dom]
                m, s = gp.predict([e(c) for c in cfgs])
                var = var + s ** 2
            else:
                m = np.full(len(cfgs), self.const.get(dom, 0.0))
            # input/comm hide under the compute floor up to overlap; compute always adds
            mu = mu + (m if dom == "compute" else np.maximum(0.0, m - self.ov))
        return mu, np.sqrt(var)


class StructuredCLBO:
    """Hardened LiveFlow controller. suggest() picks by EI on the FACTORED surrogate;
    optimize() moves the deployed INCUMBENT only when the statistical accept-guard passes,
    while the surrogate learns from EVERY measurement. measure(cfg) must return a dict:
      {"wall_mean": float, "wall_samples": [per-step...], "domains": {input,comm,compute},
       "guard_samples": [...] optional}
    'domains' should come from a synchronized-rank (clean) probe so the comm term is
    straggler-free -> no fetch_frac threshold anywhere."""
    def __init__(self, knobs, knob_domain, knob_levels, overlap=0.0, alpha=0.05, effect_floor=0.02):
        self.knobs = list(knobs)
        self.kd, self.kl, self.ov = knob_domain, knob_levels, overlap
        self.alpha, self.effect_floor = alpha, effect_floor
        self.grid = [dict(zip(self.knobs, c)) for c in itertools.product(*[knob_levels[k] for k in self.knobs])]

    def suggest(self, cfgs, domain_obs, walls):
        sur = FactoredSurrogate(self.kd, self.kl, self.ov).fit(cfgs, domain_obs)
        cand = [c for c in self.grid if c not in cfgs]
        if not cand:
            return None
        mu, sig = sur.predict(cand)
        return cand[int(np.argmax(expected_improvement(mu, sig, min(walls))))]

    def optimize(self, measure, n_init=4, n_iter=16, seed=0):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(self.grid), size=min(n_init, len(self.grid)), replace=False)
        cfgs = [self.grid[i] for i in idx]
        obs = [measure(c) for c in cfgs]
        walls = [o["wall_mean"] for o in obs]
        dom = [o["domains"] for o in obs]
        inc = int(np.argmin(walls))
        trace = [("init", dict(c), o["wall_mean"]) for c, o in zip(cfgs, obs)]
        for _ in range(n_iter):
            nxt = self.suggest(cfgs, dom, walls)
            if nxt is None:
                break
            o = measure(nxt)
            d = accept_decision(obs[inc]["wall_samples"], o["wall_samples"], self.alpha, self.effect_floor,
                                obs[inc].get("guard_samples"), o.get("guard_samples"))
            cfgs.append(nxt); obs.append(o); walls.append(o["wall_mean"]); dom.append(o["domains"])
            if d == "accept" and o["wall_mean"] < walls[inc]:
                inc = len(cfgs) - 1
            trace.append((d, dict(nxt), o["wall_mean"]))
        return cfgs[inc], walls[inc], trace
