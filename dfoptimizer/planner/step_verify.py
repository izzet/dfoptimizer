"""Per-step targeted-exposure verification of a LiveFlow move, and the side-effect ledger.

A move targets ONE family (delivery, comm, input, decode).  It is verified on that family's per-step
exposure and on the step time, comparing the steps AFTER the apply boundary with the steps BEFORE it
(the same configuration segment that preceded the apply).  Both comparisons carry a 95% interval on the
change in the mean, from a moving-block bootstrap over steps.

THE BLOCK RULE (stated, one rule, no per-workload tuning).  For a segment of n steps the block length is

    L = min( 2k - 1,  ceil(sqrt(n)) ),   k = first lag >= 1 at which the sample autocorrelation drops below 1/e

i.e. L is the width of the central run of lags whose autocorrelation is still at least 1/e (white
noise: k = 1, L = 1, the ordinary bootstrap; a constant series: L = 1; the sqrt(n) cap applies when the
autocorrelation never decays below 1/e within sqrt(n) lags, a strongly persistent or trending segment).
One more case: when, after that first drop, the autocorrelation returns to at least 1/e at a local peak P with
at least three periods in the segment (P <= n/3: an epoch-start stall or a validation gap that recurs), L = P, so
each block holds a whole period and every resample holds as many such events as the data do.
Simulated coverage of the 95% interval (AR(1), 300 steps per segment): 0.95 (phi 0), 0.90 (0.5),
0.88 (0.7), 0.85 (0.9): the e-folding lag alone (L = k) covered 0.86 / 0.84 / 0.78.  A resample draws ceil(n / L)
blocks of L consecutive steps uniformly with replacement (overlapping blocks) and takes the mean;
before and after segments are resampled independently, 2000 times (a Monte-Carlo accuracy, not a
statistical choice), with a fixed seed so a replay reproduces.  The interval is the 2.5th and 97.5th
percentile of mean(after*) - mean(before*).  95% is the conventional level; nothing here is tuned.

VERDICT.  On the STEP TIME: the interval excludes zero on the improving side (a decrease) -> confirm;
it excludes zero on the worsening side -> revert; otherwise keep collecting, bounded by the horizon.
The targeted exposure's interval is reported with every verdict as the mechanism (it is not a second
vote): the claim "this move helped because the family's exposure fell" needs both.

SIDE EFFECTS.  After an apply, the post-apply steps also price what the move cost beyond its apply
latency: the step disruption around the apply, the excess step time in windows where the kernel
reclaimed memory (vmstat allocstall / direct scan / compaction counters), and the node's anonymous
memory growth per unit of the knob.  `SideCost` is the planner's per-knob ledger; the next move of the
same knob is priced from it (gain x remaining steps must exceed apply latency + side effects).
"""
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

CONFIDENCE = 0.95
RESAMPLES = 2000
SEED = 0
_E_INV = math.exp(-1.0)


# ----------------------------------------------------------------------------------------------
# block length
# ----------------------------------------------------------------------------------------------
def acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Sample autocorrelation at lags 0..max_lag (biased estimator), by FFT."""
    n = len(x)
    x = np.asarray(x, dtype=float) - float(np.mean(x))
    var = float(np.dot(x, x))
    if n < 2 or var <= 0.0:
        return np.zeros(max_lag + 1)
    size = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x, size)
    full = np.fft.irfft(f * np.conj(f), size)[: max_lag + 1]
    return full / var


def block_length(x: np.ndarray) -> Tuple[int, str]:
    """(L, rule) for the moving-block bootstrap of one segment; see the module docstring."""
    n = len(x)
    if n < 2:
        return 1, "n<2"
    cap = int(math.ceil(math.sqrt(n)))
    r = acf(x, min(max(cap, n // 3) + 1, n - 1))
    if not np.any(r[1:]):
        return 1, "constant"
    k = next((i for i in range(1, min(cap, len(r) - 1) + 1) if r[i] < _E_INV), None)
    if k is None:
        return cap, "sqrt(n)"
    L, rule = max(1, min(2 * k - 1, cap)), "acf<1/e@%d" % k
    # A periodic structure (an epoch-start stall, a validation gap): after the first drop the autocorrelation
    # returns to >= 1/e at a local peak P with at least three periods in the segment. A block of one period then
    # holds the same number of such events in every resample, as the data do.
    for p in range(k + 1, min(n // 3, len(r) - 2) + 1):
        if r[p] >= _E_INV and r[p] >= r[p - 1] and r[p] >= r[p + 1]:
            return p, "period@%d" % p
    return L, rule


# ----------------------------------------------------------------------------------------------
# moving-block bootstrap
# ----------------------------------------------------------------------------------------------
def _block_tables(x: np.ndarray, w: np.ndarray, L: int):
    """Sums of x*w and w over every length-L window (start s covers [s, s+L))."""
    cn = np.concatenate(([0.0], np.cumsum(x * w)))
    cd = np.concatenate(([0.0], np.cumsum(w)))
    return cn[L:] - cn[:-L], cd[L:] - cd[:-L]


def mbb_means(x: np.ndarray, w: Optional[np.ndarray], L: int, rng: np.random.Generator,
              B: int = RESAMPLES) -> np.ndarray:
    """B moving-block-bootstrap means of one segment (weights `w` = steps covered by each value)."""
    n = len(x)
    w = np.ones(n) if w is None else np.asarray(w, dtype=float)
    L = max(1, min(int(L), n))
    num, den = _block_tables(np.asarray(x, dtype=float), w, L)
    nb = int(math.ceil(n / float(L)))
    starts = rng.integers(0, n - L + 1, size=(B, nb))
    return num[starts].sum(axis=1) / den[starts].sum(axis=1)


def _wmean(x, w):
    return float(np.average(x, weights=w)) if w is not None else float(np.mean(x))


def diff_ci(before: Sequence[float], after: Sequence[float],
            w_before: Optional[Sequence[float]] = None, w_after: Optional[Sequence[float]] = None,
            confidence: float = CONFIDENCE, B: int = RESAMPLES, seed: int = SEED) -> Optional[dict]:
    """Interval on mean(after) - mean(before): {est, lo, hi, L_before, L_after, n_before, n_after}."""
    xb, xa = np.asarray(before, dtype=float), np.asarray(after, dtype=float)
    if len(xb) < 1 or len(xa) < 1:
        return None
    wb = None if w_before is None else np.asarray(w_before, dtype=float)
    wa = None if w_after is None else np.asarray(w_after, dtype=float)
    Lb, rb = block_length(xb)
    La, ra = block_length(xa)
    rng = np.random.default_rng(seed)
    d = mbb_means(xa, wa, La, rng, B) - mbb_means(xb, wb, Lb, rng, B)
    tail = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(d, [tail, 1.0 - tail])
    return {"est": _wmean(xa, wa) - _wmean(xb, wb), "lo": float(lo), "hi": float(hi),
            "L_before": Lb, "L_after": La, "rule_before": rb, "rule_after": ra,
            "n_before": len(xb), "n_after": len(xa), "mean_before": _wmean(xb, wb),
            "mean_after": _wmean(xa, wa)}


def mean_lower_bound(x: Sequence[float], w: Optional[Sequence[float]] = None,
                     confidence: float = CONFIDENCE, B: int = RESAMPLES,
                     seed: int = SEED) -> Optional[float]:
    """Lower end of the interval on the mean of one segment (the measured bound a move can recover)."""
    x = np.asarray(x, dtype=float)
    if len(x) < 1:
        return None
    wv = None if w is None else np.asarray(w, dtype=float)
    L, _ = block_length(x)
    means = mbb_means(x, wv, L, np.random.default_rng(seed), B)
    return float(np.quantile(means, (1.0 - confidence) / 2.0))


def effective_blocks(x: Sequence[float]) -> Tuple[int, float]:
    """(L, sigma) of a segment: block length and per-step standard deviation."""
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return 1, 0.0
    return block_length(x)[0], float(np.std(x, ddof=1))


# ----------------------------------------------------------------------------------------------
# verdict
# ----------------------------------------------------------------------------------------------
def verdict(wall_ci: Optional[dict], min_obs: int) -> str:
    """'confirm' | 'revert' | 'collect' from the step-time interval (see the module docstring)."""
    if wall_ci is None:
        return "collect"
    for side in ("before", "after"):
        if wall_ci["n_" + side] < max(min_obs, 2 * wall_ci["L_" + side]):
            return "collect"
    if wall_ci["hi"] < 0.0:
        return "confirm"
    if wall_ci["lo"] > 0.0:
        return "revert"
    return "collect"


# ----------------------------------------------------------------------------------------------
# side effects
# ----------------------------------------------------------------------------------------------
class SideCost:
    """Measured cost of one applied move beyond its gain, from its post-apply steps.

    one_time_s     step disruption around the apply, net of the apply latency already priced
    apply_s        apply latency the application measured (0 when unknown)
    rate_s_per_step excess step time in windows where the kernel reclaimed memory, per post step
    rss_kb         growth of the node's anonymous memory over the post-apply windows (kB)
    unit           |new value - old value| of the move (the scale for the next move of this knob)
    """

    __slots__ = ("one_time_s", "apply_s", "rate_s_per_step", "rss_kb", "unit", "n_post", "reclaim_windows")

    def __init__(self, one_time_s=0.0, apply_s=0.0, rate_s_per_step=0.0, rss_kb=0.0, unit=1.0,
                 n_post=0, reclaim_windows=0):
        self.one_time_s, self.apply_s, self.rate_s_per_step = one_time_s, apply_s, rate_s_per_step
        self.rss_kb, self.unit, self.n_post, self.reclaim_windows = rss_kb, unit, n_post, reclaim_windows

    def scaled(self, unit: float) -> "SideCost":
        r = (float(unit) / self.unit) if self.unit > 0 else 1.0
        return SideCost(self.one_time_s * r, self.apply_s * r, self.rate_s_per_step * r,
                        self.rss_kb * r, unit, self.n_post, self.reclaim_windows)

    def as_dict(self) -> dict:
        return {k: round(float(getattr(self, k)), 6) for k in self.__slots__}


def measure_side_cost(pre_wall: np.ndarray, post_wall: np.ndarray, post_w: Optional[np.ndarray],
                      L_post: int, apply_s: float, unit: float,
                      windows: Sequence[dict], rss_start_kb: Optional[float]) -> Optional[SideCost]:
    """Side effects of an applied move from its post-apply per-step walls and per-window telemetry.

    `windows` is one dict per post-apply window: {n, mean_wall, reclaim (counter delta over the
    window, or None when unmeasured), anon_kb}.  Returns None until there are at least two blocks of
    post-apply steps (nothing to separate the disruption from the steady level).
    """
    n = len(post_wall)
    if n < 2 * max(L_post, 1) or len(pre_wall) < 1:
        return None
    pre_med = float(np.median(pre_wall))
    steady = float(np.median(post_wall[L_post:]))
    wts = np.ones(n) if post_w is None else np.asarray(post_w, dtype=float)
    head = np.maximum(post_wall[:L_post] - steady, 0.0) * wts[:L_post]
    disruption = float(head.sum())
    one_time = max(disruption - max(apply_s, 0.0), 0.0)
    steps = float(wts.sum())
    reclaim_s, reclaim_windows = 0.0, 0
    for win in windows:
        rec = win.get("reclaim")
        if rec is not None and rec > 0:
            reclaim_windows += 1
            reclaim_s += float(win["n"]) * max(float(win["mean_wall"]) - pre_med, 0.0)
    rss = 0.0
    anon = [win.get("anon_kb") for win in windows if win.get("anon_kb") is not None]
    if rss_start_kb is not None and anon:
        rss = max(float(anon[-1]) - float(rss_start_kb), 0.0)
    return SideCost(one_time_s=one_time, apply_s=max(apply_s, 0.0),
                    rate_s_per_step=(reclaim_s / steps if steps > 0 else 0.0), rss_kb=rss, unit=unit,
                    n_post=int(steps), reclaim_windows=reclaim_windows)


def price_next_move(side: Optional[SideCost], unit: float, delta: Optional[float],
                    remaining_steps: Optional[float], apply_s: float,
                    mem_avail_kb: Optional[float]) -> Optional[str]:
    """Veto text when a further move of the same knob is not worth its measured cost, else None.

    gain = (predicted per-step effect - recurring side cost per step) x remaining steps
    cost = apply latency + one-time side cost
    A move is declined when the recurring side cost alone consumes the predicted effect, when the gain
    does not exceed the cost, or when the knob's measured memory growth per unit would exceed the
    node's available memory (the reclaim the previous move caused would recur).
    """
    if side is None or delta is None:
        return None
    s = side.scaled(unit)
    if mem_avail_kb is not None and s.rss_kb > 0 and s.rss_kb >= mem_avail_kb:
        return "side_effects:memory_growth=%.0fkB>=avail=%.0fkB" % (s.rss_kb, mem_avail_kb)
    net = float(delta) - s.rate_s_per_step
    if net <= 0.0:
        return "side_effects:rate=%.4fs/step>=effect=%.4fs/step" % (s.rate_s_per_step, float(delta))
    if remaining_steps is None:
        return None
    cost = max(float(apply_s), 0.0) + s.one_time_s
    if net * float(remaining_steps) <= cost:
        return "side_effects:gain=%.1fs<=cost=%.1fs" % (net * float(remaining_steps), cost)
    return None
