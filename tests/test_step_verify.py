"""Per-step targeted-exposure verification: block length, moving-block bootstrap, verdict, codec,
side-effect pricing, and the planner wiring (LIVEFLOW_VERIFY=per_step | window_mean)."""
import math

import numpy as np
import pytest

from dfoptimizer import step_series
from dfoptimizer.planner import step_verify as sv
from dfoptimizer.planner.cbandit_planner import CoordinateBanditPlanner
from dfoptimizer.types import KnobDef


def _ar1(n, phi, rng, scale=1.0):
    e = rng.normal(0, scale, n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


# ------------------------------------------------------------------ block rule
def test_block_length_rule():
    rng = np.random.default_rng(1)
    assert sv.block_length(np.ones(50))[0] == 1                       # constant
    assert sv.block_length(rng.normal(size=400))[0] == 1              # white noise: ACF(1) ~ 0 < 1/e
    ar = _ar1(2000, 0.9, rng)                                         # ACF(k)=0.9^k < 1/e at k=10 -> L = 19
    L, rule = sv.block_length(ar)
    assert 13 <= L <= 25 and rule.startswith("acf<1/e")
    trend = np.arange(400, dtype=float) + rng.normal(0, 0.1, 400)     # never decays within sqrt(n) lags
    L, rule = sv.block_length(trend)
    assert L == math.ceil(math.sqrt(400)) and rule == "sqrt(n)"
    assert sv.block_length(np.array([1.0]))[0] == 1


# ------------------------------------------------------------------ interval calibration
def _coverage(phi, shift, trials, n=300, B=400):
    rng = np.random.default_rng(7)
    hits = 0
    for _ in range(trials):
        a, b = _ar1(n, phi, rng), _ar1(n, phi, rng) + shift
        ci = sv.diff_ci(a, b, B=B, seed=int(rng.integers(1 << 30)))
        hits += ci["lo"] <= shift <= ci["hi"]
    return hits / trials


def test_interval_covers_true_shift_for_iid_and_autocorrelated_steps():
    assert _coverage(0.0, 0.3, 150) >= 0.88      # nominal 0.95, finite-sample bootstrap is slightly liberal
    assert _coverage(0.7, 0.3, 150) >= 0.82      # block resampling keeps coverage under autocorrelation


def test_interval_excludes_zero_for_a_real_shift_and_not_for_none():
    rng = np.random.default_rng(3)
    base = rng.normal(0.10, 0.02, 400)
    better = rng.normal(0.08, 0.02, 400)
    ci = sv.diff_ci(base, better)
    assert ci["hi"] < 0 and ci["est"] == pytest.approx(better.mean() - base.mean())
    same = sv.diff_ci(base, rng.normal(0.10, 0.02, 400))
    assert same["lo"] < 0 < same["hi"]
    assert sv.diff_ci([], [1.0]) is None


def test_weighted_bins_equal_the_unbinned_mean():
    rng = np.random.default_rng(5)
    x = rng.normal(1.0, 0.1, 100)
    ss = step_series.decode(step_series.encode(1, {"wall": x}, max_bins=25))     # 4 steps per bin
    assert ss.bin == 4 and len(ss.weights) == 25 and ss.n == 100
    assert np.average(ss.series["wall"], weights=ss.weights) == pytest.approx(x.mean(), abs=1e-4)


def test_verdict_is_taken_on_the_step_time_interval():
    base = np.full(60, 0.10) + np.random.default_rng(0).normal(0, 0.002, 60)
    ci = lambda post: sv.diff_ci(base, post)
    assert sv.verdict(ci(base - 0.02), 3) == "confirm"
    assert sv.verdict(ci(base + 0.02), 3) == "revert"
    assert sv.verdict(ci(base + np.random.default_rng(1).normal(0, 0.05, 60)), 3) == "collect"
    assert sv.verdict(sv.diff_ci(base, base[:2] - 0.5), 3) == "collect"          # too few post steps
    assert sv.verdict(None, 3) == "collect"


# ------------------------------------------------------------------ codec
def test_codec_round_trip_and_rejects_malformed():
    series = {"wall": [0.0921, 0.0932, 0.5], "storage": [0.01, 0.0, 0.2], "comm": [0.03, 0.03, 0.03]}
    text = step_series.encode(401, series)
    ss = step_series.decode(text)
    assert (ss.first, ss.n, ss.bin, ss.last) == (401, 3, 1, 403)
    assert ss.series["wall"] == pytest.approx(series["wall"], abs=1e-5)
    assert step_series.decode(None) is None and step_series.decode("v1;first=1;n=2;b=1") is None
    assert step_series.decode(text.replace("n=3", "n=4")) is None             # length mismatch
    assert step_series.encode(1, {"storage": [1.0]}) is None                   # no wall


# ------------------------------------------------------------------ side effects and the gate
def test_side_cost_measured_from_post_apply_steps_and_reclaim_windows():
    pre = np.full(50, 0.100)
    post = np.concatenate(([0.9, 0.5], np.full(98, 0.080)))                  # disruption around the apply
    wins = [{"n": 50, "mean_wall": 0.09, "reclaim": 0, "anon_kb": 10e6},
            {"n": 50, "mean_wall": 0.14, "reclaim": 120, "anon_kb": 14e6}]   # reclaim window: 0.04 s/step above pre
    c = sv.measure_side_cost(pre, post, None, 2, apply_s=0.3, unit=8.0, windows=wins, rss_start_kb=9e6)
    assert c.one_time_s == pytest.approx((0.9 - 0.08) + (0.5 - 0.08) - 0.3)
    assert c.rate_s_per_step == pytest.approx(50 * 0.04 / 100)
    assert c.rss_kb == pytest.approx(5e6) and c.reclaim_windows == 1 and c.unit == 8.0
    assert sv.measure_side_cost(pre, post[:3], None, 2, 0.0, 1.0, wins, None) is None   # < 2 blocks


def test_gate_prices_the_next_move_from_the_measured_side_cost():
    c = sv.SideCost(one_time_s=1.0, apply_s=0.5, rate_s_per_step=0.002, rss_kb=2e6, unit=8.0)
    # moving twice as far costs twice as much
    assert sv.price_next_move(c, 8.0, 0.010, 10000, 0.5, 100e6) is None            # 0.008*10000 = 80 s > 2.5 s
    assert "gain" in sv.price_next_move(c, 16.0, 0.010, 100, 0.5, 100e6)            # 0.006*100 = 0.6 s < 0.5+2.0
    assert "rate" in sv.price_next_move(c, 8.0, 0.0015, 10000, 0.5, 100e6)          # recurring cost eats the effect
    assert "memory" in sv.price_next_move(c, 16.0, 0.010, 10000, 0.5, 3e6)          # 4e6 kB growth > 3e6 kB available
    assert sv.price_next_move(None, 8.0, 0.01, 1000, 0.5, None) is None             # nothing measured: no veto
    assert sv.price_next_move(c, 8.0, None, 1000, 0.5, None) is None


# ------------------------------------------------------------------ planner wiring
def _planner(mode):
    import os
    old = os.environ.get("LIVEFLOW_VERIFY")
    os.environ["LIVEFLOW_VERIFY"] = mode
    try:
        p = CoordinateBanditPlanner()
    finally:
        if old is None:
            os.environ.pop("LIVEFLOW_VERIFY")
        else:
            os.environ["LIVEFLOW_VERIFY"] = old
    p.register_knobs({"halo3d.decode_threads": KnobDef(id="halo3d.decode_threads", default=1, type=int,
                                                       values=[1, 2, 4, 8], cost_seconds=0.001,
                                                       target_function="halo3d.make_loader")})
    p._rate_mode = True
    p._min_samples = 3
    return p


def _feed(p, rng, pre_mean, post_mean, n_post, sigma=0.01, applied=200):
    """Window records covering steps 6..applied (pre) and applied+1.. (post), 100 steps each."""
    steps = list(range(6, applied + 1)) + list(range(applied + 1, applied + 1 + n_post))
    wall = np.array([pre_mean if s <= applied else post_mean for s in steps]) + rng.normal(0, sigma, len(steps))
    exp = np.where(np.array(steps) <= applied, pre_mean * 0.4, post_mean * 0.4) + rng.normal(0, sigma / 4, len(steps))
    for i in range(0, len(steps), 100):
        chunk = slice(i, i + 100)
        text = step_series.encode(steps[i], {"wall": wall[chunk].tolist(), "decode": np.maximum(exp[chunk], 0).tolist()})
        ss = step_series.decode(text)
        ss.window = i // 100 + 1
        p._tl.append(ss)
    p._physical_latest_cfg = {"decode_threads": 2}
    p._physical_changes = [("decode_threads", 2, applied)]
    p._move_from["decode_threads"] = (1, 0.02, 1, 0.1, 2)
    p.current_values["halo3d.decode_threads"] = 2


def test_per_step_confirms_a_faster_move_in_a_few_hundred_steps():
    p = _planner("per_step")
    _feed(p, np.random.default_rng(0), 0.100, 0.080, 150)
    assert p._unconfirmed_move(4, 3000.0) is None
    assert "decode_threads" not in p._move_from                               # confirmed and cleared
    rec = p._sv_last["decode_threads"]
    assert rec["verdict"] == "confirm" and rec["step_ci"]["hi"] < 0
    assert rec["exposure_ci"]["hi"] < 0 and rec["target_series"] == "decode"


def test_per_step_reverts_a_slower_move():
    p = _planner("per_step")
    _feed(p, np.random.default_rng(1), 0.100, 0.125, 150)
    assert p._unconfirmed_move(4, 3000.0) == ("decode_threads", 1)


def test_per_step_keeps_collecting_when_the_interval_straddles_zero():
    p = _planner("per_step")
    _feed(p, np.random.default_rng(2), 0.100, 0.100, 150, sigma=0.03)
    assert p._unconfirmed_move(4, 3000.0) is None
    assert "decode_threads" in p._move_from and p._sv_last["decode_threads"]["verdict"] == "collect"


def test_window_mean_mode_ignores_the_series():
    p = _planner("window_mean")
    _feed(p, np.random.default_rng(0), 0.100, 0.125, 150)
    p._verify = "window_mean"
    assert not p._sv_enabled()
    assert p._unconfirmed_move(4, 3000.0) is None                            # no arm samples: the old path waits
    assert "decode_threads" in p._move_from and not p._sv_last


def test_predicted_effect_and_required_steps_come_from_the_series():
    p = _planner("per_step")
    _feed(p, np.random.default_rng(0), 0.100, 0.100, 200, applied=100)
    p._physical_changes = []                                                    # one configuration throughout
    delta, src = p.predicted_effect("decode_threads", "decode")
    assert src.startswith("domain_exposed:step_exposure:decode") and 0.03 < delta < 0.045   # lb of mean 0.04
    need, why = p.required_dwell(5000.0, short="decode_threads", value=1, delta=delta)
    assert why["sigma_from"] == "step_series" and why["reachable"] and why["n_required_steps"] >= 3
    _, small = p.required_dwell(5000.0, short="decode_threads", value=1, delta=0.003)
    assert small["n_required_steps"] > 10 * why["n_required_steps"]               # smaller effect, many more steps
    huge, why2 = p.required_dwell(50.0, short="decode_threads", value=1, delta=1e-5)
    assert why2["reachable"] is False                                           # unreachable inside 50 steps
