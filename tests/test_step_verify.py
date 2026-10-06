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


def test_periodic_spikes_get_a_block_of_one_period_and_a_tight_interval():
    rng = np.random.default_rng(2)
    def epochs(n_epochs, spike):
        x = rng.normal(0.20, 0.01, 47 * n_epochs)
        x[::47] += spike                                         # one loader-restart stall per epoch
        return x
    pre, post = epochs(3, 1.1), epochs(5, 3.5)
    assert sv.block_length(pre) == (47, "period@47") and sv.block_length(post)[0] == 47
    ci = sv.diff_ci(pre, post)
    # the stall rose by 2.4 s per epoch = +51 ms/step; whole-period blocks make that significant after five epochs,
    # where an iid bootstrap (spike count random in every resample) could not separate it from zero
    assert ci["lo"] > 0 and ci["est"] == pytest.approx(0.051, abs=0.006)
    iid = sv.diff_ci(pre, post, B=2000)
    assert (ci["hi"] - ci["lo"]) < 0.3 * 0.1                    # tight


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


def _feed_exposure(p, rng, pre_exp, post_exp, n_post, wall_sigma=0.03, applied=200, exp_sigma=0.002):
    """Step time pure noise around 0.1 s; the targeted (decode) exposure moves from pre_exp to post_exp."""
    steps = list(range(6, applied + 1)) + list(range(applied + 1, applied + 1 + n_post))
    wall = 0.100 + rng.normal(0, wall_sigma, len(steps))
    exp = np.array([pre_exp if s_ <= applied else post_exp for s_ in steps]) + rng.normal(0, exp_sigma, len(steps))
    for i in range(0, len(steps), 100):
        ss = step_series.decode(step_series.encode(steps[i], {"wall": wall[i:i + 100].tolist(),
                                                              "decode": np.maximum(exp[i:i + 100], 0).tolist()}))
        ss.window = i // 100 + 1
        p._tl.append(ss)
    p._physical_latest_cfg = {"decode_threads": 2}
    p._physical_changes = [("decode_threads", 2, applied)]
    p._move_from["decode_threads"] = (1, 0.04, 1, 0.1, 2)
    p.current_values["halo3d.decode_threads"] = 2


def test_per_step_keeps_collecting_when_the_exposure_fell_but_the_step_time_is_still_noisy():
    p = _planner("per_step")
    _feed_exposure(p, np.random.default_rng(2), 0.040, 0.020, 150)
    assert p._unconfirmed_move(4, 3000.0) is None
    rec = p._sv_last["decode_threads"]
    assert "decode_threads" in p._move_from and rec["verdict"] == "collect" and rec["exposure_ci"]["hi"] < 0
    assert p._sv_mechanism_confirmed("decode_threads", p._move_from["decode_threads"])    # no longer blocks other families


def test_per_step_reverts_a_futile_move_only_once_the_steps_its_prediction_needed_have_been_collected():
    p = _planner("per_step")
    p._control_steps = 100                                                                   # K: one window = 100 steps
    _feed_exposure(p, np.random.default_rng(3), 0.040, 0.040, 60, exp_sigma=0.01)          # less than one window since the apply
    assert p._unconfirmed_move(4, 3000.0) is None and p._sv_last["decode_threads"]["verdict"] == "collect"
    q = _planner("per_step")
    q._control_steps = 100
    _feed_exposure(q, np.random.default_rng(3), 0.040, 0.040, 600, exp_sigma=0.01)          # plenty: no effect on the target
    assert q._unconfirmed_move(9, 3000.0) == ("decode_threads", 1)
    rec = q._sv_last["decode_threads"]
    assert rec["verdict"] == "revert" and rec["reason"].startswith("futile")
    assert ("decode_threads", 1, 2, q._sv_others("decode_threads")) in q._refuted


def test_per_step_reverts_a_move_that_raised_its_own_exposure():
    p = _planner("per_step")
    _feed_exposure(p, np.random.default_rng(4), 0.040, 0.060, 150)
    assert p._unconfirmed_move(4, 3000.0) == ("decode_threads", 1)
    assert p._sv_last["decode_threads"]["reason"] == "own_exposure_rose"


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


# ------------------------------------------------------------------ per-step planner policy
def test_exploration_only_raises_the_resource_in_per_step_mode():
    p = _planner("per_step")
    p._tl.append(step_series.decode(step_series.encode(1, {"wall": [0.1] * 10})))                # series present: per-step path
    p.current_values["halo3d.decode_threads"] = 2
    p._levels["decode_threads"] = [1, 2, 3, 4, 5, 6, 7, 8]
    choice, detail = p._pick("decode_threads", 2, "decode", exposed_s=0.08, remaining=3000.0)
    assert {d["level"] for d in detail} <= {2, 4}                       # no level below the current one is a candidate
    w = _planner("window_mean")
    w.current_values["halo3d.decode_threads"] = 2
    w._levels["decode_threads"] = [1, 2, 3, 4, 5, 6, 7, 8]
    assert 1 in {d["level"] for d in w._pick("decode_threads", 2, "decode", exposed_s=0.08, remaining=3000.0)[1]}


def test_a_move_whose_own_exposure_fell_no_longer_blocks_other_moves():
    p = _planner("per_step")
    _feed(p, np.random.default_rng(3), 0.100, 0.100, 150, sigma=0.05)
    p._sv_last.clear()
    p._move_from["decode_threads"] = (1, 0.04, 1, 0.1, 2)
    assert p._inflight_move() == "decode_threads"
    p._sv_last["decode_threads"] = {"to": 2, "verdict": "collect", "exposure_ci": {"hi": -0.01}}   # its own exposure fell
    assert p._inflight_move() is None
    p._sv_last["decode_threads"]["exposure_ci"] = {"hi": 0.002}
    assert p._inflight_move() == "decode_threads"


def test_escalation_follows_measured_verdicts_of_this_configuration_only():
    p = _planner("per_step")
    p._tl.append(step_series.decode(step_series.encode(1, {"wall": [0.1] * 10})))        # enable the per-step path
    cur = 2
    alts = [1, 4]
    assert p._sv_escalation_state("decode_threads", cur, alts) == (False, "untried_neighbour:4")   # raises only; 4 untried
    others = p._sv_others("decode_threads")
    p._tried[("decode_threads", 2, 4, others)] = "worse"
    assert p._sv_escalation_state("decode_threads", cur, alts) == (True, "neighbourhood_exhausted_no_help")
    p._tried[("decode_threads", 2, 4, others)] = "better"
    assert p._sv_escalation_state("decode_threads", cur, alts) == (False, "helpful_neighbour:4")
    p._physical_latest_cfg = {"decode_threads": 2, "read_threads": 3}                       # another knob moved: not evidence now
    assert p._sv_escalation_state("decode_threads", cur, alts) == (False, "untried_neighbour:4")


def test_a_refuted_move_stays_refuted_until_the_targeted_exposure_rises_above_the_refuted_interval():
    p = _planner("per_step")
    p._refuted[("decode_threads", 2, 4, p._sv_others("decode_threads"))] = {"lo": 0.03, "hi": 0.05, "window": 9}
    assert "refuted" in p._sv_refuted("decode_threads", 2, 4, 0.04)
    assert p._sv_refuted("decode_threads", 2, 4, 0.06) is None             # exposure rose above the interval: re-admitted
    assert ("decode_threads", 2, 4, p._sv_others("decode_threads")) not in p._refuted
    p._refuted[("decode_threads", 2, 4, p._sv_others("decode_threads"))] = {"lo": 0.03, "hi": 0.05, "window": 9}
    p._physical_latest_cfg = {"decode_threads": 2, "read_threads": 3}      # configuration differs: refutation does not apply
    assert p._sv_refuted("decode_threads", 2, 4, 0.04) is None


def test_side_effect_ledger_blocks_the_next_move_of_the_same_knob():
    p = _planner("per_step")
    p._side["decode_threads"] = sv.SideCost(one_time_s=2.0, apply_s=0.2, rate_s_per_step=0.004, rss_kb=1e6, unit=1.0)
    p._tele[7] = {"avail_kb": 5e8, "anon_kb": None, "reclaim": 0.0}
    assert p._sv_gate("decode_threads", 2, 4, 0.01, 5000, 0.2) is None                       # 0.006*5000 = 30 s >> 2.4 s
    assert "gain" in p._sv_gate("decode_threads", 2, 4, 0.01, 300, 0.2)                      # 1.8 s < 0.2 + 4.0 s
    assert "rate" in p._sv_gate("decode_threads", 2, 3, 0.003, 5000, 0.2)                    # recurring 0.004/step eats 0.003
    assert p._sv_gate("read_threads", 2, 4, 0.01, 10, 0.2) is None                           # a knob with no measurement
    w = _planner("window_mean")
    w._side["decode_threads"] = p._side["decode_threads"]
    assert w._sv_gate("decode_threads", 2, 4, 0.0001, 10, 0.2) is None                       # old mode never prices side effects
