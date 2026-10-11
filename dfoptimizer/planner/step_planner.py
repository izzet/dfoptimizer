"""Per-step verification state and decisions for CoordinateBanditPlanner (LIVEFLOW_VERIFY=per_step).

A mixin: it owns the per-step timeline decoded from the window records, the node telemetry deltas,
the side-cost ledger and the refuted-move memory, and answers four questions for the planner:

  * `_sv_enabled()`                 is the per-step path selected and is there data for it
  * `_sv_predicted_exposure()`      the measured bound a move of this knob can recover (per step)
  * `_sv_required_steps()`          steps a verdict on that effect needs (power rule, block-adjusted)
  * `_sv_verdict()`                 confirm / revert / collect for an applied move (step_verify.py)
  * `_sv_gate()`                    side-effect pricing of the next move of a knob

Nothing here changes the window-mean path: with LIVEFLOW_VERIFY=window_mean the planner never calls it.
"""
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import structlog

from .. import step_series
from . import step_verify as sv

logger = structlog.get_logger()

# The first steps of a run are cold (first-collective, kernel compile, page cache).  Window.stop sizes
# the cadence from steps 6..25 for the same reason; the verification excludes steps <= WARMUP_STEPS
# from every segment for the same reason.
WARMUP_STEPS = 5
_TIMELINE_WINDOWS = 256
_MODES = ("per_step", "window_mean")


def verify_mode() -> str:
    mode = os.environ.get("LIVEFLOW_VERIFY", "per_step").strip().lower()
    if mode not in _MODES:
        logger.warning("optimizer.verify.bad_mode", value=mode, using="per_step")
        return "per_step"
    return mode


class StepVerifyMixin:
    def _sv_init(self):
        self._verify = verify_mode()
        self._tl: List[step_series.StepSeries] = []
        self._sv_steady: Dict[int, np.ndarray] = {}  # verification-only eligibility by native window
        self._tele: Dict[int, dict] = {}          # window -> {reclaim, anon_kb, avail_kb}
        self._tele_prev: Optional[dict] = None
        self._side: Dict[str, sv.SideCost] = {}   # per knob, latest measured side cost
        self._side_watch: Dict[str, dict] = {}    # per knob, the applied move whose side effects are measured
        self._refuted: Dict[tuple, dict] = {}     # (short, from, to, other knobs) -> pre-move exposure interval
        self._tried: Dict[tuple, str] = {}        # (short, from, to, other knobs) -> 'better' | 'worse' (measured)
        self._sv_last: Dict[str, dict] = {}       # per knob, the latest verdict record (for logs/tests)
        self._sv_absent_logged = False

    # ------------------------------------------------------------------ ingestion
    def _sv_ingest(self, window: int, m: dict) -> None:
        """Decode this window's per-step series and node telemetry (before any early return)."""
        if self._verify != "per_step":
            return
        ss = step_series.decode(m.get("step_series"))
        if ss is not None:
            if not self._tl or ss.first > self._tl[-1].last:
                ss.window = window
                self._tl.append(ss)
                self._sv_steady[window] = self._sv_steady_mask(ss, m)
                if len(self._tl) > _TIMELINE_WINDOWS:
                    del self._tl[: len(self._tl) - _TIMELINE_WINDOWS]
                    retained = {s.window for s in self._tl}
                    self._sv_steady = {w: mask for w, mask in self._sv_steady.items() if w in retained}
        elif not self._sv_absent_logged and m.get("liveflow_v1_attribution") == 1.0:
            self._sv_absent_logged = True
            logger.info("optimizer.verify.step_series_absent", window=window,
                        note="the window record carries no per-step series; window-mean verification is used")
        # Cumulative node counters (vmstat) -> per-window deltas; anonymous memory is a level.
        cur = {k: m.get(k) for k in ("vm_allocstall", "vm_pgscan_direct", "vm_compact_stall")}
        tele = {"anon_kb": m.get("mem_anon_kb"), "avail_kb": m.get("mem_avail_kb"), "reclaim": None}
        if all(v is not None for v in cur.values()):
            if self._tele_prev is not None:
                tele["reclaim"] = sum(max(float(cur[k]) - float(self._tele_prev[k]), 0.0) for k in cur)
            self._tele_prev = {k: float(v) for k, v in cur.items()}
        self._tele[window] = tele
        if len(self._tele) > _TIMELINE_WINDOWS:
            for old in sorted(self._tele)[: len(self._tele) - _TIMELINE_WINDOWS]:
                del self._tele[old]

    def _sv_enabled(self) -> bool:
        return self._verify == "per_step" and bool(self._tl)

    def _sv_active(self) -> bool:
        """The per-step path governs this decision: selected, rate mode (liveflow-v1), and the window records carry series."""
        return self._rate_mode and self._sv_enabled()

    # ------------------------------------------------------------------ segments
    def _sv_key(self, short: str, ss: Optional[step_series.StepSeries] = None) -> Optional[str]:
        """The series that carries the targeted family's per-step exposure."""
        dom = self._domain.get(short)
        have = (ss or self._tl[-1]).series if (ss is not None or self._tl) else {}
        if dom == "input":
            return next((key for key in ("job_input", "input", "storage") if key in have), None)
        if dom == "decode":
            # A decode worker pool can expose only a combined parent fetch wait.
            # Keep that measurement labelled input; never call it pure decode.
            return next((key for key in ("decode", "input") if key in have), None)
        return dom if dom in have else None

    @staticmethod
    def _sv_steady_mask(ss: step_series.StepSeries, m: dict) -> np.ndarray:
        """Exclude declared pre-steady windows and the same first-five cold steps in each epoch.

        Local control-step coverage must match the wire series exactly before it can locate an
        epoch head. Missing or ambiguous coverage keeps the legacy global warm-up rule; an epoch
        length is never inferred. A bin containing any cold sample is excluded as a whole because
        its mean cannot be split into observed cold and steady values.
        """
        keep = np.ones(len(ss.weights), dtype=bool)
        if (m.get("planner_history_ready") == 0 or m.get("verification_epoch_index") == 0
                or m.get("epoch_index") == 0):
            return np.zeros(len(ss.weights), dtype=bool)
        try:
            first, last, count = (float(m[k]) for k in ("step_min", "step_max", "step_nunique"))
        except (KeyError, TypeError, ValueError):
            return keep
        if (m.get("epoch_nunique") != 1 or not all(math.isfinite(v) for v in (first, last, count))
                or first < 1 or first != int(first) or last != int(last)
                or count != ss.n or last - first + 1 != ss.n):
            return keep
        return first + ss.bin_starts() - ss.first > WARMUP_STEPS

    def _sv_segment(self, key: str, lo_excl: float, hi_incl: float, *, steady_only: bool = False):
        """Values/weights in (lo, hi]; optionally apply verification-only cold exclusions."""
        xs, ws = [], []
        lo = max(lo_excl, WARMUP_STEPS)
        for ss in self._tl:
            if key not in ss.series or ss.last <= lo or ss.first > hi_incl:
                continue
            starts = ss.bin_starts()
            keep = (starts > lo) & (starts <= hi_incl)
            if steady_only and ss.window in self._sv_steady:
                keep &= self._sv_steady[ss.window]
            if keep.any():
                xs.append(ss.series[key][keep])
                ws.append(ss.weights[keep])
        if not xs:
            return np.array([]), np.array([])
        return np.concatenate(xs), np.concatenate(ws)

    def _sv_boundaries(self) -> List[int]:
        changes = self._physical_changes if self._physical_latest_cfg is not None else []
        return sorted({int(c[2]) for c in changes if c[2] is not None})

    def _sv_current_segment_start(self) -> int:
        b = self._sv_boundaries()
        return b[-1] if b else 0

    # ------------------------------------------------------------------ admission
    def _sv_predicted_exposure(self, short: str) -> Optional[Tuple[float, str]]:
        """Lower bound of the targeted family's mean per-step exposure over the CURRENT configuration."""
        if not self._sv_enabled():
            return None
        key = self._sv_key(short)
        if key is None:
            return None
        x, w = self._sv_segment(key, self._sv_current_segment_start(), float("inf"))
        if len(x) < max(self._min_samples, 2 * sv.block_length(x)[0]):
            return None
        lb = sv.mean_lower_bound(x, w)
        if lb is None or lb <= 0.0:
            return 0.0, "no_exposed_time"
        return lb, "step_exposure:%s" % key

    def _sv_early_effect(self, short: str) -> Optional[Tuple[float, str]]:
        """Admit a measured domain price from one clean steady window.

        When no targeted step series exists, its native window exposure is still
        measured. Use it only if the 95% block-bootstrap interval of the mean
        wall time has half-width smaller than that price, with at least two
        blocks and the existing minimum observations. This tests precision of
        the gain's wall-time yardstick, not an exposure-specific mechanism CI.
        The ordinary horizon and apply-cost gates still price the move.
        """
        if not self._sv_active() or not self._tl or self._epoch_windows < 1:
            return None
        ss = self._tl[-1]
        if (self._win_metrics.get("planner_history_ready") != 1
                or ss.first <= max(WARMUP_STEPS, self._sv_current_segment_start())):
            return None
        own = self._domain.get(short)
        prices = self._epoch_dom.get(own) or ()
        if not prices or prices[-1] is None or prices[-1] <= 0.0:
            return None
        wall = ss.series["wall"]
        L = sv.block_length(wall)[0]
        if len(wall) < max(self._min_samples, 2 * L):
            return None
        means = sv.mbb_means(wall, ss.weights, L, np.random.default_rng(sv.SEED))
        lo, hi = np.quantile(means, [(1 - sv.CONFIDENCE) / 2, (1 + sv.CONFIDENCE) / 2])
        if (hi - lo) / 2 >= float(prices[-1]):
            return None
        return float(prices[-1]), "steady_window_exposure:%s:wall_precision95" % own

    def _sv_required_steps(self, short: str, delta: float, remaining: Optional[float]) -> Optional[Tuple[int, dict]]:
        """Steps a verdict on an effect of `delta` s/step needs, and whether the horizon holds them.

        Power rule (alpha, power as everywhere in this planner): the blocks needed to resolve `delta`
        against the per-step spread of the step time, times the block length that spread is measured
        with (the same rule the bootstrap uses).  Only the AFTER segment is still to be collected.
        """
        if not self._sv_enabled() or delta is None or delta <= 0.0:
            return None
        x, w = self._sv_segment("wall", self._sv_current_segment_start(), float("inf"))
        if len(x) < 2:
            return None
        L, sigma = sv.effective_blocks(x)
        z = self._z(1.0 - self._alpha / 2.0) + self._z(self._power)
        blocks = int(math.ceil((z * sigma / float(delta)) ** 2)) if sigma > 0 else 1
        need = max(blocks * L, self._min_samples, 2 * L)
        why = {"n_required_steps": need, "L": L, "sigma_step": round(sigma, 6), "blocks": blocks,
               "horizon_unit": "steps", "alpha": self._alpha, "power": self._power}
        if remaining is not None and remaining > 0 and need > remaining:
            why.update(reachable=False, horizon=int(remaining), reason="exceeds_horizon")
        else:
            why.update(reachable=True, reason="derived")
        return need, why

    # ------------------------------------------------------------------ verification
    def _sv_verdict(self, short: str, old_value, cur, change_step: int, window: int,
                    delta: Optional[float] = None) -> Optional[str]:
        """'confirm' | 'revert' | 'collect', or None when no per-step timeline exists.

        Missing target exposure never blocks a wall-time bootstrap verdict.
        Exposure-specific futility/refutation requires actual target samples;
        missing or too few wall samples collect rather than falling back to
        an under-sampled window test.

        confirm   the step-time interval excludes zero on the improving side
        revert    the step-time interval excludes zero on the worsening side; or the targeted exposure itself rose
                  (interval excludes zero); or FUTILE: the steps collected reach what the predicted effect needed
                  (power rule, below) and the targeted exposure has still not fallen with an interval that excludes
                  zero, i.e. the move delivered no detectable part of the effect that justified it
        collect   otherwise; collection is bounded by the horizon (admission already required the needed steps
                  to fit in it)
        """
        if not self._sv_enabled():
            return None
        key = self._sv_key(short)
        bounds = self._sv_boundaries()
        prev = max([b for b in bounds if b < change_step], default=0)
        nxt = min([b for b in bounds if b > change_step], default=float("inf"))
        wall_pre, w_pre = self._sv_segment("wall", prev, change_step, steady_only=True)
        wall_post, w_post = self._sv_segment("wall", change_step, nxt, steady_only=True)
        if len(wall_pre) < 1 or len(wall_post) < 1:
            return "collect"
        exp_pre, wx_pre = self._sv_segment(key, prev, change_step, steady_only=True) if key else (np.array([]), np.array([]))
        exp_post, wx_post = self._sv_segment(key, change_step, nxt, steady_only=True) if key else (np.array([]), np.array([]))
        wall_ci = sv.diff_ci(wall_pre, wall_post, w_pre, w_post)
        exp_ci = sv.diff_ci(exp_pre, exp_post, wx_pre, wx_post) if len(exp_pre) and len(exp_post) else None
        verdict = sv.verdict(wall_ci, self._min_samples)
        reason = verdict
        n_post = int(w_post.sum())
        if verdict == "collect" and exp_ci is not None:
            if exp_ci["n_after"] >= max(self._min_samples, 2 * exp_ci["L_after"]) and exp_ci["lo"] > 0.0:
                verdict, reason = "revert", "own_exposure_rose"
            elif exp_ci["hi"] >= 0.0 and delta:
                need = self._sv_futility_steps(exp_pre, wx_pre, float(delta), old_value, cur)
                # never before one full window of post-apply steps (the unit the planner decides in)
                if need is not None and n_post >= max(need, int(self._control_steps)):
                    verdict, reason = "revert", "futile:no_detectable_effect_after_%d_steps" % need
        rec = {"window": window, "knob": short, "from": old_value, "to": cur, "applied_step": change_step,
               "pre_steps": [prev + 1, change_step], "post_until": None if nxt == float("inf") else nxt,
               "target_series": key, "step_ci": wall_ci, "exposure_ci": exp_ci, "verdict": verdict, "reason": reason,
               "n_post": n_post, "n_pre": int(w_pre.sum())}
        self._sv_last[short] = rec
        logger.info("optimizer.cbandit.step_verify", **{k: (_clean(v)) for k, v in rec.items()})
        if verdict in ("confirm", "revert") and reason in ("confirm", "revert", "own_exposure_rose") or verdict == "confirm":
            others = self._sv_others(short)
            better = verdict == "confirm"
            self._tried[(short, old_value, cur, others)] = "better" if better else "worse"
            self._tried[(short, cur, old_value, others)] = "worse" if better else "better"
        elif verdict == "revert":                      # futile: measured no better (not worse): still not worth the resource
            others = self._sv_others(short)
            self._tried[(short, old_value, cur, others)] = "worse"
        if verdict == "revert" and exp_pre.size:
            lb = sv.mean_lower_bound(exp_pre, wx_pre)
            hi = float(np.quantile(sv.mbb_means(exp_pre, wx_pre, sv.block_length(exp_pre)[0],
                                                np.random.default_rng(sv.SEED)), 0.975))
            self._refuted[(short, old_value, cur, self._sv_others(short))] = {"lo": lb, "hi": hi, "window": window}
        return verdict

    def _sv_futility_steps(self, exp_pre, wx_pre, delta: float, old_value, cur) -> Optional[int]:
        """Steps needed to detect the smallest effect this move was admitted on, at the planner's alpha and power.

        The predicted effect is the whole exposure a raise could recover; a raise from `old` to `new` of a pool
        recovers at most the share 1 - old/new of it (overlap form), which is the effect the move must deliver at
        least in part. n = blocks x L with blocks = (z sigma / delta_min)^2, sigma and L of the PRE exposure series.
        """
        old_f, new_f = _f(old_value), _f(cur)
        share = (1.0 - old_f / new_f) if (old_f > 0.0 and new_f > old_f) else 1.0
        dmin = delta * share
        if dmin <= 0.0 or len(exp_pre) < 2:
            return None
        L, sigma = sv.effective_blocks(exp_pre)
        z = self._z(1.0 - self._alpha / 2.0) + self._z(self._power)
        blocks = int(math.ceil((z * sigma / dmin) ** 2)) if sigma > 0 else 1
        return max(blocks * L, self._min_samples, 2 * L)

    def _sv_escalation_state(self, short: str, current, alts: list) -> Optional[Tuple[bool, str]]:
        """May a costlier actuator pass this coordinate? Decided on MEASURED verdicts of this configuration.

        A neighbouring value counts as tried only when a verified move to it (or from it) was measured with
        every other knob as it is now; arm samples pooled across other knobs' values do not count. Untried
        neighbour -> stay on this coordinate; measured better -> it should move there; every neighbour
        measured worse -> exhausted.
        """
        if not self._sv_enabled():
            return None
        others = self._sv_others(short)
        if short not in self._relocating:
            alts = [lv for lv in alts if _f(lv) > _f(current)]          # only raises are explored (see _pick)
        for lv in alts:
            st = self._tried.get((short, current, lv, others))
            if st is None:
                return False, "untried_neighbour:%s" % lv
            if st == "better":
                return False, "helpful_neighbour:%s" % lv
        return True, "neighbourhood_exhausted_no_help"

    def _sv_mechanism_confirmed(self, short: str, entry) -> bool:
        """The move's own targeted exposure fell, with a 95% interval that excludes zero, since its apply.

        The net step-time verdict can stay `collect` for a long time when the step carries noise from other
        families (an epoch-start loader stall is several times the effect of a comm move). Such a move no
        longer blocks moves of other knobs: it is verified (and, if the step time ends worse, reverted) in
        parallel, and a later move is verified against the segment after it.
        """
        rec = self._sv_last.get(short)
        if rec is None or len(entry) < 5 or rec.get("to") != entry[4]:
            return False
        ci = rec.get("exposure_ci")
        return bool(ci) and ci["hi"] < 0.0 and rec["verdict"] != "revert"

    def _sv_others(self, short: str) -> tuple:
        """The rest of the configuration: a refutation holds only while the other knobs are as they were."""
        cfg = self._physical_latest_cfg or self._short_cfg()
        return tuple(sorted((k, v) for k, v in cfg.items() if k != short))

    def _sv_refuted(self, short: str, cur, to, exposure_lb: Optional[float]) -> Optional[str]:
        """Veto text while the targeted exposure has not risen above what it was when this move failed."""
        key = (short, cur, to, self._sv_others(short))
        rec = self._refuted.get(key)
        if rec is None or exposure_lb is None:
            return None
        if exposure_lb > rec["hi"]:
            self._refuted.pop(key, None)                  # the family is now measurably more exposed
            return None
        return "refuted:measured_not_better_at_exposure<=%.4f" % rec["hi"]

    # ------------------------------------------------------------------ side effects
    def _sv_sync_watches(self) -> None:
        """Watch the side effects of the most recent applied change of every knob."""
        if self._verify != "per_step" or self._physical_latest_cfg is None:
            return
        for short, new, step in self._physical_changes:
            if step is None:
                continue
            meta = self._side_watch.get(short)
            if meta is not None and meta["step"] >= int(step):
                continue
            old = self._physical_change_old.get((short, step))
            unit = abs(_f(new) - _f(old)) if old is not None else 0.0
            if unit <= 0.0:
                continue
            cost_us = self._cost_by_step.get(step)
            self._side_watch[short] = {"step": int(step), "unit": unit,
                                       "apply_s": (cost_us / 1e6) if cost_us is not None else 0.0}

    def _sv_update_side_effects(self) -> None:
        """Refresh each watched move's measured side cost from its post-apply steps and windows."""
        if not self._sv_enabled():
            return
        bounds = self._sv_boundaries()
        for short, meta in list(self._side_watch.items()):
            step = meta["step"]
            nxt = min([b for b in bounds if b > step], default=float("inf"))
            wall_pre, _ = self._sv_segment("wall", max([b for b in bounds if b < step], default=0), step)
            wall_post, w_post = self._sv_segment("wall", step, nxt)
            if len(wall_pre) < 1 or len(wall_post) < 2:
                continue
            L = sv.block_length(wall_post)[0]
            wins = []
            rss0 = None
            for ss in self._tl:
                t = self._tele.get(ss.window)
                if t is None:
                    continue
                if ss.first > step and ss.first <= nxt:
                    wins.append({"n": ss.n, "mean_wall": float(np.average(ss.series["wall"], weights=ss.weights)),
                                 "reclaim": t.get("reclaim"), "anon_kb": t.get("anon_kb")})
                elif ss.last <= step and t.get("anon_kb") is not None:
                    rss0 = t["anon_kb"]                    # the last window entirely before the apply
                elif ss.first <= step < ss.last and rss0 is None and t.get("anon_kb") is not None:
                    rss0 = t["anon_kb"]
            cost = sv.measure_side_cost(wall_pre, wall_post, w_post, L, meta["apply_s"], meta["unit"], wins, rss0)
            if cost is not None:
                self._side[short] = cost
                logger.info("optimizer.cbandit.side_cost", knob=short, applied_step=step, **cost.as_dict())

    def _sv_gate(self, short: str, cur, value, delta, remaining, apply_s: Optional[float]) -> Optional[str]:
        """Side-effect pricing of the next move of `short` (see step_verify.price_next_move)."""
        if self._verify != "per_step" or short not in self._side:
            return None
        avail = None
        for w in sorted(self._tele, reverse=True):
            if self._tele[w].get("avail_kb") is not None:
                avail = float(self._tele[w]["avail_kb"])
                break
        unit = abs(_f(value) - _f(cur))
        return sv.price_next_move(self._side[short], unit, delta, remaining, apply_s or 0.0, avail)


def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _clean(v):
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, float):
        return round(v, 6)
    return v
