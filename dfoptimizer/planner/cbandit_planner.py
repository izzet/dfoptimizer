"""CoordinateBanditPlanner: cross-layer-directed online control (DFOPTIMIZER_ALGO=cbandit).

WHY THIS EXISTS, MEASURED RATHER THAN ASSUMED

Head-to-head on StorMer's two-domain workload, 240 steps (liveflow_shm README phase 31):

    rules            0.576 s/step   sd 0.052   always 2 plans
    clbo (unbounded) 0.819 s/step   sd 0.510   4-20 plans   best run 0.308
    clbo EXPLORE=3   3.163 s/step                           binding the budget made it WORSE

CLBO's best run beats the best rule run by 43%, so the surrogate genuinely finds configurations
the hand-tuned rules never reach. But it is 10x more variable, and the variance traces to the
acquisition rather than the model:

    cand = [c for c in self.grid if c not in cfgs]     # clbo.py StructuredCLBO.suggest
    return cand[argmax(expected_improvement(...))]

`suggest()` STRUCTURALLY CANNOT RETURN THE INCUMBENT -- every already-tried config is excluded, so
every window must deploy something new -- and `self.grid` is the full Cartesian product of all knob
levels (~2300 configs for num_workers x prefetch_factor x decode_threads), of which a 240-step run
samples under 2%.

That is offline BO used as an online controller. Offline BO minimises FINAL regret: sampling is
cheap, you pay once, and optimize() correctly returns cfgs[inc] at the end. Online control
minimises CUMULATIVE regret: every suggestion is deployed and paid for in wall time and there is no
end -- the deployed config IS the cost. clbo_planner.py's own comment records the consequence: the
controller "found rt=3 (25 ms) then abandoned it for rt=1 (77 ms) and ended there, netting 0%
against a static baseline despite 73% available headroom".

WHAT THIS DOES DIFFERENTLY, AND WHAT IT KEEPS

Keeps the contribution -- the cross-layer signal chooses WHICH knob to move (argmax over per-domain
times, a comparison, not a threshold). Changes the search:

  1. ONE KNOB AT A TIME, from the dominant domain. The surrogate CLBO already uses is factored per
     domain, so a joint grid buys variance and nothing else. This cuts the search space from the
     product of all levels to one knob's neighbourhood.
  2. LOCAL MOVES. Candidates are {stay, one level up, one level down}, not all 16 levels. Response
     curves in this system are monotone and saturating (measured: liveflow_shm phase 26, stall falls
     44-48% per doubling of read_threads then flattens), so a local step is nearly as good as a
     global jump and costs far fewer samples.
  3. STAY IS A LEGAL ARM. This is the structural fix. Exploitation must be reachable by the
     acquisition itself, not forced afterwards by a separate explore-budget -- which is what
     EXPLORE=3 did, and it froze on a badly-estimated incumbent at 3.163 s/step.
  4. COST-AWARE. A decode probe is nearly free (fork-inherited limit, no worker restart) while an
     input probe rebuilds the worker set at ~3.2% of runtime (phase 21). The bonus for trying an
     arm is discounted by what trying it costs, so exploration is biased toward the cheap domain.
     This is a genuinely cross-layer decision: a single-layer tuner cannot know that the same
     nominal move is free in one domain and expensive in another.

UCB rather than Expected Improvement, because UCB is a cumulative-regret algorithm and EI is not.

WHAT THIS IS NOT: it is not a safety guard. Nothing is vetoed for being harmful and no metric
outside the objective is consulted. It is a search policy.
"""

from __future__ import annotations

import math
import os
import uuid
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import structlog

from ..types import ActionPlan, DiagnosisFindingMsg, KnobDef
from .clbo import accept_decision
from .clbo_planner import _domain_of, _levels_for

logger = structlog.get_logger()

# Relative cost of PROBING a knob, as a fraction of an epoch, from liveflow_shm phase 21:
# applying a knob through the gated/shared-memory path cost +3.2% when it forced a worker-set
# rebuild and was free (-1.5%, within noise) when it did not. Decode moves a fork-inherited limit
# and never restarts workers; input knobs rebuild the DataLoader.
_PROBE_COST = {"decode": 0.002, "comm": 0.005, "input": 0.032, "compute": 0.032}


class CoordinateBanditPlanner:
    """Drop-in alternative to Planner / CLBOPlanner for DFOPTIMIZER_ALGO=cbandit."""

    def __init__(self):
        self.knobs: Dict[str, KnobDef] = {}
        self.current_values: Dict[str, object] = {}
        self._short: Dict[str, str] = {}
        self._target_fn: Dict[str, str] = {}
        self._domain: Dict[str, str] = {}
        self._levels: Dict[str, List] = {}
        # Bandit statistics per (knob_short, level): pulls and mean observed wall.
        self._stats: Dict[Tuple[str, object], Tuple[int, float]] = {}
        # Recent wall samples per arm. The running mean above is enough to RANK arms, but not to
        # ask whether two arms differ at all, which needs a spread. Bounded because recency is what
        # matters under drift, and because an unbounded history would let a stale regime outvote
        # the current one.
        self._samples: Dict[Tuple[str, object], Deque[float]] = {}
        self._cur_window: int = -1
        self._win_metrics: Dict[str, float] = {}
        self._planned_cfg: Dict[int, dict] = {}
        self._total_pulls = 0
        # UCB exploration weight. Scales the confidence radius; 0 makes this pure greedy descent.
        self._c = float(os.environ.get("DFOPTIMIZER_CBANDIT_C", "0.35"))
        # Set to 0 to disable cost weighting and isolate its contribution.
        self._cost_w = float(os.environ.get("DFOPTIMIZER_CBANDIT_COST_W", "1.0"))
        # ACCEPTANCE TEST (clbo.accept_decision). See _accept_move for why this exists.
        self._alpha = float(os.environ.get("DFOPTIMIZER_CBANDIT_ALPHA", "0.05"))
        self._effect_floor = float(os.environ.get("DFOPTIMIZER_CBANDIT_EFFECT_FLOOR", "0.02"))
        # Visits an arm gets before it must start justifying itself. Below this it is EXPLORED on
        # the acquisition's say-so; at or above it, a move must clear the test.
        self._min_samples = int(os.environ.get("DFOPTIMIZER_CBANDIT_MIN_SAMPLES", "3"))
        self._sample_window = int(os.environ.get("DFOPTIMIZER_CBANDIT_SAMPLE_WINDOW", "32"))
        # Set to 0 to disable the gate entirely and recover the previous behaviour exactly.
        self._accept_enabled = os.environ.get("DFOPTIMIZER_CBANDIT_ACCEPT", "1") != "0"

    # --- Planner-compatible registration (identical contract to Planner/CLBOPlanner) ---
    def register_knobs(self, knob_defs, current_values=None):
        current_values = current_values or {}
        items = knob_defs.items() if hasattr(knob_defs, "items") else ((kd.id, kd) for kd in knob_defs)
        for knob_id, kd in items:
            self.knobs[knob_id] = kd
            short = knob_id.split(".")[-1]
            self._short[knob_id] = short
            self._target_fn[knob_id] = kd.target_function
            cur = current_values.get(knob_id, kd.default)
            self.current_values[knob_id] = cur
            self._domain[short] = _domain_of(short)
            self._levels[short] = _levels_for(kd, cur)
        logger.info("optimizer.cbandit.registered",
                    knobs=list(self._short.values()),
                    domains={s: self._domain[s] for s in self._short.values()},
                    levels={s: self._levels[s] for s in self._short.values()})

    def _short_cfg(self) -> dict:
        return {self._short[k]: self.current_values[k] for k in self.knobs}

    # --- window accumulation, same shape as CLBOPlanner ---
    def _accumulate(self, finding: DiagnosisFindingMsg):
        for k, v in (finding.key_metrics or {}).items():
            self._win_metrics[k] = v

    def _finalize_window(self, window: int) -> Optional[dict]:
        m = self._win_metrics
        wall = m.get("active_time")
        if wall is None or wall <= 0:
            return None
        fetch = m.get("fetch_iter_time_proc_max", 0.0)
        comm = m.get("comm_time_proc_max", 0.0)
        return {
            "cfg": dict(self._planned_cfg.get(window, self._short_cfg())),
            "domains": {
                # Same decomposition CLBOPlanner uses, so the two planners see identical signal
                # and any difference between them is the SEARCH, not the observation.
                "input": m.get("reader_posix_time_proc_max", fetch),
                "decode": m.get("reader_decode_time_proc_max", 0.0),
                "comm": comm,
                "compute": max(m.get("compute_time_proc_max", 0.0) - comm, 0.0),
            },
            "wall": float(wall),
        }

    def _record(self, cfg: dict, wall: float):
        """Credit the observed wall to every knob level that produced it.

        Attributing one wall time to each knob's level independently is the factored assumption --
        the same one CLBO's FactoredSurrogate makes. It is wrong when knobs interact strongly, and
        right enough when each knob dominates its own domain, which is the case this system is
        built for.
        """
        for short, level in cfg.items():
            n, mean = self._stats.get((short, level), (0, 0.0))
            self._stats[(short, level)] = (n + 1, mean + (wall - mean) / (n + 1))
            buf = self._samples.get((short, level))
            if buf is None:
                buf = self._samples[(short, level)] = deque(maxlen=self._sample_window)
            buf.append(float(wall))
        self._total_pulls += 1

    def _neighbourhood(self, short: str, current) -> List:
        """{stay, one level down, one level up} -- a coordinate step, not a global jump."""
        levels = self._levels.get(short) or [current]
        if current in levels:
            i = levels.index(current)
        else:
            # Current value is off-grid (e.g. an app default between levels): snap to the nearest.
            i = min(range(len(levels)), key=lambda j: abs(_num(levels[j]) - _num(current)))
        out = [levels[i]]
        if i > 0:
            out.append(levels[i - 1])
        if i + 1 < len(levels):
            out.append(levels[i + 1])
        return out

    def _pick(self, short: str, current, domain: str):
        """UCB over the neighbourhood, minimising wall, discounted by probe cost.

        An unpulled arm gets the best observed wall as an optimistic prior rather than -inf, so a
        single unknown neighbour cannot hijack the whole run -- which is how CLBO ended up
        deploying num_workers=0.
        """
        cands = self._neighbourhood(short, current)
        seen = [m for (s, _l), (n, m) in self._stats.items() if s == short and n > 0]
        optimistic = min(seen) if seen else 0.0
        # PRIOR FOR AN UNTRIED ARM = the CURRENT arm's performance, not the best ever seen.
        #
        # Using the global best made every untried neighbour score below every measured arm, and
        # because moves are LOCAL each move exposes a fresh untried neighbour -- so the planner
        # walked forever. Measured on StorMer: 16 plans in a 240-step run at 1.479 s/step against
        # the rules' 2 plans, which is the cumulative-regret failure this planner exists to avoid,
        # reproduced by my own optimism term.
        #
        # A neighbouring level is a small perturbation of the current one, so the honest prior is
        # "about the same as now"; the confidence radius, not a fabricated mean, is what should pay
        # for exploring it.
        cur_n, cur_mean = self._stats.get((short, current), (0, 0.0))
        prior = cur_mean if cur_n > 0 else (optimistic if seen else 0.0)
        # The confidence radius is a FRACTION of the objective's scale. Written in absolute units
        # it would mean something different on a 1 s/step workload than on a 10 s/step one, and the
        # probe-cost term below (which is inherently relative) could never be commensurate with it.
        scale = optimistic if optimistic > 0 else 1.0
        best, best_score = current, float("inf")
        detail = []
        for lv in cands:
            n, mean = self._stats.get((short, lv), (0, prior))
            # An untried arm is treated as one pull's worth of uncertainty, not as unboundedly
            # promising. With the prior above this makes exploration decay as evidence accumulates
            # instead of persisting forever.
            bonus = self._c * scale * math.sqrt(math.log(max(self._total_pulls, 2)) / max(n, 1))
            # Cost of CHANGING to this arm. Staying is free by construction, which is what makes
            # exploitation reachable without a separate budget.
            cost = 0.0 if lv == current else self._cost_w * _PROBE_COST.get(domain, 0.03) * scale
            score = mean - bonus + cost
            detail.append({"level": lv, "n": n, "mean": round(mean, 4),
                           "bonus": round(bonus, 4), "cost": round(cost, 4), "score": round(score, 4)})
            if score < best_score:
                best, best_score = lv, score
        return best, detail

    def _accept_move(self, short: str, current, candidate):
        """Should the acquisition's pick actually be DEPLOYED? Returns (bool, why).

        THE FAILURE THIS FIXES, measured. cbandit issued 14 plans where the rules issued 2, and the
        cause was not the search: it was that UCB will separate two arms whose means differ only by
        noise, and then every window pays to move between them. The regression test that was
        supposed to catch this (C7, "it settles") used a FLAT objective, on which UCB cannot tell
        arms apart and quietly stops caring. The real objective is noisy, and on a noisy one UCB
        chases the noise. The test encoded the failure already seen instead of the one not yet seen.

        So search and COMMITMENT are separated. `_pick` stays a pure search policy and may propose
        anything; this decides whether the proposal has earned a deployment. The rule is
        clbo.accept_decision, which CLBO has always had and has only ever used in the OFFLINE
        optimize() loop, never on the online path: a candidate must be BOTH significantly faster
        (Welch p < alpha) AND meaningfully faster (relative gain >= effect_floor).

        EXPLORATION IS NOT BLOCKED, which is the part that has to be right. An arm with fewer than
        `min_samples` observations has no distribution to test, so Welch would return p=1.0 and
        reject it forever -- the controller would freeze on whatever it happened to try first,
        which is exactly the EXPLORE=3 failure (3.163 s/step, frozen on a badly-estimated
        incumbent). Under-sampled arms are therefore allowed through on the acquisition's say-so.
        That bounds free exploration to `min_samples` visits per arm, after which the arm is known
        and must win on evidence.

        Note what the constants are. alpha and effect_floor are a significance level and a minimum
        interesting effect: standard statistical parameters that carry their meaning across
        workloads. They are NOT of the same kind as `fetch_frac > 0.20` or
        `reader_posix_time_proc_max > 20`, which encode a guess about a particular system's
        behaviour. This replaces a domain magic number with a decision rule; it does not pretend to
        have no constants at all.
        """
        if not self._accept_enabled:
            return True, "gate_disabled"

        base = list(self._samples.get((short, current), ()))
        cand = list(self._samples.get((short, candidate), ()))

        if len(cand) < self._min_samples:
            return True, "explore_under_sampled"
        if len(base) < self._min_samples:
            # The arm we are ON is not yet characterised, so there is nothing to compare against.
            # Staying costs nothing and earns the sample that makes the next comparison possible.
            return False, "incumbent_under_sampled"

        verdict = accept_decision(base, cand,
                                  alpha=self._alpha, effect_floor=self._effect_floor)
        return verdict == "accept", verdict

    # --- Planner-compatible decision entry ---
    def process_finding(self, finding: DiagnosisFindingMsg) -> List[ActionPlan]:
        if not self.knobs:
            return []
        w = finding.window_index
        if self._cur_window == -1:
            self._cur_window = w
            self._accumulate(finding)
            return []
        if w == self._cur_window:
            self._accumulate(finding)
            return []

        completed = self._cur_window
        obs = self._finalize_window(completed)
        plans: List[ActionPlan] = []
        if obs is not None:
            self._record(obs["cfg"], obs["wall"])

            # 1. CROSS-LAYER DIRECTION: the dominant domain by measured time. A comparison, not a
            #    threshold -- this is the part the paper's contribution rests on.
            doms = obs["domains"]
            actionable = {d for d in self._domain.values()}
            # COST-DISCOUNTED DOMAIN RANKING. Raw argmax picks the domain holding the most time;
            # this subtracts what PROBING that domain costs, in the same units. When one domain
            # clearly dominates, the discount is irrelevant and the ranking is unchanged. When two
            # domains are close, the cheaper-to-probe one wins -- and THAT is a decision no
            # single-layer tuner can make, because it requires knowing that the same nominal move
            # is free in one layer (fork-inherited limit, no restart) and forks a whole worker set
            # in another. Threshold-free: there is no "close enough" constant, the subtraction
            # decides.
            #
            # Placed here rather than inside _pick() on purpose. Within a single knob the probe
            # cost (3.2% of wall) is correctly swamped by the value of resolving an untried arm --
            # 3.2% is cheap against unknown headroom. Across domains it is the deciding term.
            wall = obs["wall"]
            ranked = sorted(((v - self._cost_w * _PROBE_COST.get(d, 0.03) * wall, d)
                             for d, v in doms.items() if d in actionable), reverse=True)
            if ranked:
                _v, dom = ranked[0]
                # 2. ONE KNOB, from that domain. If the domain has several, take the least-pulled
                #    so attention rotates instead of fixating.
                owned = [s for s in self._short.values() if self._domain[s] == dom]
                if owned:
                    pulls = {s: sum(n for (ss, _l), (n, _m) in self._stats.items() if ss == s)
                             for s in owned}
                    short = min(owned, key=lambda s: pulls[s])
                    cur = self._short_cfg()[short]
                    # 3. UCB over {stay, down, up}, cost-discounted.
                    choice, detail = self._pick(short, cur, dom)
                    # 4. ACCEPTANCE. The search proposes; this decides whether the proposal has
                    #    earned a deployment. Without it UCB pays to move between arms that differ
                    #    only by noise -- 14 plans against the rules' 2.
                    accepted, why = (True, "no_change")
                    if choice != cur:
                        accepted, why = self._accept_move(short, cur, choice)
                        if not accepted:
                            choice = cur
                    logger.info("optimizer.cbandit.decision",
                                window=completed, dominant_domain=dom,
                                domain_times={k: round(float(v), 4) for k, v in doms.items()},
                                knob=short, current=cur, chose=choice,
                                accept=why,
                                n_current=len(self._samples.get((short, cur), ())),
                                wall=round(obs["wall"], 4), candidates=detail)
                    nxt = dict(self._short_cfg())
                    nxt[short] = choice
                    self._planned_cfg[w + 1] = dict(nxt)
                    if choice != cur:
                        plans = self._plans_for(nxt, finding, dom)
                    # choice == cur is a real decision (exploit), and it emits no plan: the
                    # application keeps running what it is running, at zero actuation cost.

        self._cur_window = w
        self._win_metrics = {}
        self._accumulate(finding)
        return plans

    def _plans_for(self, next_short_cfg: dict, finding: DiagnosisFindingMsg, dom: str) -> List[ActionPlan]:
        plans = []
        for kid, kd in self.knobs.items():
            short = self._short[kid]
            new_v = next_short_cfg.get(short)
            old_v = self.current_values[kid]
            if new_v is None or new_v == old_v:
                continue
            new_v = kd.clamp(new_v)
            if new_v == old_v:
                continue
            self.current_values[kid] = new_v
            plans.append(ActionPlan(
                plan_id=str(uuid.uuid4()),
                knob_id=kid,
                target_function=self._target_fn.get(kid, ""),
                old_value=old_v,
                new_value=new_v,
                apply_when="window_boundary",
                rationale=f"cbandit UCB coordinate step ({dom} domain, cost-weighted)",
                finding_type=finding.finding_type,
                severity=finding.severity_score,
                opportunity_tag=(finding.opportunity_tags or [""])[0],
                window_index=finding.window_index,
            ))
        return plans


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0
