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
  4. COST-AWARE. Each candidate value carries its measured KnobDef cost in seconds. Acquisition
     compares that cost with the candidate's expected wall-time gain over the observed windows or
     epochs left, and expensive actuators wait until cheaper coordinates in the same domain have
     been tried without benefit. No fixed run horizon or per-domain proxy cost is used.

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
from .clbo_planner import _domain_of, _levels_for, domains_from_metrics

logger = structlog.get_logger()

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
        # Cheapest-first escalation within the dominant domain. 0 restores the previous rule (one
        # knob per window, chosen by least pulls -- which, because _record credits every knob every
        # window, was always the first-registered knob and never any other). See process_finding.
        self._escalate = os.environ.get("DFOPTIMIZER_CBANDIT_ESCALATE", "1") != "0"
        # THE ONLY NEW CONSTANT the barrier-attribution rule introduces, and it is a per-cluster
        # MEASUREMENT with provenance in liveflow_shm/tests/cluster_*.env, not a tuned number. Unset
        # is a supported state: the rule still redirects a comm-dominant window to the storage
        # candidates, it just cannot order them by headroom and says so in the decision record.
        def _envf(name):
            try:
                return float(os.environ.get(name, "") or 0.0)
            except ValueError:
                return 0.0

        # TWO measured constants and the run's node count, because a single-node ceiling is the
        # WRONG comparison at scale and would have made this rule systematically say the opposite
        # of the truth. Measured offline on the t47_cadence32 traces: at 32 nodes the median node
        # read 0.694 GB/s, so against Corona's single-node reader ceiling (about 7 GB/s) every one
        # of 2016 node-windows looks like it has headroom, when in fact each node is reading its
        # full share of a filesystem the cluster has already saturated. Lustre's OSTs are a fixed
        # shared resource: PAPER-SPINE measured the aggregate flat at 17.4 to 18.1 GB/s across 32
        # and 56 nodes while the per-node share fell 0.545 -> 0.323, which is what a hard shared
        # limit looks like. The ceiling that applies to a node is therefore the smaller of what one
        # node can pull alone and what the cluster's aggregate leaves it.
        self._node_ceiling_gbs = _envf("LUSTRE_NODE_READ_CEILING_GBS")
        self._agg_ceiling_gbs = _envf("LUSTRE_AGGREGATE_READ_CEILING_GBS")
        try:
            self._nodes = int(os.environ.get("LIVEFLOW_NODES", "") or 0)
        except ValueError:
            self._nodes = 0
        share = (self._agg_ceiling_gbs / self._nodes
                 if self._agg_ceiling_gbs > 0 and self._nodes > 0 else 0.0)
        cands = [c for c in (self._node_ceiling_gbs, share) if c > 0]
        self._lustre_ceiling_gbs = min(cands) if cands else 0.0
        # Consecutive application refusals after which a knob is excluded from selection. See
        # apply_ack: an unappliable knob is never sampled, so it is permanently "under-sampled",
        # so the acceptance gate never stops it, so it absorbs the escalation forever.
        self._retire_after = int(os.environ.get("DFOPTIMIZER_CBANDIT_RETIRE_AFTER", "3"))
        self._rejections: Dict[str, int] = {}

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

    def apply_ack(self, plan_id: str = "", knob_id: str = "", status: str = "unknown",
                  old_value=None, new_value=None, window_index: int = -1,
                  target_nodes=None):
        """Process an ack from the application.

        THE BUG THIS FIXES, and it was loud in the log while being invisible in the results.
        `CoordinateBanditPlanner` simply did not define this method, so `Optimizer._ack_loop` raised
        AttributeError on EVERY ack -- 27,654 times in one five-minute two-node run -- caught the
        exception, and immediately pulled the next one. Three consequences, none of which shows up
        as a failed run:

          1. the planner never learned whether anything it planned was actually applied,
          2. a REJECTED plan left `current_values` holding the value the application had refused,
             so the planner believed a knob had moved when it had not and never proposed it again
             (fatal for `data_source`: one rejection retires the staging actuator for the run),
          3. the ack thread spun on the error path, burning a core on the agent node.

        `Planner` has always had this method, so the bug appeared only when the algorithm was
        switched -- which is the configuration every evaluated arm now runs.

        Reverting on a non-applied status is the whole point. `_pick` is driven by
        `self.current_values`, so a stale entry there is not a cosmetic mismatch, it removes the
        arm from the search.
        """
        logger.info("optimizer.cbandit.ack", plan_id=plan_id, knob_id=knob_id, status=status,
                    old_value=old_value, new_value=new_value, window_index=window_index)
        if knob_id not in self.knobs:
            return
        if status == "applied":
            if new_value is not None:
                self.current_values[knob_id] = self.knobs[knob_id].clamp(new_value)
            return
        # rejected / failed / anything else: the application is still on the old value, and our
        # model has to say so or the arm is lost for the rest of the run.
        revert = old_value if old_value is not None else self.knobs[knob_id].default
        self.current_values[knob_id] = self.knobs[knob_id].clamp(revert)
        short = self._short.get(knob_id, knob_id)
        self._rejections[short] = self._rejections.get(short, 0) + 1
        logger.info("optimizer.cbandit.ack.reverted", knob_id=knob_id, status=status,
                    back_to=self.current_values[knob_id], rejections=self._rejections[short])
        # CAPABILITY IS LEARNED, NOT ASSERTED. A knob the application cannot actually apply is
        # indistinguishable, from here, from one it can: the plan is published, the ack comes back
        # "rejected", and the planner proposes it again next window. Because a never-applied arm
        # never accumulates samples it is permanently "under-sampled", so the acceptance gate lets
        # it through every single time and it ABSORBS THE ESCALATION -- the cheapest-first walk stops
        # at it and never reaches the knobs behind it.
        #
        # Measured on Corona, 2 nodes, cosmoflow: `transfer_size` is registered but is not a
        # parameter of `make_loader` (open bug #30), so `@tunable` acks it `knob_not_in_kwargs`.
        # In one epoch it took 144,048 rejections and 0 applications while `data_source`, ordered
        # behind it, was never once reached. Retiring an arm the application has refused
        # `_retire_after` times is the same move as PAPER-SPINE's "capability is LEARNED by probing
        # rather than asserted", applied to appliability instead of to benefit.
        if self._rejections[short] == self._retire_after:
            logger.warning("optimizer.cbandit.knob.retired", knob=short, knob_id=knob_id,
                           rejections=self._rejections[short],
                           reason="the application refused this knob repeatedly; it is excluded "
                                  "from selection so it cannot absorb the escalation")

    # --- window accumulation, same shape as CLBOPlanner ---
    def _accumulate(self, finding: DiagnosisFindingMsg):
        for k, v in (finding.key_metrics or {}).items():
            self._win_metrics[k] = v

    def _finalize_window(self, window: int) -> Optional[dict]:
        m = self._win_metrics
        wall = m.get("active_time")
        if wall is None or wall <= 0:
            return None
        # ONE shared definition, so the two planners cannot drift and any difference between them
        # is the SEARCH, not the observation. Both used to build this inline and both had the same
        # bug: reader_posix as the domain time instead of the exposed fetch. See
        # clbo_planner.domains_from_metrics.
        domains = domains_from_metrics(m)
        # THE PART OF THE STEP WE CANNOT NAME.
        #
        # The domain times above do not have to add up to the wall, and on DLIO they badly do not:
        # measured at 1 node, wall 6.1725 s = compute 2.5443 + fetch_iter 3.6282, of which
        # reader_posix (the "input" domain) accounts for only 0.6064. Three seconds -- 49% of the
        # step -- sat inside fetch attributed to nothing, and the planner still confidently
        # concluded "compute dominates" by comparing 2.54 against 0.61.
        #
        # Under the old threshold rules this did not decide anything: they fired on within-layer
        # ratios and never compared one domain against another. A domain-argmax planner makes
        # attribution the whole decision, so the gap has to be visible rather than silently
        # dropped. Carrying it as a domain with NO KNOBS is what makes it load-bearing: if it wins
        # the argmax, there is nothing to move and the planner declines. That is a comparison, not
        # a threshold -- no constant says how much unexplained time is "too much".
        unattributed = max(float(wall) - sum(domains.values()), 0.0)
        remaining, remaining_unit = self._remaining_run(m)
        return {
            "cfg": dict(self._planned_cfg.get(window, self._short_cfg())),
            "domains": domains,
            "unattributed": unattributed,
            "wall": float(wall),
            "remaining": remaining,
            "remaining_unit": remaining_unit,
            # The storage layer's own two numbers for this node and window, carried so the
            # barrier-attribution rule can look one layer down without a second metrics read.
            "node_read_bytes": float(m.get("node_read_bytes", 0.0) or 0.0),
            "node_read_busy_s": float(m.get("node_read_busy_s", 0.0) or 0.0),
        }

    @staticmethod
    def _remaining_run(metrics: dict) -> Tuple[Optional[float], Optional[str]]:
        """Return the remaining-run multiplier reported to the agent, without a fallback.

        A window count and an epoch count are both valid because the analyzer may publish either
        kind of aggregate. Inventing a default horizon would make the same actuator payable or
        unpayable based on a planner constant rather than the run that is actually left.
        """
        for key, unit in (("windows_remaining", "windows"),
                          ("remaining_windows", "windows"),
                          ("epochs_remaining", "epochs"),
                          ("remaining_epochs", "epochs")):
            value = metrics.get(key)
            if value is None:
                continue
            try:
                return max(float(value), 0.0), unit
            except (TypeError, ValueError):
                continue
        return None, None

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

    def _pick(self, short: str, current, domain: str, exposed_s: Optional[float] = None,
              remaining: Optional[float] = None):
        """UCB over the neighbourhood in remaining-run wall seconds.

        An unpulled arm gets the best observed wall as an optimistic prior rather than -inf, so a
        single unknown neighbour cannot hijack the whole run -- which is how CLBO ended up
        deploying num_workers=0.

        Candidate cost is the measured cost of the VALUE being requested. The candidate's
        optimistic per-window gain is bounded by the observed dominant-domain exposure, then
        multiplied by the windows or epochs left. A positive-cost move with no reported remaining
        horizon is not payable. COST_W=0 removes the cost and payback terms for the ablation.
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
        # The confidence radius is a FRACTION of the objective's scale. The acquisition below then
        # converts both predicted wall and actuation cost to seconds over the remaining run.
        scale = optimistic if optimistic > 0 else 1.0
        best, best_score = current, float("inf")
        detail = []
        for lv in cands:
            n, mean = self._stats.get((short, lv), (0, prior))
            # An untried arm is treated as one pull's worth of uncertainty, not as unboundedly
            # promising. With the prior above this makes exploration decay as evidence accumulates
            # instead of persisting forever.
            bonus = self._c * scale * math.sqrt(math.log(max(self._total_pulls, 2)) / max(n, 1))
            if lv == current and remaining is not None and self._cost_w != 0.0:
                # The incumbent is the reference, not an exploration target. Giving it an
                # optimism bonus subtracts that bonus from every candidate's apparent gain and can
                # make a move fail the payback comparison after its own gain already cleared cost.
                # Preserve the legacy UCB exactly when no priced remaining-run decision is being
                # made, including COST_W=0 and knobs declared with zero cost.
                bonus = 0.0
            cost_s = 0.0 if lv == current else self._move_cost(short, lv)
            weighted_cost_s = self._cost_w * cost_s
            gain_per_window_s = max(cur_mean - (mean - bonus), 0.0)
            if exposed_s is not None:
                gain_per_window_s = min(gain_per_window_s, max(float(exposed_s), 0.0))
            gain_s = (gain_per_window_s * remaining
                      if remaining is not None else None)
            payable = (lv == current or self._cost_w == 0.0 or cost_s <= 0.0
                       or (gain_s is not None and gain_s > weighted_cost_s))

            # Multiplying the predicted wall by the observed remaining count puts it in the same
            # units as cost_for(): wall seconds over the rest of this run. With no horizon, only a
            # zero-cost move has a defensible score.
            if remaining is not None:
                score = (mean - bonus) * remaining + weighted_cost_s
            elif weighted_cost_s <= 0.0:
                score = mean - bonus
            else:
                score = float("inf")
            if not payable:
                score = float("inf")
            detail.append({"level": lv, "n": n, "mean": round(mean, 4),
                           "bonus": round(bonus, 4), "gain_s": _rounded(gain_s),
                           "cost_s": round(cost_s, 4), "payable": payable,
                           "score": _rounded(score)})
            if score < best_score:
                best, best_score = lv, score
        return best, detail

    def _move_cost(self, short: str, value) -> float:
        """Seconds to move `short` to `value`, from the knob's own declaration. 0 for no move.

        Reads `KnobDef.cost_seconds`, which may be a scalar or a per-VALUE dict. Per value matters:
        `data_source` costs 0 for "pfs", 465 s for "local" and ~4.4 h for "restriped", and one
        number per knob would make the restripe look as cheap as the stage.
        """
        cfg = self._short_cfg()
        if short in cfg and cfg[short] == value:
            return 0.0
        for kid, kd in self.knobs.items():
            if self._short.get(kid) == short:
                try:
                    return float(kd.cost_for(value))
                except Exception:                     # noqa: BLE001 - a cost must never break a decision
                    return 0.0
        return 0.0

    def _node_read_bandwidth(self, obs: dict) -> Optional[float]:
        """This node's observed read bandwidth for the window, GB/s, or None if it did not read.

        Bytes over the reader's BUSY time, not over the window wall: the question the ceiling
        answers is "when this node was reading, was it reading as fast as the filesystem can serve
        it", and dividing by wall would answer a different question and would fall with any idle
        time. Busy time of zero means the node did not read in this window, which is an absence and
        is returned as such rather than as a bandwidth of zero.
        """
        busy = obs.get("node_read_busy_s") or 0.0
        by = obs.get("node_read_bytes") or 0.0
        if busy <= 0.0 or by <= 0.0:
            return None
        return (by / 1e9) / busy

    def _barrier_is_storage_suspect(self, dom: str, actionable: set) -> bool:
        """Option 1, cross-layer PER NODE (user decision 2026-09-03).

        At 32 nodes the per-step allreduce barrier carries the storage tail of whichever rank is
        slowest, so `comm` wins the argmax on every node while no registered knob owns comm and
        every node declines (T3-0 section 4k: comm 73.9% of the step, 1884 of 1984 windows
        declined). The measured shape rules out the obvious alternative: the straggler ROTATES,
        every one of 256 ranks was last at least once per epoch and the per-node ordering does not
        survive an epoch (Spearman 0.076, section 4l), so there is no persistent culprit to find and
        nothing to be gained by exchanging waits between nodes. Option 2 was rejected for exactly
        that reason, and because it would add a node-to-node hop the SHM design does not have.

        What is left is a LOCAL inference: a barrier this node did not cause is a barrier this node
        waited at, and the thing to check is whether this node's own storage layer has headroom.
        This predicate is the trigger only; the comparison against the ceiling is separate, so a
        run with no ceiling still gets the redirection and simply cannot order the candidates by it.
        """
        return dom == "comm" and "comm" not in actionable

    def _storage_first_order(self, owned: List[str], bw: Optional[float]) -> Tuple[List[str], str]:
        """Order the storage candidates by what the ceiling comparison says is possible.

        Below the ceiling there is bandwidth this node is not using, so more parallelism can take
        it: keep the ordinary cheapest-first escalation, which starts at the free knobs. At or above
        the ceiling there is nothing left to take and asking for more readers cannot help however
        cheap it is, so the knobs that MOVE THE DATA come first (`data_source` stages the shard off
        the shared filesystem, `transfer_size` changes what is asked of it per call).

        The boundary is the measured ceiling itself. There is no fraction of it, no margin and no
        tuned constant: `observed >= ceiling` is the whole test, which is what keeps this inside
        AGENTS.md section 2's rule against hand-set domain thresholds. With no ceiling configured or
        no read observed, the order is unchanged and the reason says so.
        """
        base = self._escalation_order(owned)
        if bw is None:
            return base, "no_read_observed"
        ceiling = self._lustre_ceiling_gbs
        if not ceiling:
            return base, "no_ceiling_configured"
        if bw < ceiling:
            return base, "headroom"
        movers = [s for s in base if s in ("data_source", "transfer_size")]
        rest = [s for s in base if s not in ("data_source", "transfer_size")]
        return movers + rest, "at_ceiling"

    def _escalation_order(self, owned: List[str]) -> List[str]:
        """The domain's knobs, cheapest available move first.

        A knob's price for this window is the cost of the CHEAPEST move it could make away from
        where it is -- not the cost of its most expensive value -- because that is what asking it
        actually risks. Ties keep registration order, so a domain whose knobs all cost the same
        behaves exactly as it did before, which is every knob set that predates `cost_seconds`.
        """
        live = [s for s in owned if self._rejections.get(s, 0) < self._retire_after]
        if not live:                    # every knob refused: keep the old list so behaviour is
            live = list(owned)          # unchanged rather than the domain silently going empty
        if not self._escalate:
            return live
        cfg = self._short_cfg()

        def price(short):
            cur = cfg.get(short)
            alts = [lv for lv in self._neighbourhood(short, cur) if lv != cur]
            return self._cost_w * min((self._move_cost(short, lv) for lv in alts),
                                      default=0.0)

        return sorted(live, key=lambda s: (price(s), owned.index(s)))

    def _escalation_state(self, short: str, current) -> Tuple[bool, str]:
        """Whether a costlier actuator may pass this cheaper coordinate.

        Every neighbouring value must have enough observations for the same CLBO acceptance test
        used at commitment time, and none may help. A coordinate with no neighbouring value is
        exhausted by construction. Cost weighting disabled means there is no cheaper tier to
        protect, so this rule does not retain a hidden cost preference in the ablation.
        """
        if self._cost_w == 0.0:
            return True, "cost_weighting_disabled"
        alts = [lv for lv in self._neighbourhood(short, current) if lv != current]
        if not alts:
            return True, "neighbourhood_exhausted"
        base = list(self._samples.get((short, current), ()))
        if len(base) < self._min_samples:
            return False, "incumbent_under_sampled"
        for lv in alts:
            cand = list(self._samples.get((short, lv), ()))
            if len(cand) < self._min_samples:
                return False, f"untried_neighbour:{lv}"
            verdict = accept_decision(base, cand,
                                      alpha=self._alpha, effect_floor=self._effect_floor)
            if verdict == "accept":
                return False, f"helpful_neighbour:{lv}"
        return True, "neighbourhood_exhausted_no_help"

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
            wall = obs["wall"]
            # The unexplained remainder COMPETES in the argmax. It owns no knob, so if it is the
            # largest share of the step the planner has nothing to move and declines -- which is
            # the correct response to "most of this step is something I cannot name". Actuation
            # cost ranks candidates only after raw exposed time has named the dominant domain.
            ranked = sorted(((v, d) for d, v in doms.items()), reverse=True)
            unattr = obs.get("unattributed", 0.0)
            if unattr > 0:
                ranked = sorted(ranked + [(unattr, "unattributed")], reverse=True)
            # BARRIER ATTRIBUTION, option 1 (user decision 2026-09-03; T3-0 section 4m). A
            # comm-dominant window with no comm knob is not "nothing to do": on a bulk-synchronous
            # step the barrier is where the slowest rank's storage tail is paid by everyone else,
            # so the window is STORAGE-SUSPECT and this node's own storage candidates are admitted.
            # The redirection is recorded on the decision, never silent, because a decision taken
            # about a domain other than the dominant one has to be visible as such.
            storage_suspect = False
            suspect_bw = None
            suspect_reason = None
            if ranked and self._barrier_is_storage_suspect(ranked[0][1], actionable):
                if any(self._domain[s] == "input" for s in self._short.values()):
                    storage_suspect = True
                    suspect_bw = self._node_read_bandwidth(obs)
                    ranked = [(doms.get("input", 0.0), "input")] + [
                        r for r in ranked if r[1] != "input"]

            if ranked and ranked[0][1] not in actionable:
                _bw = self._node_read_bandwidth(obs)
                logger.info(
                    "optimizer.cbandit.decision", window=completed, decision="decline",
                    reason="dominant share of the step is unattributed",
                    dominant=ranked[0][1], gain_s=0.0, cost_s=0.0,
                    horizon_count=obs.get("remaining"),
                    horizon_unit=obs.get("remaining_unit"), escalation_state="not_actionable",
                    unattributed=round(float(unattr), 4),
                    unattributed_frac=round(float(unattr) / wall, 3) if wall else None,
                    # On the DECLINE record too, not only on a move. A cell where every window
                    # declines -- 2 nodes, where compute is a sleep and owns no knob -- would
                    # otherwise never show these, so the hop that carries them could not be
                    # observed on the one shape cheap enough to verify it on. A measurement that
                    # is only visible when the planner acts is not verifiable.
                    node_read_gbs=(round(_bw, 4) if _bw is not None else None),
                    node_read_busy_s=round(float(obs.get("node_read_busy_s") or 0.0), 4),
                    node_read_bytes=float(obs.get("node_read_bytes") or 0.0),
                    ceiling_gbs=self._lustre_ceiling_gbs or None,
                    storage_suspect=storage_suspect,
                    domain_times={k: round(float(v), 4) for k, v in doms.items()},
                    wall=round(wall, 4))
                ranked = []
            if ranked:
                _v, dom = ranked[0]
                # 2. ONE KNOB, from that domain, CHEAPEST FIRST.
                #
                #    THE BUG THIS REPLACES, and it made the expensive actuator unreachable rather
                #    than merely unlikely. The rule was "take the least-pulled so attention rotates
                #    instead of fixating", implemented as argmin over
                #        pulls[s] = sum of _stats counts across s's levels.
                #    But `_record` credits EVERY knob in the config on EVERY completed window, so
                #    all knobs in a domain always carry the SAME pull count, `min` over equal keys
                #    returns the first item in registration order, and that knob is selected for
                #    every window of the run. On DLIO the input domain owns prefetch_size,
                #    read_threads, data_source and transfer_size, so prefetch_size was chosen every
                #    time and `data_source` -- the staging actuator, the only one that can move a
                #    storage-bound cell -- was never once considered. Nothing reported this: the
                #    loop ran, decided, actuated and never mentioned the knob it could not see.
                #
                #    The replacement is not a better tie-break, it is the rule this system already
                #    claims to follow. `KnobDef.cost_for(value)` has carried per-value costs since
                #    the ActionDef work (loader rebuild 0.21 s, stage 465 s, restripe 4.4 h) and no
                #    planner read it. Order the domain's knobs by what their cheapest available move
                #    COSTS, try them in that order, and stop at the first one that both wants to
                #    move and clears the acceptance gate. A knob that says "stay" has nothing to
                #    offer this window, so the search escalates to the next-cheapest instead of
                #    ending the window having asked one knob.
                #
                #    That produces the documented escalation -- read_threads, then prefetch, then
                #    staging -- as a CONSEQUENCE of measurement rather than as a preference: the
                #    expensive actuator is reached exactly when the cheap ones have converged, and
                #    the acceptance gate (not a threshold) decides what "converged" means.
                #
                #    DFOPTIMIZER_CBANDIT_ESCALATE=0 restores the old single-knob-per-window rule.
                owned = [s for s in self._short.values() if self._domain[s] == dom]
                if owned:
                    if storage_suspect:
                        order, suspect_reason = self._storage_first_order(owned, suspect_bw)
                        logger.info(
                            "optimizer.cbandit.storage_suspect", window=completed,
                            reason="comm dominates and no registered knob owns comm",
                            comm_s=round(float(doms.get("comm", 0.0)), 4),
                            comm_frac=round(float(doms.get("comm", 0.0)) / wall, 3) if wall else None,
                            node_read_gbs=round(suspect_bw, 4) if suspect_bw is not None else None,
                            ceiling_gbs=self._lustre_ceiling_gbs or None,
                            headroom=suspect_reason, order=order)
                    else:
                        order = self._escalation_order(owned)
                    cfg_now = self._short_cfg()
                    short, cur, choice, detail, why = None, None, None, [], "no_change"
                    tried = []
                    for cand_short in order:
                        cand_cur = cfg_now[cand_short]
                        cand_choice, cand_detail = self._pick(
                            cand_short, cand_cur, dom, exposed_s=doms[dom],
                            remaining=obs.get("remaining"))
                        if cand_choice == cand_cur:
                            may_escalate, esc_state = self._escalation_state(cand_short, cand_cur)
                            tried.append({"knob": cand_short, "verdict": "stay",
                                          "escalation": esc_state})
                            if short is None:      # remember the cheapest, for the log and the ack
                                short, cur, choice, detail = (cand_short, cand_cur,
                                                              cand_choice, cand_detail)
                                why = esc_state
                            # Staying alone is not evidence of exhaustion. A costlier actuator may
                            # pass only after all cheaper neighbours were sampled and rejected by
                            # accept_decision, or after the coordinate reaches a boundary.
                            if not self._escalate or not may_escalate:
                                break
                            continue
                        accepted, cand_why = self._accept_move(cand_short, cand_cur, cand_choice)
                        may_escalate, esc_state = self._escalation_state(cand_short, cand_cur)
                        tried.append({"knob": cand_short, "verdict": cand_why,
                                      "to": cand_choice,
                                      "cost_s": round(self._move_cost(cand_short, cand_choice), 3),
                                      "escalation": esc_state})
                        if accepted:
                            short, cur, choice, detail, why = (cand_short, cand_cur, cand_choice,
                                                               cand_detail, cand_why)
                            break
                        if short is None:
                            short, cur, choice, detail, why = (cand_short, cand_cur, cand_cur,
                                                               cand_detail, cand_why)
                        if not self._escalate or not may_escalate:
                            break
                    if short is None:              # defensive: `owned` was non-empty, so unreachable
                        short = order[0]
                        cur = choice = cfg_now[short]
                    chosen_detail = next((d for d in detail if d["level"] == choice), {})
                    gain_s = chosen_detail.get("gain_s") if choice != cur else 0.0
                    cost_s = self._move_cost(short, choice)
                    horizon_s = (float(doms[dom]) * float(obs["remaining"])
                                 if obs.get("remaining") is not None else None)
                    # One rationale record per decision. Gain and cost are remaining-run seconds.
                    logger.info("optimizer.cbandit.decision",
                                window=completed, decision=("move" if choice != cur else "stay"),
                                dominant_domain=dom,
                                domain_times={k: round(float(v), 4) for k, v in doms.items()},
                                knob=short, current=cur, chose=choice,
                                accept=why, escalation=tried,
                                gain_s=gain_s, cost_s=round(cost_s, 3),
                                horizon_s=_rounded(horizon_s),
                                horizon_count=obs.get("remaining"),
                                horizon_unit=obs.get("remaining_unit"),
                                escalation_state=tried,
                                # Recorded on EVERY decision so the release rule can be counted off
                                # the log: a window where the dominant domain was comm, no knob
                                # owned it, and the storage candidates were admitted instead.
                                storage_suspect=storage_suspect,
                                storage_headroom=suspect_reason,
                                node_read_gbs=(round(suspect_bw, 4)
                                               if suspect_bw is not None else None),
                                ceiling_gbs=self._lustre_ceiling_gbs or None,
                                unattributed=round(float(unattr), 4),
                                n_current=len(self._samples.get((short, cur), ())),
                                wall=round(obs["wall"], 4), candidates=detail)
                    nxt = dict(cfg_now)
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

    @staticmethod
    def _apply_when(kd) -> str:
        """Where this knob's plan may be applied, taken from the knob's own definition.

        THE BUG THIS FIXES, and it is silent. This was the literal string "window_boundary" for
        every knob, so a knob that declares an epoch boundary got a window-boundary plan anyway and
        the declaration did nothing. The rules Planner has always honoured it (planner.py:548,
        `apply_when=response.apply_when`); cbandit and CLBO did not, so switching planner silently
        changed WHERE an actuator fires.

        It matters for exactly one knob so far, and there it decides the outcome rather than the
        timing. `data_source` declares `epoch_boundary` because the horizon its abort test is judged
        against (`gain_per_epoch * epochs_remaining`) is only measurable once an epoch has finished:
        applied mid-epoch-1 the gain is 0, the horizon is 0, and the stage aborts after one chunk
        having paid for it. A window-boundary plan for an epoch-boundary knob is therefore not a
        late apply, it is a guaranteed abort that reports itself as a considered decision.

        Falls back to "window_boundary", so every knob that declares nothing behaves exactly as
        before and this cannot regress the parallelism knobs.
        """
        for resp in (getattr(kd, "responds_to", None) or {}).values():
            aw = getattr(resp, "apply_when", None)
            if aw:
                return aw
        return "window_boundary"

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
                apply_when=self._apply_when(kd),
                rationale=f"cbandit UCB coordinate step ({dom} domain, cost-weighted)",
                finding_type=finding.finding_type,
                severity=finding.severity_score,
                opportunity_tag=(finding.opportunity_tags or [""])[0],
                window_index=finding.window_index,
                requires_unanimity=bool(getattr(kd, "requires_unanimity", False)),
            ))
        return plans


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _rounded(value):
    if value is None or not math.isfinite(float(value)):
        return value
    return round(float(value), 4)
