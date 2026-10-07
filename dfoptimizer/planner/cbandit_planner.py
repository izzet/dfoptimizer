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
import random
import dataclasses as _dc
import time as _time
import statistics
import os
import uuid
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import structlog

from ..types import ActionPlan, DiagnosisFindingMsg, KnobDef
from .clbo import accept_decision
from .clbo_planner import _domain_of, _levels_for, domains_from_metrics
from . import sizing
from .step_planner import StepVerifyMixin

logger = structlog.get_logger()

class CoordinateBanditPlanner(StepVerifyMixin):
    """Drop-in alternative to Planner / CLBOPlanner for DFOPTIMIZER_ALGO=cbandit."""

    def __init__(self):
        self.knobs: Dict[str, KnobDef] = {}
        self.current_values: Dict[str, object] = {}
        self._short: Dict[str, str] = {}
        self._target_fn: Dict[str, str] = {}
        self._domain: Dict[str, str] = {}
        self._levels: Dict[str, List] = {}
        # Shorts whose knob declares that moving it changes the regime the objective is measured
        # in. See _reset_other_arms.
        self._regime_knobs: set = set()
        # Bandit statistics per (knob_short, level): pulls and mean observed wall.
        self._stats: Dict[Tuple[str, object], Tuple[int, float]] = {}
        # Recent wall samples per arm. The running mean above is enough to RANK arms, but not to
        # ask whether two arms differ at all, which needs a spread. Bounded because recency is what
        # matters under drift, and because an unbounded history would let a stale regime outvote
        # the current one.
        self._samples: Dict[Tuple[str, object], Deque[float]] = {}
        # (window, whole configuration, wall) per recorded window: samples keyed by CONFIGURATION, for the
        # joint verification (an arm keyed by one knob's value mixes every configuration that value ran in).
        self._series: Deque = deque(maxlen=4096)
        self._cur_window: int = -1
        self._win_metrics: Dict[str, float] = {}
        # A watermark is allowed to finalize only a real finding.  Retaining
        # that finding preserves the existing decision/provenance path while
        # making an empty native window incapable of creating a decision.
        self._pending_finding: Optional[DiagnosisFindingMsg] = None
        self._planned_cfg: Dict[int, dict] = {}
        # TAIL METRIC (same per-step family as the wall): the window's 90th-percentile per-step wall,
        # seconds, keyed like `_samples` so an arm's tail can be tested the way its mean is.
        self._tail_samples: Dict[Tuple[str, object], Deque[float]] = {}
        self._credit_tail: Dict[int, float] = {}
        # MEASURED APPLY COSTS, per knob: (|value change|, seconds training was held) of each applied
        # change, reported by the application on its ACK. A later move is priced from the latest one.
        self._apply_costs: Dict[str, List[Tuple[float, float]]] = {}
        self._physical_seen = False
        self._physical_change_old: Dict[Tuple[str, int], object] = {}
        self._cost_by_step: Dict[int, float] = {}
        self._cost_step_ids: Dict[str, int] = {}
        self._observed_initial_cfg = None
        self._applied_changes = []
        self._physical_latest_cfg = None
        self._physical_changes = []
        self._observation_first_steps = {}
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
        # BOUNDED EXPLORATION, PROPOSED AND NOT DEPLOYED (default 0 = today's behaviour exactly).
        #
        # Measured across all seven 32-node reps: every deployed move carries
        # `explore_under_sampled`, and the reason is arithmetic rather than statistical. A value is
        # held for a MEDIAN OF 2 WINDOWS before the walk steps again, while `min_samples` is 3, so
        # at the moment a move is proposed the candidate has fewer than 3 observations, the
        # acceptance test short-circuits to "explore", and the new value is then abandoned before it
        # reaches 3 either. The walk outruns its own evidence and no value in the run is ever
        # characterised. It is NOT the effect floor: the observed spread across prefetch levels is
        # 26 to 43 percent of the window wall against a 2 percent floor. It is NOT the grid: the
        # levels are well separated in measured wall.
        #
        # A minimum dwell equal to `min_samples` makes every visited value characterised by
        # construction, and bounds the whole exploration at `levels x min_samples` windows: for the
        # six prefetch levels that is 18 of an agent's ~74 windows in a 3-epoch run, leaving the
        # rest to exploit.
        #
        # APPROVED AS THE EVALUATED DEFAULT (2026-09-04), defaulting to min_samples because the two
        # are the same statement made twice: a value must be held long enough to be measured the
        # number of times the acceptance test requires. Set it to 0 to restore the no-dwell
        # behaviour reps 2 to 7 ran, which is the ablation of this rule.
        self._min_dwell = int(os.environ.get("DFOPTIMIZER_CBANDIT_MIN_DWELL",
                                             str(self._min_samples)))
        self._last_move_window: Dict[str, int] = {}
        self._sample_window = int(os.environ.get("DFOPTIMIZER_CBANDIT_SAMPLE_WINDOW", "32"))
        self._credit_observations = deque(maxlen=self._sample_window)
        # Set to 0 to disable the gate entirely and recover the previous behaviour exactly.
        self._accept_enabled = os.environ.get("DFOPTIMIZER_CBANDIT_ACCEPT", "1") != "0"
        # Cheapest-first escalation within the dominant domain. 0 restores the previous rule (one
        # knob per window, chosen by least pulls -- which, because _record credits every knob every
        # window, was always the first-registered knob and never any other). See process_finding.
        self._escalate = os.environ.get("DFOPTIMIZER_CBANDIT_ESCALATE", "1") != "0"
        # C1 ABLATION (DESIGN-MOVING-BOTTLENECK-2026-09-07, STATUS-ABLATION-MAP-2026-09-07): the
        # controller keeps its full knob set and its ordinary acquisition; it is only denied WHICH
        # domain is exposed. Collapses the per-domain breakdown to one aggregate "wall" entry that
        # no registered knob owns (`self._domain` never maps a knob to "wall"), so the existing
        # ownership check a few lines below this in `plan()` declines every window. This is not a
        # policy choice: giving a domain-blind controller some OTHER way to pick a knob (e.g. least-
        # recently-tried) would not be this planner made blind to domains, it would be a different
        # planner, so no such fallback is invented here. Default 0 = today's behaviour exactly.
        self._domain_blind = os.environ.get("DFOPTIMIZER_CBANDIT_DOMAIN_BLIND", "0") == "1"
        # C2 ABLATION: guards ONLY the unconditional decline at `if ranked and ranked[0][1] not in
        # actionable`, leaving `_accept_move`'s significance/effect-size test completely alone.
        # This is deliberately NOT `DFOPTIMIZER_CBANDIT_ACCEPT=0` (the "no-gate" run tag), which
        # disables a different mechanism (the Welch/effect-floor acceptance test) and would leave
        # the ownership decline fully intact. Default 0 = today's behaviour exactly.
        self._always_act = os.environ.get("DFOPTIMIZER_CBANDIT_ALWAYS_ACT", "0") == "1"
        # A1 (judge batch 1, 2026-09-25). The comm->input "barrier attribution" redirect lets a
        # comm-dominant window act on storage knobs.  Default 1 = historical behaviour; the TF loop
        # arm sets 0 so an input action can only come from an input-dominant window.  Every
        # decision record carries `decision_path` so the path that produced it is auditable.
        self._barrier_redirect = os.environ.get("DFOPTIMIZER_CBANDIT_BARRIER_REDIRECT", "1") != "0"
        # A2. Default 0 = historical.  1 = (a) no candidate below the value the knob was registered
        # with (the shipped Original) unless the measured explore_over_provisioned capacity path
        # admits it, and (b) exploration needs a directional prediction: a best candidate that
        # exactly ties another candidate on score is "no prediction" and the knob stays.
        # `_unconfirmed_move` (revert of a move that fails its predicted-effect check) is kept
        # and is the bounded regression budget.
        self._strict_explore = os.environ.get("DFOPTIMIZER_CBANDIT_STRICT_EXPLORE", "0") == "1"
        # The MEASURED replacement for the barrier redirect (SKEW-ATTRIBUTION-2026-09-25.md).  When the
        # analyzer reports domain_skew_coverage == 1 and A = domain_input_skew_attributed_time > 0,
        # the typical rank's vector is re-expressed as input' = input_typ + A, comm' = comm_typ - A
        # (A <= W <= comm) and dominance/predicted effect are computed on that.  A == 0 or coverage 0
        # => nothing moves.  Default ON; independent of the heuristic redirect above.
        self._skew_link = os.environ.get("DFOPTIMIZER_CBANDIT_SKEW_LINK", "1") != "0"
        # ACTIONABLE-DOMAIN PRIORITIZATION (orchestrator 2026-09-25).  Default 0 = historical: the
        # dominant domain alone decides, and a dominant domain that owns no knob declines the window.
        # 1: a dominant domain that owns no knob never vetoes.  Candidates are the domains that own a
        # registered knob AND carry measured exposure > 0 in this window (the window only reaches the
        # planner when the wired rule's fact fired); they are ranked by that measured exposure
        # (direct input wait, plus the analyzer's skew-attributed A when coverage == 1, already folded
        # into `domains` by the skew link).  No constant; every downstream gate is unchanged.
        self._joint_verify = os.environ.get("DFOPTIMIZER_CBANDIT_JOINT_VERIFY", "0") == "1"
        # COMM NEEDS MEASURED SKEW COVERAGE (opt-in; STORMER-LOOP-DESIGN sec 4).  A comm-domain action is
        # admissible only in a window whose cross-rank skew attribution was measured (domain_skew_coverage == 1;
        # A == 0 with coverage 1 is a real "no attributable skew").  Without coverage the barrier wait cannot be
        # split into transfer and late-rank input, so comm is not credited and owns nothing that window.
        self._comm_needs_skew = os.environ.get("DFOPTIMIZER_CBANDIT_COMM_NEEDS_SKEW", "0") == "1"
        self._actionable_priority = os.environ.get(
            "DFOPTIMIZER_CBANDIT_ACTIONABLE_PRIORITY", "0") == "1"
        # E2 (judge batch 4): B2 (`wall - sum-of-max fetch`, comm probe OFF) bundles compute and comm
        # into one unattributed "other" bucket, which has no owned knob.  Left in the ordinary argmax,
        # "other" usually outweighs "input" (compute alone is often ~180 of a ~300ms step) and the
        # window declines every time -- input never gets a turn even when its own exposure is real and
        # actionable.  This flag makes "input" gate on ITS OWN measured exposure (present, finite,
        # positive) plus the ordinary predicted-effect/verifiability checks that already run after
        # `dom` is chosen, and "other" NEVER competes in an argmax for this decision.  It is a
        # comm-observation-free (B2) mode: no bucket_cap_mb action can ever be reached this way,
        # by construction (see B1's own separate requirement, STORMER-LOOP-DESIGN sec 13).
        self._b2_input_only = os.environ.get(
            "DFOPTIMIZER_CBANDIT_B2_INPUT_ONLY", "0") == "1"
        # Sec 17 (full input+comm loop, orchestrator decision 2026-09-27): "a cap move can remove at
        # most the exposed comm" -- see predicted_effect's own docstring for the exact clamp.
        self._comm_effect_cap_window = os.environ.get(
            "DFOPTIMIZER_CBANDIT_COMM_EFFECT_CAP_WINDOW", "0") == "1"
        self._shipped = {}
        self._last_pick_note = None
        # C3 ABLATION: prices a relocation's gain WITHOUT weighing it against the run remaining.
        # Cost weighting (`DFOPTIMIZER_CBANDIT_COST_W`) is untouched -- this only removes the two
        # horizon terms: `_pick`'s `gain_s = gain_per_window_s * remaining` and the outer
        # `bool(rem_w)` half of `verifiable`. Default 1 = today's behaviour exactly (note the
        # inverted default: this one is ON by default, like `_escalate`/`_accept_enabled`, so
        # unsetting it is the ablation).
        self._horizon_weight = os.environ.get("DFOPTIMIZER_CBANDIT_HORIZON_WEIGHT", "1") != "0"
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
        self._last_bw_source = None
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
        # THE DESTINATION, which until 2026-09-04 was never asked about. See
        # _destination_bounded_gain. Both are measured per-cluster capacities with provenance, of
        # the same kind as the Lustre ceiling, and both are unset-safe: with no rate the planner
        # declines a positive-cost relocation rather than inventing one, exactly as the horizon
        # contract does with a missing horizon.
        self._dest_read_gbs = _envf("LOCAL_READ_CEILING_GBS")
        self._stage_rate_gbs = _envf("LIVEFLOW_STAGE_RATE_GBS")
        self._relocating: set = set()
        # THE EPOCH-TO-DATE BYTE RATE, and why a relocation may not be priced from one window.
        #
        # `un_loop_n8_10e_dev1` staged on all eight nodes at window 122 (T3-0 4s.8). Every agent
        # read 0.05 to 0.22 GB/s in THAT window against a 2.2975 GB/s ceiling and a ~2.3 GB/s epoch
        # mean, because the window was comm-dominated and, on that run, because the ring was
        # dropping 40% of the trace records. One low reading moved BOTH sides of the price the way
        # that pays: the derived cost fell to 58-241 s for a copy that took 913-948 s, and the
        # destination bound went vacuous, because a device only has to beat a source that is
        # reading nothing.
        #
        # A relocation moves an EPOCH's worth of bytes, so it is priced from an epoch's worth of
        # observation: the mean bytes this node read per window, over the windows of the current
        # epoch seen so far. More samples, no new constant. The floor on how many windows is
        # `min_samples`, already the planner's threshold for calling an arm characterised.
        self._rate_mode = False
        self._control_steps = 1
        self._epoch_key: Optional[object] = None
        self._epoch_bytes: float = 0.0
        self._epoch_windows: int = 0
        # An epoch that lost trace records did not observe the bytes it thinks it did, so it may
        # not price a relocation at all. `_drop_probe` is injected in tests and read from the
        # in-process ring reader otherwise; None means the question could not be asked, which is
        # recorded rather than assumed to be a zero.
        self._epoch_dropped: bool = False
        self._drops_at_epoch_start: Optional[int] = None
        self._drop_probe = None
        # The node's OWN observed read rate, at the same epoch-to-date scale as the bytes, because
        # the physical price needs both. Accumulated as a mean of the per-window rates rather than
        # as epoch bytes over epoch WALL: the wall on unet3d is 70.7 percent barrier (T3-0 4s.13),
        # and dividing bytes by it charges barrier time to reading, which understates the source
        # rate and inflates the gain by construction on exactly the cell where reading is off the
        # critical path. Measured: the two forms give 0 and 51 payable windows on the same records.
        self._epoch_bw_sum: float = 0.0
        self._epoch_bw_n: int = 0
        # THE DWELL IS DERIVED FROM THE OBSERVED VARIANCE, not fixed in windows.
        #
        # Orchestrator decision, 2026-09-05, on T3-0 4s.17. The acceptance test confirmed
        # `read_threads=1` 936 times on three-window samples, on a cell where the cost of that value
        # arrives in one window of six and lasts 10 to 42 s against a 4.37 s median window. One
        # stall is longer than the whole sample the dwell required, so 936 confirmations were not
        # 936 observations of anything. A larger fixed count is only another constant; the sample
        # size the test itself needs is not.
        #
        #     n_required = ceil( ((z_alpha + z_power) x sigma_wall / delta)^2 )
        #     dwell      = max(min_samples, n_required),  capped at the remaining horizon
        #
        # sigma_wall is the epoch-to-date standard deviation of the window wall, so it resets with
        # the regime; delta is `effect_floor` x the epoch's mean wall, the planner's existing
        # minimum interesting effect expressed in seconds. On a smooth cell (cosmoflow, 1.3 percent
        # spread) this reduces to min_samples. On a bursty one it grows until the sample could have
        # contained the bursts. If it exceeds the horizon the value is never characterised and the
        # planner stays on the incumbent rather than guessing.
        self._power = float(os.environ.get("DFOPTIMIZER_CBANDIT_POWER", "0.80"))
        # The predicted-effect admission (4s.23, 4s.24). ON by default: with the domain delta taken
        # as the lower confidence bound of the median, and a missing horizon treated as no horizon,
        # all three expectations hold -- unet3d admits 0 moves at every alpha and power, cosmoflow's
        # relocation needs 5 to 8 windows of a 70-window horizon, and D1 still descends to the
        # optimum in 35 windows with 8 plans. Set to 0 for the ablation.
        self._predicted_effect_gate = os.environ.get(
            "DFOPTIMIZER_CBANDIT_PREDICTED_EFFECT", "1") == "1"
        # A RELOCATION'S EFFECT BEGINS AT THE NEXT EPOCH BOUNDARY (T3-0 4s.37, 4s.39), so the
        # admission price is corrected in BOTH directions or in neither:
        #   horizon   the windows remaining AFTER the boundary, not from the decision,
        #   gain      capped by the EPOCH's exposure, not by the decision window's.
        # 4s.37 measured what half the correction does: the horizon alone refuses five of the six
        # recorded stages, r1b's and r1f's included, both of which paid. The window cap understates
        # the exposure by about half and the two errors were cancelling.
        self._reloc_effect_at_boundary = os.environ.get(
            "DFOPTIMIZER_RELOC_EFFECT_AT_BOUNDARY", "0") == "1"
        # median | hoeffding | bootstrap. See _domain_bound: measured on all three, median shipped.
        self._domain_delta_bound = os.environ.get(
            "DFOPTIMIZER_DOMAIN_DELTA_BOUND", "median").strip().lower()
        # A KNOB ABOVE A MEASURED CAPACITY (T3-0 4s.43). Both numbers are machine measurements with
        # provenance, not thresholds: the node's processor count and the ranks placed on it. Absent
        # -- which is the default -- the rule is off and nothing changes.
        self._capacity_knob = os.environ.get("DFOPTIMIZER_CAPACITY_KNOB", "").strip()
        self._capacity_procs = _num_or_none(os.environ.get("DFOPTIMIZER_CAPACITY_PROCS_PER_NODE"))
        self._capacity_ranks = _num_or_none(os.environ.get("DFOPTIMIZER_CAPACITY_RANKS_PER_NODE"))
        self._capacity_step = os.environ.get("DFOPTIMIZER_CAPACITY_STEP", "one").strip().lower()
        # Exploration needs a characterised incumbent (4s.20). Implemented, tested, DEFAULT OFF:
        # it blocks cosmoflow's staging as well as unet3d's move, which is not what was asked for.
        # The derived dwell applied to the ESCALATION verdict, which decides whether a costlier
        # actuator may be reached. Default off for the reason in _escalation_state.
        self._power_escalation = os.environ.get(
            "DFOPTIMIZER_CBANDIT_POWER_ESCALATION", "0") == "1"
        self._epoch_wall_sum: float = 0.0
        self._epoch_wall_sq: float = 0.0
        # Epoch-to-date exposed time per domain. A knob that OWNS a domain can recover at most that
        # domain's exposed time, so this is the measured BOUND on its effect (4s.23). Epoch-scale
        # for the same reason the bytes are: one window is not a regime.
        # Per-domain exposed time, kept as VALUES rather than a running sum because the statistic
        # is the MEDIAN. On unet3d the input domain is 1.9 ms in a typical window and 1.10 s on
        # average, because 16 percent of windows carry a 10 to 42 s stall (4s.19). The mean says a
        # parallelism move could recover a second per window, which is true of no window the move
        # would act in; the median says what a typical window holds. The decision's own expected
        # value for this cell is the median, and the two give opposite admissions (4s.23).
        self._epoch_dom: Dict[str, Deque[float]] = {}
        # Window walls aligned with _epoch_dom, so a measurement taken over the recent exposed run can be
        # tested against the wall noise OF THAT RUN (see _recent_run_effect / required_dwell).
        self._epoch_wall_hist: Deque[float] = deque(maxlen=self._sample_window)
        self._epoch_tail_hist: Deque[float] = deque(maxlen=self._sample_window)
        self._move_base_tails: Dict[str, List[float]] = {}
        self._recent_run_n = 0
        # Pre-move walls of the regime a recent-run move was predicted in; the verification baseline for it.
        self._move_base_walls: Dict[str, List[float]] = {}
        # After a physically observed DECREASE (a revert) the next raise of that pool is one refinement
        # step (+1), not x2; cleared by the next observed raise.
        self._refine: Dict[str, bool] = {}
        # What a move predicted, so it can be reverted if it is not confirmed at that size.
        self._move_from: Dict[str, tuple] = {}
        # A RELOCATED ARM IS SAMPLED FROM THE ACTUATOR'S COMPLETION, NOT FROM THE PLAN (4s.31).
        # The copy's own windows are neither the old regime nor the new one, so they are credited to
        # neither. Set when an ack arrives with reason `relocated`; cleared when the arm moves again.
        self._reloc_live_from: Dict[str, int] = {}
        # The actuator's own completion instant per relocating knob, and the wall-clock time each
        # window finished. Together they say which windows contain the copy (4s.35).
        self._reloc_live_at: Dict[str, Optional[float]] = {}
        # WHEN THE NEW SOURCE IS ACTUALLY IN USE, which is not when the copy ended (4s.37).
        # `data_source` resolves into a file list that the loader reads only when it is REBUILT,
        # and DLIO rebuilds at the epoch boundary. On `cf_fz_loop_n32_r1f` the copy finished at
        # 08:13:34 and the loader was rebuilt at 08:21:54, so the eight minutes between them were
        # PFS windows credited to `local`, the arms tied, and 32 nodes reverted one window before
        # the effect appeared. The arm is sampled from HERE.
        self._reloc_effective_at: Dict[str, Optional[float]] = {}
        self._reloc_pending_logged: Dict[str, int] = {}
        self._window_finished_at: Dict[int, float] = {}
        self._reloc_pending: Dict[str, object] = {}
        # The relocation value already published and not yet resolved by an ack. A second plan for
        # the SAME value is the repeat that cost five copies a node (4s.32); a plan for a different
        # value is a real decision and is not suppressed.
        self._reloc_published: Dict[str, object] = {}
        # PER-STEP VERIFICATION (LIVEFLOW_VERIFY=per_step, default; window_mean = the window-mean rule
        # the v1 campaigns ran). State and decisions live in step_planner.StepVerifyMixin.
        self._sv_init()
        # LIVEFLOW_SIZING=littles_law|geometric (default littles_law): see planner/sizing.py.  `_ll` is the CURRENT window's
        # Little's-law result per sizable knob; `_ll_partners[lead]` the other knobs moved in the same plan as an LL-sized lead
        # (restored together if the lead is reverted).
        self._sizing = sizing.sizing_mode()
        self._ll: Dict[str, dict] = {}
        self._ll_partners: Dict[str, Tuple[int, Dict[str, object]]] = {}

    # --- Planner-compatible registration (identical contract to Planner/CLBOPlanner) ---
    def register_knobs(self, knob_defs, current_values=None):
        current_values = current_values or {}
        items = knob_defs.items() if hasattr(knob_defs, "items") else ((kd.id, kd) for kd in knob_defs)
        for knob_id, kd in items:
            if os.environ.get("LIVEFLOW_PRESET") == "liveflow-v1" and knob_id.split('.')[-1] not in (
                    'read_threads', 'decode_threads', 'num_workers', 'prefetch_factor',
                    'prefetch_size', 'computation_threads', 'bucket_cap_mb', 'data_source',
                    'transfer_mb', 'delivery_mode'):
                logger.info('optimizer.liveflow_v1.knob_excluded', knob_id=knob_id,
                            reason='not in the math-preserving v1 knob whitelist')
                continue
            self.knobs[knob_id] = kd
            short = knob_id.split(".")[-1]
            self._short[knob_id] = short
            self._target_fn[knob_id] = kd.target_function
            cur = current_values.get(knob_id, kd.default)
            self.current_values[knob_id] = cur
            # The FIRST registered value is the shipped/Original start; later merges never move it.
            # The optimizer registers an env bootstrap registry BEFORE the application's own (which carries
            # the workload's real starting value), so the shipped value is re-latched until the first
            # finding arrives and never moves afterwards.
            if self._cur_window == -1:
                self._shipped[short] = cur
            else:
                self._shipped.setdefault(short, cur)
            self._domain[short] = _domain_of(short)
            self._levels[short] = _levels_for(kd, cur)
            # register_knobs MERGES and is called more than once (the optimizer bootstraps a
            # registry before the application's own arrives), so the latest declaration wins here
            # too rather than accumulating.
            if getattr(kd, "changes_regime", False):
                self._regime_knobs.add(short)
            else:
                self._regime_knobs.discard(short)
            # A knob that RELOCATES the bytes is priced against where it puts them.
            if getattr(kd, "relocates_data", False):
                self._relocating.add(short)
            else:
                self._relocating.discard(short)
        logger.info("optimizer.cbandit.registered",
                    knobs=list(self._short.values()),
                    regime_knobs=sorted(self._regime_knobs),
                    relocating_knobs=sorted(self._relocating),
                    dest_read_gbs=self._dest_read_gbs or None,
                    stage_rate_gbs=self._stage_rate_gbs or None,
                    domains={s: self._domain[s] for s in self._short.values()},
                    levels={s: self._levels[s] for s in self._short.values()})

    def _short_cfg(self) -> dict:
        return {self._short[k]: self.current_values[k] for k in self.knobs}

    def apply_ack(self, plan_id: str = "", knob_id: str = "", status: str = "unknown",
                  old_value=None, new_value=None, window_index: int = -1,
                  target_nodes=None, reason: str = "", completed_at=None,
                  effective_at=None, effective_window_id=None, applied_step=None,
                  apply_cost_s=None):
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
                    old_value=old_value, new_value=new_value, window_index=window_index,
                    reason=reason or None, completed_at=completed_at,
                    effective_at=effective_at, effective_window_id=effective_window_id)
        if knob_id not in self.knobs:
            return
        short_id = self._short.get(knob_id, knob_id)
        self._note_apply_cost(short_id, status, old_value, new_value, apply_cost_s)
        # The application may report the native boundary where the value actually
        # took effect.  It is more precise than the optimizer's receipt window;
        # malformed and absent values deliberately retain the legacy path.
        try:
            _ack_effective_window = int(effective_window_id)
        except (TypeError, ValueError):
            _ack_effective_window = -1
        # THE SECOND ACK: the actuator says the new source is IN USE (4s.37). It arrives at the
        # loader rebuild, a later call than the one that copied, so it is its own ack rather than a
        # field of the first. It reports no move -- the move already happened -- so it is handled
        # before every branch that reads `reason`, and it changes nothing but the sampling gate. A
        # run whose actuator never sends it leaves the move PENDING and says so on every window,
        # which is the honest state: nothing about the new regime has been measured yet.
        if short_id in self._relocating and effective_at is not None:
            self._reloc_effective_at[short_id] = float(effective_at)
            logger.info("optimizer.cbandit.reloc_effective", knob_id=knob_id,
                        effective_at=float(effective_at), window=self._cur_window,
                        completed_at=self._reloc_live_at.get(short_id))
            return

        # A RELOCATION THAT DID NOT RELOCATE IS NOT AN APPLICATION (4s.31). The actuator now says
        # why, and a `no_move_needed` or `resolver_declined` ack leaves the planner where it was
        # with the reason on the record rather than booking a move that never happened. It is not a
        # revert either: nothing was done, so there is nothing to undo.
        if short_id in self._relocating and status == "applied" and reason and reason != "relocated":
            logger.info("optimizer.cbandit.ack.not_applied", knob_id=knob_id, reason=reason,
                        stayed_on=self.current_values.get(knob_id),
                        note="actuator reported no relocation; incumbent unchanged")
            self._move_from.pop(short_id, None)
            self._reloc_pending.pop(short_id, None)
            self._reloc_published.pop(short_id, None)
            return
        if short_id in self._relocating and status == "applied" and reason == "relocated":
            # Sampling of the new arm starts HERE, at the window the copy finished in, not at the
            # window the plan was issued in.
            # USE THE TIMESTAMP THE ACTUATOR SENT, NOT A WINDOW INDEX (T3-0 4s.35).
            #
            # When the application supplies its native effective boundary, use it.  Older ACKs
            # retain the historical receipt-window/current-window fallback.
            #
            # `completed_at` is the moment the copy returned. Windows that finished before it
            # contain the copy and belong to neither regime, so they are credited to neither.
            #
            # BUT THE COPY ENDING IS NOT THE EFFECT BEGINNING (4s.37). `data_source` resolves into
            # a file list, and the loader reads that list only when it is REBUILT, at the epoch
            # boundary. `effective_at` below is when that happened; until it arrives the move is
            # PENDING, and a pending move is neither confirmed nor reverted.
            self._reloc_live_at[short_id] = (float(completed_at)
                                             if completed_at is not None else None)
            live = (_ack_effective_window if _ack_effective_window >= 0 else
                    window_index if window_index is not None and window_index >= 0 else
                    self._cur_window)
            self._reloc_live_from[short_id] = int(live)
            self._reloc_pending.pop(short_id, None)
            self._reloc_published.pop(short_id, None)
            self._reloc_effective_at.setdefault(short_id, None)
            logger.info("optimizer.cbandit.reloc_completed", knob_id=knob_id,
                        live_from_window=int(live), completed_at=completed_at,
                        effective_at=effective_at)
        if status == "applied":
            if self._physical_latest_cfg is not None:
                return  # ACK is a receipt; physical window state owns v1 credit/current state
            if new_value is not None:
                if self._observed_initial_cfg is None:
                    self._observed_initial_cfg = self._short_cfg()
                if old_value is not None and not any(c[0] == short_id for c in self._applied_changes):
                    self._observed_initial_cfg[short_id] = old_value
                if _ack_effective_window >= 0 or applied_step is not None:
                    self._applied_changes.append((short_id, new_value, _ack_effective_window,
                                                  int(applied_step) if applied_step is not None else None))
                applied = self.knobs[knob_id].clamp(new_value)
                changed = applied != self.current_values.get(knob_id)
                if changed:
                    # D1d: the planner adopts the ACKED observed value as its arm.  The plan and the observation
                    # differ only if the application applied something else (cross-node reconciliation); record it
                    # and repair the window plan so samples are never labelled with an arm that never ran.
                    _short = self._short.get(knob_id, knob_id)
                    logger.info("optimizer.cbandit.consensus_override", knob_id=knob_id,
                                planned=self.current_values.get(knob_id), observed=applied,
                                window=self._cur_window)
                    for _wk in [k for k in self._planned_cfg if k > self._cur_window]:
                        self._planned_cfg[_wk][_short] = applied
                self.current_values[knob_id] = applied
                if changed and self._short.get(knob_id, knob_id) in self._regime_knobs:
                    self._reset_other_arms(
                        self._short.get(knob_id, knob_id),
                        _ack_effective_window if _ack_effective_window >= 0 else window_index,
                    )
            return
        # rejected / failed / anything else: the application is still on the old value, and our
        # model has to say so or the arm is lost for the rest of the run.
        revert = old_value if old_value is not None else self.knobs[knob_id].default
        self.current_values[knob_id] = self.knobs[knob_id].clamp(revert)
        short = self._short.get(knob_id, knob_id)
        self._move_from.pop(short, None)     # a refused move is not in flight
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

    def _note_apply_cost(self, short, status, old_value, new_value, apply_cost_s, step_id=None):
        """Keep what an applied change of this knob measurably cost, from the application's clock.

        `step_id` (the applied step) lets a refined measurement replace the earlier one of the same
        apply instead of adding a second sample.
        """
        if status != "applied" or apply_cost_s is None:
            return
        try:
            cost = float(apply_cost_s)
            step = abs(float(new_value) - float(old_value))
        except (TypeError, ValueError):
            return
        if not (math.isfinite(cost) and cost >= 0.0 and step > 0.0):
            return
        hist = self._apply_costs.setdefault(short, [])
        refined = step_id is not None and bool(hist) and self._cost_step_ids.get(short) == step_id
        if refined:
            hist[-1] = (step, cost)
        else:
            hist.append((step, cost))
        if step_id is not None:
            self._cost_step_ids[short] = step_id
        logger.info("optimizer.cbandit.apply_cost", knob=short, step=step, cost_s=round(cost, 3),
                    measurements=len(hist), refined=refined)

    def _measured_apply_cost(self, short: str, current, value) -> Optional[float]:
        """Seconds a move of `short` from `current` to `value` is predicted to hold training.

        The latest measured apply of this knob, scaled by the size of the change (an apply forks
        or retires workers in proportion to the change). None until this run has measured one:
        no declared or default constant stands in for a measurement.
        """
        hist = self._apply_costs.get(short)
        if not hist:
            return None
        try:
            step = abs(float(value) - float(current))
        except (TypeError, ValueError):
            return None
        last_step, last_cost = hist[-1]
        return last_cost * (step / last_step) if last_step > 0 else last_cost

    def _inflight_move(self) -> Optional[str]:
        """The knob whose last move still awaits its verdict, or None.

        A move enters `_move_from` when it is decided and leaves it when it is confirmed, reverted,
        superseded by a different applied value, or refused by the application.
        """
        if self._joint_verify:
            # Joint verification (opt-in) verifies a coupled pair of moves as one step, so it needs the
            # second move to be allowed while the first awaits its verdict.
            return None
        for short, entry in self._move_from.items():
            if entry[1] and entry[1] > 0:
                if self._sv_active() and self._sv_mechanism_confirmed(short, entry):
                    continue     # it did what it targeted; only its net step-time verdict is still being collected
                return short
        return None

    def _apply_cost_veto(self, short, current, value, delta, remaining, window) -> Optional[str]:
        """Decline a move whose predicted gain over the run left cannot repay its measured apply cost.

        Predicted gain = the move's predicted per-step effect (`delta`, the lower confidence bound
        of the domain's exposure) x the steps remaining; cost = what this knob's last measured
        apply held training for, scaled by the size of the change. Both are measurements of this
        run. Silent when there is no measured cost yet, no predicted effect, or no horizon.
        """
        if self._cost_w == 0.0 or delta is None or remaining is None:
            return None
        cost = self._measured_apply_cost(short, current, value)
        if cost is None or cost <= 0.0:
            return None
        gain = float(delta) * float(remaining)
        if gain > self._cost_w * cost:
            return None
        logger.info("optimizer.cbandit.apply_cost_veto", window=window, knob=short, current=current,
                    to=value, predicted_gain_s=round(gain, 3), measured_cost_s=round(cost, 3),
                    remaining=float(remaining))
        return "unpayable_apply_cost:gain=%.1fs<=cost=%.1fs" % (gain, cost)

    def _reset_other_arms(self, short: str, window_index: int = -1):
        """A regime-changing knob was APPLIED: every other knob's history is now uncomparable.

        THE FINDING THIS ANSWERS (T3-0 section 4p, measured over reps 8 to 10 at 32 nodes). Each
        arm's sample deque pooled windows from before and after staging. An epoch-1 window is about
        37 s of wall and an epoch-2 window about 5.6 s, a 6.7x shift in the objective INSIDE one
        arm's samples. Within-arm variance then swamps any between-arm difference, so Welch cannot
        separate two prefetch levels no matter how well each is sampled, and the acceptance test
        rejects for a reason that has nothing to do with the knob it is testing.

        The fix is a statement about which samples are COMPARABLE, not a new constant and not a
        threshold: there is no number here to tune. Observations taken under a different data
        source are observations of a different objective, so they are discarded for every knob
        except the one that moved. That knob keeps its own arms, because its arms are exactly what
        distinguishes the two regimes and are the only place the old regime is still meaningful.

        Consequences, both intended. Every other knob returns to under-sampled, so exploration
        restarts and the first few post-staging moves are explore moves again; and the values just
        left are re-measured on the new regime rather than trusted from the old one. The dwell
        floor still spaces those moves, and `_total_pulls` is deliberately not zeroed: it counts
        observed windows and only enters the UCB radius through a logarithm, so resetting it would
        change the exploration weight without saying anything about comparability.

        `_last_move_window` is also left alone. It records when a knob last moved, which is a fact
        about the run and not an observation of the objective.
        """
        dropped = sorted({k[0] for k in self._samples if k[0] != short} |
                         {k[0] for k in self._stats if k[0] != short})
        n_samples = sum(len(v) for k, v in self._samples.items() if k[0] != short)
        self._samples = {k: v for k, v in self._samples.items() if k[0] == short}
        self._tail_samples = {k: v for k, v in self._tail_samples.items() if k[0] == short}
        self._stats = {k: v for k, v in self._stats.items() if k[0] == short}
        logger.info("optimizer.cbandit.regime_reset", knob=short, window=window_index,
                    reset_knobs=dropped, samples_discarded=n_samples,
                    reason="a regime-changing knob was applied; samples taken before it are not "
                           "observations of the same objective")

    # --- window accumulation, same shape as CLBOPlanner ---
    def _accumulate(self, finding: DiagnosisFindingMsg):
        for k, v in (finding.key_metrics or {}).items():
            self._win_metrics[k] = v

    def _observed_cfg(self, window, metrics):
        """Credit only fully post-application windows; a proposal is not application."""
        if metrics.get("liveflow_v1_attribution") != 1.0:
            return dict(self._planned_cfg.get(window, self._short_cfg()))
        fields = ["physical_" + short for short in self._short.values()]
        if fields and all(k in metrics for k in fields) and "physical_applied_step" in metrics:
            cfg = {short: int(metrics["physical_" + short]) for short in self._short.values()}
            step = int(metrics["physical_applied_step"])
            # Cumulative step of the window's first step, the same axis as `physical_applied_step`
            # (completed steps). `step_min` restarts every epoch, so across an epoch boundary it
            # would call every later window an apply window.
            first = None
            if metrics.get("completed_steps") is not None and metrics.get("step_nunique") is not None:
                first = metrics["completed_steps"] - metrics["step_nunique"] + 1
            if first is None:
                first = metrics.get("step_min")
            if first is None or first <= step:
                return None  # partial/apply-window state cannot be credited as an after sample
            if self._physical_latest_cfg != cfg:
                for kid, short in self._short.items():
                    if self._physical_latest_cfg is not None and self._physical_latest_cfg.get(short) != cfg[short]:
                        self._physical_changes.append((short, cfg[short], step))
                        _was = self._physical_latest_cfg.get(short)
                        if _was is not None:
                            self._refine[short] = cfg[short] < _was
                        self._physical_change_old[(short, step)] = _was
                        if self._sv_active():
                            # Per-step verdicts, not pooled arms, are the evidence about an interacting
                            # configuration: every other knob's arm samples (keyed by that knob's value alone)
                            # now mix configurations that no longer run (halo M: read and decode threads interact).
                            self._reset_other_arms(short, window)
                    self.current_values[kid] = cfg[short]
                self._physical_latest_cfg = dict(cfg)
            # Every node learns what the apply measurably cost from the shared applied-vector
            # snapshot (not only the node that published the plan). The value is refined once, a
            # window after the apply, when the stall of the step that followed it is known.
            cost_us = _num_or_none(metrics.get("physical_apply_cost_us"))
            if cost_us is not None and self._cost_by_step.get(step) != cost_us:
                self._cost_by_step[step] = cost_us
                for short, value, st in self._physical_changes:
                    if st == step and self._physical_change_old.get((short, st)) is not None:
                        self._note_apply_cost(short, "applied", self._physical_change_old[(short, st)],
                                              value, cost_us / 1e6, step_id=step)
            self._observation_first_steps[window] = float(first)
            return cfg
        if self._observed_initial_cfg is None:
            self._observed_initial_cfg = self._short_cfg()
        cfg = dict(self._observed_initial_cfg)
        end = metrics.get("completed_steps")
        count = metrics.get("step_nunique")
        first = float(end) - float(count) + 1 if end is not None and count is not None else None
        for short, value, effective_window, step in self._applied_changes:
            after = first > step if first is not None and step is not None else window > effective_window
            if after:
                cfg[short] = value
        return cfg

    def _finalize_window(self, window: int) -> Optional[dict]:
        m = self._win_metrics
        # The per-step series and node telemetry are evidence about the steps themselves, whether or
        # not this window can be credited to a configuration (an apply window is not, but its steps
        # after the apply boundary are post-apply steps).
        self._sv_ingest(window, m)
        if m.get("liveflow_v1_attribution") == 1.0:
            if m.get("planner_history_ready") != 1 or m.get("lf_v1_per_step") != 1:
                logger.info("optimizer.liveflow_v1.unready_history", window=window)
                return None
            self._rate_mode = True
            self._control_steps = int(m.get("configured_window_steps") or m["step_nunique"])
        if self._rate_mode and any(k.startswith("halo3d.") for k in self.knobs):
            required = ("physical_read_threads", "physical_decode_threads", "physical_applied_step")
            if not all(k in m for k in required) or self._observed_cfg(window, m) is None:
                logger.info("optimizer.liveflow_v1.missing_physical_state", window=window)
                return None
        elif self._rate_mode:
            # A workload whose application snapshots the applied vector (StorMer: num_workers,
            # prefetch_factor) is credited from it, never from per-node ACKs. Once seen, a window
            # without a coherent applied vector is not evidence about any configuration.
            have = (all(("physical_" + sh) in m for sh in self._short.values())
                    and "physical_applied_step" in m)
            if have:
                self._physical_seen = True
            if self._physical_seen and (not have or self._observed_cfg(window, m) is None):
                logger.info("optimizer.liveflow_v1.missing_physical_state", window=window,
                            reason="no coherent applied vector, or an apply window")
                return None
        wall = m.get("active_time")
        # LiveDiagnoser forwards the analyzer's coherent control vector.  Its
        # wall is authoritative for that vector and is intentionally named
        # separately from the legacy aggregate `active_time`.  Do not invent a
        # fallback for a partial vector: only use it when the explicit
        # coherence marker is present, after which domains_from_metrics below
        # validates every required coherent field.
        if wall is None and "domain_coherent" in m:
            wall = m.get("window_wall_time_proc_crit")
        if wall is None or wall <= 0:
            return None
        # ONE shared definition, so the two planners cannot drift and any difference between them
        # is the SEARCH, not the observation. Both used to build this inline and both had the same
        # bug: reader_posix as the domain time instead of the exposed fetch. See
        # clbo_planner.domains_from_metrics.
        domains = domains_from_metrics(m)
        # Explicit coherent provenance that fails validation is not equivalent to
        # missing provenance.  Do not update bandit state, epoch aggregates, or
        # emit a plan from an invalid vector.
        if domains is None:
            return None
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
        self._sizing_update(m, float(wall), domains, window)
        skew = self._skew_from_metrics(m, wall) if self._skew_link and not self._rate_mode else None
        if skew is not None:
            domains, unattributed = skew["domains"], skew["unattributed"]
        remaining, remaining_unit = self._remaining_run(m)
        # When this window finished, so a completion timestamp can be turned into a window.
        self._window_finished_at[window] = _time.time()
        # WHICH EPOCH THIS WINDOW BELONGS TO. Preferred from the analyzer's own stamp; derived from
        # the cadence when it is absent, which it is on some presets. Either way the accumulator
        # only needs a key that changes when the epoch does.
        wpe = _num_or_none(m.get("windows_per_epoch"))
        # The native window fact is indexed by `window`, so it need not carry
        # an event-level `epoch` field.  The boundary contract nevertheless
        # forwards the workload's authoritative epoch as `epoch_index`.
        # Prefer it before deriving an epoch from a global control-window
        # counter: tails/flushes make that arithmetic disagree with the real
        # workload epoch and prematurely reset cbandit's evidence.
        epoch_key = m.get("epoch")
        if epoch_key is None:
            epoch_key = m.get("epoch_index")
        if epoch_key is None and wpe:
            epoch_key = int(window // float(wpe))
        if self._rate_mode:
            epoch_key = "liveflow-v1-post-freeze"
        # THE FULL WINDOW INPUT, ON EVERY WINDOW INCLUDING A DECLINE.
        #
        # T3-0 4s.21: the replay harness could not reproduce the cosmoflow stage because
        # `optimizer.cbandit.decision` does not carry what this method consumed -- windows_per_epoch,
        # the busy time, the horizon on a decline -- so a replay had to reconstruct them and got a
        # different run. This record closes that at the source: it is everything the planner reads,
        # emitted beside the decision, so a replay is exact by construction rather than by
        # archaeology. One line per window per node; at unet3d's 4312 windows it is a few MB.
        logger.info("optimizer.cbandit.window_inputs", window=window,
                    metrics={k: v for k, v in m.items()},
                    node_ceiling_gbs=self._node_ceiling_gbs or None,
                    aggregate_ceiling_gbs=self._agg_ceiling_gbs or None,
                    nodes=self._nodes or None,
                    dest_read_gbs=self._dest_read_gbs or None,
                    stage_rate_gbs=self._stage_rate_gbs or None,
                    cfg=self._observed_cfg(window, m))
        # Establish/reset the epoch before accepting this window's domain
        # observation.  Appending first discarded the first observation of
        # every real epoch when `_note_epoch()` noticed the transition below.
        # That made the epoch-local evidence counter disagree with its sample
        # arrays and, at short cadences, could make a valid controller decision
        # impossible solely because of ordering.
        self._note_epoch(epoch_key, m.get("node_read_bytes", 0.0),
                         self._node_read_bandwidth({
                             "node_read_bytes": m.get("node_read_bytes", 0.0),
                             "node_read_bytes_any": m.get("node_read_bytes_any", 0.0),
                             "node_read_busy_s": m.get("node_read_busy_s", 0.0),
                             "wall": float(wall)}),
                         wall_s=float(wall))
        for _d, _v in domains.items():
            self._epoch_dom.setdefault(_d, deque(maxlen=self._sample_window)).append(
                float(_v or 0.0))
        self._epoch_wall_hist.append(float(wall))
        tail_us = _num_or_none(m.get("step_wall_p90_us"))
        if tail_us is not None and tail_us >= 0.0:
            self._epoch_tail_hist.append(tail_us / 1e6)
        return {
            "cfg": self._observed_cfg(window, m),
            "tail": (tail_us / 1e6) if tail_us is not None and tail_us >= 0.0 else None,
            "domains": domains,
            "skew": skew,
            "skew_coverage": _num_or_none(m.get("domain_skew_coverage")),
            "unattributed": unattributed,
            "wall": float(wall),
            "remaining": remaining,
            "remaining_unit": remaining_unit,
            # The storage layer's own two numbers for this node and window, carried so the
            # barrier-attribution rule can look one layer down without a second metrics read.
            "node_read_bytes": float(m.get("node_read_bytes", 0.0) or 0.0),
            "node_read_busy_s": float(m.get("node_read_busy_s", 0.0) or 0.0),
            # Diagnostic only; see the note in dlio-signal.yaml.
            "node_read_bytes_any": float(m.get("node_read_bytes_any", 0.0) or 0.0),
            # For the derived cost of a relocation: how many windows of these bytes make an epoch.
            "windows_per_epoch": _num_or_none(m.get("windows_per_epoch")),
            # Epoch-scale observation of the same bytes; see _epoch_bytes_per_window.
            "epoch": epoch_key,
            "epoch_windows": self._epoch_windows,
            "epoch_bytes_per_window": self._epoch_bytes_per_window(),
            "epoch_src_gbs": self._epoch_src_gbs(),
            "epoch_dropped": self._epoch_dropped,
            "drops_known": self._drops_at_epoch_start is not None,
        }

    @staticmethod
    def _remaining_run(metrics: dict) -> Tuple[Optional[float], Optional[str]]:
        """Return the remaining-run multiplier reported to the agent, without a fallback.

        A window count and an epoch count are both valid because the analyzer may publish either
        kind of aggregate. Inventing a default horizon would make the same actuator payable or
        unpayable based on a planner constant rather than the run that is actually left.
        """
        if metrics.get("lf_v1_per_step") == 1.0:
            value = _num_or_none(metrics.get("remaining_steps"))
            return (max(value, 0.0), "steps") if value is not None else (None, None)
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

    def _record(self, cfg: dict, wall: float, window=None, tail: Optional[float] = None):
        """Credit the observed wall to every knob level that produced it.

        Attributing one wall time to each knob's level independently is the factored assumption --
        the same one CLBO's FactoredSurrogate makes. It is wrong when knobs interact strongly, and
        right enough when each knob dominates its own domain, which is the case this system is
        built for.
        """
        if window is not None and self._rate_mode:
            self._credit_observations.append((window, dict(cfg), wall))
            if tail is not None:
                self._credit_tail[window] = float(tail)
                if len(self._credit_tail) > 4 * self._sample_window:
                    for old_window in sorted(self._credit_tail)[:-2 * self._sample_window]:
                        del self._credit_tail[old_window]
        for short, level in cfg.items():
            # THE COPY'S OWN WINDOWS BELONG TO NEITHER REGIME (4s.31). Between the plan and the
            # actuator's completion the node is copying hundreds of GB while still reading the old
            # source, so those windows measure the move, not the state it moves to. Crediting them
            # to the new level is what made `local` look 32.4 s when the regime it bought runs at
            # 5.5, and reverted it nine windows before its evidence existed.
            if short in self._relocating:
                live_at = self._reloc_live_at.get(short)
                live_from = self._reloc_live_from.get(short)
                eff_at = self._reloc_effective_at.get(short)
                if live_from is None and live_at is None and short in self._reloc_pending:
                    continue                      # relocation acked, completion not yet seen
                # THE COPY ENDING IS NOT THE EFFECT BEGINNING (4s.37). Between the two the loader
                # is still reading the OLD file list, so those windows measure the old source at
                # the new source's name. On `cf_fz_loop_n32_r1f` that was every window from 4 to
                # 20: PFS windows of 14 to 66 s credited to `local`, against a `pfs` arm of 25 s,
                # which is why the two arms tied and 32 nodes reverted one window before the
                # rebuild. Until the actuator says the new source is in use, the arm has no
                # samples, and that is not a gap in the record: it is the record.
                if live_at is not None and eff_at is None and short in self._reloc_effective_at:
                    if self._reloc_pending_logged.get(short) != self._cur_window:
                        self._reloc_pending_logged[short] = self._cur_window
                        logger.info("optimizer.cbandit.reloc_pending_effect", knob=short,
                                    window=self._cur_window, level_=str(level),
                                    completed_at=live_at,
                                    reason="copy finished, the loader has not been rebuilt on it")
                    continue
                gate = eff_at if eff_at is not None else live_at
                if gate is not None:
                    # The timestamp wins: a window that finished before the gate contains the copy
                    # or the old source, whatever window index the ack arrived on.
                    fin = self._window_finished_at.get(self._cur_window)
                    if fin is not None and fin < gate:
                        continue
                elif live_from is not None and self._cur_window < live_from:
                    continue
            n, mean = self._stats.get((short, level), (0, 0.0))
            self._stats[(short, level)] = (n + 1, mean + (wall - mean) / (n + 1))
            buf = self._samples.get((short, level))
            if buf is None:
                buf = self._samples[(short, level)] = deque(maxlen=self._sample_window)
            buf.append(float(wall))
            if tail is not None:
                tbuf = self._tail_samples.get((short, level))
                if tbuf is None:
                    tbuf = self._tail_samples[(short, level)] = deque(maxlen=self._sample_window)
                tbuf.append(float(tail))
        self._total_pulls += 1
        self._series.append((self._cur_window, dict(cfg), float(wall)))

    def _ring_drops(self) -> Optional[int]:
        """Cumulative trace records lost by the ring, or None if it cannot be asked.

        The optimizer stage runs in the same process as the ring reader (`node_agent.py`), so this
        is a local read and not a new hop. Off-cluster and in tests there is no ring, and None is
        the honest answer: the planner records `drops_unknown` rather than treating "I could not
        look" as "there were none".
        """
        if self._drop_probe is not None:
            try:
                return int(self._drop_probe())
            except Exception:                      # noqa: BLE001
                return None
        try:
            from liveflow_transport import shm_io as _io      # noqa: PLC0415
            totals = _io.ring_totals() or {}
        except Exception:                          # noqa: BLE001
            return None
        got = totals.get("trace")
        if not got:
            return None
        # (written, dropped, unclaimed); `dropped` already includes `unclaimed` via Reader.totals.
        try:
            return int(got[1])
        except Exception:                          # noqa: BLE001
            return None

    def _note_epoch(self, obs_epoch, node_read_bytes, node_read_gbs=None,
                    wall_s=None) -> None:
        """Accumulate this window's bytes into the current epoch, and reset on an epoch change."""
        if obs_epoch != self._epoch_key:
            self._epoch_key = obs_epoch
            self._epoch_bytes = 0.0
            self._epoch_windows = 0
            self._epoch_bw_sum = 0.0
            self._epoch_bw_n = 0
            self._epoch_wall_sum = 0.0
            self._epoch_wall_sq = 0.0
            self._epoch_dom = {}
            self._epoch_wall_hist.clear()
            self._epoch_tail_hist.clear()
            self._epoch_dropped = False
            self._drops_at_epoch_start = self._ring_drops()
        # GB, NOT BYTES. `node_read_bytes` is raw bytes and every priced quantity downstream is in
        # GB, because `_node_read_bandwidth` returns `(by / 1e9) / wall`. Accumulating raw bytes
        # here made `un_loop_n8_10e_dev2` derive a cost of 1.15e12 s and refuse every relocation
        # for the wrong reason (T3-0 4s.9): the right answer produced by a broken price, which is
        # worse than a wrong answer because it looks like the rule working.
        self._epoch_bytes += max(float(node_read_bytes or 0.0), 0.0) / 1e9
        if node_read_gbs:
            self._epoch_bw_sum += float(node_read_gbs)
            self._epoch_bw_n += 1
        if wall_s:
            self._epoch_wall_sum += float(wall_s)
            self._epoch_wall_sq += float(wall_s) ** 2
        self._epoch_windows += 1
        now = self._ring_drops()
        if now is not None:
            # Any loss since this epoch began contaminates it. A drop that happened in an earlier
            # epoch is not this epoch's problem, which is why the baseline is taken at the reset
            # and not compared against zero.
            base = self._drops_at_epoch_start
            if base is None:
                self._drops_at_epoch_start = now
            elif now > base:
                self._epoch_dropped = True

    @staticmethod
    def _z(p: float) -> float:
        """Standard normal quantile, Acklam's rational approximation (|error| < 1.15e-9).

        Written out rather than imported so the planner keeps no scipy dependency on the online
        path, and so the two constants it is used with are visible where they are used.
        """
        if p <= 0.0 or p >= 1.0:
            return 0.0
        a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
             1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
        b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
             6.680131188771972e+01, -1.328068155288572e+01)
        c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
             -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
        d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
             3.754408661907416e+00)
        pl, ph = 0.02425, 1 - 0.02425
        if p < pl:
            q = math.sqrt(-2 * math.log(p))
            return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                   ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        if p > ph:
            q = math.sqrt(-2 * math.log(1 - p))
            return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                    ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)

    def _epoch_wall_stats(self) -> Optional[Tuple[float, float, int]]:
        """(mean, sample sigma, n) of the window wall over this epoch so far, or None."""
        n = self._epoch_windows
        if n < 2 or self._epoch_wall_sum <= 0.0:
            return None
        mean = self._epoch_wall_sum / n
        var = max((self._epoch_wall_sq - n * mean * mean) / (n - 1), 0.0)
        return mean, math.sqrt(var), n

    def _within_arm_sigma(self, short: str,
                          value=None) -> Optional[Tuple[float, float, int]]:
        """(mean, pooled WITHIN-arm sigma, total n) over the sampled values of one knob.

        WHY NOT THE EPOCH'S TOTAL WALL SPREAD, which is what the decision as written asked for.
        A power calculation needs the noise the test must see THROUGH, and the epoch-to-date spread
        of the wall also contains the differences BETWEEN the values the controller visited -- which
        is the signal, not the noise. Using the total makes the dwell grow exactly because the
        controller is doing its job: measured on the D1 regression cell, sigma 1.288 on a 13.44 s
        mean gives n_required = 181 and freezes the descent that test exists to check.

        Pooling the within-arm variances is the same quantity Welch is estimating, so the dwell it
        implies is the sample size that test actually needs. Falls back to the epoch-wide spread
        when no arm of this knob has two samples yet, which is the only case where there is nothing
        better to use.
        """
        if value is not None:
            own = list(self._samples.get((short, value), ()))
            if len(own) >= 2:
                m = sum(own) / len(own)
                var = sum((x - m) ** 2 for x in own) / (len(own) - 1)
                return m, math.sqrt(var), len(own)
        vs = [list(v) for (k, _lv), v in self._samples.items() if k == short and len(v) >= 2]
        if not vs:
            return None
        num = den = 0.0
        tot = 0
        allv = []
        for v in vs:
            n = len(v)
            m = sum(v) / n
            num += sum((x - m) ** 2 for x in v)
            den += n - 1
            tot += n
            allv += v
        if den <= 0:
            return None
        return sum(allv) / len(allv), math.sqrt(num / den), tot

    def _unconfirmed_move(self, window: int,
                          remaining: Optional[float]) -> Optional[Tuple[str, object]]:
        """A knob whose last move has its predicted sample and did not confirm. Returns (knob, old).

        Only ever fires for a move this planner made and recorded a prediction for, only once the
        moved value has `n_required(delta)` observations, and only when `accept_decision` refuses.
        Returns None otherwise, which is the common case.
        """
        for _lead in [k for k in self._ll_partners if k not in self._move_from]:
            self._ll_partners.pop(_lead, None)       # the lead's move was confirmed or superseded: nothing left to revert together
        for short, entry in list(self._move_from.items()):
            old_value, delta, _w = entry[0], entry[1], entry[2]
            base_mean = entry[3] if len(entry) > 3 else None
            if not delta or delta <= 0:
                continue
            cur = (self._physical_latest_cfg or self._short_cfg()).get(short)
            if self._physical_latest_cfg is not None:
                if cur == old_value:
                    continue  # proposed change is not physically observed yet
                if len(entry) > 4 and cur != entry[4]:
                    logger.info("optimizer.cbandit.move_superseded", knob=short, value=cur,
                                predicted_value=entry[4], window=window)
                    self._move_from.pop(short, None)
                    continue
            if cur == old_value:
                self._move_from.pop(short, None)
                continue
            if self._verify == "per_step" and self._rate_mode and short not in self._relocating:
                _chg = [c for c in (self._physical_changes if self._physical_latest_cfg is not None
                                    else []) if c[0] == short]
                if _chg:
                    _sv = self._sv_verdict(short, old_value, cur, int(_chg[-1][2]), window, delta)
                    if _sv == "confirm":
                        _rec = self._sv_last.get(short, {})
                        logger.info("optimizer.cbandit.move_confirmed", window=window, knob=short, value=cur,
                                    verdict="confirmed_per_step", metric="step_time_interval",
                                    step_ci=_rec.get("step_ci"), exposure_ci=_rec.get("exposure_ci"),
                                    delta=_rounded(delta),
                                    reason="the 95% block-bootstrap interval of the step-time change excludes zero (faster)")
                        self._move_from.pop(short, None)
                        continue
                    if _sv == "revert":
                        return short, old_value
                    if _sv == "collect":
                        continue
            cand = list(self._samples.get((short, cur), ()))
            cand_tail = list(self._tail_samples.get((short, cur), ()))
            if self._rate_mode and short not in self._relocating:
                changes = [c for c in (self._physical_changes if self._physical_latest_cfg is not None
                                      else self._applied_changes) if c[0] == short]
                if not changes:
                    continue  # a proposal without an application has no after sample
                last = changes[-1]
                after = [(w, wall) for w, cfg, wall in self._credit_observations
                         if ((self._observation_first_steps.get(w, -1) > last[2])
                             if self._physical_latest_cfg is not None else w > last[2])
                         and cfg.get(short) == cur][-self._sample_window:]
                cand = [wall for _w, wall in after]
                cand_tail = [self._credit_tail[w] for w, _wall in after if w in self._credit_tail]
            base = list(self._samples.get((short, old_value), ()))
            base_tail = list(self._tail_samples.get((short, old_value), ()))
            # A MOVE PREDICTED FROM THE RECENT EXPOSED RUN IS VERIFIED AGAINST THAT RUN'S OWN PRE-MOVE
            # WINDOWS. The old value's arm also holds every window the knob spent at that value in an
            # EARLIER regime (halo-regimes X: 15 phase-1 windows at read_threads=1 and ~80 ms/step,
            # against a raw phase at ~270), so the arm-vs-arm test read a good move as worse and
            # reverted it. Same observation as for a relocation (4s.30): the baseline is the regime
            # the move left, as it was when the decision was taken.
            _snap = self._move_base_walls.get(short) if short not in self._relocating else None
            if _snap:
                base = list(_snap)
            if _snap:
                base_tail = list(self._move_base_tails.get(short, ()))

            # A RELOCATION IS VERIFIED AGAINST THE REGIME IT LEFT, NOT AGAINST A LIVE ARM.
            #
            # T3-0 4s.30. After a relocation the incumbent is never sampled again, so its arm holds
            # only PRE-move windows, and `accept_decision` then compares two different regimes as if
            # they were two arms of one. On `cf_fz_loop_n32_r1b` that compared `pfs` at 26.85 s,
            # four windows taken before the copy, against `local` at 32.38 s, ten windows taken
            # DURING the staging epoch and including the copy's own cost, and reverted the decision
            # that went on to run the cell at 5.6 s per window.
            #
            # So the baseline is the pre-move mean recorded at the move, the sample size is the
            # post-move `n_required` for the predicted effect, and the question asked is whether the
            # post-move regime is significantly WORSE than what was predicted for it. Same alpha,
            # no new constant. A knob move keeps the two-arm test unchanged, which is what D1 checks.
            if short in self._relocating:
                # PENDING IS NOT A VERDICT (4s.37). A relocation whose loader has not been rebuilt
                # has no post-move sample at all, so there is nothing to confirm and nothing to
                # refuse. Freeze f reverted 32 nodes in exactly this state, one window before the
                # rebuild would have shown the effect.
                if (short in self._reloc_effective_at
                        and self._reloc_effective_at.get(short) is None):
                    continue
                if not base_mean or base_mean <= 0:
                    continue
                need, _ = self.required_dwell(remaining, short=short, value=cur, delta=delta)
                if len(cand) < max(need, self._min_samples):
                    continue
                predicted = max(float(base_mean) - float(delta), 0.0)
                m = statistics.mean(cand)
                sd = statistics.stdev(cand) if len(cand) > 1 else 0.0
                if sd <= 0.0:
                    worse = m > predicted
                else:
                    t = (m - predicted) / (sd / math.sqrt(len(cand)))
                    worse = t > self._z(1.0 - self._alpha)
                if worse:
                    return short, old_value
                # A CONFIRMATION LEAVES A RECORD (T3-0 4s.39). Only the revert was ever logged, so
                # a run that kept a move said nothing about why, and "confirmed" had to be inferred
                # by replaying the record into a planner and looking at `_move_from` (4s.38). The
                # paper's claim is a count, and a count has to be readable from the run.
                logger.info(
                    "optimizer.cbandit.reloc_confirmed", window=window, knob=short, value=cur,
                    n=len(cand), mean=round(m, 4), base_mean=round(float(base_mean), 4),
                    predicted=round(predicted, 4), delta=round(float(delta), 4),
                    n_required=int(need), sd=round(sd, 4), verdict="confirmed",
                    reason="the relocated regime is not worse than the size it predicted")
                self._move_from.pop(short, None)
                continue

            _sig = None
            if _snap and len(_snap) >= 2:
                _m = sum(_snap) / len(_snap)
                _sig = (_m, math.sqrt(sum((x - _m) ** 2 for x in _snap) / (len(_snap) - 1)), len(_snap))
            need, _ = self.required_dwell(remaining, short=short, value=old_value, delta=delta,
                                          sigma_override=_sig)
            # JOINT VERIFICATION (opt-in, DFOPTIMIZER_CBANDIT_JOINT_VERIFY=1; STORMER-LOOP-DESIGN sec 5).
            # Relieving one domain can expose another (StorMer: a larger all-reduce bucket exposes periodic
            # input stalls), so the cheaper-domain move is followed by a move in the OTHER domain.  The
            # first move's arm then mixes windows from before and after the second move, and its own
            # verification reads the exposed stall as harm and reverts the very move the follow-up needs.
            # When another knob has moved since this one, verify the two as one step: the candidate is the
            # arm of the most recently moved knob (both moves applied, configuration stationary since),
            # the baseline is this knob's old arm (both knobs at their old values).  A failure reverts
            # the most recent move first (LIFO).  Default OFF = the historical per-arm verification.
            if self._joint_verify:
                # Configuration-keyed samples (D1b).  Candidate: windows that ran the CURRENT whole configuration
                # strictly after the most recent move of any pending knob (so a value tried, reverted and retried
                # earlier cannot leak in).  Baseline: windows at or before this move that ran the configuration
                # it left (this knob at its old value, every knob moved since restored to its own old value).
                pending = [(self._last_move_window.get(s2, -10**9), s2) for s2 in self._move_from
                           if s2 != short and self._last_move_window.get(s2, -10**9) > _w]
                t_last = max([_w] + [wm for wm, _s in pending])
                cur_cfg = dict(self._short_cfg())
                base_cfg = dict(cur_cfg)
                base_cfg[short] = old_value
                for _wm, s2 in pending:
                    base_cfg[s2] = self._move_from[s2][0]
                jcand = [wl for (wi, cf, wl) in self._series if wi > t_last and cf == cur_cfg]
                jbase = [wl for (wi, cf, wl) in self._series if wi <= _w and cf == base_cfg]
                if len(jcand) < need or len(jbase) < self._min_samples:
                    continue
                if accept_decision(jbase, jcand, alpha=self._alpha,
                                   effect_floor=self._effect_floor) != "accept":
                    if pending:                       # LIFO: the most recent move goes back first
                        latest = max(pending)[1]
                        return latest, self._move_from[latest][0]
                    return short, old_value
                logger.info(
                    "optimizer.cbandit.move_confirmed", window=window, knob=short, value=cur,
                    n=len(jcand), mean=round(statistics.mean(jcand), 4), base_n=len(jbase),
                    base_mean=round(statistics.mean(jbase), 4), delta=round(float(delta), 4),
                    n_required=int(need), verdict="confirmed_joint" if pending else "confirmed",
                    joint_with=[s2 for _wm, s2 in pending] or None,
                    reason="the stationary configuration accepts against the one it left")
                self._move_from.pop(short, None)
                continue
            # TAIL-AWARE VERIFICATION. The mean wall per step is dominated by a few long steps (a
            # stall is seconds against a ~0.3 s step), so a change that halves them can leave a
            # window mean unchanged and the mean test alone reverts it. The same per-step family
            # also gives each window's p90 step wall. A move is CONFIRMED when either the mean or
            # the p90 is significantly and meaningfully lower than the arm it left (the same
            # accept_decision, alpha and effect floor) and the mean is not significantly worse;
            # it is reverted only when neither is. Minimum-samples rule unchanged: each arm needs
            # `min_samples` windows of the metric.
            tail_ok = self._tail_accepts(base_tail, cand_tail, base, cand)
            if tail_ok and len(cand) >= self._min_samples and len(base) >= self._min_samples:
                logger.info(
                    "optimizer.cbandit.move_confirmed", window=window, knob=short, value=cur,
                    n=len(cand), mean=round(statistics.mean(cand), 4),
                    base_n=len(base), base_mean=round(statistics.mean(base), 4),
                    metric="p90_step_wall", tail_n=len(cand_tail),
                    tail=round(statistics.mean(cand_tail), 4),
                    base_tail_n=len(base_tail), base_tail=round(statistics.mean(base_tail), 4),
                    delta=round(float(delta), 4), n_required=int(need), verdict="confirmed_tail",
                    reason="the moved arm's p90 step wall accepts against the arm it left")
                self._move_from.pop(short, None)
                continue
            if len(cand) < need or len(base) < self._min_samples:
                continue
            if accept_decision(base, cand, alpha=self._alpha,
                               effect_floor=self._effect_floor) != "accept":
                return short, old_value
            # The symmetric record for an ordinary knob move: two live arms, the same test that
            # would have reverted it, and the same fields so both kinds count the same way.
            logger.info(
                "optimizer.cbandit.move_confirmed", window=window, knob=short, value=cur,
                n=len(cand), mean=round(statistics.mean(cand), 4),
                base_n=len(base), base_mean=round(statistics.mean(base), 4),
                metric="mean_wall",
                predicted=round(max(statistics.mean(base) - float(delta), 0.0), 4),
                delta=round(float(delta), 4), n_required=int(need), verdict="confirmed",
                reason="the moved arm accepts against the arm it left")
            self._move_from.pop(short, None)
        return None

    def _tail_accepts(self, base_tail, cand_tail, base, cand) -> bool:
        """Is the candidate arm's per-window p90 step wall significantly and meaningfully lower?

        Same decision rule as the mean (`accept_decision`: Welch p < alpha and relative gain >=
        effect_floor), guarded so a lower tail never excuses a significantly worse mean. False
        unless both arms have `min_samples` windows of the metric.
        """
        if len(base_tail) < self._min_samples or len(cand_tail) < self._min_samples:
            return False
        guard = len(base) > 1 and len(cand) > 1
        return accept_decision(base_tail, cand_tail, alpha=self._alpha,
                               effect_floor=self._effect_floor,
                               guard_base=base if guard else None,
                               guard_cand=cand if guard else None) == "accept"

    @staticmethod
    def _median_lower_bound(values, alpha: float) -> Optional[float]:
        """Distribution-free lower confidence bound of the median, from the order statistics.

        Orchestrator decision, 2026-09-05 (T3-0 4s.24). The domain delta is an ESTIMATE and carries
        its own uncertainty, so the planner uses the bound rather than the point estimate. 4s.23's
        eight spurious moves came from a median resting on `min_samples` windows; the remedies on
        offer were a window count, which is a constant, and the previous epoch, which does not exist
        in epoch 0. This is neither: it is the same alpha the acceptance test already documents.

        The k-th smallest observation is a lower bound for the median with confidence
        `P(Bin(n, 0.5) >= k)`, so the bound is the largest k whose tail is still at least
        `1 - alpha`. With few samples no k qualifies and the bound is the sample MINIMUM, which is
        the behaviour that matters: an early epoch of a few small windows cannot admit a move on a
        median it has not earned. As n grows the qualifying k rises and the bound climbs toward the
        median.

        n=3 at alpha=0.05 gives the minimum (P(Bin<=0) = 0.125 > 0.05); n=10 gives the second
        smallest; n=30 the tenth. No distributional assumption is made anywhere.
        """
        xs = sorted(v for v in values if v is not None)
        n = len(xs)
        if n == 0:
            return None
        if n == 1:
            return xs[0]
        best = 1
        cum = 0.0
        denom = float(2 ** n)
        for k in range(1, n + 1):
            cum += math.comb(n, k - 1) / denom
            if cum <= alpha:
                best = k
            else:
                break
        return xs[best - 1]

    def _epoch_domain_exposed(self, dom: str) -> Optional[float]:
        """Mean exposed time of one domain per window, over this epoch so far.

        The most any move of a knob that owns `dom` can recover, taken as the LOWER CONFIDENCE
        BOUND of the epoch-to-date median rather than the median itself (4s.24). A bound on a bound:
        measured, not a model of what the move would do, and used only to ask whether an effect of
        that size could be verified in the run that is left.
        """
        if self._epoch_windows < self._min_samples:
            return None
        vals = [v for v in (self._epoch_dom.get(dom) or ()) if v is not None]
        if not vals:
            return None
        lb = self._domain_bound(vals)
        return lb if lb and lb > 0.0 else None

    def _capacity_limit(self, short: str) -> Optional[float]:
        """The largest value of `short` the node's own processors can hold, or None.

        MEASURED, NOT CHOSEN (T3-0 4s.43). A worker knob spends processes: each rank runs one main
        process plus `value` workers, and the node has a processor count. So

            limit = procs_per_node / ranks_per_node - 1

        is arithmetic on two measurements, which is what the method's no-threshold rule requires.
        On Corona, 48 physical cores and 8 ranks per node give a limit of 5, and the shipped
        `num_workers=8` places 72 processes on 48 cores.

        None whenever either measurement is missing or the knob is not the one that spends them,
        and then every rule below is inert. An absent measurement is not a capacity of infinity.
        """
        if not self._capacity_knob or short != self._capacity_knob:
            return None
        if not self._capacity_procs or not self._capacity_ranks:
            return None
        if self._capacity_ranks <= 0:
            return None
        return max(float(self._capacity_procs) / float(self._capacity_ranks) - 1.0, 0.0)

    def _measured_decrease_effect(self, short: str, cur, lower) -> Optional[float]:
        """What DECREASING `short` from `cur` to `lower` was worth, from this run's own samples.

        THE DEFECT THIS ADDRESSES. `predicted_effect` prices a knob move at the exposed time of the
        domain the knob OWNS, which is a lower bound on the true effect whenever a knob's cost falls
        outside its own domain -- and over-provisioning is exactly that case: extra workers cost
        CPU that the compute stage pays for, so the input domain cannot see them. Measured twice:
        cosmoflow's relocation realised 6 to 12 s per window more than the read-time price (4s.37),
        and StorMer's W=8 to W=4 is 22 percent of the step in the static sweep against a priced
        0.5 s (4s.43).

        This is the honest form: the difference between the two arms' observed walls, which needs
        the lower value to have been VISITED. It returns None when it has not, and the caller then
        has no measured effect to admit on -- the floor the method's own rules impose.
        """
        hi = list(self._samples.get((short, cur), ()))
        lo = list(self._samples.get((short, lower), ()))
        if len(hi) < self._min_samples or len(lo) < self._min_samples:
            return None
        d = (sum(hi) / len(hi)) - (sum(lo) / len(lo))
        return d if d > 0 else None

    def _domain_bound(self, vals):

        """The lower confidence bound on the domain's per-window exposure.

        THREE ESTIMATORS, ONE SHIPPED (T3-0 4s.42). The median's bound is what 4s.24 chose and what
        every quoted result was taken on. The two mean bounds exist because a bimodal exposure --
        StorMer stalls on one step in `num_workers`, so at a cadence that does not divide the period
        some windows carry the whole stall and the rest carry none -- puts the median near zero
        while the mean is seconds. Both are distribution free at the same alpha:

            hoeffding   mean - range * sqrt(ln(1/alpha) / 2n)
            bootstrap   the alpha-quantile of the resampled mean

        MEASURED, AND NEITHER IS ADOPTED: on `stcz2_loop_r1` with the stall on the wire the three
        give 0.0143, 0.1871 and 1.0674 s against a window-wall sigma of 7.14, so `n_required` is
        1,948,687 / 11,444 / 352 against a horizon of 14. The estimator is not what blocks that
        cell; the cadence is, and a window that spans a whole stall period makes the series
        unimodal and the median sound again. Default stays `median`.
        """
        if self._domain_delta_bound == "hoeffding":
            n = len(vals)
            if n < 2:
                return None
            rng = max(vals) - min(vals)
            return (sum(vals) / n) - rng * math.sqrt(math.log(1.0 / self._alpha) / (2.0 * n))
        if self._domain_delta_bound == "bootstrap":
            n = len(vals)
            if n < 2:
                return None
            rnd = random.Random(1729)          # fixed seed: a decision must be reproducible
            means = sorted(sum(rnd.choices(vals, k=n)) / n for _ in range(2000))
            return means[int(self._alpha * len(means))]
        return self._median_lower_bound(vals, self._alpha)

    def predicted_effect(self, short: str, dom: str,
                         reloc_gain: Optional[float] = None,
                         window_exposure: Optional[float] = None) -> Tuple[Optional[float], str]:
        """The candidate's predicted per-window effect, and where it came from.

        Project owner decision, 2026-09-05 (T3-0 4s.23). The planner acts only on moves whose effect
        it can predict, verify within the horizon and afford. That replaces the 2 percent
        `effect_floor`, which 4s.20 measured as being far below the window-to-window noise of BOTH
        cells -- so no affordable sample could ever resolve it, and the floor was not a minimum
        interesting effect but a number nothing could clear.

          relocation      the physical price's gain per window (4s.14), already measured
          domain owner    the epoch-to-date exposed time of the domain it owns, a measured bound
          neither         no delta, and the candidate is not admitted

        No preferred direction: both neighbours of a knob get the same delta and the acceptance test
        decides between them.

        `window_exposure` (STORMER-LOOP-DESIGN sec 17, opt-in `DFOPTIMIZER_CBANDIT_COMM_EFFECT_CAP_WINDOW`):
        the CURRENT window's own measured exposure for `own`, when the caller has it. A `bucket_cap_mb`
        move cannot physically remove more comm time than the comm domain is currently measured to be
        paying THIS window -- the epoch-to-date accumulator above can run ahead of that (it pools across
        windows, some of which may have carried more exposure than the freshest one), so this clamps
        DOWN ONLY, never up: `ex = min(ex, window_exposure)`. Scoped to `own == "comm"` only, per the
        orchestrator's spec; other domains are unaffected. Disabled by default -- it can only make the
        planner more conservative, never license a new action, but ships opt-in like every other
        addition in this file until reviewed.
        """
        if short in self._relocating:
            return (reloc_gain, "relocation_gain") if reloc_gain and reloc_gain > 0 else (
                None, "relocation_not_payable")
        own = self._domain.get(short)
        if not own:
            return None, "no_domain"
        if self._rate_mode and self._verify == "per_step":
            # The measured bound a move of this knob can recover, from the per-step series of the
            # family it targets over the current configuration: the lower end of the 95% block-bootstrap
            # interval of the mean exposure. It needs steps, not min_samples windows.
            got = self._sv_predicted_exposure(short)
            if got is not None:
                ex, src = got
                return (ex, "domain_exposed:%s" % src) if ex > 0.0 else (None, "no_exposed_time")
        if self._rate_mode and self._epoch_windows < self._min_samples:
            return None, "insufficient_samples:%d<%d" % (self._epoch_windows, self._min_samples)
        ex = self._epoch_domain_exposed(own)
        if ex is None or ex <= 0.0:
            # The epoch-to-date lower bound is None or zero when the domain was unexposed for most of
            # the epoch and has only just become exposed (a regime shift: halo-regimes X, z9 -> raw,
            # where 15 zero-input windows put the median bound at 0 for the whole raw phase while
            # the attributed storage wait was 10-130 ms per step). That is a statement about the
            # PAST epoch, not about the exposure now being measured. See _recent_run_effect.
            rec = self._recent_run_effect(short, own)
            if rec is not None:
                return rec
            return None, "no_exposed_time"
        if (self._comm_effect_cap_window and own == "comm" and window_exposure is not None
                and math.isfinite(window_exposure) and window_exposure >= 0.0):
            ex = min(ex, window_exposure)
            if ex <= 0.0:
                return None, "no_exposed_time"
        return ex, "domain_exposed:%s" % own

    def _recent_run_effect(self, short: str, dom: str) -> Optional[Tuple[float, str]]:
        """Predicted per-window effect of the NEXT level of `short`, from the measured exposure now.

        Used only when the epoch-to-date bound of the owned domain is unusable (see predicted_effect)
        AND nothing measured exists at any higher level of this knob (a measured comparison, once it
        exists, decides instead; this is the first step, not a replacement for that test).

        Measured exposure: the exposure of the domain's current unbroken run of exposed windows, taken
        at its MINIMUM (a lower bound of what is exposed now), and only once the run is as long as the
        planner's own `min_samples` (the same evidence count every other estimate here waits for).

        Overlap/Amdahl form for a worker pool: a stage's service rate scales with its workers, so the
        wait it exposes scales as cur/next. Going cur -> next therefore removes the share
        1 - cur/next of the measured exposure. Bounded above by the measured exposure by construction,
        a lower bound on the removable wait whenever the stage is the binding one (the wait left after
        the move can only be smaller). No new constant: the levels are the knob's own, the run length
        is `min_samples`. The apply cost is NOT netted here; `_pick` prices it against the horizon.
        """
        levels = self._levels.get(short) or []
        try:
            cur = _num(self._short_cfg().get(short))
            higher = [v for v in levels if _num(v) > cur]
        except (TypeError, ValueError):
            return None
        if not higher or cur is None or cur <= 0:
            return None
        if any(self._samples.get((short, v)) for v in higher):
            return None
        run = []
        for v in reversed(list(self._epoch_dom.get(dom) or ())):
            if v is None or v <= 0.0:
                break
            run.append(float(v))
        if len(run) < self._min_samples:
            return None
        nxt = self._next_up(short, self._short_cfg().get(short))
        if nxt is None:
            return None
        share = 1.0 - cur / _num(nxt)
        if share <= 0.0:
            return None
        # The exposure SUSTAINED over the last `min_samples` windows of the run (its minimum), not the
        # run's own minimum: a run that began on a ramp would otherwise be priced by its first window.
        self._recent_run_n = len(run)
        sustained = min(run[:self._min_samples])
        return sustained * share, "domain_exposed_recent_run:%s:amdahl:%s->%s" % (dom, cur, nxt)

    def _recent_run_sigma(self) -> Optional[Tuple[float, float, int]]:
        """(mean, sigma, n) of the window wall over the recent exposed run, or None.

        The effect predicted from the recent run is tested against the noise of that same run. The
        knob's within-arm samples (and the epoch wall) span every regime the run has been through, so
        after a shift their spread is the shift itself, which is the signal and not the noise a
        verification of this move has to see through (same argument as _within_arm_sigma).
        """
        n = min(self._recent_run_n, len(self._epoch_wall_hist))
        if n < 2:
            return None
        walls = list(self._epoch_wall_hist)[-n:]
        mean = sum(walls) / n
        var = sum((x - mean) ** 2 for x in walls) / (n - 1)
        return mean, math.sqrt(var), n

    def required_dwell(self, remaining: Optional[float] = None,
                       short: Optional[str] = None, value=None,
                       delta: Optional[float] = None,
                       sigma_override: Optional[Tuple[float, float, int]] = None) -> Tuple[int, dict]:
        """How many windows a value must be held to be worth testing, from the observed variance.

        Returns (dwell, why) where `why` carries the terms so a decision can be read back. The
        dwell is `max(min_samples, n_required)` capped at the remaining horizon; `why["reachable"]`
        is False when the required sample exceeds the horizon, and the caller must then leave the
        value uncharacterised rather than declare it exhausted.

        On cosmoflow's dwell cell the steady-state spread is 1.3 percent of the wall, so
        `n_required` is 1 and this reduces to `min_samples` -- the behaviour 4p measured, unchanged.
        On unet3d dev2 the wall carries 10 to 42 s stalls on a 4.37 s median window, so it grows,
        which is the whole point: `neighbourhood_exhausted_no_help` may not be declared from a
        sample that could not have contained the cost (4s.17).
        """
        if self._sv_active() and short is not None and delta:
            got = self._sv_required_steps(short, float(delta), remaining)
            if got is not None:
                need_steps, swhy = got
                K = max(int(self._control_steps), 1)
                dwell = max(1, int(math.ceil(need_steps / K)))
                swhy.update(floor=self._min_samples, effect_floor=self._effect_floor, sigma_from="step_series",
                            n_required_steps=need_steps, dwell_steps=need_steps, dwell_windows=dwell)
                if not swhy.get("reachable", True):
                    return int(remaining // K), swhy
                return dwell, swhy
        st = sigma_override
        src = "recent_run"
        if st is None:
            st = self._within_arm_sigma(short, value) if short else None
            src = "within_arm"
        if st is None:
            st = self._epoch_wall_stats()
            src = "epoch_wall"
        why = {"floor": self._min_samples, "alpha": self._alpha, "power": self._power,
               "effect_floor": self._effect_floor, "sigma_from": src}
        if st is None:
            why.update(reason="no_wall_variance_yet", reachable=True)
            return self._min_samples, why
        mean, sigma, n = st
        # `delta` is the CANDIDATE'S PREDICTED EFFECT when the caller has one (4s.23). The
        # effect_floor fraction remains only for callers that do not, which after this change is
        # the escalation's opt-in power check.
        if delta is None:
            delta = self._effect_floor * mean
        why.update(sigma_wall=round(sigma, 5), mean_wall=round(mean, 5),
                   delta=round(delta, 5), n_observed=n)
        if delta <= 0.0:
            why.update(reason="no_effect_size", reachable=True)
            return self._min_samples, why
        z = self._z(1.0 - self._alpha / 2.0) + self._z(self._power)
        n_req = int(math.ceil((z * sigma / delta) ** 2))
        why["n_required"] = n_req
        dwell = max(self._min_samples, n_req)
        needed_steps = dwell * self._control_steps if self._rate_mode else dwell
        if self._rate_mode:
            why.update(n_required_steps=n_req * self._control_steps, dwell_steps=needed_steps, horizon_unit="steps")
        if remaining is not None and remaining > 0 and needed_steps > remaining:
            why.update(reason="exceeds_horizon", reachable=False,
                       horizon=int(remaining))
            return int(remaining // self._control_steps) if self._rate_mode else int(remaining), why
        why.update(reason="derived", reachable=True)
        return dwell, why

    def _epoch_src_gbs(self) -> Optional[float]:
        """The node's observed read rate over this epoch so far, GB/s, or None if unusable.

        Same three refusals as `_epoch_bytes_per_window`, for the same reasons: an epoch that lost
        records did not observe the reads it thinks it did, and fewer than `min_samples` windows is
        a window rather than an epoch.
        """
        if self._epoch_dropped:
            return None
        if self._epoch_windows < self._min_samples:
            return None
        if self._epoch_bw_n <= 0 or self._epoch_bw_sum <= 0.0:
            return None
        return self._epoch_bw_sum / float(self._epoch_bw_n)

    def _relocation_gain_per_window(self, bytes_per_window, src_gbs,
                                    exposed_s) -> Optional[float]:
        """What a relocation PHYSICALLY does: it changes the time the readers spend reading.

        Orchestrator decision, 2026-09-05, after T3-0 4s.13 showed the previous form assumed that
        everything above the floor was relievable by moving the data, which is the hidden assumption
        that the barrier IS storage. It held on cosmoflow and it did not on unet3d, where the
        measured input domain is 3 ms of a 4 s window and the wall correlates with comm at 1.0000.

            gain = bytes_per_window x (1 / src_gbs - 1 / dest_gbs)      clamped at 0
            gain = min(gain, input + comm)                              nothing saves more than
                                                                        the exposed time

        Two measured rates and a byte count, no constant chosen to make a cell fail and no share of
        the wall below which a domain is ignored. It refuses BY ARITHMETIC when the destination is
        not faster than the source, which is the case on unet3d: the node already reads at about
        3.69 GB/s against a 3.300 GB/s local ceiling.

        Checked against both recorded cells before deployment (4s.14): predicted 11.39, 11.58 and
        11.92 s per window on the three cosmoflow dwell reps against a realised 11.50, 11.51 and
        11.51, and 0.4024 s per window on unet3d against a realised 0.6602 -- accepting where the
        stage paid and refusing in all 305 candidate windows where it did not.

        Returns None when either rate or the byte count is missing, which the caller treats the way
        it treats a missing horizon: a positive-cost relocation is declined and nothing is invented.
        """
        if not bytes_per_window or bytes_per_window <= 0:
            return None
        if not src_gbs or src_gbs <= 0 or self._dest_read_gbs <= 0:
            return None
        gain = float(bytes_per_window) * (1.0 / float(src_gbs) - 1.0 / self._dest_read_gbs)
        gain = max(gain, 0.0)
        if exposed_s is not None:
            gain = min(gain, max(float(exposed_s), 0.0))
        return gain

    def _epoch_bytes_per_window(self) -> Optional[float]:
        """Mean bytes this node read per window so far this epoch, or None if it may not be used.

        None for three distinct reasons, all of which the caller must treat as "do not price a
        positive-cost relocation", the same discipline the horizon contract has:

          * fewer than `min_samples` windows of this epoch seen, so there is no epoch-scale
            observation yet, only a window;
          * the epoch recorded ring drops, so its byte total is a floor and not a measurement;
          * no bytes at all, which is an absence and not a rate of zero.
        """
        if self._epoch_dropped:
            return None
        if self._epoch_windows < self._min_samples:
            return None
        if self._epoch_bytes <= 0.0:
            return None
        return self._epoch_bytes / float(self._epoch_windows)

    def _destination_bounded_gain(self, wall_s, compute_s, bw_gbs,
                                  bytes_per_window=None) -> Optional[float]:
        """Per-window gain a RELOCATION may be priced at, bounded by where the bytes are going.

        THE DECISION THIS FIXES, measured on both sides. `un_loop_n8_10e_r1` staged on all eight
        nodes at window 15 with a predicted 1239 to 1549 s against a declared 465, the copy took
        936 to 956 s and bought about 38 s per epoch, and over ten epochs the loop would have run
        about 2840 s against Original's 1930 (T3-0 section 4s). `cf_loop_n32_r8` made the same kind
        of decision and won, 412 s per epoch becoming 134.

        NOTHING THE RULE LOOKED AT SEPARATED THEM. Both were at_ceiling, both had comm dominant with
        no comm knob, both had a tiny input domain. The ceiling comparison asks how bad the SOURCE
        is and the answer was "saturated" in both cases. What differs is the DESTINATION: the
        cosmoflow node was reading Lustre at 0.71 GB/s and moving to an SSD that delivers about
        2.05, while the unet3d node was already reading at 2.89 to 3.36 GB/s and moving to a
        SLOWER device.

        After a relocation the window cannot be shorter than the time the destination needs to
        deliver the bytes, nor shorter than the compute still to be done. Those two overlap in a
        pipelined loader, so the post-move window is their max and not their sum:

            bytes           = node_read_gbs * wall
            post_move_input = bytes / dest_read_gbs
            gain            = max(wall - max(compute, post_move_input), 0)

        On cosmoflow that is 10.69 s per window against a realised 11.58, where the deployed
        input+comm cap said 7.45. On unet3d it is exactly 0 on all eight agents. In one sentence:
        do not move the data to a device slower than the one you are already achieving, whatever
        comm says.

        Returns None when the destination rate or the bandwidth reading is missing, and the caller
        must then decline a positive-cost relocation rather than fall back to a looser bound.
        """
        if self._dest_read_gbs <= 0 or not wall_s or wall_s <= 0:
            return None
        # The bytes are taken from the EPOCH when an epoch-scale reading exists, and a relocation
        # is not priced at all when it does not. `bw_gbs x wall` remains the definition for every
        # non-relocating use of this helper and for the unit cases that state the formula itself.
        if bytes_per_window is not None:
            read_bytes = float(bytes_per_window)
        elif bw_gbs is not None:
            read_bytes = float(bw_gbs) * float(wall_s)
        else:
            return None
        post_move_input = read_bytes / self._dest_read_gbs
        return max(float(wall_s) - max(float(compute_s or 0.0), post_move_input), 0.0)

    def _relocation_cost_s(self, short: str, value, obs: dict, bw_gbs,
                           bytes_per_window=None) -> Optional[float]:
        """What the copy costs on THIS cell, rather than a constant measured on another one.

        465 s was measured moving 285 GB per node on cosmoflow at 32 nodes and was then applied
        unchanged to unet3d's 438 GB per node, where the copy actually took 936 to 956 s. The bytes
        are observable: this node read `bw * wall` in a window and `windows_per_epoch` of them make
        an epoch, which is one pass over its shard.

        The stage rate is itself scale dependent, 0.61 GB/s for one node's share at 32 nodes
        against 0.46 GB/s at 8 nodes each copying 438 GB, so it wants the same
        min(single, aggregate/N) treatment the Lustre ceiling has. Until that is measured this uses
        one configured rate and says so in the record.
        """
        if short not in self._relocating or self._stage_rate_gbs <= 0:
            return None
        wpe = obs.get("windows_per_epoch")
        wall = obs.get("wall")
        if not wpe or not wall:
            return None
        # One epoch's bytes, from an epoch's worth of windows. The single-window form is kept only
        # for the unit cases that state the formula, because it is what produced a 58 s estimate of
        # a 945 s copy (4s.8).
        if bytes_per_window is not None:
            per_window = float(bytes_per_window)
        elif bw_gbs is not None:
            per_window = float(bw_gbs) * float(wall)
        else:
            return None
        bytes_per_node = per_window * float(wpe)
        return bytes_per_node / self._stage_rate_gbs

    # ------------------------------------------------------------------ Little's-law sizing
    def _sizing_update(self, m: dict, wall: float, domains: dict, window: int) -> None:
        """This window's Little's-law level of every sizable parallelism knob (planner/sizing.py)."""
        self._ll = {}
        if self._sizing != "littles_law" or not self._rate_mode:
            return
        knobs = {}
        for kid, short in self._short.items():
            kd = self.knobs[kid]
            cur = self._short_cfg().get(short)
            if short in sizing.STAGES_OF_KNOB and kd.type is int and kd.range and cur is not None:
                knobs[short] = (cur, kd.range[0], kd.range[1])
        if not knobs:
            return
        self._ll = sizing.size_knobs(m, wall, domains, knobs)
        if self._ll:
            logger.info("optimizer.cbandit.sizing", window=window, knobs=self._ll,
                        exposed_wait_s={d: round(float(domains.get(d, 0.0) or 0.0), 6)
                                        for d in ("input", "decode", "delivery") if d in domains})

    def _ll_level(self, short: str, current) -> Optional[int]:
        """The Little's-law level of `short` when it lies above the current value, else None (geometric/refinement applies)."""
        if self._sizing != "littles_law" or not self._rate_mode:
            return None
        e = self._ll.get(short)
        if not e or self._refine.get(short):         # after a revert: refinement, not another model jump
            return None
        t = e["target"]
        return int(t) if t > _num(current) else None

    def _add_level(self, short: str, value) -> None:
        levels = self._levels.get(short)
        if levels is not None and value not in levels:
            self._levels[short] = sorted(set(levels) | {value}, key=_num)

    def _ll_underprovisioned(self, short: str) -> bool:
        cur = self._short_cfg().get(short)
        return cur is not None and self._ll_level(short, cur) is not None

    def _ll_partner_moves(self, lead: str, lead_to, cfg_now: dict, doms: dict) -> Dict[str, int]:
        """Other under-provisioned stages of the input side, sized in the SAME plan as an LL-sized lead (one apply, one verification).

        A partner joins only when the model sizes it above its current value, its own stage's exposed wait is positive in this
        window, and its raise to that level has not been refuted at the exposure it has now.
        """
        out: Dict[str, int] = {}
        for short in self._short.values():
            if short == lead or short not in sizing.STAGES_OF_KNOB or self._domain.get(short) not in ("input", "decode", "delivery"):
                continue
            cur = cfg_now.get(short)
            lv = self._ll_level(short, cur) if cur is not None else None
            if lv is None:
                continue
            stage = sizing.STAGES_OF_KNOB[short][0]
            if not float(doms.get(sizing.STAGE_DOMAIN[stage], 0.0) or 0.0) > 0.0:
                continue
            if self._sv_active():
                exp = self._sv_predicted_exposure(short)
                if self._sv_refuted(short, cur, lv, exp[0] if exp else None) is not None:
                    continue
            out[short] = lv
        return out

    def _neighbourhood(self, short: str, current) -> List:
        """{stay, one level down, one level up} -- a coordinate step, not a global jump."""
        levels = self._levels.get(short) or [current]
        if current in levels:
            i = levels.index(current)
        else:
            # Current value is off-grid (e.g. an app default between levels): snap to the nearest.
            i = min(range(len(levels)), key=lambda j: abs(_num(levels[j]) - _num(current)))
        out = [levels[i]]
        factor = self._geo_factor(short)
        if factor is not None:
            half = [v for v in levels if _num(v) < _num(levels[i]) and _num(v) <= _num(levels[i]) / factor]
            if not half and i > 0 and short not in _POOL_KNOBS:
                half = [levels[i - 1]]      # a refinement level (e.g. 50) steps back to its grid neighbour
            if half:
                out.append(max(half, key=_num))
            up = self._next_up(short, levels[i])
            ll = self._ll_level(short, current)
            if ll is not None:
                self._add_level(short, ll)       # the model's level joins the grid so the candidate list stays on it
                up = ll                          # and replaces the geometric raise: above it or between current and it
            if up is not None:
                out.append(up)
            return out
        if i > 0:
            out.append(levels[i - 1])
        if i + 1 < len(levels):
            out.append(levels[i + 1])
        return out

    def _geo_factor(self, short: str) -> Optional[float]:
        """Geometric step factor of `short` in rate mode, or None for a plain grid-neighbour knob.

        A KnobDef declares its own `step_factor`; otherwise the default table applies (worker/thread
        counts x2, bucket_cap_mb x4, i.e. 25 -> 100 -> 400 on the 25/50/100/200/400 grid).
        """
        if not self._rate_mode:
            return None
        for kd in self.knobs.values():
            if kd.id.split(".")[-1] == short and getattr(kd, "step_factor", None):
                return float(kd.step_factor)
        return _GEOMETRIC_FACTOR.get(short)

    def _next_up(self, short: str, current):
        """The next HIGHER level a raise of `short` proposes, or None at the declared maximum.

        Geometric knobs (see `_geo_factor`) move by their factor: current x2 for worker/thread
        counts (1->2->4->8), x4 for bucket_cap_mb (25->100->400), capped at the declared maximum
        (the largest level not above current x factor that is still above current). After an
        observed revert the next raise from the restored level is one refinement step (the next
        level on the grid), then geometric again. Other knobs keep the grid neighbour.
        """
        levels = sorted(self._levels.get(short) or [current], key=_num)
        higher = [v for v in levels if _num(v) > _num(current)]
        if not higher:
            return None
        factor = self._geo_factor(short)
        if factor is None or self._refine.get(short):
            return higher[0]
        within = [v for v in higher if _num(v) <= factor * _num(current)]
        return max(within, key=_num) if within else higher[0]

    def _pick(self, short: str, current, domain: str, exposed_s: Optional[float] = None,
              remaining: Optional[float] = None, cost_override: Optional[float] = None,
              measured_gain_only: bool = False):
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
        if self._sv_active() and short not in self._relocating:
            # A move targets one family's exposure and relieves it by RAISING the resource; lowering is
            # reached only by a verdict (revert) or the measured-capacity path, never by exploration.
            cands = [lv for lv in cands if lv == current or _num(lv) > _num(current)]
        self._last_pick_note = None
        if self._strict_explore and short in self._shipped and short not in self._relocating:
            # A2(2): never explore below the shipped value (over-provisioned capacity decreases are
            # admitted earlier, on their own measured path, and never reach _pick).
            try:
                floor = _num(self._shipped[short])
                cands = [lv for lv in cands if lv == current or _num(lv) >= floor]
            except (TypeError, ValueError):
                pass
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
            # EXPLORATION IS BOUGHT WHERE IT IS CHEAP, AND NEVER WITH A 950 s COPY.
            #
            # Orchestrator decision, 2026-09-05, on the evidence of T3-0 4s.9. An untried arm's UCB
            # bonus is optimism, not a prediction, and for a knob whose first visit costs 0.206 s
            # that is exactly the right way to pay for information. For a relocation it was buying
            # 1211 s of notional gain with a 2.90 s per-window bonus over a 417-window horizon, and
            # that is what cleared a 952 s copy twice on a cell where the copy returned nothing.
            #
            # So a relocating candidate is priced on MEASURED gain only: the exposure the move can
            # relieve, bounded by the destination, computed from the epoch-to-date byte rate. The
            # bonus is removed from its gain AND from its score, because leaving it in the score
            # would still let optimism choose the arm it may no longer price.
            if measured_gain_only:
                bonus = 0.0
            if lv == current and remaining is not None and self._cost_w != 0.0:
                # The incumbent is the reference, not an exploration target. Giving it an
                # optimism bonus subtracts that bonus from every candidate's apparent gain and can
                # make a move fail the payback comparison after its own gain already cleared cost.
                # Preserve the legacy UCB exactly when no priced remaining-run decision is being
                # made, including COST_W=0 and knobs declared with zero cost.
                bonus = 0.0
            cost_s = 0.0 if lv == current else self._move_cost(short, lv)
            # A cost derived from the bytes THIS cell moves replaces the declared constant when it
            # is computable, and is recorded as such so a decision can be read back.
            if lv != current and cost_override is not None and cost_override > 0.0:
                cost_s = float(cost_override)
            weighted_cost_s = self._cost_w * cost_s
            if measured_gain_only and lv != current:
                # What the move can actually relieve. `exposed_s` is already
                # min(the domain time this move addresses, the destination bound), both measured,
                # so it IS the gain rather than a cap on an optimistic one. Once the arm has been
                # sampled, its own measured improvement is the stricter of the two and wins.
                relievable = max(float(exposed_s), 0.0) if exposed_s is not None else 0.0
                measured = max(cur_mean - mean, 0.0)
                gain_per_window_s = relievable if n == 0 else min(measured, relievable)
            else:
                gain_per_window_s = max(cur_mean - (mean - bonus), 0.0)
                if exposed_s is not None:
                    gain_per_window_s = min(gain_per_window_s, max(float(exposed_s), 0.0))
            # C3 ABLATION (`_horizon_weight`): a relocation's one-time cost is normally repaid
            # from `gain_per_window_s * remaining` -- the run left to amortise it. With horizon
            # weighting off, price a SINGLE window's gain against the FULL cost instead, whatever
            # `remaining` says: cost weighting (`_cost_w`) is untouched, only the horizon term is
            # gone. This can fail in both directions the design predicts: a real early gain that
            # would have repaid over many windows now looks unpayable against one window's worth,
            # and (paired with the `verifiable` change below) a late-run move that should be
            # refused for lack of horizon no longer has that veto either.
            gain_s = (gain_per_window_s if not self._horizon_weight else
                      (gain_per_window_s * remaining if remaining is not None else None))
            payable = (lv == current or self._cost_w == 0.0 or cost_s <= 0.0
                       or (gain_s is not None and gain_s > weighted_cost_s))

            # Multiplying the predicted wall by the observed remaining count puts it in the same
            # units as cost_for(): wall seconds over the rest of this run. With no horizon, only a
            # zero-cost move has a defensible score.
            #
            # THE PREDICTED GAIN MUST ENTER THE SCORE, NOT ONLY `payable` (T3-0 4s.28).
            #
            # For an untried arm `mean` is the incumbent's own mean, so without a term for what the
            # move is predicted to ACHIEVE its score is the incumbent's score plus the move's cost:
            # strictly worse, always, whatever the gain. The UCB bonus used to supply that term by
            # accident, and removing it for relocations (0eefc37, "measured gain only") removed the
            # only thing that ever let a relocation win the acquisition.
            #
            # Measured on the first frozen cosmoflow loop rep: every one of 32 agents had a
            # verifiable relocation gain of about 16.7 s per window from window 2, `payable: true`
            # with gain 1225 s against a 453 s cost, and `_pick` returned `pfs` on all 1,888 of
            # them because local scored 2157.3 against pfs's 1704.3. Zero plans were published and
            # the cell did not stage.
            #
            # So a candidate priced on measured gain is scored on the wall it is predicted to
            # produce: the incumbent's mean less that gain, floored at zero. This is the same
            # quantity `payable` already uses, applied to the ranking as well as the admission.
            predicted_mean = mean - bonus
            if measured_gain_only and lv != current:
                predicted_mean = max(cur_mean - gain_per_window_s, 0.0)
            if remaining is not None:
                score = predicted_mean * remaining + weighted_cost_s
            elif weighted_cost_s <= 0.0:
                score = predicted_mean
            else:
                score = float("inf")
            if not payable:
                score = float("inf")
            detail.append({"level": lv, "n": n, "mean": round(mean, 4),
                           "bonus": round(bonus, 4), "gain_s": _rounded(gain_s),
                           "cost_s": round(cost_s, 4), "payable": payable,
                           # So a decision can be read back without inferring which rule priced it.
                           "measured_gain_only": bool(measured_gain_only and lv != current),
                           "predicted_mean": round(predicted_mean, 4),
                           "score": _rounded(score)})
            if score < best_score:
                best, best_score = lv, score
        if self._strict_explore and best != current and best_score != float("inf"):
            # A2(1): an exact tie for the best score carries no directional prediction.
            tied = [d for d in detail if d["level"] != best and d["score"] is not None
                    and d["level"] != current and d["score"] == _rounded(best_score)]
            if tied:
                self._last_pick_note = "tie_no_prediction:%s~%s" % (best, tied[0]["level"])
                return current, detail
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
        # What this knob's own applies have measurably cost beats any declaration.
        measured = self._measured_apply_cost(short, cfg.get(short), value)
        if measured is not None:
            return measured
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
        by = obs.get("node_read_bytes") or 0.0
        src = "reader_posix"
        if by <= 0.0:
            # FALL BACK TO THE BROAD POSIX LAYER, and say which was used.
            #
            # reader_posix needs `_reader` in the event category, which _fix_file_posix_category
            # writes only when the file's path matches a configured purpose, and on the streaming
            # path it does not: measured on t4m_storagegate4, reader_posix_read_size_sum was 0 on
            # all 125 windows while posix_read_size_sum was 11.56 GB per window on all 125. That is
            # an analyzer defect and it is recorded as one in T3-0 section 4m; it is not this
            # planner's to fix.
            #
            # The broad layer is a superset: it counts every read the ranks make, not only the ones
            # under the data folder. For a DLIO training run with checkpointing off that is the
            # corpus and little else, so it is the right quantity here and slightly OVER-states the
            # read, which pushes the comparison towards "at the ceiling" -- the conservative
            # direction, since it favours moving the data over asking for more readers that may not
            # help.
            by = obs.get("node_read_bytes_any") or 0.0
            src = "posix"
        wall = obs.get("wall") or 0.0
        if by <= 0.0 or wall <= 0.0:
            return None
        self._last_bw_source = src
        # Bytes over the window's ELAPSED time, which is the rate the filesystem delivered to this
        # node. An earlier version divided by the busiest rank's read busy time, which is not a
        # bandwidth when 8 ranks read concurrently: it returned 22.2 GB/s for one node, above the
        # 17.4 GB/s the whole 32-node cluster was measured at. Busy time is kept as a diagnostic.
        return (by / 1e9) / wall

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
        if not self._barrier_redirect:
            return False
        return dom == "comm" and "comm" not in actionable

    @staticmethod
    def _skew_from_metrics(m: dict, wall) -> Optional[dict]:
        """The measured skew link for this window, or None (suppressed, never invented)."""
        def f(key):
            v = m.get(key)
            try:
                v = float(v)
            except (TypeError, ValueError):
                return None
            return v if math.isfinite(v) else None
        cov, a = f("domain_skew_coverage"), f("domain_input_skew_attributed_time")
        if cov != 1.0 or a is None or not a > 0.0:
            return None
        typ = {k: f("domain_%s_time_proc_typ" % k) for k in ("input", "decode", "comm", "compute")}
        if any(v is None or v < 0.0 for v in typ.values()):
            return None
        e, w = f("domain_input_skew_excess_time"), f("domain_comm_barrier_wait_time")
        late = f("domain_skew_late_proc_input_time")
        if e is None or w is None or a > min(e, w) + 1e-9 or a > typ["comm"] + 1e-9:
            return None                       # inconsistent with A = min(E, W) <= comm: refuse
        wall_typ = f("window_wall_time_proc_typ")
        wall_typ = wall_typ if wall_typ and wall_typ > 0 else float(wall)
        primed = {"input": typ["input"] + a, "decode": typ["decode"],
                  "comm": max(typ["comm"] - a, 0.0), "compute": typ["compute"]}
        unattr = max(wall_typ - sum(typ.values()), 0.0)
        return {"A": a, "E": e, "W": w, "L": late, "domains": primed, "unattributed": unattr}

    def _decision_path(self, dom, storage_suspect: bool, via: str = "", obs=None) -> dict:
        """Which path produced a decision, recorded on EVERY decision record (A1) with this node's
        configuration at decision time (A2 per-node timeline: control is node-local)."""
        sk = (obs or {}).get("skew")
        nd = (obs or {}).get("actionable_nondominant")
        if via:
            path = via
        elif nd and dom == nd["domain"]:
            path = "actionable_nondominant(%s,exposure=%.4g,dominant=%s%s)" % (
                nd["domain"], nd["exposure"], nd["dominant"],
                (",skew_A=%.4g" % sk["A"]) if sk else "")
        elif sk and dom == "input":
            path = "skew_attributed(A=%.4g,E=%.4g,W=%.4g,L=%s)" % (
                sk["A"], sk["E"], sk["W"], "na" if sk["L"] is None else "%.4g" % sk["L"])
        elif storage_suspect:
            path = "barrier_redirect"
        elif dom == "input":
            path = "direct_input"
        elif dom == "comm":
            path = "comm"
        else:
            path = "dominant_domain:%s" % dom
        return {"decision_path": path, "skew_link_enabled": self._skew_link,
                "barrier_redirect_enabled": self._barrier_redirect,
                "strict_explore": self._strict_explore,
                "node": os.uname().nodename,
                "config_at_decision": {k: v for k, v in self._short_cfg().items()}}

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

        # Little's law: a stage the model sizes above its current value is asked before one it finds adequately provisioned.
        return sorted(live, key=lambda s: (0 if self._ll_underprovisioned(s) else 1, price(s), owned.index(s)))

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
        if self._strict_explore and short in self._shipped and short not in self._relocating:
            # A neighbour below the shipped value is never explored (`_pick`, A2(2)), so it can never be
            # sampled: requiring it made `untried_neighbour:<below shipped>` permanent and a cheaper coordinate
            # that had been tried and refused (prefetch_size 2 -> 4 reverted) could never hand over to the
            # next knob of the domain (job-level TF run: read_threads was reached in 1 of 3 runs, 47 windows in).
            try:
                floor = _num(self._shipped[short])
                alts = [lv for lv in alts if _num(lv) >= floor]
            except (TypeError, ValueError):
                pass
        if not alts:
            return True, "neighbourhood_exhausted"
        if self._sv_active():
            got = self._sv_escalation_state(short, current, alts)
            if got is not None:
                return got
        # THE SAMPLE MUST BE ABLE TO CONTAIN THE COST, not merely reach min_samples.
        #
        # 4s.17: `read_threads=1` was confirmed 936 times with `neighbourhood_exhausted_no_help` on
        # three-window samples, on a cell where the cost of that value appears in one window in six
        # and lasts longer than the sample. `exhausted` is a strong claim and it may not be made
        # from evidence that is under-powered for the effect the test is looking for.
        # DEFAULT OFF (4s.21). This breaks the escalation loop, and the escalation loop is how a
        # storage candidate is reached at all, so an under-powered cheaper knob can silently remove
        # the cosmoflow stage. Until `delta`'s meaning is settled, nothing on by default may be able
        # to do that. Enable with DFOPTIMIZER_CBANDIT_POWER_ESCALATION=1.
        need = (self.required_dwell(short=short)[0] if self._power_escalation
                else self._min_samples)
        base = list(self._samples.get((short, current), ()))
        if len(base) < need:
            return False, ("incumbent_under_sampled" if len(base) < self._min_samples
                           else "incumbent_under_powered:%d<%d" % (len(base), need))
        for lv in alts:
            cand = list(self._samples.get((short, lv), ()))
            if len(cand) < need:
                return False, (f"untried_neighbour:{lv}" if len(cand) < self._min_samples
                               else "neighbour_under_powered:%s:%d<%d" % (lv, len(cand), need))
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

        # EXPLORATION NEEDS A CHARACTERISED INCUMBENT (orchestrator, 2026-09-05, T3-0 4s.20).
        #
        # This check used to sit BELOW the exploration branch, so a move was admitted on the
        # acquisition's say-so whenever the CANDIDATE was under-sampled, whatever was known about
        # the value being left. On dev2 that let `read_threads 4 -> 1` through on a sample that
        # could not measure 4, and 4s.18's dwell then made 1 unjudgeable for the rest of the run:
        # the planner sat on an uncharacterised value it had never checked, which is a guess and
        # not a decline.
        #
        # A planner that has not measured the noise of the value it HOLDS cannot compare anything
        # against it, so it does not move. The bar is the same derived one: `n_required` from the
        # incumbent's own within-arm sigma. On a smooth cell that is `min_samples` and nothing
        # changes. On a bursty cell the incumbent is never characterised inside the horizon, no
        # exploration is admitted, and the loop holds the shipped default for the run -- which is
        # the honest result on unet3d: no knob effect is measurable within the run, so do not act.
        #
        # There is no first-move exemption. The first move has no predecessor, but it does have an
        # incumbent, and that is what this asks about.
        # DEFAULT OFF, and 4s.20 says why. The rule is right in principle and it fails its own
        # acceptance condition on the cell it was required not to disturb: cosmoflow's WITHIN-RUN
        # window walls run 13.5 to 38.1 s, a 43 percent spread, so `n_required` is about 3625
        # against 74 windows and its window-2 stage is blocked too. The 1.3 percent figure the
        # decision relied on is the steady-state spread ACROSS REPS, not the window-to-window
        # spread the power calculation needs. Enable with DFOPTIMIZER_CBANDIT_EXPLORE_GATE=1.
        # THE EXPLORE GATE IS RETIRED (4s.23). It asked whether the incumbent was characterised
        # against a 2 percent floor, which no cell could clear, so it blocked the cosmoflow stage as
        # well as unet3d's move. Admission is now decided upstream by the predicted effect and
        # whether it can be verified in the horizon, which is the same question asked with a number
        # that means something.
        if len(cand) < self._min_samples:
            return True, "explore_under_sampled"
        if len(base) < self._min_samples:
            return False, "incumbent_under_sampled"

        verdict = accept_decision(base, cand,
                                  alpha=self._alpha, effect_floor=self._effect_floor)
        if verdict != "accept" and self._tail_accepts(
                list(self._tail_samples.get((short, current), ())),
                list(self._tail_samples.get((short, candidate), ())), base, cand):
            return True, "accept_tail"
        return verdict == "accept", verdict

    # --- Planner-compatible decision entry ---
    def process_window_completion(self, window_index: int, *, terminal: bool = False) -> List[ActionPlan]:
        """Finalize one pending fact window when its native successor completes.

        A completion is not a finding.  Terminal completion is deliberately a
        no-op: end-of-stream must never manufacture a last action.
        """
        if terminal or self._cur_window == -1 or self._pending_finding is None:
            return []
        # Completion is a watermark for the *immediate* successor, not a
        # permission to silently bridge missing/reordered native windows.
        # A later marker would stamp a decision with a window whose evidence
        # belongs to an older one, so it is intentionally declined.
        # The flat view emitted beside the pending finding has the same native
        # window identity.  It is an ordinary same-window completion, not an
        # ordering fault and not a decision trigger; wait for a fact-free
        # immediate successor.  Older/duplicate markers are equally inert.
        if window_index <= self._cur_window:
            return []
        if window_index != self._cur_window + 1:
            logger.warning("optimizer.cbandit.window_completion.ordering_gap",
                           pending_window=self._cur_window,
                           completion_window=window_index)
            return []
        marker = _dc.replace(self._pending_finding, window_index=window_index)
        plans = self.process_finding(marker)
        # `process_finding` has installed the marker as a new current window.
        # Remove it: no-fact windows never accumulate control evidence.
        self._cur_window = -1
        self._win_metrics = {}
        self._pending_finding = None
        return plans

    def process_finding(self, finding: DiagnosisFindingMsg) -> List[ActionPlan]:
        if not self.knobs:
            return []
        w = finding.window_index
        if self._cur_window == -1:
            self._cur_window = w
            self._pending_finding = finding
            self._accumulate(finding)
            return []
        if w == self._cur_window:
            self._pending_finding = finding
            self._accumulate(finding)
            return []

        completed = self._cur_window
        obs = self._finalize_window(completed)
        plans: List[ActionPlan] = []
        if obs is not None:
            self._record(obs["cfg"], obs["wall"], window=completed, tail=obs.get("tail"))

            # Verification runs even when this window has no actionable exposure.
            self._sv_sync_watches()
            self._sv_update_side_effects()
            revert = self._unconfirmed_move(completed, obs.get("remaining"))
            if revert is not None:
                rshort, rback = revert
                nxt = dict(self._short_cfg())
                nxt[rshort] = rback
                joint = self._ll_partners.pop(rshort, None)
                if joint is not None:            # a jointly sized plan is verified and reverted as one
                    for pshort, pold in joint[1].items():
                        nxt[pshort] = pold
                        self._last_move_window[pshort] = completed
                self._planned_cfg[w + 1] = dict(nxt)
                self._last_move_window[rshort] = completed
                self._move_from.pop(rshort, None)
                logger.info("optimizer.cbandit.reverted", window=completed, knob=rshort,
                            back_to=rback,
                            reason="not confirmed at its predicted effect",
                            joint_with=(dict(joint[1]) if joint else None))
                self._cur_window = w
                self._win_metrics = {}
                self._pending_finding = finding
                self._accumulate(finding)
                return self._plans_for(nxt, finding, self._domain[rshort])

            if self._win_metrics.get("observation_only") == 1.0:
                logger.info("optimizer.cbandit.observation", window=completed,
                            reason="measured zero exposure; recorded without a new move")
                self._cur_window = w
                self._win_metrics = {}
                self._pending_finding = finding
                self._accumulate(finding)
                return plans

            # 1. CROSS-LAYER DIRECTION: the dominant domain by measured time. A comparison, not a
            #    threshold -- this is the part the paper's contribution rests on.
            doms = obs["domains"]
            wall = obs["wall"]
            if self._domain_blind:
                # C1 ABLATION. Collapse the breakdown to one aggregate entry that no registered
                # knob owns (see the `_domain_blind` comment in __init__). `domain_times` on the
                # decision record below will show this single "wall" key, which is the mechanism
                # signature a reader checks for -- not whether a knob moved.
                doms = {"wall": float(wall)}
            actionable = {d for d in self._domain.values()}
            if self._comm_needs_skew and obs.get("skew_coverage") != 1.0:
                actionable.discard("comm")
            # Read unconditionally: later logging on the move/stay path (`unattributed=...`) needs
            # it in both branches, even though only the non-B2 branch lets it compete in `ranked`.
            unattr = obs.get("unattributed", 0.0)
            if self._b2_input_only:
                # E2: gate on input's OWN exposure, never on an argmax against "other". "other"
                # (compute+comm+anything else B2 cannot separate) is deliberately never constructed
                # as a ranked candidate here, so it can never win or suppress this decision.
                input_v = doms.get("input")
                if input_v is not None and math.isfinite(input_v) and input_v > 0.0:
                    ranked = [(input_v, "input")]
                else:
                    logger.info(
                        "optimizer.cbandit.decision", window=completed, decision="decline",
                        reason="b2_input_only: no positive input exposure this window",
                        dominant="input", gain_s=0.0, cost_s=0.0,
                        horizon_count=obs.get("remaining"), horizon_unit=obs.get("remaining_unit"),
                        escalation_state="not_actionable",
                        decision_path="b2_input_only_absent",
                        domain_times={k: round(float(v), 4) for k, v in doms.items()},
                        wall=round(wall, 4))
                    ranked = []
            else:
                # The unexplained remainder COMPETES in the argmax. It owns no knob, so if it is the
                # largest share of the step the planner has nothing to move and declines -- which is
                # the correct response to "most of this step is something I cannot name". Actuation
                # cost ranks candidates only after raw exposed time has named the dominant domain.
                ranked = sorted(((v, d) for d, v in doms.items()), reverse=True)
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
            if (ranked and self._win_metrics.get("liveflow_v1_attribution") != 1.0
                    and self._barrier_is_storage_suspect(ranked[0][1], actionable)):
                if any(self._domain[s] == "input" for s in self._short.values()):
                    storage_suspect = True
                    suspect_bw = self._node_read_bandwidth(obs)
                    ranked = [(doms.get("input", 0.0), "input")] + [
                        r for r in ranked if r[1] != "input"]

            if self._win_metrics.get("liveflow_v1_attribution") == 1.0:
                # An exposure without a registered actuator is a finding, not a veto.
                blocked = [(value, domain) for value, domain in ranked if domain not in actionable]
                if blocked:
                    logger.info("optimizer.cbandit.unactuatable_exposure", window=completed,
                                exposures=dict((domain, value) for value, domain in blocked))
                ranked = [(value, domain) for value, domain in ranked
                          if domain in actionable and value > 0.0
                          and any(self._domain[short] == domain and self._rejections.get(short, 0) < self._retire_after
                                  for short in self._short.values())]
                storage_suspect = False

            # C2 ABLATION (`_always_act`): guards ONLY this decline, nothing past it. When it
            # fires, `ranked[0]` may still name a domain with no owned knob, and the `owned = [...]`
            # lookup a few lines down then comes back empty, so the window silently emits nothing
            # -- same numeric outcome as a decline, but with no `decision="decline"
            # reason="dominant share of the step is unattributed"` record. That absence, not a
            # changed knob value, is the mechanism signature to check (T3-0 4s.56).
            nondominant = None
            if (self._actionable_priority and ranked and ranked[0][1] not in actionable
                    and not storage_suspect):
                cands = [(float(v), d) for d, v in doms.items() if d in actionable and float(v) > 0.0]
                # GUARD (judge batch 2, C2a): the "dominant share unattributed" decline is a safety
                # guard.  Prioritization never fires when the window's unattributed residual is at least
                # the actionable exposure it would chase (a residual-dominant window still declines).
                if cands and float(obs.get("unattributed", 0.0) or 0.0) >= max(cands)[0]:
                    cands = []
                if cands:
                    _ev, _ed = max(cands)
                    nondominant = {"domain": _ed, "exposure": _ev, "dominant": ranked[0][1]}
                    obs["actionable_nondominant"] = nondominant
                    ranked = [(_ev, _ed)] + [r for r in ranked if r[1] != _ed]
            if ranked and ranked[0][1] not in actionable and not self._always_act:
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
                    node_read_bytes_any=float(obs.get("node_read_bytes_any") or 0.0),
                    ceiling_gbs=self._lustre_ceiling_gbs or None,
                    bw_source=self._last_bw_source,
                    storage_suspect=storage_suspect,
                    **self._decision_path(ranked[0][1], storage_suspect, "decline_unattributed", obs=obs),
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
                        promoted = {s for s in ("data_source", "transfer_size") if s in order}
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
                        promoted = set()
                    # VERIFY AT THE PREDICTED SIZE, OR REVERT.
                    #
                    # "Admitted because the effect is verifiable in the horizon" is only meaningful
                    # if the verification actually happens. Once the moved value has the sample its
                    # own prediction required, the acceptance test is run against the value it came
                    # from; if it does not accept, the move goes back. A move that was worth making
                    # on a prediction is not worth keeping without one.
                    cfg_now = self._short_cfg()
                    short, cur, choice, detail, why = None, None, None, [], "no_change"
                    tried = []
                    # The relocation's predicted per-window gain, kept across the candidate loop so
                    # the decision record carries the prediction the run was made on (dev3).
                    reloc_gain_last = None
                    reloc_horizon_last = None
                    dwell_last = None
                    delta_last = None
                    for cand_short in order:
                        cand_cur = cfg_now[cand_short]
                        # THE EXPOSURE A STORAGE ACTUATOR CAN RELIEVE, in a storage-suspect
                        # window, is not the input domain alone.
                        #
                        # `_pick` caps a candidate's per-window gain at `exposed_s`, and passing
                        # the input domain there is what made staging structurally unreachable at
                        # 32 nodes. Measured on cf_loop_n32_r2: input is 0.29 s per window while
                        # comm is 11.19 s, so the cap was 38x too small, `data_source -> local`
                        # priced its gain at a median 5.6 s (max 363) against a 465 s cost, came
                        # back `payable: false` in all 1873 of its appearances, and the acquisition
                        # never proposed it. The move was refused by the PRICE, not by the
                        # acceptance test, which is why no amount of sampling would have reached it.
                        #
                        # Option 1's whole claim is that on a bulk-synchronous step the barrier IS
                        # the slowest rank's storage time. A redirection that asserts that for the
                        # CHOICE of knob and then denies it for the PRICE of the knob cannot ever
                        # act, which is exactly what reps 1 and 2 measured: zero moves in 1702 and
                        # 1696 at-ceiling windows. Storage candidates are therefore priced against
                        # input + comm; every other candidate keeps the domain's own exposure.
                        cand_exposed = doms[dom]
                        if storage_suspect and cand_short in promoted:
                            cand_exposed = float(doms.get("input", 0.0)) + float(
                                doms.get("comm", 0.0))
                        # PRICED ON THE EPOCH'S EXPOSURE, NOT THE DECISION WINDOW'S (4s.39). The
                        # window a decision happens to be taken in is one sample of a distribution
                        # this cell spreads over 43 percent: on r1f windows 0 to 2 expose 34.3,
                        # 15.8 and 9.9 s, and the decision landed on the smallest. The cap is the
                        # same epoch-to-date lower-bound median the predicted effect already uses,
                        # so no new estimator and no new constant. Off by default until a rep runs.
                        if self._reloc_effect_at_boundary and cand_short in self._relocating:
                            _ep = self._epoch_domain_exposed(dom)
                            if storage_suspect and cand_short in promoted:
                                _in = self._epoch_domain_exposed("input") or 0.0
                                _cm = self._epoch_domain_exposed("comm") or 0.0
                                _ep = (_in + _cm) or None
                            if _ep is not None:
                                cand_exposed = float(_ep)
                        # AND THEN PRICED PHYSICALLY, for a knob that RELOCATES the bytes.
                        # input+comm is what a relocation may be CREDITED with at most, and it is
                        # a BOUND rather than the gain: what the move actually delivers is the
                        # change in the time the readers spend reading, two measured rates and a
                        # byte count. See _relocation_gain_per_window. A relocating knob with no
                        # epoch-scale byte count or source rate gets zero rather than a looser
                        # bound, so it is declined.
                        cand_cost_override = None
                        if cand_short in self._relocating:
                            # The bound applies in EVERY window, not only under the barrier
                            # redirect: a relocation is limited by its destination whether the
                            # dominant domain is comm or input.
                            # PRICED FROM THE EPOCH, NEVER FROM THIS WINDOW, and not at all from
                            # an epoch that lost records. With no epoch-scale reading the cap is
                            # zero and the derived cost is absent, so a positive-cost relocation is
                            # declined -- the same discipline as a missing horizon or destination
                            # rate, and the answer 4s.8 says the single-window form got wrong.
                            reloc_bytes = obs.get("epoch_bytes_per_window")
                            reloc_src = obs.get("epoch_src_gbs")
                            phys = self._relocation_gain_per_window(
                                reloc_bytes, reloc_src, cand_exposed)
                            reloc_gain_last = phys
                            cand_exposed = 0.0 if phys is None else phys
                            cand_cost_override = self._relocation_cost_s(
                                cand_short, None, obs, None,
                                bytes_per_window=reloc_bytes)
                        # THE PREDICTED EFFECT, AND WHETHER IT CAN BE VERIFIED IN THE RUN LEFT.
                        #
                        # NOT when the acceptance gate is disabled. DFOPTIMIZER_CBANDIT_ACCEPT=0 is
                        # the no-gate ablation and must recover the previous behaviour exactly, the
                        # same reason the dwell is skipped there. Caught by C8, whose ungated arm
                        # stopped thrashing on one seed and so stopped being a control.
                        # `window_exposure` is passed ONLY when the sec 17 flag is on -- the default
                        # call must stay byte-identical to before this section existed (a caller that
                        # stubs `predicted_effect` with the OLD signature, e.g.
                        # test_cbandit_no_predicted_effect.py, must keep working when the flag is off;
                        # a keyword arg present with value None still breaks a stub with no such
                        # parameter at all).
                        if self._comm_effect_cap_window:
                            delta, delta_src = (self.predicted_effect(
                                cand_short, dom, reloc_gain=reloc_gain_last,
                                window_exposure=doms.get(dom))
                                if (self._accept_enabled and self._predicted_effect_gate)
                                else (None, "gate_disabled"))
                        else:
                            delta, delta_src = (self.predicted_effect(
                                cand_short, dom, reloc_gain=reloc_gain_last)
                                if (self._accept_enabled and self._predicted_effect_gate)
                                else (None, "gate_disabled"))
                        rem_w = obs.get("remaining")
                        # THE HORIZON A RELOCATION IS PAID OVER STARTS AT THE REBUILD (4s.39). The
                        # windows left in this epoch read the old source whatever the knob says, so
                        # they cannot repay the copy. Applied to the relocating candidate only:
                        # every other knob takes effect in the window after the plan.
                        if (self._reloc_effect_at_boundary and cand_short in self._relocating
                                and rem_w is not None):
                            _wpe = obs.get("windows_per_epoch")
                            if _wpe:
                                _left = max(float(_wpe) - float(self._epoch_windows), 0.0)
                                rem_w = max(float(rem_w) - _left, 0.0)
                                reloc_horizon_last = rem_w
                        # A DECREASE BELOW A MEASURED CAPACITY IS ADMISSIBLE ON THE MEASUREMENT
                        # THAT SAYS IT IS OVER-PROVISIONED (4s.43).
                        #
                        # Priced first from this run's own two arms, which is the honest form and
                        # needs the lower value to have been visited. When it has not, the capacity
                        # comparison itself is the admission: the node cannot hold the processes
                        # this value asks for, the lower neighbour has never been tried, and the
                        # cost of finding out is one loader rebuild. That is the SAME
                        # explore-an-untried-neighbour path a relocation is admitted on, and it is
                        # what a per-domain predicted effect can never reach, because the cost of
                        # over-provisioning is not in the domain the knob owns.
                        _cap = self._capacity_limit(cand_short)
                        _cap_admit = None
                        if _cap is not None and _num(cand_cur) > _cap:
                            _lower = [v for v in (self._levels.get(cand_short) or [])
                                      if _num(v) < _num(cand_cur)]
                            if _lower:
                                # ONE COORDINATE STEP by default, which is the search this planner
                                # is: it descends 8 -> 6 -> 4 and learns what each level is worth.
                                # `to_limit` jumps straight to the highest level the capacity can
                                # hold, which reaches 4 in one move and never measures 6. Both are
                                # measured; the difference is what the run ends up knowing.
                                _fit = [v for v in _lower if _num(v) <= _cap]
                                _next_down = (max(_fit, key=_num) if (_fit and self._capacity_step
                                                                      == "to_limit")
                                              else max(_lower, key=_num))
                                _meas = self._measured_decrease_effect(
                                    cand_short, cand_cur, _next_down)
                                _cap_admit = (_next_down, _meas)
                                if _meas is not None:
                                    delta, delta_src = _meas, "measured_decrease"
                                elif _cap_admit is not None:
                                    # FREEZE I FIX (T3-0 4s.54, 2026-09-07): was `elif delta is
                                    # None`, which only blanked the per-domain price when one was
                                    # already absent -- so a capacity admission SURVIVED only when
                                    # there was nothing to override it, and was silently vetoed
                                    # whenever `predicted_effect` had already set `delta` to the
                                    # owned domain's exposure, which is exactly the over-provisioned
                                    # case (4s.43: the domain a knob owns cannot see the cost of
                                    # over-provisioning). The capacity finding is a claim about the
                                    # NODE; the domain price is a claim about one DOMAIN's exposure;
                                    # the first must not be outranked by the second. Blanking `delta`
                                    # whenever a capacity admission exists (not only when it was
                                    # already None) lets the explore-an-untried-neighbour path below
                                    # fire on its own terms, and stops `required_dwell` downstream
                                    # (guarded on `delta is not None`) from ever sizing a dwell off a
                                    # number the method has established cannot see this cost.
                                    delta, delta_src = None, "capacity_over_provisioned"
                                logger.info(
                                    "optimizer.cbandit.capacity", window=completed,
                                    knob=cand_short, value=cand_cur, limit=round(_cap, 3),
                                    to=_next_down,
                                    measured_effect=(round(_meas, 4) if _meas else None),
                                    procs_per_node=self._capacity_procs,
                                    ranks_per_node=self._capacity_ranks,
                                    reason="the node cannot hold the processes this value asks for")
                        if (delta is None and _cap_admit is not None
                                and self._accept_enabled and self._predicted_effect_gate):
                            # Untried lower neighbour under a measured capacity: explore it.
                            _to = _cap_admit[0]
                            if not self._samples.get((cand_short, _to)):
                                nxt = dict(self._short_cfg())
                                nxt[cand_short] = _to
                                self._planned_cfg[w + 1] = dict(nxt)
                                self._last_move_window[cand_short] = completed
                                _bn, _bm = self._stats.get((cand_short, cand_cur), (0, 0.0))
                                self._move_from[cand_short] = (cand_cur, None, completed, _bm)
                                logger.info(
                                    "optimizer.cbandit.decision", window=completed,
                                    decision="move", knob=cand_short, current=cand_cur, chose=_to,
                                    accept="explore_over_provisioned",
                                    **self._decision_path(dom, storage_suspect,
                                                          "capacity_over_provisioned"),
                                    dominant_domain=dom, domain_times={
                                        k: round(float(v), 4) for k, v in doms.items()},
                                    capacity_limit=round(_cap, 3), gain_s=0.0, cost_s=0.0,
                                    horizon_count=rem_w, horizon_unit=obs.get("remaining_unit"),
                                    wall=round(float(wall), 4), candidates=[])
                                return self._plans_for(nxt, finding, dom)
                        if delta is None and self._accept_enabled \
                                and self._predicted_effect_gate:
                            tried.append({"knob": cand_short, "verdict": "no_predicted_effect",
                                          "escalation": delta_src})
                            if short is None:
                                short, cur, choice, detail, why = (
                                    cand_short, cand_cur, cand_cur, [], "no_predicted_effect")
                            continue
                        need_v, vwhy = ((self.required_dwell(
                            rem_w, short=cand_short, value=cand_cur, delta=delta,
                            sigma_override=(self._recent_run_sigma()
                                            if str(delta_src).startswith("domain_exposed_recent_run")
                                            else None)))
                                        if delta is not None else (0, {}))
                        # NO HORIZON IS NOT AN UNBOUNDED ONE. "Admitted when n_required fits the
                        # windows remaining" cannot be satisfied when there are no windows
                        # remaining to compare against, and treating a missing horizon as passing
                        # is how all eight of 4s.23's spurious moves got in: every one of them was
                        # at window 530 of 539, where the record carries no horizon at all. Same
                        # discipline as the price, which declines a positive-cost move without one.
                        # C3 ABLATION (`_horizon_weight`): drop the `bool(rem_w)` half. A move at
                        # or near the end of the run, with no windows left to repay it, is no
                        # longer vetoed for that reason alone -- paired with the `gain_s` change in
                        # `_pick`, this is what lets an unpriced-horizon controller relocate where
                        # it cannot amortise.
                        verifiable = bool(vwhy.get("reachable", True)) and (
                            bool(rem_w) or not self._horizon_weight)
                        delta_last = {"delta": _rounded(delta), "source": delta_src,
                                      "n_required": vwhy.get("n_required"),
                                      "verifiable": verifiable}
                        if not verifiable:
                            tried.append({"knob": cand_short,
                                          "verdict": "unverifiable_in_horizon",
                                          "escalation": "n_required=%s horizon=%s"
                                                        % (vwhy.get("n_required"),
                                                           vwhy.get("horizon") or rem_w)})
                            if short is None:
                                short, cur, choice, detail, why = (
                                    cand_short, cand_cur, cand_cur, [], "unverifiable_in_horizon")
                            continue
                        cand_choice, cand_detail = self._pick(
                            cand_short, cand_cur, dom, exposed_s=cand_exposed,
                            remaining=rem_w, cost_override=cand_cost_override,
                            measured_gain_only=(cand_short in self._relocating))
                        if cand_choice != cand_cur:
                            # ONE CHANGE IN FLIGHT. No move on any knob until the previous move has
                            # its verdict (confirmed or reverted), which needs `min_samples` CLEAN
                            # post-apply windows of the moved arm.
                            _inflight = self._inflight_move()
                            # A RAMP CONTINUES: the same knob moving further in the direction of its
                            # unverified move is one change in flight, not a second one (halo regimes
                            # X, geometric steps: 1 -> 2 -> 4 must not wait for a verdict on 1 -> 2).
                            # The next verdict is then taken against the level it left.
                            # Only once the move is APPLIED: with an applied-vector snapshot the physical value
                            # must already be the new one (an unapplied plan is not a ramp).
                            _ramp = (_inflight == cand_short and not self._sv_active()
                                     and cand_cur > self._move_from[cand_short][0]
                                     and cand_choice > cand_cur
                                     and (self._physical_latest_cfg is None
                                          or self._physical_latest_cfg.get(cand_short) == cand_cur))
                            if _inflight is not None and not _ramp:
                                cand_choice = cand_cur
                                self._last_pick_note = "verdict_pending:%s" % _inflight
                        if cand_choice != cand_cur:
                            _veto = self._apply_cost_veto(cand_short, cand_cur, cand_choice,
                                                          delta, rem_w, completed)
                            if _veto is None and self._sv_active():
                                _veto = (self._sv_refuted(cand_short, cand_cur, cand_choice, delta)
                                         or self._sv_gate(cand_short, cand_cur, cand_choice, delta, rem_w,
                                                          self._measured_apply_cost(cand_short, cand_cur,
                                                                                    cand_choice)))
                                if _veto is not None:
                                    logger.info("optimizer.cbandit.side_effect_veto", window=completed,
                                                knob=cand_short, current=cand_cur, to=cand_choice,
                                                reason=_veto)
                            if _veto is not None:
                                cand_choice, self._last_pick_note = cand_cur, _veto
                        if cand_choice == cand_cur:
                            may_escalate, esc_state = self._escalation_state(cand_short, cand_cur)
                            tried.append({"knob": cand_short, "verdict": "stay",
                                          "escalation": esc_state,
                                          **({"pick": self._last_pick_note}
                                             if self._last_pick_note else {})})
                            if short is None:      # remember the cheapest, for the log and the ack
                                short, cur, choice, detail = (cand_short, cand_cur,
                                                              cand_choice, cand_detail)
                                why = esc_state
                            # A PROMOTED KNOB THAT STAYS MUST NOT END THE WINDOW.
                            #
                            # The escalation gate exists to stop a costlier actuator being reached
                            # before the cheaper ones are exhausted. When the ceiling verdict
                            # promotes a data-moving knob to the FRONT, that ordering inverts the
                            # gate's assumption: the expensive knob is now first, so its "wait for
                            # the cheaper ones" answer stops the cheaper ones from being asked at
                            # all. Measured on cf_loop_n32_r1 and r2: data_source stayed with
                            # `untried_neighbour:local` in 1772 and 1873 candidate slots, broke the
                            # loop each time, and NOT ONE of the 1702 and 1696 at-ceiling windows
                            # moved any knob. The promotion made the loop less active, not more.
                            #
                            # Promotion is about priority, not about exhaustion, so a promoted knob
                            # that declines hands the window on instead of ending it.
                            if cand_short in promoted:
                                continue
                            # Staying alone is not evidence of exhaustion. A costlier actuator may
                            # pass only after all cheaper neighbours were sampled and rejected by
                            # accept_decision, or after the coordinate reaches a boundary.
                            if not self._escalate or not may_escalate:
                                break
                            continue
                        accepted, cand_why = self._accept_move(cand_short, cand_cur, cand_choice)
                        # NOT when the acceptance gate is disabled. DFOPTIMIZER_CBANDIT_ACCEPT=0
                        # is the no-gate ablation and must recover the previous behaviour exactly;
                        # the dwell floor exists to make the gate's evidence reachable, so applying
                        # it with the gate off throttles the control arm and destroys the
                        # comparison. Caught by C8, whose ungated arm stopped being able to thrash.
                        if accepted and self._accept_enabled and self._min_dwell > 0:
                            held = completed - self._last_move_window.get(cand_short, -10**9)
                            need, dwell_why = self.required_dwell(
                                obs.get("remaining"), short=cand_short, value=cand_cur,
                                delta=delta)
                            if held < need:
                                accepted, cand_why = False, "dwell:%d<%d" % (held, need)
                                dwell_last = dwell_why
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
                                # The epoch-scale reading a relocation was priced from, and whether
                                # it was usable. Without these a decision cannot be read back:
                                # 4s.8's staging looked identical in the log to a sound one.
                                epoch=obs.get("epoch"),
                                epoch_windows=obs.get("epoch_windows"),
                                # The physical price's own three inputs and its prediction, so a
                                # staging decision can be read back without re-deriving it.
                                epoch_src_gbs=_rounded(obs.get("epoch_src_gbs")),
                                dest_read_gbs=self._dest_read_gbs or None,
                                reloc_gain_per_window=_rounded(reloc_gain_last),
                                reloc_horizon_windows=_rounded(reloc_horizon_last),
                                # The derived dwell's own terms, so a hold can be read back.
                                dwell=dwell_last,
                                predicted_effect=delta_last,
                                epoch_bytes_per_window=_rounded(
                                    obs.get("epoch_bytes_per_window")),
                                epoch_dropped=obs.get("epoch_dropped"),
                                drops_known=obs.get("drops_known"),
                                horizon_s=_rounded(horizon_s),
                                horizon_count=obs.get("remaining"),
                                horizon_unit=obs.get("remaining_unit"),
                                escalation_state=tried,
                                # Recorded on EVERY decision so the release rule can be counted off
                                # the log: a window where the dominant domain was comm, no knob
                                # owned it, and the storage candidates were admitted instead.
                                storage_suspect=storage_suspect,
                                **self._decision_path(dom, storage_suspect, obs=obs),
                                storage_headroom=suspect_reason,
                                node_read_gbs=(round(suspect_bw, 4)
                                               if suspect_bw is not None else None),
                                ceiling_gbs=self._lustre_ceiling_gbs or None,
                                unattributed=round(float(unattr), 4),
                                n_current=len(self._samples.get((short, cur), ())),
                                wall=round(obs["wall"], 4), candidates=detail)
                    if choice != cur:
                        self._last_move_window[short] = completed
                        # What this move predicted, so it can be checked at that size later.
                        if choice != cur:
                            # The pre-move regime's mean is recorded AT THE MOVE, because after a
                            # relocation the incumbent's arm stops being sampled and the only
                            # honest baseline is the one that existed when the decision was taken
                            # (T3-0 4s.30).
                            if short in self._relocating:
                                # Until the actuator's completion arrives, this arm records nothing.
                                self._reloc_pending[short] = completed
                                self._reloc_live_from.pop(short, None)
                                self._reloc_live_at.pop(short, None)
                                self._reloc_effective_at.pop(short, None)
                                self._reloc_pending_logged.pop(short, None)
                            _base_n, _base_mean = self._stats.get((short, cur), (0, 0.0))
                            self._move_from[short] = (cur, (delta_last or {}).get("delta"),
                                                      completed, _base_mean, choice)
                            if str((delta_last or {}).get("source", "")).startswith(
                                    "domain_exposed_recent_run"):
                                self._move_base_walls[short] = list(self._epoch_wall_hist)[
                                    -max(self._recent_run_n, self._min_samples):]
                                self._move_base_tails[short] = list(self._epoch_tail_hist)[
                                    -max(self._recent_run_n, self._min_samples):]
                            else:
                                self._move_base_walls.pop(short, None)
                                self._move_base_tails.pop(short, None)
                    nxt = dict(cfg_now)
                    nxt[short] = choice
                    if choice != cur:
                        self._ll_partners.pop(short, None)
                        if self._ll_level(short, cur) == choice:
                            # The lead is sized by Little's law: size the other under-provisioned stages of the input side
                            # in the same plan (one apply, one verification).
                            partners = self._ll_partner_moves(short, choice, cfg_now, doms)
                            for pshort, pto in partners.items():
                                self._add_level(pshort, pto)
                                nxt[pshort] = pto
                                self._last_move_window[pshort] = completed
                            if partners:
                                self._ll_partners[short] = (completed, {p: cfg_now[p] for p in partners})
                            logger.info("optimizer.cbandit.sizing_move", window=completed, lead=short, to=choice,
                                        partners=partners or None, inputs={k: self._ll.get(k) for k in [short, *partners]})
                    self._planned_cfg[w + 1] = dict(nxt)
                    if choice != cur:
                        plans = self._plans_for(nxt, finding, dom)
                    # choice == cur is a real decision (exploit), and it emits no plan: the
                    # application keeps running what it is running, at zero actuation cost.

        self._cur_window = w
        self._win_metrics = {}
        self._pending_finding = finding
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
            # NEVER RE-PUBLISH A RELOCATION THAT IS ALREADY IN FLIGHT OR DONE (T3-0 4s.32).
            #
            # `current_values` only advances when an ack comes back, so an actuator that applies a
            # relocation and fails to ack leaves the planner believing nothing happened, and it
            # re-publishes -- which on a relocation means paying the copy again. `cf_fz_loop_n32_r1c`
            # paid it five times per node. The actuator is now idempotent, and this is the other
            # half: once a relocation has been planned, no second plan for the same value is issued
            # until its ack resolves it one way or the other.
            if (short in self._relocating
                    and self._reloc_published.get(short) == new_v):
                logger.info("optimizer.cbandit.plan_suppressed", knob_id=kid, value=new_v,
                            reason="this relocation is already published and awaiting its ack")
                continue
            new_v = kd.clamp(new_v)
            if short in self._relocating:
                self._reloc_published[short] = new_v
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


# Worker/thread-count knobs: moved geometrically (x2) in rate mode; see CoordinateBanditPlanner._next_up.
_POOL_KNOBS = ("read_threads", "decode_threads", "num_workers", "computation_threads")
# Geometric step factor per knob in rate mode (a KnobDef.step_factor overrides it). Pool counts double;
# bucket_cap_mb quadruples so a DDP rebuild (seconds at 16N) is paid per 4x, not per 2x.
_GEOMETRIC_FACTOR = {**{k: 2.0 for k in _POOL_KNOBS}, "bucket_cap_mb": 4.0}


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _num_or_none(v):
    """A metric that is absent stays absent. Coercing it to 0 would price a relocation on a corpus
    of no bytes, which is the failure mode the horizon contract already refuses for the horizon."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _rounded(value):
    if value is None or not math.isfinite(float(value)):
        return value
    return round(float(value), 4)
