"""CLBOPlanner: adapt the factored cross-layer surrogate (clbo.StructuredCLBO) to the live
DFOptimizer Planner interface (register_knobs + process_finding -> [ActionPlan]).

Selected only when DFOPTIMIZER_ALGO=clbo; otherwise the default rule/Amdahl Planner runs, so
LiveIO stays byte-identical. Model-based alternative from CLBO-vs-SafeIO.md.

Online mapping (per window):
  - Every DiagnosisFindingMsg for a window carries key_metrics with ABSOLUTE per-domain times
    (fetch_iter_time_proc_max, comm_time_proc_max, compute_time_proc_max, active_time). We MERGE
    key_metrics across a window's findings (different facts carry different domains).
  - Findings for window i accumulate; when a finding with a LATER window_index arrives, window i
    is complete. We finalize it into one observation (config that RAN window i, per-domain
    domains, wall=active_time), feed the accumulated history to StructuredCLBO.suggest, and emit
    an ActionPlan per changed knob.
  - Apply latency: a plan decided at the boundary that closes window i takes effect for the NEXT
    window, so the suggestion runs window (finding.window_index + 1). We track config-per-window
    in _planned_cfg so each observation is attributed to the config that actually produced it.
  - Safety: the surrogate learns from every window, but the deployed INCUMBENT only advances on a
    real improvement (relative wall gain beyond effect_floor). Online we have per-window
    aggregates (not per-step samples), so we use the effect-floor rule rather than the Welch test
    the offline optimize()/tests use. No domain-fraction threshold anywhere.
"""
from __future__ import annotations
import math
import os
import uuid
from typing import Dict, List, Optional

import structlog

from ..types import ActionPlan, DiagnosisFindingMsg, KnobDef
from .clbo import StructuredCLBO

logger = structlog.get_logger()

_DOMAIN_OF = {
    "num_workers": "input", "prefetch_factor": "input", "prefetch_size": "input",
    "read_threads": "input", "reader_threads": "input",
    # halo3d three-way (storage<->decode<->comm): read_threads keeps domain "input" but for that
    # workload "input"==STORAGE (the analyzer emits storage_time as the input metric); decode_threads
    # is its own DECODE domain; halo_width is the COMM knob.
    "decode_threads": "decode",
    # DLIO storage-side actuators. These used to fall through to the "compute" default, which is
    # silently wrong for a domain-driven planner: when the INPUT domain dominates it would never
    # consider the very knobs that relieve it, and would look for relief in compute instead.
    # data_source relocates the corpus (pfs -> node-local); transfer_size is the POSIX read size.
    # computation_threads is deliberately absent: it defaults to "compute" and that is correct.
    "data_source": "input",
    "transfer_size": "input",
    "halo_width": "comm",
    # DELIVERY layer (worker -> trainer sample handoff). `delivery_mode` owns the delivery domain;
    # the layout knobs (`transfer_mb`, `read_threads`) change HOW the same bytes are read, so they
    # belong to the storage ("input") domain. All three are rank-local and math-neutral.
    "delivery_mode": "delivery",
    "transfer_mb": "input",
    "bucket_cap": "comm", "bucket_cap_mb": "comm", "reduce_bucket_size": "comm",
    "accumulate_grad_batches": "comm", "accum": "comm",
}


def _domain_of(knob_short: str) -> str:
    # The DLIO HDF5 pool performs both reads and decompression. Its native
    # decode exposure binds this cell; do not price it from storage alone.
    # Scope this to the explicitly enabled reader, preserving other workloads.
    if (knob_short == "read_threads"
            and os.environ.get("LIVEFLOW_WORKLOAD") == "dlio-pytorch"
            and os.environ.get("LIVEFLOW_DLIO_HDF5_DECODE", "0") == "1"):
        return "decode"
    # `computation_threads` is the TF decode map's num_parallel_calls.  DLIO's compute domain is a
    # sleep, so it stays "compute" (actuator-less) unless the TF decode knob is explicitly
    # registered; then it is priced in the INPUT domain, because the only measurable exposure of
    # decode on DLIO/TF is the consumer wait (stall.storage) that contains it.  Which of
    # read_threads / computation_threads helps is decided by the cbandit's verified effect.
    if knob_short == "computation_threads" and os.environ.get(
            "DFOPTIMIZER_REGISTER_COMPUTATION_THREADS", "0") == "1":
        return "input"
    return _DOMAIN_OF.get(knob_short, "compute")



def domains_from_metrics(m: Dict[str, float]) -> Optional[Dict[str, float]]:
    """Per-domain times for one window. THE single definition, shared by both planners.

    input = fetch - decode, NOT reader_posix. This is the documented decomposition
    (AnalyzerPresetConfigStormerXLayer):

        decode  = reader_decode_time_proc_max          (explicit region)
        input   = fetch_iter_time_proc_max - decode    (the remainder: queueing + raw chunk I/O)
        comm    = comm_time_proc_max

    Both planners had it inverted -- `reader_posix_time_proc_max` as the primary with `fetch` only
    as a fallback -- and the same preset docstring says why that cannot work:

        "syscall interception ... does NOT reliably reach forked DataLoader workers: measured, a
         workers=1 run yields 130 reader_posix-scoped findings while workers=8 runs yield ZERO
         (the fork gap) ... reader_posix is left defined and still used OPPORTUNISTICALLY when it
         IS populated."

    An opportunistic metric was being used as the load-bearing one. Measured on DLIO at 1 node:

        as coded      input=0.6064  compute=2.5443  sum=3.1507  vs wall 6.1725  -> 49% unexplained
        as documented input=3.6282  compute=2.5443  sum=6.1725  vs wall 6.1725  -> closes exactly

    So the planner compared a storage sub-component (0.61) against compute (2.54), concluded
    compute dominates, and never touched the input domain that actually held 3.63 s of the step.

    fetch is the EXPOSED input cost measured parent-side (how long next(loader) blocks), which is
    the quantity a critical-path decision needs. reader_posix explains what the storage part of
    that cost was doing and is kept for choosing WHICH knob to move inside the input domain -- it
    is a within-domain diagnostic, not a domain time.
    """
    if m.get("liveflow_v1_attribution") == 1.0:
        keys = {"input": "lf_v1_storage_wait", "decode": "lf_v1_decode_wait",
                "comm": "lf_v1_comm_wait", "compute": "lf_v1_compute_time"}
        # A producer that declares the delivery family carries its wait; a producer that does not
        # (every older window, halo3d, DLIO) has no delivery domain at all. Never a default for a
        # DECLARED family: a declared window without the measurement is invalid, not zero.
        if m.get("lf_v1_delivery_declared") == 1.0:
            keys["delivery"] = "lf_v1_delivery_wait"
        values = {domain: m.get(key) for domain, key in keys.items()}
        if not all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0
                   for v in values.values()):
            return None
        return {domain: float(value) for domain, value in values.items()}

    # THE COHERENT VECTOR FIRST (2026-09-08, STATUS-ATTRIBUTION-FIX-2026-09-08.md).
    #
    # Everything below this block reads `<layer>_time_proc_max`, which is a maximum over processes
    # taken SEPARATELY per layer. In a lockstep collective the rank that starves on input waits in
    # `stall` and the ranks waiting for it at the barrier wait in `comm`, so those two maxima come
    # from opposite ranks and the four numbers are not a decomposition of anything. Measured on
    # `stgi_capacity_gate1` they sum to 96 to 139 percent of the window while every individual rank
    # closes to 99.5 percent of it.
    #
    # `domain_*_time_proc_crit` is the whole vector of ONE process, the one on the critical path
    # through the data pipeline, so it closes on that process's wall by construction. It is the
    # right vector for pricing a knob: every other rank's comm is inflated by this rank's stall, so
    # relieving its input relieves their comm too. `domain_coherent` says whether the analyzer path
    # that produced this window could build it at all; when it could not, we fall back to the old
    # metrics rather than pricing a vector we know does not close.  Key presence
    # distinguishes an old producer (which has no coherent provenance and may use
    # the legacy path) from a producer that explicitly supplied an unusable
    # coherent vector.  The latter must fail closed: attractive legacy metrics
    # cannot authorize an action after the newer measurement path has said the
    # window is invalid.
    coherent_keys = (
        "domain_input_time_proc_crit", "domain_decode_time_proc_crit",
        "domain_comm_time_proc_crit", "domain_compute_time_proc_crit",
    )
    if "domain_coherent" in m:
        marker = m.get("domain_coherent")
        active = m.get("domain_procs_active")
        wall = m.get("window_wall_time_proc_crit")
        values = [m.get(key) for key in coherent_keys]
        valid = (
            isinstance(marker, (int, float)) and math.isfinite(float(marker)) and float(marker) == 1.0
            and isinstance(active, (int, float)) and math.isfinite(float(active)) and float(active) >= 1.0
            and isinstance(wall, (int, float)) and math.isfinite(float(wall)) and float(wall) > 0.0
            and all(isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) >= 0.0
                    for value in values)
        )
        if not valid:
            m["domain_exposure_present"] = 0.0
            m["domain_exposure_fallback"] = 0.0
            m["domain_exposure_invalid"] = 1.0
            logger.warning(
                "optimizer.domains.coherent_invalid",
                marker=marker, procs_active=active, wall=wall,
                missing=[key for key, value in zip(coherent_keys, values) if value is None],
            )
            return None
        # E1 (judge batch 4, sum-of-max, STORMER-LOOP-DESIGN sec 11/12/14): `domain_input_time_proc_crit`
        # is a window max-of-SUM (one rank, chosen once, summed over the whole window); under a
        # rotating straggler each rank's own window sum is roughly 1/R of the true critical-path cost,
        # so it understates input exposure by up to a factor of R. `domain_input_step_max_sum_time`
        # (attach_input_step_max_sum_metrics, a declared (window, step) view) is a SUM-of-max instead:
        # per step, the max over ranks, summed over the window -- structurally correct for a rotating
        # straggler and needs no comm events. But it is computed PER NODE AGENT (each node_agent sees
        # only its own local ranks' fetch_iter rows), so it still understates the job's true
        # `max over ALL ranks per step` under a rotating straggler that lands on other nodes (measured,
        # tree-v4.3 smoke: 0.129 per-node pooled vs ~0.372 all-16-rank offline). `domain_input_job_step_
        # max_sum_time` (sec 16, `attach_input_job_step_max_sum_metrics`) is the same sum-of-max but
        # reduced CROSS-NODE via a rank-side `Iallreduce(MAX)`, opt-in and off by default -- prefer it
        # whenever present and finite; else prefer the per-node sum-of-max whenever the window covered
        # more than one distinct step; else fall back to the oldest `crit` value. Every rejected tier is
        # still read and logged (`input_crit_reference`, `input_per_node_reference`), so each stays
        # checkable as a regression reference, never silently dropped.
        input_crit_reference = float(m.get("domain_input_time_proc_crit", 0.0) or 0.0)
        step_sum_of_max = m.get("domain_input_step_max_sum_time")
        step_nunique = m.get("domain_input_step_nunique")
        use_step_sum_of_max = (
            step_sum_of_max is not None and step_nunique is not None
            and isinstance(step_sum_of_max, (int, float)) and math.isfinite(float(step_sum_of_max))
            and isinstance(step_nunique, (int, float)) and math.isfinite(float(step_nunique))
            and float(step_nunique) >= 2.0 and float(step_sum_of_max) >= 0.0
        )
        input_per_node_reference = float(step_sum_of_max) if use_step_sum_of_max else input_crit_reference
        job_sum_of_max = m.get("domain_input_job_step_max_sum_time")
        use_job_sum_of_max = (
            job_sum_of_max is not None
            and isinstance(job_sum_of_max, (int, float)) and math.isfinite(float(job_sum_of_max))
            and float(job_sum_of_max) >= 0.0
        )
        if use_job_sum_of_max:
            input_value, input_source = float(job_sum_of_max), "job_sum_of_max"
        elif use_step_sum_of_max:
            input_value, input_source = float(step_sum_of_max), "step_sum_of_max"
        else:
            input_value, input_source = input_crit_reference, "domain_input_time_proc_crit"

        # E2 revision (STORMER-LOOP-DESIGN sec 17 v2, orchestrator decision 2026-09-27, full
        # input+comm loop): `domain_comm_pure_exposure_time_v2` (attach_comm_exposed_median_metrics)
        # is the window median of domain2's per-step exposed-comm event, times step count -- robust to
        # the spike-dominated raw sums the B1 agreement check found. Prefer it whenever the window
        # actually covered more than one distinct step (same suppression rule as the input side); else
        # fall back to `domain_comm_time_proc_crit` unchanged. The rejected reference is always still
        # read and logged for regression comparison.
        comm_crit_reference = float(m.get("domain_comm_time_proc_crit", 0.0) or 0.0)
        comm_exposed_v2 = m.get("domain_comm_pure_exposure_time_v2")
        comm_exposure_nunique = m.get("domain_comm_exposure_step_nunique")
        use_comm_exposed_v2 = (
            comm_exposed_v2 is not None and comm_exposure_nunique is not None
            and isinstance(comm_exposed_v2, (int, float)) and math.isfinite(float(comm_exposed_v2))
            and isinstance(comm_exposure_nunique, (int, float))
            and math.isfinite(float(comm_exposure_nunique))
            and float(comm_exposure_nunique) >= 2.0 and float(comm_exposed_v2) >= 0.0
        )
        # sec 17 v3 (orchestrator decision 2026-09-27, opt-in `DFOPTIMIZER_COMM_EXPOSURE_SOURCE=total`,
        # default "exposed" = unchanged behavior above). Domain2's exposed-comm validation found
        # exposed comm tiny and HIGHER at cap400 (1.8ms cap25/W16 vs 7.6ms cap400/W16) while TOTAL comm
        # was 173 vs 81ms -- Step 0c's cap400 gain looks like comm/compute INTERFERENCE (overlapped
        # RCCL slows backward; fewer, larger buckets interfere less), not an exposed-tail effect.
        # `domain_comm_total_median_time_v3` (attach_comm_total_median_metrics) is an UPPER BOUND on
        # what a bucket_cap change can be worth (the whole comm cost that step, not just its
        # non-overlapped tail) -- a PROPOSAL bound, not a guaranteed-real savings estimate. The bound
        # only says a move is worth PROPOSING/verifying; JOINT_VERIFY's measured wall-time improvement
        # (unmodified, existing mechanism) is the ARBITER that decides whether the move is kept (a
        # jointly-worse configuration reverts LIFO, same as always). This flag changes ONLY which
        # number feeds the argmax/predicted-effect stage; it adds no new acceptance mechanism.
        comm_total_v3 = m.get("domain_comm_total_median_time_v3")
        comm_total_nunique = m.get("domain_comm_total_step_nunique")
        use_comm_total_v3 = (
            os.environ.get("DFOPTIMIZER_COMM_EXPOSURE_SOURCE", "exposed") == "total"
            and comm_total_v3 is not None and comm_total_nunique is not None
            and isinstance(comm_total_v3, (int, float)) and math.isfinite(float(comm_total_v3))
            and isinstance(comm_total_nunique, (int, float)) and math.isfinite(float(comm_total_nunique))
            and float(comm_total_nunique) >= 2.0 and float(comm_total_v3) >= 0.0
        )
        if use_comm_total_v3:
            comm_value, comm_source = float(comm_total_v3), "comm_total_median_v3"
        elif use_comm_exposed_v2:
            comm_value, comm_source = float(comm_exposed_v2), "comm_exposed_median_v2"
        else:
            comm_value, comm_source = comm_crit_reference, "domain_comm_time_proc_crit"

        vec = {
            "input": input_value,
            "decode": float(m.get("domain_decode_time_proc_crit", 0.0) or 0.0),
            "comm": comm_value,
            "compute": float(m.get("domain_compute_time_proc_crit", 0.0) or 0.0),
        }
        logger.info(
            "optimizer.domains.input_source",
            source=input_source,
            input_job_sum_of_max=job_sum_of_max,
            input_step_sum_of_max=step_sum_of_max, input_step_nunique=step_nunique,
            input_crit_reference=input_crit_reference,
            input_per_node_reference=input_per_node_reference, input_used=input_value,
        )
        logger.info(
            "optimizer.domains.comm_source",
            source=comm_source,
            comm_exposed_median_v2=comm_exposed_v2, comm_exposure_step_nunique=comm_exposure_nunique,
            comm_total_median_v3=comm_total_v3, comm_total_step_nunique=comm_total_nunique,
            comm_crit_reference=comm_crit_reference, comm_used=comm_value,
        )
        m["domain_exposure_present"] = 1.0
        m["domain_exposure_fallback"] = 0.0
        m["domain_exposure_invalid"] = 0.0
        logger.info(
            "optimizer.domains.coherent",
            source="domain_*_time_proc_crit",
            residual=float(m.get("residual_time_proc_crit", 0.0) or 0.0),
            wall=float(m.get("window_wall_time_proc_crit", 0.0) or 0.0),
            input_spread=float(m.get("domain_input_spread_time_proc", 0.0) or 0.0),
            procs_active=float(m.get("domain_procs_active", 0.0) or 0.0),
            **vec,
        )
        return vec

    fetch = float(m.get("fetch_iter_time_proc_max", 0.0) or 0.0)
    storage_exposure_key = "stall_storage_time_proc_max"
    decode_exposure_key = "stall_decode_time_proc_max"
    # DLIO exposes the aggregate consumer wait as storage and has no separable decode stage.
    # The storage key is therefore the availability contract. A present value of zero is a real
    # measurement. Absence is the only case which may use the legacy fetch-minus-decode fallback.
    exposure_present = storage_exposure_key in m
    m["domain_exposure_present"] = float(exposure_present)
    m["domain_exposure_fallback"] = float(not exposure_present)
    m["domain_exposure_invalid"] = 0.0

    if exposure_present:
        input_time = float(m.get(storage_exposure_key, 0.0) or 0.0)
        decode = float(m.get(decode_exposure_key, 0.0) or 0.0)
        logger.info(
            "optimizer.domains.exposure",
            storage=input_time,
            decode=decode,
            decode_metric_present=decode_exposure_key in m,
            zero_exposure=(input_time == 0.0 and decode == 0.0),
        )
    else:
        decode = float(m.get("reader_decode_time_proc_max", 0.0) or 0.0)
        input_time = max(fetch - decode, 0.0)
        logger.warning(
            "optimizer.domains.exposure_fallback",
            missing=[storage_exposure_key],
            fetch=fetch,
            reader_decode=decode,
            reason="exposure metrics are absent, not zero",
        )
    comm = float(m.get("comm_time_proc_max", 0.0) or 0.0)
    compute = float(m.get("compute_time_proc_max", 0.0) or 0.0)
    return {
        "input": input_time,
        "decode": decode,
        "comm": comm,
        "compute": max(compute - comm, 0.0),
    }


def _levels_for(kdef: KnobDef, current) -> List:
    """Candidate values for a knob: explicit enum values, else a small grid over the range."""
    if kdef.values:
        return list(kdef.values)
    if kdef.range and kdef.type in (int, float):
        lo, hi = kdef.range
        if kdef.type is int:
            lo, hi = int(lo), int(hi)
            if hi <= lo:
                return [lo]
            n = min(hi - lo + 1, 6)
            step = max((hi - lo) // (n - 1), 1)
            return sorted(set(list(range(lo, hi + 1, step)) + [hi]))
        return sorted({float(lo), (lo + hi) / 2.0, float(hi), float(current)})
    return [current]


class CLBOPlanner:
    """Drop-in alternative to Planner for DFOPTIMIZER_ALGO=clbo."""

    def __init__(self):
        self.knobs: Dict[str, KnobDef] = {}
        self.current_values: Dict[str, object] = {}
        self._short: Dict[str, str] = {}
        self._target_fn: Dict[str, str] = {}
        self._domain: Dict[str, str] = {}
        self._levels: Dict[str, List] = {}
        self._clbo: Optional[StructuredCLBO] = None
        self._default_short: Dict[str, object] = {}
        self._planned_cfg: Dict[int, dict] = {}   # window_index -> config that runs it
        # Exploration budget: after this many decisions, DEPLOY the incumbent instead of probing.
        # None (default) = probe forever = historical behaviour. See process_finding().
        _ew = os.environ.get("DFOPTIMIZER_CLBO_EXPLORE_WINDOWS", "").strip()
        self._explore_windows = int(_ew) if _ew.isdigit() and int(_ew) > 0 else None
        self._decisions = 0
        self._cfgs: List[dict] = []
        self._domain_obs: List[dict] = []
        self._walls: List[float] = []
        self._inc_wall: float = float("inf")
        self._inc_cfg: Optional[dict] = None
        self._cur_window: int = -1
        self._win_metrics: Dict[str, float] = {}
        self.effect_floor = float(os.environ.get("CLBO_EFFECT_FLOOR", "0.02"))

    # --- Planner-compatible registration ---
    def register_knobs(self, knob_defs, current_values=None):
        # Matches Planner.register_knobs: knob_defs is a Dict[str knob_id, KnobDef].
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
        shorts = [self._short[k] for k in self.knobs]
        if shorts:
            self._clbo = StructuredCLBO(
                knobs=shorts, knob_domain=self._domain, knob_levels=self._levels,
                overlap=0.0, effect_floor=self.effect_floor,
            )
        self._default_short = self._short_cfg()

    def _short_cfg(self) -> dict:
        return {self._short[k]: self.current_values[k] for k in self.knobs}

    def _accumulate(self, finding: DiagnosisFindingMsg):
        for k, v in (finding.key_metrics or {}).items():
            self._win_metrics[k] = v

    def _finalize_window(self, window: int) -> Optional[dict]:
        m = self._win_metrics
        wall = m.get("active_time")
        if wall is None or wall <= 0:
            return None
        cfg = self._planned_cfg.get(window, self._default_short)
        domains = domains_from_metrics(m)
        if domains is None:
            return None
        return {"cfg": dict(cfg), "domains": domains, "wall": float(wall)}

    # --- Planner-compatible decision entry ---
    def process_finding(self, finding: DiagnosisFindingMsg) -> List[ActionPlan]:
        if self._clbo is None:
            return []
        w = finding.window_index
        if self._cur_window == -1:
            self._cur_window = w
            self._accumulate(finding)
            return []
        if w == self._cur_window:
            self._accumulate(finding)
            return []
        # boundary: the window we were accumulating is complete
        completed = self._cur_window
        obs = self._finalize_window(completed)
        plans: List[ActionPlan] = []
        if obs is not None:
            self._cfgs.append(obs["cfg"])
            self._domain_obs.append(obs["domains"])
            self._walls.append(obs["wall"])
            if obs["wall"] < self._inc_wall * (1.0 - self.effect_floor):
                self._inc_wall, self._inc_cfg = obs["wall"], obs["cfg"]
            # PROBE vs DEPLOY. Offline BO optimizes only the FINAL answer, so it deploys every
            # suggestion; ONLINE control pays for every probe in real run time. Deploying each
            # suggestion means the run lives in exploratory -- sometimes worst -- configurations:
            # measured on the storage-bound regime the controller found rt=3 (25 ms) then abandoned it
            # for rt=1 (77 ms) and ended there, netting 0% against a static baseline despite 73%
            # available headroom (LIVEFLOW-STATUS.md sec 18).
            # DFOPTIMIZER_CLBO_EXPLORE_WINDOWS=K spends the first K decisions probing, then DEPLOYS THE
            # INCUMBENT for the rest of the run. Default (unset) = probe forever = previous behaviour,
            # so StorMer/DLIO are unaffected.
            # NOTE this is deliberately NOT a safety guard/veto: nothing here is vetoed for being
            # harmful, and no guard metric separate from the objective is consulted. It is a dwell/
            # scheduling policy addressing exploration REGRET (see sec 18.7 on the LiveIO boundary).
            self._decisions += 1
            if self._explore_windows is not None and self._decisions > self._explore_windows \
                    and self._inc_cfg is not None:
                nxt = self._inc_cfg              # exploitation phase: hold the best known config
            else:
                nxt = self._clbo.suggest(self._cfgs, self._domain_obs, self._walls)
                if nxt is None:                  # grid exhausted -> steer back to the incumbent
                    nxt = self._inc_cfg
            if nxt is not None:
                self._planned_cfg[w + 1] = dict(nxt)   # takes effect for the next window
                plans = self._plans_for(nxt, finding)
        self._cur_window = w
        self._win_metrics = {}
        self._accumulate(finding)
        return plans

    def _plans_for(self, next_short_cfg: dict, finding: DiagnosisFindingMsg) -> List[ActionPlan]:
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
                rationale=f"CLBO factored-surrogate suggestion ({self._domain[short]} domain)",
                finding_type=finding.finding_type,
                severity=finding.severity_score,
                opportunity_tag=(finding.opportunity_tags or [""])[0],
                window_index=finding.window_index,
            ))
        return plans

    def apply_global_plan(self, plan) -> Optional[ActionPlan]:
        kid = getattr(plan, "knob_id", None)
        if kid in self.knobs:
            self.current_values[kid] = self.knobs[kid].clamp(plan.new_value)
            return plan
        return None
