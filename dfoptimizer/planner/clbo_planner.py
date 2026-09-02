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
    "bucket_cap": "comm", "bucket_cap_mb": "comm", "reduce_bucket_size": "comm",
    "accumulate_grad_batches": "comm", "accum": "comm",
}


def _domain_of(knob_short: str) -> str:
    return _DOMAIN_OF.get(knob_short, "compute")



def domains_from_metrics(m: Dict[str, float]) -> Dict[str, float]:
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
    fetch = float(m.get("fetch_iter_time_proc_max", 0.0) or 0.0)
    storage_exposure_key = "stall_storage_time_proc_max"
    decode_exposure_key = "stall_decode_time_proc_max"
    # DLIO exposes the aggregate consumer wait as storage and has no separable decode stage.
    # The storage key is therefore the availability contract. A present value of zero is a real
    # measurement. Absence is the only case which may use the legacy fetch-minus-decode fallback.
    exposure_present = storage_exposure_key in m
    m["domain_exposure_present"] = float(exposure_present)
    m["domain_exposure_fallback"] = float(not exposure_present)

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
        return {"cfg": dict(cfg),
                "domains": domains_from_metrics(m),
                "wall": float(wall)}

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
