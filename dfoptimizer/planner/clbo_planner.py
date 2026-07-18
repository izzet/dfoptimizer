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

from ..types import ActionPlan, DiagnosisFindingMsg, KnobDef
from .clbo import StructuredCLBO

_DOMAIN_OF = {
    "num_workers": "input", "prefetch_factor": "input", "prefetch_size": "input",
    "read_threads": "input", "reader_threads": "input",
    "bucket_cap": "comm", "bucket_cap_mb": "comm", "reduce_bucket_size": "comm",
}


def _domain_of(knob_short: str) -> str:
    return _DOMAIN_OF.get(knob_short, "compute")


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
        fetch = m.get("fetch_iter_time_proc_max", 0.0)
        comm = m.get("comm_time_proc_max", 0.0)
        pure_compute = max(m.get("compute_time_proc_max", 0.0) - comm, 0.0)
        cfg = self._planned_cfg.get(window, self._default_short)
        return {"cfg": dict(cfg),
                "domains": {"input": fetch, "comm": comm, "compute": pure_compute},
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
            nxt = self._clbo.suggest(self._cfgs, self._domain_obs, self._walls)
            if nxt is None:                      # grid exhausted -> steer back to the incumbent
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
