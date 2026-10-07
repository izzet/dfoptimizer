"""Little's-law sizing of the parallelism knobs (LIVEFLOW_SIZING=littles_law|geometric, default littles_law).

For a pipeline stage s with a parallelism knob (read_threads / num_workers, decode_threads, computation_threads) the
number of workers the stage needs to keep the trainer fed is, by Little's law,

    needed_s = ceil( lambda * S_s )

    lambda = the sample rate the trainer would consume if input were free = samples per step / (step time - exposed input wait)
    S_s    = the measured per-sample service time of stage s (busy seconds of the stage's operations per sample)

Both are measured in the SAME window: the step time and the exposed input wait are the window's per-step means after the
job-level attribution (so the wait a peer's input stall imposed on this rank counts as input wait), and the service comes from
the analyzer's per-stage service views (critical rank of the node, liveflow_v1.SERVICE_VALUE_KEYS).  Samples per step cancel:
lambda * S = (samples/step / T_free) * (service_s_per_step / samples/step) = service_s_per_step / T_free.  No safety factor, no
constant: ceil only; the knob's registry maximum caps the result.  The bootstrap verification and the side-effect ledger decide
keep or revert, exactly as for any other move.

Scope and limits: the law gives the MEAN number of workers that must be busy; it is blind to bursts (heavy-tailed service,
epoch-start stalls, peer skew), so a stage whose mean service is small but whose tail is exposed is sized at or below its
current value, and then the planner's geometric/refinement step applies unchanged.  Missing or non-finite service or a
non-positive free step time suppress the proposal (never evidence of a problem).
"""
import math
import os
from typing import Dict, Optional, Tuple

import structlog

logger = structlog.get_logger()

_MODES = ("littles_law", "geometric")

# knob -> the stage service layers whose busy time its workers carry.  A decode/map pool carries the decode stage; a read pool
# the storage stage; a DataLoader worker pool (num_workers) carries every declared worker-side stage (the delivery handoff runs in
# the trainer's main thread and is not part of a worker's service).
STAGES_OF_KNOB: Dict[str, Tuple[str, ...]] = {
    "read_threads": ("storage",),
    "decode_threads": ("decode",),
    "computation_threads": ("decode",),
    "num_workers": ("storage", "decode"),
}
# stage -> the key of the planner's per-domain exposure vector holding that stage's exposed wait
STAGE_DOMAIN = {"storage": "input", "decode": "decode", "delivery": "delivery"}


def sizing_mode() -> str:
    mode = os.environ.get("LIVEFLOW_SIZING", "littles_law").strip().lower()
    if mode not in _MODES:
        logger.warning("optimizer.sizing.bad_mode", value=mode, using="littles_law")
        return "littles_law"
    return mode


def _fin(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) and f >= 0.0 else None


def free_step_time(wall: float, domains: Dict[str, float]) -> Optional[float]:
    """Step time the trainer would take if input were free: wall minus the attributed exposed input wait of every input stage."""
    exposed = sum(float(domains.get(d, 0.0) or 0.0) for d in ("input", "decode", "delivery"))
    t = float(wall) - exposed
    return t if math.isfinite(t) and t > 0.0 else None


def stage_service(metrics: dict, short: str) -> Optional[Tuple[float, str]]:
    """(S, stages) per training step for the knob's stages, or None when a required stage was not measured."""
    stages = STAGES_OF_KNOB.get(short)
    if not stages:
        return None
    total, used = 0.0, []
    for i, st in enumerate(stages):
        v = _fin(metrics.get("lf_v1_svc_" + st))
        if v is None:
            if i == 0:
                return None
            continue
        total += v
        used.append(st)
    return (total, "+".join(used)) if total > 0.0 else None


def needed_parallelism(service_s: float, free_step_s: float) -> int:
    """ceil(lambda * S) with lambda = 1 / free_step_s per step-item and S = service_s per step-item (>= 1 worker)."""
    lam = 1.0 / free_step_s
    x = lam * service_s
    return max(int(math.ceil(x - 1e-9)), 1)


def size_knobs(metrics: dict, wall: float, domains: Dict[str, float], knobs: Dict[str, Tuple[float, float, float]]) -> Dict[str, dict]:
    """The Little's-law level of every sizable knob this window can size.

    `knobs` maps the knob's short name to (current value, registry minimum, registry maximum).  Each entry of the result
    carries the inputs (lambda, S, free step time), the uncapped need and the capped target, so a decision can be read back.
    """
    out: Dict[str, dict] = {}
    t_free = free_step_time(wall, domains)
    if t_free is None:
        return out
    for short, (cur, lo, hi) in knobs.items():
        got = stage_service(metrics, short)
        if got is None:
            continue
        service, stages = got
        need = needed_parallelism(service, t_free)
        target = int(min(max(need, int(lo) if lo is not None else 1, 1), int(hi)))
        out[short] = {"needed": need, "target": target, "current": cur, "stages": stages,
                      "lambda_per_s": round(1.0 / t_free, 3), "S_s": round(service, 6), "free_step_s": round(t_free, 6),
                      "critical_rank": bool(all(metrics.get("lf_v1_svc_%s_critical" % s) == 1.0 for s in stages.split("+"))),
                      "capped": need > int(hi)}
    return out
