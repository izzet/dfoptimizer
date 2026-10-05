"""Registry of the layout and delivery knobs for the shared StorMer/SFNO PyTorch data path.

Three rank-local, math-neutral actuators (inventory/layout-knob/HANDOFF.md, inventory/delivery-layer/
HANDOFF.md).  None needs cross-rank agreement for CORRECTNESS: every setting yields bitwise-identical
tensors in the same sample order, each worker reads the shared slot once per sample, so a change
takes effect at the next sample boundary on that rank.  Agreement is only needed for MEASUREMENT
coherence: the analyzer credits a window to a configuration only when all ranks of a node report the
same applied vector, so an application that registers these knobs must apply a plan on every rank of
the node (the same node-local reconcile it already uses for num_workers).

  layout.transfer_mb       0 = Original (one h5py read per variable); >0 = coalesced preadv calls of
                           at most that many MB.  Storage ("input") domain.
  layout.read_threads      intra-sample read concurrency.  Storage domain.
  delivery.delivery_mode   0 = torch IPC handoff (Original); 1 = preallocated shared-memory ring
                           with a handle through the queue.  Delivery domain.

Defaults equal Original, so registering them changes nothing until the planner moves one.  The
planner (cbandit, liveflow-v1) picks the knob of the most-exposed actionable family: with the
delivery family declared, a window whose fetch wait is mostly handoff is attributed to `delivery`,
not to storage, even when POSIX reads are idle.
"""
from typing import Dict

from ..types import KnobDef
from .knob import knob, knob_def_from_dict

# Value grids are the measured settings (inventory/layout-knob): 0 plus the 16/64/128 MB transfers
# and 1/2/4 threads, not tuned thresholds.
TRANSFER_MB_VALUES = (0, 16, 64, 128)
READ_THREADS_VALUES = (1, 2, 4)
DELIVERY_MODE_VALUES = (0, 1)


def layout_delivery_knob_defs(workload: str, current: Dict[str, int] = None,
                              delivery: bool = True, layout: bool = True) -> Dict[str, KnobDef]:
    """KnobDefs for the layout/delivery actuators of `workload` ('sfno' | 'stormer').

    `current` supplies the shipped (Original) values: transfer_mb, read_threads, delivery_mode.
    """
    cur = dict(transfer_mb=0, read_threads=1, delivery_mode=0)
    cur.update(current or {})
    target = f"{workload}.make_loader"
    defs = {}
    if layout:
        defs["layout.transfer_mb"] = knob(
            default=int(cur["transfer_mb"]), type=int, values=list(TRANSFER_MB_VALUES),
            responds_to={"storage_pressure": {"direction": "increase", "step_mode": "evidence",
                                              "min_persistence": 2, "cooldown_windows": 3,
                                              "apply_when": "window_boundary"}})
        defs["layout.read_threads"] = knob(
            default=int(cur["read_threads"]), type=int, values=list(READ_THREADS_VALUES),
            responds_to={"storage_pressure": {"direction": "increase", "step_mode": "evidence",
                                              "min_persistence": 2, "cooldown_windows": 3,
                                              "apply_when": "window_boundary"}})
    if delivery:
        defs["delivery.delivery_mode"] = knob(
            default=int(cur["delivery_mode"]), type=int, values=list(DELIVERY_MODE_VALUES),
            responds_to={"delivery_pressure": {"direction": "increase", "step_mode": "evidence",
                                               "min_persistence": 2, "cooldown_windows": 3,
                                               "apply_when": "window_boundary"}})
    return {k: knob_def_from_dict(k, v, target_function=target) for k, v in defs.items()}
