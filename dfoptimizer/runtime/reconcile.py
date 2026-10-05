"""All-rank reconciliation of per-node planner proposals (LiveFlow v1).

Each node's planner proposes independently, so at a window boundary the ranks carry different
`want` values for the same knob. Raises are reconciled as before: MAX over ranks, with -1
meaning "no proposal". A proposal BELOW the knob's current value is a node planner's REVERT
verdict (a move it judged not confirmed). A revert is a VOTE: it is sticky until the knob's value
changes (nodes reach a verdict in different windows), and it is applied only when a strict
MAJORITY of the voting ranks voted for it. A tie keeps the change; silence (a confirmation, or a
node that has not decided) counts as keep. So one node cannot undo a move that most nodes
confirmed. Orchestrator decision 2026-10-04 (halo3d regime X); the same rule ports to every
workload that reconciles per-node proposals (StorMer `_sync_apply_window`, the SFNO loop).

`halo3d_train.arbitrate_revert` is the same rule (kept there verbatim because the halo3d arm was
validated with it); this module is the shared copy for the PyTorch workloads.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

SENT = -1


def arbitrate_revert(raise_max: int, lower_max: int, votes: int, voters: int,
                     sentinel: int = SENT) -> Tuple[int, Optional[dict]]:
    """Job-wide verdict on node planners' REVERT votes. A raise wins as before; a revert is applied
    only on a strict majority of the voters. Returns (value, event-or-None)."""
    if votes == 0:
        return raise_max, None
    win = raise_max == sentinel and 2 * votes > voters
    return (lower_max if win else raise_max), dict(votes=votes, ranks=voters, applied=bool(win),
                                                   to=lower_max)


class RevertVotes:
    """This rank's sticky revert votes, one per knob."""

    def __init__(self) -> None:
        self._vote: Dict[str, Tuple[int, int]] = {}      # knob -> (value voted for, current at vote)

    def local(self, knob: str, want: int, cur: int) -> Tuple[int, int]:
        """-> (raise proposal, revert vote), each SENT when absent."""
        held = self._vote.get(knob)
        if held is not None and held[1] != cur:
            self._vote.pop(knob, None)                    # the knob changed: the vote is spent
        if SENT < want < cur:
            self._vote[knob] = (int(want), int(cur))
        elif want >= cur:
            self._vote.pop(knob, None)                    # a confirmation or a raise withdraws it
        raise_v = int(want) if want > cur else SENT
        vote = self._vote.get(knob)
        return raise_v, (vote[0] if vote else SENT)


def reconcile(knobs: Sequence[str], wants: Dict[str, int], current: Dict[str, int],
              votes: RevertVotes, voter: bool,
              all_max: Callable[[np.ndarray], np.ndarray],
              all_sum: Callable[[np.ndarray], np.ndarray]) -> Tuple[Dict[str, int], list]:
    """Collective (every rank, every window boundary). `wants[k]` is this rank's proposal (SENT
    for none), `voter` whether this rank carries a node planner's context. Returns the agreed
    value per knob (SENT = no change) and the revert-vote events (identical on every rank)."""
    raises, rev, mine = [], [], []
    for k in knobs:
        r, v = votes.local(k, int(wants.get(k, SENT)), int(current[k])) if voter else (SENT, SENT)
        raises.append(r)
        rev.append(v)
        mine.append(1 if v > SENT else 0)
    mx = all_max(np.array(raises + rev, dtype=np.int64))
    sm = all_sum(np.array(mine + [1 if voter else 0], dtype=np.int64))
    voters = int(sm[-1])
    out, events = {}, []
    n = len(knobs)
    for i, k in enumerate(knobs):
        value, ev = arbitrate_revert(int(mx[i]), int(mx[n + i]), int(sm[i]), voters)
        out[k] = value
        if ev is not None:
            events.append(dict(knob=k, **ev))
    return out, events
