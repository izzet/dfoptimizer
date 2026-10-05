"""Majority-revert reconciliation shared by the PyTorch workloads (halo3d keeps its own copy)."""
import numpy as np
from dfoptimizer.runtime.reconcile import SENT, RevertVotes, arbitrate_revert, reconcile


def _world(per_rank_wants, current, books, voters=None):
    """Run reconcile on every simulated rank with real MAX/SUM over the stacked contributions."""
    n = len(per_rank_wants)
    voters = voters if voters is not None else [True] * n
    contrib = []
    for r in range(n):
        raises, rev, mine = [], [], []
        for k in sorted(current):
            a, b = books[r].local(k, per_rank_wants[r].get(k, SENT), current[k]) if voters[r] else (SENT, SENT)
            raises.append(a); rev.append(b); mine.append(1 if b > SENT else 0)
        contrib.append((np.array(raises + rev), np.array(mine + [1 if voters[r] else 0])))
    mx = np.max([c[0] for c in contrib], axis=0)
    sm = np.sum([c[1] for c in contrib], axis=0)
    outs = []
    for r in range(n):
        books_copy = books[r]
        out, ev = reconcile(sorted(current), per_rank_wants[r], current, _Frozen(books_copy, voters[r]),
                            voters[r], lambda a: mx, lambda a: sm)
        outs.append((out, ev))
    return outs


class _Frozen:
    """reconcile() calls votes.local() again; replay without re-mutating (already done above)."""
    def __init__(self, book, voter):
        self.book, self.voter = book, voter
    def local(self, knob, want, cur):
        held = self.book._vote.get(knob)
        raise_v = int(want) if want > cur else SENT
        return raise_v, (held[0] if held else SENT)


def test_arbitrate_matches_halo_rule():
    assert arbitrate_revert(SENT, 8, 0, 8) == (SENT, None)
    assert arbitrate_revert(SENT, 8, 4, 8)[0] == SENT          # tie keeps
    assert arbitrate_revert(SENT, 8, 5, 8)[0] == 8             # strict majority reverts
    assert arbitrate_revert(24, 8, 7, 8)[0] == 24              # a raise wins


def test_one_dissenting_node_cannot_undo_a_move():
    books = [RevertVotes() for _ in range(8)]
    wants = [{"num_workers": 8}] + [{} for _ in range(7)]       # node 0 votes to revert 16 -> 8
    out = _world(wants, {"num_workers": 16}, books)
    assert all(o[0]["num_workers"] == SENT for o in out)


def test_majority_reverts_and_votes_are_sticky():
    books = [RevertVotes() for _ in range(8)]
    cur = {"num_workers": 16}
    wants = [{"num_workers": 8}] * 3 + [{}] * 5
    assert _world(wants, cur, books)[0][0]["num_workers"] == SENT            # 3 of 8
    wants = [{}] * 3 + [{"num_workers": 8}] * 3 + [{}] * 2                    # three more vote later
    out = _world(wants, cur, books)
    assert out[0][0]["num_workers"] == 8                                      # 6 of 8 sticky votes
    assert out[0][1][0]["applied"] is True


def test_vote_spent_when_value_changes():
    b = RevertVotes()
    assert b.local("w", 8, 16) == (SENT, 8)
    assert b.local("w", SENT, 24) == (SENT, SENT)


def test_only_context_ranks_count_as_voters():
    books = [RevertVotes() for _ in range(8)]
    voters = [True, False, False, False, True, False, False, False]           # 2 nodes x 4 ranks
    wants = [{"num_workers": 8}, {}, {}, {}, {"num_workers": 8}, {}, {}, {}]
    out = _world(wants, {"num_workers": 16}, books, voters=voters)
    assert out[0][0]["num_workers"] == 8                                      # 2 of 2 voters
