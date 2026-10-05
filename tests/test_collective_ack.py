#!/usr/bin/env python3
"""Unit contract for opt-in node-local collective acknowledgement handling."""
import json
import os
import tempfile

from dfoptimizer.optimizer import Optimizer


class Planner:
    def __init__(self):
        self.acks = []

    def apply_ack(self, **kwargs):
        self.acks.append(kwargs)


with tempfile.TemporaryDirectory() as td:
    old_expected = os.environ.get("DFOPTIMIZER_EXPECTED_LOCAL_RANKS")
    old_ledger = os.environ.get("DFOPTIMIZER_ACK_LEDGER")
    os.environ["DFOPTIMIZER_EXPECTED_LOCAL_RANKS"] = "4"
    os.environ["DFOPTIMIZER_ACK_LEDGER"] = f"{td}/acks.jsonl"
    try:
        optimizer = Optimizer()
        optimizer.planner = Planner()
        base = {"host": "node0", "plan_id": "p1", "knob_id": "dlio.read_threads",
                "status": "applied", "old_value": 4, "new_value": 2}
        for rank in range(3):
            optimizer._handle_ack_message({**base, "local_rank": rank, "global_rank": rank})
        assert not optimizer.planner.acks
        optimizer._handle_ack_message({**base, "local_rank": 3, "global_rank": 3})
        assert len(optimizer.planner.acks) == 1
        optimizer._handle_ack_message({**base, "local_rank": 3, "global_rank": 3})
        assert len(optimizer.planner.acks) == 1
        receipts = [json.loads(line) for line in open(os.environ["DFOPTIMIZER_ACK_LEDGER"])]
        assert len(receipts) == 5
        print("PASS collective ACK waits for four unique ranks and forwards once")

        inconsistent = Optimizer()
        inconsistent.planner = Planner()
        inconsistent._handle_ack_message({**base, "plan_id": "p2", "local_rank": 0, "global_rank": 4})
        inconsistent._handle_ack_message({**base, "plan_id": "p2", "new_value": 3,
                                          "local_rank": 1, "global_rank": 5})
        for rank in (2, 3):
            inconsistent._handle_ack_message({**base, "plan_id": "p2", "local_rank": rank,
                                              "global_rank": rank + 4})
        assert not inconsistent.planner.acks
        print("PASS inconsistent values permanently fail a logical collective plan")

        malformed = Optimizer()
        malformed.planner = Planner()
        malformed._handle_ack_message({**base, "plan_id": "p3", "local_rank": 4, "global_rank": 8})
        for rank in range(4):
            malformed._handle_ack_message({**base, "plan_id": "p3", "local_rank": rank,
                                           "global_rank": rank + 8})
        assert not malformed.planner.acks
        print("PASS malformed rank permanently fails a logical collective plan")
    finally:
        if old_expected is None:
            os.environ.pop("DFOPTIMIZER_EXPECTED_LOCAL_RANKS", None)
        else:
            os.environ["DFOPTIMIZER_EXPECTED_LOCAL_RANKS"] = old_expected
        if old_ledger is None:
            os.environ.pop("DFOPTIMIZER_ACK_LEDGER", None)
        else:
            os.environ["DFOPTIMIZER_ACK_LEDGER"] = old_ledger
