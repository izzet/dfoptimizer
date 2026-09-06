import json
import os
import queue
import socket
import threading
from collections import defaultdict
from typing import Dict, List, Optional

import structlog

from ..types import ActionAck, ActionPlan, KnobDef

logger = structlog.get_logger()

# Global context singleton
_global_context: Optional["OptimizerContext"] = None
_context_lock = threading.Lock()


class OptimizerContext:
    """Bridge between the optimizer service and the instrumented app.

    Spawns a background thread that subscribes to Mofka ``optimizer.plans``
    topic.  Incoming ActionPlan messages are parsed and queued per target
    function.  The @tunable decorator drains the queue on each call.

    On first call to a @tunable function, publishes the knob registry
    (including responds_to rules) to the ``optimizer.registry`` topic so
    the optimizer service can build its planner dynamically.
    """

    def __init__(
        self,
        namespace: str = "app",
        protocol: str = "",
        group_file: str = "",
        topic_plans: str = "optimizer.plans",
        topic_acks: str = "optimizer.acks",
        topic_registry: str = "optimizer.registry",
    ):
        self.namespace = namespace
        self.protocol = protocol
        self.group_file = group_file
        # THE OUTBOUND PATHS MUST NOT GATE ON A MOFKA CONCEPT.
        #
        # `_publish_ack` and `_publish_registration` both used to start with
        # `if self._noop or not self.group_file: return`. Under any transport that addresses
        # topics by name rather than through a Mofka group file, that silently discarded every
        # ack and every knob registration BEFORE they were even enqueued -- so the app applied
        # knobs and logged optimizer.action.ack while the optimizer received nothing, and the
        # cooldown and effectiveness gates (which only record on an ack, sec 83.5) could never
        # fire. Three separate places had the same assumption; this is the single predicate they
        # all now share.
        self._transport = os.environ.get("LIVEFLOW_TRANSPORT", "mofka").strip().lower()
        self._can_publish = bool(group_file) or self._transport != "mofka"
        # Reconcile-to-desired-state rather than drain-a-queue. On by default for broadcast
        # transports, where every rank reads the same slot and last-value-wins is the point;
        # off under Mofka so existing behaviour is untouched. See desired_actions_for().
        self._reconcile = os.environ.get(
            "LIVEFLOW_RECONCILE", "1" if self._transport != "mofka" else "0") == "1"
        self.topic_plans = topic_plans
        self.topic_acks = topic_acks
        self.topic_registry = topic_registry

        # Per-function action queue: func_name -> {knob_id -> ActionPlan}
        # Only the *latest* action per knob is kept (latest-wins)
        self._queues: Dict[str, Dict[str, ActionPlan]] = defaultdict(dict)
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._consumer = None
        self._listener_driver = None
        self._producer = None  # for acks
        self._registry_producer = None  # for knob registration
        self._global_ack_producer = None
        self._global_registry_producer = None
        self._noop = False
        # ---- NON-BLOCKING ACK PATH (LIVEFLOW-DECODE-RCA.md sec 11) --------------------------
        # An ack is a CONFIRMATION, not a barrier: losing one costs the controller a confirmation,
        # while blocking on one costs the entire job. Yet `_publish_ack` used to push + `flush()`
        # synchronously ON THE TRAINING THREAD, at a window boundary, only when a knob changed --
        # an unbounded blocking call sitting on the workload's critical path.
        #
        # Principle adopted for this codebase: no control-plane call on the training thread may
        # block without a bound; where a bound cannot be guaranteed the call moves off-thread and
        # becomes droppable. So acks are enqueued to a BOUNDED ring and published by a daemon
        # thread; on overflow they are DROPPED and COUNTED. Degrade by dropping, never by waiting.
        #
        # The drop counter is not cosmetic: the paper claims near-zero overhead for the
        # observability plane, and a silent drop would be exactly the kind of unmeasured loss that
        # claim cannot afford. It is reported at teardown.
        self._ack_q: "queue.Queue" = queue.Queue(
            maxsize=int(os.environ.get("DFOPTIMIZER_ACK_QUEUE", "1024") or 1024))
        self._ack_dropped = 0
        self._ack_published = 0
        self._ack_pub_thread: Optional[threading.Thread] = None
        # Pause flag: when set, the background listener stays ALIVE but stops pulling from Mofka. Lets
        # a caller confine the consumer's Margo progress to safe moments (e.g. window boundaries, off
        # the per-step NCCL-P2P critical path) WITHOUT tearing down/recreating the consumer thread
        # (which desyncs collective apply at scale). Opt-in; default = not paused (unchanged behavior).
        self._paused = threading.Event()

    def pause(self):
        """Stop the listener from pulling (thread stays alive)."""
        self._paused.set()

    def resume(self):
        """Resume pulling."""
        self._paused.clear()

    def start(self):
        """Start the background Mofka listener thread.

        LIVEFLOW_CTX_LISTEN=0 constructs the context but never starts the listener, so the process
        holds NO Mofka consumer and never pulls. Diagnostic only (LIVEFLOW-STATUS.md sec 57): every
        arm carrying an in-rank context wedges at a control-window boundary and every arm without
        one completes, but "context present" bundles two different things -- a second margo/thallium
        ENGINE existing in the address space, and a consumer actively PULLING on it. Nothing so far
        separates them, and they imply different fixes: an engine that is merely resident argues for
        sharing one runtime, whereas an active consumer argues the control plane must leave the
        process entirely. The driver is still created lazily by the ack path, so this isolates the
        pull loop rather than the engine.
        """
        if self._noop:
            return
        if os.environ.get("LIVEFLOW_CTX_LISTEN", "1") == "0":
            logger.info("optimizer.context.listen_disabled", namespace=self.namespace)
            self._running = False
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._listen_loop, daemon=True, name="dfoptimizer-ctx"
        )
        self._thread.start()
        logger.info("optimizer.context.started", namespace=self.namespace)

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        self._release_producer("_producer")
        self._release_producer("_registry_producer")
        self._release_producer("_global_ack_producer")
        self._release_producer("_global_registry_producer")
        self._consumer = None
        self._listener_driver = None
        logger.info("optimizer.context.stopped")

    def drain_actions_for(
        self, func_name: str, at: str = "epoch_boundary"
    ) -> List[ActionPlan]:
        """Return pending actions whose ``apply_when`` matches *at*.

        Actions that don't match remain in the queue for a later drain
        with the appropriate *at* value.  For example, ``epoch_boundary``
        actions stay queued when called with ``at="window_boundary"`` and are
        only drained when the caller passes ``at="epoch_boundary"``.
        """
        with self._lock:
            pending = self._queues[func_name]
            matched = []
            remaining = {}
            for knob_id, plan in pending.items():
                if plan.apply_when == at or at == "epoch_boundary":
                    # epoch_boundary drains everything (superset of step)
                    matched.append(plan)
                else:
                    remaining[knob_id] = plan
            self._queues[func_name] = remaining
        return matched

    def desired_actions_for(self, func_name: str, at: str = "epoch_boundary") -> List[ActionPlan]:
        """Return the current DESIRED plans without consuming them.

        RECONCILIATION, NOT DELIVERY. `drain_actions_for` removes a plan as it hands it over, so
        each rank consumes independently at whatever moment it reaches a window boundary. With
        more than one rank that is a divergence generator: if rank A drains at window 10 and rank
        B at window 11 with a new plan published in between, the two ranks apply DIFFERENT values
        and stay that way. That is the sec 78 rank-asymmetry failure re-created by timing instead
        of by flags, and no amount of care in the actuator prevents it.

        Holding the plan and letting every rank reconcile to it makes the apply idempotent and
        convergent: whenever a rank next looks, it sees the same current value as every other
        rank, regardless of when it looked. It also matches the hop-4 mailbox exactly, whose
        semantics are last-value-wins because a knob is STATE, not an event stream.

        LIMIT, stated because it decides where this is safe: reconciliation gives CONVERGENCE,
        not SIMULTANEITY. Ranks agree eventually, not at the same instant. That is fine for a
        knob whose apply is local (read_threads rebuilds one process's DataLoader). It is NOT
        sufficient for a knob whose apply contains a collective -- `bucket_cap_mb` ->
        `_rebuild_ddp` (sec 68) -- where a rank arriving a window late deadlocks the ones that
        arrived on time. Those need a deterministic apply point carried in the plan, not just
        convergence. Not implemented here; such knobs must keep using drain semantics until it is.
        """
        with self._lock:
            pending = self._queues[func_name]
            return [
                plan for plan in pending.values()
                if plan.apply_when == at or at == "epoch_boundary"
            ]

    def actions_for(self, func_name: str, at: str = "epoch_boundary") -> List[ActionPlan]:
        """Reconcile when the transport is a broadcast one, drain otherwise.

        Defaults to the historical drain behaviour under Mofka so existing runs are unchanged.
        """
        if self._reconcile:
            return self.desired_actions_for(func_name, at)
        return self.drain_actions_for(func_name, at)

    def ack_action(self, plan: ActionPlan, old_value, new_value, status="applied",
                   reason="", completed_at=None):
        ack = ActionAck(
            plan_id=plan.plan_id,
            knob_id=plan.knob_id,
            status=status,
            old_value=old_value,
            new_value=new_value,
            reason=reason or "",
            completed_at=completed_at,
        )
        logger.info(
            "optimizer.action.ack",
            plan_id=ack.plan_id,
            knob_id=ack.knob_id,
            old_value=old_value,
            new_value=new_value,
            status=status,
            reason=reason or None,
            completed_at=completed_at,
        )
        self._publish_ack(ack)

    def _ack_publisher_loop(self):
        """Drain the ack ring and publish. Runs OFF the training thread, so a slow or stuck
        Mofka push delays acks only -- it can no longer stall the workload."""
        while True:
            item = self._ack_q.get()
            if item is None:                      # shutdown sentinel
                return
            payload, metadata = item
            try:
                if self._producer is None:
                    from ..streaming.transport import open_producer
                    _, self._producer = open_producer(self.group_file, self.topic_acks)
                self._producer.push(metadata=metadata, data=payload)
                self._producer.flush()
                self._ack_published += 1
                self._publish_global_mirror(
                    producer_attr="_global_ack_producer",
                    topic=os.environ.get("DFOPTIMIZER_GLOBAL_ACKS_TOPIC", "optimizer_acks"),
                    payload=payload,
                    metadata=metadata,
                )
            except Exception:
                logger.warning("optimizer.ack.publish_failed", exc_info=True)

    def _ack_stats(self) -> dict:
        return {"published": self._ack_published, "dropped": self._ack_dropped,
                "queued": self._ack_q.qsize()}

    def enqueue_action(self, plan: ActionPlan):
        """Manually enqueue an action (used by tests or direct injection)."""
        with self._lock:
            self._queues[plan.target_function][plan.knob_id] = plan

    def register_knobs(
        self,
        func_name: str,
        knob_defs: Dict[str, KnobDef],
        current_values: Optional[Dict[str, object]] = None,
    ):
        """Publish knob definitions to the optimizer registry topic."""
        from .knob import knob_def_to_wire

        knobs_wire = {}
        for param_name, kdef in knob_defs.items():
            knobs_wire[param_name] = knob_def_to_wire(kdef)

        registration = {
            "namespace": self.namespace,
            "function_name": func_name,
            "knobs": knobs_wire,
            "current_values": current_values or {},
        }

        logger.info(
            "optimizer.knobs.registering",
            namespace=self.namespace,
            function=func_name,
            knobs=list(knobs_wire.keys()),
            current_values=registration["current_values"],
        )

        self._publish_registration(registration)

    # -- Mofka listener --

    def _listen_loop(self):
        try:
            from ..streaming.transport import ack_event, open_consumer
            driver, consumer = open_consumer(
                self.group_file,
                self.topic_plans,
                consumer_name=f"dfoptimizer_ctx_{os.getpid()}",
            )
        except Exception:
            logger.warning(
                "optimizer.context.mofka_connect_failed",
                exc_info=True,
            )
            self._noop = True
            return

        self._listener_driver = driver
        self._consumer = consumer

        future = consumer.pull()
        while self._running:
            if self._paused.is_set():
                # Paused: do NOT drive Margo progress (keeps the consumer off the NCCL-P2P path).
                import time as _t
                _t.sleep(0.02)
                continue
            try:
                # POLL TIMEOUT IS A BUSY-SPIN DIAL, NOT JUST A LATENCY DIAL
                # (LIVEFLOW-STATUS.md sec 57/58).
                #
                # mofka v0.8.2 `Promise::wait(timeout_ms)` computes its deadline as
                #     deadline.tv_nsec = now.tv_nsec + timeout_ms*1e6
                # and never normalises it. Whenever that sum exceeds 1e9 the timespec is INVALID:
                # ABT_cond_timedwait returns immediately while the loop guard
                # (now.tv_nsec < deadline.tv_nsec) can never go false, because a real tv_nsec tops
                # out at 999,999,999. The "timed wait" degenerates into a tight spin holding a core
                # until the second rolls over. Upstream fixed exactly this in b677527 (v0.9.0+).
                #
                # The overflow probability is timeout_ms/1000, so this value IS the spin rate:
                #   500 ms -> ~50% of polls spin      10 ms -> ~1%
                # Until the Mofka upgrade lands, keeping this small is what stops a consumer thread
                # from starving RCCL's proxy and the Python decode pool in the same process.
                _poll_ms = int(os.environ.get("DFOPTIMIZER_POLL_TIMEOUT_MS", "10") or 10)
                event = future.wait(timeout_ms=_poll_ms)
            except Exception as ex:
                if "timeout" in str(ex).lower():
                    continue
                logger.error("optimizer.context.listen_error", error=str(ex))
                break

            if event is None:
                continue

            try:
                self._handle_plan_event(event)
            except Exception:
                logger.exception("optimizer.context.plan_event_error")

            ack_event(event)
            future = consumer.pull()

        del consumer
        del driver
        self._consumer = None
        self._listener_driver = None

    def _handle_plan_event(self, event):
        payload = event.data
        if payload is None:
            return
        if isinstance(payload, list):
            payload = b"".join(payload)

        msg = json.loads(payload.decode("utf-8"))

        # A window's plans arrive as ONE batch message, because the broadcast mailbox is a single
        # versioned slot and per-plan publishes overwrite each other (see optimizer._publish_plans).
        # A bare plan dict is still accepted: other transports are real queues and older publishers
        # send that shape, and silently ignoring them would be the same class of bug again.
        if isinstance(msg, dict) and isinstance(msg.get("plans"), list):
            for sub in msg["plans"]:
                self._handle_plan_message(sub)
            return
        self._handle_plan_message(msg)

    def _handle_plan_message(self, msg):
        plan = ActionPlan(
            plan_id=msg["plan_id"],
            knob_id=msg["knob_id"],
            target_function=msg["target_function"],
            old_value=msg.get("old_value"),
            new_value=msg["new_value"],
            apply_when=msg.get("apply_when", "epoch_boundary"),
            rationale=msg.get("rationale", ""),
            finding_type=msg.get("finding_type", ""),
            severity=msg.get("severity", 0),
            opportunity_tag=msg.get("opportunity_tag", ""),
            window_index=msg.get("window_index", 0),
            target_nodes=msg.get("target_nodes", []),
            requires_unanimity=bool(msg.get("requires_unanimity", False)),
        )

        # Per-node plan filter: skip plans not targeting this node.
        if plan.target_nodes and socket.gethostname() not in plan.target_nodes:
            logger.debug(
                "optimizer.context.plan_filtered",
                plan_id=plan.plan_id,
                target_nodes=plan.target_nodes,
                hostname=socket.gethostname(),
            )
            return

        logger.info(
            "optimizer.context.plan_received",
            plan_id=plan.plan_id,
            knob_id=plan.knob_id,
            new_value=plan.new_value,
            target_nodes=plan.target_nodes or "all",
            # WHY the plan exists, logged where it ARRIVES.
            #
            # The rationale names the finding that produced the plan ("decode_pressure: ...
            # -> decode_parallelism"), which is the only record of WHICH DOMAIN the controller
            # attributed the problem to. It was previously visible only if the application's own
            # apply path happened to print it: DLIO's did, StorMer's did not, so the phase 29
            # bottleneck-shift result could show that two domains were actuated but had to INFER
            # from the knob-to-tag mapping which rule drove each.
            #
            # The planner logs it too (planner.py:152) but that is the agent process, and no agent
            # log in this campaign has ever captured those records. Logging it on arrival makes
            # domain attribution observable from the application side, for any application,
            # independent of how it applies the knob.
            rationale=plan.rationale,
        )

        with self._lock:
            # Latest-wins: overwrite any older action for same knob+function
            self._queues[plan.target_function][plan.knob_id] = plan

    def _publish_ack(self, ack: ActionAck):
        """NON-BLOCKING. Serialise, enqueue, return. The training thread never touches Mofka.

        Was: push + flush() inline, i.e. an unbounded blocking control-plane call on the
        workload's critical path, firing only when a knob changed. See sec 11 of
        LIVEFLOW-DECODE-RCA.md for why that is the wrong shape regardless of whether it was the
        cause of any particular hang.
        """
        if self._noop or not self._can_publish:
            return
        try:
            import dataclasses
            payload = json.dumps(dataclasses.asdict(ack)).encode("utf-8")
            metadata = {"type": "action_ack", "plan_id": ack.plan_id}
            if self._ack_pub_thread is None:
                self._ack_pub_thread = threading.Thread(
                    target=self._ack_publisher_loop, daemon=True, name="dfoptimizer-ackpub")
                self._ack_pub_thread.start()
            try:
                self._ack_q.put_nowait((payload, metadata))
            except queue.Full:
                # Degrade by DROPPING, never by waiting. Counted so the loss is measurable --
                # an unmeasured drop would undermine the near-zero-overhead claim.
                self._ack_dropped += 1
                if self._ack_dropped in (1, 10, 100) or self._ack_dropped % 1000 == 0:
                    logger.warning("optimizer.ack.dropped", total=self._ack_dropped)
        except Exception:
            logger.warning("optimizer.ack.enqueue_failed", plan_id=ack.plan_id, exc_info=True)

    def _publish_registration(self, registration: dict):
        if self._noop or not self._can_publish:
            return
        try:
            if self._registry_producer is None:
                from ..streaming.transport import open_producer
                _, self._registry_producer = open_producer(
                    self.group_file, self.topic_registry
                )

            payload = json.dumps(registration).encode("utf-8")
            metadata = {
                "type": "knob_registration",
                "namespace": registration["namespace"],
                "function": registration["function_name"],
            }
            self._registry_producer.push(metadata=metadata, data=payload)
            self._registry_producer.flush()
            self._publish_global_mirror(
                producer_attr="_global_registry_producer",
                topic=os.environ.get("DFOPTIMIZER_GLOBAL_REGISTRY_TOPIC", self.topic_registry),
                payload=payload,
                metadata=metadata,
            )
            logger.info(
                "optimizer.knobs.published",
                namespace=registration["namespace"],
                function=registration["function_name"],
                knob_count=len(registration["knobs"]),
            )
        except Exception:
            logger.warning("optimizer.knobs.publish_failed", exc_info=True)

    def _publish_global_mirror(
        self,
        producer_attr: str,
        topic: str,
        payload: bytes,
        metadata: dict,
    ):
        global_group_file = os.environ.get("INFRA_CXI_GROUP", "").strip()
        if not global_group_file or global_group_file == self.group_file:
            return
        try:
            producer = getattr(self, producer_attr, None)
            if producer is None:
                from ..streaming.transport import open_producer
                _, producer = open_producer(global_group_file, topic)
                setattr(self, producer_attr, producer)
            producer.push(metadata=metadata, data=payload)
            producer.flush()
            logger.info(
                "optimizer.global_mirror.published",
                topic=topic,
                global_group_file=global_group_file,
                metadata_type=metadata.get("type", ""),
            )
        except Exception:
            logger.warning(
                "optimizer.global_mirror.publish_failed",
                topic=topic,
                global_group_file=global_group_file,
                exc_info=True,
            )

    def _release_producer(self, attr_name: str):
        producer = getattr(self, attr_name, None)
        if producer is None:
            return
        try:
            producer.flush()
        except Exception:
            logger.warning("optimizer.context.producer_flush_failed", producer_attr=attr_name, exc_info=True)
        setattr(self, attr_name, None)


class _NoopContext(OptimizerContext):
    """Lightweight stand-in when Mofka is not available."""

    def __init__(self, namespace="app"):
        super().__init__(namespace=namespace)
        self._noop = True

    def start(self):
        logger.info("optimizer.context.noop")

    def stop(self):
        pass


def optimizer_context(
    namespace: str = "app",
    protocol: str = "",
    group_file: str = "",
    topic_plans: str = "optimizer.plans",
    topic_acks: str = "optimizer.acks",
    topic_registry: str = "optimizer.registry",
) -> OptimizerContext:
    """Create or return the global OptimizerContext.

    Returns a no-op context when no transport can serve it, so the app runs
    unchanged.

    THE GROUP FILE IS A MOFKA CONCEPT, NOT A PRECONDITION FOR ACTUATION.
    This used to gate solely on `group_file`, which silently disabled the whole
    actuation path under any other transport: the context logged
    `no_group_file` and became a no-op BEFORE the streaming/transport selector
    was ever consulted, so plans published to a shared-memory mailbox were
    never read and knobs never moved. That is what made the first Option C
    end-to-end run stop at "plan published" (see liveflow_shm/README.md).

    Under LIVEFLOW_TRANSPORT the backend addresses topics by name and needs no
    group file at all, so the gate has to be per transport.
    """
    global _global_context
    with _context_lock:
        if _global_context is not None:
            return _global_context

        _transport = os.environ.get("LIVEFLOW_TRANSPORT", "mofka").strip().lower()
        _needs_group_file = _transport == "mofka"

        if _needs_group_file and (not group_file or not os.path.exists(group_file)):
            logger.info("optimizer.context.no_group_file", transport=_transport)
            ctx = _NoopContext(namespace=namespace)
        else:
            ctx = OptimizerContext(
                namespace=namespace,
                protocol=protocol,
                group_file=group_file,
                topic_plans=topic_plans,
                topic_acks=topic_acks,
                topic_registry=topic_registry,
            )

        ctx.start()
        _global_context = ctx
        return _global_context
