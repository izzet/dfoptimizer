import json
import os
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
        self._noop = False

    def start(self):
        """Start the background Mofka listener thread."""
        if self._noop:
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
        self._consumer = None
        self._listener_driver = None
        logger.info("optimizer.context.stopped")

    def drain_actions_for(self, func_name: str) -> List[ActionPlan]:
        """Return and clear all pending actions for a given target function."""
        with self._lock:
            actions = list(self._queues[func_name].values())
            self._queues[func_name].clear()
        return actions

    def ack_action(self, plan: ActionPlan, old_value, new_value, status="applied"):
        ack = ActionAck(
            plan_id=plan.plan_id,
            knob_id=plan.knob_id,
            status=status,
            old_value=old_value,
            new_value=new_value,
        )
        logger.info(
            "optimizer.action.ack",
            plan_id=ack.plan_id,
            knob_id=ack.knob_id,
            old_value=old_value,
            new_value=new_value,
            status=status,
        )
        self._publish_ack(ack)

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
            from ..streaming.mofka_io import open_consumer
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
            try:
                event = future.wait(timeout_ms=500)
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

            event.acknowledge()
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
        )

        logger.info(
            "optimizer.context.plan_received",
            plan_id=plan.plan_id,
            knob_id=plan.knob_id,
            new_value=plan.new_value,
        )

        with self._lock:
            # Latest-wins: overwrite any older action for same knob+function
            self._queues[plan.target_function][plan.knob_id] = plan

    def _publish_ack(self, ack: ActionAck):
        if self._noop or not self.group_file:
            return
        try:
            if self._producer is None:
                from ..streaming.mofka_io import open_producer
                _, self._producer = open_producer(self.group_file, self.topic_acks)

            import dataclasses
            payload = json.dumps(dataclasses.asdict(ack)).encode("utf-8")
            metadata = {"type": "action_ack", "plan_id": ack.plan_id}
            self._producer.push(metadata=metadata, data=payload)
            self._producer.flush()
        except Exception:
            logger.warning("optimizer.ack.publish_failed", plan_id=ack.plan_id, exc_info=True)

    def _publish_registration(self, registration: dict):
        if self._noop or not self.group_file:
            return
        try:
            if self._registry_producer is None:
                from ..streaming.mofka_io import open_producer
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
            logger.info(
                "optimizer.knobs.published",
                namespace=registration["namespace"],
                function=registration["function_name"],
                knob_count=len(registration["knobs"]),
            )
        except Exception:
            logger.warning("optimizer.knobs.publish_failed", exc_info=True)

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

    If Mofka is not available (no group_file or import fails), returns a
    no-op context so the app runs unchanged.
    """
    global _global_context
    with _context_lock:
        if _global_context is not None:
            return _global_context

        if not group_file or not os.path.exists(group_file):
            logger.info("optimizer.context.no_group_file")
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
