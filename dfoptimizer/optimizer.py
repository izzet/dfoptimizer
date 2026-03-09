"""DFOptimizer service: consumes DiagnosisFindings, produces ActionPlans.

Runs as a standalone process alongside the pipeline:
  DFTracer -> Mofka -> DFAnalyzer -> Mofka -> DFDiagnoser -> Mofka -> DFOptimizer

The optimizer starts with an empty planner. Apps register their knobs
(including responds_to rules) via the ``optimizer.registry`` Mofka topic.
Once knobs are registered, the planner can match incoming findings to
knob responses and emit ActionPlans.
"""

import dataclasses
import json
import os
import signal
import threading
import time
from typing import Dict, List

import structlog

from .types import ActionPlan, DiagnosisFindingMsg, KnobDef
from .runtime.knob import knob_def_from_wire
from .planner.planner import Planner

logger = structlog.get_logger()

_shutdown_requested = False


def _sigterm_handler(signum, frame):
    del signum, frame
    global _shutdown_requested
    _shutdown_requested = True


def install_shutdown_handler():
    global _shutdown_requested
    _shutdown_requested = False
    signal.signal(signal.SIGTERM, _sigterm_handler)


class Optimizer:
    def __init__(self):
        self.planner = Planner()
        self._plans_produced = 0

    def run_mofka(
        self,
        group_file: str,
        input_topic: str = "diagnosis.findings",
        output_topic: str = "optimizer.plans",
        registry_topic: str = "optimizer.registry",
        consumer_name: str = "",
        idle_timeout_sec: int = 0,
        pull_timeout_ms: int = 1000,
    ):
        from .streaming.mofka_io import open_consumer, open_producer

        # Open ALL Mofka connections in the main thread to avoid GIL
        # blocking from Mofka C extension network calls in background threads
        _, consumer = open_consumer(
            group_file, input_topic,
            consumer_name=consumer_name or f"dfoptimizer_{os.getpid()}",
        )
        _, producer = open_producer(group_file, output_topic)
        _, registry_consumer = open_consumer(
            group_file, registry_topic,
            consumer_name=f"dfoptimizer_registry_{os.getpid()}",
        )

        install_shutdown_handler()

        # Start registry listener in background thread (consumer already open)
        registry_thread = threading.Thread(
            target=self._registry_loop,
            args=(group_file, registry_topic, registry_consumer),
            daemon=True,
            name="optimizer-registry",
        )
        registry_thread.start()

        event_count = 0
        plan_count = 0
        error_count = 0
        last_event_time = None
        timeout_count = 0
        wait_ms = pull_timeout_ms if pull_timeout_ms > 0 else 1000

        logger.info(
            "optimizer.stream.start",
            input_topic=input_topic,
            output_topic=output_topic,
            registry_topic=registry_topic,
            idle_timeout_sec=idle_timeout_sec,
        )

        try:
            future = consumer.pull()
            while not _shutdown_requested:
                now = time.monotonic()
                if (
                    last_event_time is not None
                    and idle_timeout_sec > 0
                    and (now - last_event_time) >= idle_timeout_sec
                ):
                    logger.info(
                        "optimizer.stream.idle_timeout",
                        idle_sec=round(now - last_event_time, 1),
                    )
                    break

                try:
                    event = future.wait(timeout_ms=wait_ms)
                except Exception as ex:
                    if "timeout" in str(ex).lower():
                        timeout_count += 1
                        continue
                    raise

                if event is None:
                    timeout_count += 1
                    continue

                last_event_time = time.monotonic()
                event_count += 1

                try:
                    finding = self._parse_finding(event)
                    if finding is not None:
                        plans = self.planner.process_finding(finding)
                        for plan in plans:
                            self._publish_plan(producer, plan)
                            plan_count += 1
                except Exception:
                    error_count += 1
                    logger.exception("optimizer.event.error", event_index=event_count)

                event.acknowledge()
                future = consumer.pull()

            if _shutdown_requested:
                logger.info("optimizer.stream.stop_signal", signal="SIGTERM")

        finally:
            producer.flush()
            registry_thread.join(timeout=wait_ms / 1000.0 + 1.0)
            logger.info(
                "optimizer.stream.done",
                event_count=event_count,
                plan_count=plan_count,
                error_count=error_count,
                knob_state=dict(self.planner.current_values),
            )
            del consumer
            del producer

    def _registry_loop(self, group_file: str, registry_topic: str, consumer=None):
        """Background thread: listen for knob registrations from apps."""
        if consumer is None:
            try:
                from .streaming.mofka_io import open_consumer
                _, consumer = open_consumer(
                    group_file, registry_topic,
                    consumer_name=f"dfoptimizer_registry_{os.getpid()}",
                )
            except Exception:
                logger.warning("optimizer.registry.connect_failed", topic=registry_topic, exc_info=True)
                return

        logger.info("optimizer.registry.listening", topic=registry_topic)
        future = consumer.pull()

        while not _shutdown_requested:
            try:
                event = future.wait(timeout_ms=1000)
            except Exception as ex:
                if "timeout" in str(ex).lower():
                    continue
                logger.error("optimizer.registry.listen_error", error=str(ex))
                break

            if event is None:
                continue

            try:
                self._handle_registration(event)
            except Exception:
                logger.exception("optimizer.registry.error")

            event.acknowledge()
            future = consumer.pull()

        if _shutdown_requested:
            logger.info("optimizer.registry.stop_signal", signal="SIGTERM")

        del consumer

    def _handle_registration(self, event):
        payload = event.data
        if payload is None:
            return
        if isinstance(payload, list):
            payload = b"".join(payload)

        msg = json.loads(payload.decode("utf-8"))
        namespace = msg.get("namespace", "app")
        func_name = msg.get("function_name", "")
        knobs_wire = msg.get("knobs", {})

        knob_defs = {}
        for param_name, kw in knobs_wire.items():
            kdef = knob_def_from_wire(kw)
            knob_defs[kdef.id] = kdef

        logger.info(
            "optimizer.registry.received",
            namespace=namespace,
            function=func_name,
            knobs=list(knob_defs.keys()),
        )

        self.planner.register_knobs(knob_defs)

    def _parse_finding(self, event) -> DiagnosisFindingMsg | None:
        raw_metadata = event.metadata if hasattr(event, "metadata") else None
        if isinstance(raw_metadata, dict):
            metadata = raw_metadata
        elif isinstance(raw_metadata, str):
            try:
                metadata = json.loads(raw_metadata)
            except (ValueError, TypeError):
                metadata = {}
        else:
            metadata = {}

        # Check for stop sentinel
        if metadata.get("name") == "end" or metadata.get("type") == "stop":
            logger.info("optimizer.stream.stop_sentinel")
            return None

        payload = event.data
        if payload is None:
            return None
        if isinstance(payload, list):
            payload = b"".join(payload)

        msg = json.loads(payload.decode("utf-8"))

        return DiagnosisFindingMsg(
            finding_type=msg.get("finding_type", ""),
            scope=msg.get("scope", ""),
            layer=msg.get("layer"),
            motif=msg.get("motif", "unclassified"),
            severity=msg.get("severity", "unknown"),
            confidence=msg.get("confidence", 0),
            prevalence=msg.get("prevalence", 0),
            persistence=msg.get("persistence", 0),
            support_windows=msg.get("support_windows", 0),
            trend_direction=msg.get("trend_direction", "insufficient_data"),
            last_seen_window=msg.get("last_seen_window", msg.get("window_index", 0)),
            contributing_facts=[
                tuple(f) for f in msg.get("contributing_facts", [])
            ],
            recommendation_bundle=msg.get("recommendation_bundle", ""),
            opportunity_tags=msg.get("opportunity_tags", []),
            summary=msg.get("summary", ""),
            window_index=msg.get("window_index", 0),
            publish_mode=msg.get("publish_mode", "control"),
        )

    def _publish_plan(self, producer, plan: ActionPlan):
        payload = json.dumps(dataclasses.asdict(plan)).encode("utf-8")
        metadata = {
            "type": "action_plan",
            "plan_id": plan.plan_id,
            "knob_id": plan.knob_id,
            "target_function": plan.target_function,
        }
        producer.push(metadata=metadata, data=payload)
        logger.info(
            "optimizer.plan.published",
            plan_id=plan.plan_id,
            knob_id=plan.knob_id,
            old_value=plan.old_value,
            new_value=plan.new_value,
            target_function=plan.target_function,
            rationale=plan.rationale,
        )
        self._plans_produced += 1
