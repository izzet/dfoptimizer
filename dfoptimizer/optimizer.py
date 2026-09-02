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

from .streaming.transport import ack_event

from .types import ActionPlan, DiagnosisFindingMsg, KnobDef
from .runtime.knob import knob_def_from_dict, knob_def_from_wire
from .planner.planner import Planner

logger = structlog.get_logger()


def _make_planner():
    """Select the optimizer algorithm. Default = the rule/Amdahl Planner (LiveIO byte-identical).
    DFOPTIMIZER_ALGO=clbo selects the factored cross-layer surrogate (LiveFlow), imported lazily
    so LiveIO runs never load it."""
    algo = os.environ.get("DFOPTIMIZER_ALGO", "planner").lower()
    if algo == "clbo":
        from .planner.clbo_planner import CLBOPlanner
        logger.info("optimizer.algo.selected", algo="clbo")
        return CLBOPlanner()
    if algo == "cbandit":
        # Cross-layer-directed coordinate bandit: argmax domain picks the knob, UCB over
        # {stay, down, up} picks the value, probe cost discounts the exploration bonus.
        # Built because CLBO's acquisition cannot return the incumbent and so pays cumulative
        # regret every window; see cbandit_planner.py for the measurements.
        from .planner.cbandit_planner import CoordinateBanditPlanner
        logger.info("optimizer.algo.selected", algo="cbandit")
        return CoordinateBanditPlanner()
    return Planner()

_shutdown_requested = False


def _sigterm_handler(signum, frame):
    del signum, frame
    global _shutdown_requested
    _shutdown_requested = True


def install_shutdown_handler():
    """Install the SIGTERM handler. No-op off the main thread.

    `signal.signal` raises ValueError anywhere but the main thread, and under the Option C node
    agent (LIVEFLOW-ARCHITECTURE.md sec 6) this optimizer runs as ONE THREAD alongside the
    analyzer and diagnoser instead of owning a process. Signal disposition is process-wide, so a
    stage thread was never the right place to set it: whoever owns the process installs the
    handler and shuts the stages down through the same flag.

    Same change as dfanalyzer's install_shutdown_handler, for the same reason.
    """
    global _shutdown_requested
    _shutdown_requested = False
    if threading.current_thread() is not threading.main_thread():
        return
    signal.signal(signal.SIGTERM, _sigterm_handler)


class Optimizer:
    def __init__(self):
        self.planner = _make_planner()
        self._plans_produced = 0
        self._output_topic = ""
        self._plan_fanout = 1

    def _maybe_bootstrap_knobs(self):
        if os.environ.get("DFOPTIMIZER_BOOTSTRAP_DLIO", "0") != "1":
            return

        if self.planner.knobs:
            return

        knob_boundary = os.environ.get("DLIO_KNOB_BOUNDARY", "window_boundary")
        knob_min_threads = int(os.environ.get("DLIO_KNOB_MIN_THREADS", "0"))
        knob_max_threads = int(
            os.environ.get("DLIO_KNOB_MAX_THREADS", str(os.cpu_count() or 8))
        )
        if knob_min_threads > knob_max_threads:
            knob_min_threads = knob_max_threads

        bootstrap_read_threads = int(
            os.environ.get(
                "DFOPTIMIZER_BOOTSTRAP_READ_THREADS",
                str(max(knob_min_threads, 1 if knob_max_threads > 0 else 0)),
            )
        )
        bootstrap_prefetch_size = int(
            os.environ.get("DFOPTIMIZER_BOOTSTRAP_PREFETCH_SIZE", "2")
        )
        bootstrap_data_source = os.environ.get(
            "DFOPTIMIZER_BOOTSTRAP_DATA_SOURCE", "pfs"
        )
        # See main.py: data_source gets its own boundary because its apply contains a world
        # collective and its abort test needs a FINISHED epoch to compute a horizon from.
        data_source_boundary = os.environ.get(
            "DLIO_DATA_SOURCE_BOUNDARY", "epoch_boundary"
        )
        # Costs, seconds, all measured on Corona; see main.py for the derivations. Carried here so
        # the bootstrap registry is not a cheaper-looking copy of the application's.
        stage_cost = float(os.environ.get("LIVEFLOW_STAGE_COST", "465"))
        restripe_cost = float(os.environ.get("LIVEFLOW_RESTRIPE_COST", "15840"))
        rebuild_cost = float(os.environ.get("LIVEFLOW_REBUILD_COST", "0.206"))

        knob_defs = {
            "dlio.prefetch_size": knob_def_from_dict(
                "dlio.prefetch_size",
                {
                    "default": 2,
                    "type": int,
                    "range": (1, 16),
                    "scope": "job",
                    "responds_to": {
                        "dataloader_prefetch": {
                            "direction": "increase",
                            "step_mode": "add",
                            "step": 2,
                            "min_persistence": 1,
                            "cooldown_windows": 2,
                            "apply_when": knob_boundary,
                        },
                    },
                },
                target_function="DLIOBenchmark.make_loader",
            ),
            "dlio.read_threads": knob_def_from_dict(
                "dlio.read_threads",
                {
                    "default": max(0, knob_min_threads),
                    "type": int,
                    "range": (knob_min_threads, knob_max_threads),
                    "scope": "job",
                    "responds_to": {
                        "reader_parallelism": {
                            "direction": "increase",
                            "step_mode": "evidence",
                            "min_persistence": 1,
                            "cooldown_windows": 1,
                            "apply_when": knob_boundary,
                        },
                        "reader_contention": {
                            "direction": "decrease",
                            "step": 1,
                            "min_persistence": 2,
                            "cooldown_windows": 3,
                            "apply_when": knob_boundary,
                        },
                    },
                },
                target_function="DLIOBenchmark.make_loader",
            ),
            # WHERE THE BYTES COME FROM. Declared here as well as in the application because the
            # bootstrap WINS THE RACE: it runs the moment the optimizer stage starts, and the
            # application's own five-knob registry does not arrive until its first make_loader call,
            # about 20 s later under the agent harness. `_maybe_bootstrap_knobs` returns early once
            # any knob is registered, so with only two knobs declared here the planner's first view
            # of the world had no data-location actuator in it and could not have proposed one, and
            # nothing anywhere reported that. register_knobs MERGES, so the application's definition
            # still lands and wins on arrival; this only closes the window before it.
            "dlio.data_source": knob_def_from_dict(
                "dlio.data_source",
                {
                    "default": "pfs",
                    "type": str,
                    # "restriped" is opt-in: its resolver always declines (registered and costed,
                    # not implemented), so as a declared level it is an arm that can never be
                    # satisfied, never accumulates samples, and therefore never stops being
                    # explored. It took the epoch boundary from "local" twice at 2 nodes.
                    "values": (["pfs", "local", "restriped"]
                               if os.environ.get("DFOPTIMIZER_REGISTER_RESTRIPED", "0") == "1"
                               else ["pfs", "local"]),
                    "scope": "node",
                    # Cost per VALUE, not per knob: "pfs" is free, "local" is a 465 s collective
                    # copy, "restriped" is a 4.4 h in-place rewrite. One number per knob would make
                    # the restripe look as cheap as a stage or the stage as dear as a restripe.
                    "cost_seconds": {"pfs": 0.0, "local": stage_cost,
                                     "restriped": restripe_cost},
                    "responds_to": {
                        "storage_starved": {
                            "direction": "set",
                            "set_to": "local",
                            "min_persistence": 1,
                            "cooldown_windows": 999,
                            "apply_when": data_source_boundary,
                        },
                    },
                },
                target_function="DLIOBenchmark.make_loader",
            ),
            # Read SHAPE, OFF BY DEFAULT: `make_loader` has no such parameter, so @tunable acks
            # every plan `knob_not_in_kwargs` (open bug #30). Measured at 2 nodes: 144,048
            # rejections, 0 applications, and because a never-applied arm stays under-sampled the
            # acceptance gate never stops it and it absorbed the whole escalation.
            **({"dlio.transfer_size": knob_def_from_dict(
                "dlio.transfer_size",
                {
                    "default": 262144,
                    "type": int,
                    "values": [262144, 1048576, 4194304, 16777216],
                    "scope": "job",
                    "cost_seconds": rebuild_cost,
                    "responds_to": {},
                },
                target_function="DLIOBenchmark.make_loader",
            )} if os.environ.get("DFOPTIMIZER_REGISTER_TRANSFER_SIZE", "0") == "1" else {}),
            # Decode parallelism, OFF BY DEFAULT -- see main.py for the measurement. In short: it
            # falls to the default domain "compute", DLIO's compute is a time.sleep that no actuator
            # can shorten, so a compute-dominant window handed the planner one knob and it drove it
            # to 1 in 44 of 46 windows and made the run 50% slower. Unregistered, the compute domain
            # owns nothing and a compute-dominant window declines, which is the honest answer.
            **({"dlio.computation_threads": knob_def_from_dict(
                "dlio.computation_threads",
                {
                    "default": 1,
                    "type": int,
                    "range": (1, knob_max_threads),
                    "scope": "job",
                    "cost_seconds": rebuild_cost,
                    "responds_to": {
                        "decode_pressure": {
                            "direction": "increase",
                            "step_mode": "add",
                            "step": 2,
                            "min_persistence": 1,
                            "cooldown_windows": 2,
                            "apply_when": knob_boundary,
                        },
                    },
                },
                target_function="DLIOBenchmark.make_loader",
            )} if os.environ.get("DFOPTIMIZER_REGISTER_COMPUTATION_THREADS", "0") == "1" else {}),
        }
        current_values = {
            "dlio.prefetch_size": bootstrap_prefetch_size,
            "dlio.read_threads": bootstrap_read_threads,
            "dlio.data_source": bootstrap_data_source,
        }
        if "dlio.transfer_size" in knob_defs:
            current_values["dlio.transfer_size"] = int(
                os.environ.get("DFOPTIMIZER_BOOTSTRAP_TRANSFER_SIZE", "262144"))
        if "dlio.computation_threads" in knob_defs:
            current_values["dlio.computation_threads"] = int(
                os.environ.get("DFOPTIMIZER_BOOTSTRAP_COMPUTATION_THREADS", "1"))
        self.planner.register_knobs(knob_defs, current_values=current_values)
        logger.info(
            "optimizer.registry.bootstrapped",
            source="env",
            knobs=list(knob_defs.keys()),
            current_values=current_values,
        )

    def _wait_for_event(self, future, wait_ms: int):
        try:
            return future.wait(timeout_ms=wait_ms)
        except Exception as ex:
            if "timeout" in str(ex).lower():
                return None
            raise

    def _process_global_plan_event(self, event, local_producer):
        import socket

        local_hostname = socket.gethostname()
        metadata = event.metadata
        payload = event.data
        if isinstance(payload, list):
            payload = b"".join(payload)

        msg = json.loads(payload.decode("utf-8"))

        target_nodes = msg.get("target_nodes", [])
        if target_nodes and local_hostname not in target_nodes:
            logger.debug(
                "optimizer.global_relay.not_targeted",
                target_nodes=target_nodes,
                local_hostname=local_hostname,
            )
            return False

        plan = ActionPlan(
            plan_id=msg.get("plan_id", ""),
            knob_id=msg.get("knob_id", ""),
            target_function=msg.get("target_function", ""),
            old_value=msg.get("old_value"),
            new_value=msg.get("new_value"),
            apply_when=msg.get("apply_when", "next_window"),
            rationale=msg.get("rationale", "global optimizer"),
            finding_type=msg.get("finding_type", "global"),
            severity=msg.get("severity", 0.0),
            opportunity_tag=msg.get("opportunity_tag", ""),
            window_index=msg.get("window_index", -1),
            target_nodes=target_nodes,
        )

        accepted = self.planner.apply_global_plan(plan)
        if accepted is not None:
            self._publish_plan(local_producer, accepted)
            return True
        return False

    def run_mofka(
        self,
        group_file: str,
        input_topic: str = "diagnosis.findings",
        output_topic: str = "optimizer.plans",
        registry_topic: str = "optimizer.registry",
        consumer_name: str = "",
        idle_timeout_sec: int = 0,
        pull_timeout_ms: int = 1000,
        no_registry: bool = False,
        global_group_file: str = "",
        global_input_topic: str = "global_plans",
        relay_only: bool = False,
    ):
        from .streaming.transport import open_consumer, open_producer

        self._output_topic = output_topic
        if output_topic == "global_plans":
            self._plan_fanout = max(
                1,
                int(os.environ.get("DFOPTIMIZER_GLOBAL_PLAN_FANOUT", "1")),
            )
        else:
            self._plan_fanout = 1

        # Open ALL Mofka connections in the main thread to avoid GIL
        # blocking from Mofka C extension network calls in background threads
        consumer = None
        if not relay_only:
            _, consumer = open_consumer(
                group_file, input_topic,
                consumer_name=consumer_name or f"dfoptimizer_{os.getpid()}",
            )
        _, producer = open_producer(group_file, output_topic)

        registry_consumer = None
        ack_consumer = None
        if not no_registry:
            _, registry_consumer = open_consumer(
                group_file, registry_topic,
                consumer_name=f"dfoptimizer_registry_{os.getpid()}",
            )

        # ACKS ARE NOT PART OF THE REGISTRY. This used to sit inside the `if not no_registry`
        # block above, so --no-registry -- documented as "for a global optimizer that has no knob
        # defs" -- silently switched off the acknowledgement path as well.
        #
        # That matters more than it looks: the cooldown timer and the effectiveness veto only
        # record on an ack (sec 83.5). With no ack consumer they never record, `_last_action`
        # stays empty, and both gates return not-blocked forever -- the exact symptom sec 83
        # spent a session chasing. Two unrelated concerns behind one flag.
        _, ack_consumer = open_consumer(
            group_file, "optimizer_acks",
            consumer_name=f"dfoptimizer_acks_{os.getpid()}",
        )

        # Open global plan consumer (ofi+cxi) in main thread if configured
        global_consumer = None
        if global_group_file:
            try:
                import mochi.mofka.client as mofka
                logger.info("optimizer.global_driver.create", group_file=global_group_file)
                global_driver = mofka.MofkaDriver(
                    group_file=global_group_file, use_progress_thread=True,
                )
                global_topic = global_driver.open_topic(global_input_topic)
                global_consumer = global_topic.consumer(
                    name=f"dfoptimizer_global_{os.getpid()}",
                    batch_size=mofka.AdaptiveBatchSize,
                    data_allocator=mofka.ByteArrayAllocator,
                    data_selector=mofka.FullDataSelector,
                    thread_pool=global_driver.default_thread_pool,
                )
                logger.info("optimizer.global_consumer.ready", topic=global_input_topic)
            except Exception:
                logger.warning("optimizer.global_consumer.failed", exc_info=True)

        install_shutdown_handler()

        # Start registry listener in background thread (consumer already open)
        registry_thread = None
        if registry_consumer is not None:
            registry_thread = threading.Thread(
                target=self._registry_loop,
                args=(group_file, registry_topic, registry_consumer),
                daemon=True,
                name="optimizer-registry",
            )
            registry_thread.start()

        self._maybe_bootstrap_knobs()

        # Start ack listener in background thread
        ack_thread = None
        if ack_consumer is not None:
            ack_thread = threading.Thread(
                target=self._ack_loop,
                args=(ack_consumer,),
                daemon=True,
                name="optimizer-acks",
            )
            ack_thread.start()

        # Start global plan relay if consumer available
        global_thread = None
        if global_consumer is not None and not relay_only:
            global_thread = threading.Thread(
                target=self._global_plans_loop,
                args=(global_consumer, producer),
                daemon=True,
                name="optimizer-global-relay",
            )
            global_thread.start()

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
            relay_only=relay_only,
        )

        try:
            if relay_only:
                global_future = global_consumer.pull() if global_consumer is not None else None
                while not _shutdown_requested:
                    if global_future is None:
                        time.sleep(wait_ms / 1000.0)
                        continue

                    event = self._wait_for_event(global_future, wait_ms)
                    if event is None:
                        continue

                    try:
                        self._process_global_plan_event(event, producer)
                    except Exception:
                        logger.exception("optimizer.global_relay.process_error")

                    ack_event(event)
                    global_future = global_consumer.pull()
            else:
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
                            # ONE message for the whole window, not one per plan. See
                            # _publish_plans: the mailbox is a single slot and per-plan
                            # publishes overwrite each other.
                            self._publish_plans(producer, plans)
                            plan_count += len(plans)
                    except Exception:
                        error_count += 1
                        logger.exception("optimizer.event.error", event_index=event_count)

                    ack_event(event)
                    future = consumer.pull()

            if _shutdown_requested:
                logger.info("optimizer.stream.stop_signal", signal="SIGTERM")

        finally:
            producer.flush()
            if registry_thread is not None:
                registry_thread.join(timeout=wait_ms / 1000.0 + 1.0)
            if ack_thread is not None:
                ack_thread.join(timeout=wait_ms / 1000.0 + 1.0)
            if global_thread is not None:
                global_thread.join(timeout=wait_ms / 1000.0 + 1.0)
            logger.info(
                "optimizer.stream.done",
                event_count=event_count,
                plan_count=plan_count,
                error_count=error_count,
                knob_state=dict(self.planner.current_values),
            )
            if consumer is not None:
                del consumer
            del producer

    def _registry_loop(self, group_file: str, registry_topic: str, consumer=None):
        """Background thread: listen for knob registrations from apps."""
        if consumer is None:
            try:
                from .streaming.transport import open_consumer
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

            ack_event(event)
            future = consumer.pull()

        if _shutdown_requested:
            logger.info("optimizer.registry.stop_signal", signal="SIGTERM")

        del consumer

    def _ack_loop(self, consumer):
        """Background thread: listen for plan acks from apps."""
        logger.info("optimizer.acks.listening")
        future = consumer.pull()

        while not _shutdown_requested:
            try:
                event = future.wait(timeout_ms=1000)
            except Exception as ex:
                if "timeout" in str(ex).lower():
                    continue
                logger.error("optimizer.acks.listen_error", error=str(ex))
                break

            if event is None:
                continue

            try:
                payload = event.data
                if isinstance(payload, list):
                    payload = b"".join(payload)
                msg = json.loads(payload.decode("utf-8"))
                self.planner.apply_ack(
                    plan_id=msg.get("plan_id", ""),
                    knob_id=msg.get("knob_id", ""),
                    status=msg.get("status", "unknown"),
                    old_value=msg.get("old_value"),
                    new_value=msg.get("new_value"),
                    window_index=msg.get("window_index", -1),
                    target_nodes=msg.get("target_nodes"),
                )
            except Exception:
                logger.exception("optimizer.acks.error")

            ack_event(event)
            future = consumer.pull()

        if _shutdown_requested:
            logger.info("optimizer.acks.stop_signal", signal="SIGTERM")
        del consumer

    def _global_plans_loop(self, consumer, local_producer):
        """Background thread: receive global plans, validate via planner, publish locally."""
        import socket
        local_hostname = socket.gethostname()
        logger.info("optimizer.global_relay.listening", local_hostname=local_hostname)
        future = consumer.pull()
        accepted_count = 0
        skipped_count = 0

        while not _shutdown_requested:
            try:
                event = self._wait_for_event(future, 1000)
            except Exception as ex:
                logger.error("optimizer.global_relay.error", error=str(ex))
                break

            if event is None:
                continue

            try:
                if self._process_global_plan_event(event, local_producer):
                    accepted_count += 1
                else:
                    skipped_count += 1
            except Exception:
                logger.exception("optimizer.global_relay.process_error")

            ack_event(event)
            future = consumer.pull()

        logger.info(
            "optimizer.global_relay.done",
            accepted=accepted_count,
            skipped=skipped_count,
        )

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
        wire_current_values = msg.get("current_values", {})

        # DFTracer sends knobs as a JSON array of objects (each with "id"),
        # while DLIO sends a dict keyed by param name.  Normalize to dict.
        if isinstance(knobs_wire, list):
            knobs_wire = {k.get("id", f"knob_{i}"): k for i, k in enumerate(knobs_wire)}

        knob_defs = {}
        current_values = {}
        for param_name, kw in knobs_wire.items():
            # DFTracer uses "knob_response" (single response applied to all
            # severity tags), while DLIO uses "responds_to" (tag -> response
            # dict).  Normalize to responds_to format.
            if "knob_response" in kw and "responds_to" not in kw:
                resp = kw.pop("knob_response")
                kw["responds_to"] = {"io_bottleneck": resp}
            kdef = knob_def_from_wire(kw)
            knob_defs[kdef.id] = kdef
            if param_name in wire_current_values:
                current_values[kdef.id] = wire_current_values[param_name]
            elif kdef.id in wire_current_values:
                current_values[kdef.id] = wire_current_values[kdef.id]

        logger.info(
            "optimizer.registry.received",
            namespace=namespace,
            function=func_name,
            knobs=list(knob_defs.keys()),
            current_values=current_values,
        )

        self.planner.register_knobs(knob_defs, current_values=current_values)

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
            severity_score=float(msg.get("severity_score", 0.0)),
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
            suppresses_tags=msg.get("suppresses_tags", []),
            key_metrics=msg.get("key_metrics", {}),
        )

    def _publish_plans(self, producer, plans):
        """Publish a window's plans as ONE message.

        One message per plan loses all but the last under the broadcast-mailbox transport, which is
        a single versioned SLOT rather than a queue (shm_mailbox.ShmMailbox, "One versioned
        broadcast slot"). Three plans pushed back to back overwrite one another and the application
        polls once, so it sees only whichever was published last.

        Measured: window 0 published read_threads 1->2, prefetch_size 1->2 and data_source
        pfs->local, and the app logged plan_received for data_source ALONE. The knobs never moved.

        The single-slot semantic is right for ONE knob -- a reader that was paused wants the current
        value, not a queue of superseded ones -- but wrong across DISTINCT knobs, because
        read_threads=2 is not superseded by data_source=local. They are independent state. Sending
        the window's whole desired state in one message is what the mailbox docstring already says
        plans ARE: state, not events. The receiver keys them per knob (context._queues), so
        latest-wins is preserved exactly as before.
        """
        plans = list(plans)
        if not plans:
            return
        payload = json.dumps(
            {"type": "action_plan_batch",
             "plans": [dataclasses.asdict(p) for p in plans]}
        ).encode("utf-8")
        metadata = {
            "type": "action_plan_batch",
            "plan_count": len(plans),
            "knob_ids": [p.knob_id for p in plans],
            "target_function": plans[0].target_function,
        }
        # target_nodes is per-plan and the receiver filters on the plan body, so the metadata copy
        # is advisory only. Carry it when every plan agrees, which is the case the agent produces.
        node_sets = {tuple(p.target_nodes or ()) for p in plans}
        if len(node_sets) == 1 and plans[0].target_nodes:
            metadata["target_nodes"] = plans[0].target_nodes
        for _ in range(self._plan_fanout):
            producer.push(metadata=metadata, data=payload)
        for plan in plans:
            self._log_published(plan)
            self._plans_produced += 1

    def _publish_plan(self, producer, plan: ActionPlan):
        self._publish_plans(producer, [plan])

    def _log_published(self, plan: ActionPlan):
        logger.info(
            "optimizer.plan.published",
            plan_id=plan.plan_id,
            knob_id=plan.knob_id,
            old_value=plan.old_value,
            new_value=plan.new_value,
            target_function=plan.target_function,
            target_nodes=plan.target_nodes or "all",
            fanout=self._plan_fanout,
            rationale=plan.rationale,
        )
