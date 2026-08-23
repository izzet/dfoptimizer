import json
import os
import threading
from typing import Optional

import structlog

logger = structlog.get_logger()

_driver_singleton = None
_driver_pid = None
_driver_lock = threading.Lock()


def _load_mofka():
    try:
        import mochi.mofka.client as mofka
    except ModuleNotFoundError as exc:
        raise RuntimeError("Mofka client is not available. Install mochi.mofka.") from exc
    return mofka


def _get_driver(group_file: str, use_progress_thread: bool):
    global _driver_singleton, _driver_pid
    # DFOPTIMIZER_MOFKA_PROGRESS_THREAD overrides: "0" disables the dedicated (busy-spinning) Mercury
    # progress thread -> progress is driven on-demand by the calling thread. Used to test whether the
    # progress thread starves NCCL-P2P at scale (halo3d 16-way hang). Default unchanged (True).
    _env = os.environ.get("DFOPTIMIZER_MOFKA_PROGRESS_THREAD")
    if _env is not None:
        use_progress_thread = _env == "1"
    with _driver_lock:
        if _driver_singleton is None or _driver_pid != os.getpid():
            mofka = _load_mofka()
            _driver_pid = os.getpid()
            logger.info("mofka.driver.create", pid=_driver_pid, group=group_file,
                        use_progress_thread=use_progress_thread)
            # THE OPTION MUST BE NESTED UNDER "margo" (LIVEFLOW-STATUS.md sec 57).
            #
            # MofkaDriver.__init__(**kwargs) forwards kwargs verbatim as the driver's JSON options,
            # and MofkaDriver.cpp reads ONLY `config.value("margo", {})` before handing it to
            # thallium. A TOP-LEVEL `use_progress_thread` is therefore parsed by nobody and
            # silently discarded -- which is how DFOPTIMIZER_MOFKA_PROGRESS_THREAD came to be a
            # no-op for its entire life, and why toggling it (G2) changed nothing. DFTracer's C++
            # writer gets this right: options.json()["margo"]["use_progress_thread"].
            #
            # Consequence: this engine has been running on margo DEFAULTS all along, so "we tried
            # disabling the progress thread" was never actually true. DFOPTIMIZER_MOFKA_CONFIG
            # accepts a JSON object merged into the margo block, so the rest of the margo/argobots
            # surface (rpc_thread_count, progress_timeout_ub_msec, argobots pools/xstreams) is
            # reachable for tuning without patching Mofka.
            _margo = {"use_progress_thread": bool(use_progress_thread)}
            _extra = os.environ.get("DFOPTIMIZER_MOFKA_CONFIG", "").strip()
            if _extra:
                try:
                    _margo.update(json.loads(_extra))
                except Exception as _e:
                    logger.warning("mofka.driver.config_parse_failed", error=str(_e), raw=_extra)
            logger.info("mofka.driver.margo_config", margo=_margo)
            _driver_singleton = mofka.MofkaDriver(
                group_file=group_file,
                margo=_margo,
            )
        else:
            logger.debug("mofka.driver.reuse", pid=_driver_pid)
    return _driver_singleton


def open_consumer(
    group_file: str,
    topic_name: str,
    consumer_name: Optional[str] = None,
    use_progress_thread: bool = True,
):
    mofka = _load_mofka()
    driver = _get_driver(group_file, use_progress_thread)
    topic = driver.open_topic(topic_name)
    effective_name = consumer_name or f"dfoptimizer_{os.getpid()}"
    logger.info("mofka.consumer.open", topic=topic_name, name=effective_name)
    consumer = topic.consumer(
        name=effective_name,
        thread_pool=driver.default_thread_pool,
        batch_size=mofka.AdaptiveBatchSize,
        data_allocator=mofka.ByteArrayAllocator,
        data_selector=mofka.FullDataSelector,
    )
    return driver, consumer


def open_producer(
    group_file: str,
    topic_name: str,
    use_progress_thread: bool = True,
):
    mofka = _load_mofka()
    driver = _get_driver(group_file, use_progress_thread)
    topic = driver.open_topic(topic_name)
    logger.info("mofka.producer.open", topic=topic_name)
    producer = topic.producer(
        batch_size=mofka.AdaptiveBatchSize,
        thread_pool=driver.default_thread_pool,
        ordering=mofka.Ordering.Strict,
    )
    return driver, producer


# Acknowledgement is DISABLED by default. See LIVEFLOW-STATUS sec 60.
#
# Measured on diaspora-stream-api 0.5.1 with liveflow_validation/repro_consumer_stall.sh
# (local na+sm bedrock, one single-partition topic, one producer, one consumer whose loop
# mirrors optimizer.py::run_mofka), 100 events published in flushes of 5:
#
#     ACK on   ->  1 event received of 100, consumer never delivered another
#     ACK off  -> 100 events received of 100, clean exit
#
# Acking an event that belongs to a multi-event batch permanently stops delivery to that
# consumer. It also leaves the group in a state where the producer process hangs on exit.
# This is what starved the optimizer in production: the diagnoser published 129 findings,
# the optimizer received exactly 1 (in 1.8 ms) and then nothing for 17 minutes, on every
# node. The diagnoser's own consumer was unaffected only because the analyzer feeds it one
# event per flush, so its batches never hold more than one event.
#
# Dropping the ack costs LiveFlow nothing. An acknowledgement exists so a restarted
# consumer of the same name can resume after its last acked event. Our consumers are named
# per-PID (`dfoptimizer_{pid}`), topics are created per run, and partitions are `--type
# memory`, so no consumer ever resumes anything. The ack was carrying no information and
# breaking the control plane.
#
# Set LIVEFLOW_MOFKA_ACK=1 to restore the old behaviour (for A/B against a fixed stack).
def ack_event(event) -> None:
    if os.environ.get("LIVEFLOW_MOFKA_ACK", "0") == "1":
        event.acknowledge()
