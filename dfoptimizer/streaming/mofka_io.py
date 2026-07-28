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
            _driver_singleton = mofka.MofkaDriver(
                group_file=group_file,
                use_progress_thread=use_progress_thread,
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
