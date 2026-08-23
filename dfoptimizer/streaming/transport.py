"""Transport selector for the streaming pipeline.

LIVEFLOW_TRANSPORT picks the backend and DEFAULTS TO "mofka", so this file is a no-op unless a
run opts in. That default matters: it keeps every existing result reproducible with the same
code, and makes the Option C path an addition rather than a replacement.

    mofka      the existing bus (unchanged)
    liveflow   no bus at all -- shared-memory ring across the app/agent boundary, in-process
               queues between agent stages. Routing is per TOPIC, see liveflow_transport.

See LIVEFLOW-ARCHITECTURE.md sec 6 (Option C) and liveflow_shm/README.md.

`ack_event` is optional on purpose: dfdiagnoser and dfoptimizer export one from their mofka_io,
dfanalyzer defines its own privately (analyzer.py:25) and its mofka_io has none. Rather than
force all three to look alike, this falls back to a no-op, which is what acking has been on every
backend since sec 60 anyway.
"""

import os

_BACKEND = os.environ.get("LIVEFLOW_TRANSPORT", "mofka").strip().lower()

if _BACKEND == "mofka":
    from .mofka_io import open_consumer, open_producer  # noqa: F401
    try:
        from .mofka_io import ack_event  # noqa: F401
    except ImportError:
        def ack_event(event):  # noqa: D103
            pass
else:
    import sys as _sys

    _p = os.environ.get("LIVEFLOW_SHM_PYTHONPATH")
    if _p and _p not in _sys.path:
        _sys.path.insert(0, _p)
    from liveflow_transport import open_consumer, open_producer, ack_event  # noqa: F401

__all__ = ["open_consumer", "open_producer", "ack_event"]
