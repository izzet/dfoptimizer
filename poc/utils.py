"""utils.py
Small utilities for logging and timing.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

class JsonlLogger:
    def __init__(self, out_dir: str, filename: str):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.out_dir / filename
        self.f = self.path.open("w", encoding="utf-8")

    def log(self, rec: Dict[str, Any]) -> None:
        self.f.write(json.dumps(rec) + "\n")
        self.f.flush()

    def close(self) -> None:
        try:
            self.f.close()
        except Exception:
            pass

class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()

    def reset(self):
        self.t0 = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0
