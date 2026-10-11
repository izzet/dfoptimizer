"""Wire codec for the per-step series a LiveFlow window carries (window.stop string arg `step_series`).

The window already times every step (`Window._window_step_walls`, the source of `step_wall_p90_us`).
This module is the one definition of how those per-step arrays, and the per-step exposure of each
family the application measures, travel with the window record to the analyzer, the diagnoser and
the planner.  The planner verifies a move on them (planner/step_verify.py); nothing else reads them.

Format (ASCII, one string):

    v1;first=<completed-step index of the first step>;n=<steps>;b=<steps per bin>;
       wall=<csv>;input=<csv>;storage=<csv>;decode=<csv>;delivery=<csv>;comm=<csv>;job_input=<csv>

Values are integers in units of 10 microseconds.  `wall` is the critical rank's step time; each other
key is that rank's per-step exposed wait (`input` is combined parent fetch wait, never pure storage
or decode; `job_input` is the job-wide per-step maximum of
the input wait over all ranks, which every rank holds after the window's Allreduce(MAX)).  A window of
more than MAX_BINS steps is carried as bin means of b = ceil(n / MAX_BINS) consecutive steps: the ring
that carries window events is small, and a mean over consecutive steps loses nothing the verification
uses (it compares means).  MAX_BINS is a transport size cap, not a statistical choice.
"""
import math
from typing import Dict, Optional, Sequence

try:  # the planner and the analyzer have numpy; the application side encodes without it
    import numpy as np
except Exception:  # pragma: no cover
    np = None

UNIT_S = 1e-5
MAX_BINS = 512
VERSION = "v1"
FAMILIES = ("input", "storage", "decode", "delivery", "comm", "job_input")


def _bin_means(values: Sequence[float], b: int):
    if b <= 1:
        return [float(v) for v in values]
    out = []
    for i in range(0, len(values), b):
        chunk = values[i:i + b]
        out.append(sum(chunk) / len(chunk))
    return out


def encode(first_step: int, series: Dict[str, Sequence[float]], max_bins: int = MAX_BINS) -> Optional[str]:
    """Encode per-step series (seconds). `series` must contain `wall`; returns None when it cannot."""
    wall = series.get("wall")
    if wall is None or len(wall) == 0:
        return None
    n = len(wall)
    b = max(1, int(math.ceil(n / float(max_bins))))
    parts = [VERSION, "first=%d" % int(first_step), "n=%d" % n, "b=%d" % b]
    for key in ("wall",) + tuple(k for k in series if k != "wall"):
        vals = series[key]
        if vals is None or len(vals) != n:
            continue
        ints = [max(int(round(v / UNIT_S)), 0) for v in _bin_means(vals, b)]
        parts.append("%s=%s" % (key, ",".join(str(i) for i in ints)))
    return ";".join(parts)


class StepSeries:
    """Decoded series. `weights[i]` is the number of steps bin i covers; arrays are in seconds."""

    __slots__ = ("first", "n", "bin", "weights", "series", "window")

    def __init__(self, first, n, b, weights, series, window=None):
        self.first, self.n, self.bin, self.weights, self.series = first, n, b, weights, series
        self.window = window

    @property
    def last(self) -> int:
        return self.first + self.n - 1

    def bin_starts(self):
        return self.first + np.arange(len(self.weights)) * self.bin


def decode(text) -> Optional[StepSeries]:
    """Parse a `step_series` string; None for anything that is not a well-formed v1 record."""
    if not isinstance(text, str) or not text.startswith(VERSION + ";") or np is None:
        return None
    fields = {}
    for part in text.split(";")[1:]:
        key, sep, value = part.partition("=")
        if sep:
            fields[key] = value
    try:
        first, n, b = int(fields.pop("first")), int(fields.pop("n")), int(fields.pop("b"))
    except (KeyError, ValueError):
        return None
    if n < 1 or b < 1:
        return None
    nbins = int(math.ceil(n / float(b)))
    series = {}
    for key, value in fields.items():
        try:
            arr = np.array([int(x) for x in value.split(",")], dtype=float) * UNIT_S
        except ValueError:
            return None
        if len(arr) != nbins:
            return None
        series[key] = arr
    if "wall" not in series:
        return None
    weights = np.full(nbins, float(b))
    if n % b:
        weights[-1] = float(n - b * (nbins - 1))
    return StepSeries(first, n, b, weights, series)
