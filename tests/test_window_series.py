"""Window target parameter (LIVEFLOW_WINDOW_TARGET_S) and the per-step series / node telemetry on window.stop."""
import pytest

from dfoptimizer import step_series
from dfoptimizer.runtime import window as window_mod
from dfoptimizer.runtime.window import Window, freeze_window_steps, window_target_seconds


class FakeDft:
    def __init__(self):
        self.t, self.events = 0, []

    def get_time(self):
        self.t += 1000
        return self.t

    def enter_event(self):
        pass

    def exit_event(self):
        pass

    def log_event(self, **kw):
        self.events.append(kw)


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("LIVEFLOW_PRESET", "liveflow-v1")
    monkeypatch.delenv("LIVEFLOW_WINDOW_TARGET_S", raising=False)
    monkeypatch.setattr(Window, "_agreed_step_time", staticmethod(lambda s: s))
    monkeypatch.setattr(window_mod, "read_node_telemetry", lambda: {"vm_allocstall": 7, "vm_pgscan_direct": 11,
                                                                    "vm_compact_stall": 2, "mem_anon_kb": 1234,
                                                                    "mem_avail_kb": 99999})
    return monkeypatch


def test_window_target_rule_for_t10_t30_t60(env):
    assert window_target_seconds() == 30.0                       # unchanged default
    assert freeze_window_steps(0.092) == 327                     # ceil(30 / 0.092)
    assert freeze_window_steps(0.092, 10.0) == 109
    assert freeze_window_steps(0.092, 60.0) == 653
    assert freeze_window_steps(5.0, 10.0) == 20                  # floor of 20 steps
    env.setenv("LIVEFLOW_WINDOW_TARGET_S", "10")
    assert window_target_seconds() == 10.0 and freeze_window_steps(0.092) == 109
    for bad in ("0", "-5", "abc", "inf", "nan"):
        env.setenv("LIVEFLOW_WINDOW_TARGET_S", bad)
        with pytest.raises(ValueError):
            window_target_seconds()


def _run(env, target, steps, step_s=0.1):
    clock = {"t": 0.0}
    env.setattr(window_mod.time, "perf_counter", lambda: clock["t"])
    if target is not None:
        env.setenv("LIVEFLOW_WINDOW_TARGET_S", str(target))
    dft = FakeDft()
    w = Window(dft)
    w.set_horizon(2, steps)
    w.set_epoch(1)
    for i in range(steps):
        w.start()
        clock["t"] += step_s
        w.note_step_exposure(storage=0.01 * (i % 3), comm=0.02)
        w.stop()
    return w, dft


def test_cadence_freezes_from_the_target(env):
    w30, _ = _run(env, None, 60)
    assert w30.every_n == 300
    w10, _ = _run(env, 10, 60)
    assert w10.every_n == 100
    w60, _ = _run(env, 60, 60)
    assert w60.every_n == 600


def test_window_stop_carries_the_per_step_series_and_telemetry(env):
    w, dft = _run(env, None, 40)                                  # two 20-step windows
    stops = [e for e in dft.events if e["name"] == "window.stop"]
    assert len(stops) == 2
    ev = stops[1]
    ss = step_series.decode(ev["string_args"]["step_series"][1])
    assert (ss.first, ss.n, ss.bin) == (21, 20, 1)
    assert ss.series["wall"] == pytest.approx([0.1] * 20, abs=1e-5)
    assert ss.series["comm"] == pytest.approx([0.02] * 20, abs=1e-5)
    assert ss.series["storage"][:3] == pytest.approx([0.02, 0.0, 0.01], abs=1e-5)   # steps 21.. have i % 3 = 2, 0, 1
    assert ev["int_args"]["vm_allocstall"][1] == 7 and ev["int_args"]["mem_anon_kb"][1] == 1234
    assert ev["int_args"]["step_wall_p90_us"][1] == 100000


def test_a_long_window_is_binned_and_telemetry_can_be_disabled(env):
    env.setenv("LIVEFLOW_NODE_TELEMETRY", "0")
    w, dft = _run(env, 1000, 40)
    stops = [e for e in dft.events if e["name"] == "window.stop"]
    assert "vm_allocstall" not in stops[0]["int_args"]
    text = step_series.encode(1, {"wall": [0.1] * 2000})
    ss = step_series.decode(text)
    assert ss.bin == 4 and len(ss.weights) == 500 and ss.n == 2000


def test_series_off_outside_liveflow_v1(env):
    env.delenv("LIVEFLOW_PRESET")
    clock = {"t": 0.0}
    env.setattr(window_mod.time, "perf_counter", lambda: clock["t"])
    dft = FakeDft()
    w = Window(dft)
    for _ in range(3):
        w.start()
        w.note_step_exposure(storage=0.5)
        w.stop()
    assert all("string_args" not in e or not e["string_args"] for e in dft.events if e["name"] == "window.stop")
