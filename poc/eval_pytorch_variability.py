"""eval_pytorch_variability.py
Scenario: variability/straggler in the input pipeline.

We run 3 variants:
1) baseline: no injected delay, fixed knobs
2) delayed : inject extra delay for worker 0 (head-of-line / straggler proxy), fixed knobs
3) ea      : same delay, but tune knobs between epochs with EvolutionaryOptimizer

Telemetry recorded per epoch:
- epoch_time_sec
- samples_per_sec
- data_wait_time_sec (time waiting for next batch)
- compute_time_sec (simulated)
- gpu_idle_frac (proxy = data_wait / (data_wait + compute))
- posix_read_bw_mb_s (bytes/time inside __getitem__)

Notes:
- Compute is simulated via time.sleep() for repeatability (acts like GPU compute time).
- IO is simulated by reading blocks from a local file via os.pread.
"""

from __future__ import annotations

import argparse
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, DataLoader, get_worker_info
except Exception as e:
    raise SystemExit("PyTorch is required for this script. Install torch first.") from e

from model import EvolutionaryOptimizer, load_knob_defs, load_fitness_spec
from utils import JsonlLogger

@dataclass
class ScenarioCfg:
    n_epochs: int = 8
    n_samples: int = 20000
    batch_size: int = 256
    compute_ms_per_batch: float = 8.0
    base_io_bytes: int = 64 * 1024
    variable_io: bool = True
    slow_worker_id: int = 0
    slow_extra_ms: float = 2.0

class SyntheticIODataset(Dataset):
    def __init__(self, data_path: str, cfg: ScenarioCfg, inject_delay: bool):
        self.data_path = data_path
        self.cfg = cfg
        self.inject_delay = inject_delay
        self.fd = os.open(self.data_path, os.O_RDONLY)

    def __len__(self):
        return self.cfg.n_samples

    def __getitem__(self, idx: int):
        wi = get_worker_info()
        wid = wi.id if wi is not None else 0

        # Inject deterministic straggler: one worker always slower
        if self.inject_delay and wid == self.cfg.slow_worker_id:
            time.sleep(self.cfg.slow_extra_ms / 1000.0)

        # Variable-length IO: occasionally read larger blocks
        nbytes = self.cfg.base_io_bytes
        if self.cfg.variable_io and (idx % 50 == 0):
            nbytes = self.cfg.base_io_bytes * 8

        # Read from file (pread: doesn't move a shared file offset across workers)
        offset = (idx * self.cfg.base_io_bytes) % (128 * 1024 * 1024)
        t0 = time.perf_counter()
        _ = os.pread(self.fd, nbytes, offset)
        read_time = time.perf_counter() - t0
        # Return a tiny tensor payload + IO stats (bytes, read_time)
        x = torch.tensor([idx % 1024], dtype=torch.int64)
        return x, nbytes, read_time

    def __del__(self):
        try:
            os.close(self.fd)
        except Exception:
            pass

def ensure_data_file(path: Path, size_mb: int = 256) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size >= size_mb * 1024 * 1024:
        return
    # Create a deterministic file (zeros) once
    with path.open("wb") as f:
        f.truncate(size_mb * 1024 * 1024)

def run_epoch(dl: DataLoader, cfg: ScenarioCfg) -> Dict[str, float]:
    t_epoch0 = time.perf_counter()
    data_wait = 0.0
    compute_time = 0.0
    bytes_read = 0
    read_time = 0.0
    samples = 0

    it = iter(dl)
    while True:
        t0 = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            break
        data_wait += (time.perf_counter() - t0)

        x, nbytes, rtime = batch
        # batch is collated: nbytes and rtime are tensors
        bytes_read += int(nbytes.sum().item())
        read_time += float(rtime.sum().item())
        samples += int(x.shape[0])

        # Simulate compute
        t1 = time.perf_counter()
        time.sleep(cfg.compute_ms_per_batch / 1000.0)
        compute_time += (time.perf_counter() - t1)

    epoch_time = time.perf_counter() - t_epoch0
    samples_per_sec = samples / max(epoch_time, 1e-9)
    io_bw = (bytes_read / (1024 * 1024)) / max(read_time, 1e-9)  # MB/s
    idle_frac = data_wait / max(data_wait + compute_time, 1e-9)

    return {
        "epoch_time_sec": epoch_time,
        "samples_per_sec": samples_per_sec,
        "data_wait_time_sec": data_wait,
        "compute_time_sec": compute_time,
        "gpu_idle_frac": idle_frac,
        "posix_read_bw_mb_s": io_bw,
        "oom_count": 0,
        "error_count": 0,
    }

def make_dataloader(ds: Dataset, batch_size: int, config: Dict[str, Any]) -> DataLoader:
    # For num_workers=0, prefetch_factor/persistent_workers are invalid; handle gracefully
    num_workers = int(config.get("num_workers", 0))
    kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "drop_last": True,
        "num_workers": num_workers,
        "pin_memory": bool(config.get("pin_memory", False)),
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = int(config.get("prefetch_factor", 2))
        kwargs["persistent_workers"] = bool(config.get("persistent_workers", False))
    return DataLoader(ds, **kwargs)

def run_variant(
    framework: str,
    scenario: str,
    variant: str,
    cfg: ScenarioCfg,
    out_dir: Path,
    data_file: Path,
    knob_defs_path: str,
    fitness_spec_path: str,
    fixed_config: Optional[Dict[str, Any]] = None,
    use_ea: bool = False,
    inject_delay: bool = False,
):
    run_id = str(uuid.uuid4())[:8]
    logger = JsonlLogger(str(out_dir), f"{framework}_{scenario}_{variant}_{run_id}.jsonl")

    knob_defs = load_knob_defs(knob_defs_path)
    fitness_spec = load_fitness_spec(fitness_spec_path)
    ea = EvolutionaryOptimizer(knob_defs, fitness_spec, population_size=16, seed=0) if use_ea else None

    config = fixed_config or {k: kd.values[0] for k, kd in knob_defs.items()}

    for epoch in range(cfg.n_epochs):
        if ea is not None:
            config = ea.ask()

        ds = SyntheticIODataset(str(data_file), cfg, inject_delay=inject_delay)
        dl = make_dataloader(ds, cfg.batch_size, config)

        metrics = run_epoch(dl, cfg)
        meta = {"epoch": epoch}

        if ea is not None:
            _ = ea.tell(config, metrics, meta=meta)
            best = ea.best()
        else:
            best = None

        rec = {
            "run_id": run_id,
            "framework": framework,
            "scenario": scenario,
            "variant": variant,
            "epoch": epoch,
            "config": config,
            "metrics": metrics,
            "best_so_far": (best.config if best else None),
            "best_fitness": (best.fitness if best else None),
        }
        logger.log(rec)
        print(f"[{variant}] epoch={epoch} config={config} samples/s={metrics['samples_per_sec']:.1f} idle={metrics['gpu_idle_frac']:.3f}")

    logger.close()
    return run_id

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output folder for run logs")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--slow-extra-ms", type=float, default=2.0)
    ap.add_argument("--compute-ms", type=float, default=8.0)
    ap.add_argument("--data-file-mb", type=int, default=256)
    ap.add_argument("--knobs", default="knobs_pytorch.json")
    ap.add_argument("--metrics", default="metrics_spec.json")
    args = ap.parse_args()

    cfg = ScenarioCfg(
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        slow_extra_ms=args.slow_extra_ms,
        compute_ms_per_batch=args.compute_ms,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_file = out_dir / "synthetic_data.bin"
    ensure_data_file(data_file, size_mb=args.data_file_mb)

    # Baseline fixed config (reasonable default)
    fixed = {"num_workers": 2, "prefetch_factor": 4, "pin_memory": False, "persistent_workers": True}

    run_variant("pytorch", "variability", "baseline", cfg, out_dir, data_file, args.knobs, args.metrics,
                fixed_config=fixed, use_ea=False, inject_delay=False)

    run_variant("pytorch", "variability", "delayed", cfg, out_dir, data_file, args.knobs, args.metrics,
                fixed_config=fixed, use_ea=False, inject_delay=True)

    run_variant("pytorch", "variability", "ea", cfg, out_dir, data_file, args.knobs, args.metrics,
                fixed_config=fixed, use_ea=True, inject_delay=True)

if __name__ == "__main__":
    main()
