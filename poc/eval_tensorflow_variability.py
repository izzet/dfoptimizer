"""eval_tensorflow_variability.py
TensorFlow version of the variability/straggler input pipeline scenario.

Variants:
1) baseline: fixed map parallelism, interleave, prefetch
2) autotune: use tf.data.AUTOTUNE where applicable
3) ea      : tune knobs between epochs using EvolutionaryOptimizer

Notes:
- Uses tf.py_function to insert a sleep for selected elements to simulate slow preprocessing.
- Uses a file dataset + interleave to better represent IO-ish pipelines.

This script gracefully exits if TensorFlow is not installed.
"""

from __future__ import annotations

import argparse
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

try:
    import tensorflow as tf
except Exception:
    raise SystemExit("TensorFlow is required for this script. Install tensorflow first.")

from model import EvolutionaryOptimizer, load_knob_defs, load_fitness_spec
from utils import JsonlLogger

def ensure_shards(out_dir: Path, n_shards: int = 8, shard_size_mb: int = 32) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_shards):
        p = out_dir / f"shard_{i:02d}.bin"
        if p.exists() and p.stat().st_size >= shard_size_mb * 1024 * 1024:
            continue
        with p.open("wb") as f:
            f.truncate(shard_size_mb * 1024 * 1024)

def slow_map_py(idx, x, slow_every=50, extra_ms=2.0):
    # deterministic slow element: every Nth element is slower
    i = int(idx)
    if i % slow_every == 0:
        time.sleep(extra_ms / 1000.0)
    return x

def build_dataset(
    shard_paths,
    batch_size: int,
    config: Dict[str, Any],
    inject_delay: bool,
    use_autotune: bool,
):
    cycle_length = int(config.get("interleave_cycle_length", 2))
    map_parallel = int(config.get("map_parallel_calls", 2))
    prefetch_buf = int(config.get("prefetch_buffer", 2))

    if use_autotune:
        cycle_length = tf.data.AUTOTUNE
        map_parallel = tf.data.AUTOTUNE
        prefetch_buf = tf.data.AUTOTUNE

    ds_files = tf.data.Dataset.from_tensor_slices(shard_paths)

    # Interleave to simulate reading from multiple files
    ds = ds_files.interleave(
        lambda p: tf.data.FixedLengthRecordDataset(p, record_bytes=1024),
        cycle_length=cycle_length,
        num_parallel_calls=tf.data.AUTOTUNE if use_autotune else None,
        deterministic=False,
    )

    # Add an index to create deterministic "slow elements"
    ds = ds.enumerate()

    def map_fn(idx, rec):
        # Convert bytes -> int just to do a tiny parse
        x = tf.strings.length(rec)
        if inject_delay:
            y = tf.py_function(func=slow_map_py, inp=[idx, x], Tout=tf.int32)
            y.set_shape(())
            return y
        return x

    ds = ds.map(map_fn, num_parallel_calls=map_parallel, deterministic=False)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(prefetch_buf)
    return ds

def run_epoch(ds, compute_ms_per_batch: float):
    t0 = time.perf_counter()
    data_wait = 0.0
    compute_time = 0.0
    batches = 0
    samples = 0

    it = iter(ds)
    while True:
        tw0 = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            break
        data_wait += (time.perf_counter() - tw0)

        # Simulate compute
        tc0 = time.perf_counter()
        time.sleep(compute_ms_per_batch / 1000.0)
        compute_time += (time.perf_counter() - tc0)

        batches += 1
        samples += int(batch.shape[0])

        # Stop after some batches for repeatability
        if batches >= 200:
            break

    epoch_time = time.perf_counter() - t0
    sps = samples / max(epoch_time, 1e-9)
    idle = data_wait / max(data_wait + compute_time, 1e-9)

    return {
        "epoch_time_sec": epoch_time,
        "samples_per_sec": sps,
        "data_wait_time_sec": data_wait,
        "compute_time_sec": compute_time,
        "gpu_idle_frac": idle,
        # IO bw is not directly measured here; leave None or 0
        "posix_read_bw_mb_s": None,
        "oom_count": 0,
        "error_count": 0,
    }

def run_variant(
    scenario: str,
    variant: str,
    out_dir: Path,
    shards_dir: Path,
    knob_defs_path: str,
    fitness_spec_path: str,
    n_epochs: int = 8,
    batch_size: int = 256,
    compute_ms: float = 8.0,
    inject_delay: bool = False,
    use_autotune: bool = False,
    use_ea: bool = False,
    fixed_config: Optional[Dict[str, Any]] = None,
):
    run_id = str(uuid.uuid4())[:8]
    logger = JsonlLogger(str(out_dir), f"tensorflow_{scenario}_{variant}_{run_id}.jsonl")

    shard_paths = [str(p) for p in sorted(shards_dir.glob("shard_*.bin"))]
    if not shard_paths:
        raise RuntimeError("No shard files found; call ensure_shards first.")

    knob_defs = load_knob_defs(knob_defs_path)
    fitness_spec = load_fitness_spec(fitness_spec_path)
    ea = EvolutionaryOptimizer(knob_defs, fitness_spec, population_size=16, seed=0) if use_ea else None
    config = fixed_config or {k: kd.values[0] for k, kd in knob_defs.items()}

    for epoch in range(n_epochs):
        if ea is not None:
            config = ea.ask()

        ds = build_dataset(shard_paths, batch_size, config, inject_delay=inject_delay, use_autotune=use_autotune)
        metrics = run_epoch(ds, compute_ms_per_batch=compute_ms)

        if ea is not None:
            _ = ea.tell(config, metrics, meta={"epoch": epoch})
            best = ea.best()
        else:
            best = None

        rec = {
            "run_id": run_id,
            "framework": "tensorflow",
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
    ap.add_argument("--knobs", default="knobs_tensorflow.json")
    ap.add_argument("--metrics", default="metrics_spec.json")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = out_dir / "tf_shards"
    ensure_shards(shards_dir, n_shards=8, shard_size_mb=32)

    fixed = {"map_parallel_calls": 2, "prefetch_buffer": 2, "interleave_cycle_length": 2}

    run_variant("variability", "baseline", out_dir, shards_dir, args.knobs, args.metrics,
                n_epochs=args.epochs, batch_size=args.batch_size, compute_ms=args.compute_ms,
                inject_delay=False, use_autotune=False, use_ea=False, fixed_config=fixed)

    run_variant("variability", "autotune", out_dir, shards_dir, args.knobs, args.metrics,
                n_epochs=args.epochs, batch_size=args.batch_size, compute_ms=args.compute_ms,
                inject_delay=True, use_autotune=True, use_ea=False, fixed_config=fixed)

    run_variant("variability", "ea", out_dir, shards_dir, args.knobs, args.metrics,
                n_epochs=args.epochs, batch_size=args.batch_size, compute_ms=args.compute_ms,
                inject_delay=True, use_autotune=False, use_ea=True, fixed_config=fixed)

if __name__ == "__main__":
    main()
