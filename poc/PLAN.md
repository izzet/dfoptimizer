## What’s inside

* `data_prep.py`
  Loads `runs/*.jsonl` logs and produces a flattened epoch-level dataframe (config_* and metric_* columns), saved to CSV/Parquet.

* `data_valid.py`
  Validates required columns, reports missingness, aggregates by `(framework, scenario, variant)`, and prints heuristic bottleneck hints.

* `model.py`
  A **steady-state evolutionary optimizer** with `ask()/tell()` that supports:

  * mixed knob types (int/bool/enum),
  * knobs appearing/disappearing (dynamic registry),
  * fitness computed from a **metric spec JSON** (so knobs can be arbitrary).

* `eval_pytorch_variability.py`
  Implements the “variability/straggler” scenario and logs per-epoch metrics for:

  * `baseline` (no injected delay),
  * `delayed` (one worker always slower),
  * `ea` (same delay, EA tunes DataLoader knobs per epoch).

* `eval_tensorflow_variability.py`
  Similar scenario in TF:

  * `baseline` (fixed knobs),
  * `autotune` (`tf.data.AUTOTUNE`),
  * `ea` (EA tunes knobs per epoch).
    Uses `tf.py_function` sleep to simulate slow elements.

* `metrics_spec.json`
  Defines which metrics matter (weights, directions, constraints). This avoids “hardcoding” a reward function into the optimizer.

* `knobs_pytorch.json`, `knobs_tensorflow.json`
  Example knob registries.

## How to run

```bash
# 1) PyTorch scenario (creates run logs)
python eval_pytorch_variability.py --out runs

# 2) TensorFlow scenario (if TF installed)
python eval_tensorflow_variability.py --out runs

# 3) Build epoch dataframe
python data_prep.py --in runs --out artifacts/epoch_stats.parquet

# 4) Validate + report
python data_valid.py --in artifacts/epoch_stats.parquet --out artifacts/report.md
```

## About “dynamic knob detection”

In the PoC, knobs are loaded from `knobs_*.json`, but the EA engine is written to support **knobs changing at runtime**:

* If a new knob appears, it gets added to the genome automatically (default first value).
* If a knob disappears, it gets dropped.
* No feature-dimension issues (unlike contextual bandits).

In your real system, the decorator/instrumentation layer would:

* maintain a runtime knob registry (per scope/job/process),
* push knob updates to the optimizer (`optimizer.update_knobs(...)`),
* “activate” new knobs when certain metric conditions trigger (your x,y,z).

If you want, next step I can add **two more scenario scripts** into this same structure:

1. `eval_pytorch_io_stall.py` (slow storage / small-file overhead simulation)
2. `eval_pytorch_cpu_preproc.py` (heavy CPU map/augment stage simulation)
   with the same baseline/delayed/EA structure and logging format.
