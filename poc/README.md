# DL Autotune PoC (PyTorch + TensorFlow)

This folder contains a small, repeatable PoC for:
- generating per-epoch telemetry (data + compute + IO-ish signals),
- validating/aggregating that telemetry into a dataframe,
- running a simple evolutionary optimizer between epochs,
- comparing against baselines (and tf.data AUTOTUNE when available).

## Folder layout
- `data_prep.py`   : load run logs -> dataframe -> save (csv/parquet)
- `data_valid.py`  : validate dataframe + produce quick analysis report
- `model.py`       : evolutionary optimizer + knob/metric specs + fitness
- `eval_pytorch_variability.py` : PyTorch scenario runner (baseline vs delayed vs EA)
- `eval_tensorflow_variability.py` : TF scenario runner (baseline vs AUTOTUNE vs EA)
- `metrics_spec.json` : example metric weights/constraints
- `knobs_pytorch.json` : example knobs for PyTorch eval
- `knobs_tensorflow.json` : example knobs for TF eval

## Quick start
### 1) Run PyTorch scenario (creates logs under `runs/`)
```bash
python eval_pytorch_variability.py --out runs
```

### 2) (Optional) Run TensorFlow scenario (if TensorFlow installed)
```bash
python eval_tensorflow_variability.py --out runs
```

### 3) Build a dataframe from logs
```bash
python data_prep.py --in runs --out artifacts/epoch_stats.parquet
```

### 4) Validate + analyze
```bash
python data_valid.py --in artifacts/epoch_stats.parquet --out artifacts/report.md
```

## Notes
- This PoC uses **synthetic compute** (sleep) and **synthetic IO-ish work** (reading blocks from a local file) so it’s repeatable on any machine.
- Real systems would plug in actual GPU counters and OS IO stats (e.g. via psutil + NVML + eBPF).
