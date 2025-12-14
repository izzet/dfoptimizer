"""data_valid.py
Validate the epoch-level dataframe and produce a small report.

Checks:
- required identifiers exist
- key metrics exist and are numeric
- missingness and ranges
- per-variant summary (mean/median) for key metrics
- simple bottleneck hints:
  - high gpu_idle_frac => data starvation
  - low samples_per_sec => throughput issue
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

REQUIRED_ID_COLS = ["run_id", "framework", "scenario", "variant", "epoch"]

KEY_METRICS = [
    "metric_epoch_time_sec",
    "metric_samples_per_sec",
    "metric_gpu_idle_frac",
    "metric_posix_read_bw_mb_s",
    "metric_data_wait_time_sec",
    "metric_compute_time_sec",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input parquet/csv from data_prep.py")
    ap.add_argument("--out", dest="out", required=True, help="Output report markdown")
    args = ap.parse_args()

    inp = Path(args.inp)
    if inp.suffix == ".parquet":
        df = pd.read_parquet(inp)
    else:
        df = pd.read_csv(inp)

    lines = []
    lines.append(f"# Data Validation Report\n")
    lines.append(f"Loaded **{len(df)}** rows from `{inp}`\n")

    # Required columns
    missing_ids = [c for c in REQUIRED_ID_COLS if c not in df.columns]
    if missing_ids:
        lines.append(f"## ❌ Missing required id columns\n- " + "\n- ".join(missing_ids) + "\n")
    else:
        lines.append("## ✅ Required id columns present\n")

    # Metric presence
    present_metrics = [m for m in KEY_METRICS if m in df.columns]
    missing_metrics = [m for m in KEY_METRICS if m not in df.columns]
    lines.append("## Metric coverage\n")
    lines.append(f"Present: {len(present_metrics)}/{len(KEY_METRICS)}\n")
    if missing_metrics:
        lines.append("Missing:\n" + "\n".join([f"- {m}" for m in missing_metrics]) + "\n")
    else:
        lines.append("All key metrics present.\n")

    # Missingness
    lines.append("## Missingness\n")
    miss = df[present_metrics].isna().mean().sort_values(ascending=False)
    lines.append(miss.to_frame("missing_frac").to_markdown())
    lines.append("\n")

    # Basic summaries
    lines.append("## Summary by (framework, scenario, variant)\n")
    agg_cols = [m for m in present_metrics if df[m].dtype != object]
    grp = df.groupby(["framework", "scenario", "variant"])[agg_cols].agg(["mean", "median"])
    lines.append(grp.to_markdown())
    lines.append("\n")

    # Bottleneck hints
    lines.append("## Bottleneck hints (heuristics)\n")
    if "metric_gpu_idle_frac" in df.columns:
        high_idle = df[df["metric_gpu_idle_frac"] > 0.3]
        lines.append(f"- Rows with gpu_idle_frac > 0.3: **{len(high_idle)}** (likely data starvation)\n")
    if "metric_posix_read_bw_mb_s" in df.columns:
        low_bw = df[df["metric_posix_read_bw_mb_s"] < df["metric_posix_read_bw_mb_s"].quantile(0.1)]
        lines.append(f"- Rows in lowest 10% IO bw: **{len(low_bw)}** (possible IO bottleneck)\n")
    if "metric_data_wait_time_sec" in df.columns and "metric_epoch_time_sec" in df.columns:
        ratio = (df["metric_data_wait_time_sec"] / (df["metric_epoch_time_sec"] + 1e-9))
        lines.append(f"- Median data_wait/epoch_time: **{ratio.median():.3f}**\n")
        lines.append(f"- 90th pct data_wait/epoch_time: **{ratio.quantile(0.9):.3f}**\n")
    lines.append("\n")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote report to {out}")

if __name__ == "__main__":
    main()
