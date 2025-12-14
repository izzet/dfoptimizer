"""data_prep.py
Load run logs (JSONL or CSV) and build an epoch-level dataframe.

Expected per-epoch log record schema (JSONL):
{
  "run_id": "...",
  "framework": "pytorch|tensorflow",
  "scenario": "variability",
  "variant": "baseline|autotune|ea",
  "epoch": 0,
  "config": {...},
  "metrics": {...}
}

Outputs:
- Parquet or CSV with flattened columns: config_* and metric_*
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def flatten_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: rec.get(k) for k in ["run_id", "framework", "scenario", "variant", "epoch"]}
    cfg = rec.get("config", {}) or {}
    met = rec.get("metrics", {}) or {}
    for k, v in cfg.items():
        out[f"config_{k}"] = v
    for k, v in met.items():
        out[f"metric_{k}"] = v
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input folder containing *.jsonl run logs")
    ap.add_argument("--out", dest="out", required=True, help="Output file (.parquet or .csv)")
    args = ap.parse_args()

    inp = Path(args.inp)
    paths = sorted(list(inp.rglob("*.jsonl"))) + sorted(list(inp.rglob("*.csv")))

    if not paths:
        raise SystemExit(f"No logs found under {inp} (expected *.jsonl or *.csv).")

    all_rows: List[Dict[str, Any]] = []
    for p in paths:
        if p.suffix == ".jsonl":
            rows = load_jsonl(p)
            all_rows.extend([flatten_record(r) for r in rows])
        elif p.suffix == ".csv":
            df = pd.read_csv(p)
            all_rows.extend(df.to_dict(orient="records"))

    df = pd.DataFrame(all_rows)

    # Basic ordering
    if "epoch" in df.columns:
        df = df.sort_values(["run_id", "framework", "scenario", "variant", "epoch"], kind="stable")

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    if outp.suffix == ".parquet":
        df.to_parquet(outp, index=False)
    else:
        df.to_csv(outp, index=False)

    print(f"Wrote {len(df)} rows to {outp}")

if __name__ == "__main__":
    main()
