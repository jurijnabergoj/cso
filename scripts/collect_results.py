"""
Print a summary table of all experiment results.

Usage: python scripts/collect_results.py
"""

import json
from pathlib import Path

OUTPUTS_DIR = Path("/d/hpc/home/jn16867/cso/outputs")

COLS = ["MAE", "NAE", "SRE", "sMAPE", "R2", "best_val_mae"]

results = []
for results_file in sorted(OUTPUTS_DIR.glob("*/*/*_results.json")):
    with open(results_file) as f:
        data = json.load(f)
    group = results_file.parts[-3]
    run = results_file.stem.replace("_results", "")
    row = {"experiment": f"{group}/{run}"}
    metrics = data.get("metrics", {})
    for col in COLS:
        if col in metrics:
            row[col] = metrics[col]
        elif col in data:
            row[col] = data[col]
        else:
            row[col] = float("nan")
    results.append(row)

if not results:
    print("No results found.")
else:
    col_w = max(len(r["experiment"]) for r in results) + 2
    header = f"{'experiment':<{col_w}}" + "".join(f"{c:>12}" for c in COLS)
    print(header)
    print("-" * len(header))
    for row in sorted(results, key=lambda r: r.get("best_val_mae", float("inf"))):
        line = f"{row['experiment']:<{col_w}}"
        for col in COLS:
            v = row[col]
            line += f"{v:>12.4f}" if v == v else f"{'N/A':>12}"
        print(line)
