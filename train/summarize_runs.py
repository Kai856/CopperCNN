"""Print validation/test metrics for several runs, plus hole depth alone as a reference score.

Usage:
  python -m train.summarize_runs --data-dir /cluster/scratch/$USER/copper_dataset64 \
      --runs-dir /cluster/scratch/$USER/copper_runs base32 wide64 wide64_rock wide64_rock_depth
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def main():
    user = os.environ.get("USER", "user")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=f"/cluster/scratch/{user}/copper_dataset64")
    p.add_argument("--runs-dir", default=f"/cluster/scratch/{user}/copper_runs")
    p.add_argument("runs", nargs="+")
    args = p.parse_args()

    rows = []
    for run in args.runs:
        path = os.path.join(args.runs_dir, run, "metrics.json")
        if not os.path.exists(path):
            print(f"missing: {path}")
            continue
        with open(path) as f:
            m = json.load(f)
        rows.append({"run": run, "best_epoch": m["best_epoch"],
                     **{f"{s}_{k}": m[s][k] for s in ("val", "test") for k in ("roc_auc_bin", "pr_auc_bin", "roc_auc_log")}})

    cells = pd.read_csv(os.path.join(args.data_dir, "fold_cells.csv"))
    with open(os.path.join(args.data_dir, "norm_stats.json")) as f:
        stats = json.load(f)
    if "depth_m" in cells:
        ref = {"run": "depth_only", "best_epoch": np.nan}
        for s, fold in (("val", stats["val_fold"]), ("test", stats["test_fold"])):
            c = cells[(cells["fold"] == fold) & cells["depth_m"].notna()]
            ref[f"{s}_roc_auc_bin"] = roc_auc_score(c["y_bin"], c["depth_m"])
            ref[f"{s}_pr_auc_bin"] = average_precision_score(c["y_bin"], c["depth_m"])
            ref[f"{s}_roc_auc_log"] = np.nan
        rows.append(ref)

    table = pd.DataFrame(rows).set_index("run")
    pos = {s: cells.loc[cells["fold"] == stats[f"{s}_fold"], "y_bin"].mean() for s in ("val", "test")}
    print(f"Random PR-AUC: val {pos['val']:.3f}, test {pos['test']:.3f}; random ROC-AUC 0.5")
    print(table.to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
