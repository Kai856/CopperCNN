"""Random-forest baseline on each cell's own values (no spatial context).

Features: gravity, TMI and fault distance at the drillhole cell, optionally the
4 rock codes (--rock) and hole depth (--depth). Same folds and buffer as train.py.
Writes metrics.json in the same format so summarize_runs.py can compare runs.

Usage (from the repo root):
  python -m train.random_forest --run-name rf_rock --rock
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score


def main():
    user = os.environ.get("USER", "user")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=f"/cluster/scratch/{user}/copper_dataset64")
    p.add_argument("--out-root", default=f"/cluster/scratch/{user}/copper_runs")
    p.add_argument("--run-name", default="rf")
    p.add_argument("--rock", action="store_true")
    p.add_argument("--depth", action="store_true")
    p.add_argument("--buffer-km", type=float, default=6.4, help="same as the 64x64 CNN runs")
    p.add_argument("--trees", type=int, default=500)
    p.add_argument("--min-leaf", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    out_dir = os.path.join(args.out_root, args.run_name)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(args.data_dir, "norm_stats.json")) as f:
        stats = json.load(f)
    cells = pd.read_csv(os.path.join(args.data_dir, "fold_cells.csv"))
    tiles = np.load(os.path.join(args.data_dir, "tiles.npy"), mmap_mode="r")
    c = tiles.shape[-1] // 2
    feats = {"gravity": tiles[:, 0, c, c], "tmi": tiles[:, 1, c, c], "fault_dist_km": tiles[:, 2, c, c]}
    if args.rock:
        with open(os.path.join(args.data_dir, "rock_layers.json")) as f:
            layers = json.load(f)["layers"]
        rock = np.load(os.path.join(args.data_dir, "rock.npy"), mmap_mode="r")
        for i, name in enumerate(layers):
            feats[name.replace(".tif", "")] = rock[:, i, c, c]
    if args.depth:
        feats["log10_depth"] = np.log10(np.clip(cells["depth_m"].values, 1, None))
    X = pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in feats.items()})
    X = X.fillna(-999).values

    folds = cells["fold"].values
    val_idx = np.flatnonzero(folds == stats["val_fold"])
    test_idx = np.flatnonzero(folds == stats["test_fold"])
    train_idx = np.flatnonzero(np.isin(folds, stats["train_folds"]))
    xy = cells[["x", "y"]].values
    near = cKDTree(xy[np.concatenate([val_idx, test_idx])]).query(xy[train_idx], distance_upper_bound=args.buffer_km * 1000)[0]
    train_idx = train_idx[~np.isfinite(near)]
    y = cells["y_bin"].values
    print(f"features {list(feats)}; train {len(train_idx)}, val {len(val_idx)}, test {len(test_idx)}")

    rf = RandomForestClassifier(n_estimators=args.trees, min_samples_leaf=args.min_leaf,
                                class_weight="balanced_subsample", n_jobs=-1, random_state=args.seed)
    rf.fit(X[train_idx], y[train_idx])

    results = {"best_epoch": None, "model": "random_forest", "features": list(feats), "args": vars(args)}
    for name, idx in [("val", val_idx), ("test", test_idx)]:
        p_bin = rf.predict_proba(X[idx])[:, 1]
        results[name] = {"roc_auc_bin": float(roc_auc_score(y[idx], p_bin)),
                         "pr_auc_bin": float(average_precision_score(y[idx], p_bin)),
                         "roc_auc_log": float("nan"), "pos_rate": float(y[idx].mean()), "n": int(len(idx))}
        print(f"  {name:4s}: ROC-AUC {results[name]['roc_auc_bin']:.3f}, PR-AUC {results[name]['pr_auc_bin']:.3f} "
              f"(random = {results[name]['pos_rate']:.3f})")
    print("  importance: " + ", ".join(f"{k} {v:.2f}" for k, v in
                                        sorted(zip(feats, rf.feature_importances_), key=lambda t: -t[1])))
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Outputs in {out_dir}")


if __name__ == "__main__":
    main()
