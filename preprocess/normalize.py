"""Compute per-channel normalisation statistics from the training folds.

Channels in tiles.npy: 0 gravity, 1 tmi, 2 fault_dist_km, 3 valid_mask.
  gravity: (x - mean) / std
  tmi:     clip to [p0.5, p99.5], then (x - mean) / std
  fault:   log1p(km), then (x - mean) / std
  mask:    unchanged
Pixels where the mask is 0 are set to 0 after normalisation.

Outputs: norm_stats.json next to the tiles, and figures/example_tiles.png.

Usage:
  python preprocess/normalize.py --data-dir /cluster/scratch/$USER/copper_dataset
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

CHANNEL_NAMES = ["gravity", "tmi", "fault_dist"]


def normalize_tiles(tiles, stats):
    valid = tiles[:, 3] == 1
    g = (tiles[:, 0] - stats["gravity"]["mean"]) / stats["gravity"]["std"]
    t = np.clip(tiles[:, 1], stats["tmi"]["clip_lo"], stats["tmi"]["clip_hi"])
    t = (t - stats["tmi"]["mean"]) / stats["tmi"]["std"]
    f = (np.log1p(tiles[:, 2]) - stats["fault_dist"]["mean"]) / stats["fault_dist"]["std"]
    out = np.stack([np.where(valid, g, 0), np.where(valid, t, 0), f, tiles[:, 3]], axis=1)
    return out.astype(np.float32)


def compute_stats(train_tiles):
    valid = train_tiles[:, 3] == 1
    g = train_tiles[:, 0][valid]
    t = train_tiles[:, 1][valid]
    f = np.log1p(train_tiles[:, 2]).ravel()
    lo, hi = np.percentile(t, [0.5, 99.5])
    t = np.clip(t, lo, hi)
    return {
        "gravity": {"mean": float(g.mean()), "std": float(g.std())},
        "tmi": {"clip_lo": float(lo), "clip_hi": float(hi), "mean": float(t.mean()), "std": float(t.std())},
        "fault_dist": {"transform": "log1p_km", "mean": float(f.mean()), "std": float(f.std())},
    }


def save_examples(norm, cells, train_idx, path, n=4, seed=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    labels = cells["y_bin"].values[train_idx]
    pick = np.concatenate([
        rng.choice(train_idx[labels == 1], n, replace=False),
        rng.choice(train_idx[labels == 0], n, replace=False),
    ])
    fig, axes = plt.subplots(len(pick), 3, figsize=(7, 2.1 * len(pick)))
    for r, i in enumerate(pick):
        for c, name in enumerate(CHANNEL_NAMES):
            ax = axes[r, c]
            im = ax.imshow(norm[np.searchsorted(train_idx, i), c], cmap="RdBu_r", vmin=-3, vmax=3)
            ax.set_xticks([]), ax.set_yticks([])
            if r == 0:
                ax.set_title(name)
            if c == 0:
                ax.set_ylabel(f"{'Cu>=1000' if cells['y_bin'].iat[i] else 'Cu<1000'}\n{cells['cu_ppm'].iat[i]:,.0f} ppm", fontsize=8)
    fig.colorbar(im, ax=axes, shrink=0.3, label="normalised value")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")


def main():
    user = os.environ.get("USER", "user")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=f"/cluster/scratch/{user}/copper_dataset")
    p.add_argument("--test-fold", type=int, default=2)
    p.add_argument("--val-fold", type=int, default=3)
    p.add_argument("--figure", default="figures/example_tiles.png")
    args = p.parse_args()

    tiles = np.load(os.path.join(args.data_dir, "tiles.npy"), mmap_mode="r")
    cells = pd.read_csv(os.path.join(args.data_dir, "fold_cells.csv"))
    folds = cells["fold"].values
    split_idx = {
        "train": np.flatnonzero((folds != args.test_fold) & (folds != args.val_fold)),
        "val": np.flatnonzero(folds == args.val_fold),
        "test": np.flatnonzero(folds == args.test_fold),
    }
    train_folds = sorted(set(folds[split_idx["train"]].tolist()))
    print(f"train folds {train_folds}, val fold {args.val_fold}, test fold {args.test_fold}: "
          + ", ".join(f"{k} {len(v)} tiles" for k, v in split_idx.items()))

    train_tiles = np.asarray(tiles[split_idx["train"]])
    stats = compute_stats(train_tiles)
    stats.update({"train_folds": train_folds, "val_fold": args.val_fold, "test_fold": args.test_fold})
    with open(os.path.join(args.data_dir, "norm_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps({k: stats[k] for k in CHANNEL_NAMES}, indent=2))

    print("Normalised channels (valid pixels): mean / std / min / max")
    for split, idx in split_idx.items():
        norm = normalize_tiles(np.asarray(tiles[idx]), stats)
        valid = norm[:, 3] == 1
        parts = []
        for c, name in enumerate(CHANNEL_NAMES):
            v = norm[:, c][valid]
            parts.append(f"{name} {v.mean():+.2f} / {v.std():.2f} / {v.min():+.1f} / {v.max():+.1f}")
        print(f"  {split:5s}: " + " | ".join(parts))
        if split == "train":
            save_examples(norm, cells, idx, args.figure)
    print(f"Saved {os.path.join(args.data_dir, 'norm_stats.json')} and {args.figure}")


if __name__ == "__main__":
    main()
