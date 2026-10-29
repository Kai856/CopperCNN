"""Assign every tile to one of K spatial folds using 50 km blocks.

Blocks are sorted by positives (most first). Each block with positives goes to the
fold with the fewest positives so far; each block without goes to the fold with the
fewest cells so far.

Output: fold_cells.csv = cells.csv + block_id, fold (same row order as tiles.npy).

Usage:
  python preprocess/make_folds.py --data-dir /cluster/scratch/$USER/copper_dataset
"""
import argparse
import os

import numpy as np
import pandas as pd


def main():
    user = os.environ.get("USER", "user")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=f"/cluster/scratch/{user}/copper_dataset")
    p.add_argument("--block-km", type=float, default=50.0)
    p.add_argument("--res", type=float, default=100.0, help="grid cell size in metres")
    p.add_argument("--folds", type=int, default=5)
    args = p.parse_args()

    cells = pd.read_csv(os.path.join(args.data_dir, "cells.csv"))
    block_cells = int(round(args.block_km * 1000 / args.res))
    cells["block_id"] = (cells["row"] // block_cells).astype(str) + "_" + (cells["col"] // block_cells).astype(str)

    blocks = (cells.groupby("block_id")
              .agg(n_cells=("y_bin", "size"), n_pos=("y_bin", "sum"))
              .sort_values(["n_pos", "n_cells"], ascending=False))

    fold_pos = np.zeros(args.folds)
    fold_cells = np.zeros(args.folds)
    block_fold = {}
    for block_id, n_cells, n_pos in zip(blocks.index, blocks["n_cells"], blocks["n_pos"]):
        if n_pos > 0:
            k = int(np.lexsort((fold_cells, fold_pos))[0])
        else:
            k = int(np.argmin(fold_cells))
        block_fold[block_id] = k
        fold_pos[k] += n_pos
        fold_cells[k] += n_cells

    cells["fold"] = cells["block_id"].map(block_fold)
    cells.to_csv(os.path.join(args.data_dir, "fold_cells.csv"), index=False)

    blocks["fold"] = blocks.index.map(block_fold)
    summary = blocks.groupby("fold").agg(
        blocks=("n_cells", "size"),
        blocks_with_pos=("n_pos", lambda s: int((s > 0).sum())),
        cells=("n_cells", "sum"),
        positives=("n_pos", "sum"),
    )
    summary["pos_rate"] = (summary["positives"] / summary["cells"]).map("{:.1%}".format)
    print(f"{len(cells)} cells in {len(blocks)} blocks of {args.block_km:g} km")
    print(summary.to_string())
    top = blocks.iloc[0]
    print(f"Largest block {blocks.index[0]}: {top['n_pos']} positives "
          f"({top['n_pos'] / blocks['n_pos'].sum():.0%} of all), in fold {top['fold']}")
    print(f"Saved {os.path.join(args.data_dir, 'fold_cells.csv')}")


if __name__ == "__main__":
    main()
