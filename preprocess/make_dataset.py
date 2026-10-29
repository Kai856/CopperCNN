"""Cut a tile around every drillhole cell and save it with its copper labels.

Outputs:
  tiles.npy  float32, shape (N, 4, T, T), channels: gravity, tmi, fault_dist_km, valid_mask.
             Physical units; NaN where gravity/TMI have no data.
  cells.csv  one row per tile: row, col, x, y, cu_ppm, y_bin, y_log

Usage:
  python preprocess/make_dataset.py --grid-dir /cluster/scratch/$USER/copper_grid \
      --out-dir /cluster/scratch/$USER/copper_dataset
"""
import argparse
import os

import numpy as np
import pandas as pd
import rasterio

CHANNELS = ["gravity.tif", "tmi.tif", "fault_dist_km.tif", "valid_mask.tif"]


def main():
    user = os.environ.get("USER", "user")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid-dir", default=f"/cluster/scratch/{user}/copper_grid")
    p.add_argument("--out-dir", default=f"/cluster/scratch/{user}/copper_dataset")
    p.add_argument("--tile", type=int, default=32)
    p.add_argument("--threshold-ppm", type=float, default=1000.0)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    holes = pd.read_csv(os.path.join(args.grid_dir, "labels.csv"))
    cells = holes.groupby(["row", "col"], as_index=False)["cu_ppm"].max()
    print(f"{len(holes)} drillholes -> {len(cells)} cells")

    with rasterio.open(os.path.join(args.grid_dir, CHANNELS[0])) as src:
        transform, height, width = src.transform, src.height, src.width

    half = args.tile // 2
    r0, c0 = cells["row"].values - half, cells["col"].values - half
    inside = (r0 >= 0) & (c0 >= 0) & (r0 + args.tile <= height) & (c0 + args.tile <= width)
    if (~inside).any():
        print(f"Dropping {(~inside).sum()} cells whose tile runs off the grid")
    cells = cells[inside].reset_index(drop=True)
    r0, c0 = r0[inside], c0[inside]

    tiles = np.empty((len(cells), len(CHANNELS), args.tile, args.tile), dtype=np.float32)
    for ch, name in enumerate(CHANNELS):
        with rasterio.open(os.path.join(args.grid_dir, name)) as src:
            grid = src.read(1)
        for i, (r, c) in enumerate(zip(r0, c0)):
            tiles[i, ch] = grid[r:r + args.tile, c:c + args.tile]
        del grid
        print(f"  {name}: done")

    cells["x"] = transform.c + (cells["col"] + 0.5) * transform.a
    cells["y"] = transform.f + (cells["row"] + 0.5) * transform.e
    cells["y_bin"] = (cells["cu_ppm"] >= args.threshold_ppm).astype(np.int8)
    cells["y_log"] = np.log10(cells["cu_ppm"])
    cells = cells[["row", "col", "x", "y", "cu_ppm", "y_bin", "y_log"]]

    np.save(os.path.join(args.out_dir, "tiles.npy"), tiles)
    cells.to_csv(os.path.join(args.out_dir, "cells.csv"), index=False)

    centre_valid = tiles[:, 3, half, half] == 1
    print(f"Saved {len(cells)} tiles, shape {tiles.shape}, {tiles.nbytes / 1e9:.2f} GB")
    print(f"  y_bin = 1: {cells['y_bin'].sum()} ({cells['y_bin'].mean():.1%})")
    print(f"  y_log: median {cells['y_log'].median():.2f}, range {cells['y_log'].min():.2f} to {cells['y_log'].max():.2f}")
    print(f"  centre pixel valid: {centre_valid.mean():.2%}; "
          f"tiles fully valid: {(tiles[:, 3] == 1).all(axis=(1, 2)).mean():.1%}")
    print(f"Outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
