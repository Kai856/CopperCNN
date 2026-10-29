"""Cut a tile around every drillhole cell and save it with its copper labels.

Outputs:
  tiles.npy        float32, shape (N, 4, T, T), channels: gravity, tmi, fault_dist_km, valid_mask.
                   Physical units; NaN where gravity/TMI have no data.
  rock.npy         int16, shape (N, 4, T, T), rock unit codes per age layer (0 = unmapped).
  rock_layers.json layer names and number of codes per layer (from grid.json).
  cells.csv        one row per tile: row, col, x, y, cu_ppm, y_bin, y_log, depth_m
                   (depth_m = deepest hole in the cell from the Mineral drillholes layer; NaN if unmatched)

Usage:
  python preprocess/make_dataset.py --grid-dir /cluster/scratch/$USER/copper_grid \
      --out-dir /cluster/scratch/$USER/copper_dataset64 --sarig-dir /cluster/scratch/$USER/sarig_data
"""
import argparse
import glob
import json
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from sklearn.metrics import roc_auc_score

CHANNELS = ["gravity.tif", "tmi.tif", "fault_dist_km.tif", "valid_mask.tif"]


def cut_tiles(path, r0, c0, tile, dtype):
    with rasterio.open(path) as src:
        grid = src.read(1)
    out = np.empty((len(r0), tile, tile), dtype=dtype)
    for i, (r, c) in enumerate(zip(r0, c0)):
        out[i] = grid[r:r + tile, c:c + tile]
    return out


def hole_depths(sarig_dir, id_field, depth_field):
    hits = sorted(glob.glob(os.path.join(sarig_dir, "**", "Mineral drillholes*.shp"), recursive=True))
    if not hits:
        raise FileNotFoundError(f"No 'Mineral drillholes*.shp' under {sarig_dir}")
    dh = gpd.read_file(hits[0], columns=[id_field, depth_field], ignore_geometry=True)
    dh = dh.dropna(subset=[id_field])
    return dh.groupby(dh[id_field].astype(np.int64))[depth_field].max()


def main():
    user = os.environ.get("USER", "user")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid-dir", default=f"/cluster/scratch/{user}/copper_grid")
    p.add_argument("--out-dir", default=f"/cluster/scratch/{user}/copper_dataset64")
    p.add_argument("--sarig-dir", default=f"/cluster/scratch/{user}/sarig_data")
    p.add_argument("--dh-id-field", default="DHNUMBER", help="matches DRILLHOL01 in copper_maxdh")
    p.add_argument("--dh-depth-field", default="MAXDEPTH", help="total hole depth, m")
    p.add_argument("--tile", type=int, default=64)
    p.add_argument("--threshold-ppm", type=float, default=1000.0)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    holes = pd.read_csv(os.path.join(args.grid_dir, "labels.csv"))
    depths = hole_depths(args.sarig_dir, args.dh_id_field, args.dh_depth_field)
    holes["depth_m"] = holes["drillhole_id"].astype(np.int64).map(depths)
    print(f"Hole depth matched for {holes['depth_m'].notna().sum()} of {len(holes)} drillholes "
          f"({holes['depth_m'].notna().mean():.1%})")
    cells = holes.groupby(["row", "col"], as_index=False).agg(cu_ppm=("cu_ppm", "max"), depth_m=("depth_m", "max"))
    print(f"{len(holes)} drillholes -> {len(cells)} cells")

    with open(os.path.join(args.grid_dir, "grid.json")) as f:
        rock_lookups = json.load(f)["rock_lookups"]
    rock_layers = list(rock_lookups)
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
        tiles[:, ch] = cut_tiles(os.path.join(args.grid_dir, name), r0, c0, args.tile, np.float32)
        print(f"  {name}: done")
    rock = np.empty((len(cells), len(rock_layers), args.tile, args.tile), dtype=np.int16)
    for ch, name in enumerate(rock_layers):
        rock[:, ch] = cut_tiles(os.path.join(args.grid_dir, name), r0, c0, args.tile, np.int16)
        print(f"  {name}: {len(rock_lookups[name]['codes'])} codes, centre pixel mapped in "
              f"{(rock[:, ch, half, half] > 0).mean():.1%} of cells")

    cells["x"] = transform.c + (cells["col"] + 0.5) * transform.a
    cells["y"] = transform.f + (cells["row"] + 0.5) * transform.e
    cells["y_bin"] = (cells["cu_ppm"] >= args.threshold_ppm).astype(np.int8)
    cells["y_log"] = np.log10(cells["cu_ppm"])
    cells = cells[["row", "col", "x", "y", "cu_ppm", "y_bin", "y_log", "depth_m"]]

    np.save(os.path.join(args.out_dir, "tiles.npy"), tiles)
    np.save(os.path.join(args.out_dir, "rock.npy"), rock)
    with open(os.path.join(args.out_dir, "rock_layers.json"), "w") as f:
        json.dump({"layers": rock_layers, "n_codes": [len(rock_lookups[n]["codes"]) for n in rock_layers]}, f, indent=2)
    cells.to_csv(os.path.join(args.out_dir, "cells.csv"), index=False)

    centre_valid = tiles[:, 3, half, half] == 1
    print(f"Saved {len(cells)} tiles, shape {tiles.shape}, {tiles.nbytes / 1e9:.2f} GB (+ rock {rock.nbytes / 1e9:.2f} GB)")
    print(f"  y_bin = 1: {cells['y_bin'].sum()} ({cells['y_bin'].mean():.1%})")
    print(f"  y_log: median {cells['y_log'].median():.2f}, range {cells['y_log'].min():.2f} to {cells['y_log'].max():.2f}")
    print(f"  centre pixel valid: {centre_valid.mean():.2%}; "
          f"tiles fully valid: {(tiles[:, 3] == 1).all(axis=(1, 2)).mean():.1%}")
    d = cells["depth_m"]
    has = d.notna()
    print(f"  depth_m: {has.mean():.1%} of cells, median {d.median():.0f} m, p10 {d.quantile(0.1):.0f} m, "
          f"p90 {d.quantile(0.9):.0f} m")
    print(f"  depth alone as a score: ROC-AUC {roc_auc_score(cells['y_bin'][has], d[has]):.3f} (cells with depth)")
    print(f"Outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
