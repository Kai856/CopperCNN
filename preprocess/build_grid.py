"""Build a common GDA2020 / SA Lambert (EPSG:8059) grid from the SARIG downloads.

Outputs (one GeoTIFF per channel, all on the same grid):
  gravity.tif        Bouguer gravity, um/s^2, bilinear
  tmi.tif            Total magnetic intensity, nT, area-averaged
  valid_mask.tif     1 where both gravity and TMI have data
  fault_dist_km.tif  Distance to the nearest fault (all age layers), km
  rock_<layer>.tif   Integer rock-group code per age layer (0 = no polygon)
  labels.csv         One row per drillhole: max Cu ppm, grid row/col
  grid.json          Grid definition, channel list and category lookups

Usage:
  python preprocess/build_grid.py --data-dir ~/sarig_data --out-dir /cluster/scratch/$USER/copper_grid
"""
import argparse
import glob
import json
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin, rowcol
from rasterio.warp import reproject, transform_bounds
from scipy.ndimage import distance_transform_edt

SRC_CRS = CRS.from_epsg(7844)
DST_CRS = CRS.from_epsg(8059)
SRC_NODATA = -99999.0
SA_BOUNDS_LONLAT = (129.0, -38.6, 141.0, -25.99)

AGE_LAYERS = {
    "archaean_early_meso": "Archaean - Early Mesoproterozoic",
    "middle_meso": "Middle Mesoproterozoic",
    "late_meso": "Late Mesoproterozoic",
    "neo_ordovician": "Neoproterozoic - Ordovician",
}


def find_one(data_dir, pattern):
    hits = sorted(glob.glob(os.path.join(data_dir, pattern), recursive=True))
    if not hits:
        raise FileNotFoundError(f"No file matching {pattern} under {data_dir}")
    return hits[0]


def make_grid(res):
    xmin, ymin, xmax, ymax = transform_bounds(SRC_CRS, DST_CRS, *SA_BOUNDS_LONLAT, densify_pts=200)
    xmin, ymin = np.floor(xmin / res) * res, np.floor(ymin / res) * res
    xmax, ymax = np.ceil(xmax / res) * res, np.ceil(ymax / res) * res
    width, height = int(round((xmax - xmin) / res)), int(round((ymax - ymin) / res))
    return from_origin(xmin, ymax, res, res), (height, width)


def write_tif(path, array, transform, nodata=None):
    is_float = np.issubdtype(array.dtype, np.floating)
    profile = dict(
        driver="GTiff", height=array.shape[0], width=array.shape[1], count=1,
        dtype=array.dtype, crs=DST_CRS, transform=transform, nodata=nodata,
        compress="deflate", predictor=3 if is_float else 2, tiled=True,
        blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER",
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array, 1)


def reproject_grid(src_path, transform, shape, resampling, threads):
    out = np.full(shape, np.nan, dtype=np.float32)
    with rasterio.open(src_path) as src:
        print(f"  source {os.path.basename(src_path)}: {src.width}x{src.height}, bounds {src.bounds}")
        reproject(
            source=rasterio.band(src, 1), destination=out,
            src_transform=src.transform, src_crs=SRC_CRS, src_nodata=SRC_NODATA,
            dst_transform=transform, dst_crs=DST_CRS, dst_nodata=np.nan,
            resampling=resampling, num_threads=threads,
        )
    return out


def summarize(name, a):
    v = a[np.isfinite(a)]
    print(f"  {name}: {v.size / a.size:.1%} valid, min {v.min():.2f}, "
          f"p1 {np.percentile(v, 1):.2f}, median {np.median(v):.2f}, p99 {np.percentile(v, 99):.2f}, max {v.max():.2f}")


def build_fault_distance(data_dir, transform, shape, res):
    shapes = []
    for label in AGE_LAYERS.values():
        faults = gpd.read_file(find_one(data_dir, f"{label} faults_GDA2020_shp/*.shp"))
        faults = faults.set_crs(SRC_CRS, allow_override=True).to_crs(DST_CRS)
        shapes.extend((geom, 1) for geom in faults.geometry if geom is not None)
        print(f"  {label}: {len(faults)} faults")
    burned = rasterize(shapes, out_shape=shape, transform=transform, fill=0, all_touched=True, dtype="uint8")
    dist_m = distance_transform_edt(burned == 0, sampling=res)
    return (dist_m / 1000.0).astype(np.float32)


def build_rock_codes(data_dir, out_dir, transform, shape, field):
    lookups = {}
    for key, label in AGE_LAYERS.items():
        polys = gpd.read_file(find_one(data_dir, f"{label} polygons_GDA2020_shp/*.shp"))
        polys = polys.set_crs(SRC_CRS, allow_override=True).to_crs(DST_CRS)
        values = polys[field].fillna("unknown").astype(str)
        categories = sorted(values.unique())
        codes = {c: i + 1 for i, c in enumerate(categories)}
        shapes = ((g, codes[v]) for g, v in zip(polys.geometry, values) if g is not None)
        raster = rasterize(shapes, out_shape=shape, transform=transform, fill=0, dtype="int16")
        write_tif(os.path.join(out_dir, f"rock_{key}.tif"), raster, transform, nodata=0)
        lookups[f"rock_{key}.tif"] = {"field": field, "codes": codes}
        print(f"  {label}: {len(polys)} polygons, {len(categories)} {field} categories, "
              f"{(raster > 0).mean():.1%} of grid covered")
    return lookups


def build_labels(data_dir, transform, shape, valid):
    cu = gpd.read_file(find_one(data_dir, "copper_maxdh_GDA2020_shp/*.shp"))
    cu = cu.set_crs(SRC_CRS, allow_override=True).to_crs(DST_CRS)
    n_raw = len(cu)
    cu = cu.sort_values("CHEM_VALUE", ascending=False).drop_duplicates("DRILLHOL01")
    rows, cols = rowcol(transform, cu.geometry.x.values, cu.geometry.y.values)
    rows, cols = np.asarray(rows), np.asarray(cols)
    inside = (rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1])
    labels = pd.DataFrame({
        "drillhole_id": cu["DRILLHOL01"].values,
        "drillhole_name": cu["DRILLHOLE_"].values,
        "x": cu.geometry.x.values,
        "y": cu.geometry.y.values,
        "row": rows,
        "col": cols,
        "cu_ppm": cu["CHEM_VALUE"].values,
        "log10_cu_ppm": np.log10(cu["CHEM_VALUE"].values),
        "depth_from_m": cu["DH_DEPTH_F"].values,
        "depth_to_m": cu["DH_DEPTH_T"].values,
        "analysis_method": cu["ANALYSIS_M"].values,
    })[inside]
    labels["valid"] = valid[labels["row"], labels["col"]].astype(bool)
    print(f"  {n_raw} rows -> {len(cu)} unique drillholes -> {len(labels)} inside grid, "
          f"{labels['valid'].sum()} on valid geophysics")
    return labels


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=os.path.expanduser("~/sarig_data"))
    p.add_argument("--out-dir", default=f"/cluster/scratch/{os.environ.get('USER', 'user')}/copper_grid")
    p.add_argument("--res", type=float, default=100.0, help="cell size in metres")
    p.add_argument("--rock-field", default="SG_PROVINC", help="polygon attribute used as rock group")
    p.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    transform, shape = make_grid(args.res)
    print(f"Grid: EPSG:8059, {args.res:g} m, {shape[1]} x {shape[0]} cells, origin ({transform.c}, {transform.f})")

    print("Gravity")
    grav = reproject_grid(find_one(args.data_dir, "sa_grav_ers/**/*GEODETIC.ers"),
                          transform, shape, Resampling.bilinear, args.threads)
    summarize("gravity (um/s^2)", grav)
    write_tif(os.path.join(args.out_dir, "gravity.tif"), grav, transform, nodata=np.nan)

    print("TMI")
    tmi = reproject_grid(find_one(args.data_dir, "SA_TMI_ERS_GDA2020/SA_TMI.ers"),
                         transform, shape, Resampling.average, args.threads)
    summarize("tmi (nT)", tmi)
    write_tif(os.path.join(args.out_dir, "tmi.tif"), tmi, transform, nodata=np.nan)

    valid = (np.isfinite(grav) & np.isfinite(tmi)).astype(np.uint8)
    print(f"Valid mask: {valid.mean():.1%} of grid")
    write_tif(os.path.join(args.out_dir, "valid_mask.tif"), valid, transform)
    del grav, tmi

    print("Faults")
    fault = build_fault_distance(args.data_dir, transform, shape, args.res)
    summarize("fault distance (km)", np.where(valid == 1, fault, np.nan))
    write_tif(os.path.join(args.out_dir, "fault_dist_km.tif"), fault, transform)
    del fault

    print(f"Rock units ({args.rock_field})")
    rock_lookups = build_rock_codes(args.data_dir, args.out_dir, transform, shape, args.rock_field)

    print("Labels")
    labels = build_labels(args.data_dir, transform, shape, valid)
    labels.to_csv(os.path.join(args.out_dir, "labels.csv"), index=False)

    meta = {
        "crs": "EPSG:8059",
        "res_m": args.res,
        "shape": list(shape),
        "transform": list(transform)[:6],
        "channels": {
            "gravity.tif": "Bouguer gravity, um/s^2",
            "tmi.tif": "Total magnetic intensity, nT",
            "valid_mask.tif": "1 = gravity and TMI both present",
            "fault_dist_km.tif": "distance to nearest fault, km",
            **{c: "categorical rock group code, 0 = none" for c in rock_lookups},
        },
        "rock_lookups": rock_lookups,
    }
    with open(os.path.join(args.out_dir, "grid.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Done. Outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
