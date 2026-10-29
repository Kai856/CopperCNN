"""Print GeoTIFF metadata using only the Python standard library.

Usage: python3 inspect_tif.py file1.tif [file2.tif ...]
"""
import struct
import sys

TYPES = {1: "B", 2: "s", 3: "H", 4: "I", 5: "II", 6: "b", 7: "B", 8: "h",
         9: "i", 10: "ii", 11: "f", 12: "d", 16: "Q", 17: "q", 18: "Q"}
SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8,
         11: 4, 12: 8, 16: 8, 17: 8, 18: 8}
TAGS = {256: "width", 257: "height", 258: "bits_per_sample", 259: "compression",
        277: "samples_per_pixel", 322: "tile_width", 323: "tile_length",
        339: "sample_format", 33550: "pixel_scale", 33922: "tiepoint",
        34264: "transform", 34735: "geokeys", 34736: "geo_doubles",
        34737: "geo_ascii", 42112: "gdal_metadata", 42113: "gdal_nodata"}
GEOKEYS = {1024: "ModelType(1=projected,2=geographic)", 1025: "RasterType(1=area,2=point)",
           2048: "GeographicCRS_EPSG", 2049: "GeogCitation", 2054: "AngularUnits",
           3072: "ProjectedCRS_EPSG", 3073: "PCSCitation", 3076: "LinearUnits",
           4096: "VerticalCRS_EPSG"}
SAMPLE_FORMAT = {1: "unsigned int", 2: "signed int", 3: "float"}


def read_tags(path):
    with open(path, "rb") as f:
        order = f.read(2)
        e = "<" if order == b"II" else ">"
        magic = struct.unpack(e + "H", f.read(2))[0]
        big = magic == 43
        if big:
            f.read(4)
            offset = struct.unpack(e + "Q", f.read(8))[0]
        else:
            offset = struct.unpack(e + "I", f.read(4))[0]
        f.seek(offset)
        n = struct.unpack(e + ("Q" if big else "H"), f.read(8 if big else 2))[0]
        out = {}
        for _ in range(n):
            if big:
                tag, typ, count = struct.unpack(e + "HHQ", f.read(12))
                raw = f.read(8)
            else:
                tag, typ, count = struct.unpack(e + "HHI", f.read(8))
                raw = f.read(4)
            if tag not in TAGS or typ not in SIZES:
                continue
            size = SIZES[typ] * count
            pos = f.tell()
            if size > len(raw):
                f.seek(struct.unpack(e + ("Q" if big else "I"), raw)[0])
                data = f.read(size)
                f.seek(pos)
            else:
                data = raw[:size]
            if typ == 2:
                val = data.rstrip(b"\x00").decode("latin-1")
            elif typ in (5, 10):
                nums = struct.unpack(e + TYPES[typ][0] * (2 * count), data)
                val = [nums[i] / nums[i + 1] for i in range(0, len(nums), 2)]
            else:
                val = list(struct.unpack(e + TYPES[typ] * count, data))
            out[TAGS[tag]] = val
        return out


def describe(path):
    t = read_tags(path)
    print(f"=== {path}")
    w, h = t["width"][0], t["height"][0]
    print(f"  size: {w} x {h} pixels, bands: {t.get('samples_per_pixel', [1])[0]}")
    fmt = SAMPLE_FORMAT.get(t.get("sample_format", [1])[0], "?")
    print(f"  dtype: {t['bits_per_sample'][0]}-bit {fmt}, compression code: {t.get('compression', ['?'])[0]}")
    if "gdal_nodata" in t:
        print(f"  nodata: {t['gdal_nodata']}")
    if "pixel_scale" in t and "tiepoint" in t:
        sx, sy = t["pixel_scale"][:2]
        tp = t["tiepoint"]
        x0, y0 = tp[3] - tp[0] * sx, tp[4] + tp[1] * sy
        print(f"  pixel size: {sx} x {sy}")
        print(f"  extent: x {x0} .. {x0 + w * sx}, y {y0 - h * sy} .. {y0}")
    if "geokeys" in t:
        k = t["geokeys"]
        for i in range(4, len(k), 4):
            key, loc, cnt, val = k[i:i + 4]
            if loc == 34737:
                val = t.get("geo_ascii", "")[val:val + cnt].rstrip("|")
            elif loc == 34736:
                val = t.get("geo_doubles", [])[val:val + cnt]
            print(f"  geokey {GEOKEYS.get(key, key)}: {val}")
    if "gdal_metadata" in t:
        print(f"  gdal metadata: {t['gdal_metadata'][:500]}")


for p in sys.argv[1:]:
    describe(p)
