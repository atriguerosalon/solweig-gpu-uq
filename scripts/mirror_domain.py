"""Enlarge a domain by mirroring it at its edges, for timing and memory tests only.

Run time depends on tile size and on the height range (which sets the length of the shadow
loops), not on where the buildings are, so a mirrored domain is a fair stand-in when a bigger
real one cannot be downloaded (the USGS elevation service refused 3050 x 3050 px on 1 Nov 2026).
Never use it for heat results: the city it shows does not exist.

Usage:
    python scripts/mirror_domain.py --src data/la2000/pacoima --size 3000 --out data/la3000_mirror/pacoima
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

NAMES = ["Building_DSM", "DEM", "Trees", "Landcover"]


def main() -> None:
    from osgeo import gdal

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    gdal.UseExceptions()
    args.out.mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        ds = gdal.Open(str(args.src / f"{name}.tif"))
        a = ds.GetRasterBand(1).ReadAsArray()
        pad = args.size - a.shape[0]
        if pad < 0 or a.shape[0] != a.shape[1]:
            raise ValueError("source must be square and smaller than --size")
        b = np.pad(a, ((pad // 2, pad - pad // 2), (pad // 2, pad - pad // 2)), mode="reflect")
        gt = list(ds.GetGeoTransform())
        gt[0] -= (pad // 2) * gt[1]
        gt[3] -= (pad // 2) * gt[5]
        out = gdal.GetDriverByName("GTiff").Create(str(args.out / f"{name}.tif"), args.size, args.size, 1,
                                                    ds.GetRasterBand(1).DataType)
        out.SetGeoTransform(gt)
        out.SetProjection(ds.GetProjection())
        out.GetRasterBand(1).WriteArray(b)
        out = None
    (args.out / "MIRRORED_NOT_A_REAL_CITY.txt").write_text(
        f"Mirrored from {args.src} to {args.size} px for timing tests only (scripts/mirror_domain.py).\n")
    print(f"[mirror] {args.src} -> {args.out}, {args.size} px")


if __name__ == "__main__":
    main()
