"""Crop the four input rasters to the densest built-up square window.

Usage:
    python scripts/crop_domain.py --src data/austin/Input_rasters --dst data/austin_crop --size 1000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from osgeo import gdal

gdal.UseExceptions()
NAMES = ["Building_DSM", "DEM", "Trees", "Landcover"]


def densest_window(dsm: np.ndarray, dem: np.ndarray, size: int, step: int = 50) -> tuple[int, int]:
    """Top-left corner of the size x size window with the largest building fraction."""
    built = ((dsm - dem) > 2.0).astype(np.float64)
    integral = np.pad(built.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    best, corner = -1.0, (0, 0)
    for r in range(0, built.shape[0] - size + 1, step):
        for c in range(0, built.shape[1] - size + 1, step):
            s = integral[r + size, c + size] - integral[r, c + size] - integral[r + size, c] + integral[r, c]
            if s > best:
                best, corner = s, (r, c)
    print(f"window at row {corner[0]}, col {corner[1]}: {100 * best / size**2:.1f}% buildings")
    return corner


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--size", type=int, default=1000)
    args = ap.parse_args()
    args.dst.mkdir(parents=True, exist_ok=True)

    dsm = gdal.Open(str(args.src / "Building_DSM.tif")).ReadAsArray().astype(np.float32)
    dem = gdal.Open(str(args.src / "DEM.tif")).ReadAsArray().astype(np.float32)
    r, c = densest_window(dsm, dem, args.size)
    for name in NAMES:
        gdal.Translate(str(args.dst / f"{name}.tif"), str(args.src / f"{name}.tif"),
                       srcWin=[c, r, args.size, args.size])


if __name__ == "__main__":
    main()
