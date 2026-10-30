"""Read SOLWEIG-GPU outputs and summarise how each case differs from the baseline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .perturb import Perturbation

# UTCI heat-stress category lower bounds (degC), Broede et al. 2012.
UTCI_STRONG = 32.0
UTCI_VERY_STRONG = 38.0


def read_stack(run_dir: Path, var: str) -> np.ndarray:
    """Read one output variable as (hours, rows, cols), mosaicking tiles if needed.

    Tiles are placed by their geotransform; the overlap strips are written twice,
    which is harmless because neighbouring tiles agree there.
    """
    from osgeo import gdal

    files = sorted((run_dir / "output_folder").glob(f"*/{var}_*.tif"))
    if not files:
        raise FileNotFoundError(f"no {var} outputs under {run_dir}")
    dss = [gdal.Open(str(f)) for f in files]
    gts = [ds.GetGeoTransform() for ds in dss]
    px, py = gts[0][1], gts[0][5]
    x0 = min(g[0] for g in gts)
    y0 = max(g[3] for g in gts)
    width = max(round((g[0] - x0) / px) + ds.RasterXSize for g, ds in zip(gts, dss))
    height = max(round((g[3] - y0) / py) + ds.RasterYSize for g, ds in zip(gts, dss))
    bands = dss[0].RasterCount
    out = np.full((bands, height, width), np.nan, dtype=np.float32)
    for g, ds in zip(gts, dss):
        c, r = round((g[0] - x0) / px), round((g[3] - y0) / py)
        out[:, r:r + ds.RasterYSize, c:c + ds.RasterXSize] = ds.ReadAsArray().reshape(bands, ds.RasterYSize, ds.RasterXSize)
    return out


def is_complete(run_dir: Path) -> bool:
    """True if every UTCI tile exists and its last hour holds data.

    A run killed part-way leaves the output folder behind, so its existence is not enough.
    """
    from osgeo import gdal

    files = sorted((run_dir / "output_folder").glob("*/UTCI_*.tif"))
    if not files:
        return False
    for f in files:
        try:
            ds = gdal.Open(str(f))
            last = ds.GetRasterBand(ds.RasterCount).ReadAsArray()
        except RuntimeError:
            return False
        if not np.any(np.isfinite(last) & (last != 0) & (last != -999)):
            return False
    return True


def outdoor_mask(raster_dir: Path, shape: tuple[int, int]) -> np.ndarray:
    """True where a pedestrian could stand: not on a building roof."""
    from osgeo import gdal

    dsm = gdal.Open(str(raster_dir / "Building_DSM.tif")).ReadAsArray().astype(np.float32)
    dem = gdal.Open(str(raster_dir / "DEM.tif")).ReadAsArray().astype(np.float32)
    mask = (dsm - dem) < 0.5
    return mask[: shape[0], : shape[1]]


def summarise(runs_dir: Path, raster_dir: Path, cases: list[Perturbation], hour: int) -> pd.DataFrame:
    """Mean change in UTCI and Tmrt at one hour, overall and split by sun and shade."""
    base_utci = read_stack(runs_dir / "baseline", "UTCI")
    base_tmrt = read_stack(runs_dir / "baseline", "TMRT")
    shadow = read_stack(runs_dir / "baseline", "Shadow")[hour]
    outdoor = outdoor_mask(raster_dir, base_utci.shape[1:])
    valid = outdoor & np.isfinite(base_utci[hour])
    sun = valid & (shadow > 0.5)  # SOLWEIG shadow: 1 = sunlit, 0 = shaded
    shade = valid & (shadow <= 0.5)

    rows = []
    for case in cases:
        d = runs_dir / case.name
        du = read_stack(d, "UTCI")[hour] - base_utci[hour]
        dt = read_stack(d, "TMRT")[hour] - base_tmrt[hour]
        u = read_stack(d, "UTCI")[hour]
        rows.append({
            "case": case.name,
            "input": case.group,
            "value": case.value,
            "dUTCI_mean": float(np.nanmean(du[valid])),
            "dUTCI_sun": float(np.nanmean(du[sun])),
            "dUTCI_shade": float(np.nanmean(du[shade])),
            "dUTCI_p95_abs": float(np.nanpercentile(np.abs(du[valid]), 95)),
            "dTmrt_mean": float(np.nanmean(dt[valid])),
            "area_ge38_pct": float(100 * np.mean(u[valid] >= UTCI_VERY_STRONG)),
        })
    df = pd.DataFrame(rows)
    df.attrs["baseline_area_ge38_pct"] = float(100 * np.mean(base_utci[hour][valid] >= UTCI_VERY_STRONG))
    df.attrs["baseline_utci_mean"] = float(np.nanmean(base_utci[hour][valid]))
    df.attrs["sun_fraction"] = float(sun.sum() / valid.sum())
    return df
