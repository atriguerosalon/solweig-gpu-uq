"""Geometry and tree inputs in Pacoima: building heights, tree heights, tree transmissivity.

The other sensitivity runs keep the geometry fixed, because changing it means recomputing the
sky view factor. Here each geometry case gets its own rasters, sky view factor and day run.

    baseline     rasters as built by build_la_domain.py
    bldg_lo/hi   building heights above ground x 0.8 / x 1.2 (LA County heights; a pitched
                 roof's eave is lower than its ridge, so 20% is a plausible definition error)
    trees_lo/hi  canopy height -2.8 / +2.8 m (mean absolute error of the Meta canopy-height
                 map, Tolan et al. 2024); footprint kept, heights not below 2 m
    trans_hi     tree shortwave transmissivity 10% instead of SOLWEIG's hard-coded 3%
                 (geometry unchanged, so it reuses the baseline sky view factor)

Runs use the fast functions and compact masks (verified equal to the original).

Usage:
    python scripts/run_la_geometry.py --rasters data/la/pacoima --met data/la/met_pacoima_20190730.txt
"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import sys
import textwrap
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solweig_sens.analysis import is_complete  # noqa: E402
from solweig_sens.run import Setup, prepare  # noqa: E402

CASES = {
    "baseline": {},
    "bldg_lo": {"bldg_scale": 0.8},
    "bldg_hi": {"bldg_scale": 1.2},
    "trees_lo": {"tree_add": -2.8},
    "trees_hi": {"tree_add": 2.8},
    "trans_hi": {"trans": 0.10},
}
TREE_MIN = 2.0  # build_la_domain.py drops canopy below 2 m


def write_rasters(src: Path, dst: Path, bldg_scale: float = 1.0, tree_add: float = 0.0) -> None:
    from osgeo import gdal

    dst.mkdir(parents=True, exist_ok=True)
    for name in ("DEM", "Landcover", "Building_DSM", "Trees"):
        ds = gdal.Open(str(src / f"{name}.tif"))
        arr = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
        if name == "Building_DSM":
            ground = gdal.Open(str(src / "DEM.tif")).ReadAsArray().astype(np.float32)
            arr = ground + (arr - ground) * bldg_scale
        if name == "Trees" and tree_add:
            arr = np.where(arr > 0, np.maximum(arr + tree_add, TREE_MIN), 0).astype(np.float32)
        out = gdal.GetDriverByName("GTiff").Create(str(dst / f"{name}.tif"), ds.RasterXSize, ds.RasterYSize, 1,
                                                    gdal.GDT_Float32)
        out.SetGeoTransform(ds.GetGeoTransform())
        out.SetProjection(ds.GetProjection())
        out.GetRasterBand(1).WriteArray(arr)
        out = None


@contextlib.contextmanager
def tree_transmissivity(value: float):
    """compute_utci with a different hard-coded transVeg (it is a local, not a module global)."""
    import solweig_gpu.utci_process as up

    src = textwrap.dedent(inspect.getsource(up.compute_utci))
    old = "transVeg = 3. / 100."
    if src.count(old) != 1:
        raise RuntimeError("cannot patch transVeg in compute_utci")
    original = up.compute_utci
    try:
        exec(compile(src.replace(old, f"transVeg = {value!r}"), up.__file__, "exec"), vars(up))
        yield
    finally:
        up.compute_utci = original


def main() -> None:
    from solweig_gpu import run_utci_tiles
    from solweig_sens.compact_masks import compact_masks
    from solweig_sens.fast import fast_solweig

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rasters", type=Path, required=True)
    ap.add_argument("--met", type=Path, required=True)
    ap.add_argument("--runs", type=Path, default=Path("runs_la_geom"))
    ap.add_argument("--date", default="2019-07-30")
    args = ap.parse_args()
    runs = args.runs.resolve()
    runs.mkdir(parents=True, exist_ok=True)
    (runs / "cases.json").write_text(json.dumps(CASES, indent=2))
    timings_path = runs / "timings.json"
    timings = json.loads(timings_path.read_text()) if timings_path.exists() else {}

    for name, p in CASES.items():
        out_dir = runs / name / "out"
        if is_complete(out_dir):
            continue
        geometry = "trans" not in p
        case_dir = runs / (name if geometry else "baseline")
        setup = Setup(raster_dir=case_dir / "rasters", met_file=args.met.resolve(), runs_dir=case_dir, date=args.date)
        t0 = time.perf_counter()
        if geometry:
            write_rasters(args.rasters.resolve(), setup.raster_dir, p.get("bldg_scale", 1.0), p.get("tree_add", 0.0))
            svf_dir = setup.preprocess_dir / "SVF"
            if not (svf_dir.exists() and any(svf_dir.glob("SkyViewFactor_*.tif"))):
                with compact_masks("bits", svf=True):
                    prepare(setup)
        t1 = time.perf_counter()
        out_dir.mkdir(parents=True, exist_ok=True)
        run = lambda: run_utci_tiles(base_path=str(out_dir), preprocess_dir=str(setup.preprocess_dir),  # noqa: E731
                                     selected_date_str=args.date, save_tmrt=True, save_shadow=True)
        if geometry:
            with fast_solweig(), compact_masks("bits", svf=False):
                run()
        else:
            with fast_solweig(), tree_transmissivity(p["trans"]):
                run()
        timings[name] = {"geometry_s": t1 - t0, "day_s": time.perf_counter() - t1}
        timings_path.write_text(json.dumps(timings, indent=2))
        print(f"[geom] {name}: geometry {t1 - t0:.0f} s, day {time.perf_counter() - t1:.0f} s", flush=True)


if __name__ == "__main__":
    main()
