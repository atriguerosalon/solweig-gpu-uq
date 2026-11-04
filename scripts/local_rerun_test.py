"""Can SOLWEIG be rerun only around a new tree? Windowed rerun against the full rerun.

A new tree changes the sky view factor and the shadows only within its shadow reach (about
height / tan(altitude): ~95 m for a 10 m tree at the lowest sky patches, ~6 deg), and Tmrt with a
small extra halo (ground and wall view factors). So a window around the tree, padded by a halo,
should reproduce the full rerun exactly in its core, at a fraction of the cost. This tests it on
the tree-planting runs of run_la_treeplant.py: for one tree, the window = core (+-core_m around the
tree) + halo, cut from that tree's full-rerun rasters and run with the same code (fast functions,
compact masks, tight bound). In the core, sky view factor, hourly Shadow and Tmrt are compared
with the full rerun (bit-identical or max |difference|), and both are timed.

Usage:
    python scripts/local_rerun_test.py --runs ~/treeplant --halos 25 50 100 150
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from solweig_sens.analysis import read_stack  # noqa: E402
from solweig_sens.run import Setup, prepare  # noqa: E402

NAMES = ["Building_DSM", "DEM", "Trees", "Landcover"]


def run(raster_dir: Path, d: Path, met: Path, date: str, tile: int) -> dict:
    import torch
    from solweig_gpu import run_utci_tiles
    from solweig_sens.compact_masks import compact_masks
    from solweig_sens.fast import fast_solweig
    from solweig_sens.tight_bound import tight_bound

    shutil.rmtree(d, ignore_errors=True)
    setup = Setup(raster_dir=raster_dir, met_file=met, runs_dir=d, date=date, tile_size=tile)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with compact_masks("bits", svf=True), tight_bound():
        prepare(setup)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    with fast_solweig(), compact_masks("bits", svf=False), tight_bound():
        run_utci_tiles(base_path=str(d / "out"), preprocess_dir=str(setup.preprocess_dir), selected_date_str=date,
                       save_tmrt=True, save_shadow=True)
    torch.cuda.synchronize()
    return {"svf_s": t1 - t0, "day_s": time.perf_counter() - t1}


def main() -> None:
    from osgeo import gdal

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path.home() / "treeplant")
    ap.add_argument("--tree", type=int, default=None, help="tree index (default: the one closest to the centre)")
    ap.add_argument("--core-m", type=float, default=40.0)
    ap.add_argument("--halos", type=float, nargs="*", default=[25, 50, 100, 150])
    ap.add_argument("--work", type=Path, default=Path.home() / "localrerun")
    ap.add_argument("--out", type=Path, default=ROOT / "figures")
    args = ap.parse_args()
    gdal.UseExceptions()
    design = json.loads((args.runs / "design.json").read_text())
    full_rasters = args.runs / "single_00" / "rasters"
    n = gdal.Open(str(full_rasters / "DEM.tif")).RasterXSize
    res_m = gdal.Open(str(full_rasters / "DEM.tif")).GetGeoTransform()[1]
    sites = np.array(design["sites"])
    i = args.tree if args.tree is not None else int(np.argmin(np.hypot(*(sites - n / 2).T)))
    full = args.runs / f"single_{i:02d}"
    cy, cx = (int(v) for v in sites[i])
    core = int(round(args.core_m / res_m))
    ref = {"svf": gdal.Open(str(full / "shared/processed_inputs/SVF/SkyViewFactor_0_0.tif")).ReadAsArray(),
           "Shadow": read_stack(full / "out", "Shadow"), "TMRT": read_stack(full / "out", "TMRT")}
    met = Path(design["met"])
    res = {"tree": i, "site_px": [cy, cx], "core_m": args.core_m, "domain_px": n,
           "full": run(full / "rasters", args.work / "full", met, design["date"], n), "halos": {}}
    print(f"[local] tree {i} at {cy},{cx}; full rerun {n} px: sky view {res['full']['svf_s']:.0f} s, "
          f"day {res['full']['day_s']:.0f} s", flush=True)
    for h in args.halos:
        half = core + int(round(h / res_m))
        r0, r1, c0, c1 = max(cy - half, 0), min(cy + half + 1, n), max(cx - half, 0), min(cx + half + 1, n)
        win = args.work / f"halo_{int(h)}m"
        (win / "rasters").mkdir(parents=True, exist_ok=True)
        for name in NAMES:
            gdal.Translate(str(win / "rasters" / f"{name}.tif"), str(full / "rasters" / f"{name}.tif"),
                           srcWin=[c0, r0, c1 - c0, r1 - r0])
        t = run(win / "rasters", win / "run", met, design["date"], max(r1 - r0, c1 - c0))
        loc = {"svf": gdal.Open(str(win / "run/shared/processed_inputs/SVF/SkyViewFactor_0_0.tif")).ReadAsArray(),
               "Shadow": read_stack(win / "run" / "out", "Shadow"), "TMRT": read_stack(win / "run" / "out", "TMRT")}
        ys, xs = slice(cy - core, cy + core + 1), slice(cx - core, cx + core + 1)
        ly, lx = slice(cy - core - r0, cy + core + 1 - r0), slice(cx - core - c0, cx + core + 1 - c0)
        cmp = {}
        for k in ("svf", "Shadow", "TMRT"):
            a = ref[k][..., ys, xs]
            b = loc[k][..., ly, lx]
            cmp[k] = {"identical": bool(np.array_equal(a, b, equal_nan=True)),
                      "max_abs": float(np.nanmax(np.abs(a.astype(np.float64) - b)))}
        res["halos"][int(h)] = {"window_px": [r1 - r0, c1 - c0], **t, **cmp}
        verdict = lambda v: "identical" if v["identical"] else "max |d| %.3g" % v["max_abs"]  # noqa: E731
        print(f"[local] halo {h:4.0f} m, window {r1 - r0}x{c1 - c0} px: sky view {t['svf_s']:.0f} s, day {t['day_s']:.0f} s; "
              + ", ".join(f"{k} {verdict(v)}" for k, v in cmp.items()), flush=True)
        (args.out / "local_rerun.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
