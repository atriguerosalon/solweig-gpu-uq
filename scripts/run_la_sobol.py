"""Sobol pilot: which uncertain inputs drive street-scale heat stress in Pacoima, also above thresholds?

Morris (run_morris.py) ranks inputs; Sobol indices give each input's share of the output variance:
first order (the input alone) and total (with its interactions). On the Austin Morris runs a linear
fit already explains all the variance of mean UTCI, but only 82-98% of the share of area above
38 C (docs/internal/night_2026-11-02.md), so thresholds are where the interactions are.

Design: Sobol' sequence with Saltelli's scheme (SALib), N base samples, k inputs, first and total
order only: N (k + 2) runs. Inputs and ranges: RANGES of run_la_ensemble.py (each with its source),
regular asphalt only (k = 8). Model: the structure-fixed one of run_la_ensemble.py --structure
fitted (ground heating fitted to the measured surface temperatures, heat kept after sunset with a
4 h time constant), so the spread decomposed here belongs to a model whose daytime bias against
the Pacoima measurements is about one spread. Domain: centred 500 x 500 crop of the Pacoima tile
(1 x 1 km at 2 m), 30 July 2019, full day. Speed: C1-C8 and the packed mask cache (all
bit-identical to v2.0.0).

Per run, hourly: outdoor mean UTCI; share of outdoor area with UTCI above 32, 38 and 46 C (strong,
very strong, extreme heat stress); median Tmrt on open sunlit ground; and the UTCI field at 13, 16
and 21 h (float16) for maps of the indices. analyse_la_sobol.py computes the indices.

    prepare   crop, preprocess and sky view factor once; write the design
    run       one worker (--worker i of --workers n), on its own copy of the inputs, since the
              met-file perturbation edits them in place; resumes, skipping finished runs

Usage:
    python scripts/run_la_sobol.py prepare --n 128
    python scripts/run_la_sobol.py run --worker 0 --workers 2
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

WORK = Path.home() / "sobol_la"
SIZE = 500
DATE = "2019-07-30"
MAP_HOURS = (13, 16, 21)
THRESHOLDS = (32.0, 38.0, 46.0)


def problem() -> dict:
    from run_la_ensemble import RANGES

    names = [k for k in RANGES if k != "coated_albedo"]
    return {"num_vars": len(names), "names": names, "bounds": [list(RANGES[k]) for k in names]}


def prepare(args) -> None:
    from bound_test import centred_crop
    from solweig_sens.compact_masks import compact_masks
    from solweig_sens.fast_sky_shadow import fast_sky_shadow
    from solweig_sens.run import Setup, prepare as prepare_tile
    from solweig_sens.tight_bound import tight_bound

    try:
        from SALib.sample import sobol as sampler
    except ImportError:  # older SALib
        from SALib.sample import saltelli as sampler
    WORK.mkdir(parents=True, exist_ok=True)
    raster = WORK / "rasters"
    centred_crop((ROOT / "data" / "la" / "pacoima").resolve(), raster, SIZE)
    setup = Setup(raster_dir=raster, met_file=(ROOT / "data" / "la" / "met_pacoima_20190730.txt").resolve(),
                  runs_dir=WORK / "base", date=DATE, tile_size=SIZE)
    with compact_masks("bits", svf=True), fast_sky_shadow(), tight_bound():
        prepare_tile(setup)
    p = problem()
    X = sampler.sample(p, args.n, calc_second_order=False, seed=args.seed)
    np.save(WORK / "X.npy", X)
    (WORK / "problem.json").write_text(json.dumps(p | {"n_base": args.n, "seed": args.seed, "runs": len(X),
                                                       "size_px": SIZE, "date": DATE, "structure": "fitted, tau 4 h",
                                                       "map_hours": MAP_HOURS, "thresholds": THRESHOLDS}, indent=2))
    (WORK / "preprocess_dir.txt").write_text(str(setup.preprocess_dir))
    print(f"[sobol] prepared: {len(X)} runs ({args.n} x ({p['num_vars']} + 2)), inputs {p['names']}", flush=True)


def masks(pre: Path, raster: Path):
    from osgeo import gdal

    lc = gdal.Open(str(raster / "Landcover.tif")).ReadAsArray()
    trees = gdal.Open(str(raster / "Trees.tif")).ReadAsArray()
    svf = gdal.Open(str(next((pre / "SVF").glob("SkyViewFactor_*.tif")))).ReadAsArray()
    r, c = svf.shape
    lc, trees = lc[:r, :c], trees[:r, :c]
    return lc != 2, (lc != 2) & (trees <= 0) & (svf > 0.75)


def run(args) -> None:
    from run_la_pavement import TS_FIT
    from solweig_gpu import run_utci_tiles
    from solweig_sens.analysis import read_stack
    from solweig_sens.compact_masks import compact_masks
    from solweig_sens.fast import fast_solweig
    from solweig_sens.heat_storage import ground_heat_storage
    from solweig_sens.hoist_geometry import hoist_geometry
    from solweig_sens.perturb import PAVED_CLASSES, applied_many, patched_landcover
    from solweig_sens.tight_bound import tight_bound

    p = json.loads((WORK / "problem.json").read_text())
    X = np.load(WORK / "X.npy")
    shared = Path((WORK / "preprocess_dir.txt").read_text())
    mine = WORK / f"worker_{args.worker}"
    pre = mine / "processed_inputs"
    if not pre.exists():
        shutil.copytree(shared, pre)
    outdoor, open_ground = masks(pre, WORK / "rasters")
    (WORK / "runs").mkdir(exist_ok=True)
    (WORK / "fields").mkdir(exist_ok=True)
    todo = [i for i in range(len(X)) if i % args.workers == args.worker and not (WORK / "runs" / f"{i:05d}.json").exists()]
    t_start = time.perf_counter()
    for n_done, i in enumerate(todo):
        v = dict(zip(p["names"], map(float, X[i])))
        sample = {k: v[k] for k in ("ta_add", "rh_add", "wind_scale", "kdown_scale", "albedo_b", "ewall")}
        paved = {"albedo": v["regular_albedo"], "emissivity": v["paved_emissivity"], "ts_deg": TS_FIT}
        tmp = mine / "tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        with contextlib.ExitStack() as st:
            for c in (fast_solweig(), compact_masks("bits", svf=False, packed_cache=True), hoist_geometry(),
                      tight_bound()):
                st.enter_context(c)
            st.enter_context(applied_many(sample, tmp, pre / "metfiles"))  # met, walls
            st.enter_context(patched_landcover(tmp, PAVED_CLASSES, paved))
            st.enter_context(ground_heat_storage(4.0))
            run_utci_tiles(base_path=str(tmp), preprocess_dir=str(pre), selected_date_str=DATE, save_tmrt=True,
                           save_shadow=True)
        utci, tmrt, shadow = (read_stack(tmp, var) for var in ("UTCI", "TMRT", "Shadow"))
        hours = range(utci.shape[0])
        rec = {"x": X[i].tolist(),
               "utci_mean": [float(np.nanmean(utci[h][outdoor])) for h in hours],
               "tmrt_open_sunlit_median": [float(np.nanmedian(tmrt[h][open_ground & ((shadow[h] > 0.5) | (h < 7) | (h > 19))]))
                                           for h in hours]}
        for t in THRESHOLDS:
            rec[f"area_utci_above_{t:g}_pct"] = [float((utci[h][outdoor] > t).mean() * 100) for h in hours]
        np.save(WORK / "fields" / f"{i:05d}.npy", utci[list(MAP_HOURS)].astype(np.float16))
        (WORK / "runs" / f"{i:05d}.json").write_text(json.dumps(rec))
        shutil.rmtree(tmp, ignore_errors=True)
        if n_done % 20 == 0:
            rate = (time.perf_counter() - t_start) / (n_done + 1)
            print(f"[sobol] worker {args.worker}: {n_done + 1}/{len(todo)} runs, {rate:.1f} s each, "
                  f"~{rate * (len(todo) - n_done - 1) / 60:.0f} min left", flush=True)
    print(f"[sobol] worker {args.worker} done", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--n", type=int, default=128, help="base samples (a power of 2)")
    a.add_argument("--seed", type=int, default=0)
    b = sub.add_parser("run")
    b.add_argument("--worker", type=int, default=0)
    b.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    {"prepare": prepare, "run": run}[args.cmd](args)


if __name__ == "__main__":
    main()
