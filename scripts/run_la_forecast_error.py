"""Street-scale skill against forecast error: how much air-temperature error can a heat-stress map take?

The skill front of a forecast: its skill falls at finer length scales and longer lead times.
SOLWEIG takes air temperature as one value per hour for the whole domain, and it is the input that
moves UTCI most (Morris; regression on the Morris runs: 62-80% of the variance of mean UTCI). So
for street-scale heat stress the question is how an air-temperature forecast error of a given size
(a stand-in for lead time; the mapping to lead times comes from published HRRR verification, not
from this script) degrades threshold maps, and down to which spatial scale they stay skillful.

Perfect-model ensemble on Pacoima (1000 x 1000 at 2 m, 30 July 2019), the structure-fixed model of
run_la_ensemble.py --structure fitted (ground heating fitted to the measured surface temperatures,
heat kept after sunset, tau 4 h), C1-C9 for speed (bit-identical). Members: the hourly air
temperature plus an error e(t), AR(1) in time with lag-1 correlation RHO and standard deviation
sigma (forecast errors persist over hours); the same random numbers for every sigma, scaled by it,
so levels differ only in error size. Relative humidity is kept as forced (a simplification:
humidity errors are left out here, so the spread is a lower bound). One unperturbed run.

Per run: hours of UTCI above 32 and 38 C per pixel (strong and very strong heat stress), the share
of outdoor area above each threshold every hour, mean outdoor UTCI every hour, UTCI at 15:00.
analyse_la_forecast_error.py scores members against each other (fractions skill score by
neighbourhood size, Roberts & Lean 2008) per sigma.

Usage:
    python scripts/run_la_forecast_error.py run --worker 0 --workers 2
    python scripts/run_la_forecast_error.py run --work ~/forecast_error_expo --src <processed inputs>         --rasters data/la/expo_park                      (another tile, e.g. the Exposition Park venue)
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

WORK = Path.home() / "forecast_error_la"
SRC = Path.home() / "solweig_bench" / "processed_inputs"  # Pacoima 1000 px with its sky view cache
RASTERS = ROOT / "data" / "la" / "pacoima"
DATE = "2019-07-30"
SIGMAS = (0.5, 1.0, 1.5, 2.0, 3.0)
MEMBERS = 20
RHO = 0.9
THRESHOLDS = (32.0, 38.0)
SEED = 0


def design() -> list[dict]:
    """Every run: (sigma, member) and its hourly error series; member -1 is the unperturbed run."""
    rng = np.random.default_rng(SEED)
    z = rng.standard_normal((MEMBERS, 24))
    unit = np.zeros_like(z)  # AR(1) with unit standard deviation, stationary from the first hour
    unit[:, 0] = z[:, 0]
    for t in range(1, 24):
        unit[:, t] = RHO * unit[:, t - 1] + np.sqrt(1 - RHO ** 2) * z[:, t]
    runs = [{"id": "base", "sigma": 0.0, "member": -1, "error": [0.0] * 24}]
    for s in SIGMAS:
        for m in range(MEMBERS):
            runs.append({"id": f"s{s:g}_m{m:02d}", "sigma": s, "member": m, "error": (s * unit[m]).tolist()})
    return runs


@contextlib.contextmanager
def ta_error(metfile_dir: Path, error: list[float]):
    """Add error[hour] to the air temperature of every per-tile met file; restore afterwards."""
    from solweig_sens.perturb import MET_COLS, _read_met, _write_met

    backup = metfile_dir.parent / (metfile_dir.name + "_orig")
    if backup.exists():  # a killed run left its backup: restore first
        shutil.rmtree(metfile_dir)
        shutil.copytree(backup, metfile_dir)
        shutil.rmtree(backup)
    shutil.copytree(metfile_dir, backup)
    try:
        for f in sorted(metfile_dir.glob("*.txt")):
            header, data = _read_met(f)
            hour = data[:, 2].astype(int) % 24  # UMEP columns: iy, id, it (hour), imin, ...
            col = MET_COLS["ta"]
            valid = data[:, col] != -999.0
            data[valid, col] = data[valid, col] + np.asarray(error)[hour[valid]]
            _write_met(f, header, data)
        yield
    finally:
        shutil.rmtree(metfile_dir)
        shutil.copytree(backup, metfile_dir)
        shutil.rmtree(backup)


def run(args) -> None:
    global WORK, SRC, RASTERS, DATE
    WORK, SRC, RASTERS, DATE = args.work, args.src, args.rasters, args.date
    from osgeo import gdal
    from run_la_pavement import TS_FIT
    from solweig_gpu import run_utci_tiles
    from solweig_sens.analysis import read_stack
    from solweig_sens.compact_masks import compact_masks
    from solweig_sens.fast import fast_solweig
    from solweig_sens.heat_storage import ground_heat_storage
    from solweig_sens.hoist_geometry import hoist_geometry
    from solweig_sens.perturb import PAVED_CLASSES, patched_landcover
    from solweig_sens.tight_bound import tight_bound

    runs = design()
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "design.json").write_text(json.dumps({"sigmas": SIGMAS, "members": MEMBERS, "rho": RHO, "seed": SEED,
                                                 "thresholds": THRESHOLDS, "date": DATE, "structure": args.structure,
                                                 "src": str(SRC).replace(str(Path.home()), "~"), "runs": runs},
                                                indent=1))
    mine = WORK / f"worker_{args.worker}"
    pre = mine / "processed_inputs"
    if not pre.exists():
        shutil.copytree(SRC, pre)
    (WORK / "runs").mkdir(exist_ok=True)
    (WORK / "maps").mkdir(exist_ok=True)
    lc = gdal.Open(str(RASTERS / "Landcover.tif")).ReadAsArray()
    todo = [r for i, r in enumerate(runs) if i % args.workers == args.worker
            and not (WORK / "runs" / f"{r['id']}.json").exists()]
    t0 = time.perf_counter()
    for k, r in enumerate(todo):
        tmp = mine / "tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        with contextlib.ExitStack() as st:
            for c in (fast_solweig(), compact_masks("bits", svf=False, packed_cache=True), hoist_geometry(),
                      tight_bound()):
                st.enter_context(c)
            st.enter_context(ta_error(pre / "metfiles", r["error"]))
            if args.structure == "fitted":  # Pacoima's measured asphalt; does not transfer to other sites
                st.enter_context(patched_landcover(tmp, PAVED_CLASSES, {"ts_deg": TS_FIT}))
                st.enter_context(ground_heat_storage(4.0))
            run_utci_tiles(base_path=str(tmp), preprocess_dir=str(pre), selected_date_str=DATE, save_tmrt=False)
        utci = read_stack(tmp, "UTCI")
        outdoor = (lc[: utci.shape[1], : utci.shape[2]] != 2) & np.isfinite(utci).all(axis=0)
        rec = {k2: r[k2] for k2 in ("id", "sigma", "member", "error")}
        rec["utci_mean"] = [float(np.nanmean(utci[h][outdoor])) for h in range(utci.shape[0])]
        maps = {}
        for t in THRESHOLDS:
            above = utci > t
            rec[f"area_above_{t:g}_pct"] = [float(above[h][outdoor].mean() * 100) for h in range(utci.shape[0])]
            maps[f"hours_above_{t:g}"] = np.where(outdoor, above.sum(axis=0), 255).astype(np.uint8)
        maps["utci_15h"] = utci[15].astype(np.float16)
        np.savez_compressed(WORK / "maps" / f"{r['id']}.npz", **maps)
        (WORK / "runs" / f"{r['id']}.json").write_text(json.dumps(rec))
        shutil.rmtree(tmp, ignore_errors=True)
        if k % 10 == 0:
            rate = (time.perf_counter() - t0) / (k + 1)
            print(f"[fcst] worker {args.worker}: {k + 1}/{len(todo)} runs, {rate:.0f} s each", flush=True)
    print(f"[fcst] worker {args.worker} done", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--work", type=Path, default=WORK)
    ap.add_argument("--src", type=Path, default=SRC, help="preprocessed tile with its sky view cache")
    ap.add_argument("--rasters", type=Path, default=RASTERS, help="the tile's rasters (land cover)")
    ap.add_argument("--date", default=DATE)
    ap.add_argument("--structure", choices=["fitted", "default"], default="fitted",
                    help="fitted: ground heating fitted to the Pacoima measurements and heat storage; "
                         "default: SOLWEIG as is (for other sites)")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
