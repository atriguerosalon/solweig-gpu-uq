"""Cool-pavement test in Pacoima, Los Angeles, against the measurements of Middel et al. (2020).

Middel et al. (2020, Environ. Res. Lett. 15, 064016) measured mean radiant temperature
(Tmrt) over solar-reflective coated asphalt and regular asphalt in Pacoima on 30 July 2019
with the MaRTy cart. In Pacoima: albedo 0.25 coated versus about 0.07 regular; coated
surfaces 4.2 to 5.8 K cooler at midday; Tmrt over coated pavement 2.6 to 4.2 K higher from
11:30 to 15:30 local standard time, about 1 K lower at 20:00.

SOLWEIG's ground temperature depends on land-cover class and sun altitude, not on albedo.
So changing albedo alone changes reflected shortwave but not emitted longwave. Four runs
separate the two:

    regular_default  albedo 0.07, default asphalt ground heating (ts_deg 0.58)
    coated_default   albedo 0.25, default ground heating           -> SOLWEIG as is
    regular_fit      albedo 0.07, ground heating fitted to the measured asphalt peak (55.9 C)
    coated_fit       albedo 0.25, fitted heating minus the measured coating effect (about 5.3 K)

Usage:
    python scripts/run_la_pavement.py --rasters data/la/pacoima --met data/la/met_pacoima_20190730.txt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solweig_sens.analysis import is_complete  # noqa: E402
from solweig_sens.perturb import PAVED_CLASSES, patched_landcover  # noqa: E402
from solweig_sens.run import Setup, prepare  # noqa: E402

# Ground heating: surface minus air temperature at its daily peak = ts_deg * altmax + tstart.
# Pacoima, 30 July: max sun altitude about 74.1 degrees; asphalt tstart = -9.78.
ALTMAX, TSTART = 74.1, -9.78
TA_PEAK, TS_ASPHALT_PEAK, TS_COATING = 31.0, 55.9, -5.3
TS_FIT = round((TS_ASPHALT_PEAK - TA_PEAK - TSTART) / ALTMAX, 3)          # about 0.47
TS_FIT_COATED = round(TS_FIT + TS_COATING / ALTMAX, 3)                       # about 0.40

CASES = {
    "regular_default": {"albedo": 0.07, "ts_deg": 0.58},
    "coated_default": {"albedo": 0.25, "ts_deg": 0.58},
    "regular_fit": {"albedo": 0.07, "ts_deg": TS_FIT},
    "coated_fit": {"albedo": 0.25, "ts_deg": TS_FIT_COATED},
}


def main() -> None:
    from solweig_gpu import run_utci_tiles

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rasters", type=Path, required=True)
    ap.add_argument("--met", type=Path, required=True)
    ap.add_argument("--runs", type=Path, default=Path("runs_la"))
    ap.add_argument("--date", default="2019-07-30")
    args = ap.parse_args()
    runs = args.runs.resolve()
    setup = Setup(raster_dir=args.rasters.resolve(), met_file=args.met.resolve(), runs_dir=runs, date=args.date)
    svf_dir = setup.preprocess_dir / "SVF"
    if not (svf_dir.exists() and any(svf_dir.glob("SkyViewFactor_*.tif"))):  # an interrupted run leaves it empty
        prepare(setup)
    (runs / "cases.json").write_text(json.dumps(CASES, indent=2))

    timings = {}
    for name, values in CASES.items():
        out_dir = runs / name
        if is_complete(out_dir):
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        with patched_landcover(out_dir, PAVED_CLASSES, values):
            run_utci_tiles(base_path=str(out_dir), preprocess_dir=str(setup.preprocess_dir),
                           selected_date_str=setup.date, save_tmrt=True,
                           save_shadow=name == "regular_default", save_svf=name == "regular_default")
        timings[name] = time.perf_counter() - t0
        print(f"[la] {name} done in {timings[name] / 60:.1f} min", flush=True)
    (runs / "timings.json").write_text(json.dumps(timings, indent=2))


if __name__ == "__main__":
    main()
