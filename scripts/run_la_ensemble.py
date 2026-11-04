"""Does SOLWEIG's input uncertainty cover the Pacoima measurements? An ensemble for bias and spread.

Uncertain inputs and their ranges (RANGES below, each with its source) are sampled by Latin
hypercube. Body shortwave absorption is not among them: the measured Tmrt is defined with
0.70 (Middel et al. 2020), which is SOLWEIG's value, so for this comparison it is a definition,
not an uncertainty. Each sample runs twice, over regular and coated asphalt, as in
run_la_pavement.py, full day on the Pacoima tile with SOLWEIG's default ground heating. Paired
runs give the coating effect per sample. The ranges are centred on what a user of the station
and default parameters would assume; the measured mean offset of the station (street +0.65 K)
is deliberately left out, so that it shows up as bias, not spread.

Per run only hourly statistics are kept, on the ground the measurements of Middel et al.
(2020) stand for (open: no building, no tree, sky view factor above 0.75; sunlit 07-19 h):
25th/50th/75th percentile of Tmrt and UTCI. analyse_la_ensemble.py compares the ensemble
with the measurements: bias = ensemble median minus measurement, spread = ensemble interval.

Usage:
    python scripts/run_la_ensemble.py --rasters data/la/pacoima --met data/la/met_pacoima_20190730.txt
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from solweig_sens.analysis import read_stack  # noqa: E402
from solweig_sens.run import Setup  # noqa: E402

RANGES = {
    # Street minus Van Nuys station air temperature at the seven Pacoima stops (Middel et al. 2020,
    # table 1, against our met file): +0.65 +- 1.43 K, range -2.3..+2.0. Uniform +-2.5 has std 1.44.
    "ta_add": (-2.5, 2.5),
    # Not measured on the street; vapour pressure varies less than temperature, so RH follows Ta:
    # 1.4 K at RH 40% is about 3.5 points; plus station hygrometer accuracy (a few %).
    "rh_add": (-6.0, 6.0),
    # Street wind below 1.6 m/s at 1.1 m (Middel et al. 2020), about 2.4 m/s at 10 m with UTCI's log
    # profile, against 2.3-5.7 m/s at Van Nuys in the afternoon; GLIDE-SOL found ERA5 wind 2.2 m/s too
    # strong at street level. Wind does not enter Tmrt, only UTCI.
    "wind_scale": (0.3, 1.0),
    # ERA5 global radiation: hourly peaks off by up to 50-100 W/m2 at some sites; +-10% (assumed).
    "kdown_scale": (0.9, 1.1),
    # Asphalt emissivity: 0.9 typical (LBNL), most building and paving materials 0.90-0.98.
    "paved_emissivity": (0.90, 0.98),
    # Walls: no survey of LA walls; 0.10 is the LA climate-modelling baseline (Zhang et al. 2018),
    # light stucco/paint about 0.5 (LBNL cool walls). Weakest-sourced range here.
    "albedo_b": (0.10, 0.50),
    "ewall": (0.85, 0.95),
    # Pavement albedo as measured (Middel et al. 2020): regular 0.06-0.08, coated 0.25 (+-0.02 assumed).
    "regular_albedo": (0.06, 0.08),
    "coated_albedo": (0.23, 0.27),
}
NAMES = list(RANGES)
BOUNDS = np.array(list(RANGES.values()))
PAVEMENT = ("regular", "coated")


def open_sunlit(runs_la: Path, rasters: Path) -> np.ndarray:
    """(hours, rows, cols) mask: the open ground of analyse_la_pavement.py, sunlit by day."""
    from osgeo import gdal

    shadow = read_stack(runs_la / "regular_default", "Shadow")
    shape = shadow.shape[1:]
    svf = gdal.Open(str(runs_la / "shared/processed_inputs/SVF/SkyViewFactor_0_0.tif")).ReadAsArray()[: shape[0], : shape[1]]
    lc = gdal.Open(str(rasters / "Landcover.tif")).ReadAsArray()[: shape[0], : shape[1]]
    trees = gdal.Open(str(rasters / "Trees.tif")).ReadAsArray()[: shape[0], : shape[1]]
    open_ground = (lc != 2) & (trees <= 0) & (svf > 0.75)
    hours = np.arange(shadow.shape[0])[:, None, None]
    return open_ground[None] & ((shadow > 0.5) | (hours < 7) | (hours > 19))


def main() -> None:
    from scipy.stats import qmc
    from solweig_gpu import run_utci_tiles
    from solweig_sens.compact_masks import compact_masks
    from solweig_sens.fast import fast_solweig
    from run_la_pavement import TS_FIT, TS_FIT_COATED
    from solweig_sens.heat_storage import ground_heat_storage
    from solweig_sens.perturb import PAVED_CLASSES, applied_many, patched_landcover
    from solweig_sens.tight_bound import tight_bound

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rasters", type=Path, required=True)
    ap.add_argument("--met", type=Path, required=True)
    ap.add_argument("--runs-la", type=Path, default=Path("runs_la"), help="run_la_pavement.py output (shared cache)")
    ap.add_argument("--out", type=Path, default=Path("runs_la_ens"))
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--date", default="2019-07-30")
    ap.add_argument("--structure", choices=["default", "fitted"], default="default",
                    help="fitted: ground heating fitted to the measured surface temperatures (run_la_pavement.py) "
                         "and ground heat kept after sunset (heat_storage.py, --tau)")
    ap.add_argument("--tau", type=float, default=4.0, help="heat storage time constant (h), with --structure fitted")
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    X = qmc.scale(qmc.LatinHypercube(d=len(NAMES), seed=args.seed).random(args.n), BOUNDS[:, 0], BOUNDS[:, 1])
    np.save(out / "X.npy", X)
    (out / "design.json").write_text(json.dumps({"names": NAMES, "bounds": BOUNDS.tolist(), "n": args.n,
                                                 "seed": args.seed, "structure": args.structure,
                                                 "tau_h": args.tau if args.structure == "fitted" else None}, indent=2))
    setup = Setup(raster_dir=args.rasters.resolve(), met_file=args.met.resolve(), runs_dir=args.runs_la.resolve(),
                  date=args.date)
    mask = open_sunlit(args.runs_la.resolve(), args.rasters.resolve())

    for i, x in enumerate(X):
        for case in PAVEMENT:
            target = out / f"{case}_{i:03d}.json"
            if target.exists():
                continue
            tmp = out / "tmp"
            if tmp.exists():
                shutil.rmtree(tmp)
            tmp.mkdir()
            v = dict(zip(NAMES, map(float, x)))
            sample = {k: v[k] for k in NAMES if k not in ("regular_albedo", "coated_albedo")}
            sample["paved_albedo"] = v[f"{case}_albedo"]
            t0 = time.perf_counter()
            with contextlib.ExitStack() as st:
                for c in (fast_solweig(), compact_masks("bits", svf=False), tight_bound()):
                    st.enter_context(c)
                if args.structure == "default":
                    st.enter_context(applied_many(sample, tmp, setup.preprocess_dir / "metfiles"))
                else:  # one land-cover table with the sampled albedo and emissivity and the fitted heating
                    paved = {"albedo": sample.pop("paved_albedo"), "emissivity": sample.pop("paved_emissivity"),
                             "ts_deg": TS_FIT if case == "regular" else TS_FIT_COATED}
                    st.enter_context(applied_many(sample, tmp, setup.preprocess_dir / "metfiles"))  # met, walls, body
                    st.enter_context(patched_landcover(tmp, PAVED_CLASSES, paved))
                    st.enter_context(ground_heat_storage(args.tau))
                    sample |= {f"paved_{k}": val for k, val in paved.items()}
                run_utci_tiles(base_path=str(tmp), preprocess_dir=str(setup.preprocess_dir),
                               selected_date_str=args.date, save_tmrt=True)
            rec = {"sample": sample}
            for var in ("TMRT", "UTCI"):
                f = read_stack(tmp, var)
                rec[var] = [np.percentile(f[h][mask[h]], [25, 50, 75]).tolist() for h in range(f.shape[0])]
            target.write_text(json.dumps(rec))
            shutil.rmtree(tmp)
            print(f"[ens] {case} {i:03d} in {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
