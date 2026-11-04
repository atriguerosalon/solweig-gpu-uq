"""Does a linear emulator fitted on one day hold on another? Pacoima, two very different days.

    2019-07-30  the Middel et al. measurement day (Van Nuys max 31 C)
    2020-09-06  San Fernando Valley record heat (Van Nuys max 48 C, 7% humidity)

On each day, on the same 0.8 x 0.8 km window: the 100-run Morris design of run_morris.py (same
seed, same input ranges) and 50 independent Latin hypercube samples, up to 14:00. The Latin
hypercube runs are the test set for emulators fitted on the Morris runs, of the same day and of
the other day (analyse_la_emulator_days.py).

Usage:
    python scripts/run_la_emulator_days.py --rasters data/la/pacoima
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_morris import PROBLEM, crop, trim_met  # noqa: E402
from solweig_sens.analysis import read_stack  # noqa: E402
from solweig_sens.perturb import applied_many  # noqa: E402
from solweig_sens.run import Setup, prepare  # noqa: E402

DAYS = {"2019-07-30": "data/la/met_pacoima_20190730.txt", "2020-09-06": "data/la/met_pacoima_20200906.txt"}
HOUR, SIZE, N_LHS = 14, 400, 50


def designs(root: Path) -> dict[str, np.ndarray]:
    from SALib.sample import morris as morris_sample
    from scipy.stats import qmc

    b = np.array(PROBLEM["bounds"])
    X = {"morris": morris_sample.sample(PROBLEM, N=10, num_levels=4, seed=0),
         "lhs": b[:, 0] + qmc.LatinHypercube(d=len(b), seed=7).random(N_LHS) * (b[:, 1] - b[:, 0])}
    for k, v in X.items():
        np.save(root / f"X_{k}.npy", v)
    (root / "problem.json").write_text(json.dumps(PROBLEM, indent=2))
    return X


def main() -> None:
    from solweig_gpu import run_utci_tiles
    from solweig_sens.compact_masks import compact_masks
    from solweig_sens.fast import fast_solweig

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rasters", type=Path, required=True)
    ap.add_argument("--runs", type=Path, default=Path("runs_la_emu"))
    args = ap.parse_args()
    root = args.runs.resolve()
    root.mkdir(parents=True, exist_ok=True)
    X = designs(root)

    for day, met_src in DAYS.items():
        runs = root / day
        raster_dir = runs / "rasters"
        if not (raster_dir / "Landcover.tif").exists():
            crop(args.rasters.resolve(), raster_dir, SIZE)
        met = runs / f"met_0to{HOUR:02d}.txt"
        trim_met(Path(met_src).resolve(), met, HOUR)
        setup = Setup(raster_dir=raster_dir, met_file=met, runs_dir=runs, date=day, tile_size=SIZE)
        if not (setup.preprocess_dir / "SVF").exists() or not any((setup.preprocess_dir / "SVF").glob("Sky*.tif")):
            with compact_masks("bits", svf=True):
                prepare(setup)
        fields = runs / "fields"
        fields.mkdir(exist_ok=True)
        jobs = [("baseline", None)] + [(f"{k}_{i:03d}", dict(zip(PROBLEM["names"], map(float, x))))
                                       for k, xs in X.items() for i, x in enumerate(xs)]
        t_start, done = time.perf_counter(), 0
        for name, sample in jobs:
            target = fields / f"{name}.npy"
            if target.exists():
                continue
            tmp = runs / "tmp"
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir()
            kwargs = dict(base_path=str(tmp), preprocess_dir=str(setup.preprocess_dir), selected_date_str=day,
                          save_tmrt=True, save_shadow=sample is None)
            with fast_solweig(), compact_masks("bits", svf=False):
                if sample is None:
                    run_utci_tiles(**kwargs)
                else:
                    with applied_many(sample, tmp, setup.preprocess_dir / "metfiles"):
                        run_utci_tiles(**kwargs)
            if sample is None:
                np.save(fields / "shadow.npy", read_stack(tmp, "Shadow")[HOUR])
            np.save(fields / f"{name}_tmrt.npy", read_stack(tmp, "TMRT")[HOUR])
            np.save(target, read_stack(tmp, "UTCI")[HOUR])
            shutil.rmtree(tmp)
            done += 1
            left = sum(not (fields / f"{n}.npy").exists() for n, _ in jobs)
            rate = (time.perf_counter() - t_start) / done
            print(f"[emu] {day} {name} done, {left} left, ~{rate * left / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
