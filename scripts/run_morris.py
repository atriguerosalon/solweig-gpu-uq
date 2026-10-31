"""Morris screening of nine uncertain inputs, on a small window and up to one hour.

Changes all inputs together along random trajectories (Morris 1991, with the
mu* measure of Campolongo et al. 2007), so it also sees interactions and
nonlinear effects that one-at-a-time runs miss. Each run's UTCI field at the
target hour is saved, so the screening can also be mapped per pixel.

Ground and wall temperatures carry state from hour to hour, so every run starts
at midnight and stops at the target hour.

Usage:
    python scripts/run_morris.py --rasters data/austin_crop --met data/ownmet_Forcing_data.txt
The run resumes where it stopped if interrupted.
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

from crop_domain import NAMES, densest_window  # noqa: E402
from solweig_sens.analysis import UTCI_VERY_STRONG, outdoor_mask, read_stack  # noqa: E402
from solweig_sens.perturb import applied_many  # noqa: E402
from solweig_sens.run import Setup, prepare  # noqa: E402

PROBLEM = {
    "num_vars": 9,
    "names": ["albedo_b", "ewall", "absK", "paved_albedo", "paved_emissivity",
              "ta_add", "rh_add", "wind_scale", "kdown_scale"],
    "labels": ["wall albedo", "wall emissivity", "body shortwave absorption", "paved albedo",
               "paved emissivity", "air temperature", "relative humidity", "wind speed", "global radiation"],
    "bounds": [[0.10, 0.40], [0.85, 0.95], [0.60, 0.80], [0.10, 0.35], [0.90, 0.98],
               [-2.0, 2.0], [-15.0, 15.0], [0.5, 1.5], [0.9, 1.1]],
}


def crop(src: Path, dst: Path, size: int) -> None:
    from osgeo import gdal

    gdal.UseExceptions()
    dst.mkdir(parents=True, exist_ok=True)
    dsm = gdal.Open(str(src / "Building_DSM.tif")).ReadAsArray().astype(np.float32)
    dem = gdal.Open(str(src / "DEM.tif")).ReadAsArray().astype(np.float32)
    r, c = densest_window(dsm, dem, size, step=25)
    for name in NAMES:
        gdal.Translate(str(dst / f"{name}.tif"), str(src / f"{name}.tif"), srcWin=[c, r, size, size])


def trim_met(src: Path, dst: Path, last_hour: int) -> None:
    lines = src.read_text().splitlines()
    dst.write_text("\n".join(lines[: last_hour + 2]) + "\n")  # header + hours 0..last_hour


def run_once(setup: Setup, out_dir: Path, sample: dict | None, save_shadow: bool = False) -> None:
    from solweig_gpu import run_utci_tiles

    out_dir.mkdir(parents=True, exist_ok=True)
    kwargs = dict(base_path=str(out_dir), preprocess_dir=str(setup.preprocess_dir),
                  selected_date_str=setup.date, save_tmrt=False, save_shadow=save_shadow)
    if sample is None:
        run_utci_tiles(**kwargs)
    else:
        with applied_many(sample, out_dir, setup.preprocess_dir / "metfiles"):
            run_utci_tiles(**kwargs)


def main() -> None:
    from SALib.sample import morris as morris_sample

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rasters", type=Path, required=True)
    ap.add_argument("--met", type=Path, required=True)
    ap.add_argument("--runs", type=Path, default=Path("runs_morris"))
    ap.add_argument("--size", type=int, default=400)
    ap.add_argument("--hour", type=int, default=14)
    ap.add_argument("--trajectories", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all-hours", action="store_true",
                    help="also save every hour up to --hour (float16, run_XXX_hours.npy)")
    args = ap.parse_args()
    runs = args.runs.resolve()
    runs.mkdir(parents=True, exist_ok=True)

    raster_dir = runs / "rasters"
    if not (raster_dir / "Landcover.tif").exists():
        crop(args.rasters.resolve(), raster_dir, args.size)
    met = runs / f"met_0to{args.hour:02d}.txt"
    trim_met(args.met.resolve(), met, args.hour)

    setup = Setup(raster_dir=raster_dir, met_file=met, runs_dir=runs, tile_size=args.size)
    if not setup.preprocess_dir.exists():
        prepare(setup)

    X = morris_sample.sample(PROBLEM, N=args.trajectories, num_levels=4, seed=args.seed)
    np.save(runs / "X.npy", X)
    (runs / "problem.json").write_text(json.dumps(PROBLEM, indent=2))

    fields = runs / "fields"
    fields.mkdir(exist_ok=True)
    if not (fields / "baseline.npy").exists():
        tmp = runs / "tmp_baseline"
        run_once(setup, tmp, None, save_shadow=True)
        utci = read_stack(tmp, "UTCI")
        np.save(fields / "baseline.npy", utci[args.hour])
        np.save(fields / "shadow.npy", read_stack(tmp, "Shadow")[args.hour])
        if args.all_hours:
            np.save(fields / "baseline_hours.npy", utci.astype(np.float16))
            np.save(fields / "shadow_hours.npy", read_stack(tmp, "Shadow").astype(np.float16))
        shutil.rmtree(tmp)

    t_start = time.perf_counter()
    for i, x in enumerate(X):
        target = fields / f"run_{i:03d}.npy"
        if target.exists():
            continue
        tmp = runs / "tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        run_once(setup, tmp, dict(zip(PROBLEM["names"], map(float, x))))
        utci = read_stack(tmp, "UTCI")
        if args.all_hours:
            np.save(fields / f"run_{i:03d}_hours.npy", utci.astype(np.float16))
        np.save(target, utci[args.hour])
        shutil.rmtree(tmp)
        done = i + 1
        rate = (time.perf_counter() - t_start) / done
        print(f"[morris] run {done}/{len(X)} done, ~{rate * (len(X) - done) / 60:.0f} min left", flush=True)

    outdoor = outdoor_mask(raster_dir, np.load(fields / "baseline.npy").shape)
    print(f"[morris] all {len(X)} runs saved; outdoor fraction {outdoor.mean():.2f}; "
          f"baseline area >= {UTCI_VERY_STRONG:.0f} C: "
          f"{100 * np.mean(np.load(fields / 'baseline.npy')[outdoor] >= UTCI_VERY_STRONG):.1f}%")


if __name__ == "__main__":
    main()
