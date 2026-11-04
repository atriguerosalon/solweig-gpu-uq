"""How wrong is the "add up single trees" shortcut for planting many trees? Pacoima, full reruns.

Tools that place many trees (UMEP TreePlanter, Wallenberg et al. 2022; the optimisation of
Schrodi et al. 2023) estimate a planting from per-tree effects, without rerunning SOLWEIG for
the combination. Here every combination is also rerun in full (new canopy, new sky view
factor, full day), so the shortcut's error is measured directly.

On the 0.8 x 0.8 km Pacoima window of the Morris and emulator runs: 12 new street trees
(10 m tall, 12 m crowns, on paved ground away from buildings and existing canopy), in four
clusters of three 14 m apart, so that some trees shade each other's ground. Runs:

    base        no new trees
    single_XX   each tree alone (12)
    cluster_X   each cluster of three (4)
    subset_XX   random subsets of 2-8 trees (16, seeded)
    all         all 12

Each run writes its own rasters, sky view factor and full-day Tmrt/Shadow (fast functions,
compact masks and the tight loop bound, all verified equal to the original). Analysis:
analyse_la_treeplant.py.

Usage:
    python scripts/run_la_treeplant.py --base runs_la_emu/2019-07-30/rasters --met data/la/met_pacoima_20190730.txt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solweig_sens.analysis import is_complete  # noqa: E402
from solweig_sens.run import Setup, prepare  # noqa: E402

HEIGHT, RADIUS, SPACING = 10.0, 3.0, 7  # m, pixels (12 m crown at 2 m), pixels (14 m)
N_CLUSTERS, PER_CLUSTER, N_SUBSETS, SEED = 4, 3, 16, 0
EDGE, BLDG_GAP = 40, 4  # pixels from the window edge (80 m) and from buildings (8 m)


def disk(shape, cy, cx, r=RADIUS):
    y, x = np.ogrid[: shape[0], : shape[1]]
    return (y - cy) ** 2 + (x - cx) ** 2 <= r ** 2


def place_trees(lc: np.ndarray, trees: np.ndarray, dsm: np.ndarray, dem: np.ndarray, rng) -> list[tuple[int, int]]:
    """Cluster centres far apart; three trees per cluster, SPACING apart, all on clear paved ground."""
    from scipy.ndimage import binary_dilation

    bldg = (dsm - dem) > 0.5
    ok = (lc == 1) & (trees == 0) & ~binary_dilation(bldg, iterations=BLDG_GAP)
    ok &= ~binary_dilation(trees > 0, iterations=int(RADIUS) + 1)
    ok[:EDGE], ok[-EDGE:], ok[:, :EDGE], ok[:, -EDGE:] = False, False, False, False
    crown_ok = lambda cy, cx: ok[disk(ok.shape, cy, cx)].all()  # noqa: E731
    offsets = [[(0, 0), (0, SPACING), (0, 2 * SPACING)], [(0, 0), (SPACING, 0), (2 * SPACING, 0)],
               [(0, 0), (0, SPACING), (SPACING, 0)], [(0, 0), (SPACING, 0), (SPACING, SPACING)]]
    ys, xs = np.nonzero(ok)
    placed: list[tuple[int, int]] = []
    for _ in range(20000):
        if len(placed) == N_CLUSTERS * PER_CLUSTER:
            return placed
        i = rng.integers(len(ys))
        cy, cx = int(ys[i]), int(xs[i])
        if any(np.hypot(cy - py, cx - px) < 60 for py, px in placed):  # clusters >= 120 m apart
            continue
        for off in offsets:
            cand = [(cy + dy, cx + dx) for dy, dx in off]
            if all(0 <= y < ok.shape[0] and 0 <= x < ok.shape[1] and crown_ok(y, x) for y, x in cand):
                placed += cand
                break
    raise RuntimeError(f"placed only {len(placed)} trees")


def cases(n: int, rng) -> dict[str, list[int]]:
    out = {"base": []}
    out |= {f"single_{i:02d}": [i] for i in range(n)}
    out |= {f"cluster_{c}": list(range(c * PER_CLUSTER, (c + 1) * PER_CLUSTER)) for c in range(N_CLUSTERS)}
    for s in range(N_SUBSETS):
        k = int(rng.integers(2, 9))
        out[f"subset_{s:02d}"] = sorted(int(i) for i in rng.choice(n, size=k, replace=False))
    out["all"] = list(range(n))
    return out


def write_rasters(src: Path, dst: Path, sites: list[tuple[int, int]]) -> None:
    from osgeo import gdal

    gdal.UseExceptions()
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("DEM", "Landcover", "Building_DSM", "Trees"):
        ds = gdal.Open(str(src / f"{name}.tif"))
        arr = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
        if name == "Trees":
            for cy, cx in sites:
                arr[disk(arr.shape, cy, cx)] = HEIGHT
        out = gdal.GetDriverByName("GTiff").Create(str(dst / f"{name}.tif"), ds.RasterXSize, ds.RasterYSize, 1,
                                                    gdal.GDT_Float32)
        out.SetGeoTransform(ds.GetGeoTransform())
        out.SetProjection(ds.GetProjection())
        out.GetRasterBand(1).WriteArray(arr)
        out = None


def main() -> None:
    from osgeo import gdal
    from solweig_gpu import run_utci_tiles
    from solweig_sens.compact_masks import compact_masks
    from solweig_sens.fast import fast_solweig
    from solweig_sens.tight_bound import tight_bound

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, required=True, help="raster folder of the window")
    ap.add_argument("--met", type=Path, required=True)
    ap.add_argument("--runs", type=Path, default=Path.home() / "treeplant")
    ap.add_argument("--date", default="2019-07-30")
    args = ap.parse_args()
    runs = args.runs.resolve()
    runs.mkdir(parents=True, exist_ok=True)
    rd = lambda n: gdal.Open(str(args.base / f"{n}.tif")).ReadAsArray().astype(np.float32)  # noqa: E731
    rng = np.random.default_rng(SEED)
    design_path = runs / "design.json"
    if design_path.exists():
        design = json.loads(design_path.read_text())
    else:
        sites = place_trees(rd("Landcover"), rd("Trees"), rd("Building_DSM"), rd("DEM"), rng)
        design = {"sites": sites, "cases": cases(len(sites), rng), "height_m": HEIGHT, "crown_radius_px": RADIUS,
                  "base": str(args.base.resolve()), "met": str(args.met.resolve()), "date": args.date}
        design_path.write_text(json.dumps(design, indent=1))
    timings_path = runs / "timings.json"
    timings = json.loads(timings_path.read_text()) if timings_path.exists() else {}

    for name, idx in design["cases"].items():
        out_dir = runs / name / "out"
        if is_complete(out_dir):
            continue
        setup = Setup(raster_dir=runs / name / "rasters", met_file=args.met.resolve(), runs_dir=runs / name,
                      date=args.date)
        t0 = time.perf_counter()
        write_rasters(args.base.resolve(), setup.raster_dir, [tuple(design["sites"][i]) for i in idx])
        with compact_masks("bits", svf=True), tight_bound():
            prepare(setup)
        t1 = time.perf_counter()
        out_dir.mkdir(parents=True, exist_ok=True)
        with fast_solweig(), compact_masks("bits", svf=False), tight_bound():
            run_utci_tiles(base_path=str(out_dir), preprocess_dir=str(setup.preprocess_dir),
                           selected_date_str=args.date, save_tmrt=True, save_shadow=True)
        timings[name] = {"svf_s": t1 - t0, "day_s": time.perf_counter() - t1}
        timings_path.write_text(json.dumps(timings, indent=2))
        print(f"[plant] {name} ({len(idx)} trees): sky view {t1 - t0:.0f} s, day {time.perf_counter() - t1:.0f} s",
              flush=True)


if __name__ == "__main__":
    main()
