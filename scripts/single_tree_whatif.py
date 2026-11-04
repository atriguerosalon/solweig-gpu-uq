"""Does the vegetation sky-patch fix change what an added tree does? Synthetic scenes, v2.0.0 against the fix.

PR #137 (SOLWEIG-GPU) reports that SOLWEIG-GPU "over-cools added tree shade relative to canonical
UMEP-SOLWEIG: approximately 2x stronger cooling across 4x more pixels", and attributes it to the
vegetation scheme of the 153 sky-patch masks. On the Pacoima tile the fix changes the tree effect by
1-2% (eval_fork_fix.py). Here the trees are added to simple scenes where the sky-patch masks
matter most:

    open     one tree (12 m crown, 10 m tall) on open paved ground
    grove    a 3 x 3 group of the same trees, 40 m apart
    street   one tree in a 20 m wide north-south street between 12 m buildings

Each scene with and without its trees, full day (Pacoima forcing, 30 July 2019), 400 x 400 m at 2 m.
Run once per SOLWEIG-GPU version (selected with PYTHONPATH), then compare:

    python scripts/single_tree_whatif.py run v200
    PYTHONPATH=~/solweig_versions/fix python scripts/single_tree_whatif.py run fix
    python scripts/single_tree_whatif.py compare --out figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from solweig_sens.analysis import outdoor_mask, read_stack  # noqa: E402
from solweig_sens.run import Setup, prepare  # noqa: E402

WORK = Path.home() / "single_tree"
GEO = ROOT / "data" / "la" / "pacoima" / "DEM.tif"  # georeference (Pacoima, UTM 11N)
MET = ROOT / "data" / "la" / "met_pacoima_20190730.txt"
DATE, N, GROUND = "2019-07-30", 200, 320.0
SCENES = ("open", "grove", "street")


def disk(cy: int, cx: int, r: float) -> np.ndarray:
    y, x = np.ogrid[:N, :N]
    return (y - cy) ** 2 + (x - cx) ** 2 <= r ** 2


def scene(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Building heights and canopy heights above ground (m)."""
    bld, trees = np.zeros((N, N), np.float32), np.zeros((N, N), np.float32)
    crown = 3.0  # pixels: 12 m diameter at 2 m
    if name == "open":
        trees[disk(N // 2, N // 2, crown)] = 10.0
    elif name == "grove":
        for cy in (80, 100, 120):
            for cx in (80, 100, 120):
                trees[disk(cy, cx, crown)] = 10.0
    elif name == "street":
        bld[20:180, 60:95] = 12.0  # street between columns 95 and 105 (20 m)
        bld[20:180, 105:140] = 12.0
        trees[disk(N // 2, N // 2, crown)] = 10.0
    return bld, trees


def write_rasters(dst: Path, bld: np.ndarray, trees: np.ndarray) -> None:
    from osgeo import gdal

    gdal.UseExceptions()
    dst.mkdir(parents=True, exist_ok=True)
    ref = gdal.Open(str(GEO))
    ground = np.full((N, N), GROUND, np.float32)
    layers = {"DEM": ground, "Building_DSM": ground + bld, "Trees": trees,
              "Landcover": np.where(bld > 0, 2, 1).astype(np.float32)}  # 1 paved, 2 building
    for name, arr in layers.items():
        out = gdal.GetDriverByName("GTiff").Create(str(dst / f"{name}.tif"), N, N, 1, gdal.GDT_Float32)
        out.SetGeoTransform(ref.GetGeoTransform())
        out.SetProjection(ref.GetProjection())
        out.GetRasterBand(1).WriteArray(arr)
        out = None


def run(version: str) -> None:
    import solweig_gpu
    from solweig_gpu import run_utci_tiles

    print(f"[tree] {version}: solweig_gpu from {Path(solweig_gpu.__file__).parent}", flush=True)
    for name in SCENES:
        bld, trees = scene(name)
        for case, t in (("trees", trees), ("notrees", np.zeros_like(trees))):
            d = WORK / version / f"{name}_{case}"
            write_rasters(d / "rasters", bld, t)
            setup = Setup(raster_dir=d / "rasters", met_file=MET, runs_dir=d, date=DATE, tile_size=N)
            prepare(setup)
            run_utci_tiles(base_path=str(d / "out"), preprocess_dir=str(setup.preprocess_dir), selected_date_str=DATE,
                           save_tmrt=True, save_shadow=True)
            print(f"[tree] {version} {name} {case} done", flush=True)


def compare(out: Path) -> None:
    versions = ("v200", "fix")
    res = {}
    for name in SCENES:
        bld, trees = scene(name)
        outdoor = outdoor_mask(WORK / versions[0] / f"{name}_trees" / "rasters", (N, N))
        crown, ground = outdoor & (trees > 0), outdoor & (trees == 0)
        res[name] = {}
        for v in versions:
            d = (read_stack(WORK / v / f"{name}_trees" / "out", "TMRT")
                 - read_stack(WORK / v / f"{name}_notrees" / "out", "TMRT"))  # effect of the trees, per hour
            res[name][v] = [{"hour": h,
                             "px_cooled_0.5K": int((outdoor & (d[h] < -0.5)).sum()),
                             "mean_cooling_there": float(d[h][outdoor & (d[h] < -0.5)].mean()) if (outdoor & (d[h] < -0.5)).any() else 0.0,
                             "px_warmed_0.5K": int((outdoor & (d[h] > 0.5)).sum()),
                             "under_crown": float(d[h][crown].mean()),
                             "open_ground_sum": float(d[h][ground].sum())}  # K x pixels, area-integrated
                            for h in range(d.shape[0])]
    path = out / "single_tree_whatif.json"
    path.write_text(json.dumps(res, indent=2))
    for name, r in res.items():
        print(f"== {name}: hour | px cooled >0.5 K v200 / fix | mean there v200 / fix | under crown v200 / fix"
              f" | px warmed >0.5 K v200 / fix")
        for a, b in zip(r["v200"], r["fix"]):
            print(f"   {a['hour']:2d} | {a['px_cooled_0.5K']:6d} {b['px_cooled_0.5K']:6d} | {a['mean_cooling_there']:+6.2f}"
                  f" {b['mean_cooling_there']:+6.2f} | {a['under_crown']:+6.2f} {b['under_crown']:+6.2f}"
                  f" | {a['px_warmed_0.5K']:6d} {b['px_warmed_0.5K']:6d}")
    print(f"[tree] report: {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["run", "compare"])
    ap.add_argument("version", nargs="?", default="v200")
    ap.add_argument("--out", type=Path, default=ROOT / "figures")
    args = ap.parse_args()
    run(args.version) if args.mode == "run" else compare(args.out)


if __name__ == "__main__":
    main()
