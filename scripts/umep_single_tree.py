"""The single-tree scenes in UMEP's own SOLWEIG, to compare with SOLWEIG-GPU.

PR #137 (SOLWEIG-GPU) says SOLWEIG-GPU "over-cools added tree shade relative to canonical
UMEP-SOLWEIG: approximately 2x stronger cooling across 4x more pixels". single_tree_whatif.py
showed that the PR's own fix changes the tree effect by 1-4%. This runs the same scenes, on the
same rasters and forcing, in umep-core (UMEP-dev, the QGIS-free port of UMEP-processing's
SOLWEIG, PyPI ``umep`` 0.0.1b47), with SOLWEIG-GPU v2.0.0's settings:

    anisotropic sky (153 patches), only global radiation (diffuse and direct estimated),
    cylinder person, land cover, 3% canopy transmissivity, 25% trunk zone, deciduous
    (leaves DOY 97-300), no wall scheme, UTC-7. Walls Ts_deg set to SOLWEIG-GPU's 0.58
    (UMEP's JSON has 0.37); everything else in the parameter JSON already matches.

Needs the rasters written by ``single_tree_whatif.py run v200`` and a separate env with umep:

    micromamba run -n umep python scripts/umep_single_tree.py run
    micromamba run -n umep python scripts/umep_single_tree.py compare --out figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORK = Path.home() / "single_tree"
MET = ROOT / "data" / "la" / "met_pacoima_20190730.txt"
PARAMS = Path.home() / "umep-core" / "demos" / "data" / "Goteborg_SWEREF99_1200" / "parametersforsolweig.json"
SCENES = ("open", "grove", "street")
CASES = ("trees", "notrees")


def config(r: Path, w: Path) -> str:
    return f"""[DEFAULT]
output_dir={w}/output/
working_dir={w}/working/
dsm_path={r}/Building_DSM.tif
cdsm_path={r}/Trees.tif
tdsm_path=
dem_path={r}/DEM.tif
lc_path={r}/Landcover.tif
wh_path={w}/walls/wall_hts.tif
wa_path={w}/walls/wall_aspects.tif
svf_path={w}/svf/svfs.zip
aniso_path={w}/svf/shadowmats.npz
poi_path=
poi_field=
wall_path=
woi_path=
woi_field=
met_path={MET}
only_global=1
use_veg_dem=1
conifer=0
person_cylinder=1
utc=-7
use_landcover=1
use_dem_for_buildings=0
use_aniso=1
use_wall_scheme=0
wall_type=Brick
output_tmrt=1
output_kup=0
output_kdown=0
output_lup=0
output_ldown=0
output_sh=1
save_buildings=0
output_kdiff=0
output_tree_planter=0
wall_netcdf=0
use_epw_file=0
epw_start_date=
epw_end_date=
epw_hours=
"""


def run() -> None:
    import umep
    from umep import skyviewfactor_algorithm, wall_heightaspect_algorithm
    from umep.functions.SOLWEIGpython import solweig_runner_core

    print(f"[umep] umep from {Path(umep.__path__[0])}", flush=True)
    params = json.loads(PARAMS.read_text())
    params["Ts_deg"]["Value"]["Walls"] = 0.58  # SOLWEIG-GPU's landcoverclasses_2016a.txt
    for name in SCENES:
        for case in CASES:
            r, w = WORK / "v200" / f"{name}_{case}" / "rasters", WORK / "umep" / f"{name}_{case}"
            (w / "output").mkdir(parents=True, exist_ok=True)
            (w / "working").mkdir(parents=True, exist_ok=True)
            (w / "params.json").write_text(json.dumps(params, indent=2))
            (w / "config.ini").write_text(config(r, w))
            wall_heightaspect_algorithm.generate_wall_hts(dsm_path=str(r / "Building_DSM.tif"), bbox=None,
                                                          out_dir=str(w / "walls"))
            skyviewfactor_algorithm.generate_svf(dsm_path=str(r / "Building_DSM.tif"), bbox=None,
                                                 out_dir=str(w / "svf"), dem_path=str(r / "DEM.tif"),
                                                 cdsm_path=str(r / "Trees.tif"), trans_veg_perc=3,
                                                 trunk_ratio_perc=25)
            solweig_runner_core.SolweigRunCore(config_path_str=str(w / "config.ini"),
                                               params_json_path=str(w / "params.json")).run()
            print(f"[umep] {name} {case} done", flush=True)


def tmrt(version: str, run_dir: str) -> np.ndarray:
    """Tmrt as (hours, rows, cols)."""
    import rasterio

    d = WORK / version / run_dir
    if version == "umep":
        files = sorted((d / "output").glob("Tmrt_*.tif"))
        files = [f for f in files if "average" not in f.name]
        return np.stack([rasterio.open(f).read(1) for f in files])
    (f,) = (d / "out" / "output_folder").glob("*/TMRT_*.tif")
    return rasterio.open(f).read()


def compare(out: Path) -> None:
    import rasterio

    versions = ("v200", "fix", "umep")
    res = {}
    for name in SCENES:
        r = WORK / "v200" / f"{name}_trees" / "rasters"
        dsm, dem = (rasterio.open(r / f).read(1) for f in ("Building_DSM.tif", "DEM.tif"))
        trees = rasterio.open(r / "Trees.tif").read(1)
        outdoor = (dsm - dem) < 0.5
        crown = outdoor & (trees > 0)
        base = {v: tmrt(v, f"{name}_notrees") for v in versions}
        res[name] = {"hours": base["umep"].shape[0]}
        for v in versions:
            d = tmrt(v, f"{name}_trees") - base[v]
            cooled = outdoor & (d < -0.5)
            res[name][v] = [{"hour": h,
                             "px_cooled_0.5K": int(cooled[h].sum()),
                             "mean_cooling_there": float(d[h][cooled[h]].mean()) if cooled[h].any() else 0.0,
                             "px_warmed_0.5K": int((outdoor & (d[h] > 0.5)).sum()),
                             "under_crown": float(d[h][crown].mean()),
                             "area_sum": float(d[h][outdoor].sum())}  # K x pixels
                            for h in range(d.shape[0])]
        # do the two codes agree without trees? (otherwise a tree-effect gap means little)
        res[name]["baseline_gpu_minus_umep"] = [{"hour": h, "mean": float((base["v200"][h] - base["umep"][h])[outdoor].mean()),
                                                 "max_abs": float(np.abs(base["v200"][h] - base["umep"][h])[outdoor].max())}
                                                for h in range(base["umep"].shape[0])]
    path = out / "umep_single_tree.json"
    path.write_text(json.dumps(res, indent=2))
    for name, r in res.items():
        print(f"== {name}: hour | px cooled >0.5 K v200 fix umep | under crown v200 fix umep"
              f" | area sum (K px) v200 fix umep | no-tree Tmrt gpu-umep mean / max|.|")
        for h in range(r["hours"]):
            a, b, c, z = r["v200"][h], r["fix"][h], r["umep"][h], r["baseline_gpu_minus_umep"][h]
            print(f"   {h:2d} | {a['px_cooled_0.5K']:5d} {b['px_cooled_0.5K']:5d} {c['px_cooled_0.5K']:5d}"
                  f" | {a['under_crown']:+6.2f} {b['under_crown']:+6.2f} {c['under_crown']:+6.2f}"
                  f" | {a['area_sum']:8.0f} {b['area_sum']:8.0f} {c['area_sum']:8.0f}"
                  f" | {z['mean']:+6.2f} {z['max_abs']:6.2f}")
    print(f"[umep] report: {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["run", "compare"])
    ap.add_argument("--out", type=Path, default=ROOT / "figures")
    args = ap.parse_args()
    run() if args.mode == "run" else compare(args.out)


if __name__ == "__main__":
    main()
