"""Effect of the sky-patch and vegetation-shadow fix (iadicarlo fork, commit 5932ef9) on a real tile.

Run once per SOLWEIG-GPU version, selected with PYTHONPATH (each version extracted with git archive):
    PYTHONPATH=~/solweig_versions/parent python scripts/eval_fork_fix.py run parent
    PYTHONPATH=~/solweig_versions/fix    python scripts/eval_fork_fix.py run fix
Each run computes the sky view factor from scratch, then a full day, saving Tmrt, UTCI and shadow.
Then compare, with the installed v2.0.0 output as a third reference:
    python scripts/eval_fork_fix.py compare

The fix commit's parent already carries the fork's Apple-GPU changes, so "fix vs parent" isolates the
fix; "parent vs v2.0.0" checks those other changes do not alter results on CUDA.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SRC = Path.home() / "solweig_verify" / "pacoima_summer"  # processed inputs of the Pacoima tile
WORK = Path.home() / "solweig_forkeval"
DATE = "2019-07-30"
REFERENCE = ROOT / "runs_la" / "baseline"  # v2.0.0, same inputs and default land-cover table


def run(version: str) -> None:
    import solweig_gpu
    import torch
    from solweig_gpu import calculate_svf, run_utci_tiles

    print(f"[fork] {version}: solweig_gpu from {Path(solweig_gpu.__file__).parent}", flush=True)
    pre = WORK / version / "processed_inputs"
    shutil.rmtree(WORK / version, ignore_errors=True)
    shutil.copytree(SRC, pre, ignore=shutil.ignore_patterns("SVF"))
    if version.endswith("_notrees"):  # tree what-if: the same tile with every tree removed
        from osgeo import gdal

        for f in (pre / "Trees").glob("*.tif"):
            ds = gdal.Open(str(f), gdal.GA_Update)
            ds.GetRasterBand(1).WriteArray(np.zeros((ds.RasterYSize, ds.RasterXSize), np.float32))
            ds = None
    t0 = time.perf_counter()
    calculate_svf(str(pre), patch_option=2, overwrite=True)
    t1 = time.perf_counter()
    run_utci_tiles(base_path=str(WORK / version / "out"), preprocess_dir=str(pre), selected_date_str=DATE,
                   save_tmrt=True, save_shadow=True)
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    (WORK / version / "timing.json").write_text(json.dumps({"svf_s": t1 - t0, "day_s": t2 - t1}))
    print(f"[fork] {version}: SVF {t1 - t0:.0f} s, day {t2 - t1:.0f} s", flush=True)


def svf_rasters(pre: Path) -> dict:
    from osgeo import gdal

    z = next((pre / "SVF").glob("svfs_*.zip"))
    out = {}
    with zipfile.ZipFile(z) as zf:
        for name in zf.namelist():
            out[name] = gdal.Open(f"/vsizip/{z}/{name}").ReadAsArray().astype(np.float64)
    return out


def classes(rasters: Path, shadow: np.ndarray) -> dict:
    from osgeo import gdal

    lc = gdal.Open(str(rasters / "Landcover.tif")).ReadAsArray()
    trees = gdal.Open(str(rasters / "Trees.tif")).ReadAsArray()
    h, w = shadow.shape[1:]
    lc, trees = lc[:h, :w], trees[:h, :w]
    outdoor = lc != 2
    return {"all outdoor": outdoor, "under trees": outdoor & (trees > 0),
            "open (no trees)": outdoor & (trees <= 0)}


def stats(d: np.ndarray) -> dict:
    d = d[np.isfinite(d)]
    return {"mean": float(d.mean()), "mean_abs": float(np.abs(d).mean()), "p1": float(np.percentile(d, 1)),
            "p99": float(np.percentile(d, 99)), "max_abs": float(np.abs(d).max())}


def compare() -> None:
    from solweig_sens.analysis import read_stack

    versions = [v for v in ("parent", "fix", "patchcount", "zenith") if (WORK / v / "timing.json").exists()]
    runs = {"v2.0.0": REFERENCE, **{v: WORK / v / "out" for v in versions}}
    out = {"timing": {v: json.loads((WORK / v / "timing.json").read_text()) for v in versions}}
    shadow = read_stack(REFERENCE, "Shadow")
    cls = classes(ROOT / "data" / "la" / "pacoima", shadow)
    pairs = [("parent", "fix"), ("v2.0.0", "parent"), ("v2.0.0", "fix")]
    if "patchcount" in versions:  # v2.0.0 with only the patch-count line of the fix
        pairs += [("v2.0.0", "patchcount"), ("patchcount", "fix")]
    if "zenith" in versions:  # ... plus altitude in float64 in shadow(): float32 tan(90 deg) < 0
        pairs += [("patchcount", "zenith"), ("zenith", "fix")]
    for a, b in pairs:
        res = {}
        for var in ("TMRT", "UTCI"):
            x, y = read_stack(runs[a], var), read_stack(runs[b], var)
            d = y - x
            res[var] = {"identical": bool(np.array_equal(x, y, equal_nan=True))}
            for cname, m in cls.items():
                res[var][cname] = {
                    "day (10-16 h)": stats(d[10:17][:, m]),
                    "14:00 sunlit": stats(d[14][m & (shadow[14] > 0.5)]),
                    "14:00 shaded": stats(d[14][m & (shadow[14] <= 0.5)]),
                    "night (21-05 h)": stats(np.concatenate([d[h][m] for h in [21, 22, 23, 0, 1, 2, 3, 4, 5]])),
                }
        out[f"{b} minus {a}"] = res
        d = read_stack(runs[b], "TMRT") - read_stack(runs[a], "TMRT")
        m = cls["all outdoor"]
        res["TMRT hourly mean, sunlit / shaded"] = {
            h: [float(np.nanmean(d[h][m & (shadow[h] > 0.5)])) if (m & (shadow[h] > 0.5)).any() else None,
                float(np.nanmean(d[h][m & (shadow[h] <= 0.5)]))] for h in range(24)}
    sa, sb = svf_rasters(WORK / "parent" / "processed_inputs"), svf_rasters(WORK / "fix" / "processed_inputs")
    out["svf fix minus parent"] = {k: stats(sb[k] - sa[k]) for k in sorted(sa)}
    s0 = svf_rasters(SRC)  # v2.0.0
    out["svf fix minus v2.0.0"] = {k: stats(sb[k] - s0[k]) for k in sorted(s0)}
    if "patchcount" in versions:
        sp = svf_rasters(WORK / "patchcount" / "processed_inputs")
        out["svf patchcount minus v2.0.0"] = {k: stats(sp[k] - s0[k]) for k in sorted(s0)}
    if "zenith" in versions:
        sz = svf_rasters(WORK / "zenith" / "processed_inputs")
        out["svf zenith minus v2.0.0"] = {k: stats(sz[k] - s0[k]) for k in sorted(s0)}
    tree_runs = {"v2.0.0": (REFERENCE, WORK / "v200_notrees" / "out"), "fix": (runs.get("fix"), WORK / "fix_notrees" / "out")}
    if all((w / "..").resolve().joinpath("timing.json").exists() for _, w in tree_runs.values()):
        res = {}
        for v, (with_, without) in tree_runs.items():
            d = read_stack(with_, "TMRT")[14] - read_stack(without, "TMRT")[14]  # effect of the trees
            m = cls["all outdoor"]
            cool = m & (d < -0.5)
            res[v] = {"pixels cooled > 0.5 K": int(cool.sum()), "mean cooling there": float(d[cool].mean()),
                      "mean effect, all outdoor": float(d[m].mean()), "mean effect under trees": float(d[cls["under trees"]].mean()),
                      "mean effect, open ground": float(d[cls["open (no trees)"]].mean())}
        out["tree effect at 14:00 (with minus without trees)"] = res
        for v, r in res.items():
            print(f"== trees, {v}: " + ", ".join(f"{k} {x:.3f}" if isinstance(x, float) else f"{k} {x}" for k, x in r.items()))
    path = ROOT / "figures" / "profile" / "fork_fix_effect.json"
    path.write_text(json.dumps(out, indent=2))
    for k in [f"{b} minus {a}" for a, b in pairs]:
        r = out[k]
        print(f"== {k}: Tmrt identical {r['TMRT']['identical']}, UTCI identical {r['UTCI']['identical']}")
        for cname in cls:
            t = r["TMRT"][cname]
            print(f"   Tmrt {cname:16s} day mean {t['day (10-16 h)']['mean']:+.3f} K (1-99%: {t['day (10-16 h)']['p1']:+.2f} to "
                  f"{t['day (10-16 h)']['p99']:+.2f}, max |{t['day (10-16 h)']['max_abs']:.2f}|), 14h shaded mean "
                  f"{t['14:00 shaded']['mean']:+.3f}, night mean {t['night (21-05 h)']['mean']:+.3f}")
    for name in [k for k in out if k.startswith("svf ")]:
        print(f"== {name} (mean, max abs):")
        for k, s in out[name].items():
            print(f"   {k:14s} {s['mean']:+.4f}  {s['max_abs']:.3f}")
    print(f"[fork] report: {path}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "run":
        run(sys.argv[2])
    elif len(sys.argv) == 2 and sys.argv[1] == "compare":
        compare()
    else:
        print(__doc__)
        sys.exit(0 if any(a in ("-h", "--help") for a in sys.argv[1:]) else 2)
