"""Compute a tile's sky view factor cache with the original and with compact masks; compare and measure.

The cache (SkyViewFactor tif, svfs zip of 15 rasters, shadowmats npz of three 153-patch masks) must be
identical. Records peak allocated and reserved GPU memory and wall time of the SVF step.

Usage:
    python scripts/verify_compact_svf.py [--src ~/solweig_verify/pacoima_summer] [--mode bits]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from solweig_sens.compact_masks import compact_masks  # noqa: E402

GIB = 2**30


def compute(pre: Path, compact: str | None) -> dict:
    import contextlib

    import torch
    from solweig_gpu import calculate_svf

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    t0 = time.perf_counter()
    with (compact_masks(compact) if compact else contextlib.nullcontext()):
        calculate_svf(str(pre), patch_option=2, overwrite=True)
    torch.cuda.synchronize()
    return {"seconds": time.perf_counter() - t0,
            "peak_allocated_gib": (torch.cuda.max_memory_allocated() - base) / GIB,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / GIB}


def compare_caches(a: Path, b: Path) -> dict:
    from osgeo import gdal

    out = {}
    for fa in sorted((a / "SVF").iterdir()):
        fb = b / "SVF" / fa.name
        if fa.suffix == ".npz":
            with np.load(fa) as x, np.load(fb) as y:
                for k in x.files:
                    out[f"{fa.name}:{k}"] = bool(np.array_equal(x[k], y[k])) and x[k].dtype == y[k].dtype
        elif fa.suffix == ".zip":
            with zipfile.ZipFile(fa) as za:
                for name in za.namelist():
                    xa = gdal.Open(f"/vsizip/{fa}/{name}").ReadAsArray()
                    xb = gdal.Open(f"/vsizip/{fb}/{name}").ReadAsArray()
                    out[f"{fa.name}:{name}"] = bool(np.array_equal(xa, xb))
        elif fa.suffix == ".tif":
            out[fa.name] = bool(np.array_equal(gdal.Open(str(fa)).ReadAsArray(), gdal.Open(str(fb)).ReadAsArray()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=Path.home() / "solweig_verify" / "pacoima_summer")
    ap.add_argument("--mode", default="bits")
    args = ap.parse_args()

    import torch
    from osgeo import gdal

    gdal.UseExceptions()
    torch.zeros(1, device="cuda")
    work = args.src.parent / "svf_check"
    dirs = {}
    for label in ("original", args.mode):
        d = work / label
        shutil.rmtree(d, ignore_errors=True)
        shutil.copytree(args.src, d, ignore=shutil.ignore_patterns("SVF"))
        dirs[label] = d
    res = {}
    for label, d in dirs.items():
        print(f"[svf] {label}", flush=True)
        res[label] = compute(d, None if label == "original" else args.mode)
        print(f"[svf]   {res[label]}", flush=True)
    cmp = compare_caches(dirs["original"], dirs[args.mode])
    res["identical"] = cmp
    res["all_identical"] = all(cmp.values())
    print(f"[svf] cache files identical: {res['all_identical']} ({sum(cmp.values())}/{len(cmp)} arrays)", flush=True)
    out = ROOT / "figures" / "profile" / "verify_compact_svf.json"
    out.write_text(json.dumps(res, indent=2))
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
