"""Can a tile too big for the GPU with the original code run with compact masks? A 2000 x 2000 test.

Steps, each recording peak GPU memory and wall time:
  1. preprocessing and walls (CPU), then the sky view factor with compact masks (bits)
  2. one simulated day with the fast functions and bit masks
  3. the original code on the same tile: the day run, then the sky view factor, each expected to run
     out of memory on an 8 GB GPU; the out-of-memory error is caught and recorded

Writes figures/profile/bigtile_demo.json.

Usage:
    python scripts/bigtile_demo.py --rasters data/la2000/pacoima --met data/la/met_pacoima_20190730.txt
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from solweig_sens.compact_masks import compact_masks  # noqa: E402
from solweig_sens.fast import fast_solweig  # noqa: E402
from solweig_sens.run import Setup  # noqa: E402

GIB = 2**30


def measured(fn) -> dict:
    import torch

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    t0 = time.perf_counter()
    status = "ok"
    try:
        fn()
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError as e:
        status = f"out of memory: {str(e).splitlines()[0][:160]}"
    out = {"status": status, "seconds": time.perf_counter() - t0,
           "peak_allocated_gib": (torch.cuda.max_memory_allocated() - base) / GIB,
           "peak_reserved_gib": torch.cuda.max_memory_reserved() / GIB}
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rasters", type=Path, required=True)
    ap.add_argument("--met", type=Path, required=True)
    ap.add_argument("--runs", type=Path, default=Path.home() / "solweig_bigtile")
    ap.add_argument("--size", type=int, default=2000)
    args = ap.parse_args()

    import torch
    from solweig_gpu import calculate_svf, preprocess, run_utci_tiles, run_walls_aspect

    torch.zeros(1, device="cuda")
    setup = Setup(raster_dir=args.rasters.resolve(), met_file=args.met.resolve(), runs_dir=args.runs,
                  date="2019-07-30", tile_size=args.size)
    pre = setup.preprocess_dir
    report = {"tile": f"{args.size} x {args.size}", "gpu": torch.cuda.get_device_name(0),
              "gpu_total_gib": torch.cuda.get_device_properties(0).total_memory / GIB}

    if not (pre / "walls").exists():
        print("[big] preprocessing and walls (CPU)", flush=True)
        t0 = time.perf_counter()
        setup.shared_dir.mkdir(parents=True, exist_ok=True)
        preprocess(base_path=str(setup.shared_dir), selected_date_str=setup.date,
                   building_dsm_filename=str(setup.raster_dir / "Building_DSM.tif"),
                   dem_filename=str(setup.raster_dir / "DEM.tif"), trees_filename=str(setup.raster_dir / "Trees.tif"),
                   landcover_filename=str(setup.raster_dir / "Landcover.tif"), tile_size=setup.tile_size,
                   overlap=setup.overlap, use_own_met=True, own_met_file=str(setup.met_file), use_uhi=False)
        run_walls_aspect(str(pre))
        report["preprocess_seconds"] = time.perf_counter() - t0

    def svf_compact():
        with compact_masks("bits"):
            calculate_svf(str(pre), patch_option=2, overwrite=True)

    print("[big] sky view factor, compact masks", flush=True)
    report["svf_compact"] = measured(svf_compact)
    print(f"[big]   {report['svf_compact']}", flush=True)

    out_fast = args.runs / "fast_bits"
    shutil.rmtree(out_fast, ignore_errors=True)

    def day_fast():
        with fast_solweig(), compact_masks("bits"):
            run_utci_tiles(base_path=str(out_fast), preprocess_dir=str(pre), selected_date_str=setup.date, save_tmrt=True)

    print("[big] day, fast + bit masks", flush=True)
    report["day_fast_bits"] = measured(day_fast)
    print(f"[big]   {report['day_fast_bits']}", flush=True)

    out_orig = args.runs / "original"
    shutil.rmtree(out_orig, ignore_errors=True)

    def day_original():
        run_utci_tiles(base_path=str(out_orig), preprocess_dir=str(pre), selected_date_str=setup.date, save_tmrt=True)

    print("[big] day, original (expected out of memory)", flush=True)
    report["day_original"] = measured(day_original)
    print(f"[big]   {report['day_original']}", flush=True)

    svf_copy = args.runs / "svf_original_attempt"
    shutil.rmtree(svf_copy, ignore_errors=True)
    shutil.copytree(pre, svf_copy, ignore=shutil.ignore_patterns("SVF"))

    def svf_original():
        calculate_svf(str(svf_copy), patch_option=2, overwrite=True)

    print("[big] sky view factor, original (expected out of memory)", flush=True)
    report["svf_original"] = measured(svf_original)
    print(f"[big]   {report['svf_original']}", flush=True)
    shutil.rmtree(svf_copy, ignore_errors=True)

    out = ROOT / "figures" / "profile" / "bigtile_demo.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
