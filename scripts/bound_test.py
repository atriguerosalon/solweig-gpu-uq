"""What limits SOLWEIG-GPU's speed? Tile-size scaling and the GPU's busy fraction.

Two classical tests, for v2.0.0 and for ours (fast functions, compact masks, tight bound):

    scaling   sky view factor and day (00-14 h) time and peak GPU memory against tile size, on
              centred crops of one Pacoima domain (125 px up to the domain, until out of memory).
              Time flat in size: overhead-bound (Python, kernel launches, CPU-GPU syncs). Time
              proportional to pixels: throughput-bound.
    profile   the day under the PyTorch profiler: GPU kernel time / unprofiled wall time (busy
              fraction), number of kernel launches, number of CPU-GPU syncs (.item()).

On Windows (WDDM) a GPU that runs out of memory spills into system RAM and runs ~30x slower
instead of failing, so PyTorch is capped at 95% of the card: past that, a run stops with a real
out-of-memory error, recorded as such, and the larger sizes are skipped for that version.

Needs the GPU to itself (check nvidia-smi first); timings are medians of --repeats runs.

Usage:
    python scripts/bound_test.py --rasters data/la3000/pacoima --met data/la/met_pacoima_20190730.txt
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from crop_domain import NAMES  # noqa: E402
from run_morris import trim_met  # noqa: E402
from solweig_sens.run import Setup, prepare  # noqa: E402

SIZES, PROFILE_SIZES, LAST_HOUR = [125, 250, 500, 1000, 1500, 2000, 2500, 3000], [250, 1000, 2000], 14


def centred_crop(src: Path, dst: Path, size: int) -> float:
    """Crop the centre of the tile; returns its height range (m), which sets the shadow loop length."""
    from osgeo import gdal

    gdal.UseExceptions()
    dst.mkdir(parents=True, exist_ok=True)
    ds = gdal.Open(str(src / "DEM.tif"))
    c, r = (ds.RasterXSize - size) // 2, (ds.RasterYSize - size) // 2
    for name in NAMES:
        gdal.Translate(str(dst / f"{name}.tif"), str(src / f"{name}.tif"), srcWin=[c, r, size, size])
    dsm = gdal.Open(str(dst / "Building_DSM.tif")).ReadAsArray()
    trees = gdal.Open(str(dst / "Trees.tif")).ReadAsArray()
    return float(max(dsm.max(), (dsm + trees).max()) - dsm.min())


def contexts(cfg: str, stage: str) -> list:
    if cfg == "v200":
        return []
    from solweig_sens.compact_masks import compact_masks
    from solweig_sens.fast import fast_solweig
    from solweig_sens.tight_bound import tight_bound

    if stage == "svf":
        return [compact_masks("bits", svf=True), tight_bound()]
    return [fast_solweig(), compact_masks("bits", svf=False), tight_bound()]


def one_run(work: Path, raster: Path, met: Path, size: int, cfg: str, date: str, profile: bool = False) -> dict:
    import torch

    d = work / f"{cfg}_{size}"
    shutil.rmtree(d, ignore_errors=True)
    setup = Setup(raster_dir=raster, met_file=met, runs_dir=d, date=date, tile_size=size)
    try:
        return timed(setup, d, cfg, date, profile)
    except torch.OutOfMemoryError:
        return {"oom": True}
    finally:
        shutil.rmtree(d, ignore_errors=True)
        torch.cuda.empty_cache()


def timed(setup: Setup, d: Path, cfg: str, date: str, profile: bool) -> dict:
    import torch
    from solweig_gpu import run_utci_tiles

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with contextlib.ExitStack() as st:
        for c in contexts(cfg, "svf"):
            st.enter_context(c)
        prepare(setup)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    svf_peak = torch.cuda.max_memory_reserved() / 2 ** 30
    torch.cuda.reset_peak_memory_stats()
    out = d / "out"
    out.mkdir(parents=True)
    prof = None
    with contextlib.ExitStack() as st:
        for c in contexts(cfg, "day"):
            st.enter_context(c)
        if profile:
            from torch.profiler import ProfilerActivity, profile as tprofile
            prof = st.enter_context(tprofile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]))
        run_utci_tiles(base_path=str(out), preprocess_dir=str(setup.preprocess_dir), selected_date_str=date,
                       save_tmrt=False)
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    rec = {"svf_s": t1 - t0, "day_s": t2 - t1, "svf_peak_gib": svf_peak,
           "day_peak_gib": torch.cuda.max_memory_reserved() / 2 ** 30}
    if prof is not None:
        ev = prof.key_averages()
        dev = lambda e: getattr(e, "self_device_time_total", getattr(e, "self_cuda_time_total", 0))  # noqa: E731
        kernels = [e for e in ev if str(e.device_type).endswith("CUDA")]
        # Under WSL, CUPTI does not initialise, so the profiler sees no GPU kernels: report None, not 0.
        rec |= {"kernel_time_s": sum(dev(e) for e in kernels) / 1e6 if kernels else None,
                "kernel_launches": sum(e.count for e in kernels) if kernels else None,
                "syncs_item": sum(e.count for e in ev if e.key == "aten::_local_scalar_dense"),
                "aten_ops": sum(e.count for e in ev if e.key.startswith("aten::"))}
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rasters", type=Path, required=True)
    ap.add_argument("--met", type=Path, required=True)
    ap.add_argument("--work", type=Path, default=Path.home() / "boundtest")
    ap.add_argument("--date", default="2019-07-30")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--out", type=Path, default=ROOT / "figures")
    ap.add_argument("--sizes", type=int, nargs="*", default=SIZES, help="tile sizes to (re)run")
    ap.add_argument("--profile-sizes", type=int, nargs="*", default=PROFILE_SIZES)
    args = ap.parse_args()
    import torch
    from osgeo import gdal

    torch.cuda.set_per_process_memory_fraction(0.95)  # fail with out-of-memory instead of spilling to RAM
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    met = work / f"met_0to{LAST_HOUR}.txt"
    trim_met(args.met.resolve(), met, LAST_HOUR)
    domain = gdal.Open(str(args.rasters.resolve() / "DEM.tif")).RasterXSize
    sizes = [s for s in args.sizes if s <= domain]
    needed = sorted(set(sizes) | set(args.profile_sizes))
    heights = {s: centred_crop(args.rasters.resolve(), work / f"rasters_{s}", s) for s in needed}
    one_run(work, work / f"rasters_{needed[0]}", met, needed[0], "ours", args.date)  # warm-up

    path = args.out / "bound_test.json"
    # results of earlier sizes are kept, so an interrupted test can be resumed with --sizes
    res = json.loads(path.read_text()) if path.exists() else {"scaling": {}, "profile": {}, "height_range_m": {}}
    res |= {"gpu": torch.cuda.get_device_name(0), "domain_px": domain}
    res["height_range_m"] |= {str(s): h for s, h in heights.items()}
    oom_at = {}  # smallest size each configuration ran out of memory at
    for k, v in res["scaling"].items():
        if v.get("oom"):
            cfg, size = k.split("_")
            oom_at[cfg] = min(oom_at.get(cfg, int(size)), int(size))
    for s in sizes:
        for cfg in ("v200", "ours"):
            if s >= oom_at.get(cfg, s + 1):
                continue
            runs = [one_run(work, work / f"rasters_{s}", met, s, cfg, args.date) for _ in range(args.repeats)]
            if any(r.get("oom") for r in runs):
                oom_at[cfg] = s
                res["scaling"][f"{cfg}_{s}"] = {"oom": True}
                print(f"[bound] {cfg:4s} {s:5d} px: out of GPU memory; larger sizes skipped", flush=True)
                continue
            res["scaling"][f"{cfg}_{s}"] = {k: statistics.median(r[k] for r in runs) for k in runs[0]}
            r = res["scaling"][f"{cfg}_{s}"]
            print(f"[bound] {cfg:4s} {s:5d} px: sky view {r['svf_s']:6.1f} s, day {r['day_s']:6.1f} s, peak "
                  f"{r['svf_peak_gib']:.2f}/{r['day_peak_gib']:.2f} GiB (height range {heights[s]:.0f} m)", flush=True)
            path.write_text(json.dumps(res, indent=2))
    for s in args.profile_sizes:
        for cfg in ("v200", "ours"):
            if "svf_s" not in res["scaling"].get(f"{cfg}_{s}", {}):
                continue
            r = one_run(work, work / f"rasters_{s}", met, s, cfg, args.date, profile=True)
            if r.get("oom"):
                continue
            # the profiler slows the CPU side, so divide by the unprofiled day time
            day = res["scaling"][f"{cfg}_{s}"]["day_s"]
            r["busy_fraction"] = r["kernel_time_s"] / day if r["kernel_time_s"] is not None else None
            res["profile"][f"{cfg}_{s}"] = r
            busy = f"busy {100 * r['busy_fraction']:.0f}%" if r["busy_fraction"] is not None else "GPU kernels not visible"
            print(f"[bound] profile {cfg:4s} {s:5d} px: {busy}, {r['syncs_item']} syncs, {r['aten_ops']} torch ops "
                  f"(day {day:.1f} s)", flush=True)
            path.write_text(json.dumps(res, indent=2))
    path.write_text(json.dumps(res, indent=2))
    print(f"[bound] report: {path}")


if __name__ == "__main__":
    main()
