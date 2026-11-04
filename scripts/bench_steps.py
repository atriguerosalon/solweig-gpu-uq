"""Time every performance step on one Pacoima tile, with repeats, for the summary figures.

Day steps (one simulated day, cached sky view factor, outputs on the Linux disk unless noted):
    A   original, outputs on the Windows drive (/mnt/c): the WSL setup cost, not a code change
    B   original
    C   + fast sky (C1), chunk 64 as first written
    D   + fast ground view factors (C2)
    E   + fast hourly shadows (C3)
    E8  sky chunk 64 -> 8 (the default since 30 Oct)
    F   + bit-packed masks (C4/C5), masks decompressed one after another as committed on 30 Oct
    G   + tight loop bound (C6)
    G2  + masks decompressed in parallel (compact_masks default since 2 Nov)
    H   + sky-patch geometry once per run (C8)
    I   + bit-packed copy of the mask cache (compact_masks(packed_cache=True)); repeat 0 writes
        the copy (cold, reported apart), later repeats read it
    K<n> chunk sweep of the fast sky: the code of I with n = 8, 16, 32, 64 or 153 patches per chunk
    J   + hourly shadow step compiled with torch.compile (scripts/compile_shadow.py, experiment);
        compiled once per process before the first run, so later runs reuse it
Sky view factor steps (the one-off geometry step, fresh copy of the inputs each run):
    svf_original, svf_compact (C5), svf_tight (+ C6), svf_c7 (+ C7)

Repeats are interleaved, in a shuffled order each round (seeded), so slow drift on the machine
spreads over all steps. Each run is written to the JSON report as soon as it ends, and a rerun
skips the runs already there, so a stopped benchmark resumes. The first repeat of every step
keeps its outputs, compared afterwards: each day step against the one before it, and every
sky view step against the original, file by file and inside the zip caches. Needs a quiet GPU.

Usage:
    python scripts/bench_steps.py --repeats 5 --svf-repeats 3
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import json
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_fast_sky import compare as compare_day  # noqa: E402
from verify_fast_sky_shadow import compare as compare_files  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DAY = ["A", "B", "C", "D", "E", "E8", "F", "G", "G2", "H", "I", "J"]
SVF = ["svf_original", "svf_compact", "svf_tight", "svf_c7"]


@contextlib.contextmanager
def sky_chunk(chunk: int):
    """Run the fast sky with a given chunk size (the patch swapped in last wins)."""
    import solweig_gpu.solweig as sw
    from solweig_sens.fast_sky import define_patch_characteristics

    old = sw.define_patch_characteristics
    sw.define_patch_characteristics = functools.partial(define_patch_characteristics, chunk=chunk)
    try:
        yield
    finally:
        sw.define_patch_characteristics = old


def patches(step: str) -> list:
    """Context managers for a step, outermost first; tight_bound must stay innermost."""
    from solweig_sens.compact_masks import compact_masks
    from solweig_sens.fast import fast_solweig
    from solweig_sens.fast_gvf import fast_gvf
    from solweig_sens.fast_sky import fast_sky
    from solweig_sens.fast_sky_shadow import fast_sky_shadow
    from solweig_sens.hoist_geometry import hoist_geometry
    from solweig_sens.tight_bound import tight_bound
    from compile_shadow import compiled_shadow
    from verify_c8 import serial_load

    return {
        "A": [], "B": [],
        "C": [fast_sky(), sky_chunk(64)],
        "D": [fast_sky(), fast_gvf(), sky_chunk(64)],
        "E": [fast_solweig(), sky_chunk(64)],
        "E8": [fast_solweig()],
        "F": [fast_solweig(), compact_masks("bits", svf=False), serial_load()],
        "G": [fast_solweig(), compact_masks("bits", svf=False), serial_load(), tight_bound()],
        "G2": [fast_solweig(), compact_masks("bits", svf=False), tight_bound()],
        "H": [fast_solweig(), compact_masks("bits", svf=False), hoist_geometry(), tight_bound()],
        "I": [fast_solweig(), compact_masks("bits", svf=False, packed_cache=True), hoist_geometry(), tight_bound()],
        "J": [fast_solweig(), compact_masks("bits", svf=False, packed_cache=True), hoist_geometry(), compiled_shadow(),
              tight_bound()],
        # chunk sweep of the fast sky (K<n>): the current code (as I) with n patches per chunk
        **{f"K{n}": [fast_solweig(), sky_chunk(n), compact_masks("bits", svf=False, packed_cache=True), hoist_geometry(),
                     tight_bound()] for n in (8, 16, 32, 64, 153)},
        "svf_original": [],
        "svf_compact": [compact_masks("bits", svf=True)],
        "svf_tight": [compact_masks("bits", svf=True), tight_bound()],
        "svf_c7": [compact_masks("bits", svf=True), fast_sky_shadow(), tight_bound()],
    }[step]


def run(step: str, args, keep: bool) -> dict:
    import torch
    from solweig_gpu import calculate_svf, run_utci_tiles

    is_svf = step.startswith("svf")
    if step == "A":
        work = REPO / "runs_la" / "bench_steps_A"
    else:
        work = args.work / (f"{step}_kept" if keep else "tmp")
    shutil.rmtree(work, ignore_errors=True)
    if is_svf:
        pre = work / "processed_inputs"
        shutil.copytree(args.svf_src, pre, ignore=shutil.ignore_patterns("SVF"))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with contextlib.ExitStack() as st:
        for cm in patches(step):
            st.enter_context(cm)
        if is_svf:
            calculate_svf(str(pre), patch_option=2, overwrite=True)
        else:
            pre = REPO / "runs_la" / "shared" / "processed_inputs" if step == "A" else args.day_pre
            run_utci_tiles(base_path=str(work), preprocess_dir=str(pre), selected_date_str=args.date, save_tmrt=True)
    torch.cuda.synchronize()
    sec = time.perf_counter() - t0
    out = {"seconds": sec, "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
           "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2 ** 30}
    if not keep:
        shutil.rmtree(work, ignore_errors=True)
    return out


def compare_zips(a: Path, b: Path) -> tuple[int, int]:
    """Rasters inside the sky view zip caches, member by member: (members, members that differ)."""
    import zipfile

    from osgeo import gdal

    bad = 0
    names = sorted(zipfile.ZipFile(a).namelist())
    for n in names:
        x, y = gdal.Open(f"/vsizip/{a}/{n}"), gdal.Open(f"/vsizip/{b}/{n}")
        if y is None or not np.array_equal(x.ReadAsArray(), y.ReadAsArray(), equal_nan=True):
            bad += 1
    return len(names), bad


def environment() -> dict:
    import solweig_gpu
    import torch

    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    return {"gpu_driver": smi, "torch": torch.__version__, "cuda": torch.version.cuda,
            "solweig_gpu": getattr(solweig_gpu, "__version__", "?"), "python": platform.python_version(),
            "started": time.strftime("%Y-%m-%d %H:%M")}


def main() -> None:
    import torch

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeats", type=int, default=5, help="repeats of every day step")
    ap.add_argument("--svf-repeats", type=int, default=3, help="repeats of every sky view step")
    ap.add_argument("--day-pre", type=Path, default=Path.home() / "solweig_bench" / "processed_inputs",
                    help="preprocessed Pacoima tile with its sky view cache, on the Linux disk")
    ap.add_argument("--svf-src", type=Path, default=Path.home() / "solweig_verify" / "pacoima_summer",
                    help="preprocessed Pacoima tile; its SVF folder is left out and recomputed")
    ap.add_argument("--work", type=Path, default=Path.home() / "bench_steps")
    ap.add_argument("--date", default="2019-07-30")
    ap.add_argument("--steps", nargs="*", default=DAY + SVF)
    ap.add_argument("--out", type=Path, default=REPO / "figures" / "profile" / "bench_steps.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rep_out = json.loads(args.out.read_text()) if args.out.exists() else {"environment": environment(), "runs": []}
    if not rep_out["runs"] and "I" in args.steps:  # fresh start: step I's first run must find no packed copy (cold)
        for f in (args.day_pre / "SVF").glob("*.bits.npz"):
            f.unlink()
    done = {(r["step"], r["repeat"]) for r in rep_out["runs"]}
    torch.zeros(1, device="cuda")  # CUDA context before timing
    if "J" in args.steps:  # build the compiled shadow function from SOLWEIG's own source, before any patch
        from compile_shadow import compiled_shadow
        with compiled_shadow():
            pass
    rng = random.Random(args.seed)
    for rep in range(max(args.repeats, args.svf_repeats)):
        order = [s for s in args.steps if rep < (args.svf_repeats if s.startswith("svf") else args.repeats)]
        rng.shuffle(order)  # drawn every round, so the order does not depend on what is resumed
        for step in order:
            if (step, rep) in done:
                continue
            r = run(step, args, keep=rep == 0)
            rep_out["runs"].append({"step": step, "repeat": rep, **r})
            args.out.write_text(json.dumps(rep_out, indent=2))
            print(f"[steps] {step:12s} rep {rep}: {r['seconds']:6.1f} s, peak reserved "
                  f"{r['peak_reserved_gib']:.2f} GiB", flush=True)

    # I's first run writes the packed copy, J's first run compiles: both reported, not in the medians
    warm = [r for r in rep_out["runs"] if not (r["step"] in ("I", "J") and r["repeat"] == 0)]
    rep_out["median_seconds"] = {s: statistics.median(r["seconds"] for r in warm if r["step"] == s)
                                 for s in args.steps if any(r["step"] == s for r in warm)}
    kept = lambda s: REPO / "runs_la" / "bench_steps_A" if s == "A" else args.work / f"{s}_kept"  # noqa: E731
    cmp = {}
    day = [s for s in DAY if s in args.steps]
    for a, b in zip(day, day[1:]):
        if (kept(a) / "output_folder").exists() and (kept(b) / "output_folder").exists():
            cmp[f"{b}_vs_{a}"] = compare_day(kept(a), kept(b))
    for s in SVF[1:]:
        if s in args.steps and (kept(s) / "processed_inputs").exists():
            n, bad = compare_files(kept("svf_original") / "processed_inputs" / "SVF",
                                   kept(s) / "processed_inputs" / "SVF")
            zips = sorted((kept("svf_original") / "processed_inputs" / "SVF").glob("*.zip"))
            zn = zb = 0
            for z in zips:
                m, d = compare_zips(z, kept(s) / "processed_inputs" / "SVF" / z.name)
                zn, zb = zn + m, zb + d
            cmp[f"{s}_vs_svf_original"] = {"files": n, "differ": bad, "zip_members": zn, "zip_members_differ": zb}
    rep_out["comparisons"] = cmp
    rep_out["finished"] = time.strftime("%Y-%m-%d %H:%M")
    args.out.write_text(json.dumps(rep_out, indent=2))
    print(json.dumps({"median_seconds": rep_out["median_seconds"], "comparisons": cmp}, indent=2))


if __name__ == "__main__":
    main()
