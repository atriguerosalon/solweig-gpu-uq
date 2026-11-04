"""Does running several SOLWEIG jobs at once on one GPU finish the work sooner?

Without NVIDIA's MPS (not available on a laptop under Windows/WSL), kernels from different
processes do not run at the same instant: the GPU time-slices between them. Sharing still pays
when each job leaves the GPU idle part of the time (SOLWEIG waits on Python, launches and syncs),
because another job's kernels fill the gaps; it stops paying once the GPU is busy all the time,
and each process also costs its own CUDA context memory.

Test: k identical jobs started together (k = 1, 2, 3, 4, 6), each a full SOLWEIG run (sky view
factor + day to 14 h) on the same 400 x 400 px Pacoima window, with --config c1c6 (fast functions,
compact masks, tight bound; as on 2 Nov) or current (also C7 in the sky view step, C8 and the packed
mask cache in the day). Throughput = k / wall time. nvidia-smi samples utilisation, power and memory
every second. Needs the GPU otherwise idle.

Usage:
    python scripts/bench_concurrency.py --rasters runs_la_emu/2019-07-30/rasters --met data/la/met_pacoima_20190730.txt
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

LEVELS = [1, 2, 3, 4, 6]


def worker(rasters: Path, met: Path, d: Path, date: str, config: str) -> None:
    from solweig_gpu import run_utci_tiles
    from solweig_sens.compact_masks import compact_masks
    from solweig_sens.fast import fast_solweig
    from solweig_sens.fast_sky_shadow import fast_sky_shadow
    from solweig_sens.hoist_geometry import hoist_geometry
    from solweig_sens.run import Setup, prepare
    from solweig_sens.tight_bound import tight_bound

    current = config == "current"
    shutil.rmtree(d, ignore_errors=True)
    setup = Setup(raster_dir=rasters, met_file=met, runs_dir=d, date=date, tile_size=400)
    with contextlib.ExitStack() as st:
        for c in [compact_masks("bits", svf=True)] + ([fast_sky_shadow()] if current else []) + [tight_bound()]:
            st.enter_context(c)
        prepare(setup)
    day = ([fast_solweig(), compact_masks("bits", svf=False, packed_cache=True), hoist_geometry(), tight_bound()]
           if current else [fast_solweig(), compact_masks("bits", svf=False), tight_bound()])
    with contextlib.ExitStack() as st:
        for c in day:
            st.enter_context(c)
        run_utci_tiles(base_path=str(d / "out"), preprocess_dir=str(setup.preprocess_dir), selected_date_str=date,
                       save_tmrt=False)
    shutil.rmtree(d, ignore_errors=True)


def samples(log: Path) -> dict:
    rows = [l.split(", ") for l in log.read_text().splitlines()[1:] if l.count(",") == 3]
    vals = lambda i: [float(r[i].split()[0]) for r in rows if r[i].split()[0].replace(".", "").isdigit()]  # noqa: E731
    u, p, m = vals(1), vals(2), vals(3)
    mean = lambda x: sum(x) / len(x) if x else float("nan")  # noqa: E731
    return {"gpu_util_mean_pct": mean(u), "power_mean_w": mean(p), "mem_max_mib": max(m) if m else float("nan"),
            "samples": len(rows)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", nargs="?", default="bench", choices=["bench", "worker"])
    ap.add_argument("--rasters", type=Path, default=ROOT / "runs_la_emu" / "2019-07-30" / "rasters")
    ap.add_argument("--met", type=Path, default=ROOT / "data" / "la" / "met_pacoima_20190730.txt")
    ap.add_argument("--dir", type=Path)
    ap.add_argument("--work", type=Path, default=Path.home() / "bench_conc")
    ap.add_argument("--date", default="2019-07-30")
    ap.add_argument("--config", choices=["c1c6", "current"], default="c1c6")
    ap.add_argument("--out", type=Path, default=ROOT / "figures" / "bench_concurrency.json")
    args = ap.parse_args()
    if args.mode == "worker":
        worker(args.rasters.resolve(), args.met.resolve(), args.dir, args.date, args.config)
        return

    from run_morris import trim_met

    work = args.work
    work.mkdir(parents=True, exist_ok=True)
    met = work / "met_0to14.txt"
    trim_met(args.met.resolve(), met, 14)
    res = {}
    for k in LEVELS:
        log = work / f"smi_{k}.csv"
        smi = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,power.draw,memory.used",
                                "--format=csv", "-l", "1"], stdout=log.open("w"), stderr=subprocess.DEVNULL)
        t0 = time.perf_counter()
        procs = [subprocess.Popen([sys.executable, "-W", "ignore", __file__, "worker", "--rasters", str(args.rasters),
                                   "--met", str(met), "--dir", str(work / f"job_{k}_{i}"), "--date", args.date,
                                   "--config", args.config],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) for i in range(k)]
        codes = [p.wait() for p in procs]
        wall = time.perf_counter() - t0
        smi.terminate()
        smi.wait()
        res[k] = {"wall_s": wall, "jobs_per_hour": 3600 * k / wall, "failed": sum(c != 0 for c in codes),
                  **samples(log)}
        r = res[k]
        print(f"[conc] {k} at once: {wall:.0f} s, {r['jobs_per_hour']:.1f} jobs/h "
              f"(x{r['jobs_per_hour'] / res[1]['jobs_per_hour']:.2f} vs one), GPU {r['gpu_util_mean_pct']:.0f}%, "
              f"{r['power_mean_w']:.0f} W, max {r['mem_max_mib']:.0f} MiB, failed {r['failed']}", flush=True)
        args.out.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
