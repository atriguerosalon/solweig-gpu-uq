"""Cache loading and sky-patch geometry (C8) on a real tile: same outputs, how much faster?

One simulated day on the Pacoima tile (cached sky view factor), four configurations, each on its
own copy of the inputs so the packed mask copy of one does not leak into another:

    base     C1-C6 as committed: fast functions, bit-packed masks, tight bound; masks
             decompressed one after another (as before 2 Nov)
    par      the same, masks decompressed in parallel (compact_masks default since 2 Nov)
    hoist    par + sky-patch geometry once per run (hoist_geometry)
    cached   hoist + bit-packed copy of the mask cache (compact_masks(packed_cache=True));
             repeat 0 writes the copy (cold), later repeats read it (warm)

Repeats are interleaved in a shuffled order (seeded). The first repeat of each keeps its hourly
UTCI, Tmrt and shadow rasters, compared with base byte for byte. Each run is written to the
JSON report as it ends, and a rerun resumes. Needs a quiet machine for the timings.

Usage:
    python scripts/verify_c8.py --src ~/solweig_bench/processed_inputs --repeats 5
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import json
import random
import shutil
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from verify_fast_sky_shadow import compare  # noqa: E402

CONFIGS = ["base", "par", "hoist", "cached"]


@contextlib.contextmanager
def serial_load():
    """Decompress the masks one after another, as compact_masks did before 2 Nov."""
    from solweig_sens import compact_masks as cm

    old = cm.ThreadPoolExecutor
    cm.ThreadPoolExecutor = lambda n: concurrent.futures.ThreadPoolExecutor(1)
    try:
        yield
    finally:
        cm.ThreadPoolExecutor = old


def patches(cfg: str) -> list:
    from solweig_sens.compact_masks import compact_masks
    from solweig_sens.fast import fast_solweig
    from solweig_sens.hoist_geometry import hoist_geometry
    from solweig_sens.tight_bound import tight_bound

    cms = [fast_solweig(), compact_masks("bits", svf=False, packed_cache=cfg == "cached")]
    extra = {"base": [serial_load()], "par": [], "hoist": [hoist_geometry()], "cached": [hoist_geometry()]}[cfg]
    return cms + extra + [tight_bound()]  # tight_bound innermost


def run(cfg: str, pre: Path, out: Path, date: str) -> float:
    import torch
    from solweig_gpu import run_utci_tiles

    shutil.rmtree(out, ignore_errors=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with contextlib.ExitStack() as st:
        for cm in patches(cfg):
            st.enter_context(cm)
        run_utci_tiles(base_path=str(out), preprocess_dir=str(pre), selected_date_str=date, save_tmrt=True,
                       save_shadow=True)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def main() -> None:
    import torch

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=Path.home() / "solweig_bench" / "processed_inputs",
                    help="preprocessed tile with its sky view cache")
    ap.add_argument("--work", type=Path, default=Path.home() / "verify_c8")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--date", default="2019-07-30")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "figures" / "verify_c8.json")
    args = ap.parse_args()

    res = json.loads(args.out.read_text()) if args.out.exists() else {"src": str(args.src).replace(str(Path.home()), "~"), "runs": []}
    done = {(r["config"], r["repeat"]) for r in res["runs"]}
    for cfg in CONFIGS:
        pre = args.work / cfg / "processed_inputs"
        if not pre.exists():
            shutil.copytree(args.src, pre, ignore=shutil.ignore_patterns("*.bits.npz"))
    torch.zeros(1, device="cuda")
    rng = random.Random(args.seed)
    for rep in range(args.repeats):
        order = list(CONFIGS)
        rng.shuffle(order)
        for cfg in order:
            if (cfg, rep) in done:
                continue
            out = args.work / cfg / ("kept" if rep == 0 else "tmp")
            sec = run(cfg, args.work / cfg / "processed_inputs", out, args.date)
            res["runs"].append({"config": cfg, "repeat": rep, "seconds": sec})
            args.out.write_text(json.dumps(res, indent=2))
            print(f"[c8] {cfg:6s} rep {rep}: {sec:.1f} s", flush=True)
            if rep:
                shutil.rmtree(out, ignore_errors=True)

    warm = {c: [r["seconds"] for r in res["runs"] if r["config"] == c and not (c == "cached" and r["repeat"] == 0)]
            for c in CONFIGS}  # cached's first run writes the packed copy
    res["median_seconds"] = {c: statistics.median(xs) if xs else None for c, xs in warm.items()}
    res["cached_cold_seconds"] = next(r["seconds"] for r in res["runs"] if r["config"] == "cached" and r["repeat"] == 0)
    res["comparisons"] = {}
    for cfg in CONFIGS[1:]:
        n, bad = compare(args.work / "base" / "kept", args.work / cfg / "kept")
        res["comparisons"][f"{cfg}_vs_base"] = {"files": n, "differ": bad}
    args.out.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("median_seconds", "cached_cold_seconds", "comparisons")}, indent=2))


if __name__ == "__main__":
    main()
