"""C7 on a real tile: same sky view factor and day, faster? Pacoima summer, with and without C7.

Both configurations use the compact masks, the fast functions and the tight loop bound; "c7" adds
fast_sky_shadow(). Repeats are interleaved. The first repeat of each configuration keeps its
outputs (every SVF raster, the sky-patch shadow matrices, hourly Tmrt and Shadow), which are then
compared byte for byte; timings and peak memory go to a JSON file. Needs a quiet GPU for the
timings; the comparison does not.

Usage:
    python scripts/verify_fast_sky_shadow.py --src ~/solweig_verify/pacoima_summer --repeats 3
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def run(src: Path, work: Path, c7: bool, date: str) -> dict:
    import torch
    from solweig_gpu import calculate_svf, run_utci_tiles
    from solweig_sens.compact_masks import compact_masks
    from solweig_sens.fast import fast_solweig
    from solweig_sens.fast_sky_shadow import fast_sky_shadow
    from solweig_sens.tight_bound import tight_bound

    shutil.rmtree(work, ignore_errors=True)
    pre = work / "processed_inputs"
    shutil.copytree(src, pre, ignore=shutil.ignore_patterns("SVF"))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with contextlib.ExitStack() as st:
        st.enter_context(compact_masks("bits", svf=True))
        if c7:
            st.enter_context(fast_sky_shadow())
        st.enter_context(tight_bound())  # innermost
        calculate_svf(str(pre), patch_option=2, overwrite=True)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    svf_peak = torch.cuda.max_memory_allocated() / 2 ** 30
    with fast_solweig(), compact_masks("bits", svf=False), tight_bound():
        run_utci_tiles(base_path=str(work / "out"), preprocess_dir=str(pre), selected_date_str=date,
                       save_tmrt=True, save_shadow=True)
    torch.cuda.synchronize()
    return {"svf_s": t1 - t0, "day_s": time.perf_counter() - t1, "svf_peak_gib": svf_peak}


def compare(a: Path, b: Path) -> tuple[int, int]:
    from osgeo import gdal

    files = sorted(p.relative_to(a) for p in a.rglob("*") if p.suffix in (".tif", ".npz", ".npy"))
    bad = 0
    for rel in files:
        p, q = a / rel, b / rel
        if not q.exists():
            print("[c7] missing", rel)
            bad += 1
            continue
        if rel.suffix == ".tif":
            same = np.array_equal(gdal.Open(str(p)).ReadAsArray(), gdal.Open(str(q)).ReadAsArray(), equal_nan=True)
        elif rel.suffix == ".npz":
            with np.load(p) as x, np.load(q) as y:
                same = x.files == y.files and all(np.array_equal(x[k], y[k]) for k in x.files)
        else:
            same = np.array_equal(np.load(p), np.load(q), equal_nan=True)
        if not same:
            print("[c7] DIFF", rel)
            bad += 1
    return len(files), bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=Path.home() / "solweig_verify" / "pacoima_summer")
    ap.add_argument("--work", type=Path, default=Path.home() / "verify_c7")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--date", default="2019-07-30")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "figures")
    args = ap.parse_args()
    times = {"tight": [], "c7": []}
    for rep in range(args.repeats):
        for cfg in ("tight", "c7"):
            keep = args.work / f"{cfg}_kept"
            r = run(args.src, keep if rep == 0 else args.work / f"{cfg}_tmp", cfg == "c7", args.date)
            times[cfg].append(r)
            print(f"[c7] {cfg:5s} rep {rep}: sky view {r['svf_s']:.1f} s, day {r['day_s']:.1f} s, "
                  f"sky view peak {r['svf_peak_gib']:.2f} GiB", flush=True)
            shutil.rmtree(args.work / f"{cfg}_tmp", ignore_errors=True)
    n, bad = compare(args.work / "tight_kept", args.work / "c7_kept")
    print(f"[c7] {n} output files compared, {bad} differ")
    res = {cfg: {k: statistics.median(r[k] for r in rs) for k in rs[0]} | {"runs": rs} for cfg, rs in times.items()}
    res["files_compared"], res["files_differ"] = n, bad
    (args.out / "verify_c7.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
