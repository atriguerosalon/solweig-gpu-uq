"""Where does SOLWEIG-GPU spend its time? Profile one 24-hour run on an existing tile.

Two views:
- cProfile over the whole day: wall time per Python function. Because the code syncs with
  the GPU often, wall time per function is a fair proxy for GPU time spent in it.
- torch.profiler on one daytime hour (default 13:00): kernel launches, CUDA time, and the
  largest kernels.

Run with nothing else on the GPU.

Usage:
    python scripts/profile_solweig.py --runs runs_la --date 2019-07-30 --hour 13
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import shutil
import time
from pathlib import Path


def main() -> None:
    import torch
    import solweig_gpu.utci_process as up
    from solweig_gpu import run_utci_tiles
    from torch.profiler import ProfilerActivity, profile

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path("runs_la"))
    ap.add_argument("--date", default="2019-07-30")
    ap.add_argument("--hour", type=int, default=13)
    ap.add_argument("--out", type=Path, default=Path("figures/profile"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    runs = args.runs.resolve()
    work = runs / "profile_tmp"

    original = up.Solweig_2022a_calc
    traces = {}

    def wrapped(i, *a, **k):
        if int(i) != args.hour:
            return original(i, *a, **k)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            result = original(i, *a, **k)
            torch.cuda.synchronize()
        traces["prof"] = prof
        return result

    up.Solweig_2022a_calc = wrapped
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    run_utci_tiles(base_path=str(work), preprocess_dir=str(runs / "shared" / "processed_inputs"),
                   selected_date_str=args.date, save_tmrt=False)
    pr.disable()
    total = time.perf_counter() - t0
    up.Solweig_2022a_calc = original
    shutil.rmtree(work, ignore_errors=True)

    pr.dump_stats(str(args.out / "cprofile_day.prof"))
    s = io.StringIO()
    stats = pstats.Stats(pr, stream=s)
    stats.sort_stats("cumulative").print_stats(r"solweig_gpu", 40)
    s.write("\n\n==== All functions by own time (tottime) ====\n")
    stats.sort_stats("tottime").print_stats(30)
    (args.out / "cprofile_day.txt").write_text(s.getvalue())

    # CUPTI does not initialise under WSL, so GPU-side events are missing; count operator
    # launches on the CPU side instead (each aten compute op on a CUDA tensor is about one kernel).
    prof = traces["prof"]
    events = prof.key_averages()
    skip = {"aten::empty", "aten::empty_strided", "aten::to", "aten::_to_copy", "aten::lift_fresh",
            "aten::detach", "aten::detach_", "detach", "detach_", "aten::result_type", "aten::item",
            "aten::_local_scalar_dense", "aten::view", "aten::slice", "aten::select", "aten::as_strided",
            "aten::expand", "aten::unsqueeze", "aten::squeeze", "aten::reshape", "aten::alias"}
    n_ops = sum(e.count for e in events if e.key.startswith("aten::") and e.key not in skip)
    n_items = sum(e.count for e in events if e.key in ("aten::item", "aten::_local_scalar_dense"))
    n_copies = sum(e.count for e in events if e.key == "aten::copy_")
    table = events.table(sort_by="cpu_time_total", row_limit=25)
    summary = (
        f"Whole day (24 h, incl. reading inputs and writing outputs): {total:.1f} s\n"
        f"Hour {args.hour}: about {n_ops} operator launches (CPU-side count), "
        f"{n_items} scalar reads back to the CPU (.item and similar), {n_copies} copies\n"
    )
    (args.out / f"torch_profile_hour{args.hour}.txt").write_text(summary + "\n" + table)
    print(summary)
    print(table)


if __name__ == "__main__":
    main()
