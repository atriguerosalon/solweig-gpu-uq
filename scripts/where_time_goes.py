"""Where does the day's time go, function by function? Launch-bound, bandwidth-bound or neither.

For ours (--config c1c6: C1-C6; current: also C8 and the packed mask cache; C7 only changes the sky
view step) on centred Pacoima crops, 00-14 h as in bound_test.py, three passes per tile size:

    clean    no instrumentation: the reference wall time (after a discarded warm-up day; once
             before and once after the other passes, to see drift)
    timed    every listed SOLWEIG function synchronises the GPU on entry and exit, so its wall
             time includes its GPU work; self time = own time minus listed callees. With
             --repeats n, n timed passes alternate with clean days and self times are their median
             (every run kept as self_s_runs)
    counted  a TorchDispatchMode counts, per function, the CUDA operators that launch work (views
             excluded), the bytes they read and write, and the values read back to the CPU

Two microbenchmarks give the machine's constants: the CPU cost of one small launch, and the
achievable memory bandwidth (large elementwise op). With them each function gets two lower
bounds on its time, launches x launch cost and bytes / bandwidth, and is called launch-bound,
bandwidth-bound, or neither (Python, syncs, or something the model misses) by which bound comes
close to its measured self time. No profiler and no CUPTI, so this works under WSL; it is a
model, checked by how much of the measured time the bounds explain.

Usage:
    python scripts/where_time_goes.py --rasters data/la3000_mirror/pacoima --met data/la/met_pacoima_20190730.txt
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import functools
import json
import shutil
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from bound_test import centred_crop  # noqa: E402
from run_morris import trim_met  # noqa: E402
from solweig_sens.run import Setup, prepare  # noqa: E402

# Functions timed and counted; wrapped wherever the name is bound (utci_process imports some by name).
FUNCS = ["Solweig_2022a_calc", "define_patch_characteristics", "Lcyl_v2022a", "Kside_veg_v2022a",
         "Lside_veg_v2022a", "gvf_2018a", "sunonsurface_2018a", "shadowingfunction_wallheight_23", "Perez_v3",
         "cylindric_wedge", "Kup_veg_2015a", "create_patches", "utci_calculator", "load_cached_svf_outputs",
         "Tgmaps_v1"]
LAST_HOUR = 14


def modules():
    import solweig_gpu.calculate_utci as cu
    import solweig_gpu.shadow as sh
    import solweig_gpu.solweig as sw
    import solweig_gpu.utci_process as up

    return [sw, sh, up, cu]


@contextlib.contextmanager
def wrapped(on_enter, on_exit):
    """Wrap every FUNCS binding; on_enter(name) and on_exit(name) run around each call."""
    saved = []
    for m in modules():
        for name in FUNCS:
            f = getattr(m, name, None)
            if f is None or not callable(f):
                continue

            def make(f=f, name=name):
                @functools.wraps(f)
                def w(*a, **k):
                    on_enter(name)
                    try:
                        return f(*a, **k)
                    finally:
                        on_exit(name)
                return w

            saved.append((m, name, f))
            setattr(m, name, make())
    try:
        yield
    finally:
        for m, name, f in reversed(saved):
            setattr(m, name, f)


class Timer:
    """Synchronised wall time per function, inclusive and self (minus listed callees)."""

    def __init__(self):
        import torch

        self.sync = torch.cuda.synchronize
        self.stack = []  # [name, start, time spent in listed callees]
        self.incl, self.self_, self.calls = collections.Counter(), collections.Counter(), collections.Counter()

    def enter(self, name):
        self.sync()
        self.stack.append([name, time.perf_counter(), 0.0])

    def exit(self, name):
        self.sync()
        n, t0, child = self.stack.pop()
        dt = time.perf_counter() - t0
        self.incl[n] += dt
        self.self_[n] += dt - child
        self.calls[n] += 1
        if self.stack:
            self.stack[-1][2] += dt


def counting_mode(stack: list, launches, bytes_rw, syncs):
    """A TorchDispatchMode attributing CUDA work to the innermost listed function."""
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_flatten

    def nbytes(xs):
        return sum(x.numel() * x.element_size() for x in xs if isinstance(x, torch.Tensor) and x.is_cuda)

    class Count(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            where = stack[-1] if stack else "outside listed functions"
            ins = tree_flatten((args, kwargs))[0]
            outs = tree_flatten(out)[0]
            on_gpu = any(isinstance(x, torch.Tensor) and x.is_cuda for x in ins + outs)
            if func is torch.ops.aten._local_scalar_dense.default and on_gpu:
                syncs[where] += 1
            elif on_gpu and not func.is_view:
                launches[where] += 1
                bytes_rw[where] += nbytes(ins) + nbytes(outs)
            return out

    return Count()


def machine_constants() -> dict:
    """CPU cost of one tiny launch (s) and achievable bandwidth (bytes/s), best of several tries."""
    import torch

    a = torch.zeros(1, device="cuda")
    per_launch = []
    for _ in range(5):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20000):
            a.add_(1.0)
        per_launch.append((time.perf_counter() - t0) / 20000)  # CPU side only: the launch cost
        torch.cuda.synchronize()
    x = torch.rand(64 * 2 ** 20, device="cuda")  # 256 MB
    y = torch.empty_like(x)
    bw = []
    for _ in range(10):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        torch.mul(x, 2.0, out=y)
        e.record()
        torch.cuda.synchronize()
        bw.append(2 * x.numel() * 4 / (s.elapsed_time(e) / 1e3))
    return {"launch_s": min(per_launch), "launch_s_median": sorted(per_launch)[2], "bandwidth_Bps": max(bw)}


def day(setup: Setup, out: Path, date: str):
    from solweig_gpu import run_utci_tiles

    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    run_utci_tiles(base_path=str(out), preprocess_dir=str(setup.preprocess_dir), selected_date_str=date,
                   save_tmrt=False)


def day_patches(config: str) -> list:
    """c1c6: as measured first on 2 Nov; current: + geometry once per run (C8) and the packed mask cache."""
    from solweig_sens.compact_masks import compact_masks
    from solweig_sens.fast import fast_solweig
    from solweig_sens.hoist_geometry import hoist_geometry
    from solweig_sens.tight_bound import tight_bound

    if config == "c1c6":
        return [fast_solweig(), compact_masks("bits", svf=False), tight_bound()]
    return [fast_solweig(), compact_masks("bits", svf=False, packed_cache=True), hoist_geometry(), tight_bound()]


def main() -> None:
    import torch
    from solweig_sens.compact_masks import compact_masks
    from solweig_sens.fast_sky_shadow import fast_sky_shadow
    from solweig_sens.tight_bound import tight_bound

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rasters", type=Path, required=True)
    ap.add_argument("--met", type=Path, required=True)
    ap.add_argument("--sizes", type=int, nargs="*", default=[500, 1000, 2000])
    ap.add_argument("--repeats", type=int, default=1,
                    help="timed passes per size, each followed by a clean day; self times are their median")
    ap.add_argument("--work", type=Path, default=Path.home() / "where_time_goes")
    ap.add_argument("--date", default="2019-07-30")
    ap.add_argument("--out", type=Path, default=ROOT / "figures" / "time_budget.json")
    ap.add_argument("--config", choices=["c1c6", "current"], default="current",
                    help="c1c6: C1-C6 as on 2 Nov afternoon; current: + C8 and the packed mask cache")
    args = ap.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    met = args.work / f"met_0to{LAST_HOUR}.txt"
    trim_met(args.met.resolve(), met, LAST_HOUR)

    torch.zeros(1, device="cuda")
    res = json.loads(args.out.read_text()) if args.out.exists() else {}
    res["machine"] = machine_constants()
    res.setdefault("sizes", {})
    print(f"[budget] launch {res['machine']['launch_s'] * 1e6:.1f} us, bandwidth "
          f"{res['machine']['bandwidth_Bps'] / 1e9:.0f} GB/s", flush=True)
    for size in args.sizes:
        raster = args.work / f"rasters_{size}"
        centred_crop(args.rasters.resolve(), raster, size)
        d = args.work / f"run_{size}"
        shutil.rmtree(d, ignore_errors=True)
        setup = Setup(raster_dir=raster, met_file=met, runs_dir=d, date=args.date, tile_size=size)
        with compact_masks("bits", svf=True), fast_sky_shadow(), tight_bound():
            prepare(setup)
        rec = {}
        def clean() -> float:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            day(setup, d / "out", args.date)
            torch.cuda.synchronize()
            return time.perf_counter() - t0

        with contextlib.ExitStack() as st:
            for cm in day_patches(args.config):
                st.enter_context(cm)
            rec["warmup_s"] = clean()  # first day pays one-off costs (compiling patches, first kernels)
            rec["clean_s"] = [clean()]

            timers, timed = [], []
            for _ in range(args.repeats):  # timed passes, each followed by a clean one (interleaved: drift spreads)
                tm = Timer()
                t0 = time.perf_counter()
                with wrapped(tm.enter, tm.exit):
                    day(setup, d / "out", args.date)
                torch.cuda.synchronize()
                timed.append(time.perf_counter() - t0)
                timers.append(tm)
                if len(timers) < args.repeats:
                    rec["clean_s"].append(clean())
            rec["timed_s"] = statistics.median(timed)
            rec["timed_s_runs"] = timed

            stack, launches, rw, syncs = [], collections.Counter(), collections.Counter(), collections.Counter()
            t0 = time.perf_counter()
            with wrapped(lambda n: stack.append(n), lambda n: stack.pop()), counting_mode(stack, launches, rw, syncs):
                day(setup, d / "out", args.date)
            torch.cuda.synchronize()
            rec["counted_s"] = time.perf_counter() - t0
            rec["clean_s"].append(clean())  # again at the end: drift check
        names = set(launches).union(*(t.calls for t in timers))
        rec["functions"] = {n: {"calls": timers[0].calls[n],
                                "incl_s": statistics.median(t.incl[n] for t in timers),
                                "self_s": statistics.median(t.self_[n] for t in timers),
                                "self_s_runs": [t.self_[n] for t in timers],
                                "launches": launches[n], "bytes": rw[n], "syncs": syncs[n]}
                            for n in names}
        rec["outside"] = {"launches": launches["outside listed functions"], "bytes": rw["outside listed functions"],
                          "syncs": syncs["outside listed functions"]}
        res["sizes"][str(size)] = rec
        args.out.write_text(json.dumps(res, indent=2))
        m = res["machine"]
        print(f"[budget] {size} px: clean {' / '.join(f'{x:.1f}' for x in rec['clean_s'])} s, timed {rec['timed_s']:.1f} s "
              f"(median of {len(timed)}), "
              f"counted {rec['counted_s']:.1f} s", flush=True)
        for n, f in sorted(rec["functions"].items(), key=lambda kv: -kv[1]["self_s"])[:10]:
            print(f"[budget]   {n:34s} self {f['self_s']:6.2f} s  launches {f['launches']:8d} "
                  f"(>= {f['launches'] * m['launch_s']:5.2f} s)  {f['bytes'] / 1e9:7.1f} GB "
                  f"(>= {f['bytes'] / m['bandwidth_Bps']:5.2f} s)  syncs {f['syncs']}", flush=True)
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    main()
