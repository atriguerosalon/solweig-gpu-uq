"""Experiment A: torch.compile on one step of the hourly shadow loop. Same bits? Faster?

The hourly shadow function (shadowingfunction_wallheight_23, with C3) steps away from the sun;
each step runs ~30 full-tile operations (shift four rasters, maxima, comparisons, wheres), each a
separate GPU kernel that reads and writes whole arrays. The time budget (where_time_goes.py) found
it bandwidth-bound on large tiles. torch.compile can fuse a step into a few kernels that read each
input once. The step only subtracts, compares, takes maxima and adds small whole numbers, which
fusion cannot round differently, so the outputs should stay bit-identical; this checks it.

    capture   run one day (C1-C6 and C8) and save the arguments of every k-th hourly shadow call
    test      for each saved call: C3 as is (reference), the same step written as a function
              (eager), and that function compiled; outputs compared bit for bit, time per call
              (CUDA-synchronised, median of repeats), compile time, number of compiled graphs.
              --variant slices keeps C3's slicing (first try, 2 Nov: 4-6x faster per call, but a
              new compile almost every hour); --variant fixed builds the shifted rasters with
              fixed shapes and passes the window as GPU data; --variant graphs compiles the fixed
              step with mode "reduce-overhead" (CUDA graphs), copying each step's outputs before
              the next replay overwrites them
    day       whole days with everything bit-identical so far (C1-C8, packed mask cache), with and
              without the compiled step, interleaved; outputs compared byte for byte. The first
              compiled day pays the compile; the later ones reuse it (same process). --mode
              reduce-overhead uses CUDA graphs

Results (3-4 Nov, quiet machine; figures/compile_shadow_*.json): outputs byte-identical in every
case. Whole day: about -5 to -7% at 1000 px, 84.2 -> 66.7 s (-21%) at 2000 px; first compilation
with empty caches about 3-4.5 s, once. CUDA graphs: -1.4% (1000 px) and -10% (2000 px), never better
than plain compile; profile_compiled_steps.py bounds what graphs can save at 0.5-3.6% of a step.

Usage:
    python scripts/compile_shadow.py capture --pre ~/solweig_bench/processed_inputs --tag 1000
    python scripts/compile_shadow.py test --tag 1000
    python scripts/compile_shadow.py day --pre ~/solweig_bench/processed_inputs --tag 1000
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import statistics
import sys
import textwrap
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

WORK = Path.home() / "compile_shadow"
BODY = [  # the C3 loop body, between the step bounds and index += 1
    "tempvegdem.zero_()",
    "tempvegdem2.zero_()",
    "temp.zero_()",
    "templastfabovea.zero_()",
    "templastgabovea.zero_()",
    "tempvegdem[xp1:xp2, yp1:yp2] = vegdem[xc1:xc2, yc1:yc2] - dz",
    "tempvegdem2[xp1:xp2, yp1:yp2] = vegdem2[xc1:xc2, yc1:yc2] - dz",
    "temp[xp1:xp2, yp1:yp2] = a[xc1:xc2, yc1:yc2] - dz",
    "f = torch.maximum(f, temp) # Moving building shadow",
    "shvoveg = torch.maximum(shvoveg, tempvegdem) # moving vegetation shadow volume",
    "sh = torch.where(f > a, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))",
    "fabovea = (tempvegdem > a).float()   # vegdem above DEM",
    "gabovea = (tempvegdem2 > a).float()   # vegdem2 above DEM",
    "templastfabovea[xp1:xp2, yp1:yp2] = vegdem[xc1:xc2, yc1:yc2] - dzprev",
    "templastgabovea[xp1:xp2, yp1:yp2] = vegdem2[xc1:xc2, yc1:yc2] - dzprev",
    "lastfabovea = templastfabovea > a",
    "lastgabovea = templastgabovea > a",
    "dzprev = dz",
    "vegsh2 = fabovea + gabovea + lastfabovea.float() + lastgabovea.float()",
    "vegsh2 = torch.where(vegsh2 == 4, torch.tensor(0.0, device=device), vegsh2)",
    "vegsh2 = torch.where(vegsh2 > 0, torch.tensor(1.0, device=device), vegsh2)",
    "vegsh = torch.maximum(vegsh, vegsh2)",
    "vegsh = torch.where(vegsh * sh > 0, torch.tensor(0.0, device=device), vegsh)",
    "vbshvegsh = vbshvegsh + vegsh",
]


def shadow_step(f, shvoveg, vegsh, vbshvegsh, a, vegdem, vegdem2, dz, dzprev, xc1, xc2, yc1, yc2, xp1, xp2, yp1, yp2):
    """One step of the loop, the same operations in the same order, as a function of its state."""
    device = a.device
    tempvegdem = torch.zeros(a.shape, device=device)
    tempvegdem2 = torch.zeros(a.shape, device=device)
    temp = torch.zeros(a.shape, device=device)
    templastfabovea = torch.zeros(a.shape, device=device)
    templastgabovea = torch.zeros(a.shape, device=device)
    tempvegdem[xp1:xp2, yp1:yp2] = vegdem[xc1:xc2, yc1:yc2] - dz
    tempvegdem2[xp1:xp2, yp1:yp2] = vegdem2[xc1:xc2, yc1:yc2] - dz
    temp[xp1:xp2, yp1:yp2] = a[xc1:xc2, yc1:yc2] - dz
    f = torch.maximum(f, temp)
    shvoveg = torch.maximum(shvoveg, tempvegdem)
    sh = torch.where(f > a, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))
    fabovea = (tempvegdem > a).float()
    gabovea = (tempvegdem2 > a).float()
    templastfabovea[xp1:xp2, yp1:yp2] = vegdem[xc1:xc2, yc1:yc2] - dzprev
    templastgabovea[xp1:xp2, yp1:yp2] = vegdem2[xc1:xc2, yc1:yc2] - dzprev
    lastfabovea = templastfabovea > a
    lastgabovea = templastgabovea > a
    vegsh2 = fabovea + gabovea + lastfabovea.float() + lastgabovea.float()
    vegsh2 = torch.where(vegsh2 == 4, torch.tensor(0.0, device=device), vegsh2)
    vegsh2 = torch.where(vegsh2 > 0, torch.tensor(1.0, device=device), vegsh2)
    vegsh = torch.maximum(vegsh, vegsh2)
    vegsh = torch.where(vegsh * sh > 0, torch.tensor(0.0, device=device), vegsh)
    vbshvegsh = vbshvegsh + vegsh
    return f, shvoveg, sh, vegsh, vbshvegsh


def shadow_step_fixed(f, shvoveg, vegsh, vbshvegsh, a, vegdem, vegdem2, dz, dzprev, b):
    """The same step with fixed shapes: no slicing by step-dependent sizes.

    ``b`` holds (row offset, column offset, xp1, xp2, yp1, yp2) as a GPU tensor, so the compiled
    graph sees data, not Python integers that change every step. Inside the window each shifted
    raster is the source value minus dz, the same single subtraction as the slice assignment;
    outside it is 0, as in the zeroed buffers.
    """
    n0, n1 = a.shape
    r = torch.arange(n0, device=a.device)
    c = torch.arange(n1, device=a.device)
    inside = ((r >= b[2]) & (r < b[3]))[:, None] & ((c >= b[4]) & (c < b[5]))[None, :]
    rs = (r - b[0]).clamp(0, n0 - 1)[:, None]
    cs = (c - b[1]).clamp(0, n1 - 1)[None, :]

    def shifted(x, d):
        return torch.where(inside, x[rs, cs] - d, 0.0)

    device = a.device
    tempvegdem = shifted(vegdem, dz)
    tempvegdem2 = shifted(vegdem2, dz)
    temp = shifted(a, dz)
    f = torch.maximum(f, temp)
    shvoveg = torch.maximum(shvoveg, tempvegdem)
    sh = torch.where(f > a, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))
    fabovea = (tempvegdem > a).float()
    gabovea = (tempvegdem2 > a).float()
    templastfabovea = shifted(vegdem, dzprev)
    templastgabovea = shifted(vegdem2, dzprev)
    lastfabovea = templastfabovea > a
    lastgabovea = templastgabovea > a
    vegsh2 = fabovea + gabovea + lastfabovea.float() + lastgabovea.float()
    vegsh2 = torch.where(vegsh2 == 4, torch.tensor(0.0, device=device), vegsh2)
    vegsh2 = torch.where(vegsh2 > 0, torch.tensor(1.0, device=device), vegsh2)
    vegsh = torch.maximum(vegsh, vegsh2)
    vegsh = torch.where(vegsh * sh > 0, torch.tensor(0.0, device=device), vegsh)
    vbshvegsh = vbshvegsh + vegsh
    return f, shvoveg, sh, vegsh, vbshvegsh


def step_source(fixed: bool = False, clone: bool = False) -> str:
    """C3's function with the loop body replaced by one call of _step.

    clone: copy the step's outputs before the next call. Needed with CUDA graphs (mode="reduce-overhead"):
    a replayed graph writes its outputs into the same static buffers every time, so feeding step k's outputs
    into step k+1 would read buffers the next replay overwrites.
    """
    from solweig_sens.fast_shadow import patched_source

    lines = patched_source().splitlines()
    starts = [i for i, l in enumerate(lines) if l.strip() == BODY[0]]
    if len(starts) != 1:
        raise RuntimeError("unexpected C3 source")
    s = starts[0]
    code = [i for i in range(s, len(lines)) if lines[i].strip() and not lines[i].strip().startswith("#")]
    e = code[len(BODY) - 1] + 1  # the body ends at its last code line; blank and comment lines skipped
    if [lines[i].strip() for i in code[:len(BODY)]] != BODY:
        raise RuntimeError("unexpected C3 loop body")
    ind = lines[s][: len(lines[s]) - len(lines[s].lstrip())]
    if not fixed:
        call = [f"{ind}f, shvoveg, sh, vegsh, vbshvegsh = _step(f, shvoveg, vegsh, vbshvegsh, a, vegdem, vegdem2, dz, dzprev,",
                f"{ind}                                         xc1, xc2, yc1, yc2, xp1, xp2, yp1, yp2)",
                f"{ind}dzprev = dz"]
        return "\n".join(lines[:s] + call + lines[e:]) + "\n"
    # Fixed shapes: all steps' offsets and windows go to the GPU once, one row per step.
    loop = [i for i in range(s) if lines[i].strip().startswith("for _k, (xc1, xc2, yc1, yc2, xp1, xp2, yp1, yp2)")]
    if len(loop) != 1:
        raise RuntimeError("unexpected C3 loop header")
    li = lines[loop[0]][: len(lines[loop[0]]) - len(lines[loop[0]].lstrip())]
    pre = [f"{li}_B = torch.tensor([[xp1 - xc1, yp1 - yc1, xp1, xp2, yp1, yp2]",
           f"{li}                   for (xc1, xc2, yc1, yc2, xp1, xp2, yp1, yp2) in _steps] or [[0] * 6], device=device)"]
    call = [f"{ind}f, shvoveg, sh, vegsh, vbshvegsh = _step(f, shvoveg, vegsh, vbshvegsh, a, vegdem, vegdem2, dz, dzprev,",
            f"{ind}                                         _B[_k])",
            f"{ind}dzprev = dz"]
    if clone:
        call.insert(2, f"{ind}f, shvoveg, sh, vegsh, vbshvegsh = (_o.clone() for _o in (f, shvoveg, sh, vegsh, vbshvegsh))")
    return "\n".join(lines[:loop[0]] + pre + lines[loop[0]:s] + call + lines[e:]) + "\n"


def build(step, fixed: bool = False, clone: bool = False) -> callable:
    import solweig_gpu.solweig as sw
    from solweig_sens.fast_shadow import _shadow_steps

    ns = dict(vars(sw))
    ns["_shadow_steps"], ns["_step"] = _shadow_steps, step
    exec(compile(textwrap.dedent(step_source(fixed, clone)), sw.__file__, "exec"), ns)
    return ns["shadowingfunction_wallheight_23"]


_COMPILED = {}


@contextlib.contextmanager
def compiled_shadow(mode: str = "default"):
    """The hourly shadow function with its loop step compiled (fixed shapes). Enter after fast_solweig().

    The function is built the first time this is called, which must be before any patch is entered:
    building reads SOLWEIG's own source, and inside fast_solweig() the module holds C3's copy instead
    (whose source C3's check rightly refuses).
    """
    import solweig_gpu.solweig as sw

    if mode not in _COMPILED:  # one compiled step per process and mode, reused across runs
        _COMPILED[mode] = build(torch.compile(shadow_step_fixed, mode=mode), fixed=True, clone=mode == "reduce-overhead")
    old = sw.shadowingfunction_wallheight_23
    sw.shadowingfunction_wallheight_23 = _COMPILED[mode]
    try:
        yield
    finally:
        sw.shadowingfunction_wallheight_23 = old


def day(args) -> None:
    """Whole days: everything bit-identical so far (C1-C8, packed cache), with and without the compiled step."""
    import random

    from solweig_gpu import run_utci_tiles
    from solweig_sens.compact_masks import compact_masks
    from solweig_sens.fast import fast_solweig
    from solweig_sens.hoist_geometry import hoist_geometry
    from solweig_sens.tight_bound import tight_bound
    from verify_fast_sky_shadow import compare

    def patches(cfg):
        cms = [fast_solweig(), compact_masks("bits", svf=False, packed_cache=True), hoist_geometry()]
        return cms + ([compiled_shadow(args.mode)] if cfg == "compiled" else []) + [tight_bound()]

    work = WORK / f"day_{args.tag}"
    pre = work / "processed_inputs"
    if not pre.exists():
        shutil.copytree(args.pre, pre)
    torch.zeros(1, device="cuda")
    with compiled_shadow(args.mode):  # build it now, from SOLWEIG's own source, before any patch is entered
        pass
    res = {"tag": args.tag, "pre": str(args.pre), "runs": []}
    rng = random.Random(0)
    for rep in range(args.repeats):
        order = ["base", "compiled"]
        rng.shuffle(order)
        for cfg in order:
            out = work / (f"{cfg}_kept" if rep == 0 else "tmp")
            shutil.rmtree(out, ignore_errors=True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with contextlib.ExitStack() as st:
                for cm in patches(cfg):
                    st.enter_context(cm)
                run_utci_tiles(base_path=str(out), preprocess_dir=str(pre), selected_date_str=args.date,
                               save_tmrt=True, save_shadow=True)
            torch.cuda.synchronize()
            sec = time.perf_counter() - t0
            res["runs"].append({"config": cfg, "repeat": rep, "seconds": sec})
            print(f"[compile-day] {cfg:8s} rep {rep}: {sec:.1f} s", flush=True)
            if rep:
                shutil.rmtree(out, ignore_errors=True)
    n, bad = compare(work / "base_kept", work / "compiled_kept")
    res["files_compared"], res["files_differ"] = n, bad
    for cfg in ("base", "compiled"):
        xs = [r["seconds"] for r in res["runs"] if r["config"] == cfg]
        res[f"{cfg}_first_s"], res[f"{cfg}_median_after_first_s"] = xs[0], statistics.median(xs[1:])
    out = ROOT / "figures" / f"compile_shadow_day_{args.tag}.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"[compile-day] {n} output files compared, {bad} differ; wrote {out}")


def capture(args) -> None:
    import solweig_gpu.solweig as sw
    from solweig_gpu import run_utci_tiles
    from solweig_sens.compact_masks import compact_masks
    from solweig_sens.fast import fast_solweig
    from solweig_sens.hoist_geometry import hoist_geometry
    from solweig_sens.tight_bound import tight_bound

    out = WORK / args.tag
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    calls = []
    with contextlib.ExitStack() as st:
        for cm in (fast_solweig(), compact_masks("bits", svf=False), hoist_geometry(), tight_bound()):
            st.enter_context(cm)
        inner = sw.shadowingfunction_wallheight_23

        def spy(*a, **k):
            calls.append(len(calls))
            if (len(calls) - 1) % args.every == 0:
                torch.save({"args": a, "kwargs": k}, out / f"call_{len(calls) - 1:02d}.pt")
            return inner(*a, **k)

        sw.shadowingfunction_wallheight_23 = spy
        run_utci_tiles(base_path=str(WORK / f"run_{args.tag}"), preprocess_dir=str(args.pre),
                       selected_date_str=args.date, save_tmrt=False)
    shutil.rmtree(WORK / f"run_{args.tag}", ignore_errors=True)
    print(f"[compile] {len(calls)} hourly shadow calls, saved every {args.every}th to {out}")


def timed(fn, a, k, repeats):
    out = None
    ts = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn(*a, **k)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return out, ts


def test(args) -> None:
    import torch._dynamo
    from torch._dynamo.utils import counters
    from solweig_sens.fast_shadow import build as build_c3

    torch._dynamo.config.cache_size_limit = 256  # do not fall back to eager silently; count graphs instead
    c3 = build_c3()
    fixed = args.variant in ("fixed", "graphs")
    step = shadow_step_fixed if fixed else shadow_step
    eager = build(step, fixed)
    # graphs: reduce-overhead mode records the compiled step as a CUDA graph and replays it
    compiled_step = torch.compile(step, dynamic=not fixed, mode="reduce-overhead" if args.variant == "graphs" else None)
    compiled = build(compiled_step, fixed, clone=args.variant == "graphs")
    files = sorted((WORK / args.tag).glob("call_*.pt"))
    res = {"tag": args.tag, "variant": args.variant, "torch": torch.__version__, "calls": []}
    for i, p in enumerate(files):
        d = torch.load(p, map_location="cuda", weights_only=False)
        a, k = d["args"], d["kwargs"]
        ref, t_ref = timed(c3, a, k, args.repeats)
        eag, t_eag = timed(eager, a, k, args.repeats)
        t0 = time.perf_counter()
        com, t_first = timed(compiled, a, k, 1)  # first call: compiles if needed
        first = time.perf_counter() - t0
        com, t_com = timed(compiled, a, k, args.repeats)
        same_eager = all(torch.equal(x, y) for x, y in zip(ref, eag))
        same_comp = all(torch.equal(x, y) for x, y in zip(ref, com))
        maxdiff = max(float((x.double() - y.double()).abs().max()) for x, y in zip(ref, com))
        rec = {"call": p.stem, "shape": list(a[0].shape) if hasattr(a[0], "shape") else None,
               "c3_s": statistics.median(t_ref), "eager_step_s": statistics.median(t_eag),
               "compiled_s": statistics.median(t_com), "compiled_first_s": first,
               "bit_identical_eager": same_eager, "bit_identical_compiled": same_comp, "max_abs_diff": maxdiff,
               "graphs_so_far": int(counters["stats"]["unique_graphs"])}
        res["calls"].append(rec)
        print(f"[compile] {p.stem}: C3 {rec['c3_s'] * 1e3:7.1f} ms  eager step {rec['eager_step_s'] * 1e3:7.1f} ms  "
              f"compiled {rec['compiled_s'] * 1e3:7.1f} ms (first {first:5.1f} s)  identical eager/compiled "
              f"{same_eager}/{same_comp}  max diff {maxdiff:.3g}  graphs {rec['graphs_so_far']}", flush=True)
    res["recompiles"] = {k: int(v) for k, v in counters["recompiles"].items()} if "recompiles" in counters else {}
    out = ROOT / "figures" / f"compile_shadow_{args.tag}_{args.variant}.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"[compile] wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--pre", type=Path, required=True, help="preprocessed tile with its sky view cache")
    c.add_argument("--tag", required=True)
    c.add_argument("--every", type=int, default=3, help="save every k-th hourly call")
    c.add_argument("--date", default="2019-07-30")
    t = sub.add_parser("test")
    t.add_argument("--tag", required=True)
    t.add_argument("--repeats", type=int, default=5)
    t.add_argument("--variant", choices=["slices", "fixed", "graphs"], default="fixed",
                   help="slices: C3's slicing as is (recompiles as slice sizes change); fixed: fixed shapes")
    d = sub.add_parser("day")
    d.add_argument("--pre", type=Path, required=True, help="preprocessed tile with its sky view cache")
    d.add_argument("--tag", required=True)
    d.add_argument("--repeats", type=int, default=4)
    d.add_argument("--date", default="2019-07-30")
    d.add_argument("--mode", choices=["default", "reduce-overhead"], default="default",
                   help="torch.compile mode; reduce-overhead also replays the step as a CUDA graph")
    args = ap.parse_args()
    {"capture": capture, "test": test, "day": day}[args.cmd](args)


if __name__ == "__main__":
    main()
