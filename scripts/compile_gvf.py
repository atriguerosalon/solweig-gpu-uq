"""Experiment D: torch.compile on one step of the ground view factor loop (sunonsurface_2018a, C2).

The ground view factor steps away from each pixel in the sun direction, 18 times per hour, and
each step runs about 25 whole-tile operations (a bandwidth-bound function on large tiles, see
where_time_goes.py). Unlike the shadow steps, this one multiplies and adds (weight sums), and a
compiler may fuse a multiply and an add into one fused multiply-add, which rounds once instead
of twice. So this is the first experiment where bit-identity is not expected; the limits set
before running apply: max |dUTCI|, |dTmrt| <= 0.01 C and 99.99th percentile <= 0.001 C.

Two details of the original are kept: the temporary rasters are not cleared between steps (out of
the current window they keep the previous step's values), so they are part of the step's state;
and the window is passed as GPU data, so the compiled step has one shape for every hour.

    all     capture every 20th call of a Pacoima day (1000 x 1000), per call: C2 against the
            compiled step (largest difference, time); then whole days at 1000 and 2000 px:
            base (C1-C8, packed cache), + compiled ground view factor, + compiled ground view
            factor and shadow step; outputs compared with base (largest and 99.99th percentile
            difference in UTCI and Tmrt)

Result (4 Nov; figures/compile_gvf.json): exactly identical outputs after all (UTCI and Tmrt differ
by 0 at 1000 and 2000 px), but no gain: 1.01x per call, -5% of a day at 2000 px. The temporaries
that carry over between steps are state, so fusion cannot drop their traffic, and the fixed-shape
gathers are slower than the original's contiguous slices. A negative result.

Usage:
    python scripts/compile_gvf.py all
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import shutil
import statistics
import sys
import textwrap
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

WORK = Path.home() / "compile_gvf"
PRE_1000 = Path.home() / "solweig_bench" / "processed_inputs"
PRE_2000 = Path.home() / "verify_c7_la2000" / "tight_kept" / "processed_inputs"
LOOP = "for n, (xc1, xc2, yc1, yc2, xp1, xp2, yp1, yp2) in enumerate(_offsets):"
BODY = [
    "tempbu[xp1:xp2, yp1:yp2] = buildings[xc1:xc2, yc1:yc2]",
    "tempsh[xp1:xp2, yp1:yp2] = shadow[xc1:xc2, yc1:yc2]",
    "tempLupsh[xp1:xp2, yp1:yp2] = Lup[xc1:xc2, yc1:yc2]",
    "tempalbsh[xp1:xp2, yp1:yp2] = albshadow[xc1:xc2, yc1:yc2]",
    "tempalbnosh[xp1:xp2, yp1:yp2] = alb[xc1:xc2, yc1:yc2]",
    "f = torch.min(f, tempbu)",
    "shadow2 = tempsh * f",
    "weightsumsh += shadow2",
    "Lupsh = tempLupsh * f",
    "weightsumLupsh += Lupsh",
    "albsh = tempalbsh * f",
    "weightsumalbsh += albsh",
    "albnosh = tempalbnosh * f",
    "weightsumalbnosh += albnosh",
    "tempwallsun[xp1:xp2, yp1:yp2] = sunwall[xc1:xc2, yc1:yc2]",
    "tempb = tempwallsun * f",
    "tempbwall = f * -1 + 1",
    "tempbub = ((tempb + tempbub) > 0).float()",
    "tempbubwall = ((tempbwall + tempbubwall) > 0).float()",
    "weightsumLwall += tempbub * Lwall",
    "weightsumalbwall += tempbub * albedo_b",
    "weightsumwall += tempbub",
    "weightsumalbwallnosh += tempbubwall * albedo_b",
]
STATE = ["f", "weightsumsh", "weightsumLupsh", "weightsumalbsh", "weightsumalbnosh", "tempbub", "tempbubwall",
         "weightsumLwall", "weightsumalbwall", "weightsumwall", "weightsumalbwallnosh",
         "tempbu", "tempsh", "tempLupsh", "tempalbsh", "tempalbnosh", "tempwallsun"]


def gvf_step(f, weightsumsh, weightsumLupsh, weightsumalbsh, weightsumalbnosh, tempbub, tempbubwall,
             weightsumLwall, weightsumalbwall, weightsumwall, weightsumalbwallnosh,
             tempbu, tempsh, tempLupsh, tempalbsh, tempalbnosh, tempwallsun,
             buildings, shadow, Lup, albshadow, alb, sunwall, Lwall, albedo_b, b):
    """One step of C2's loop, same operations and order; the temporaries keep their old values
    outside the window, as the original's never-cleared buffers do."""
    n0, n1 = buildings.shape
    r = torch.arange(n0, device=buildings.device)
    c = torch.arange(n1, device=buildings.device)
    inside = ((r >= b[2]) & (r < b[3]))[:, None] & ((c >= b[4]) & (c < b[5]))[None, :]
    rs = (r - b[0]).clamp(0, n0 - 1)[:, None]
    cs = (c - b[1]).clamp(0, n1 - 1)[None, :]

    def shifted(x, prev):
        return torch.where(inside, x[rs, cs], prev)

    tempbu = shifted(buildings, tempbu)
    tempsh = shifted(shadow, tempsh)
    tempLupsh = shifted(Lup, tempLupsh)
    tempalbsh = shifted(albshadow, tempalbsh)
    tempalbnosh = shifted(alb, tempalbnosh)
    f = torch.min(f, tempbu)
    weightsumsh = weightsumsh + tempsh * f
    weightsumLupsh = weightsumLupsh + tempLupsh * f
    weightsumalbsh = weightsumalbsh + tempalbsh * f
    weightsumalbnosh = weightsumalbnosh + tempalbnosh * f
    tempwallsun = shifted(sunwall, tempwallsun)
    tempb = tempwallsun * f
    tempbwall = f * -1 + 1
    tempbub = ((tempb + tempbub) > 0).float()
    tempbubwall = ((tempbwall + tempbubwall) > 0).float()
    weightsumLwall = weightsumLwall + tempbub * Lwall
    weightsumalbwall = weightsumalbwall + tempbub * albedo_b
    weightsumwall = weightsumwall + tempbub
    weightsumalbwallnosh = weightsumalbwallnosh + tempbubwall * albedo_b
    return (f, weightsumsh, weightsumLupsh, weightsumalbsh, weightsumalbnosh, tempbub, tempbubwall,
            weightsumLwall, weightsumalbwall, weightsumwall, weightsumalbwallnosh,
            tempbu, tempsh, tempLupsh, tempalbsh, tempalbnosh, tempwallsun)


def step_source() -> str:
    """C2's function with the loop body replaced by one call of _step."""
    from solweig_sens.fast_gvf import patched_source

    lines = patched_source().splitlines()
    loop = [i for i, l in enumerate(lines) if l.strip() == LOOP]
    starts = [i for i, l in enumerate(lines) if l.strip() == BODY[0]]
    if len(loop) != 1 or len(starts) != 1 or not loop[0] < starts[0]:
        raise RuntimeError("unexpected C2 source")
    s = starts[0]
    code = [i for i in range(s, len(lines)) if lines[i].strip() and not lines[i].strip().startswith("#")]
    if [lines[i].strip() for i in code[:len(BODY)]] != BODY:
        raise RuntimeError("unexpected C2 loop body")
    e = code[len(BODY) - 1] + 1
    li = lines[loop[0]][: len(lines[loop[0]]) - len(lines[loop[0]].lstrip())]
    ind = lines[s][: len(lines[s]) - len(lines[s].lstrip())]
    pre = [f"{li}_B = torch.tensor([[xp1 - xc1, yp1 - yc1, xp1, xp2, yp1, yp2]",
           f"{li}                   for (xc1, xc2, yc1, yc2, xp1, xp2, yp1, yp2) in _offsets] or [[0] * 6], device=device)"]
    state = ", ".join(STATE)
    call = [f"{ind}({state}) = _step(",
            f"{ind}    {state},",
            f"{ind}    buildings, shadow, Lup, albshadow, alb, sunwall, Lwall, albedo_b, _B[n])",
            f"{ind}shadow2 = Lupsh = albsh = albnosh = tempb = tempbwall = None  # deleted after the loop"]
    return "\n".join(lines[:loop[0]] + pre + lines[loop[0]:s] + call + lines[e:]) + "\n"


def build(step):
    import solweig_gpu.solweig as sw
    from solweig_sens.fast_gvf import _gvf_offsets

    ns = dict(vars(sw))
    ns["_gvf_offsets"], ns["_step"] = _gvf_offsets, step
    exec(compile(textwrap.dedent(step_source()), sw.__file__, "exec"), ns)
    return ns["sunonsurface_2018a"]


_COMPILED = {}


@contextlib.contextmanager
def compiled_gvf():
    """sunonsurface_2018a with its loop step compiled. Built on first use, before any patch is
    entered (building reads SOLWEIG's own source); enter after fast_solweig()."""
    import solweig_gpu.solweig as sw

    if "fn" not in _COMPILED:
        _COMPILED["fn"] = build(torch.compile(gvf_step))
    old = sw.sunonsurface_2018a
    sw.sunonsurface_2018a = _COMPILED["fn"]
    try:
        yield
    finally:
        sw.sunonsurface_2018a = old


def day_patches(cfg: str) -> list:
    from compile_shadow import compiled_shadow
    from solweig_sens.compact_masks import compact_masks
    from solweig_sens.fast import fast_solweig
    from solweig_sens.hoist_geometry import hoist_geometry
    from solweig_sens.tight_bound import tight_bound

    cms = [fast_solweig(), compact_masks("bits", svf=False, packed_cache=True), hoist_geometry()]
    if cfg in ("gvf", "gvf+shadow"):
        cms.append(compiled_gvf())
    if cfg == "gvf+shadow":
        cms.append(compiled_shadow())
    return cms + [tight_bound()]


def differences(a: Path, b: Path) -> dict:
    import numpy as np
    from solweig_sens.analysis import read_stack

    out = {}
    for var in ("UTCI", "TMRT", "Shadow"):
        x, y = read_stack(a, var), read_stack(b, var)
        d = np.abs(x.astype(np.float64) - y)
        d = d[np.isfinite(d)]
        out[var] = {"identical": bool(np.array_equal(x, y, equal_nan=True)), "max_abs": float(d.max()),
                    "p99_99": float(np.percentile(d, 99.99)), "n_above_0.001": int((d > 0.001).sum())}
    return out


def per_call(args, log) -> dict:
    import solweig_gpu.solweig as sw
    from solweig_gpu import run_utci_tiles
    from solweig_sens.fast_gvf import build as build_c2
    from torch._dynamo.utils import counters

    c2 = build_c2()
    compiled = build(torch.compile(gvf_step))
    out = WORK / "calls"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    n = [0]
    with contextlib.ExitStack() as st:
        for cm in day_patches("base"):
            st.enter_context(cm)
        inner = sw.sunonsurface_2018a

        def spy(*a):
            if n[0] % args.every == 0:
                torch.save(a, out / f"call_{n[0]:03d}.pt")  # before the call: it edits some inputs in place
            n[0] += 1
            return inner(*a)

        sw.sunonsurface_2018a = spy
        run_utci_tiles(base_path=str(WORK / "capture_run"), preprocess_dir=str(PRE_1000), selected_date_str=args.date,
                       save_tmrt=False)
    shutil.rmtree(WORK / "capture_run", ignore_errors=True)
    log(f"[gvf] {n[0]} ground view factor calls, saved every {args.every}th")
    recs = []
    for p in sorted(out.glob("call_*.pt")):
        def fresh():
            return list(torch.load(p, map_location="cuda", weights_only=False))

        def timed(fn):
            ts, res = [], None
            for _ in range(args.repeats):
                a = fresh()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                res = fn(*a)
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
            return res, statistics.median(ts)

        ref, t_ref = timed(c2)
        t0 = time.perf_counter()
        compiled(*fresh())
        torch.cuda.synchronize()
        first = time.perf_counter() - t0
        com, t_com = timed(compiled)
        diffs = [float((x.double() - y.double()).abs().max()) for x, y in zip(ref, com) if isinstance(x, torch.Tensor)]
        same = all(torch.equal(x, y) for x, y in zip(ref, com) if isinstance(x, torch.Tensor))
        rel = max(float(((x.double() - y.double()).abs() / x.double().abs().clamp_min(1e-6)).max())
                  for x, y in zip(ref, com) if isinstance(x, torch.Tensor))
        rec = {"call": p.stem, "c2_s": t_ref, "compiled_s": t_com, "first_s": first, "bit_identical": same,
               "max_abs_diff": max(diffs), "max_rel_diff": rel, "graphs_so_far": int(counters["stats"]["unique_graphs"])}
        recs.append(rec)
        log(f"[gvf] {p.stem}: C2 {t_ref * 1e3:6.1f} ms  compiled {t_com * 1e3:6.1f} ms (first {first:4.1f} s)  "
            f"identical {same}  max abs diff {rec['max_abs_diff']:.3g}  max rel {rel:.3g}  graphs {rec['graphs_so_far']}")
    return {"calls": recs}


def days(pre: Path, tag: str, repeats: int, date: str, log) -> dict:
    from solweig_gpu import run_utci_tiles
    from compile_shadow import compiled_shadow

    with compiled_gvf(), compiled_shadow():  # build both from SOLWEIG's own source, before any patch
        pass
    work = WORK / f"day_{tag}"
    p = work / "processed_inputs"
    if not p.exists():
        shutil.copytree(pre, p, ignore=shutil.ignore_patterns("*.bits.npz"))
    res = {"runs": []}
    rng = random.Random(0)
    cfgs = ["base", "gvf", "gvf+shadow"]
    for rep in range(repeats):
        order = list(cfgs)
        rng.shuffle(order)
        for cfg in order:
            out = work / (f"{cfg}_kept" if rep == 0 else "tmp")
            shutil.rmtree(out, ignore_errors=True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with contextlib.ExitStack() as st:
                for cm in day_patches(cfg):
                    st.enter_context(cm)
                run_utci_tiles(base_path=str(out), preprocess_dir=str(p), selected_date_str=date,
                               save_tmrt=True, save_shadow=True)
            torch.cuda.synchronize()
            sec = time.perf_counter() - t0
            res["runs"].append({"config": cfg, "repeat": rep, "seconds": sec})
            log(f"[gvf] day {tag} {cfg:10s} rep {rep}: {sec:.1f} s")
            if rep:
                shutil.rmtree(out, ignore_errors=True)
    for cfg in cfgs[1:]:
        res[f"{cfg}_vs_base"] = differences(work / "base_kept", work / f"{cfg}_kept")
        u, t = res[f"{cfg}_vs_base"]["UTCI"], res[f"{cfg}_vs_base"]["TMRT"]
        log(f"[gvf] day {tag} {cfg} vs base: UTCI max {u['max_abs']:.2g} p99.99 {u['p99_99']:.2g}; "
            f"Tmrt max {t['max_abs']:.2g} p99.99 {t['p99_99']:.2g}; shadow identical "
            f"{res[f'{cfg}_vs_base']['Shadow']['identical']}")
    for cfg in cfgs:
        xs = [r["seconds"] for r in res["runs"] if r["config"] == cfg]
        res[f"{cfg}_first_s"], res[f"{cfg}_median_after_first_s"] = xs[0], statistics.median(xs[1:])
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["all"])
    ap.add_argument("--every", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--date", default="2019-07-30")
    args = ap.parse_args()
    import torch._dynamo
    torch._dynamo.config.cache_size_limit = 256
    torch.zeros(1, device="cuda")
    out = ROOT / "figures" / "compile_gvf.json"
    res = {"torch": torch.__version__}

    def log(msg):
        print(msg, flush=True)

    res["per_call_1000"] = per_call(args, log)
    out.write_text(json.dumps(res, indent=2))
    res["days_1000"] = days(PRE_1000, "1000", 4, args.date, log)
    out.write_text(json.dumps(res, indent=2))
    res["days_2000"] = days(PRE_2000, "2000", 3, args.date, log)
    out.write_text(json.dumps(res, indent=2))
    log(f"[gvf] wrote {out}")


if __name__ == "__main__":
    main()
