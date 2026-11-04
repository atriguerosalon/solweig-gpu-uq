"""Experiment C: torch.compile on one step of the sky-patch shadow loop (the sky view factor step).

Computing a tile's sky view factor casts a shadow for each of the 153 sky patches (shadow.shadow,
with C7), and each step of that loop runs about 25 whole-tile operations. As for the hourly
shadow step (compile_shadow.py), the step is written as a function with fixed shapes (shifted
rasters by index arithmetic, the window passed as GPU data) and compiled. Its operations are
subtractions, comparisons, maxima, small whole-number sums and one maximum over the tile, which
fusion cannot round differently, so the sky view factor cache should stay bit-identical.

    calls   compute one sky view factor (C5, C6, C7), save every k-th patch's arguments, then for
            each saved call: C7 as is against the compiled step; outputs bit for bit, time per
            call, compile time, number of compiled graphs
    svf     whole sky view factor steps, with and without the compiled step, interleaved; every
            cache file and every raster inside the zip compared with the first configuration

Results (3-4 Nov; figures/compile_sky_*.json): cache files byte-identical in every case. Per call
1.7-2.8x faster; whole sky view step 21.2 -> 17.5 s (-17%) at 1000 px, about 200 -> 95 s (-52%) at
2000 px. With CUDA graphs (--mode reduce-overhead) the step is slower than plain C7 (21.3 -> 25.3 s).
The svf mode needs scripts/bench_steps.py (compare_zips).

Usage:
    python scripts/compile_sky.py calls --pre ~/solweig_bench/processed_inputs --tag 1000
    python scripts/compile_sky.py svf --src ~/solweig_verify/pacoima_summer --tag 1000 --repeats 3
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

WORK = Path.home() / "compile_sky"
BODY = [  # C7's loop body, after dz = _dzs[_k], up to index += 1.
    "tempvegdem.zero_()",
    "tempvegdem2.zero_()",
    "temp.zero_()",
    "tempvegdem[xp1:xp2, yp1:yp2] = vegdem[xc1:xc2, yc1:yc2] - dz",
    "tempvegdem2[xp1:xp2, yp1:yp2] = vegdem2[xc1:xc2, yc1:yc2] - dz",
    "temp[xp1:xp2, yp1:yp2] = a[xc1:xc2, yc1:yc2] - dz",
    "f = torch.max(f, temp)",
    "sh[f > a] = 1.",
    "sh[f <= a] = 0.",
    "fabovea = tempvegdem > a",
    "gabovea = tempvegdem2 > a",
    "vegsh2 = fabovea.float() - gabovea.float()",
    "vegsh = torch.max(vegsh, vegsh2)",
    "vegsh[(vegsh * sh > 0.)] = 0.",
    "vbshvegsh = vegsh + vbshvegsh",
    "if index == 1.:",
    "firstvegdem = tempvegdem - temp",
    "firstvegdem[firstvegdem <= 0.] = 1000.",
    "vegsh[firstvegdem < dz] = 1.",
    "vegsh = vegsh * (vegdem2 > a).float()",
    "vbshvegsh.zero_()",
    "if _bush_any:  # same update, chosen on the GPU instead of read back every step",
    "_hit = torch.max(fabovea * bush) > 0.",
    "tempbush.zero_()",
    "tempbush[xp1:xp2, yp1:yp2] = bush[xc1:xc2, yc1:yc2] - dz",
    "g = torch.where(_hit, torch.max(g, tempbush) * bushplant, g)",
]
LOOP = "for _k, (xc1, xc2, yc1, yc2, xp1, xp2, yp1, yp2) in enumerate(_steps):"


def sky_step(f, sh, vegsh, vbshvegsh, g, a, vegdem, vegdem2, bush, bushplant, dz, b, first: bool, bush_any: bool):
    """One step of the sky-patch loop with fixed shapes; same operations and order as C7's body.

    ``b`` holds (row offset, column offset, xp1, xp2, yp1, yp2) as a GPU tensor. Masked
    assignments become torch.where with the same values (a cell keeps its value where neither
    condition holds, e.g. NaN, as with the two assignments).
    """
    n0, n1 = a.shape
    r = torch.arange(n0, device=a.device)
    c = torch.arange(n1, device=a.device)
    inside = ((r >= b[2]) & (r < b[3]))[:, None] & ((c >= b[4]) & (c < b[5]))[None, :]
    rs = (r - b[0]).clamp(0, n0 - 1)[:, None]
    cs = (c - b[1]).clamp(0, n1 - 1)[None, :]

    def shifted(x, d):
        return torch.where(inside, x[rs, cs] - d, 0.0)

    tempvegdem = shifted(vegdem, dz)
    tempvegdem2 = shifted(vegdem2, dz)
    temp = shifted(a, dz)
    f = torch.max(f, temp)
    sh = torch.where(f > a, 1.0, torch.where(f <= a, 0.0, sh))
    fabovea = tempvegdem > a
    gabovea = tempvegdem2 > a
    vegsh2 = fabovea.float() - gabovea.float()
    vegsh = torch.max(vegsh, vegsh2)
    vegsh = torch.where(vegsh * sh > 0.0, 0.0, vegsh)
    vbshvegsh = vegsh + vbshvegsh
    if first:
        firstvegdem = tempvegdem - temp
        firstvegdem = torch.where(firstvegdem <= 0.0, 1000.0, firstvegdem)
        vegsh = torch.where(firstvegdem < dz, 1.0, vegsh)
        vegsh = vegsh * (vegdem2 > a).float()
        vbshvegsh = torch.zeros_like(vbshvegsh)
    if bush_any:
        hit = torch.max(fabovea * bush) > 0.0
        tempbush = shifted(bush, dz)
        g = torch.where(hit, torch.max(g, tempbush) * bushplant, g)
    return f, sh, vegsh, vbshvegsh, g


def step_source(clone: bool = False) -> str:
    """C7's function with the loop body replaced by one call of _step.

    clone: copy the step's outputs before the next call, needed with CUDA graphs (see compile_shadow.py).
    """
    from solweig_sens.fast_sky_shadow import patched_source

    lines = patched_source().splitlines()
    loop = [i for i, l in enumerate(lines) if l.strip() == LOOP]
    starts = [i for i, l in enumerate(lines) if l.strip() == BODY[0]]
    if len(loop) != 1 or len(starts) != 1 or not loop[0] < starts[0]:
        raise RuntimeError("unexpected C7 source")
    s = starts[0]
    code = [i for i in range(s, len(lines)) if lines[i].strip() and not lines[i].strip().startswith("#")]
    if [lines[i].strip() for i in code[:len(BODY)]] != BODY:
        raise RuntimeError("unexpected C7 loop body")
    e = code[len(BODY) - 1] + 1
    li = lines[loop[0]][: len(lines[loop[0]]) - len(lines[loop[0]].lstrip())]
    ind = lines[s][: len(lines[s]) - len(lines[s].lstrip())]
    pre = [f"{li}_B = torch.tensor([[xp1 - xc1, yp1 - yc1, xp1, xp2, yp1, yp2]",
           f"{li}                   for (xc1, xc2, yc1, yc2, xp1, xp2, yp1, yp2) in _steps] or [[0] * 6], device=device)"]
    call = [f"{ind}f, sh, vegsh, vbshvegsh, g = _step(f, sh, vegsh, vbshvegsh, g, a, vegdem, vegdem2, bush, bushplant,",
            f"{ind}                                   dz, _B[_k], index == 1., _bush_any)"]
    if clone:
        call.append(f"{ind}f, sh, vegsh, vbshvegsh, g = (_o.clone() for _o in (f, sh, vegsh, vbshvegsh, g))")
    return "\n".join(lines[:loop[0]] + pre + lines[loop[0]:s] + call + lines[e:]) + "\n"


def build(step, clone: bool = False):
    import solweig_gpu.shadow as shm
    from solweig_sens.fast_sky_shadow import _shadow_steps

    ns = dict(vars(shm))
    ns["_shadow_steps"], ns["_step"] = _shadow_steps, step
    exec(compile(textwrap.dedent(step_source(clone)), shm.__file__, "exec"), ns)
    return ns["shadow"]


_COMPILED = {}


@contextlib.contextmanager
def compiled_sky(mode: str = "default"):
    """shadow.shadow with its loop step compiled. Build happens on first use, which must be before
    any patch is entered (building reads SOLWEIG's own source); enter before tight_bound()."""
    import solweig_gpu.shadow as shm

    if mode not in _COMPILED:
        _COMPILED[mode] = build(torch.compile(sky_step, mode=mode), clone=mode == "reduce-overhead")
    old = shm.shadow
    shm.shadow = _COMPILED[mode]
    try:
        yield
    finally:
        shm.shadow = old


def svf_patches(compiled: bool, mode: str = "default") -> list:
    from solweig_sens.compact_masks import compact_masks
    from solweig_sens.fast_sky_shadow import fast_sky_shadow
    from solweig_sens.tight_bound import tight_bound

    sky = compiled_sky(mode) if compiled else fast_sky_shadow()
    return [compact_masks("bits", svf=True), sky, tight_bound()]


def timed(fn, a, repeats):
    out, ts = None, []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn(*a)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return out, ts


def calls(args) -> None:
    import solweig_gpu.shadow as shm
    import torch._dynamo
    from solweig_gpu import calculate_svf
    from solweig_sens.fast_sky_shadow import build as build_c7
    from solweig_sens.tight_bound import tight
    from torch._dynamo.utils import counters

    torch._dynamo.config.cache_size_limit = 256
    c7 = build_c7()
    compiled = build(torch.compile(sky_step, mode=args.mode), clone=args.mode == "reduce-overhead")
    out = WORK / args.tag
    shutil.rmtree(out, ignore_errors=True)
    pre = out / "processed_inputs"
    shutil.copytree(args.pre, pre, ignore=shutil.ignore_patterns("SVF"))
    n = [0]
    with contextlib.ExitStack() as st:
        for cm in svf_patches(False):
            st.enter_context(cm)
        inner = shm.shadow

        def spy(*a):
            if n[0] % args.every == 0:
                torch.save(a, out / f"call_{n[0]:03d}.pt")
            n[0] += 1
            return inner(*a)

        shm.shadow = spy
        calculate_svf(str(pre), patch_option=2, overwrite=True)
    print(f"[sky] {n[0]} sky-patch shadow calls, saved every {args.every}th", flush=True)
    res = {"tag": args.tag, "mode": args.mode, "calls": []}
    for p in sorted(out.glob("call_*.pt")):
        a = list(torch.load(p, map_location="cuda", weights_only=False))
        a[0] = tight(a[0], a[1], a[2])  # as tight_bound passes it
        ref, t_ref = timed(c7, a, args.repeats)
        t0 = time.perf_counter()
        timed(compiled, a, 1)
        first = time.perf_counter() - t0
        com, t_com = timed(compiled, a, args.repeats)
        same = all(torch.equal(x, y) for x, y in zip(ref, com))
        rec = {"call": p.stem, "altitude": float(a[6]), "c7_s": statistics.median(t_ref),
               "compiled_s": statistics.median(t_com), "first_s": first, "bit_identical": same,
               "graphs_so_far": int(counters["stats"]["unique_graphs"])}
        res["calls"].append(rec)
        print(f"[sky] {p.stem} alt {rec['altitude']:4.0f}: C7 {rec['c7_s'] * 1e3:7.1f} ms  compiled "
              f"{rec['compiled_s'] * 1e3:7.1f} ms (first {first:5.1f} s)  identical {same}  graphs {rec['graphs_so_far']}",
              flush=True)
    path = ROOT / "figures" / f"compile_sky_calls_{args.tag}_{args.mode}.json"
    path.write_text(json.dumps(res, indent=2))
    print(f"[sky] wrote {path}")


def svf(args) -> None:
    from bench_steps import compare_zips
    from solweig_gpu import calculate_svf
    from verify_fast_sky_shadow import compare as compare_files

    with compiled_sky(args.mode):  # build it from SOLWEIG's own source, before any patch is entered
        pass
    torch.zeros(1, device="cuda")
    res = {"tag": args.tag, "mode": args.mode, "src": str(args.src), "runs": []}
    rng = random.Random(0)
    for rep in range(args.repeats):
        order = ["c7", "compiled"]
        rng.shuffle(order)
        for cfg in order:
            work = WORK / f"svf_{args.tag}" / (f"{cfg}_kept" if rep == 0 else "tmp")
            shutil.rmtree(work, ignore_errors=True)
            pre = work / "processed_inputs"
            shutil.copytree(args.src, pre, ignore=shutil.ignore_patterns("SVF"))
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with contextlib.ExitStack() as st:
                for cm in svf_patches(cfg == "compiled", args.mode):
                    st.enter_context(cm)
                calculate_svf(str(pre), patch_option=2, overwrite=True)
            torch.cuda.synchronize()
            sec = time.perf_counter() - t0
            res["runs"].append({"config": cfg, "repeat": rep, "seconds": sec})
            print(f"[sky-svf] {cfg:8s} rep {rep}: {sec:.1f} s", flush=True)
            if rep:
                shutil.rmtree(work, ignore_errors=True)
    base = WORK / f"svf_{args.tag}"
    a, b = base / "c7_kept" / "processed_inputs" / "SVF", base / "compiled_kept" / "processed_inputs" / "SVF"
    n, bad = compare_files(a, b)
    zn = zb = 0
    for z in sorted(a.glob("*.zip")):
        m, d = compare_zips(z, b / z.name)
        zn, zb = zn + m, zb + d
    res |= {"files": n, "files_differ": bad, "zip_members": zn, "zip_members_differ": zb}
    for cfg in ("c7", "compiled"):
        xs = [r["seconds"] for r in res["runs"] if r["config"] == cfg]
        res[f"{cfg}_first_s"] = xs[0]
        res[f"{cfg}_median_after_first_s"] = statistics.median(xs[1:]) if len(xs) > 1 else None
    path = ROOT / "figures" / f"compile_sky_svf_{args.tag}_{args.mode}.json"
    path.write_text(json.dumps(res, indent=2))
    print(f"[sky-svf] {n} files and {zn} zip members compared, {bad + zb} differ; wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calls")
    c.add_argument("--pre", type=Path, required=True, help="preprocessed tile (its SVF folder is left out)")
    c.add_argument("--tag", required=True)
    c.add_argument("--every", type=int, default=19, help="save every k-th of the 153 patch calls")
    c.add_argument("--repeats", type=int, default=5)
    c.add_argument("--mode", choices=["default", "reduce-overhead"], default="default")
    s = sub.add_parser("svf")
    s.add_argument("--src", type=Path, required=True, help="preprocessed tile (its SVF folder is left out)")
    s.add_argument("--tag", required=True)
    s.add_argument("--repeats", type=int, default=3)
    s.add_argument("--mode", choices=["default", "reduce-overhead"], default="default")
    args = ap.parse_args()
    {"calls": calls, "svf": svf}[args.cmd](args)


if __name__ == "__main__":
    main()
