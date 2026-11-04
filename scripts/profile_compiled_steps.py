"""How much of a compiled loop step is launch overhead? The most CUDA graphs could ever save.

The shadow step (compile_shadow.py, shadow_step_fixed) and the sky-patch step (compile_sky.py,
sky_step) run in a loop of N steps with the state carried from step to step, as in SOLWEIG, on
synthetic rasters of 1000 x 1000 and 2000 x 2000 (the cost of these steps does not depend on the
values: elementwise operations and fixed-shape gathers). Three variants:

    eager      the step as written, no compilation
    compiled   torch.compile, default mode
    graphs     torch.compile, mode "reduce-overhead" (CUDA graphs), outputs copied after each step
               as compile_shadow.py does (a replay overwrites its output buffers)

Per step: wall time (one synchronisation after the N steps, not per step). For compiled,
the same N-step loop is also captured by hand as one CUDA graph (torch.cuda.graph, state in fixed
buffers, no copies) and replayed: that is the loop's GPU time with no launch overhead. wall minus
that is the launch (and Python) overhead: the most CUDA graphs can remove, whatever the
implementation. (torch.profiler records no GPU kernels under WSL2 here, so it is not used.)

Usage:
    python scripts/profile_compiled_steps.py --sizes 1000 2000
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

N = 60


def inputs(n: int, kind: str):
    g = torch.Generator(device="cuda").manual_seed(0)
    r = lambda: torch.rand((n, n), device="cuda", generator=g)  # noqa: E731
    a = r() * 20
    vegdem, vegdem2 = a + r() * 8, a + r() * 2
    state = [torch.zeros((n, n), device="cuda") for _ in range(4 if kind == "shadow" else 5)]
    bs = [torch.tensor([k + 1, k // 2 + 1, k + 1, n, k // 2 + 1, n], device="cuda") for k in range(N)]
    dzs = [torch.tensor(0.5 * (k + 1), device="cuda") for k in range(N + 1)]
    bush, bushplant = (r() > 0.9).float(), (r() > 0.5).float()
    return a, vegdem, vegdem2, state, bs, dzs, bush, bushplant


def loop(step, kind, data, clone=False):
    a, vegdem, vegdem2, state, bs, dzs, bush, bushplant = data
    if kind == "shadow":
        f, shvoveg, vegsh, vbshvegsh = state
        for k in range(N):
            f, shvoveg, sh, vegsh, vbshvegsh = step(f, shvoveg, vegsh, vbshvegsh, a, vegdem, vegdem2, dzs[k + 1],
                                                    dzs[k], bs[k])
            if clone:
                f, shvoveg, sh, vegsh, vbshvegsh = (x.clone() for x in (f, shvoveg, sh, vegsh, vbshvegsh))
        return vbshvegsh
    f, sh, vegsh, vbshvegsh, g = state
    for k in range(N):
        f, sh, vegsh, vbshvegsh, g = step(f, sh, vegsh, vbshvegsh, g, a, vegdem, vegdem2, bush, bushplant, dzs[k + 1],
                                          bs[k], False, True)
        if clone:
            f, sh, vegsh, vbshvegsh, g = (x.clone() for x in (f, sh, vegsh, vbshvegsh, g))
    return vbshvegsh


def captured(run):
    """The whole N-step loop captured as one CUDA graph by hand (torch.cuda.graph): its replay has no
    launch overhead and no copies in or out, so it is the GPU time of the loop itself."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):  # warm-up on a side stream, as the CUDA graph docs ask
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run()
    return graph, out


def main() -> None:
    from compile_shadow import shadow_step_fixed
    from compile_sky import sky_step

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="+", default=[1000, 2000])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", type=Path, default=ROOT / "figures" / "profile_compiled_steps.json")
    args = ap.parse_args()
    torch._dynamo.config.cache_size_limit = 64
    res = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "steps_per_loop": N, "results": []}
    steps = {"shadow": shadow_step_fixed, "sky": sky_step}
    for n in args.sizes:
        for kind, fn in steps.items():
            data = inputs(n, kind)
            variants = {"eager": (fn, False), "compiled": (torch.compile(fn), False),
                        "graphs": (torch.compile(fn, mode="reduce-overhead"), True)}
            ref = None
            for name, (step, clone) in variants.items():
                run = lambda: loop(step, kind, data, clone)  # noqa: E731
                for _ in range(3):  # compile, record graphs, warm up
                    out = run()
                torch.cuda.synchronize()
                same = None if ref is None else bool(torch.equal(out, ref))
                ref = out if ref is None else ref
                ts = []
                for _ in range(args.repeats):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    run()
                    torch.cuda.synchronize()
                    ts.append((time.perf_counter() - t0) / N * 1e3)
                wall = statistics.median(ts)
                rec = {"size": n, "step": kind, "variant": name, "wall_ms_per_step": wall, "wall_ms_runs": ts,
                       "same_as_eager": same}
                if name == "compiled":  # the same loop as one hand-captured graph (the eager step copies host scalars: not capturable)
                    graph, gout = captured(run)
                    gs = []
                    for _ in range(args.repeats):
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        graph.replay()
                        torch.cuda.synchronize()
                        gs.append((time.perf_counter() - t0) / N * 1e3)
                    rec["captured_ms_per_step"] = statistics.median(gs)
                    rec["captured_ms_runs"] = gs
                    rec["captured_same_as_eager"] = bool(torch.equal(gout, ref))
                    rec["launch_overhead_ms_per_step"] = wall - rec["captured_ms_per_step"]
                    rec["launch_overhead_share"] = 1 - rec["captured_ms_per_step"] / wall
                    del graph
                res["results"].append(rec)
                extra = (f", one captured graph {rec['captured_ms_per_step']:6.3f} ms/step: launch overhead "
                         f"{100 * rec['launch_overhead_share']:4.1f}% (same: {rec['captured_same_as_eager']})"
                         if "captured_ms_per_step" in rec else "")
                print(f"[prof] {n} px {kind:6s} {name:8s}: {wall:6.3f} ms/step{extra}"
                      f"{'' if same is None else f', same as eager: {same}'}", flush=True)
            del data
            torch.cuda.empty_cache()
    args.out.write_text(json.dumps(res, indent=2))
    print(f"[prof] wrote {args.out}")


if __name__ == "__main__":
    main()
