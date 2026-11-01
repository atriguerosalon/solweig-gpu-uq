"""GPU memory and time, original SOLWEIG-GPU against each fast replacement and all combined.

For each variant (original; + fast sky; + fast ground view factors; all three), separate runs so
that instrumentation does not distort the other measurements:
  1. whole-day peak allocated and peak reserved memory, wall time (no instrumentation)
  2. per component: time (with GPU sync at the boundaries) and transient peak memory per call
  3. allocator trace of one daytime hour (torch.cuda.memory._record_memory_history), every
     allocation attributed to the SOLWEIG function that made it (innermost matching frame)
Plus a chunk-size sweep for the fast sky (memory against speed), and peaks on a second tile size
to fit memory per pixel and predict the largest tile per GPU memory size.

Run with the GPU to itself. Writes figures/profile/memory_report.json and memory_*.png.

Usage:
    python scripts/profile_memory.py [--hour 13] [--skip-trace] [--skip-sweep] [--skip-scaling]
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from solweig_sens.fast import build_fast_functions, no_tf32  # noqa: E402

GIB = 2**30
LIN = Path.home() / "solweig_verify"
CONFIGS = {"pacoima_1000": (LIN / "pacoima_summer", "2019-07-30"),
           "pacoima_tiles600": (LIN / "tiles600" / "shared" / "processed_inputs", "2019-07-30")}
OUT = ROOT / "figures" / "profile"

# Components for attribution, by function name (innermost matching frame wins).
COMPONENTS = {
    "shadow casting": {"shadowingfunction_wallheight_23", "shadowingfunction_wallheight_13", "_shadow_steps"},
    "ground view factors": {"gvf_2018a", "sunonsurface_2018a", "_gvf_offsets"},
    "sky longwave (patches)": {"Lcyl_v2022a", "define_patch_characteristics", "_wsum", "model2"},
    "shortwave and side fluxes": {"Kside_veg_v2022a", "Kup_veg_2015a", "Perez_v3", "cylindric_wedge", "Kvikt_veg",
                                  "Lside_veg_v2022a", "Lvikt_veg", "clearnessindex_2013b", "diffusefraction",
                                  "TsWaveDelay_2015a", "daylen"},
    "UTCI": {"utci_calculator", "utci_polynomial"},
}
COLORS = {"geometry and state (before the hour)": "#2E6FD8", "shadow casting": "#E8663D",
          "ground view factors": "#1FAD7E", "sky longwave (patches)": "#E77FA8",
          "shortwave and side fluxes": "#E8A317", "UTCI": "#0B7A1E", "other (in Solweig_2022a_calc)": "#B5B5B5"}
# Functions timed and measured per call (none of them calls another one in this list).
MEASURED = ["shadowingfunction_wallheight_23", "gvf_2018a", "Lcyl_v2022a", "Kside_veg_v2022a"]


def variants(fast: dict) -> dict:
    sky, gvf, shadow = (fast["define_patch_characteristics"], fast["sunonsurface_2018a"],
                        fast["shadowingfunction_wallheight_23"])
    return {
        "original": {},
        "+ fast sky": {"define_patch_characteristics": sky},
        "+ fast sky + view factors": {"define_patch_characteristics": sky, "sunonsurface_2018a": gvf},
        "all three": {"define_patch_characteristics": sky, "sunonsurface_2018a": gvf,
                      "shadowingfunction_wallheight_23": shadow},
    }


@contextlib.contextmanager
def installed(replacements: dict):
    import solweig_gpu.solweig as sw

    old = {k: getattr(sw, k) for k in replacements}
    try:
        for k, v in replacements.items():
            setattr(sw, k, v)
        with no_tf32():
            yield
    finally:
        for k, v in old.items():
            setattr(sw, k, v)


def run_day(pre: Path, date: str, out: Path) -> float:
    import torch
    from solweig_gpu import run_utci_tiles

    shutil.rmtree(out, ignore_errors=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    run_utci_tiles(base_path=str(out), preprocess_dir=str(pre), selected_date_str=date, save_tmrt=True)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    shutil.rmtree(out, ignore_errors=True)
    return dt


def measure_peak(pre: Path, date: str, replacements: dict) -> dict:
    import torch

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    with installed(replacements):
        seconds = run_day(pre, date, LIN / "mem_tmp")
    return {"seconds": seconds,
            "peak_allocated_gib": (torch.cuda.max_memory_allocated() - base) / GIB,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / GIB}


def measure_components(pre: Path, date: str, replacements: dict) -> dict:
    import solweig_gpu.solweig as sw
    import torch

    stats = defaultdict(lambda: {"calls": 0, "seconds": 0.0, "max_transient_gib": 0.0, "max_level_gib": 0.0})

    def wrap(name, fn):
        @functools.wraps(fn)
        def w(*a, **k):
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            r = fn(*a, **k)
            torch.cuda.synchronize()
            s = stats[name]
            s["calls"] += 1
            s["seconds"] += time.perf_counter() - t0
            peak = torch.cuda.max_memory_allocated()
            s["max_transient_gib"] = max(s["max_transient_gib"], (peak - before) / GIB)
            s["max_level_gib"] = max(s["max_level_gib"], peak / GIB)
            return r
        return w

    with installed(replacements):
        old = {n: getattr(sw, n) for n in MEASURED}
        try:
            for n in MEASURED:
                setattr(sw, n, wrap(n, getattr(sw, n)))
            run_day(pre, date, LIN / "mem_tmp")
        finally:
            for n, f in old.items():
                setattr(sw, n, f)
    return dict(stats)


def trace_hour(pre: Path, date: str, replacements: dict, hour: int, compact: str | None = None) -> dict:
    """Allocator events during one call of Solweig_2022a_calc, attributed to components."""
    import solweig_gpu.utci_process as up
    import torch

    from solweig_sens.compact_masks import compact_masks

    result = {}
    original_calc = up.Solweig_2022a_calc

    def wrapped(i, *a, **k):
        if int(i) != hour:
            return original_calc(i, *a, **k)
        torch.cuda.synchronize()
        result["start_allocated"] = torch.cuda.memory_allocated()
        torch.cuda.memory._record_memory_history(enabled="all", context="all", stacks="python", max_entries=3_000_000)
        try:
            r = original_calc(i, *a, **k)
            torch.cuda.synchronize()
            result["snapshot"] = torch.cuda.memory._snapshot()
        finally:
            torch.cuda.memory._record_memory_history(enabled=None)
        return r

    with installed(replacements), (compact_masks(compact) if compact else contextlib.nullcontext()):
        up.Solweig_2022a_calc = wrapped
        try:
            run_day(pre, date, LIN / "mem_tmp")
        finally:
            up.Solweig_2022a_calc = original_calc

    return replay(result["snapshot"], result["start_allocated"])


def component_of(frames) -> str:
    for f in frames or []:
        name = f.get("name", "")
        for comp, names in COMPONENTS.items():
            if name in names:
                return comp
    return "other (in Solweig_2022a_calc)"


def replay(snapshot: dict, start_allocated: int) -> dict:
    events = [e for e in snapshot["device_traces"][0] if e["action"] in ("alloc", "free_completed")]
    comps = list(COLORS)
    level = {c: 0 for c in comps}
    level["geometry and state (before the hour)"] = start_allocated
    owner = {}
    series = []
    for e in events:
        if e["action"] == "alloc":
            c = component_of(e.get("frames"))
            owner[e["addr"]] = (c, e["size"])
            level[c] += e["size"]
        else:
            c, size = owner.pop(e["addr"], ("geometry and state (before the hour)", e["size"]))
            level[c] -= size
        series.append([level[c] for c in comps])
    arr = np.array(series, dtype=np.float64) / GIB if series else np.zeros((0, len(comps)))
    # Downsample for plotting and storage.
    step = max(1, len(arr) // 4000)
    total = arr.sum(axis=1) if len(arr) else np.zeros(0)
    return {"components": comps, "n_events": len(arr), "series": arr[::step].tolist(), "step": step,
            "peak_gib": float(total.max()) if len(total) else 0.0,
            "start_gib": start_allocated / GIB}


def plot_traces(traces: dict, path: Path, budget_gib: float = 8.0) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(traces)
    fig, axes = plt.subplots(1, len(names), figsize=(max(6.4, 4.6 * len(names)), 4.4), sharey=True)
    axes = np.atleast_1d(axes)
    ymax = max(budget_gib, max(t["peak_gib"] for t in traces.values())) * 1.12
    for ax, name in zip(axes, names):
        t = traces[name]
        arr = np.array(t["series"])
        x = np.arange(len(arr)) * t["step"] / 1000.0
        ax.stackplot(x, arr.T, colors=[COLORS[c] for c in t["components"]], linewidth=0)
        ax.axhline(budget_gib, color="0.35", ls="--", lw=1)
        ax.text(x[-1] if len(x) else 0, budget_gib, f"{budget_gib:.0f} GiB laptop GPU ", ha="right", va="bottom",
                fontsize=8, color="0.35")
        ax.set_title(name, loc="left", fontsize=11)
        ax.text(0.5, 0.93, f"peak {t['peak_gib']:.2f} GiB", transform=ax.transAxes, ha="center",
                fontsize=10, fontweight="bold")
        ax.set_xlabel(f"allocator events (thousands), {t['n_events']:,} in total")
        ax.set_ylim(0, ymax)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("active GPU memory (GiB)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[c]) for c in COLORS]
    fig.legend(handles, list(COLORS), loc="lower center", ncol=4 if len(names) >= 3 else (3 if len(names) == 2 else 2),
               frameon=False, fontsize=9)
    title = ("SOLWEIG-GPU memory during one daytime hour (13:00), 1000 x 1000 tile" if len(names) > 1
             else "GPU memory, 13:00 hour, 1000 x 1000 tile")
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, {1: 0.22, 2: 0.14}.get(len(names), 0.1), 1, 0.95))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hour", type=int, default=13)
    ap.add_argument("--skip-trace", action="store_true")
    ap.add_argument("--skip-sweep", action="store_true")
    ap.add_argument("--skip-scaling", action="store_true")
    ap.add_argument("--final-trace", action="store_true",
                    help="only trace the original against all fixes + bit-packed masks, then exit")
    args = ap.parse_args()

    import torch

    torch.zeros(1, device="cuda")
    OUT.mkdir(parents=True, exist_ok=True)
    fast = build_fast_functions()
    vs = variants(fast)
    if args.final_trace:
        pre, date = CONFIGS["pacoima_1000"]
        traces = {"original": trace_hour(pre, date, vs["original"], args.hour),
                  "all fixes + bit-packed masks": trace_hour(pre, date, vs["all three"], args.hour, compact="bits")}
        budget = torch.cuda.get_device_properties(0).total_memory / GIB
        for k, t in traces.items():
            print(f"[mem] {k}: events {t['n_events']:,}, start {t['start_gib']:.2f} GiB, peak {t['peak_gib']:.2f} GiB")
            slug = "original" if k == "original" else "final"
            np.savez_compressed(OUT / f"memory_trace_{slug}.npz", series=np.array(t["series"]), step=t["step"],
                                n_events=t["n_events"], components=np.array(t["components"]))
            plot_traces({k: t}, OUT / f"memory_trace_{slug}.png", budget_gib=budget)  # each on its own
        plot_traces(traces, OUT / "memory_trace_compare.png", budget_gib=budget)  # side by side
        (OUT / "memory_trace_final.json").write_text(json.dumps(
            {k: {kk: vv for kk, vv in v.items() if kk != "series"} for k, v in traces.items()}, indent=2))
        return
    pre, date = CONFIGS["pacoima_1000"]
    report = {"gpu": torch.cuda.get_device_name(0), "total_gib": torch.cuda.get_device_properties(0).total_memory / GIB,
              "peak": {}, "components": {}, "traces": {}, "chunk_sweep": {}, "scaling": {}}

    for name, rep in vs.items():
        print(f"[mem] peak: {name}", flush=True)
        report["peak"][name] = measure_peak(pre, date, rep)
        print(f"[mem]   {report['peak'][name]}", flush=True)
    for name, rep in vs.items():
        print(f"[mem] components: {name}", flush=True)
        report["components"][name] = measure_components(pre, date, rep)

    if not args.skip_sweep:
        from solweig_sens.fast_sky import define_patch_characteristics as sky_fn
        for chunk in (8, 16, 32, 64, 153):
            rep = dict(vs["all three"])
            rep["define_patch_characteristics"] = functools.partial(sky_fn, chunk=chunk)
            print(f"[mem] chunk {chunk}", flush=True)
            report["chunk_sweep"][str(chunk)] = measure_peak(pre, date, rep)
            print(f"[mem]   {report['chunk_sweep'][str(chunk)]}", flush=True)

    if not args.skip_scaling and CONFIGS["pacoima_tiles600"][0].exists():
        pre6, date6 = CONFIGS["pacoima_tiles600"]
        for name in ("original", "all three"):
            print(f"[mem] scaling, 620-pixel tiles: {name}", flush=True)
            report["scaling"][name] = measure_peak(pre6, date6, vs[name])

    if not args.skip_trace:
        for name in ("original", "+ fast sky", "all three"):
            print(f"[mem] trace hour {args.hour}: {name}", flush=True)
            report["traces"][name] = trace_hour(pre, date, vs[name], args.hour)
            print(f"[mem]   events {report['traces'][name]['n_events']:,}, peak {report['traces'][name]['peak_gib']:.2f} GiB",
                  flush=True)
        plot_traces(report["traces"], OUT / "memory_trace_hour.png", budget_gib=report["total_gib"])

    slim = {k: v for k, v in report.items() if k != "traces"}
    slim["traces"] = {k: {kk: vv for kk, vv in v.items() if kk != "series"} for k, v in report["traces"].items()}
    (OUT / "memory_report.json").write_text(json.dumps(slim, indent=2))
    print(json.dumps(slim, indent=2))


if __name__ == "__main__":
    main()
