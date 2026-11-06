"""Performance bounds: where SOLWEIG-GPU's time goes, what limits it, and what it would take to go lower.

Two views, from measurements already on disk:

1. The day, function by function (figures/time_budget_current_r5.json, 5 timed passes, or else
   time_budget_current.json, one; where_time_goes.py, our code C1-C9,
   00-14 h, Pacoima crops of 1000 and 2000 px): measured self time (median of the timed passes, each
   run drawn as a dot) against two lower bounds, the bytes
   its operations read and write over the memory bandwidth (~230 GB/s measured) and its kernel launches
   times the cost of one launch (~7 us measured). Bytes are counted per operator as SOLWEIG writes them;
   the GPU's 32 MB L2 cache serves some reuse, so bytes / bandwidth can exceed the measured time a
   little. A function near its bandwidth line is memory-bound: only moving fewer bytes makes it faster.
   Launch costs overlap with GPU work, so the launch line is what CUDA graphs could remove at most.

2. One step of the hourly shadow loop at 2000 px (figures/profile_compiled_steps.json): eager,
   compiled (torch.compile fuses the step's ~30 operators into few kernels), the compiled loop
   captured by hand as one CUDA graph (no launch cost at all), and the bandwidth time of the fused
   step's traffic (state and inputs read once and written once per step). Below that only a
   different algorithm helps, e.g. keeping the loop state on chip across steps (one kernel for the
   whole loop: reads of the three shifted inputs only); that line is an estimate, not built.

Writes figures/profile/bounds.png (and _day.png, _step.png) and figures/profile/bounds.json.

Usage:
    python scripts/plot_bounds.py
"""

from __future__ import annotations

import sys as _sys

if __name__ == "__main__" and any(a in ("-h", "--help") for a in _sys.argv[1:]):
    print(__doc__)
    raise SystemExit(0)

import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
P = ROOT / "figures" / "profile"
INK = "#1B232C"
C_MEAS, C_BW, C_LAUNCH, C_EST = "#0E6A78", "#E8663D", "#7A5AA6", "#9AA6B2"

NAMES = {"shadowingfunction_wallheight_23": "hourly shadows (C3)", "define_patch_characteristics": "sky patches, longwave",
         "sunonsurface_2018a": "ground view factors (C2)", "Kside_veg_v2022a": "shortwave on a person",
         "utci_calculator": "UTCI", "Solweig_2022a_calc": "SOLWEIG step, rest", "load_cached_svf_outputs": "load sky view cache",
         "Lside_veg_v2022a": "longwave on a person", "gvf_2018a": "ground view, rest", "Lcyl_v2022a": "longwave, cylinder",
         "cylindric_wedge": "cylindric wedge", "Kup_veg_2015a": "reflected shortwave", "Tgmaps_v1": "ground temperature maps",
         "Perez_v3": "sky luminance (Perez)", "create_patches": "sky patch grid"}
RASTER_BYTES_2000 = 2000 * 2000 * 4
FUSED_RASTERS = 12   # shadow step: reads a, vegdem, vegdem2 (shifted), f, shvoveg, vegsh, vbshvegsh; writes 5
ONCHIP_RASTERS = 3   # loop state kept on chip: per step only the three shifted inputs are read


def day_bounds(tb: dict, size: str, top: int = 9) -> tuple[list[dict], dict]:
    m, v = tb["machine"], tb["sizes"][size]
    rows = []
    for f, x in v["functions"].items():
        rows.append({"function": f, "label": NAMES.get(f, f), "self_s": x["self_s"],
                     "self_s_runs": x.get("self_s_runs", [x["self_s"]]),
                     "bandwidth_s": x["bytes"] / m["bandwidth_Bps"], "launch_s": x["launches"] * m["launch_s"],
                     "syncs": x["syncs"], "bytes": x["bytes"], "launches": x["launches"]})
    rows.sort(key=lambda r: -r["self_s"])
    rest = rows[top:]
    keys = ("self_s", "bandwidth_s", "launch_s", "syncs", "bytes", "launches")
    rest_row = {"function": "rest", "label": f"{len(rest)} others", **{k: sum(r[k] for r in rest) for k in keys}}
    rest_row["self_s_runs"] = [sum(run) for run in zip(*(r["self_s_runs"] for r in rest))] if rest else []
    rows = rows[:top] + [rest_row]
    tot = {k: sum(r[k] for r in rows) for k in ("self_s", "bandwidth_s", "launch_s")}
    tot["clean_s"] = statistics.median(v["clean_s"])
    tot["clean_n"] = len(v["clean_s"])
    tot["clean_runs"] = list(v["clean_s"])
    tot["timed_n"] = len(v.get("timed_s_runs", [v["timed_s"]]))
    return rows, tot


def draw_day(ax, rows, tot, size):
    y = np.arange(len(rows))
    ax.barh(y, [r["self_s"] for r in rows], color=C_MEAS, height=0.62, alpha=0.85,
            label=("measured" if tot["timed_n"] == 1 else f"measured, median of {tot['timed_n']} runs (dots: each run)"))
    for yi, r in zip(y, rows):  # every timed run, spread a little vertically so they do not overlap
        pts = r["self_s_runs"]
        if len(pts) < 2:
            continue
        offs = np.linspace(-0.18, 0.18, len(pts)) if len(pts) > 1 else [0.0]
        ax.scatter(pts, [yi + o for o in offs], s=9, color=INK, zorder=4, edgecolor="white", linewidth=0.3)
    ax.scatter([r["bandwidth_s"] for r in rows], y, marker="|", s=260, linewidths=2.2, color=C_BW, zorder=3,
               label="bytes moved / bandwidth")
    ax.scatter([r["launch_s"] for r in rows], y, marker="|", s=260, linewidths=2.2, color=C_LAUNCH, zorder=3,
               label="kernel launches x launch cost")
    ax.set_yticks(y, [r["label"] for r in rows], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("time over the day, 00-14 h (s)")
    ax.set_title(f"{size} x {size} px", loc="left", fontsize=10)
    ax.grid(axis="x", alpha=0.3)


def draw_step(ax, prof):
    r = {x["variant"]: x for x in prof["results"] if x["size"] == 2000 and x["step"] == "shadow"}
    fused = FUSED_RASTERS * RASTER_BYTES_2000 / BW * 1e3
    onchip = ONCHIP_RASTERS * RASTER_BYTES_2000 / BW * 1e3
    bars = [("not compiled\n(fixed-shape\nform)", r["eager"]["wall_ms_per_step"], C_MEAS, None, r["eager"]["wall_ms_runs"]),
            ("torch.compile\n(fused)", r["compiled"]["wall_ms_per_step"], C_MEAS, None, r["compiled"]["wall_ms_runs"]),
            ("compiled,\none CUDA graph\n(no launches)", r["compiled"]["captured_ms_per_step"], C_MEAS, None,
             r["compiled"]["captured_ms_runs"]),
            ("bandwidth time\nof fused step\n(12 rasters)", fused, C_BW, None, []),
            ("state on chip\n(3 rasters/step)\nestimate", onchip, C_EST, "//", [])]
    x = np.arange(len(bars))
    for xi, (lab, v, c, h, runs) in zip(x, bars):
        ax.bar(xi, v, color=c if h is None else "white", edgecolor=c, hatch=h, width=0.65, alpha=0.85 if h is None else 1)
        if runs:  # every repeat (median shown as the bar)
            ax.scatter(xi + np.linspace(-0.2, 0.2, len(runs)), runs, s=9, color=INK, zorder=4, edgecolor="white",
                       linewidth=0.3)
        ax.text(xi, v * 1.08, f"{v:.2f} ms", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_yscale("log")
    ax.set_ylim(0.1, 20)
    ax.set_yticks([0.1, 0.2, 0.5, 1, 2, 5, 10], ["0.1", "0.2", "0.5", "1", "2", "5", "10"])
    ax.minorticks_off()
    ax.set_xticks(x, [b[0] for b in bars], fontsize=7.5)
    ax.set_ylabel("one step of the shadow loop (ms)")
    ax.set_title("2000 x 2000 px", loc="left", fontsize=10)
    ax.grid(axis="y", alpha=0.3, which="both")


BW = None


def main() -> None:
    global BW
    from solweig_sens import plotstyle

    # the 5-repeat run on a quiet machine (where_time_goes.py --repeats 5) when there is one, else the single-pass night run
    tb_name = next(n for n in ("time_budget_current_r5.json", "time_budget_current.json") if (ROOT / "figures" / n).exists())
    tb = json.loads((ROOT / "figures" / tb_name).read_text())
    prof = json.loads((ROOT / "figures" / "profile_compiled_steps.json").read_text())
    BW = tb["machine"]["bandwidth_Bps"]
    out = {"machine": tb["machine"], "day": {}, "shadow_step_2000": {}}
    for s in ("1000", "2000"):
        rows, tot = day_bounds(tb, s)
        out["day"][s] = {"rows": rows, "totals": tot}
    r = {x["variant"]: x for x in prof["results"] if x["size"] == 2000 and x["step"] == "shadow"}
    out["shadow_step_2000"] = {"eager_ms": r["eager"]["wall_ms_per_step"], "compiled_ms": r["compiled"]["wall_ms_per_step"],
                               "captured_graph_ms": r["compiled"]["captured_ms_per_step"],
                               "bandwidth_fused_ms": FUSED_RASTERS * RASTER_BYTES_2000 / BW * 1e3,
                               "estimate_state_on_chip_ms": ONCHIP_RASTERS * RASTER_BYTES_2000 / BW * 1e3}
    (P / "bounds.json").write_text(json.dumps(out, indent=2))

    plotstyle.use()
    # no title or caption in the image; the caption (machine, sources, what the lines mean) is in docs/FIGURES.md
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.4), gridspec_kw={"width_ratios": [1, 1, 0.9]})
    for ax, s in zip(axes[:2], ("1000", "2000")):
        draw_day(ax, out["day"][s]["rows"], out["day"][s]["totals"], s)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    draw_step(axes[2], prof)
    for ax, letter in zip(axes, "abc"):
        ax.text(-0.02 if letter == "c" else -0.25, 1.03, letter, transform=ax.transAxes, fontsize=13,
                fontweight="bold", va="bottom", ha="right" if letter == "c" else "left")
    fig.tight_layout()
    fig.savefig(P / "bounds.png", dpi=150)
    plt.close(fig)
    for name in ("bounds_day.png", "bounds_step.png"):
        if name == "bounds_day.png":
            fig, axes2 = plt.subplots(1, 2, figsize=(14, 5.2))
            for ax, s in zip(axes2, ("1000", "2000")):
                draw_day(ax, out["day"][s]["rows"], out["day"][s]["totals"], s)
            axes2[0].legend(frameon=False, fontsize=8, loc="lower right")
        else:
            fig, ax = plt.subplots(figsize=(7, 5.2))
            draw_step(ax, prof)
        fig.tight_layout()
        fig.savefig(P / name, dpi=150)
        plt.close(fig)
    print(f"wrote {P / 'bounds.png'}, bounds_day.png, bounds_step.png, bounds.json")


if __name__ == "__main__":
    main()
