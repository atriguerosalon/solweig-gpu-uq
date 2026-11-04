"""Summary figures of the SOLWEIG-GPU performance work, from the measured JSON reports.

Each figure is saved on its own, and all four together in one overview:
  summary_time.png      time per tile-day, step by step (C1-C9, then torch.compile as an experiment)
  summary_memory.png    peak GPU memory, step by step
  summary_svf.png       the one-off sky view factor step, step by step
  summary_bigtile.png   the 2000 x 2000 tile on an 8 GB GPU
  summary.png           the four together

Inputs (figures/profile/): bench_steps.json and bench_steps_b.json (scripts/bench_steps.py, two
sessions on a quiet machine on AC, 3-4 Nov 2026, 5 runs per day step and 3 per sky view step each),
bigtile_demo.json with bigtile_compare_current.json (C1-C9, 4 Nov) or bigtile_compare.json (30 Oct code)
(optional).

Usage:
    python scripts/plot_summary.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

P = Path(__file__).resolve().parents[1] / "figures" / "profile"
INK, MUTED = "#1B232C", "#56616D"
C_ORIG, C_FAST, C_BEST, C_EXP = "#9AA6B2", "#4C78A8", "#0E6A78", "#B07AA1"
SESSION_DOTS = (INK, "#E8663D")  # session 1, session 2
SESSIONS = ("bench_steps.json", "bench_steps_b.json")
NOTE = ("RTX 4070 Laptop GPU (8 GB), WSL2, quiet machine on AC, 3-4 Nov 2026. Bars: median of all runs; dots: each "
        "run, session 1 dark, session 2 orange.")
COLD = {"I", "J"}  # first run writes the packed cache / compiles: left out of the median and the dots

# (step, label, colour). G2 and I are the two halves of C9; J is an experiment, not one of the patches.
DAY = [("A", "original,\nWindows\ndrive", C_ORIG), ("B", "original,\nLinux\ndisk", C_ORIG),
       ("C", "C1\nfast sky\n(chunk 64)", C_FAST), ("D", "C2\nfast view\nfactors", C_FAST),
       ("E", "C3\nfast\nshadows", C_FAST), ("E8", "sky\nchunk\n64 → 8", C_FAST),
       ("F", "C4/C5\nbit-packed\nmasks", C_BEST), ("G", "C6\ntight loop\nbound", C_BEST),
       ("G2", "C9\nparallel\nmask load", C_BEST), ("H", "C8\ngeometry\nonce", C_BEST),
       ("I", "C9\npacked\ncache", C_BEST), ("J", "torch.\ncompile\nshadows", C_EXP)]
SVF = [("svf_original", "original", C_ORIG), ("svf_compact", "C5 compact\nmasks", C_BEST),
       ("svf_tight", "+ C6 tight\nbound", C_BEST), ("svf_c7", "+ C7 no syncs\nin patch loop", C_BEST)]


def load(name):
    f = P / name
    return json.loads(f.read_text()) if f.exists() else None


def runs_by_step(reports):
    """{step: [(session, seconds, peak_allocated, peak_reserved), ...]}, cold first runs of I and J left out."""
    out = {}
    for k, rep in enumerate(reports):
        for r in rep["runs"]:
            if r["step"] in COLD and r["repeat"] == 0:
                continue
            out.setdefault(r["step"], []).append((k, r["seconds"], r["peak_allocated_gib"], r["peak_reserved_gib"]))
    return out


def bars_with_runs(ax, x, rows, by, label_fmt="{:.1f} s"):
    """Bar = median over all runs of both sessions; dots = each run, coloured by session."""
    meds = [statistics.median(r[1] for r in by[s]) for s, _, _ in rows]
    ax.bar(x, meds, color=[c for _, _, c in rows], alpha=0.85, width=0.7)
    for xi, (s, _, _), m in zip(x, rows, meds):
        pts = sorted(by[s])
        offs = np.linspace(-0.24, 0.24, len(pts)) if len(pts) > 1 else [0.0]
        for o, (k, sec, _, _) in zip(offs, pts):
            ax.scatter(xi + o, sec, s=12, color=SESSION_DOTS[k], zorder=3, edgecolor="white", linewidth=0.3)
        top = max(m, max(p[1] for p in pts))
        ax.text(xi, top * 1.02 + 0.5, label_fmt.format(m), ha="center", va="bottom", fontsize=8, color=INK)
    return meds


def draw_time(ax, by):
    # The first bar is a setup change (where WSL writes outputs), not a code change, and the last is the
    # torch.compile experiment: both set apart.
    x = [0] + [i + 0.5 for i in range(1, len(DAY) - 1)] + [len(DAY) - 1 + 1.0]
    meds = bars_with_runs(ax, x, DAY, by)
    ymax = max(max(r[1] for r in by[s]) for s, _, _ in DAY) * 1.15
    for xv in (0.75, x[-1] - 0.75):
        ax.axvline(xv, color=MUTED, lw=0.8, ls=":")
    ax.text(0, meds[0] / 2, "setup,\nnot\ncode", ha="center", va="center", fontsize=8, color="white")
    ax.text(x[1] - 0.35, ymax * 0.97, "the patches (C1-C9), outputs on the Linux disk", ha="left", va="top",
            fontsize=8, color=MUTED)
    ax.text(x[-1], ymax * 0.97, "experiment", ha="center", va="top", fontsize=8, color=C_EXP)
    ax.set_xticks(x, [lab for _, lab, _ in DAY], fontsize=8)
    ax.set_ylabel("one simulated day, 1000 x 1000 tile (s)")
    ax.set_ylim(0, ymax)
    ax.set_title(f"Time per tile-day: {meds[1]:.0f} s → {meds[-2]:.1f} s with C1-C9 ({meds[1] / meds[-2]:.1f}x), "
                 "same outputs", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)


def draw_memory(ax, by):
    x = np.arange(len(DAY))
    w = 0.38
    alloc = [statistics.median(r[2] for r in by[s]) for s, _, _ in DAY]
    resv = [statistics.median(r[3] for r in by[s]) for s, _, _ in DAY]
    cols = [c for _, _, c in DAY]
    ax.bar(x - w / 2, alloc, w, color=cols, label="peak allocated")
    ax.bar(x + w / 2, resv, w, color=cols, alpha=0.45, label="peak reserved (decides out-of-memory)")
    for i, r in enumerate(resv):
        ax.text(i + w / 2, r + 0.06, f"{r:.2f}", ha="center", fontsize=7.5, color=INK)
    ax.axhline(resv[1], color=MUTED, ls="--", lw=0.9)
    ax.set_xticks(x, [lab for _, lab, _ in DAY], fontsize=7.5)
    ax.set_ylabel("GPU memory over the day (GiB)")
    ax.set_ylim(0, max(resv) * 1.2)
    ax.set_title("Peak memory: chunk 64 traded memory for speed until profiled; compact masks cut it further",
                 loc="left", fontweight="bold", fontsize=10.5)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)


def draw_svf(ax, by, ident):
    x = list(range(len(SVF)))
    meds = bars_with_runs(ax, x, SVF, by, "{:.0f} s")
    for xi, (s, _, _), m in zip(x, SVF, meds):
        mem = statistics.median(r[3] for r in by[s])
        ax.text(xi, m / 2, f"{mem:.2f}\nGiB", ha="center", va="center", fontsize=8, color="white")
    ax.set_xticks(x, [lab for _, lab, _ in SVF], fontsize=8)
    ax.set_ylabel("sky view step, 1000 x 1000 tile (s)")
    ax.set_ylim(0, max(max(r[1] for r in by[s]) for s, _, _ in SVF) * 1.18)
    ax.set_title(f"One-off sky view step: {meds[0]:.0f} s → {meds[-1]:.0f} s, {ident}", loc="left",
                 fontweight="bold", fontsize=10.5)
    ax.grid(axis="y", alpha=0.3)


def draw_bigtile(ax, big, cmp):
    total = big.get("gpu_total_gib", 8.0)
    if "day_current" in big:  # the code of C1-C9 (bigtile_demo.py --current); else the 30 Oct code
        ours = [("sky view, C5-C7", big["svf_current"], C_BEST), ("day, C1-C9", big["day_current"], C_BEST)]
    else:
        ours = [("sky view, compact", big["svf_compact"], C_BEST), ("day, fast + compact", big["day_fast_bits"], C_BEST)]
    rows = [("sky view, original", big["svf_original"], C_ORIG), ("day, original", big["day_original"], C_ORIG), *ours]
    vmax = max(total, max(r[1]["peak_reserved_gib"] for r in rows))
    for i, (lab, r, c) in enumerate(rows):
        ok = r["status"] == "ok"
        v = r["peak_reserved_gib"] if ok else total
        over = ok and v > total
        ax.barh(i, v, color=c if ok else "white", edgecolor=c, hatch="//" if (over or not ok) else None)
        if not ok:
            txt = "out of memory"
        else:
            m = r["seconds"] / 60
            txt = f"{v:.1f} GiB, {m:.0f} min (n = 1)" if m >= 10 else f"{v:.1f} GiB, {m:.1f} min (n = 1)"
            if over:
                txt += "\nover the GPU's 8 GB: spilled to system RAM\n(Windows driver allows it; Linux would fail)"
        ax.text(v + 0.15, i, txt, va="center", fontsize=8.5)
    ax.axvline(total, color=MUTED, ls="--", lw=0.9)
    ax.text(total, len(rows) - 0.45, f" {total:.0f} GiB GPU", fontsize=8, color=MUTED)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=9)
    ax.set_xlim(0, vmax * 1.75)
    ax.set_xlabel("peak reserved GPU memory (GiB)")
    title = f"A {big['tile']} tile (4 x 4 km) on the 8 GB laptop GPU"
    if cmp:
        title += f"\noutputs agree to {max(cmp['UTCI']['max_abs'], cmp['TMRT']['max_abs']):.0e} °C"
    ax.set_title(title, loc="left", fontweight="bold", fontsize=10.5)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)


def main() -> None:
    reports = [r for r in (load(n) for n in SESSIONS) if r]
    by = runs_by_step(reports)
    same = all(c.get("differ", 1) == 0 and c.get("zip_members_differ", 1) == 0
               for rep in reports for k, c in rep["comparisons"].items() if k.startswith("svf_"))
    ident = "identical cache files" if same else "CACHE FILES DIFFER"
    big = load("bigtile_demo.json")
    current = big is not None and "day_current" in big
    cmp = (load("bigtile_compare_current.json") or {}).get("original_vs_current") if current else load("bigtile_compare.json")
    big_note = ("Big tile: original 30 Oct (GPU alone), C1-C9 4 Nov, single runs" if current
                else "Big tile: 30 Oct, code of that day, single runs")
    singles = [("summary_time.png", (11, 4.8), lambda ax: draw_time(ax, by)),
               ("summary_memory.png", (11, 4.6), lambda ax: draw_memory(ax, by)),
               ("summary_svf.png", (6, 4.4), lambda ax: draw_svf(ax, by, ident))]
    if big:
        singles.append(("summary_bigtile.png", (7.8, 4.4), lambda ax: draw_bigtile(ax, big, cmp)))
    for name, size, draw in singles:
        fig, ax = plt.subplots(figsize=size)
        draw(ax)
        note = f"RTX 4070 Laptop GPU (8 GB), WSL2. {big_note}." if name == "summary_bigtile.png" else NOTE
        fig.text(0.01, 0.005, note, fontsize=7, color=MUTED)
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        fig.savefig(P / name, dpi=150)
        plt.close(fig)
        print(P / name)

    fig, axes = plt.subplots(2, 2, figsize=(16, 9), gridspec_kw={"width_ratios": [1.9, 1]})
    draw_time(axes[0, 0], by)
    draw_memory(axes[1, 0], by)
    draw_svf(axes[0, 1], by, ident)
    if big:
        draw_bigtile(axes[1, 1], big, cmp)
    else:
        axes[1, 1].set_axis_off()
    fig.suptitle("SOLWEIG-GPU v2.0.0: speed and memory, same results (Pacoima tile)", x=0.01, ha="left",
                 fontweight="bold", fontsize=13)
    fig.text(0.01, 0.005, NOTE + " First runs of the packed cache copy and of torch.compile (cold) left out.\n"
             f"Outputs bit-identical step to step except C1 and the chunk change (rounding, ≤0.0001 °C). {big_note}. "
             "Sources: figures/profile/*.json", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    fig.savefig(P / "summary.png", dpi=150)
    plt.close(fig)
    print(P / "summary.png")


if __name__ == "__main__":
    main()
