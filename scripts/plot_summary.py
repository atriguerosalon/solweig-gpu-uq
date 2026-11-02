"""Summary figures of the SOLWEIG-GPU performance work, from the measured JSON reports.

Each figure is saved on its own, and all four together in one overview:
  summary_time.png      time per tile-day, step by step
  summary_memory.png    peak GPU memory, step by step
  summary_svf.png       the one-off sky view factor step, original against compact masks
  summary_bigtile.png   the 2000 x 2000 tile on an 8 GB GPU
  summary.png           the four together

Inputs (figures/profile/): bench_history.json, memory_report.json, verify_compact.json,
verify_compact_svf.json, bigtile_demo.json, bigtile_compare.json (optional).

Usage:
    python scripts/plot_summary.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

P = Path(__file__).resolve().parents[1] / "figures" / "profile"
INK, MUTED = "#1B232C", "#56616D"
C_ORIG, C_FAST, C_BEST = "#9AA6B2", "#4C78A8", "#0E6A78"
NOTE = "RTX 4070 Laptop GPU (8 GB), WSL2; single runs, about ±2 s run-to-run noise."


def load(name):
    f = P / name
    return json.loads(f.read_text()) if f.exists() else None


def steps(hist, mem, comp):
    t = hist["summary"]
    pac = comp["pacoima_summer"]["runs"]
    return [
        ("Original,\nWindows drive", t["A"], mem["peak"]["original"], C_ORIG),
        ("Original,\nLinux disk", t["B"], mem["peak"]["original"], C_ORIG),
        ("+ fast sky\n(chunk 64)", t["C"], mem["peak"]["+ fast sky"], C_FAST),
        ("+ fast view\nfactors", t["D"], mem["peak"]["+ fast sky + view factors"], C_FAST),
        ("+ fast\nshadows", t["E"], mem["peak"]["all three"], C_FAST),
        ("sky chunk\n64 → 8", t.get("E8", mem["chunk_sweep"]["8"]["seconds"]), mem["chunk_sweep"]["8"], C_FAST),
        ("+ bit-packed\nmasks", t.get("F", pac["fast + bit masks"]["seconds"]), pac["fast + bit masks"], C_BEST),
    ]


def draw_time(ax, st):
    x = range(len(st))
    bars = ax.bar(x, [s[1] for s in st], color=[s[3] for s in st])
    for b, s in zip(bars, st):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{s[1]:.0f} s", ha="center", fontsize=9, color=INK)
    ax.set_xticks(list(x), [s[0] for s in st], fontsize=8.5)
    ax.set_ylabel("one simulated day, 1000 x 1000 tile (s)")
    ax.set_title("Time per tile-day: 103 s → about 23 s", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)


def draw_memory(ax, st):
    x = range(len(st))
    w = 0.38
    alloc = [s[2]["peak_allocated_gib"] for s in st]
    resv = [s[2]["peak_reserved_gib"] for s in st]
    ax.bar([i - w / 2 for i in x], alloc, w, color=[s[3] for s in st], label="peak allocated")
    ax.bar([i + w / 2 for i in x], resv, w, color=[s[3] for s in st], alpha=0.45, label="peak reserved (decides out-of-memory)")
    for i, r in enumerate(resv):
        ax.text(i + w / 2, r + 0.06, f"{r:.2f}", ha="center", fontsize=8, color=INK)
    ax.axhline(st[1][2]["peak_reserved_gib"], color=MUTED, ls="--", lw=0.9)
    ax.set_xticks(list(x), [s[0] for s in st], fontsize=8.5)
    ax.set_ylabel("GPU memory over the day (GiB)")
    ax.set_title("Peak memory: chunks traded memory for speed until profiled; compact masks cut it 3x", loc="left",
                 fontweight="bold", fontsize=10.5)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)


def draw_svf(ax, svf):
    names = ["original", "bits"]
    vals = [svf[n]["peak_reserved_gib"] for n in names]
    secs = [svf[n]["seconds"] for n in names]
    b = ax.bar([0, 1], vals, color=[C_ORIG, C_BEST], width=0.55)
    for bb, v, s in zip(b, vals, secs):
        ax.text(bb.get_x() + bb.get_width() / 2, v + 0.04, f"{v:.2f} GiB\n{s:.0f} s", ha="center", fontsize=9)
    ax.set_xticks([0, 1], ["original", "compact masks"])
    ax.set_ylabel("peak reserved GPU memory (GiB)")
    ax.set_ylim(0, max(vals) * 1.35)
    ident = "identical cache files" if svf.get("all_identical") else "CACHE FILES DIFFER"
    ax.set_title(f"One-off sky view step, 1000 x 1000: {ident}", loc="left", fontweight="bold", fontsize=10.5)
    ax.grid(axis="y", alpha=0.3)


def draw_bigtile(ax, big, cmp):
    total = big.get("gpu_total_gib", 8.0)
    rows = [("sky view, original", big["svf_original"], C_ORIG), ("day, original", big["day_original"], C_ORIG),
            ("sky view, compact", big["svf_compact"], C_BEST), ("day, fast + compact", big["day_fast_bits"], C_BEST)]
    vmax = max(total, max(r[1]["peak_reserved_gib"] for r in rows))
    for i, (lab, r, c) in enumerate(rows):
        ok = r["status"] == "ok"
        v = r["peak_reserved_gib"] if ok else total
        over = ok and v > total
        ax.barh(i, v, color=c if ok else "white", edgecolor=c, hatch="//" if (over or not ok) else None)
        if not ok:
            txt = "out of memory"
        else:
            txt = f"{v:.1f} GiB, {r['seconds'] / 60:.0f} min"
            if over:
                txt += "\nmore than the GPU has: the Windows driver\nspilled to system RAM (would fail on Linux)"
        ax.text(v + 0.15, i, txt, va="center", fontsize=8.5)
    ax.axvline(total, color=MUTED, ls="--", lw=0.9)
    ax.text(total, len(rows) - 0.45, f" {total:.0f} GiB GPU", fontsize=8, color=MUTED)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=9)
    ax.set_xlim(0, vmax * 1.9)
    ax.set_xlabel("peak reserved GPU memory (GiB)")
    title = f"A {big['tile']} tile (4 x 4 km) on the 8 GB laptop GPU"
    if cmp:
        title += f"\noutputs agree to {max(cmp['UTCI']['max_abs'], cmp['TMRT']['max_abs']):.0e} °C"
    ax.set_title(title, loc="left", fontweight="bold", fontsize=10.5)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)


def main() -> None:
    hist, mem, comp, svf, big, cmp = (load(n) for n in (
        "bench_history.json", "memory_report.json", "verify_compact.json", "verify_compact_svf.json",
        "bigtile_demo.json", "bigtile_compare.json"))
    st = steps(hist, mem, comp)
    singles = [("summary_time.png", (9, 4.6), lambda ax: draw_time(ax, st)),
               ("summary_memory.png", (9, 4.6), lambda ax: draw_memory(ax, st)),
               ("summary_svf.png", (5.5, 4.4), lambda ax: draw_svf(ax, svf))]
    if big:
        singles.append(("summary_bigtile.png", (7, 4.4), lambda ax: draw_bigtile(ax, big, cmp)))
    for name, size, draw in singles:
        fig, ax = plt.subplots(figsize=size)
        draw(ax)
        fig.text(0.01, 0.005, NOTE, fontsize=7.5, color=MUTED)
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        fig.savefig(P / name, dpi=150)
        plt.close(fig)
        print(P / name)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.6), gridspec_kw={"width_ratios": [1.6, 1]})
    draw_time(axes[0, 0], st)
    draw_memory(axes[1, 0], st)
    draw_svf(axes[0, 1], svf)
    if big:
        draw_bigtile(axes[1, 1], big, cmp)
    else:
        axes[1, 1].set_axis_off()
    fig.suptitle("SOLWEIG-GPU v2.0.0: speed and memory, same results (Pacoima tile)", x=0.01, ha="left",
                 fontweight="bold", fontsize=13)
    fig.text(0.01, 0.005, NOTE + " Outputs bit-identical to the original except the fast sky (≤0.0001 °C). "
             "Sources: figures/profile/*.json", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(P / "summary.png", dpi=150)
    plt.close(fig)
    print(P / "summary.png")


if __name__ == "__main__":
    main()
