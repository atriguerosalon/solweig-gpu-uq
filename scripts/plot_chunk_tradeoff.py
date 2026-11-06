"""Plot the sky chunk-size trade-off from figures/profile/bench_chunks.json.

The chunk sweep of scripts/bench_steps.py (steps K8 ... K153): the code of C1-C9 with 8, 16, 32,
64 or 153 sky patches per chunk, 5 runs each on a quiet machine on AC. The original's memory comes
from step B of figures/profile/bench_steps.json.

Saves each figure on its own and both together:
  chunk_memory.png     peak memory against chunk size
  chunk_time.png       time against chunk size, every run
  chunk_tradeoff.png   the two side by side

Usage:
    python scripts/plot_chunk_tradeoff.py
"""

from __future__ import annotations

import sys as _sys

if __name__ == "__main__" and any(a in ("-h", "--help") for a in _sys.argv[1:]):
    print(__doc__)
    raise SystemExit(0)

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "figures" / "profile"
INK, MUTED = "#1B232C", "#56616D"
DEFAULT = 8


def sweep():
    """{chunk: [(seconds, peak_allocated, peak_reserved), ...]} and the original's (allocated, reserved)."""
    rep = json.loads((P / "bench_chunks.json").read_text())
    out = {}
    for r in rep["runs"]:
        if r["step"].startswith("K"):
            out.setdefault(int(r["step"][1:]), []).append((r["seconds"], r["peak_allocated_gib"], r["peak_reserved_gib"]))
    base = [r for r in json.loads((P / "bench_steps.json").read_text())["runs"] if r["step"] == "B"]
    orig = (statistics.median(r["peak_allocated_gib"] for r in base), statistics.median(r["peak_reserved_gib"] for r in base))
    return dict(sorted(out.items())), orig


def draw_memory(ax, sw, orig):
    chunks = list(sw)
    for k, (color, label) in enumerate([("#2E6FD8", "peak allocated"),
                                        ("#E8663D", "peak reserved (what triggers out-of-memory)")]):
        ax.plot(chunks, [statistics.median(r[k + 1] for r in sw[c]) for c in chunks], "o-", color=color, label=label)
        ax.axhline(orig[k], color=color, ls="--", lw=1)
    ax.text(chunks[0], orig[1] + 0.1, "original, before any patch (dashed)", ha="left", va="bottom", fontsize=8,
            color=MUTED)
    ax.set_xscale("log", base=2)
    ax.set_xticks(chunks, [str(c) for c in chunks])
    ax.set_xlabel("sky patches processed per chunk (153 = all at once)")
    ax.set_ylabel("GPU memory over the day (GiB)")
    ax.set_ylim(0, max(r[2] for v in sw.values() for r in v) * 1.15)
    ax.set_title("Memory grows with chunk size", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, 0.9))
    ax.grid(alpha=0.3)


def draw_time(ax, sw, orig):
    chunks = list(sw)
    meds = [statistics.median(r[0] for r in sw[c]) for c in chunks]
    for c in chunks:
        pts = [r[0] for r in sw[c]]
        offs = np.linspace(0.94, 1.06, len(pts))
        ax.scatter([c * o for o in offs], pts, s=16, color=INK, zorder=3, edgecolor="white", linewidth=0.4)
    ax.plot(chunks, meds, "-", color="#1FAD7E", lw=2, label="median of 5 runs")
    for c, m in zip(chunks, meds):
        ax.annotate(f"{m:.1f} s", (c, m), xytext=(0, 16), textcoords="offset points", ha="center", fontsize=8)
    ax.axvline(DEFAULT, color=MUTED, ls=":", lw=1)
    ax.text(DEFAULT, 0.6, " default", fontsize=8, color=MUTED, transform=ax.get_xaxis_transform())
    ax.set_xscale("log", base=2)
    ax.set_xticks(chunks, [str(c) for c in chunks])
    ax.set_xlabel("sky patches processed per chunk (153 = all at once)")
    ax.set_ylabel("one tile-day, C1-C9 (s)")
    ax.set_ylim(0, max(r[0] for v in sw.values() for r in v) * 1.25)
    ax.set_title(f"...while time barely moves: {meds[0]:.1f} s → {meds[-1]:.1f} s", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)


def main() -> None:
    sw, orig = sweep()
    note = ("Fast sky, 1000 x 1000 Pacoima tile, code of C1-C9, RTX 4070 Laptop (8 GB), quiet machine on AC, 4 Nov 2026. "
            "Dots: each run.")
    for name, draw in (("chunk_memory.png", draw_memory), ("chunk_time.png", draw_time)):
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        draw(ax, sw, orig)
        fig.text(0.01, 0.005, note, fontsize=6.5, color=MUTED)
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        fig.savefig(P / name, dpi=150)
        plt.close(fig)
        print(P / name)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    draw_memory(axes[0], sw, orig)
    draw_time(axes[1], sw, orig)
    fig.suptitle("Sky patches per chunk: memory against time", x=0.01, ha="left", fontweight="bold")
    fig.text(0.01, 0.005, note, fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(P / "chunk_tradeoff.png", dpi=150)
    plt.close(fig)
    print(P / "chunk_tradeoff.png")


if __name__ == "__main__":
    main()
