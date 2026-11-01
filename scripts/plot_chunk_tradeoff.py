"""Plot the sky chunk-size trade-off from figures/profile/memory_report.json.

Saves each figure on its own and both together:
  chunk_memory.png     peak memory against chunk size
  chunk_time.png       time against peak reserved memory
  chunk_tradeoff.png   the two side by side

Usage:
    python scripts/plot_chunk_tradeoff.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "figures" / "profile"


def draw_memory(ax, sweep, orig):
    chunks = sorted(sweep)
    for key, color, label in [("peak_allocated_gib", "#2E6FD8", "peak allocated"),
                              ("peak_reserved_gib", "#E8663D", "peak reserved (what triggers out-of-memory)")]:
        ax.plot(chunks, [sweep[c][key] for c in chunks], "o-", color=color, label=label)
        ax.axhline(orig[key], color=color, ls="--", lw=1)
    ax.text(chunks[-1], orig["peak_reserved_gib"] - 0.12, "original (dashed)", ha="right", va="top", fontsize=8, color="0.3")
    ax.set_xscale("log", base=2)
    ax.set_xticks(chunks, [str(c) for c in chunks])
    ax.set_xlabel("sky patches processed per chunk (153 = all at once)")
    ax.set_ylabel("GPU memory over the day (GiB)")
    ax.set_title("Memory grows with chunk size", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)


def draw_time(ax, sweep, orig):
    chunks = sorted(sweep)
    xs = [sweep[c]["peak_reserved_gib"] for c in chunks]
    ys = [sweep[c]["seconds"] for c in chunks]
    ax.plot(xs, ys, "o-", color="#1FAD7E")
    for c, x, y in zip(chunks, xs, ys):
        ax.annotate(f"chunk {c}", (x, y), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.axvline(orig["peak_reserved_gib"], color="0.4", ls="--", lw=1)
    ax.text(orig["peak_reserved_gib"], max(ys) + 0.4, " original memory", fontsize=8, color="0.3", va="bottom")
    ax.set_xlabel("peak reserved GPU memory (GiB)")
    ax.set_ylabel("one tile-day, all three fixes (s)")
    ax.set_ylim(min(ys) - 2, max(ys) + 2)
    ax.set_title("...while time barely moves (single runs; noise about ±2 s)", loc="left")
    ax.grid(alpha=0.3)


def main() -> None:
    r = json.loads((P / "memory_report.json").read_text())
    sweep = {int(k): v for k, v in r["chunk_sweep"].items()}
    orig = r["peak"]["original"]
    note = "Fast sky, 1000 x 1000 Pacoima tile, RTX 4070 Laptop (8 GB)"
    for name, draw in (("chunk_memory.png", draw_memory), ("chunk_time.png", draw_time)):
        fig, ax = plt.subplots(figsize=(6, 4.2))
        draw(ax, sweep, orig)
        fig.text(0.01, 0.005, note, fontsize=7.5, color="0.4")
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        fig.savefig(P / name, dpi=150)
        plt.close(fig)
        print(P / name)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    draw_memory(axes[0], sweep, orig)
    draw_time(axes[1], sweep, orig)
    fig.suptitle(f"{note}: patches per chunk", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(P / "chunk_tradeoff.png", dpi=150)
    plt.close(fig)
    print(P / "chunk_tradeoff.png")


if __name__ == "__main__":
    main()
