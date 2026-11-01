"""Re-plot the saved one-hour memory traces (figures/profile/memory_trace_*.npz) without the GPU.

Writes memory_trace_original.png, memory_trace_final.png (each on its own) and memory_trace_compare.png.

Usage:
    python scripts/replot_traces.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from profile_memory import OUT, plot_traces  # noqa: E402

LABELS = {"original": "original", "final": "all fixes + bit-packed masks"}


def load(slug: str, label: str) -> dict:
    z = np.load(OUT / f"memory_trace_{slug}.npz")
    series = z["series"]
    # The saved series is downsampled for plotting; the exact peak is in memory_trace_final.json.
    exact = json.loads((OUT / "memory_trace_final.json").read_text())
    return {"components": [str(c) for c in z["components"]], "series": series.tolist(), "step": int(z["step"]),
            "n_events": int(z["n_events"]), "peak_gib": float(exact[label]["peak_gib"])}


def main() -> None:
    traces = {LABELS[s]: load(s, LABELS[s]) for s in LABELS}
    for slug, label in LABELS.items():
        plot_traces({label: traces[label]}, OUT / f"memory_trace_{slug}.png", budget_gib=8.0)
    plot_traces(traces, OUT / "memory_trace_compare.png", budget_gib=8.0)
    print("written:", [f"memory_trace_{s}.png" for s in LABELS] + ["memory_trace_compare.png"])


if __name__ == "__main__":
    main()
