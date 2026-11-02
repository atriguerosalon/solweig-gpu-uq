"""Shared look for the figures: readable labels, one colour per input, map scale bars.

The result figures use the same names and colours for the nine inputs, so a reader can follow an
input from one figure to the next. Colours are from the Okabe-Ito palette (distinguishable with
colour-vision deficiency and in greyscale print), with its yellow replaced by brown, which reads
better as a line on white.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

INPUT_COLORS = {
    "air temperature": "#D55E00",
    "relative humidity": "#0072B2",
    "wind speed": "#000000",
    "global radiation": "#E69F00",
    "body shortwave absorption": "#CC79A7",
    "paved albedo": "#009E73",
    "paved emissivity": "#56B4E9",
    "wall albedo": "#8B5A2B",
    "wall emissivity": "#999999",
}
# SOLWEIG-GPU v2.0.0 defaults of the parameters perturbed by value (utci_process globals; dark asphalt in the
# land-cover table), so a label can say "0.18 -> 0.35" rather than only the new value.
DEFAULTS = {"wall albedo": 0.20, "wall emissivity": 0.90, "body shortwave absorption": 0.70,
            "paved albedo": 0.18, "paved emissivity": 0.95}
LOW, HIGH = "#4C78A8", "#E45756"  # low and high value of an input; also shaded and sunlit ground
SUN, SHADE = HIGH, LOW


def use() -> None:
    """Common fonts and frames."""
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False,
        "savefig.dpi": 160, "savefig.bbox": "tight", "figure.constrained_layout.use": True,
    })


def input_label(group: str) -> str:
    """'air temperature' -> 'Air temperature'."""
    return group[:1].upper() + group[1:]


def case_label(group: str, value: float, kind: str) -> str:
    """A perturbed case in words: 'Air temperature +2 K', 'Wind speed x0.5', 'Paved albedo 0.18 -> 0.35'."""
    if kind == "met" and group == "air temperature":
        text = f"{input_label(group)} {value:+g} K"
    elif kind == "met" and group == "relative humidity":
        text = f"{input_label(group)} {value:+g} points"
    elif kind == "met":
        text = f"{input_label(group)} ×{value:g}"
    elif group in DEFAULTS:
        text = f"{input_label(group)} {DEFAULTS[group]:g} → {value:g}"
    else:
        text = f"{input_label(group)} {value:g}"
    return text.replace("-", "−")  # typographic minus


def map_axes(ax) -> None:
    """A map panel: no ticks, thin frame."""
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.6)


def scalebar(ax, pixel_m: float, length_m: float | None = None) -> None:
    """A white-backed scale bar in the lower left of an imshow map."""
    nx = ax.get_xlim()[1] - ax.get_xlim()[0]
    if length_m is None:  # about a fifth of the width, rounded to 100 / 200 / 500 m ...
        target = abs(nx) * pixel_m / 5
        length_m = min((c for c in (50, 100, 200, 250, 500, 1000, 2000) if c >= target / 2), default=1000)
    px = length_m / pixel_m
    x0, y0 = ax.get_xlim()[0] + 0.05 * nx, ax.get_ylim()[0] - 0.06 * (ax.get_ylim()[0] - ax.get_ylim()[1])
    ax.plot([x0, x0 + px], [y0, y0], color="black", lw=3, solid_capstyle="butt", zorder=5)
    ax.text(x0 + px / 2, y0 - 0.025 * abs(ax.get_ylim()[0] - ax.get_ylim()[1]), f"{length_m:g} m",
            ha="center", va="bottom", fontsize=8, zorder=5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5})


def symmetric_limit(*fields: np.ndarray, q: float = 99.5) -> float:
    """Colour limit for a diverging map: the q-th percentile of |values|, so a few pixels cannot set it."""
    return float(max(np.nanpercentile(np.abs(f), q) for f in fields))
