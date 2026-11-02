"""Compare the Pacoima cool-pavement runs with the measurements of Middel et al. (2020).

Measurements were taken at fully sun-exposed street stops, so the model is sampled on open
ground: not a building, no tree, sky view factor above 0.75, and sunlit at that hour.
Measured times are local standard time; they are shifted by +1 h to daylight time, which is
what the model's hourly output uses.

Usage:
    python scripts/analyse_la_pavement.py --runs runs_la --rasters data/la/pacoima --out figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from osgeo import gdal  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solweig_sens import plotstyle  # noqa: E402
from solweig_sens.analysis import read_stack  # noqa: E402

# Middel et al. (2020), Pacoima, coated minus regular asphalt; time in PDT (LST + 1 h).
MEAS_DTMRT = {12.5: 4.2, 13.5: 4.0, 14.5: 3.7, 15.5: 2.8, 16.5: 2.6, 17.5: 1.2, 21.0: -1.0}
MEAS_DTMRT_SV = {12.5: 4.2, 13.5: 4.2, 14.5: 3.0, 15.5: 2.2, 16.5: 2.3, 17.5: 1.2, 18.5: 0.2, 21.0: -0.9}  # Sun Valley, albedo 0.18
# Absolute Tmrt over regular asphalt, Pacoima: 11:30 LST, peak at 15:30 LST, 20:00 LST.
MEAS_TMRT_ASPHALT = {12.5: 57.6, 16.5: 64.4, 21.0: 24.9}
CASES = ["regular_default", "coated_default", "regular_fit", "coated_fit"]


def at(series: np.ndarray, t: float) -> float:
    """Linear interpolation of an hourly series (index = hour) at time t."""
    return float(np.interp(t, np.arange(len(series)), series))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path("runs_la"))
    ap.add_argument("--rasters", type=Path, default=Path("data/la/pacoima"))
    ap.add_argument("--out", type=Path, default=Path("figures"))
    args = ap.parse_args()

    tmrt = {c: read_stack(args.runs / c, "TMRT") for c in CASES}
    utci = {c: read_stack(args.runs / c, "UTCI") for c in CASES}
    shadow = read_stack(args.runs / "regular_default", "Shadow")
    shape = shadow.shape[1:]
    # Sky view factor from the shared cache (not written per run when cached); single tile here.
    svf = gdal.Open(str(args.runs / "shared/processed_inputs/SVF/SkyViewFactor_0_0.tif")).ReadAsArray()[: shape[0], : shape[1]]
    lc = gdal.Open(str(args.rasters / "Landcover.tif")).ReadAsArray()[: shape[0], : shape[1]]
    trees = gdal.Open(str(args.rasters / "Trees.tif")).ReadAsArray()[: shape[0], : shape[1]]
    open_ground = (lc != 2) & (trees <= 0) & (svf > 0.75)
    outdoor = lc != 2
    print(f"open ground: {100 * open_ground.mean():.1f}% of pixels")

    hours = np.arange(shadow.shape[0])

    def stat(field: np.ndarray, h: int, mask: np.ndarray) -> tuple[float, float, float]:
        m = mask & ((shadow[h] > 0.5) if 7 <= h <= 19 else True)
        v = field[h][m]
        return tuple(np.percentile(v, [25, 50, 75]))

    rows = {}
    for label, a, b in [("default", "regular_default", "coated_default"), ("fit", "regular_fit", "coated_fit")]:
        d = tmrt[b] - tmrt[a]
        rows[label] = np.array([stat(d, h, open_ground) for h in hours])
    absolute = {c: np.array([stat(tmrt[c], h, open_ground) for h in hours]) for c in ["regular_default", "regular_fit"]}

    print("\nCoated minus regular Tmrt on open sunlit ground (median), versus Middel et al. 2020, Pacoima")
    print("time PDT  measured  SOLWEIG default  SOLWEIG fitted ground temperature")
    for t, m in MEAS_DTMRT.items():
        print(f"{t:8.1f} {m:9.1f} {at(rows['default'][:, 1], t):16.2f} {at(rows['fit'][:, 1], t):17.2f}")
    print("\nAbsolute Tmrt over regular asphalt (median)")
    for t, m in MEAS_TMRT_ASPHALT.items():
        print(f"{t:8.1f} measured {m:5.1f}  default {at(absolute['regular_default'][:, 1], t):5.1f}  "
              f"fitted {at(absolute['regular_fit'][:, 1], t):5.1f}")

    # Neighbourhood averages, all outdoor ground, sun and shade, 12:00-18:00 (for Taha 2024's framing).
    for label, a, b in [("default", "regular_default", "coated_default"), ("fit", "regular_fit", "coated_fit")]:
        dt = np.mean([np.mean((tmrt[b][h] - tmrt[a][h])[outdoor]) for h in range(12, 19)])
        du = np.mean([np.mean((utci[b][h] - utci[a][h])[outdoor]) for h in range(12, 19)])
        dn = np.mean([np.mean((tmrt[b][h] - tmrt[a][h])[outdoor]) for h in list(range(21, 24)) + list(range(0, 6))])
        print(f"{label}: all outdoor ground, 12-18 h mean dTmrt {dt:+.2f} C, dUTCI {du:+.2f} C; night (21-05 h) dTmrt {dn:+.2f} C")

    plotstyle.use()
    band = plt.Rectangle((0, 0), 1, 1, color="0.6", alpha=0.35, lw=0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    for label, color, name in [("default", plotstyle.HIGH, "SOLWEIG as is (albedo only)"),
                               ("fit", plotstyle.LOW, "SOLWEIG + measured surface cooling")]:
        r = rows[label]
        ax.fill_between(hours, r[:, 0], r[:, 2], color=color, alpha=0.2, lw=0)
        ax.plot(hours, r[:, 1], color=color, label=name)
    ax.plot(list(MEAS_DTMRT), list(MEAS_DTMRT.values()), "ko", label="measured, Pacoima (albedo 0.25)")
    ax.plot(list(MEAS_DTMRT_SV), list(MEAS_DTMRT_SV.values()), "o", mfc="none", mec="0.4",
            label="measured, Sun Valley (albedo 0.18)")
    handles, names = ax.get_legend_handles_labels()
    ax.legend(handles + [band], names + ["model: middle half of open sunlit pixels"], frameon=False,
              loc="upper right")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlim(6, 23)
    ax.set_xlabel("Hour, 30 July 2019 (PDT)")
    ax.set_ylabel("Tmrt, coated minus regular asphalt (°C)")
    ax.set_title("Coating effect on pedestrians")
    ax.grid(alpha=0.3)

    ax = axes[1]
    for c, color, name in [("regular_default", plotstyle.HIGH, "SOLWEIG, default ground heating"),
                           ("regular_fit", plotstyle.LOW, "SOLWEIG, ground heating fitted")]:
        r = absolute[c]
        ax.fill_between(hours, r[:, 0], r[:, 2], color=color, alpha=0.2, lw=0)
        ax.plot(hours, r[:, 1], color=color, label=name)
    ax.plot(list(MEAS_TMRT_ASPHALT), list(MEAS_TMRT_ASPHALT.values()), "ko", label="measured, Pacoima asphalt")
    ax.set_xlim(0, 23)
    ax.set_xlabel("Hour, 30 July 2019 (PDT)")
    ax.set_ylabel("Tmrt over regular asphalt (°C)")
    ax.set_title("Tmrt itself")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.text(0.5, -0.03, "Pacoima, Los Angeles, on the day of the MaRTy measurements (Middel et al. 2020, "
             "Environ. Res. Lett. 15, 064016). Model: median over open sunlit ground (no building, no tree, sky view "
             "above 0.75); shading: 25th to 75th percentile.", ha="center", fontsize=8, color="0.35")
    fig.savefig(args.out / "la_pavement_vs_middel.png")
    plt.close(fig)

    # Map of the effect at 13:00 for the fitted case, all outdoor ground.
    h = 13
    px = abs(gdal.Open(str(args.rasters / "Building_DSM.tif")).GetGeoTransform()[1])
    d = np.where(outdoor, tmrt["coated_fit"][h] - tmrt["regular_fit"][h], np.nan)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    im = axes[0].imshow(np.where(outdoor, tmrt["regular_fit"][h], np.nan), cmap="inferno")
    axes[0].set_title("Tmrt at 13:00, regular asphalt everywhere")
    fig.colorbar(im, ax=axes[0], shrink=0.8, label="Tmrt (°C)")
    cmap = plt.get_cmap("OrRd").copy()
    cmap.set_bad("0.75")  # roofs grey here: white would read as "no change"
    im = axes[1].imshow(d, cmap=cmap, vmin=0, vmax=np.nanpercentile(d, 99.5))
    axes[1].set_title("If all asphalt were coated: change in Tmrt")
    fig.colorbar(im, ax=axes[1], shrink=0.8, label="ΔTmrt (°C)", extend="both")
    for ax in axes:
        plotstyle.map_axes(ax)
    plotstyle.scalebar(axes[0], px)
    fig.text(0.5, -0.03, "Pacoima, 2 x 2 km at 2 m, 30 July 2019. Coating: albedo 0.07 → 0.25, with the surface "
             "cooling measured by Middel et al. 2020. Roofs: white (left), grey (right).", ha="center", fontsize=8,
             color="0.35")
    fig.savefig(args.out / "la_pavement_map_13h.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
