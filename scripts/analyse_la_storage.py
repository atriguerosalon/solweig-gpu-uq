"""Does keeping ground heat after sunset close the evening gap in Pacoima?

Compares the fitted runs without and with the heat-storage patch (several time constants)
against Middel et al. (2020) at 21:00 PDT, on open ground as in analyse_la_pavement.py.

Usage:
    python scripts/analyse_la_storage.py --runs runs_la --rasters data/la/pacoima --out figures
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyse_la_pavement import tmrt_error  # noqa: E402
from run_la_storage import TAUS  # noqa: E402
from solweig_sens import plotstyle  # noqa: E402
from solweig_sens.analysis import read_stack  # noqa: E402

MEAS = {"tmrt_asphalt_21": 24.9, "dtmrt_21": -1.0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path("runs_la"))
    ap.add_argument("--rasters", type=Path, default=Path("data/la/pacoima"))
    ap.add_argument("--out", type=Path, default=Path("figures"))
    args = ap.parse_args()

    shadow = read_stack(args.runs / "regular_default", "Shadow")
    shape = shadow.shape[1:]
    lc = gdal.Open(str(args.rasters / "Landcover.tif")).ReadAsArray()[: shape[0], : shape[1]]
    trees = gdal.Open(str(args.rasters / "Trees.tif")).ReadAsArray()[: shape[0], : shape[1]]
    svf = gdal.Open(str(args.runs / "shared/processed_inputs/SVF/SkyViewFactor_0_0.tif")).ReadAsArray()[: shape[0], : shape[1]]
    open_ground = (lc != 2) & (trees <= 0) & (svf > 0.75)
    outdoor = lc != 2

    variants = {"no storage": ""} | {f"τ = {t:g} h": f"_storage_tau{t:g}h" for t in TAUS}
    tm, ut = {}, {}
    for label, suffix in variants.items():
        for base in ("regular_fit", "coated_fit"):
            d = args.runs / f"{base}{suffix}"
            tm[label, base] = read_stack(d, "TMRT")
            ut[label, base] = read_stack(d, "UTCI")

    hours = np.arange(tm["no storage", "regular_fit"].shape[0])
    print("variant      Tmrt asphalt 21h  dTmrt 21h  |  mean outdoor UTCI change vs no storage, 20-23 h  |  max |change| 14h")
    curves = {}
    for label in variants:
        reg, coat = tm[label, "regular_fit"], tm[label, "coated_fit"]
        curves[label] = np.array([np.median(reg[h][open_ground]) for h in hours])
        t21 = np.interp(21.0, hours, curves[label])
        d21 = np.interp(21.0, hours, [np.median((coat[h] - reg[h])[open_ground]) for h in hours])
        du = np.mean([np.mean((ut[label, "regular_fit"][h] - ut["no storage", "regular_fit"][h])[outdoor]) for h in range(20, 24)])
        day = np.nanmax(np.abs(reg[14] - tm["no storage", "regular_fit"][14]))
        print(f"{label:12s} {t21:16.1f} {d21:10.2f}  |  {du:+.2f} C  |  {day:.3f}")
    print(f"measured     {MEAS['tmrt_asphalt_21']:16.1f} {MEAS['dtmrt_21']:10.2f}")

    # The patch only acts after sunset, so show the evening; the measurement with its uncertainty.
    plotstyle.use()
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    for (label, c), color in zip(curves.items(), ["0.45", "#9ecae1", "#4292c6", "#08519c"]):
        reg = tm[label, "regular_fit"]
        q = np.array([np.percentile(reg[h][open_ground], [25, 75]) for h in hours])
        if label == "no storage":
            ax.fill_between(hours, q[:, 0], q[:, 1], color=color, alpha=0.2, lw=0)
        ax.plot(hours, c, color=color, lw=2.4 if label == "no storage" else 1.6,
                label="SOLWEIG as is (ground = air temperature after sunset)" if label == "no storage"
                else f"heat kept, decay time {label.split('= ')[1]}")
    t21 = MEAS["tmrt_asphalt_21"]
    ax.errorbar([21.0], [t21], yerr=[tmrt_error(t21)], fmt="ko", capsize=3,
                label="measured, Pacoima asphalt (radiometers ±10%)")
    ax.set_xlim(18, 23)
    ax.set_xticks(range(18, 24))
    ax.set_xlabel("Hour, 30 July 2019 (PDT)")
    ax.set_ylabel("Tmrt over regular asphalt (°C)")
    ax.set_title("Evening Tmrt with and without the ground's stored heat")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.grid(alpha=0.3)
    fig.text(0.5, -0.1, "Pacoima, open sunlit ground, ground heating fitted to the measured surface temperatures. "
             "Lines: median over pixels; grey band: 25th to 75th percentile.\nExperimental patch (off by default): "
             "the daytime excess of ground over air temperature decays after sunset instead of vanishing.",
             ha="center", fontsize=8, color="0.35")
    fig.savefig(args.out / "la_storage_evening.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
