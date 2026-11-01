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

from run_la_storage import TAUS  # noqa: E402
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

    fig, ax = plt.subplots(figsize=(7, 4))
    for (label, c), color in zip(curves.items(), ["0.4", "#9ecae1", "#4292c6", "#08519c"]):
        ax.plot(hours, c, color=color, label=label, lw=2 if label == "no storage" else 1.5)
    ax.plot([21.0], [MEAS["tmrt_asphalt_21"]], "ko", label="measured, Pacoima asphalt")
    ax.set_xlim(16, 23)
    ax.set_xlabel("Hour, 30 July 2019 (PDT)")
    ax.set_ylabel("Tmrt over regular asphalt (°C)")
    ax.set_title("Evening Tmrt with ground heat kept after sunset")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out / "la_storage_evening.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
