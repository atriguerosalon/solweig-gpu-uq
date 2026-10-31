"""What if air temperature varied in space? An illustrative test, not a validated field.

SOLWEIG-GPU uses one air temperature per hour for the whole domain. Inside the
model, Ta enters Tmrt only through domain-wide longwave terms, and the UTCI step
(Ta_mat in utci_process.py). So a spatial Ta field can be tested by recomputing
UTCI from the saved baseline Tmrt, with no GPU rerun; Tmrt itself is left as it
is (a stated limitation).

Field shape: the term the SOLWEIG-GPU authors drafted and commented out,
    dT = uhii * (2 - svf - fveg) * wind**-0.25
with the domain mean removed, so only the spatial pattern changes, never the
mean. svf is the building sky view factor; fveg is 1 under trees. uhii is the
amplitude, tried at 1, 2 and 3 K.

Usage:
    python scripts/spatial_ta.py --runs runs --out figures
"""

from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solweig_sens.analysis import outdoor_mask, read_stack  # noqa: E402
from solweig_sens.perturb import MET_COLS  # noqa: E402


def utci(ta, rh, tmrt, wind) -> np.ndarray:
    from solweig_gpu.calculate_utci import utci_calculator

    t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32))  # noqa: E731
    va = torch.clamp(t(wind), min=0.15)  # as in utci_process.py
    return utci_calculator(t(ta), t(rh), t(tmrt), va).numpy()


def read_svf(svf_dir: Path) -> np.ndarray:
    from osgeo import gdal

    with zipfile.ZipFile(next(svf_dir.glob("svfs_*.zip"))) as z:
        data = z.read("svf.tif")
    gdal.FileFromMemBuffer("/vsimem/svf.tif", data)
    arr = gdal.Open("/vsimem/svf.tif").ReadAsArray().astype(np.float32)
    gdal.Unlink("/vsimem/svf.tif")
    return arr


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    from osgeo import gdal

    gdal.UseExceptions()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--met", type=Path, default=Path("data/ownmet_Forcing_data.txt"))
    ap.add_argument("--rasters", type=Path, default=Path("data/austin_crop"))
    ap.add_argument("--out", type=Path, default=Path("figures"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    pre = args.runs / "shared" / "processed_inputs"
    met = np.loadtxt(args.met, skiprows=1)
    ta, rh, ws = met[:, MET_COLS["ta"]], met[:, MET_COLS["rh"]], met[:, MET_COLS["wind"]]
    tmrt = read_stack(args.runs / "baseline", "TMRT")
    u_base = read_stack(args.runs / "baseline", "UTCI")
    shape = u_base.shape[1:]
    outdoor = outdoor_mask(args.rasters, shape) & np.isfinite(u_base[14])

    # 1. Check: recomputing UTCI with uniform Ta must reproduce the model's output.
    check = max(float(np.nanmax(np.abs(utci(np.full(shape, ta[h]), np.full(shape, rh[h]), tmrt[h],
                                             np.full(shape, ws[h])) - u_base[h])[outdoor]))
                for h in (6, 14, 22))
    print(f"reproduction check, max |UTCI recomputed - model| = {check:.4f} C")

    # 2. Field shape from the commented-out SOLWEIG-GPU term.
    svf = read_svf(pre / "SVF")[: shape[0], : shape[1]]
    trees = gdal.Open(str(next((pre / "Trees").glob("*.tif")))).ReadAsArray()[: shape[0], : shape[1]]
    fveg = (trees > 0).astype(np.float32)
    s = 2.0 - svf - fveg
    s = s - s[outdoor].mean()

    rows, maps = [], {}
    for hour, thr in [(14, 38.0), (22, 32.0)]:
        wind_factor = max(ws[hour], 0.15) ** -0.25
        for uhii in (1.0, 2.0, 3.0):
            dT = uhii * s * wind_factor
            u_new = utci(ta[hour] + dT, np.full(shape, rh[hour]), tmrt[hour], np.full(shape, ws[hour]))
            du = (u_new - u_base[hour])[outdoor]
            b, n = u_base[hour][outdoor], u_new[outdoor]
            rows.append({
                "hour": hour, "uhii_K": uhii,
                "dT_std": float(dT[outdoor].std()),
                "dT_p5": float(np.percentile(dT[outdoor], 5)), "dT_p95": float(np.percentile(dT[outdoor], 95)),
                "dUTCI_mean": float(du.mean()), "dUTCI_absmean": float(np.abs(du).mean()),
                "dUTCI_p95_abs": float(np.percentile(np.abs(du), 95)),
                "slope_dUTCI_per_dT": float(np.polyfit(dT[outdoor], du, 1)[0]),
                "threshold_C": thr,
                "area_above_base_pct": float(100 * (b >= thr).mean()),
                "area_above_new_pct": float(100 * (n >= thr).mean()),
                "pixels_crossing_pct": float(100 * ((b >= thr) != (n >= thr)).mean()),
                "spearman_rank": spearman(b, n),
            })
            if uhii == 2.0:
                maps[hour] = (np.where(outdoor, dT, np.nan), np.where(outdoor, u_new - u_base[hour], np.nan))
    df = pd.DataFrame(rows)
    df.to_csv(args.out / "spatial_ta.csv", index=False, float_format="%.3f")
    print(df.round(3).to_string(index=False))

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.4))
    for r, hour in enumerate((14, 22)):
        dT, du = maps[hour]
        lim = np.nanpercentile(np.abs(dT), 99)
        im = axes[r, 0].imshow(dT, cmap="RdBu_r", vmin=-lim, vmax=lim)
        axes[r, 0].set_title(f"Air temperature anomaly, {hour:02d}:00 (uhii = 2 K)")
        fig.colorbar(im, ax=axes[r, 0], shrink=0.8, label="K")
        im = axes[r, 1].imshow(du, cmap="RdBu_r", vmin=-lim, vmax=lim)
        axes[r, 1].set_title(f"Resulting ΔUTCI, {hour:02d}:00")
        fig.colorbar(im, ax=axes[r, 1], shrink=0.8, label="°C")
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(args.out / "spatial_ta_maps.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
