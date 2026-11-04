"""Effect of building heights, tree heights and tree transmissivity in Pacoima (run_la_geometry.py).

Each case minus the baseline, for Tmrt and UTCI, by place (all outdoor, under trees, open ground
within 10 m of a building, other open ground) and time (10-16 h, 14:00 sunlit and shaded by the
baseline shadow, 21-05 h).

Usage:
    python scripts/analyse_la_geometry.py --runs runs_la_geom --out figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solweig_sens.analysis import read_stack  # noqa: E402

NIGHT = [21, 22, 23, 0, 1, 2, 3, 4, 5]
LABELS = {"bldg_lo": "buildings x0.8", "bldg_hi": "buildings x1.2", "trees_lo": "trees -2.8 m",
          "trees_hi": "trees +2.8 m", "trans_hi": "tree transmissivity 10%"}


def places(raster_dir: Path, shape: tuple[int, int]) -> dict:
    from osgeo import gdal
    from scipy.ndimage import distance_transform_edt

    rd = lambda n: gdal.Open(str(raster_dir / f"{n}.tif")).ReadAsArray()[: shape[0], : shape[1]]  # noqa: E731
    lc, trees = rd("Landcover"), rd("Trees")
    building = lc == 2
    near = distance_transform_edt(~building) * 2.0 <= 10.0  # 2 m pixels
    outdoor = ~building
    return {"all outdoor": outdoor, "under trees": outdoor & (trees > 0),
            "open, near buildings": outdoor & (trees <= 0) & near,
            "open, away from buildings": outdoor & (trees <= 0) & ~near}


def stats(d: np.ndarray) -> dict:
    d = d[np.isfinite(d)]
    return {"mean": float(d.mean()), "mean_abs": float(np.abs(d).mean()), "p1": float(np.percentile(d, 1)),
            "p99": float(np.percentile(d, 99)), "n": int(d.size)}


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path("runs_la_geom"))
    ap.add_argument("--out", type=Path, default=Path("figures"))
    args = ap.parse_args()
    cases = [c for c in json.loads((args.runs / "cases.json").read_text()) if c != "baseline"]
    base = args.runs / "baseline" / "out"
    shadow = read_stack(base, "Shadow")
    cls = places(args.runs / "baseline" / "rasters", shadow.shape[1:])
    out = {"pixels": {k: int(v.sum()) for k, v in cls.items()}}
    for var in ("TMRT", "UTCI"):
        b = read_stack(base, var)
        for case in cases:
            d = read_stack(args.runs / case / "out", var) - b
            res = out.setdefault(case, {}).setdefault(var, {})
            for cname, m in cls.items():
                res[cname] = {
                    "day (10-16 h)": stats(d[10:17][:, m]),
                    "14:00 sunlit": stats(d[14][m & (shadow[14] > 0.5)]),
                    "14:00 shaded": stats(d[14][m & (shadow[14] <= 0.5)]),
                    "night (21-05 h)": stats(np.concatenate([d[h][m] for h in NIGHT])),
                }
            res["hourly mean, all outdoor"] = [float(np.nanmean(d[h][cls["all outdoor"]])) for h in range(24)]
    path = args.out / "la_geometry.json"
    path.write_text(json.dumps(out, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    places_ = [c for c in cls if c != "all outdoor"]
    for ax, w in zip(axes, ["day (10-16 h)", "night (21-05 h)"]):
        x = np.arange(len(cases))
        for i, cname in enumerate(places_):
            ax.bar(x + (i - 1) * 0.27, [out[c]["TMRT"][cname][w]["mean"] for c in cases], 0.27, label=cname)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(x, [LABELS.get(c, c) for c in cases], rotation=25, ha="right")
        ax.set_title(f"Tmrt change, {w}")
    axes[0].set_ylabel("mean change from baseline (K)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out / "la_geometry.png", dpi=150)

    print(f"pixels: {out['pixels']}")
    for case in cases:
        for var in ("TMRT", "UTCI"):
            r = out[case][var]
            parts = [f"{c}: day {r[c]['day (10-16 h)']['mean']:+.2f} (1-99% {r[c]['day (10-16 h)']['p1']:+.1f}..{r[c]['day (10-16 h)']['p99']:+.1f}), "
                     f"14h shade {r[c]['14:00 shaded']['mean']:+.2f}, night {r[c]['night (21-05 h)']['mean']:+.2f}" for c in cls]
            print(f"== {LABELS.get(case, case)} {var}\n   " + "\n   ".join(parts))
    print(f"[geom] report: {path}")


if __name__ == "__main__":
    main()
