"""Overview of a domain built by build_la_domain.py: ground, buildings, trees and land cover.

Usage:
    python scripts/plot_la_domain.py --rasters data/la/pacoima --out figures/la_pacoima_inputs.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solweig_sens import plotstyle  # noqa: E402

# SOLWEIG land-cover codes used by build_la_domain.py, with map colours.
CLASSES = {1: ("Paved", "#8C8C8C"), 2: ("Building", "#C0504D"), 5: ("Grass", "#6BA543"),
           6: ("Bare soil", "#D9B26F"), 7: ("Water", "#3C78B4")}


def main() -> None:
    from osgeo import gdal

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rasters", type=Path, default=Path("data/la/pacoima"))
    ap.add_argument("--out", type=Path, default=Path("figures/la_pacoima_inputs.png"))
    args = ap.parse_args()
    gdal.UseExceptions()
    read = lambda n: gdal.Open(str(args.rasters / f"{n}.tif")).ReadAsArray().astype(np.float32)  # noqa: E731
    dsm, dem, trees, lc = read("Building_DSM"), read("DEM"), read("Trees"), read("Landcover")
    px = abs(gdal.Open(str(args.rasters / "DEM.tif")).GetGeoTransform()[1])
    size_km = dem.shape[1] * px / 1000

    plotstyle.use()
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6))
    land = matplotlib.colors.LinearSegmentedColormap.from_list(  # terrain without its sea-blue end
        "land", plt.get_cmap("terrain")(np.linspace(0.25, 1.0, 256)))
    im = axes[0].imshow(dem, cmap=land)
    axes[0].set_title("Ground elevation")
    fig.colorbar(im, ax=axes[0], shrink=0.8, label="m above sea level")
    bld = np.ma.masked_less(dsm - dem, 0.5)
    cmap = plt.get_cmap("magma_r").copy()
    cmap.set_bad("0.93")
    im = axes[1].imshow(bld, cmap=cmap, vmin=0, vmax=np.percentile(bld.compressed(), 99.5))
    axes[1].set_title("Building height")
    fig.colorbar(im, ax=axes[1], shrink=0.8, label="m above ground", extend="max")
    cmap = plt.get_cmap("Greens").copy()
    cmap.set_bad("0.93")
    im = axes[2].imshow(np.ma.masked_less_equal(trees, 0), cmap=cmap, vmin=0, vmax=np.percentile(trees[trees > 0], 99.5))
    axes[2].set_title(f"Tree canopy ({100 * (trees > 0).mean():.0f}% of the area)")
    fig.colorbar(im, ax=axes[2], shrink=0.8, label="m above ground", extend="max")
    present = [c for c in CLASSES if (lc == c).any()]
    rgb = np.ones(lc.shape + (3,))
    for c in present:
        rgb[lc == c] = matplotlib.colors.to_rgb(CLASSES[c][1])
    axes[3].imshow(rgb)
    axes[3].set_title("Land cover")
    axes[3].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=CLASSES[c][1]) for c in present],
                   labels=[f"{CLASSES[c][0]} ({100 * (lc == c).mean():.0f}%)" for c in present],
                   loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    for ax in axes:
        plotstyle.map_axes(ax)
    plotstyle.scalebar(axes[0], px)
    fig.text(0.5, -0.03, f"{args.rasters.name.capitalize()}, Los Angeles: {size_km:g} x {size_km:g} km at {px:g} m, "
             "built from public data only: USGS 3DEP elevation, LA County building outlines, Meta / WRI 1 m canopy "
             "height (Tolan et al. 2024), ESA WorldCover 2021.", ha="center", fontsize=8, color="0.35")
    fig.savefig(args.out)
    plt.close(fig)
    print(f"[domain] {args.out}")


if __name__ == "__main__":
    main()
