"""Summary table and figures from finished runs.

Usage:
    python scripts/make_figures.py --rasters data/austin/Input_rasters --hour 14
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
from solweig_sens.analysis import outdoor_mask, read_stack, summarise  # noqa: E402
from solweig_sens.perturb import default_cases  # noqa: E402


def pixel_size(rasters: Path) -> float:
    from osgeo import gdal

    return abs(gdal.Open(str(rasters / "Building_DSM.tif")).GetGeoTransform()[1])


def tornado(df, path: Path, hour: int) -> None:
    """Low and high case per input, split into sunlit and shaded ground."""
    groups = (
        df.assign(span=df["dUTCI_mean"].abs())
        .groupby("input")["span"].max().sort_values().index.tolist()
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.45 * len(groups) + 1.6), sharey=True)
    for ax, col, title in zip(axes, ["dUTCI_sun", "dUTCI_shade"], ["Sunlit", "Shaded"]):
        for i, g in enumerate(groups):
            sub = df[df["input"] == g].sort_values("value")
            lo, hi = sub.iloc[0], sub.iloc[-1]
            ax.barh(i, lo[col], color=plotstyle.LOW, height=0.6)
            ax.barh(i, hi[col], color=plotstyle.HIGH, height=0.6)
        ax.axvline(0, color="0.3", lw=0.8)
        ax.set_title(f"{title} ground, {hour:02d}:00")
        ax.set_xlabel("Change in mean UTCI (°C)")
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_yticks(range(len(groups)), [plotstyle.input_label(g) for g in groups])
    axes[1].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in (plotstyle.LOW, plotstyle.HIGH)],
                   labels=["low end of range", "high end of range"], loc="lower right", frameon=False)
    fig.savefig(path)
    plt.close(fig)


def maps(runs: Path, rasters: Path, cases: list, hour: int, path: Path) -> None:
    """Baseline UTCI and the change for the given cases, each on its own symmetric colour scale."""
    base = read_stack(runs / "baseline", "UTCI")[hour]
    outdoor = outdoor_mask(rasters, base.shape)
    base = np.where(outdoor, base, np.nan)
    px = pixel_size(rasters)
    n = 1 + len(cases)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.9))
    im = axes[0].imshow(base, cmap="inferno")
    axes[0].set_title(f"UTCI at {hour:02d}:00, defaults")
    fig.colorbar(im, ax=axes[0], shrink=0.8, label="UTCI (°C)")
    for ax, c in zip(axes[1:], cases):
        d = np.where(outdoor, read_stack(runs / c.name, "UTCI")[hour], np.nan) - base
        lim = plotstyle.symmetric_limit(d)
        im = ax.imshow(d, cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_title(plotstyle.case_label(c.group, c.value, c.kind))
        fig.colorbar(im, ax=ax, shrink=0.8, label="ΔUTCI (°C)", extend="both")
    for ax in axes:
        plotstyle.map_axes(ax)
    plotstyle.scalebar(axes[0], px)
    fig.text(0.5, -0.02, "Left: UTCI with default inputs. Right: change in UTCI when one input is changed as titled, "
             "each on its own scale (99.5th percentile of |Δ|); magnitudes are compared in the tornado plot. "
             "White: roofs.", ha="center", fontsize=8, color="0.35")
    fig.savefig(path)
    plt.close(fig)


def diurnal(runs: Path, rasters: Path, path: Path) -> None:
    """Domain-mean UTCI over the day for baseline and the air-temperature and wind cases."""
    base = read_stack(runs / "baseline", "UTCI")
    outdoor = outdoor_mask(rasters, base.shape[1:])
    cases = {c.name: c for c in default_cases()}
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    for name, ls in [("baseline", "-"), ("ta_lo", "--"), ("ta_hi", "-"), ("wind_lo", "--"), ("wind_hi", "-")]:
        u = base if name == "baseline" else read_stack(runs / name, "UTCI")
        c = cases.get(name)
        color = "0.55" if c is None else plotstyle.INPUT_COLORS[c.group]
        label = "Defaults" if c is None else plotstyle.case_label(c.group, c.value, c.kind)
        ax.plot([np.nanmean(u[h][outdoor]) for h in range(u.shape[0])], ls, color=color,
                lw=3 if c is None else 1.5, label=label, zorder=1 if c is None else 2)
    ax.axhline(38, color="0.5", lw=0.8)
    ax.text(23, 38.3, "very strong heat stress (UTCI 38 °C)", fontsize=8, color="0.4", ha="right")
    ax.set_xlabel("Hour (local time)")
    ax.set_ylabel("Mean outdoor UTCI (°C)")
    ax.set_xlim(0, 23)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.grid(alpha=0.3)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rasters", type=Path, required=True)
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--out", type=Path, default=Path("figures"))
    ap.add_argument("--hour", type=int, default=14)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plotstyle.use()

    cases = [c for c in default_cases() if (args.runs / c.name / "output_folder").exists()]
    df = summarise(args.runs, args.rasters, cases, args.hour)
    df.to_csv(args.out / f"summary_{args.hour:02d}h.csv", index=False, float_format="%.3f")
    print(df.round(2).to_string(index=False))
    print({k: round(v, 3) for k, v in df.attrs.items()})

    tornado(df, args.out / f"tornado_{args.hour:02d}h.png", args.hour)
    # Map the cases whose effect varies most across space; uniform shifts make flat maps.
    base = read_stack(args.runs / "baseline", "UTCI")[args.hour]
    outdoor = outdoor_mask(args.rasters, base.shape)
    spread = {c.name: float(np.nanstd((read_stack(args.runs / c.name, "UTCI")[args.hour] - base)[outdoor]))
              for c in cases}
    top = (df.assign(s=df["case"].map(spread)).sort_values("s", ascending=False)
           .drop_duplicates("input")["case"].head(3).tolist())
    by_name = {c.name: c for c in cases}
    maps(args.runs, args.rasters, [by_name[n] for n in top], args.hour, args.out / f"maps_{args.hour:02d}h.png")
    if all((args.runs / n).exists() for n in ["ta_lo", "ta_hi", "wind_lo", "wind_hi"]):
        diurnal(args.runs, args.rasters, args.out / "diurnal.png")


if __name__ == "__main__":
    main()
