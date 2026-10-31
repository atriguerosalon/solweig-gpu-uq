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

from solweig_sens.analysis import outdoor_mask, read_stack, summarise  # noqa: E402
from solweig_sens.perturb import default_cases  # noqa: E402


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
            ax.barh(i, lo[col], color="#4C78A8", height=0.6)
            ax.barh(i, hi[col], color="#E45756", height=0.6)
        ax.axvline(0, color="0.3", lw=0.8)
        ax.set_title(f"{title} ground, {hour:02d}:00")
        ax.set_xlabel("Change in mean UTCI (°C)")
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_yticks(range(len(groups)), groups)
    axes[1].legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color="#4C78A8"), plt.Rectangle((0, 0), 1, 1, color="#E45756")],
        labels=["low value", "high value"], loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def maps(runs: Path, rasters: Path, cases: list[str], hour: int, path: Path) -> None:
    """Baseline UTCI and the change for the given cases."""
    base = read_stack(runs / "baseline", "UTCI")[hour]
    outdoor = outdoor_mask(rasters, base.shape)
    base = np.where(outdoor, base, np.nan)
    n = 1 + len(cases)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.4))
    im = axes[0].imshow(base, cmap="inferno")
    axes[0].set_title(f"Baseline UTCI, {hour:02d}:00")
    fig.colorbar(im, ax=axes[0], shrink=0.75, label="°C")
    diffs = [np.where(outdoor, read_stack(runs / c, "UTCI")[hour], np.nan) - base for c in cases]
    lim = max(np.nanpercentile(np.abs(d), 99) for d in diffs)
    for ax, c, d in zip(axes[1:], cases, diffs):
        im = ax.imshow(d, cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_title(f"ΔUTCI: {c}")
        fig.colorbar(im, ax=ax, shrink=0.75, label="°C")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def diurnal(runs: Path, rasters: Path, path: Path) -> None:
    """Domain-mean UTCI over the day for baseline and the air-temperature and wind cases."""
    base = read_stack(runs / "baseline", "UTCI")
    outdoor = outdoor_mask(rasters, base.shape[1:])
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for name, style in [("baseline", "k-"), ("ta_lo", "b--"), ("ta_hi", "r--"), ("wind_lo", "m:"), ("wind_hi", "c:")]:
        u = base if name == "baseline" else read_stack(runs / name, "UTCI")
        ax.plot([np.nanmean(u[h][outdoor]) for h in range(u.shape[0])], style, label=name)
    ax.axhline(38, color="0.5", lw=0.8)
    ax.text(0.3, 38.3, "very strong heat stress (38 °C)", fontsize=8, color="0.4")
    ax.set_xlabel("Hour (local)")
    ax.set_ylabel("Mean outdoor UTCI (°C)")
    ax.legend(frameon=False, fontsize=8, ncol=3)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rasters", type=Path, required=True)
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--out", type=Path, default=Path("figures"))
    ap.add_argument("--hour", type=int, default=14)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

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
    maps(args.runs, args.rasters, top, args.hour, args.out / f"maps_{args.hour:02d}h.png")
    if all((args.runs / n).exists() for n in ["ta_lo", "ta_hi", "wind_lo", "wind_hi"]):
        diurnal(args.runs, args.rasters, args.out / "diurnal.png")


if __name__ == "__main__":
    main()
