"""Morris indices for domain-mean UTCI (sun, shade) and per-pixel maps of the dominant input.

Usage:
    python scripts/analyse_morris.py --runs runs_morris --out figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solweig_sens import plotstyle  # noqa: E402
from solweig_sens.analysis import outdoor_mask  # noqa: E402


def elementary_effects(X: np.ndarray, Y: np.ndarray, bounds: np.ndarray, k: int) -> np.ndarray:
    """Elementary effects per trajectory step: array (trajectories, k, *Y.shape[1:]).

    Inputs are scaled to [0, 1], so each effect is the output change per full range of that input.
    """
    Xn = (X - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    n_traj = len(X) // (k + 1)
    ee = np.full((n_traj, k) + Y.shape[1:], np.nan, dtype=np.float64)
    for t in range(n_traj):
        rows = slice(t * (k + 1), (t + 1) * (k + 1))
        xs, ys = Xn[rows], Y[rows]
        for j in range(k):
            dx = xs[j + 1] - xs[j]
            p = int(np.argmax(np.abs(dx)))
            ee[t, p] = (ys[j + 1] - ys[j]) / dx[p]
    return ee


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path("runs_morris"))
    ap.add_argument("--out", type=Path, default=Path("figures"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    problem = json.loads((args.runs / "problem.json").read_text())
    k, labels = problem["num_vars"], problem["labels"]
    bounds = np.array(problem["bounds"])
    X = np.load(args.runs / "X.npy")
    fields = np.stack([np.load(args.runs / "fields" / f"run_{i:03d}.npy") for i in range(len(X))])
    base = np.load(args.runs / "fields" / "baseline.npy")
    shadow = np.load(args.runs / "fields" / "shadow.npy")
    outdoor = outdoor_mask(args.runs / "rasters", base.shape) & np.isfinite(base)
    sun, shade = outdoor & (shadow > 0.5), outdoor & (shadow <= 0.5)

    # Domain-level indices, with a bootstrap 95% interval for mu* over the trajectories (only 10 of them,
    # so mu* itself is uncertain; sigma is a different thing: the spread of the effects, i.e. interactions).
    rng = np.random.default_rng(0)
    rows = []
    for region, m in [("all", outdoor), ("sun", sun), ("shade", shade)]:
        Y = fields[:, m].mean(axis=1)
        ee = elementary_effects(X, Y, bounds, k)
        boot = rng.integers(0, len(ee), size=(2000, len(ee)))
        for p in range(k):
            mu_b = np.nanmean(np.abs(ee[boot, p]), axis=1)
            rows.append({"region": region, "input": labels[p],
                         "mu_star": np.nanmean(np.abs(ee[:, p])), "mu": np.nanmean(ee[:, p]),
                         "sigma": np.nanstd(ee[:, p], ddof=1),
                         "mu_star_lo95": np.percentile(mu_b, 2.5), "mu_star_hi95": np.percentile(mu_b, 97.5)})
    Y38 = 100 * (fields[:, outdoor] >= 38.0).mean(axis=1)
    ee38 = elementary_effects(X, Y38, bounds, k)
    for p in range(k):
        rows.append({"region": "area>=38C (pct points)", "input": labels[p],
                     "mu_star": np.nanmean(np.abs(ee38[:, p])), "mu": np.nanmean(ee38[:, p]),
                     "sigma": np.nanstd(ee38[:, p], ddof=1)})
    df = pd.DataFrame(rows)
    df.to_csv(args.out / "morris_indices.csv", index=False, float_format="%.3f")
    print(df.pivot(index="input", columns="region", values="mu_star").round(2)
          .sort_values("all", ascending=False).to_string())

    # Left: mu* with its bootstrap 95% interval, sun and shade side by side. Right: mu* against sigma
    # (the classic Morris plot: far right = important, high up = interacting or non-linear).
    plotstyle.use()
    order = df[df.region == "all"].sort_values("mu_star").input.tolist()
    y = np.arange(len(order))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 0.42 * len(order) + 1.8), gridspec_kw={"width_ratios": [1.25, 1]})
    for dy, region, color, name in [(0.2, "sun", plotstyle.SUN, "sunlit"), (-0.2, "shade", plotstyle.SHADE, "shaded")]:
        sub = df[df.region == region].set_index("input").loc[order]
        ax.barh(y + dy, sub.mu_star, height=0.38, color=color, label=f"{name} ground")
        ax.errorbar(sub.mu_star, y + dy, xerr=[sub.mu_star - sub.mu_star_lo95, sub.mu_star_hi95 - sub.mu_star],
                    fmt="none", ecolor="0.2", elinewidth=1, capsize=2.5)
    ax.set_yticks(y, [plotstyle.input_label(o) for o in order])
    ax.set_xlabel("μ*: change in mean UTCI over the full input range (°C)")
    ax.set_title("Importance, with 95% bootstrap interval")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(frameon=False, loc="lower right")
    for region, marker, name in [("sun", "o", "sunlit"), ("shade", "s", "shaded")]:
        sub = df[df.region == region].set_index("input")
        for inp, r in sub.iterrows():
            bx.scatter(r.mu_star, r.sigma, marker=marker, s=46, color=plotstyle.INPUT_COLORS[inp],
                       edgecolor="white", linewidth=0.6, zorder=3)
    ss = df[df.region.isin(["sun", "shade"])]
    xmax, ymax = ss.mu_star.max() * 1.08, ss.sigma.max() * 1.35
    bx.plot([0, xmax], [0, 0.1 * xmax], color="0.6", lw=0.8, ls="--")
    xl = min(xmax, 0.95 * ymax / 0.1)  # where the line leaves the panel
    bx.text(xl * 0.97, 0.1 * xl, "σ = 0.1 μ*", ha="right", va="bottom", fontsize=8, color="0.45",
            rotation=0)
    bx.set_xlim(0, xmax)
    bx.set_ylim(0, ymax)
    bx.set_xlabel("μ* (°C)")
    bx.set_ylabel("σ (°C): spread of the effect")
    bx.set_title("Importance against interaction")
    bx.grid(alpha=0.3)
    handles = [plt.Line2D([], [], ls="", marker="o", color=plotstyle.INPUT_COLORS[o], label=plotstyle.input_label(o))
               for o in reversed(order)]
    handles += [plt.Line2D([], [], ls="", marker=mk, color="0.4", mfc="none", label=f"{nm} ground")
                for mk, nm in (("o", "sunlit"), ("s", "shaded"))]
    bx.legend(handles=handles, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.text(0.5, -0.03, "Morris screening on the Austin sample at 14:00: 10 trajectories, 100 SOLWEIG-GPU runs. "
             "Intervals: 2000 bootstrap resamples of the trajectories. Below the dashed line an input acts almost "
             "additively (little interaction or non-linearity).", ha="center", fontsize=8, color="0.35")
    fig.savefig(args.out / "morris_mu_sigma.png")
    plt.close(fig)

    # Per-pixel. Air temperature leads everywhere, so map the leading surface or body input
    # instead, and the summed mu* of those inputs (how much the uncertain materials matter).
    ee_px = elementary_effects(X, fields.astype(np.float64), bounds, k)
    mu_px = np.nanmean(np.abs(ee_px), axis=0)  # (k, rows, cols)
    weather = {"air temperature", "relative humidity", "wind speed", "global radiation"}
    surf = [p for p in range(k) if labels[p] not in weather]
    lead_ta = 100 * (np.argmax(mu_px, axis=0)[outdoor] == labels.index("air temperature")).mean()
    dom = np.where(outdoor, np.array(surf)[np.argmax(mu_px[surf], axis=0)], -1)
    present = [p for p in surf if (dom == p).any()]
    share = lambda m: 100 * m.sum() / outdoor.sum()  # noqa: E731
    pct = lambda m: f"{share(m):.0f}%" if share(m) >= 0.5 else "<1%"  # noqa: E731
    colors ={p: matplotlib.colors.to_rgb(plotstyle.INPUT_COLORS[labels[p]]) for p in present}
    img = np.ones(dom.shape + (3,))
    for p in present:
        img[dom == p] = colors[p]
    img[~outdoor] = 0.85
    from osgeo import gdal

    px = abs(gdal.Open(str(args.runs / "rasters" / "Building_DSM.tif")).GetGeoTransform()[1])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.9))
    im = axes[0].imshow(np.where(outdoor, base, np.nan), cmap="inferno")
    axes[0].set_title("UTCI at 14:00, defaults")
    fig.colorbar(im, ax=axes[0], shrink=0.8, label="UTCI (°C)")
    axes[1].imshow(img)
    axes[1].set_title("Which uncertain material or body input matters most")
    axes[1].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=colors[p]) for p in present],
                   labels=[f"{plotstyle.input_label(labels[p])} ({pct(dom == p)})" for p in present],
                   loc="upper center", bbox_to_anchor=(0.5, -0.01), frameon=False,
                   ncol=2)
    im = axes[2].imshow(np.where(outdoor, mu_px[surf].sum(axis=0), np.nan), cmap="viridis")
    axes[2].set_title("How much they matter together")
    fig.colorbar(im, ax=axes[2], shrink=0.8, label="sum of μ* of the five (°C)")
    for ax in axes:
        plotstyle.map_axes(ax)
    plotstyle.scalebar(axes[0], px)
    fig.text(0.5, -0.04, f"Austin sample, Morris screening (100 runs). Air temperature is the leading input at "
             f"{lead_ta:.0f}% of pixels, so the middle map ranks the five surface and body properties "
             "(albedo, emissivity, body absorption). Grey and white: roofs.", ha="center", fontsize=8, color="0.35")
    fig.savefig(args.out / "morris_dominant_input.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
