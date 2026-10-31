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

    # Domain-level indices.
    rows = []
    for region, m in [("all", outdoor), ("sun", sun), ("shade", shade)]:
        Y = fields[:, m].mean(axis=1)
        ee = elementary_effects(X, Y, bounds, k)
        for p in range(k):
            rows.append({"region": region, "input": labels[p],
                         "mu_star": np.nanmean(np.abs(ee[:, p])), "mu": np.nanmean(ee[:, p]),
                         "sigma": np.nanstd(ee[:, p], ddof=1)})
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

    # mu* as bars, sigma as whiskers, sun and shade side by side.
    order = df[df.region == "all"].sort_values("mu_star").input.tolist()
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(order) + 1.4))
    for dy, region, color, name in [(0.2, "sun", "#E45756", "sunlit"), (-0.2, "shade", "#4C78A8", "shaded")]:
        sub = df[df.region == region].set_index("input").loc[order]
        ax.barh(y + dy, sub.mu_star, height=0.38, color=color, label=f"{name} ground")
        ax.errorbar(sub.mu_star, y + dy, xerr=[np.zeros(len(order)), sub.sigma], fmt="none",
                    ecolor="0.25", elinewidth=0.9, capsize=2)
    ax.set_yticks(y, order)
    ax.set_xlabel("μ*: mean |change in mean UTCI| per full input range (°C); whisker = σ")
    ax.set_title("Morris screening, 14:00, 100 runs")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(args.out / "morris_mu_sigma.png", dpi=160)
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
    cmap = plt.get_cmap("tab10")
    img = np.ones(dom.shape + (3,))
    for p in present:
        img[dom == p] = cmap(p)[:3]
    img[~outdoor] = 0.75
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.9))
    im = axes[0].imshow(np.where(outdoor, base, np.nan), cmap="inferno")
    axes[0].set_title("Baseline UTCI at 14:00 (roofs blank)")
    fig.colorbar(im, ax=axes[0], shrink=0.75, label="°C")
    axes[1].imshow(img)
    axes[1].set_title(f"Leading surface or body input\n(air temperature leads at {lead_ta:.0f}% of pixels)")
    axes[1].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=cmap(p)) for p in present],
                   labels=[f"{labels[p]} ({100 * (dom == p).sum() / outdoor.sum():.0f}%)" for p in present],
                   loc="upper center", bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=8, ncol=2)
    im = axes[2].imshow(np.where(outdoor, mu_px[surf].sum(axis=0), np.nan), cmap="viridis")
    axes[2].set_title("Summed μ* of the five surface and body inputs")
    fig.colorbar(im, ax=axes[2], shrink=0.75, label="°C")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(args.out / "morris_dominant_input.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
