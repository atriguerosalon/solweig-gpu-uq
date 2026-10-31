"""Morris indices for every hour of the day: which input moves mean outdoor UTCI, and when.

Needs a run of run_morris.py with --hour 23 --all-hours.

Usage:
    python scripts/analyse_morris_day.py --runs runs_morris_day --out figures
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyse_morris import elementary_effects  # noqa: E402
from solweig_sens.analysis import outdoor_mask  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path("runs_morris_day"))
    ap.add_argument("--out", type=Path, default=Path("figures"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    problem = json.loads((args.runs / "problem.json").read_text())
    k, labels, bounds = problem["num_vars"], problem["labels"], np.array(problem["bounds"])
    X = np.load(args.runs / "X.npy")
    base = np.load(args.runs / "fields" / "baseline_hours.npy").astype(np.float64)
    outdoor = outdoor_mask(args.runs / "rasters", base.shape[1:]) & np.isfinite(base).all(axis=0)

    # Domain-mean UTCI per run and hour: (runs, hours).
    Y = np.stack([np.load(args.runs / "fields" / f"run_{i:03d}_hours.npy").astype(np.float64)[:, outdoor].mean(axis=1)
                  for i in range(len(X))])
    ee = elementary_effects(X, Y, bounds, k)  # (trajectories, k, hours)
    mu_star = np.nanmean(np.abs(ee), axis=0)
    sigma = np.nanstd(ee, axis=0, ddof=1)
    hours = np.arange(Y.shape[1])
    df = pd.DataFrame([{"hour": h, "input": labels[p], "mu_star": mu_star[p, h], "sigma": sigma[p, h]}
                       for p in range(k) for h in hours])
    df.to_csv(args.out / "morris_day_indices.csv", index=False, float_format="%.3f")

    order = np.argsort(-mu_star.mean(axis=1))
    print("mu* (C UTCI per full range) at selected hours")
    print(df[df.hour.isin([3, 9, 14, 18, 22])].pivot(index="input", columns="hour", values="mu_star")
          .loc[[labels[p] for p in order]].round(2).to_string())
    print("\nbaseline mean outdoor UTCI by hour:", np.round(base[:, outdoor].mean(axis=1), 1).tolist())

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
    im = axes[0].imshow(mu_star[order], aspect="auto", cmap="viridis")
    axes[0].set_yticks(range(k), [labels[p] for p in order])
    axes[0].set_xticks(hours[::2], hours[::2])
    axes[0].set_xlabel("Hour (local)")
    axes[0].set_title("μ* by hour (°C UTCI per full input range)")
    fig.colorbar(im, ax=axes[0], shrink=0.85)
    cmap = plt.get_cmap("tab10")
    for p in order:
        if labels[p] in ("air temperature", "relative humidity"):
            continue
        axes[1].plot(hours, mu_star[p], color=cmap(p), label=labels[p])
    axes[1].set_xlabel("Hour (local)")
    axes[1].set_ylabel("μ* (°C)")
    axes[1].set_title("Without air temperature and humidity")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out / "morris_day.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
