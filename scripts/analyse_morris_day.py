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
from solweig_sens import plotstyle  # noqa: E402
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
    # bootstrap 95% interval of mu* over the trajectories (only 10 of them)
    boot = np.random.default_rng(0).integers(0, len(ee), size=(2000, len(ee)))
    mu_b = np.nanmean(np.abs(ee[boot]), axis=1)  # (resamples, k, hours)
    lo95, hi95 = np.percentile(mu_b, 2.5, axis=0), np.percentile(mu_b, 97.5, axis=0)
    hours = np.arange(Y.shape[1])
    df = pd.DataFrame([{"hour": h, "input": labels[p], "mu_star": mu_star[p, h], "sigma": sigma[p, h],
                        "mu_star_lo95": lo95[p, h], "mu_star_hi95": hi95[p, h]}
                       for p in range(k) for h in hours])
    df.to_csv(args.out / "morris_day_indices.csv", index=False, float_format="%.3f")

    order = np.argsort(-mu_star.mean(axis=1))
    print("mu* (C UTCI per full range) at selected hours")
    print(df[df.hour.isin([3, 9, 14, 18, 22])].pivot(index="input", columns="hour", values="mu_star")
          .loc[[labels[p] for p in order]].round(2).to_string())
    print("\nbaseline mean outdoor UTCI by hour:", np.round(base[:, outdoor].mean(axis=1), 1).tolist())

    plotstyle.use()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
    im = axes[0].imshow(mu_star[order], aspect="auto", cmap="viridis")
    axes[0].set_yticks(range(k), [plotstyle.input_label(labels[p]) for p in order])
    axes[0].set_xticks(hours[::2], hours[::2])
    axes[0].set_xlabel("Hour (local time)")
    axes[0].set_title("All nine inputs, every hour")
    fig.colorbar(im, ax=axes[0], shrink=0.85, label="μ* (°C)")
    rest = [p for p in order if labels[p] not in ("air temperature", "relative humidity")]
    for p in sorted(rest, key=lambda p: -mu_star[p].max()):  # legend in order of peak importance
        c = plotstyle.INPUT_COLORS[labels[p]]
        axes[1].fill_between(hours, lo95[p], hi95[p], color=c, alpha=0.18, lw=0)
        axes[1].plot(hours, mu_star[p], color=c, lw=1.6, label=plotstyle.input_label(labels[p]))
    axes[1].set_xlabel("Hour (local time)")
    axes[1].set_ylabel("μ* (°C)")
    axes[1].set_xlim(0, 23)
    axes[1].set_title("The seven others, with 95% bootstrap bands")
    axes[1].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    axes[1].grid(alpha=0.3)
    fig.text(0.5, -0.03, "Morris screening on the Austin sample, 100 full-day runs. μ*: change in domain-mean "
             "UTCI over the full range of an input. Air temperature and humidity lead at every hour (left).",
             ha="center", fontsize=8, color="0.35")
    fig.savefig(args.out / "morris_day.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
