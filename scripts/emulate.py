"""Per-pixel linear emulator of SOLWEIG-GPU, fitted on the Morris runs, and Monte Carlo maps from it.

Morris showed the inputs barely interact, so UTCI at each pixel may be close to linear
in the nine inputs. If so, a linear fit on the 100 runs reproduces the model, and
propagating input uncertainty becomes a matrix product instead of thousands of runs.

Checked by leaving out one Morris trajectory at a time. Inputs are drawn uniformly over
their ranges: our assumption, not measured uncertainty.

Usage:
    python scripts/emulate.py --runs runs_morris --out figures
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solweig_sens.analysis import UTCI_VERY_STRONG, outdoor_mask  # noqa: E402


def design(Xn: np.ndarray, quadratic: bool = False) -> np.ndarray:
    """Intercept, linear terms and, optionally, squares and pairwise products."""
    cols = [np.ones(len(Xn)), *Xn.T]
    if quadratic:
        k = Xn.shape[1]
        cols += [Xn[:, i] * Xn[:, j] for i in range(k) for j in range(i, k)]
    return np.stack(cols, axis=1)


def loto_errors(Xn: np.ndarray, Y: np.ndarray, k: int, quadratic: bool, ridge: float = 0.0) -> np.ndarray:
    """Leave-one-trajectory-out residuals, shape like Y."""
    traj = np.arange(len(Xn)) // (k + 1)
    resid = np.empty_like(Y)
    for t in np.unique(traj):
        tr, te = traj != t, traj == t
        A = design(Xn[tr], quadratic)
        coef = np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]), A.T @ Y[tr])
        resid[te] = Y[te] - design(Xn[te], quadratic) @ coef
    return resid


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path("runs_morris"))
    ap.add_argument("--out", type=Path, default=Path("figures"))
    ap.add_argument("--samples", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    problem = json.loads((args.runs / "problem.json").read_text())
    k, bounds = problem["num_vars"], np.array(problem["bounds"])
    X = np.load(args.runs / "X.npy")
    Xn = (X - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    base = np.load(args.runs / "fields" / "baseline.npy")
    outdoor = outdoor_mask(args.runs / "rasters", base.shape) & np.isfinite(base)
    Y = np.stack([np.load(args.runs / "fields" / f"run_{i:03d}.npy") for i in range(len(X))])[:, outdoor]
    Y = Y.astype(np.float64)

    # How well does the emulator reproduce held-out runs?
    print(f"spread of UTCI across runs (std, median pixel): {np.median(Y.std(axis=0)):.2f} C")
    for name, quad, ridge in [("linear", False, 0.0), ("quadratic", True, 1e-2)]:
        r = loto_errors(Xn, Y, k, quad, ridge)
        r38 = ((Y >= UTCI_VERY_STRONG) != (Y - r >= UTCI_VERY_STRONG)).mean()
        print(f"{name:9s} held-out RMSE: median pixel {np.median(np.sqrt((r ** 2).mean(0))):.3f} C, "
              f"99th pct pixel {np.percentile(np.sqrt((r ** 2).mean(0)), 99):.3f} C, "
              f"max |error| {np.abs(r).max():.2f} C, domain mean {np.sqrt((r.mean(1) ** 2).mean()):.3f} C, "
              f"wrong side of 38 C: {100 * r38:.2f}% of pixel-runs")

    # Monte Carlo through the linear emulator, in chunks to keep memory small.
    A = design(Xn)
    coef = np.linalg.lstsq(A, Y, rcond=None)[0]  # (k+1, pixels)
    rng = np.random.default_rng(args.seed)
    S = rng.random((args.samples, k))
    n, s1, s2 = 0, np.zeros(Y.shape[1]), np.zeros(Y.shape[1])
    exceed, area = np.zeros(Y.shape[1]), []
    q_lo, q_hi = [], []
    for chunk in np.array_split(S, max(1, args.samples // 500)):
        U = design(chunk) @ coef
        n += len(U)
        s1 += U.sum(0)
        s2 += (U ** 2).sum(0)
        exceed += (U >= UTCI_VERY_STRONG).sum(0)
        area.append(100 * (U >= UTCI_VERY_STRONG).mean(1))
        q_lo.append(np.percentile(U, 5, axis=0))
        q_hi.append(np.percentile(U, 95, axis=0))
    mean, std = s1 / n, np.sqrt(s2 / n - (s1 / n) ** 2)
    p38 = exceed / n
    width = np.mean(q_hi, axis=0) - np.mean(q_lo, axis=0)  # chunk-averaged 5-95% width
    area = np.concatenate(area)
    base_o = base[outdoor]
    print(f"Monte Carlo ({n} samples): UTCI std median pixel {np.median(std):.2f} C "
          f"(5-95% of pixels {np.percentile(std, 5):.2f}-{np.percentile(std, 95):.2f}); "
          f"5-95% interval width median {np.median(width):.2f} C")
    print(f"area >= 38 C: baseline {100 * (base_o >= UTCI_VERY_STRONG).mean():.1f}%, "
          f"Monte Carlo median {np.median(area):.1f}%, 5-95% {np.percentile(area, 5):.1f}-{np.percentile(area, 95):.1f}%")
    unsure = (p38 > 0.1) & (p38 < 0.9)
    print(f"pixels where P(UTCI >= 38) is between 0.1 and 0.9: {100 * unsure.mean():.1f}%; "
          f"baseline says >= 38 but P < 0.5: {100 * ((base_o >= UTCI_VERY_STRONG) & (p38 < 0.5)).mean():.1f}%")

    def to_map(v):
        m = np.full(base.shape, np.nan)
        m[outdoor] = v
        return m

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.9))
    for ax, data, title, cmap, label, lim in [
        (axes[0], base_o, "Baseline UTCI, 14:00", "inferno", "°C", None),
        (axes[1], std, "Spread of UTCI from input uncertainty (std)", "viridis", "°C", None),
        (axes[2], p38, "Probability of very strong heat stress (UTCI ≥ 38 °C)", "RdBu_r", "probability", (0, 1)),
    ]:
        im = ax.imshow(to_map(data), cmap=cmap, vmin=None if lim is None else lim[0], vmax=None if lim is None else lim[1])
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.75, label=label)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Linear emulator fitted on 100 SOLWEIG-GPU runs; inputs uniform over the assumed ranges", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out / "emulator_uq_maps.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
