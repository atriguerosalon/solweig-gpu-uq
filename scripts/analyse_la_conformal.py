"""Are the emulator's error bars right? Gaussian against conformal intervals, Pacoima, UTCI at 14:00.

The emulator is the best one of analyse_la_emulator_days.py: Tmrt linear in the nine inputs
(fitted on 100 Morris runs, of the same day or of the other day), then the exact UTCI
polynomial. Its error on the 50 Latin hypercube runs of each day is the thing to bound. Some
of those runs calibrate the interval, the rest test it (random splits, repeated).

Intervals, all aimed at 90%, symmetric around the prediction:

    gauss          +-1.645 x RMS of the calibration errors, pooled over pixels
    conformal      the 90% conformal quantile of |error|, pooled over pixels
    mondrian       conformal, separately for pixels that switch between sun and shade from
                   one day to the other (14:00) and for the rest: the free flag of section 4
    pixel conf.    conformal per pixel (needs at least 9 calibration runs for 90%)

Scenarios: the emulator fitted on the same day; and fitted on the other day ("transferred"),
calibrated with k runs of the new day, or with k = 0, on the old day's own transfer errors
(no new runs at all). Coverage is reported over all test pixels and runs, and on the switching
pixels alone.

Usage:
    python scripts/analyse_la_conformal.py --runs runs_la_emu --out figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyse_la_emulator_days import X0, load, utci_at  # noqa: E402
from emulate import design  # noqa: E402
from solweig_sens.analysis import outdoor_mask  # noqa: E402

ALPHA, SPLITS, SEED = 0.10, 20, 0


def conf_q(scores: np.ndarray, axis=None) -> np.ndarray:
    """Split-conformal quantile: the ceil((n+1)(1-alpha))-th smallest score."""
    n = scores.size if axis is None else scores.shape[axis]
    level = np.ceil((n + 1) * (1 - ALPHA)) / n
    return np.full(scores.shape[1:], np.inf) if level > 1 and axis is not None else \
        np.quantile(scores, min(level, 1.0), axis=axis, method="higher")


def intervals(cal: np.ndarray, flip: np.ndarray) -> dict[str, np.ndarray]:
    """Half-widths per pixel from calibration errors (runs, pixels)."""
    a = np.abs(cal)
    hw = {"gauss": np.full(cal.shape[1], 1.645 * np.sqrt((cal ** 2).mean())),
          "conformal": np.full(cal.shape[1], conf_q(a)),
          "mondrian": np.where(flip, conf_q(a[:, flip]), conf_q(a[:, ~flip]))}
    if cal.shape[0] >= int(np.ceil(1 / ALPHA)) - 1:
        hw["pixel conf."] = conf_q(a, axis=0)
    return hw


def evaluate(hw: dict, test: np.ndarray, flip: np.ndarray) -> dict:
    out = {}
    for k, h in hw.items():
        inside = np.abs(test) <= h
        out[k] = {"coverage": float(inside.mean()), "coverage_switching": float(inside[:, flip].mean()),
                  "coverage_rest": float(inside[:, ~flip].mean()),
                  "width_median": float(2 * np.median(h)), "width_switching": float(2 * np.median(h[flip]))}
    return out


def mean_over_splits(results: list[dict]) -> dict:
    return {m: {s: float(np.mean([r[m][s] for r in results])) for s in results[0][m]} for m in results[0]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path("runs_la_emu"))
    ap.add_argument("--out", type=Path, default=Path("figures"))
    args = ap.parse_args()
    problem = json.loads((args.runs / "problem.json").read_text())
    b = np.array(problem["bounds"])
    norm = lambda X: (X - b[:, 0]) / (b[:, 1] - b[:, 0])  # noqa: E731
    Xm, Xl, x0 = norm(np.load(args.runs / "X_morris.npy")), norm(np.load(args.runs / "X_lhs.npy")), norm(X0[None])
    X_lhs = np.load(args.runs / "X_lhs.npy")
    days = sorted(p.name for p in args.runs.iterdir() if p.is_dir() and (p / "fields").exists())
    shape = np.load(args.runs / days[0] / "fields" / "baseline.npy").shape
    outdoor = outdoor_mask(args.runs / days[0] / "rasters", shape)
    for d in days:
        outdoor &= np.isfinite(np.load(args.runs / d / "fields" / "baseline.npy"))
    utci = {d: load(args.runs / d, len(Xm), len(Xl), outdoor) for d in days}
    tmrt = {d: load(args.runs / d, len(Xm), len(Xl), outdoor, "tmrt") for d in days}
    coef = {d: np.linalg.lstsq(design(Xm), tmrt[d]["morris"], rcond=None)[0] for d in days}
    sun = {d: np.load(args.runs / d / "fields" / "shadow.npy")[outdoor] > 0.5 for d in days}
    flip = sun[days[0]] != sun[days[1]]

    def errors(d: str, fitted_on: str) -> np.ndarray:
        """UTCI error of the emulator on day d's test runs (runs, pixels)."""
        tp = design(Xl) @ coef[d] if fitted_on == d else tmrt[d]["base"] + (design(Xl) - design(x0)) @ coef[fitted_on]
        met = args.runs / d / f"met_0to{problem.get('hour', 14):02d}.txt"
        return np.stack([utci_at(met, problem["names"], x, t) for x, t in zip(X_lhs, tp)]) - utci[d]["lhs"]

    err = {(d, f): errors(d, f) for d in days for f in days}
    rng = np.random.default_rng(SEED)
    out = {"pixels": int(outdoor.sum()), "switching_pixels": int(flip.sum()), "alpha": ALPHA, "splits": SPLITS,
           "results": {}}
    n = len(Xl)
    for d in days:
        other = [o for o in days if o != d][0]
        for label, e in (("same day", err[(d, d)]), (f"transferred from {other}", err[(d, other)])):
            for k in (5, 10, 25):
                res = []
                for _ in range(SPLITS):
                    idx = rng.permutation(n)
                    res.append(evaluate(intervals(e[idx[:k]], flip), e[idx[k:]], flip))
                out["results"][f"{d} | {label} | k={k}"] = mean_over_splits(res)
            if label.startswith("transferred"):
                # no runs on the new day: calibrate on the other day's own transfer errors
                out["results"][f"{d} | {label} | k=0 (calibrated on {other})"] = \
                    evaluate(intervals(err[(other, d)], flip), e, flip)
    path = args.out / "la_conformal.json"
    path.write_text(json.dumps(out, indent=2))

    print(f"outdoor pixels {out['pixels']}, switching sun/shade between the days {out['switching_pixels']}; "
          f"target 90%, mean over {SPLITS} splits")
    print(f"{'':62s} {'cover':>6s} {'switch':>6s} {'rest':>6s} {'width':>6s} {'w.sw':>6s}")
    for key, r in out["results"].items():
        print(f"== {key}")
        for m, s in r.items():
            print(f"   {m:58s} {100 * s['coverage']:5.1f}% {100 * s['coverage_switching']:5.1f}% "
                  f"{100 * s['coverage_rest']:5.1f}% {s['width_median']:6.2f} {s['width_switching']:6.2f}")
    print(f"[conformal] report: {path}")


if __name__ == "__main__":
    main()
