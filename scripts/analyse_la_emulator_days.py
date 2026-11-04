"""Linear emulators across two Pacoima days (runs from run_la_emulator_days.py), UTCI at 14:00.

Test set: the 50 Latin hypercube runs of each day. Predictors, from cheapest to fullest:

    baseline only      the day's default run, ignoring the inputs
    transferred        the day's default run plus the per-pixel sensitivities fitted on the
                       other day's 100 Morris runs (one SOLWEIG run on the new day)
    same day           per-pixel linear fit on the same day's 100 Morris runs

and the last two again with the linear fit on Tmrt instead of UTCI, followed by the exact UTCI
polynomial (Ta, RH and wind are known inputs; with no wind-coefficient rasters the wind is the
same everywhere). Also compares the fitted sensitivities themselves between the days.

Usage:
    python scripts/analyse_la_emulator_days.py --runs runs_la_emu --out figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from emulate import design  # noqa: E402
from solweig_gpu.calculate_utci import utci_calculator  # noqa: E402
from solweig_sens.analysis import UTCI_VERY_STRONG, outdoor_mask  # noqa: E402

# SOLWEIG-GPU defaults for the nine inputs (utci_process globals; dark asphalt, class 1, in the table)
X0 = np.array([0.20, 0.90, 0.70, 0.18, 0.95, 0.0, 0.0, 1.0, 1.0])


def load(day_dir: Path, n_morris: int, n_lhs: int, outdoor: np.ndarray, var: str = "") -> dict:
    sfx = f"_{var}" if var else ""
    f = lambda name: np.load(day_dir / "fields" / f"{name}{sfx}.npy")[outdoor].astype(np.float64)  # noqa: E731
    return {"base": f("baseline"), "morris": np.stack([f(f"morris_{i:03d}") for i in range(n_morris)]),
            "lhs": np.stack([f(f"lhs_{i:03d}") for i in range(n_lhs)])}


def utci_at(met_file: Path, names: list[str], x: np.ndarray, tmrt: np.ndarray) -> np.ndarray:
    """UTCI at the last hour of a met file for one input sample x (raw units), as SOLWEIG-GPU computes it."""
    ta, rh, ws = np.loadtxt(met_file, skiprows=1)[-1, [11, 10, 9]]
    v = dict(zip(names, x))
    t = lambda a: torch.full(tmrt.shape, float(a), dtype=torch.float32)  # noqa: E731
    return utci_calculator(t(ta + v["ta_add"]), t(np.clip(rh + v["rh_add"], 0, 100)),
                           torch.as_tensor(tmrt, dtype=torch.float32), t(max(ws * v["wind_scale"], 0.15))).numpy()


def scores(pred: np.ndarray, truth: np.ndarray) -> dict:
    r = pred - truth
    rmse_px = np.sqrt((r ** 2).mean(0))
    return {"rmse_median_pixel": float(np.median(rmse_px)), "rmse_p99_pixel": float(np.percentile(rmse_px, 99)),
            "max_abs": float(np.abs(r).max()), "bias": float(r.mean()),
            "wrong_side_of_38C_pct": float(100 * ((pred >= UTCI_VERY_STRONG) != (truth >= UTCI_VERY_STRONG)).mean())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path("runs_la_emu"))
    ap.add_argument("--out", type=Path, default=Path("figures"))
    args = ap.parse_args()
    problem = json.loads((args.runs / "problem.json").read_text())
    b = np.array(problem["bounds"])
    norm = lambda X: (X - b[:, 0]) / (b[:, 1] - b[:, 0])  # noqa: E731
    Xm, Xl = norm(np.load(args.runs / "X_morris.npy")), norm(np.load(args.runs / "X_lhs.npy"))
    x0 = norm(X0[None])
    days = sorted(p.name for p in args.runs.iterdir() if p.is_dir() and (p / "fields").exists())
    shape = np.load(args.runs / days[0] / "fields" / "baseline.npy").shape
    outdoor = outdoor_mask(args.runs / days[0] / "rasters", shape)
    for d in days:
        outdoor &= np.isfinite(np.load(args.runs / d / "fields" / "baseline.npy"))
    data = {d: load(args.runs / d, len(Xm), len(Xl), outdoor) for d in days}
    tmrt = {d: load(args.runs / d, len(Xm), len(Xl), outdoor, "tmrt") for d in days}
    coef = {d: np.linalg.lstsq(design(Xm), data[d]["morris"], rcond=None)[0] for d in days}  # (k+1, pixels)
    coef_t = {d: np.linalg.lstsq(design(Xm), tmrt[d]["morris"], rcond=None)[0] for d in days}
    X_lhs = np.load(args.runs / "X_lhs.npy")

    out = {"pixels": int(outdoor.sum()), "test": {}, "sensitivities": {}}
    for d in days:
        other = [o for o in days if o != d][0]
        truth = data[d]["lhs"]
        dx = design(Xl) - design(x0)  # intercept cancels
        preds = {
            "baseline only": np.broadcast_to(data[d]["base"], truth.shape),
            "transferred from " + other: data[d]["base"] + dx @ coef[other],
            "same day": design(Xl) @ coef[d],
        }
        met = args.runs / d / f"met_0to{problem.get('hour', 14):02d}.txt"
        for k, tp in {"Tmrt transferred from " + other: tmrt[d]["base"] + dx @ coef_t[other],
                      "Tmrt same day": design(Xl) @ coef_t[d]}.items():
            preds[k + ", exact UTCI"] = np.stack([utci_at(met, problem["names"], x, t) for x, t in zip(X_lhs, tp)])
        out["test"][d] = {"spread_std_median_pixel": float(np.median(truth.std(0))),
                          "tmrt_spread_std_median_pixel": float(np.median(tmrt[d]["lhs"].std(0))),
                          **{k: scores(v, truth) for k, v in preds.items()}}
    for i, name in enumerate(problem["labels"], 1):
        a, c = coef[days[0]][i], coef[days[1]][i]
        out["sensitivities"][name] = {f"median {d}": float(np.median(coef[d][i])) for d in days}
        out["sensitivities"][name]["pixel correlation"] = float(np.corrcoef(a, c)[0, 1]) if a.std() and c.std() else None
    path = args.out / "la_emulator_days.json"
    path.write_text(json.dumps(out, indent=2))

    print(f"outdoor pixels: {out['pixels']}")
    for d, r in out["test"].items():
        print(f"== {d}: spread across test runs (std, median pixel) UTCI {r['spread_std_median_pixel']:.2f} C, "
              f"Tmrt {r['tmrt_spread_std_median_pixel']:.2f} K")
        for k, s in r.items():
            if isinstance(s, dict):
                print(f"   {k:46s} RMSE median px {s['rmse_median_pixel']:.3f}, 99th px {s['rmse_p99_pixel']:.3f}, "
                      f"max {s['max_abs']:.2f}, bias {s['bias']:+.3f}, wrong side of 38 C {s['wrong_side_of_38C_pct']:.2f}%")
    print("== sensitivity, UTCI change over the full input range (median pixel), and correlation of the maps")
    for name, s in out["sensitivities"].items():
        vals = "  ".join(f"{v:+.2f}" for k, v in s.items() if k.startswith("median"))
        corr = s["pixel correlation"]
        print(f"   {name:28s} {vals}   r = {corr:.2f}" if corr is not None else f"   {name:28s} {vals}")
    print(f"[emu] report: {path}")


if __name__ == "__main__":
    main()
