"""Sobol indices from a Gaussian-process surrogate of the Pacoima Sobol runs (run_la_sobol.py).

The direct estimators (analyse_la_sobol.py) resolve the mean outdoor UTCI and the area above 38 C
with N = 128 (1280 runs), but not the area above 32 C: a threshold quantity near saturation, whose
estimates have intervals wider than the indices. Here a Gaussian process is fitted to all 1280 runs
for each quantity and hour (inputs scaled to [0, 1], Matern 5/2 kernel with one length scale per
input, hyperparameters by maximum marginal likelihood, in PyTorch, on the GPU if there is one), checked by 5-fold
cross-validation, and the Sobol indices are computed from the surrogate's mean with a large
Saltelli design (N = 8192, SALib). Two checks that the surrogate can be trusted: its held-out R^2,
and agreement with the direct estimates where those are resolved (mean UTCI; area above 38 C at 13
and 16 h). Uncertainty: indices from 5 surrogates, each fitted without one fold, give a spread.

Usage:
    python scripts/sobol_surrogate.py --runs ~/sobol_la --out figures
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

QUANTITIES = ("utci_mean", "area_utci_above_32_pct", "area_utci_above_38_pct")
HOURS = (9, 13, 16, 19)


def matern52(a, b, ls, var):
    d = torch.cdist(a / ls, b / ls)
    s5 = math.sqrt(5.0) * d
    return var * (1 + s5 + s5 ** 2 / 3) * torch.exp(-s5)


class GP:
    def __init__(self, x, y, steps=300):
        self.x = x
        self.mu, self.sd = y.mean(), y.std().clamp(min=1e-9)
        yn = (y - self.mu) / self.sd
        d = x.shape[1]
        p = {k: torch.zeros(n, dtype=x.dtype, device=x.device, requires_grad=True) for k, n in (("ls", d), ("var", 1), ("noise", 1))}
        with torch.no_grad():
            p["ls"].fill_(math.log(0.5)); p["noise"].fill_(math.log(1e-2))
        opt = torch.optim.Adam(p.values(), lr=0.05)
        eye = torch.eye(len(x), dtype=x.dtype, device=x.device)
        for _ in range(steps):
            opt.zero_grad()
            k = matern52(x, x, p["ls"].exp(), p["var"].exp()) + (p["noise"].exp() + 1e-6) * eye
            L = torch.linalg.cholesky(k)
            a = torch.cholesky_solve(yn[:, None], L)
            nll = 0.5 * (yn[:, None] * a).sum() + torch.log(torch.diagonal(L)).sum()
            nll.backward()
            opt.step()
        with torch.no_grad():
            self.ls, self.var, self.noise = p["ls"].exp(), p["var"].exp(), p["noise"].exp()
            k = matern52(x, x, self.ls, self.var) + (self.noise + 1e-6) * eye
            self.alpha = torch.cholesky_solve(yn[:, None], torch.linalg.cholesky(k))

    @torch.no_grad()
    def mean(self, xs):
        out = []
        for i in range(0, len(xs), 8192):
            out.append(matern52(xs[i:i + 8192], self.x, self.ls, self.var) @ self.alpha)
        return (torch.cat(out)[:, 0] * self.sd + self.mu)


def main() -> None:
    from SALib.analyze import sobol as sobol_analyze
    from SALib.sample import sobol as sobol_sample

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path.home() / "sobol_la")
    ap.add_argument("--out", type=Path, default=Path("figures"))
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--hours", type=int, nargs="*", default=list(HOURS),
                    help="hours to fit; with --all-hours every hour where the quantity varies")
    ap.add_argument("--all-hours", action="store_true")
    ap.add_argument("--quantities", nargs="*", default=list(QUANTITIES))
    ap.add_argument("--tag", default="", help="suffix for the output file")
    args = ap.parse_args()
    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p = json.loads((args.runs / "problem.json").read_text())
    names, bounds = p["names"], np.array(p["bounds"])
    X = np.load(args.runs / "X.npy")
    recs = [json.loads((args.runs / "runs" / f"{i:05d}.json").read_text()) for i in range(len(X))]
    xs = torch.tensor((X - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0]), dtype=torch.float64, device=dev)
    unit = {"num_vars": p["num_vars"], "names": names, "bounds": [[0.0, 1.0]] * p["num_vars"]}
    design = torch.tensor(sobol_sample.sample(unit, args.n, calc_second_order=False, seed=1), dtype=torch.float64, device=dev)
    direct = json.loads((args.out / "la_sobol.json").read_text())["quantities"]
    folds = np.array_split(np.random.default_rng(0).permutation(len(X)), 5)
    res = {"n_design": args.n, "runs": len(X), "kernel": "Matern 5/2, ARD", "results": {}}
    for q in args.quantities:
        for h in (range(24) if args.all_hours else args.hours):
            y = torch.tensor([r[q][h] for r in recs], dtype=torch.float64, device=dev)
            if float(y.std()) < 1e-6:
                continue
            pred = torch.zeros_like(y)
            st_folds, s1_folds = [], []
            for f in folds:
                tr = torch.as_tensor(np.setdiff1d(np.arange(len(X)), f), device=dev)
                f = torch.as_tensor(f, device=dev)
                g = GP(xs[tr], y[tr])
                pred[f] = g.mean(xs[f])
                si = sobol_analyze.analyze(unit, g.mean(design).cpu().numpy(), calc_second_order=False, print_to_console=False)
                st_folds.append(si["ST"]); s1_folds.append(si["S1"])
            r2 = float(1 - ((pred - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())
            st, s1 = np.array(st_folds), np.array(s1_folds)
            dst = np.array(direct[q]["ST"][h])
            rec = {"cv_r2": r2, "ST_mean": st.mean(0).tolist(), "ST_min": st.min(0).tolist(), "ST_max": st.max(0).tolist(),
                   "S1_mean": s1.mean(0).tolist(), "S1_sum": float(s1.mean(0).sum()), "direct_ST": dst.tolist(),
                   "direct_ST_lo": direct[q]["st_lo"][h], "direct_ST_hi": direct[q]["st_hi"][h]}
            res["results"][f"{q}|{h}"] = rec
            top = np.argsort(-st.mean(0))[:3]
            print(f"[gp] {q:24s} {h:2d} h: CV R2 {r2:.3f}, sum S1 {rec['S1_sum']:.2f} | " +
                  ", ".join(f"{names[j]} ST {st.mean(0)[j]:.2f} [{st.min(0)[j]:.2f}-{st.max(0)[j]:.2f}] (direct {dst[j]:.2f})"
                            for j in top), flush=True)
    out = args.out / f"la_sobol_surrogate{args.tag}.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"[gp] wrote {out}")


if __name__ == "__main__":
    main()
