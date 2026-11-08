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
    python scripts/sobol_surrogate.py --runs ~/sobol_la --out figures --all-hours --quantities area_utci_above_32_pct \
        area_utci_above_38_pct --tag _allhours
    python scripts/sobol_surrogate.py --check-figure figures
"""

from __future__ import annotations

import argparse
import json
import math
import sys
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


def check_figure(out: Path) -> None:
    """Can the surrogate be trusted? Its indices against the direct ones where those are resolved, and its held-out R^2."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import sys as _s

    _s.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from solweig_sens import plotstyle

    four = json.loads((out / "la_sobol_surrogate.json").read_text())["results"]
    allh = json.loads((out / "la_sobol_surrogate_allhours.json").read_text())["results"]
    plotstyle.use()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    big = out / "la_sobol_n512_new.json"
    if big.exists():
        # out of sample: against the direct estimators on the 3840 runs of the N = 512 design that the surrogate was
        # not fitted to (analyse_la_sobol.py --n 512 --skip-base 128), every hour where the quantity varies
        D = json.loads(big.read_text())["quantities"]
        sets = [("mean outdoor UTCI", "utci_mean", "o", four, "#4c78a8"),
                ("area with UTCI > 32 °C", "area_utci_above_32_pct", "^", allh, "#c4501b"),
                ("area with UTCI > 38 °C", "area_utci_above_38_pct", "s", allh, "#0e6a78")]
        for lab, q, mk, src, col in sets:
            xs, ys, lo, hi = [], [], [], []
            for key, r in src.items():
                qq, h = key.split("|")
                h = int(h)
                if qq != q or r.get("cv_r2", 1.0) < 0.9 or (q != "utci_mean" and D[q]["spread_sd"][h] < 1.0):
                    continue
                xs += D[q]["ST"][h]; ys += r["ST_mean"]; lo += D[q]["st_lo"][h]; hi += D[q]["st_hi"][h]
            xs, ys, lo, hi = (np.array(v, dtype=float) for v in (xs, ys, lo, hi))
            ok = np.isfinite(xs) & np.isfinite(ys)
            xs, ys, lo, hi = xs[ok], ys[ok], lo[ok], hi[ok]
            ax.errorbar(xs, ys, xerr=[np.clip(xs - lo, 0, None), np.clip(hi - xs, 0, None)], fmt=mk, ms=5, lw=0.8,
                        capsize=0, alpha=0.8, color=col, label=lab)
        xlabel = "total Sobol index, direct estimator, 3840 other runs"
    else:
        # resolved by the direct estimators on the same runs: mean outdoor UTCI, area above 38 C at 13 and 16 h
        sets = [("mean outdoor UTCI (9, 13, 16, 19 h)", "utci_mean", (9, 13, 16, 19), "o", four, "#4c78a8"),
                ("area with UTCI > 38 °C (13, 16 h)", "area_utci_above_38_pct", (13, 16), "s", allh, "#0e6a78")]
        for lab, q, hours, mk, src, col in sets:
            xs, ys, lo, hi = [], [], [], []
            for h in hours:
                r = src.get(f"{q}|{h}")
                if not r:
                    continue
                xs += r["direct_ST"]; ys += r["ST_mean"]; lo += r["direct_ST_lo"]; hi += r["direct_ST_hi"]
            xs, ys, lo, hi = map(np.array, (xs, ys, lo, hi))
            ax.errorbar(xs, ys, xerr=[xs - lo, hi - xs], fmt=mk, ms=5, lw=0.8, capsize=0, alpha=0.85, color=col, label=lab)
        xlabel = "total Sobol index, direct estimator"
    ax.plot([0, 1], [0, 1], color="0.4", lw=0.8, ls=":")
    ax.set_xlim(-0.05, 1.1)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("total Sobol index, surrogate, 1280 runs")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    ax = axes[1]
    for q, lab, c in (("area_utci_above_32_pct", "area with UTCI > 32 °C", "#c4501b"), ("area_utci_above_38_pct", "area with UTCI > 38 °C", "#0e6a78")):
        hs = sorted(int(k.split("|")[1]) for k in allh if k.startswith(q + "|"))
        ax.plot(hs, [allh[f"{q}|{h}"]["cv_r2"] for h in hs], "o-", color=c, label=lab)
    ax.axhline(0.9, color="0.4", lw=0.8, ls=":")
    ax.set_xlim(0, 23)
    ax.set_ylim(0.8, 1.005)
    ax.set_xlabel("hour (PDT)")
    ax.set_ylabel("held-out $R^2$")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)
    for a_, letter in zip(axes, "ab"):
        a_.text(-0.12, 1.02, letter, transform=a_.transAxes, fontsize=13, fontweight="bold", va="bottom")
    fig.tight_layout()  # caption: docs/FIGURES.md
    fig.savefig(out / "la_sobol_surrogate_check.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[gp] wrote {out / 'la_sobol_surrogate_check.png'}")


if __name__ == "__main__":
    if "--check-figure" in sys.argv:
        check_figure(Path(sys.argv[sys.argv.index("--check-figure") + 1]))
    else:
        main()
