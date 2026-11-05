"""Street-scale skill against air-temperature forecast error (run_la_forecast_error.py).

Perfect-model test: within each error size sigma, every member in turn is "the truth" and the
others are forecasts. The event scored is strong heat stress for at least four hours (UTCI above
32 C for 4 h or more: a duration-based heat-stress criterion), per pixel.

Fractions skill score (Roberts & Lean 2008, Mon. Wea. Rev. 136, 78-97): the event fraction in a
square neighbourhood of n x n pixels is compared between forecast and truth,
FSS = 1 - mean((P_f - P_o)^2) / (mean(P_f^2) + mean(P_o^2)), from 0 (no skill) to 1. A forecast is
"useful" at a scale where FSS >= 0.5 + f/2, f the observed event frequency (their uniform-forecast
target). Two forecasts per truth: one other member (a deterministic forecast with the same error
statistics) and the probability from the other members (the ensemble). Neighbourhoods from 2 m
(one pixel) to 1 km. Output: FSS against scale for every sigma, and the smallest useful scale
against sigma: a street-scale version of the skill front, with sigma standing for lead time.

Also, per sigma: the spread of the share of outdoor area above 32 and 38 C at 15:00, and of mean
outdoor UTCI, across members.

Second event: each map's own hottest 10% of outdoor pixels at 15:00 (a percentile threshold, Roberts & Lean 2008),
which removes the level of the map and scores only where the heat is. UTCI above 38 C for 4 h or more is scored
too (in the JSON), but on these days it is too rare to be informative.

Usage:
    python scripts/analyse_la_forecast_error.py --runs ~/forecast_error_la --out figures
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
from scipy.ndimage import uniform_filter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solweig_sens import plotstyle  # noqa: E402

SCALES_PX = (1, 5, 13, 25, 51, 125, 251, 501)  # odd sizes: 2 m to 1 km at 2 m
PIXEL_M = 2.0
EVENT_HOURS = 4
TOP_SHARE = 0.10  # percentile event: each member's hottest 10% of outdoor pixels at 15 h


def fractions(event: np.ndarray, valid: np.ndarray, n: int) -> np.ndarray:
    """Event fraction over valid pixels in an n x n neighbourhood."""
    if n == 1:
        return np.where(valid, event, 0.0)
    num = uniform_filter(np.where(valid, event, 0.0), size=n, mode="constant")
    den = uniform_filter(valid.astype(float), size=n, mode="constant")
    return np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)


def fss(pf: np.ndarray, po: np.ndarray, valid: np.ndarray) -> float:
    a, b = pf[valid], po[valid]
    den = np.mean(a ** 2) + np.mean(b ** 2)
    return float(1 - np.mean((a - b) ** 2) / den) if den > 0 else np.nan


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path.home() / "forecast_error_la")
    ap.add_argument("--out", type=Path, default=Path("figures"))
    ap.add_argument("--site", default="Pacoima", help="tile name for the caption")
    args = ap.parse_args()
    design = json.loads((args.runs / "design.json").read_text())
    sigmas = design["sigmas"]
    report = {"design": {k: design[k] for k in ("sigmas", "members", "rho", "seed", "date")},
              "scales_m": [n * PIXEL_M for n in SCALES_PX], "sigma": {}}
    base = np.load(args.runs / "maps" / "base.npz")
    valid = base["hours_above_32"] != 255
    def hours_event(key):
        return lambda m: m[key] >= EVENT_HOURS

    def top_event(m):  # the hottest TOP_SHARE of each map's own outdoor pixels at 15 h: the pattern, not the level
        u = m["utci_15h"].astype(np.float32)
        return u >= np.nanpercentile(u[valid], 100 * (1 - TOP_SHARE))

    def event_skill(ids, event):
        """FSS against scale for a binary event defined on each member's maps."""
        ev = np.stack([event(np.load(args.runs / "maps" / f"{i}.npz")) for i in ids]).astype(float)
        det, ens, freq = {n: [] for n in SCALES_PX}, {n: [] for n in SCALES_PX}, []
        for t in range(len(ids)):
            truth = ev[t]
            freq.append(truth[valid].mean())
            other = [k for k in range(len(ids)) if k != t]
            prob = ev[other].mean(axis=0)
            fc = ev[other[t % len(other)]]  # one other member, a different one for each truth
            for n in SCALES_PX:
                po = fractions(truth, valid, n)
                det[n].append(fss(fractions(fc, valid, n), po, valid))
                ens[n].append(fss(fractions(prob, valid, n), po, valid))
        f = float(np.mean(freq))
        target = 0.5 + f / 2
        rec = {"event_frequency": f, "useful_target": target,
               "fss_member": [float(np.nanmean(det[n])) for n in SCALES_PX],
               "fss_member_p10_p90": [[float(np.nanpercentile(det[n], q)) for q in (10, 90)] for n in SCALES_PX],
               "fss_ensemble": [float(np.nanmean(ens[n])) for n in SCALES_PX],
               "fss_ensemble_p10_p90": [[float(np.nanpercentile(ens[n], q)) for q in (10, 90)] for n in SCALES_PX]}
        for kind in ("member", "ensemble"):
            ok = [n * PIXEL_M for n, v in zip(SCALES_PX, rec[f"fss_{kind}"]) if v >= target]
            rec[f"useful_scale_m_{kind}"] = ok[0] if ok else None
        return rec

    for s in sigmas:
        ids = [r["id"] for r in design["runs"] if r["sigma"] == s]
        recs = [json.loads((args.runs / "runs" / f"{i}.json").read_text()) for i in ids]
        rec = event_skill(ids, hours_event("hours_above_32"))
        rec["event_38"] = event_skill(ids, hours_event("hours_above_38"))  # too rare on these days to score; kept
        rec["event_top10"] = event_skill(ids, top_event)
        f = rec["event_frequency"]
        for q in ("area_above_32_pct", "area_above_38_pct", "utci_mean"):
            vals = np.array([r[q][15] for r in recs])
            rec[f"{q}_15h_mean_sd"] = [float(vals.mean()), float(vals.std())]
        # level against pattern: each member's UTCI change at 15 h from the unperturbed run, split into its
        # domain mean (the level) and its spread across pixels (the pattern), plus the rank correlation
        from scipy.stats import spearmanr
        u0 = base["utci_15h"].astype(np.float32)[valid]
        shift, spread, rho = [], [], []
        for i in ids:
            u = np.load(args.runs / "maps" / f"{i}.npz")["utci_15h"].astype(np.float32)[valid]
            d = u - u0
            shift.append(abs(float(d.mean())))
            spread.append(float(d.std()))
            rho.append(float(spearmanr(u0[::25], u[::25]).correlation))
        rec["utci_15h_level_shift_abs_mean"] = float(np.mean(shift))
        rec["utci_15h_pattern_spread_mean"] = float(np.mean(spread))
        rec["utci_15h_rank_correlation_min"] = float(np.min(rho))
        report["sigma"][str(s)] = rec
        print(f"sigma {s:3.1f} K: event frequency {f:.2f}, useful scale member {rec['useful_scale_m_member']} m, "
              f"ensemble {rec['useful_scale_m_ensemble']} m; area > 32 C at 15 h "
              f"{rec['area_above_32_pct_15h_mean_sd'][0]:.1f} +- {rec['area_above_32_pct_15h_mean_sd'][1]:.1f} %; "
              f"UTCI 15 h change: level {rec['utci_15h_level_shift_abs_mean']:.2f} C, across pixels "
              f"{rec['utci_15h_pattern_spread_mean']:.3f} C, rank correlation >= {rec['utci_15h_rank_correlation_min']:.4f}")
    (args.out / "la_forecast_error.json").write_text(json.dumps(report, indent=2))

    plotstyle.use()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.6), gridspec_kw={"width_ratios": [1.4, 1]})
    cmap = plt.get_cmap("viridis")
    scales = np.array(report["scales_m"])
    rows = ((lambda r: r, "UTCI above 32 C for 4 h or more", "32 C, 4 h"),
            (lambda r: r["event_top10"], f"Hottest {100 * TOP_SHARE:.0f}% of outdoor pixels at 15:00", "hottest 10%"))
    for row, (get, title, short) in enumerate(rows):
        ax, ax2 = axes[row]
        ev = [get(report["sigma"][str(s)]) for s in sigmas]
        for k, (s, rec) in enumerate(zip(sigmas, ev)):
            c = cmap(k / max(len(sigmas) - 1, 1))
            lo, hi = np.array(rec["fss_member_p10_p90"]).T
            ax.fill_between(scales, lo, hi, color=c, alpha=0.12, lw=0)
            ax.plot(scales, rec["fss_member"], color=c, lw=1.4, ls="--")
            ax.plot(scales, rec["fss_ensemble"], color=c, lw=2, label=f"error {s:g} K")
        ax.axhline(np.mean([r["useful_target"] for r in ev]), color="0.4", lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel("Neighbourhood size (m)")
        ax.set_ylabel("Fractions skill score")
        fr = [r["event_frequency"] for r in ev]
        ax.set_title(f"{title} (event at {100 * min(fr):.0f}-{100 * max(fr):.0f}% of pixels)", loc="left", fontsize=10)
        ax.grid(alpha=0.3)
        for kind, ls, lab in (("ensemble", "-", "ensemble probability"), ("member", "--", "single forecast")):
            ys = [r[f"useful_scale_m_{kind}"] for r in ev]
            ax2.plot([s for s, y in zip(sigmas, ys) if y], [y for y in ys if y], "o" + ls, color="k", label=lab)
        ax2.set_yscale("log")
        ax2.set_yticks(scales, [f"{v:g}" for v in scales])  # the scales tested: the curve can only take these values
        ax2.minorticks_off()
        ax2.set_ylim(scales[0] * 0.8, scales[-1] * 1.25)
        ax2.set_xlabel("Air-temperature forecast error, standard deviation (K)")
        ax2.set_ylabel("Smallest useful scale tested (m)")
        ax2.set_title(f"Skill front, {short}", loc="left", fontsize=10)
        ax2.grid(alpha=0.3)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower right")
    axes[0, 1].legend(frameon=False, fontsize=8)
    structure = design.get("structure", "fitted")
    struct_txt = ("ground heating fitted to measurements, heat kept after sunset" if structure == "fitted"
                  else "SOLWEIG's own ground model")
    fig.text(0.5, -0.08, f"{args.site}, 2 x 2 km at 2 m, {design['date']}, SOLWEIG-GPU ({struct_txt}). "
             "Top: a fixed threshold, met by most pixels, so it measures the level of the map. Bottom: each map's own hottest "
             "10% (Roberts & Lean 2008's percentile threshold), which removes the level and tests whether the hot spots are in the right "
             "streets. (UTCI above 38 C for 4 h is too rare on these days to score; it is in the JSON.) "
             "Right: the smallest of the tested scales (2, 10, 26 ... m) where FSS reaches the useful line; 2 m means useful "
             "at the finest scale tested; a missing point, at no scale tested. "
             f"{design['members']} members per error size, errors AR(1) in time (lag-1 correlation {design['rho']}); "
             "each member in turn taken as the truth. Lines: mean over truths; band: 10th to 90th percentile "
             "(single forecast). Dotted: useful skill (FSS = 0.5 + f/2, Roberts & Lean 2008, mean over error sizes). "
             "Lead time for each error size: from HRRR verification, not shown here.", ha="center", fontsize=8,
             color="0.35", wrap=True)
    fig.savefig(args.out / "la_forecast_error_fss.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out / 'la_forecast_error.json'} and la_forecast_error_fss.png")


if __name__ == "__main__":
    main()
