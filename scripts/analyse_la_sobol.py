"""Sobol indices of the Pacoima pilot (run_la_sobol.py): which inputs drive heat stress, and where?

First-order index S1 (the share of the output variance an input explains on its own) and total
index ST (its share including all its interactions), for every hour, for the domain-level
quantities: mean outdoor UTCI, the share of outdoor area above 32, 38 and 46 C UTCI, and the median
Tmrt on open sunlit ground. SALib's estimators (Saltelli 2010 for S1, Jansen for ST) with 95%
bootstrap intervals. ST - S1 is the input's share in interactions; if S1 sums to about 1, the
output is additive in the inputs.

Per pixel, at 13, 16 and 21 h, the same estimators vectorised over the UTCI fields: maps of ST for
each input (which input dominates where), and the sum of S1 (how additive each pixel is).

Usage:
    python scripts/analyse_la_sobol.py --runs ~/sobol_la --out figures
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

from solweig_sens import plotstyle  # noqa: E402

LABELS = {"ta_add": "air temperature", "rh_add": "relative humidity", "wind_scale": "wind speed",
          "kdown_scale": "global radiation", "paved_emissivity": "paved emissivity", "albedo_b": "wall albedo",
          "ewall": "wall emissivity", "regular_albedo": "paved albedo"}
TOP = 3  # inputs drawn per panel in the daily figure
PANEL = {"utci_mean": "mean outdoor UTCI", "area_utci_above_32_pct": "outdoor area with UTCI > 32 °C",
         "area_utci_above_38_pct": "outdoor area with UTCI > 38 °C"}
QUANTITIES = {"utci_mean": "mean outdoor UTCI", "area_utci_above_32_pct": "area above 32 C UTCI",
              "area_utci_above_38_pct": "area above 38 C UTCI", "area_utci_above_46_pct": "area above 46 C UTCI",
              "tmrt_open_sunlit_median": "Tmrt, open sunlit ground"}


def blocks(Y: np.ndarray, d: int):
    """Saltelli ordering without second order: A, AB_1..AB_d, B per base sample (first axis)."""
    step = d + 2
    return Y[0::step], np.stack([Y[j + 1::step] for j in range(d)]), Y[step - 1::step]


def indices(Y: np.ndarray, d: int, boot: int = 0, seed: int = 0):
    """S1, ST (each (d, ...)) by Saltelli 2010 and Jansen; with boot > 0 also 95% intervals."""
    A, AB, B = blocks(Y, d)

    def est(A, AB, B):
        var = np.var(np.concatenate([A, B]), axis=0)
        var = np.where(var > 0, var, np.nan)
        s1 = np.mean(B[None] * (AB - A[None]), axis=1) / var
        st = 0.5 * np.mean((A[None] - AB) ** 2, axis=1) / var
        return s1, st

    s1, st = est(A, AB, B)
    if not boot:
        return s1, st, None
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    bs = [est(A[i], AB[:, i], B[i]) for i in (rng.integers(0, n, n) for _ in range(boot))]
    lo = [np.percentile([b[k] for b in bs], 2.5, axis=0) for k in (0, 1)]
    hi = [np.percentile([b[k] for b in bs], 97.5, axis=0) for k in (0, 1)]
    return s1, st, {"s1_lo": lo[0], "s1_hi": hi[0], "st_lo": lo[1], "st_hi": hi[1]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path.home() / "sobol_la")
    ap.add_argument("--out", type=Path, default=Path("figures"))
    ap.add_argument("--n", type=int, default=None,
                    help="use only the first n base samples (a power of 2): a complete smaller design")
    ap.add_argument("--tag", default="", help="suffix for the output files")
    ap.add_argument("--json-only", action="store_true", help="indices and printout only; no per-pixel maps or figures "
                    "(the maps load every field: about 3 GB per 1000 runs)")
    ap.add_argument("--skip-base", type=int, default=0, help="leave out the first k base samples (k (d + 2) runs), e.g. the "
                    "runs a surrogate was fitted to, for an out-of-sample check")
    ap.add_argument("--no-maps", action="store_true", help="the daily figure but not the per-pixel maps (which load every field)")
    ap.add_argument("--direct", action="store_true", help="daily figure from the direct estimators in every panel (enough "
                    "runs to resolve the threshold areas, e.g. N = 512), instead of the surrogate for the areas")
    ap.add_argument("--day-out", type=Path, default=None, help="file for the daily figure (default <out>/la_sobol_day<tag>.png)")
    args = ap.parse_args()
    p = json.loads((args.runs / "problem.json").read_text())
    names, d = p["names"], p["num_vars"]
    if args.n:
        p["n_base"] = args.n
    n_runs = p["n_base"] * (d + 2)
    recs = [json.loads((args.runs / "runs" / f"{i:05d}.json").read_text()) for i in range(n_runs)
            if (args.runs / "runs" / f"{i:05d}.json").exists()]
    if len(recs) != n_runs:
        raise SystemExit(f"{len(recs)} of {n_runs} runs on disk; the estimators need the complete design")

    if args.skip_base:  # Saltelli blocks are whole base samples, so dropping the first k keeps a valid design
        recs = recs[args.skip_base * (d + 2):]
        n_runs = len(recs)
        p["n_base"] -= args.skip_base
    report = {"design": {k: p[k] for k in ("n_base", "size_px", "date", "structure")} | {"runs": n_runs, "skip_base": args.skip_base}, "names": names,
              "quantities": {}}
    for q in QUANTITIES:
        Y = np.array([r[q] for r in recs], dtype=np.float64)  # (runs, hours)
        s1, st, ci = indices(Y, d, boot=1000)
        report["quantities"][q] = {
            "spread_sd": np.std(Y, axis=0).tolist(),
            "S1": s1.T.tolist(), "ST": st.T.tolist(),
            **{k: v.T.tolist() for k, v in ci.items()}}
    (args.out / f"la_sobol{args.tag}.json").write_text(json.dumps(report))

    print("hour  quantity                    sd    sum S1 | ST per input")
    for q in ("utci_mean", "area_utci_above_32_pct", "area_utci_above_38_pct"):
        r = report["quantities"][q]
        for h in (9, 13, 16, 19, 21):
            st = np.array(r["ST"][h])
            top = np.argsort(-np.nan_to_num(st))[:3]
            print(f"{h:4d}  {QUANTITIES[q]:26s} {r['spread_sd'][h]:5.2f}  {np.nansum(r['S1'][h]):5.2f} | "
                  + ", ".join(f"{LABELS[names[j]]} {st[j]:.2f}" for j in top))

    if args.json_only:
        return
    plotstyle.use()
    # 1. indices through the day for the three heat-stress quantities. The threshold areas are not resolved by the
    # direct estimators at this N; where a Gaussian-process surrogate of the same runs exists (sobol_surrogate.py,
    # --all-hours), its indices are drawn (band: spread over its 5 cross-validation fits), the direct ones faintly.
    sur_path = args.out / "la_sobol_surrogate_allhours.json"
    sur = json.loads(sur_path.read_text())["results"] if sur_path.exists() else {}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    hrs = np.arange(len(recs[0]["utci_mean"]))
    for ax, q in zip(axes, ("utci_mean", "area_utci_above_32_pct", "area_utci_above_38_pct")):
        r = report["quantities"][q]
        st, lo, hi = (np.array(r[k], dtype=float) for k in ("ST", "st_lo", "st_hi"))
        s1sum = np.array([np.nan if np.all(np.isnan(np.array(row, dtype=float))) else np.nansum(row) for row in r["S1"]])
        use_sur = q != "utci_mean" and not args.direct and any(k.startswith(q + "|") for k in sur)
        if q != "utci_mean" and not use_sur:  # no index where the area hardly varies between runs (sd < 1 point)
            flat = np.array(r["spread_sd"]) < 1.0
            st[flat], lo[flat], hi[flat], s1sum[flat] = np.nan, np.nan, np.nan, np.nan
        if use_sur:  # surrogate indices, only where its held-out R^2 is at least 0.9; NaN (a gap) elsewhere
            sst, slo, shi = (np.full((len(hrs), d), np.nan) for _ in range(3))
            ss1 = np.full(len(hrs), np.nan)
            for h in hrs:
                rec = sur.get(f"{q}|{h}")
                if rec and rec["cv_r2"] >= 0.9:
                    sst[h], slo[h], shi[h], ss1[h] = rec["ST_mean"], rec["ST_min"], rec["ST_max"], rec["S1_sum"]
            order = np.argsort(-np.nan_to_num(np.nanmean(sst, axis=0)))  # by the average over the day
            for j in order[:TOP]:  # the direct estimates are compared with these in la_sobol_surrogate_check.png
                c = plotstyle.INPUT_COLORS.get(LABELS[names[j]], "0.5")
                ax.fill_between(hrs, slo[:, j], shi[:, j], color=c, alpha=0.25, lw=0)
                ax.plot(hrs, sst[:, j], "o-", ms=3, color=c, lw=1.8, label=LABELS[names[j]])
            ax.plot(hrs, ss1, color="0.45", lw=1.2, ls="--", label="sum of first-order indices")
        else:
            order = np.argsort(-np.nanmean(st, axis=0))
            for j in order[:TOP]:
                c = plotstyle.INPUT_COLORS.get(LABELS[names[j]], "0.5")
                ax.fill_between(hrs, lo[:, j], hi[:, j], color=c, alpha=0.18, lw=0)
                ax.plot(hrs, st[:, j], "o-", ms=3, color=c, lw=1.6, label=LABELS[names[j]])
            ax.plot(hrs, s1sum, color="0.45", lw=1.2, ls="--", label="sum of first-order indices")
        ax.set_title(PANEL[q], loc="left", fontsize=10)
        ax.set_xlabel("hour (PDT)")
        ax.set_xlim(0, 23)
        ax.set_ylim(-0.05, 1.15)
        ax.grid(alpha=0.3)
    for ax, letter in zip(axes, "abc"):
        ax.text(-0.08, 1.06, letter, transform=ax.transAxes, fontsize=13, fontweight="bold", va="bottom")
    axes[0].set_ylabel("total Sobol index")
    handles, labels = [], []
    for ax in axes:  # one legend for the inputs that appear in any panel
        for hnd, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels:
                handles.append(hnd)
                labels.append(lab)
    k = labels.index("sum of first-order indices")  # inputs first, the sum last
    handles.append(handles.pop(k))
    labels.append(labels.pop(k))
    fig.legend(handles, labels, frameon=False, loc="upper left", bbox_to_anchor=(0.995, 0.92), fontsize=9)
    # caption: docs/FIGURES.md (no title or caption in the image)
    fig.savefig(args.day_out or args.out / f"la_sobol_day{args.tag}.png", bbox_inches="tight")
    plt.close(fig)
    if args.no_maps:
        print(f"wrote {args.out / f'la_sobol{args.tag}.json'} and {args.day_out or f'la_sobol_day{args.tag}.png'}")
        return

    # per-pixel maps
    if args.skip_base:
        raise SystemExit("per-pixel maps with --skip-base are not supported; use --no-maps or --json-only")
    fields = np.stack([np.load(args.runs / "fields" / f"{i:05d}.npy").astype(np.float32) for i in range(n_runs)])
    hours = p["map_hours"]
    s1m, stm, _ = indices(fields, d)  # (d, hours, rows, cols)

    # 2. maps at 16 h: total index of the leading inputs, and how additive each pixel is
    k = hours.index(16) if 16 in hours else 0
    lead = np.argsort(-np.nanmean(stm[:, k], axis=(1, 2)))[:3]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    for ax, j in zip(axes, lead):
        im = ax.imshow(stm[j, k], cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"{plotstyle.input_label(LABELS[names[j]])}, total index")
        plotstyle.map_axes(ax)
    fig.colorbar(im, ax=axes[:3], shrink=0.8, label="share of the pixel's UTCI variance")
    im = axes[3].imshow(np.nansum(s1m[:, k], axis=0), cmap="magma", vmin=0.5, vmax=1.05)
    axes[3].set_title("Sum of first-order indices")
    plotstyle.map_axes(axes[3])
    fig.colorbar(im, ax=axes[3], shrink=0.8, label="1 = additive, lower = interactions")
    plotstyle.scalebar(axes[0], 2.0)
    fig.text(0.5, -0.03, f"UTCI at {hours[k]}:00. Roofs blank. Same runs as the daily curves.", ha="center",
             fontsize=8, color="0.35")
    fig.savefig(args.out / f"la_sobol_maps_16h{args.tag}.png")
    plt.close(fig)
    print(f"wrote {args.out / f'la_sobol{args.tag}.json'}, la_sobol_day{args.tag}.png, la_sobol_maps_16h{args.tag}.png")


if __name__ == "__main__":
    main()
