"""Calibrate the inputs or the decision? Hourly air temperature at held-out stations, summer 2022, Los Angeles.

A model gives hourly air temperature where there is no sensor; people act on numbers derived from a whole day of it
(the day's maximum, hours above 32 C, an event of 4 or more consecutive hours above 32 C). Three ways to get
calibrated uncertainty for those numbers are compared on stations held out by 5 km blocks:

    A  input level, hours independent: each hour's error drawn from that hour's calibration errors
    B  input level, joint: whole-day error trajectories (24 hours together) drawn from calibration station-days
    C  decision level: split conformal on each decision quantity directly (per network too: C_net), and isotonic
       regression of the event on the predicted daily maximum

Design, hypotheses and limits, written before running: docs/internal/decision_calibration_design_2026-11-12.md.
Same stations, quality control and predictor as street_ta_structure.py (HRRR 1 h forecast + mean bias + each
station's offset predicted from its surroundings, per local hour, no live reports).

Protocols: P1 space only (June-September everywhere); P2 space and time (fit and calibrate on June-August, test on
September, which holds the 31 Aug-9 Sep 2022 heatwave).

Outputs: figures/street_ta/decision_calibration.json and decision_calibration.png.

Usage (env with scikit-learn, pyproj, matplotlib; needs street_ta_structure.py's hourly 2022 download):
    python scripts/decision_calibration.py [--draws 400] [--boot 500]
"""

from __future__ import annotations

import sys as _sys

if __name__ == "__main__" and any(a in ("-h", "--help") for a in _sys.argv[1:]):
    print(__doc__)
    raise SystemExit(0)

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys_path = Path(__file__).resolve().parent
_sys.path.insert(0, str(sys_path))
import street_ta_structure as sta  # noqa: E402

THRESH = (32.0, 35.0)
RUN_H = 4
METHODS = ("A", "B", "C", "C_net")


def event(T, thr):
    """True where a day (last axis = 24 hours) has RUN_H or more consecutive hours above thr."""
    above = T > thr
    run = np.zeros(above.shape[:-1], int)
    best = np.zeros(above.shape[:-1], int)
    for h in range(above.shape[-1]):
        run = np.where(above[..., h], run + 1, 0)
        best = np.maximum(best, run)
    return best >= RUN_H


def conf_q(scores, alpha=0.1):
    s = np.sort(np.asarray(scores))
    n = len(s)
    if n == 0:
        return np.nan
    return float(s[min(n - 1, int(np.ceil((1 - alpha) * (n + 1))) - 1)])


def build():
    """Station-day cubes of observed and HRRR temperature, summer 2022, local (PDT) days."""
    D = sta.load()
    names, st = D["names"], D["st"]
    sel = np.array([t.startswith("2022-") for t in D["times"]])
    times = [t for t, s in zip(D["times"], sel) if s]
    R, TH = D["R"][sel], D["TH"][sel]
    utc_h = np.array([int(t[-2:]) for t in times])
    Rc = np.full_like(R, np.nan)
    qc = {}
    for h in range(24):  # quality control per UTC hour (offsets differ by hour)
        m = utc_h == h
        if m.sum() == 0:
            continue
        Rc[m], q = sta.clean(R[m], D["keep"])
        qc[h] = q
    OBS = TH + Rc
    loc = [datetime.strptime(t, "%Y-%m-%dT%H") - timedelta(hours=7) for t in times]
    days = sorted({x.date() for x in loc})
    di = {d: k for k, d in enumerate(days)}
    S, ND = len(names), len(days)
    obs = np.full((S, ND, 24), np.nan, np.float32)
    hr = np.full((S, ND, 24), np.nan, np.float32)
    for k, x in enumerate(loc):
        obs[:, di[x.date()], x.hour] = OBS[k]
        hr[:, di[x.date()], x.hour] = TH[k]
    keep_day = np.array([datetime(2022, 6, 1).date() <= d <= datetime(2022, 9, 30).date() for d in days])
    obs, hr, days = obs[:, keep_day], hr[:, keep_day], [d for d, k in zip(days, keep_day) if k]
    return {"names": names, "st": st, "obs": obs, "hr": hr, "days": days, "z_s": D["z_s"], "z_h": D["z_h"],
            "qc_reports_used": int(np.isfinite(Rc).sum()), "hours_with_data": int(len(set(times)))}


def offsets_model(F_fit, M_fit):
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    ok = np.isfinite(M_fit)
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))).fit(F_fit[ok], M_fit[ok])


def predict(C, fit, others, fit_days):
    """Predicted hourly temperature for stations `others`: HRRR + mean bias + offset model, per local hour,
    fitted on stations `fit` and days `fit_days` only."""
    obs, hr, F = C["obs"], C["hr"], C["F"]
    out = np.full((len(others),) + obs.shape[1:], np.nan, np.float32)
    for h in range(24):
        r = (obs[fit][:, fit_days, h] - hr[fit][:, fit_days, h]).T  # days x stations
        c, m, _ = sta.decompose(r)
        mdl = offsets_model(F[fit], m)
        out[:, :, h] = hr[others][:, :, h] + float(np.nanmean(c)) + mdl.predict(F[others])[:, None]
    return out


def decisions(T):
    return {"D1": np.max(T, axis=-1), "D2": np.sum(T > THRESH[0], axis=-1).astype(float),
            "D2_35": np.sum(T > THRESH[1], axis=-1).astype(float), "E32": event(T, THRESH[0]), "E35": event(T, THRESH[1])}


def run_protocol(C, fit_days, test_days, args, rng):
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import GroupKFold, GroupShuffleSplit

    obs, prov, blocks = C["obs"], C["prov"], C["blocks"]
    usable = C["usable"]
    rows = []  # one dict of arrays per fold, concatenated after
    hourly = {"covered": 0, "n": 0, "covered_hot": 0, "n_hot": 0}
    for tr_i, te_i in GroupKFold(n_splits=5).split(usable, groups=blocks[usable]):
        tr, te = usable[tr_i], usable[te_i]
        fi, ci = next(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=0).split(tr, groups=blocks[tr]))
        fit, cal = tr[fi], tr[ci]
        P_cal = predict(C, fit, cal, fit_days)
        P_te = predict(C, fit, te, fit_days)
        # calibration material, from calibration stations on the fit days
        E_cal = obs[cal][:, fit_days] - P_cal[:, fit_days]                     # stations x days x 24
        pools = [E_cal[:, :, h][np.isfinite(E_cal[:, :, h])] for h in range(24)]
        q_hour = np.array([conf_q(np.abs(p)) for p in pools])
        full_cal = np.all(np.isfinite(E_cal), axis=-1)
        traj = E_cal[full_cal]                                                  # n_traj x 24
        O_cal, Ph_cal = obs[cal][:, fit_days][full_cal], P_cal[:, fit_days][full_cal]
        d_obs_cal, d_hat_cal = decisions(O_cal), decisions(Ph_cal)
        prov_cal = np.broadcast_to(prov[cal][:, None], full_cal.shape)[full_cal]
        qC = {k: conf_q(np.abs(d_obs_cal[k] - d_hat_cal[k])) for k in ("D1", "D2", "D2_35")}
        qN = {p: {k: conf_q(np.abs(d_obs_cal[k] - d_hat_cal[k])[prov_cal == p]) for k in ("D1", "D2", "D2_35")}
              for p in set(prov_cal)}
        iso = {e: IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(d_hat_cal["D1"], d_obs_cal[e].astype(float))
               for e in ("E32", "E35")}
        # test station-days
        O_te, P_te_d = obs[te][:, test_days], P_te[:, test_days]
        full_te = np.all(np.isfinite(O_te), axis=-1)
        # hourly coverage (H1) on all valid test hours
        Eh = O_te - P_te_d
        fin = np.isfinite(Eh)
        cov_h = np.abs(Eh) <= q_hour[None, None, :]
        hourly["covered"] += int(cov_h[fin].sum())
        hourly["n"] += int(fin.sum())
        hot_te = C["hot"][test_days]
        hourly["covered_hot"] += int(cov_h[:, hot_te][fin[:, hot_te]].sum())
        hourly["n_hot"] += int(fin[:, hot_te].sum())
        O, Ph = O_te[full_te], P_te_d[full_te]
        if len(O) == 0 or len(traj) == 0:
            print("[dc] fold skipped: no complete test or calibration station-days", flush=True)
            continue
        st_idx = np.broadcast_to(te[:, None], full_te.shape)[full_te]
        day_idx = np.broadcast_to(np.where(test_days)[0][None, :], full_te.shape)[full_te]
        d_obs, d_hat = decisions(O), decisions(Ph)
        res = {"station": st_idx, "day": day_idx, "obs": d_obs, "hat": d_hat}
        # A and B by Monte Carlo, in chunks
        K = args.draws
        for meth in ("A", "B"):
            lo = {k: np.empty(len(O)) for k in ("D1", "D2", "D2_35")}
            hi = {k: np.empty(len(O)) for k in ("D1", "D2", "D2_35")}
            pr = {e: np.empty(len(O)) for e in ("E32", "E35")}
            for a in range(0, len(O), 1000):
                b = min(len(O), a + 1000)
                if meth == "A":
                    err = np.stack([rng.choice(pools[h], size=(b - a, K)) for h in range(24)], axis=-1)
                else:
                    err = traj[rng.integers(0, len(traj), size=(b - a, K))]
                dd = decisions(Ph[a:b, None, :] + err.astype(np.float32))
                for k in lo:
                    lo[k][a:b], hi[k][a:b] = np.percentile(dd[k], 5, axis=1), np.percentile(dd[k], 95, axis=1)
                for e in pr:
                    pr[e][a:b] = dd[e].mean(axis=1)
            res[meth] = {"lo": lo, "hi": hi, "p": pr}
        res["C"] = {"lo": {k: d_hat[k] - qC[k] for k in qC}, "hi": {k: d_hat[k] + qC[k] for k in qC},
                    "p": {e: iso[e].predict(d_hat["D1"]) for e in iso}}
        pn = prov[st_idx]
        qn = {k: np.array([qN.get(p, qC)[k] if np.isfinite(qN.get(p, qC)[k]) else qC[k] for p in pn]) for k in qC}
        res["C_net"] = {"lo": {k: d_hat[k] - qn[k] for k in qC}, "hi": {k: d_hat[k] + qn[k] for k in qC},
                        "p": res["C"]["p"]}
        for meth in ("C", "C_net"):
            for k in ("D2", "D2_35"):
                res[meth]["lo"][k] = np.clip(res[meth]["lo"][k], 0, 24)
                res[meth]["hi"][k] = np.clip(res[meth]["hi"][k], 0, 24)
        rows.append(res)
        print(f"[dc] fold done: {len(te)} test stations, {len(O)} test station-days, {len(traj)} calibration days", flush=True)
    if not rows:
        return None
    # concatenate folds
    cat = lambda f: np.concatenate([f(r) for r in rows])
    out = {"station": cat(lambda r: r["station"]), "day": cat(lambda r: r["day"]),
           "obs": {k: cat(lambda r, k=k: r["obs"][k]) for k in rows[0]["obs"]},
           "hat": {k: cat(lambda r, k=k: r["hat"][k]) for k in rows[0]["hat"]}}
    for m in METHODS:
        out[m] = {"lo": {k: cat(lambda r, m=m, k=k: r[m]["lo"][k]) for k in rows[0][m]["lo"]},
                  "hi": {k: cat(lambda r, m=m, k=k: r[m]["hi"][k]) for k in rows[0][m]["hi"]},
                  "p": {e: cat(lambda r, m=m, e=e: r[m]["p"][e]) for e in rows[0][m]["p"]}}
    out["hourly"] = hourly
    return out


def score(C, out, args, rng):
    """Coverage, width, Brier and its reliability part, overall / hottest tenth / per network, with a station-block
    bootstrap (95% intervals) on the differences that the hypotheses are about."""
    blocks = C["blocks"][out["station"]]
    prov = C["prov"][out["station"]]
    hot = C["hot"][out["day"]]
    groups = {"all": np.ones(len(blocks), bool), "hottest_tenth": hot,
              "MesoWest": prov == "MesoWest", "CWOP": prov == "APRSWXNET"}
    rep = {"station_days": int(len(blocks)), "test_stations": int(len(set(out["station"]))),
           "hourly_coverage_90": round(out["hourly"]["covered"] / max(1, out["hourly"]["n"]), 3),
           "hourly_coverage_90_hottest_tenth": round(out["hourly"]["covered_hot"] / max(1, out["hourly"]["n_hot"]), 3),
           "base_rate": {e: round(float(out["obs"][e].mean()), 3) for e in ("E32", "E35")}, "methods": {}}
    ind = {}  # per-station-day indicators for the bootstrap
    for m in METHODS:
        r = {}
        for k in ("D1", "D2", "D2_35"):
            cov = (out["obs"][k] >= out[m]["lo"][k] - 1e-9) & (out["obs"][k] <= out[m]["hi"][k] + 1e-9)
            wid = out[m]["hi"][k] - out[m]["lo"][k]
            ind[(m, k)] = cov.astype(float)
            r[k] = {g: {"coverage_90": round(float(cov[s].mean()), 3), "width": round(float(wid[s].mean()), 2)}
                    for g, s in groups.items() if s.sum()}
        for e in ("E32", "E35"):
            p, y = out[m]["p"][e], out["obs"][e].astype(float)
            bs = (p - y) ** 2
            ind[(m, e)] = bs
            rr = {}
            for g, s in groups.items():
                if not s.sum():
                    continue
                bins = np.clip((p[s] * 10).astype(int), 0, 9)
                rel = sum((np.sum(bins == b) / s.sum()) * (p[s][bins == b].mean() - y[s][bins == b].mean()) ** 2
                          for b in range(10) if np.sum(bins == b))
                rr[g] = {"brier": round(float(bs[s].mean()), 4), "reliability": round(float(rel), 4),
                         "mean_p": round(float(p[s].mean()), 3), "rate": round(float(y[s].mean()), 3)}
            r[e] = rr
        rep["methods"][m] = r
    # bootstrap over 5 km blocks of test stations
    ub = np.unique(blocks)
    bidx = {b: np.where(blocks == b)[0] for b in ub}
    comps = [("B", "A", "D2"), ("B", "A", "E32"), ("C", "B", "D2"), ("C", "B", "D1"), ("C", "B", "E32"),
             ("B", "A", "D1"), ("C_net", "C", "D1")]
    boots = {c: [] for c in comps}
    for _ in range(args.boot):
        take = np.concatenate([bidx[b] for b in rng.choice(ub, size=len(ub), replace=True)])
        for c in comps:
            boots[c].append(ind[(c[0], c[2])][take].mean() - ind[(c[1], c[2])][take].mean())
    rep["differences"] = {f"{a}-{b} {k} ({'Brier' if k.startswith('E') else 'coverage'})":
                          {"diff": round(float(ind[(a, k)].mean() - ind[(b, k)].mean()), 4),
                           "ci95": [round(float(np.percentile(boots[(a, b, k)], q)), 4) for q in (2.5, 97.5)]}
                          for (a, b, k) in comps}
    return rep


def figure(rep_all, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 4, figsize=(13.5, 3.2), constrained_layout=True)
    col = {"A": "#b3b3b3", "B": "#2f5d8a", "C": "#c8553d", "C_net": "#e3a08f"}
    lab = {"A": "A hours independent", "B": "B whole-day trajectories", "C": "C decision level", "C_net": "C per network"}
    cases = [c for c in [("P1", "all", "all days"), ("P1", "hottest_tenth", "hottest tenth"),
                         ("P2", "all", "September (P2)"), ("P2", "hottest_tenth", "Sept. hottest (P2)")] if c[0] in rep_all]
    x = np.arange(len(cases))
    for j, (k, title) in enumerate((("D1", "daily maximum"), ("D2", "hours above 32 °C"), ("D2_35", "hours above 35 °C"))):
        for i, m in enumerate(METHODS):
            ys = [rep_all[p]["methods"][m][k].get(g, {}).get("coverage_90", np.nan) for p, g, _ in cases]
            ax[j].plot(x + (i - 1.5) * 0.12, ys, "o", color=col[m], ms=4, label=lab[m])
        ax[j].axhline(0.9, color="0.5", lw=0.7, ls="--")
        ax[j].set_ylim(0.55, 1.0)
        ax[j].set_xticks(x, [c[2] for c in cases], rotation=20)
        ax[j].set_ylabel("coverage of 90% intervals")
        ax[j].set_title(title, fontsize=8)
        ax[j].text(-0.15, 1.02, "abc"[j], transform=ax[j].transAxes, fontweight="bold", fontsize=10)
    ax[0].legend(frameon=False, fontsize=7, loc="lower right")
    for i, m in enumerate(("A", "B", "C")):
        ys = [rep_all[p]["methods"][m]["E32"].get(g, {}).get("brier", np.nan) for p, g, _ in cases]
        ax[3].plot(x + (i - 1) * 0.12, ys, "o", color=col[m], ms=4)
    ax[3].set_xticks(x, [c[2] for c in cases], rotation=20)
    ax[3].set_ylabel("Brier score (lower is better)")
    ax[3].set_title("4 h or more above 32 °C", fontsize=8)
    ax[3].text(-0.15, 1.02, "d", transform=ax[3].transAxes, fontweight="bold", fontsize=10)
    fig.savefig(path, dpi=200)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draws", type=int, default=400)
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--tag", default="")
    ap.add_argument("--replot", action="store_true", help="redraw the figure from the saved JSON only")
    args = ap.parse_args()
    if args.replot:
        rep = json.loads((sta.OUT / f"decision_calibration{args.tag}.json").read_text())
        figure(rep["protocols"], sta.OUT / f"decision_calibration{args.tag}.png")
        return
    rng = np.random.default_rng(0)
    C = build()
    names, st = C["names"], C["st"]
    pred = {p["station"]: p for p in json.loads((sta.RAW / "predictors.json").read_text())}
    has = np.array([s in pred for s in names])
    F = np.array([[pred[s][f] for f in sta.FEATS] if s in pred else [np.nan] * len(sta.FEATS) for s in names])
    C["F"] = np.hstack([F, (C["z_s"] - C["z_h"])[:, None], C["z_s"][:, None]])
    lat = np.array([st[s]["lat"] for s in names])
    lon = np.array([st[s]["lon"] for s in names])
    C["blocks"] = (np.floor(lat / 0.045) * 1000 + np.floor(lon / 0.054)).astype(int)
    C["prov"] = np.array([st[s]["provider"] for s in names])
    full = np.all(np.isfinite(C["obs"]), axis=-1)
    C["usable"] = np.where(has & (full.sum(axis=1) >= 20))[0]
    daymax = np.where(full, np.nanmax(C["obs"], axis=-1), np.nan)
    basin = np.nanmean(daymax[C["usable"]], axis=0)
    C["hot"] = basin >= np.nanpercentile(basin, 90)
    months = np.array([d.month for d in C["days"]])
    print(f"[dc] {len(C['days'])} days, {len(C['usable'])} usable stations, {int(full[C['usable']].sum())} complete "
          f"station-days, hours with data {C['hours_with_data']}", flush=True)
    rep = {"days": len(C["days"]), "usable_stations": int(len(C["usable"])),
           "complete_station_days": int(full[C["usable"]].sum()), "hours_with_data": C["hours_with_data"],
           "hottest_tenth_days": [str(d) for d, h in zip(C["days"], C["hot"]) if h],
           "draws": args.draws, "boot": args.boot, "protocols": {}}
    for name, fit_days, test_days in (("P1", np.ones(len(months), bool), np.ones(len(months), bool)),
                                      ("P2", months <= 8, months == 9)):
        out = run_protocol(C, fit_days, test_days, args, rng)
        if out is None:
            print(f"[dc] {name}: no test data, skipped", flush=True)
            continue
        rep["protocols"][name] = score(C, out, args, rng)
        print(f"[dc] {name} scored", flush=True)
    path = sta.OUT / f"decision_calibration{args.tag}.json"
    path.write_text(json.dumps(rep, indent=1))
    figure(rep["protocols"], sta.OUT / f"decision_calibration{args.tag}.png")
    print(json.dumps({p: {"hourly": rep["protocols"][p]["hourly_coverage_90"], "diffs": rep["protocols"][p]["differences"]}
                      for p in rep["protocols"]}, indent=1))


if __name__ == "__main__":
    main()
