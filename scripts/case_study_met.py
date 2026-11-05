"""A forecast ensemble with errors in air temperature, wind and solar radiation: how few physics runs?

Follows case_study.py (same venue tile, same 20 air-temperature error series, our code C1-C9,
SOLWEIG's own physics). There, a forecast ensemble that perturbs only the air temperature was
reproduced from two physics runs: the unperturbed run, plus one with Ta + 1 K giving each pixel's
dTmrt/dTa. A forecast also gets wind and sunshine wrong. Here every member carries three errors,
each AR(1) in time (lag-1 correlation 0.9) and independent of the others:

    air temperature   + SIGMA_TA x unit series (the members of case_study.py)
    wind speed        x exp(SIGMA_LOG_WIND x unit series)
    global radiation  x (1 + SIGMA_KDOWN x unit series), at least 0

The error sizes are illustrative (a day-ahead forecast: about 1.5 K, 30% in wind, 10% in global
radiation); their lead time is to be taken from published verification. Shortcuts against the
full-physics ensemble (one SOLWEIG run per member):

    (b)  the unperturbed run's Tmrt; UTCI per member from the member's Ta and wind (1 physics run)
    (c)  + each pixel's dTmrt/dTa per hour from the Ta + 1 K run (2 physics runs)
    (d)  + each pixel's dTmrt/d(radiation fraction) per hour from a run with radiation x 1.1 (3 runs)

Wind enters only UTCI in SOLWEIG-GPU v2.0.0 (no wind in Tmrt); the full-physics members test that.
Recorded per shortcut: member UTCI error against full physics, and the ensemble products (P of
UTCI above 32 C for 4 h or more, P above 38 C at 15:00), as in case_study.py.

Usage (after case_study.py, whose unperturbed and Ta + 1 K runs it reuses):
    python scripts/case_study_met.py --rasters data/la/expo_park --met data/la/met_expo_20190730.txt --tag expo
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

SIGMA_TA, SIGMA_LOG_WIND, SIGMA_KDOWN = 1.5, 0.3, 0.1
K_STEP = 0.1  # radiation x (1 + K_STEP) for the slope run
SEED_WIND, SEED_KDOWN = 11, 12


def ar1(seed: int, members: int, rho: float) -> np.ndarray:
    """Unit-variance AR(1) series, stationary from the first hour, (members, 24)."""
    z = np.random.default_rng(seed).standard_normal((members, 24))
    u = np.zeros_like(z)
    u[:, 0] = z[:, 0]
    for t in range(1, 24):
        u[:, t] = rho * u[:, t - 1] + np.sqrt(1 - rho ** 2) * z[:, t]
    return u


def members(n: int) -> list[dict]:
    from case_study import SIGMA
    from run_la_forecast_error import RHO, design

    assert SIGMA == SIGMA_TA
    ta = [r for r in design() if r["sigma"] == SIGMA_TA and r["member"] < n]
    uw, uk = ar1(SEED_WIND, n, RHO), ar1(SEED_KDOWN, n, RHO)
    return [{"id": f"met_m{r['member']:02d}", "member": r["member"], "ta_add": r["error"],
             "wind_scale": np.exp(SIGMA_LOG_WIND * uw[r["member"]]).tolist(),
             "kdown_scale": np.maximum(1 + SIGMA_KDOWN * uk[r["member"]], 0).tolist()} for r in ta]


@contextlib.contextmanager
def met_error(metfile_dir: Path, ta_add, wind_scale, kdown_scale):
    """Per-hour errors on every per-tile met file (Ta added, wind and radiation scaled); restored afterwards."""
    from solweig_sens.perturb import MET_COLS, _read_met, _write_met

    backup = metfile_dir.parent / (metfile_dir.name + "_orig")
    if backup.exists():  # a killed run left its backup: restore first
        shutil.rmtree(metfile_dir)
        shutil.copytree(backup, metfile_dir)
        shutil.rmtree(backup)
    shutil.copytree(metfile_dir, backup)
    try:
        for f in sorted(metfile_dir.glob("*.txt")):
            header, data = _read_met(f)
            hour = data[:, 2].astype(int) % 24
            for col, vals, add in (("ta", ta_add, True), ("wind", wind_scale, False), ("kdown", kdown_scale, False)):
                c = MET_COLS[col]
                ok = data[:, c] != -999.0
                v = np.asarray(vals)[hour[ok]]
                data[ok, c] = data[ok, c] + v if add else data[ok, c] * v
            _write_met(f, header, data)
        yield
    finally:
        shutil.rmtree(metfile_dir)
        shutil.copytree(backup, metfile_dir)
        shutil.rmtree(backup)


def main() -> None:
    import torch
    from osgeo import gdal
    from solweig_gpu import run_utci_tiles
    from case_study import HALF_MEMBER, day_patches, timed
    from solweig_sens.analysis import read_stack
    from solweig_sens.perturb import MET_COLS
    from spatial_ta import utci as utci_fn

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rasters", type=Path, required=True)
    ap.add_argument("--met", type=Path, required=True)
    ap.add_argument("--date", default="2019-07-30")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--work", type=Path, default=Path.home() / "case_study")
    ap.add_argument("--out", type=Path, default=ROOT / "figures")
    ap.add_argument("--members", type=int, default=20)
    args = ap.parse_args()
    work = args.work / args.tag
    pre = work / "day" / "processed_inputs"
    days = work / "days" / "ours"
    if not (days / "base" / "output_folder").exists() or not (days / "plus1K" / "output_folder").exists():
        raise SystemExit("run case_study.py first (its unperturbed and Ta + 1 K runs are reused)")
    out_json = args.out / f"case_study_met_{args.tag}.json"
    res = json.loads(out_json.read_text()) if out_json.exists() else {
        "tag": args.tag, "date": args.date, "members": args.members,
        "errors": {"ta_sd_K": SIGMA_TA, "wind_log_sd": SIGMA_LOG_WIND, "kdown_fraction_sd": SIGMA_KDOWN,
                   "rho": 0.9, "seeds": {"wind": SEED_WIND, "kdown": SEED_KDOWN}}, "runs": {}}
    res["members"] = args.members
    torch.zeros(1, device="cuda")
    zero = [0.0] * 24
    one = [1.0] * 24
    todo = [{"id": "kdown_plus", "ta_add": zero, "wind_scale": one, "kdown_scale": [1 + K_STEP] * 24},
            *members(args.members)]
    for r in todo:
        dst = days / r["id"]
        if r["id"] in res["runs"] and (dst / "output_folder").exists():
            continue
        shutil.rmtree(dst, ignore_errors=True)

        def day(r=r, dst=dst):
            with contextlib.ExitStack() as st:
                for c in day_patches("ours"):
                    st.enter_context(c)
                st.enter_context(met_error(pre / "metfiles", r["ta_add"], r["wind_scale"], r["kdown_scale"]))
                run_utci_tiles(base_path=str(dst), preprocess_dir=str(pre), selected_date_str=args.date, save_tmrt=True)

        res["runs"][r["id"]] = timed(day)
        out_json.write_text(json.dumps(res, indent=2))
        print(f"[met] {r['id']:11s} {res['runs'][r['id']]['seconds']:5.1f} s", flush=True)

    # the shortcuts
    met = np.loadtxt(args.met, skiprows=1)
    hour = met[:, 2].astype(int) % 24
    ta, rh, ws = (np.zeros(24) for _ in range(3))
    ta[hour], rh[hour], ws[hour] = (met[:, MET_COLS[c]] for c in ("ta", "rh", "wind"))
    tmrt0 = read_stack(days / "base", "TMRT").astype(np.float64)
    s_ta = read_stack(days / "plus1K", "TMRT") - tmrt0
    s_k = (read_stack(days / "kdown_plus", "TMRT") - tmrt0) / K_STEP
    lc = gdal.Open(str(args.rasters / "Landcover.tif")).ReadAsArray()[: tmrt0.shape[1], : tmrt0.shape[2]]
    outdoor = (lc != 2) & np.isfinite(tmrt0).all(axis=0)
    shape = outdoor.shape
    # products as sums of member indicators, and member errors, one member at a time (fields are large)
    def indicators(u):
        return {"p_4h_above_32": ((u > 32.0).sum(axis=0) >= 4).astype(np.float64), "p_above_38_at_15": (u[15] > 38.0) * 1.0}

    names = ("b", "c", "d")
    pf = {k: np.zeros(shape) for k in ("p_4h_above_32", "p_above_38_at_15")}
    psum = {n: {k: np.zeros(shape) for k in pf} for n in names}
    err = {n: [] for n in names}
    t0 = time.perf_counter()
    ms = members(args.members)
    for r in ms:
        f = read_stack(days / r["id"], "UTCI")
        for k, v in indicators(f).items():
            pf[k] += v / len(ms)
        e, w, kf = (np.asarray(r[x]) for x in ("ta_add", "wind_scale", "kdown_scale"))
        tm = {"b": tmrt0, "c": tmrt0 + e[:, None, None] * s_ta,
              "d": tmrt0 + e[:, None, None] * s_ta + (kf - 1)[:, None, None] * s_k}
        for name, t in tm.items():
            u = np.stack([utci_fn(np.full(shape, ta[h] + e[h]), np.full(shape, rh[h]), t[h],
                                  np.full(shape, ws[h] * w[h])) for h in range(t.shape[0])])
            for k, v in indicators(u).items():
                psum[name][k] += v / len(ms)
            d = np.abs(u - f)[:, outdoor]
            err[name].append((float(np.nanmean(d)), float(np.nanpercentile(d, 99)), float(np.nanmax(d))))
    t_short = time.perf_counter() - t0

    res["full_physics_products"] = {k: float(np.nanmean(v[outdoor])) for k, v in pf.items()}
    res["shortcuts"] = {}
    labels = {"b": "1 physics run: unperturbed Tmrt, member Ta and wind in UTCI",
              "c": "2 physics runs: + per-pixel dTmrt/dTa", "d": "3 physics runs: + per-pixel dTmrt/d(radiation fraction)"}
    for name in names:
        ps = psum[name]
        e = np.array(err[name])
        res["shortcuts"][name] = {
            "what": labels[name],
            "member_utci_error": {"mean_abs": float(e[:, 0].mean()), "p99_abs": float(e[:, 1].mean()),
                                  "max_abs": float(e[:, 2].max())},
            "products_abs_diff": {k: {"mean": float(np.abs(ps[k] - pf[k])[outdoor].mean()),
                                      "max": float(np.abs(ps[k] - pf[k])[outdoor].max()),
                                      "pixels_differing_pct": float((np.abs(ps[k] - pf[k])[outdoor] > HALF_MEMBER).mean() * 100)}
                                  for k in pf}}
        s = res["shortcuts"][name]
        print(f"[met] ({name}) {labels[name]}: member UTCI error mean {s['member_utci_error']['mean_abs']:.3f} C, "
              f"p99 {s['member_utci_error']['p99_abs']:.3f}, max {s['member_utci_error']['max_abs']:.2f} C; "
              f"P(4 h > 32) differs at {s['products_abs_diff']['p_4h_above_32']['pixels_differing_pct']:.1f}% of pixels, "
              f"P(> 38 at 15 h) at {s['products_abs_diff']['p_above_38_at_15']['pixels_differing_pct']:.2f}%", flush=True)
    sk = s_k[:, outdoor]
    res["slope_dtmrt_dkfrac_at_13h"] = {q: float(np.nanpercentile(sk[13], p)) for q, p in (("p1", 1), ("median", 50), ("p99", 99))}
    res["shortcut_seconds"] = t_short
    out_json.write_text(json.dumps(res, indent=2))
    print(f"[met] wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
