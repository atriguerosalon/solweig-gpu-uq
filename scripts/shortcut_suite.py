"""Forecast-ensemble shortcuts against full physics, for any tile, day and error model, with a noise yardstick.

Brings together case_study.py, case_study_met.py, radiation_table.py and ensemble_noise.py so the
same test can be repeated on other days, venues and error models. On one tile (our code C1-C9,
SOLWEIG's own physics), two independent ensembles A and B of MEMBERS members each are run in full;
every member carries errors in air temperature, wind and global radiation, AR(1) in time (lag-1
correlation 0.9). Error models:

    independent   Ta sd 1.5 K, wind x exp(0.3 u), radiation x (1 + 0.1 u), independent; RH held
    correlated    the radiation error correlated with the Ta error (correlation 0.6: cloudier
                  hours are cooler), and the dew point held instead of RH (RH recomputed from the
                  member's Ta, Magnus formula), as a forecast error in Ta alone would do

Shortcuts for ensemble A, from a few physics runs (the slope and table runs use the same RH rule):

    (b) 1 run: unperturbed Tmrt; member Ta, RH, wind in UTCI
    (c) 2 runs: + per-pixel dTmrt/dTa per hour (run with Ta + 1 K)
    (d) 3 runs: + one-sided radiation slope (run with radiation x 1.1)
    (e) 4 runs: + two-sided radiation slope (runs x 1.1 and x 0.9)
    (f) 14 runs: per-hour table in the radiation scale (0.70 to 1.30, step 0.05) + Ta slope

Scored on the products P(UTCI above 32 or 38 C for 4 h or more), P(above 38 or 46 C at 15:00): share of outdoor
pixels off by one member or more and mean |dp|, against the same for ensemble A vs ensemble B
(two equally valid ensembles: the sampling noise).

With --structure fitted, every run uses the paved ground heating fitted to the Pacoima surface
measurements and ground heat storage (tau 4 h), as run_la_forecast_error.py (site-specific).

Usage:
    python scripts/shortcut_suite.py --rasters data/la/expo_park --met data/la/met_expo_20190615.txt \
        --date 2019-06-15 --tag expo_0615 --errors independent
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

MEMBERS = 20
RHO = 0.9
SIGMA_TA, SIGMA_LOG_WIND, SIGMA_KDOWN = 1.5, 0.3, 0.1
CORR_TA_K = 0.6
GRID = np.round(np.arange(0.70, 1.3001, 0.05), 4)
SEEDS = {"A": (100, 101, 102), "B": (200, 201, 202)}


def es(t):
    """Saturation vapour pressure (hPa), Magnus formula (Alduchov and Eskridge 1996)."""
    return 6.1094 * np.exp(17.625 * t / (t + 243.04))


def ensemble(name: str, errors: str) -> list[dict]:
    from case_study_met import ar1

    s_ta, s_w, s_k = SEEDS[name]
    u_ta, u_w, z_k = ar1(s_ta, MEMBERS, RHO), ar1(s_w, MEMBERS, RHO), ar1(s_k, MEMBERS, RHO)
    u_k = CORR_TA_K * u_ta + np.sqrt(1 - CORR_TA_K ** 2) * z_k if errors == "correlated" else z_k
    return [{"id": f"{name}_m{m:02d}", "ta_add": (SIGMA_TA * u_ta[m]).tolist(),
             "wind_scale": np.exp(SIGMA_LOG_WIND * u_w[m]).tolist(),
             "kdown_scale": np.maximum(1 + SIGMA_KDOWN * u_k[m], 0).tolist()} for m in range(MEMBERS)]


@contextlib.contextmanager
def met_errors(metfile_dir: Path, ta_add, wind_scale, kdown_scale, dewpoint: bool):
    """Per-hour errors on every per-tile met file; with dewpoint, RH follows the new Ta at fixed vapour pressure.

    Crash-safe: the unperturbed files are copied once to a pristine folder (made right after
    preprocessing, before any perturbation); every run reads from there and each file is replaced by an
    atomic rename, so a crash never leaves a half-written or perturbed file behind for the next run.
    """
    import os

    from solweig_sens.perturb import MET_COLS, _read_met, _write_met

    pristine = metfile_dir.parent / (metfile_dir.name + "_pristine")
    if not pristine.exists():
        raise RuntimeError(f"no pristine met files at {pristine}: make them right after preprocessing")

    def restore():
        for p in sorted(pristine.glob("*.txt")):
            tmp = metfile_dir / (p.name + ".tmp")
            shutil.copyfile(p, tmp)
            os.replace(tmp, metfile_dir / p.name)

    restore()
    try:
        for p in sorted(pristine.glob("*.txt")):
            f = metfile_dir / p.name
            header, data = _read_met(p)
            hour = data[:, 2].astype(int) % 24
            ok = data[:, MET_COLS["ta"]] != -999.0
            ta0 = data[ok, MET_COLS["ta"]].copy()
            add = np.asarray(ta_add)[hour[ok]]
            data[ok, MET_COLS["ta"]] = ta0 + add
            if dewpoint:
                c = MET_COLS["rh"]
                data[ok, c] = np.clip(data[ok, c] * es(ta0) / es(ta0 + add), 1.0, 100.0)
            for col, vals in (("wind", wind_scale), ("kdown", kdown_scale)):
                c = MET_COLS[col]
                v = data[:, c] != -999.0
                data[v, c] = data[v, c] * np.asarray(vals)[hour[v]]
            tmp = metfile_dir / (p.name + ".tmp")
            _write_met(tmp, header, data)
            os.replace(tmp, f)
        yield
    finally:
        restore()


def main() -> None:
    global MEMBERS
    import torch
    from osgeo import gdal
    from solweig_gpu import run_utci_tiles
    from case_study import HALF_MEMBER, day_patches, svf_patches
    from run_la_pavement import TS_FIT
    from solweig_sens.analysis import read_stack
    from solweig_sens.heat_storage import ground_heat_storage
    from solweig_sens.perturb import MET_COLS, PAVED_CLASSES, patched_landcover
    from solweig_sens.run import Setup, prepare
    from spatial_ta import utci as utci_fn

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rasters", type=Path, required=True)
    ap.add_argument("--met", type=Path, required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--errors", choices=["independent", "correlated"], default="independent")
    ap.add_argument("--perez", choices=["coded", "published"], default="coded",
                    help="published: the Perez sky clearness as published (perez_formula_effect.py)")
    ap.add_argument("--structure", choices=["default", "fitted"], default="default",
                    help="fitted: paved ground heating fitted to the Pacoima surface measurements and ground heat "
                         "storage (tau 4 h), as run_la_forecast_error.py; site-specific")
    ap.add_argument("--work", type=Path, default=Path.home() / "shortcut_suite")
    ap.add_argument("--out", type=Path, default=ROOT / "figures" / "shortcuts")
    ap.add_argument("--members", type=int, default=MEMBERS, help="fewer only to check the script (same first members)")
    args = ap.parse_args()
    MEMBERS = args.members
    dew = args.errors == "correlated"
    work = args.work / args.tag
    args.out.mkdir(parents=True, exist_ok=True)
    sfx = ("" if args.structure == "default" else "_fitted") + ("" if args.perez == "coded" else "_perez_published")
    out_json = args.out / f"shortcut_suite_{args.tag}_{args.errors}{sfx}.json"
    torch.zeros(1, device="cuda")
    setup = Setup(raster_dir=args.rasters.resolve(), met_file=args.met.resolve(), runs_dir=work / "prep", date=args.date,
                  tile_size=1000)
    if not (setup.preprocess_dir / "SVF").exists():
        with contextlib.ExitStack() as st:
            for c in svf_patches("ours"):
                st.enter_context(c)
            prepare(setup)
    pre = setup.preprocess_dir
    if not (pre / "metfiles_pristine").exists():  # the unperturbed met files, before any run perturbs them
        if (pre / "metfiles_orig").exists():
            raise SystemExit(f"{pre} holds a leftover metfiles_orig: check which copy is unperturbed first")
        shutil.copytree(pre / "metfiles", pre / "metfiles_pristine")
    runs = work / f"runs_{args.errors}{sfx}"
    zero, one = [0.0] * 24, [1.0] * 24
    jobs = [("base", zero, one, one), ("plus1K", [1.0] * 24, one, one), ("kplus", zero, one, [1.1] * 24),
            ("kminus", zero, one, [0.9] * 24)]
    jobs += [(f"table_{f:.2f}", zero, one, [float(f)] * 24) for f in GRID if f != 1.0]
    ens = {k: ensemble(k, args.errors) for k in ("A", "B")}
    jobs += [(r["id"], r["ta_add"], r["wind_scale"], r["kdown_scale"]) for k in "AB" for r in ens[k]]
    def complete(dst: Path) -> bool:
        """A finished run: both stacks readable with 24 hours. Always read, marker or not: after a crash of the
        WSL VM a run can have its marker while its files never reached the disk (4 Nov, B_m10)."""
        try:
            return all(read_stack(dst, v).shape[0] == 24 for v in ("UTCI", "TMRT"))
        except Exception:
            return False

    t_start = time.perf_counter()
    for i, (rid, ta_add, ws_, ks_) in enumerate(jobs):
        dst = runs / rid
        if complete(dst):  # a stopped chain resumes; a run cut off half-way is redone
            continue
        shutil.rmtree(dst, ignore_errors=True)
        with contextlib.ExitStack() as st:
            for c in day_patches("ours"):
                st.enter_context(c)
            st.enter_context(met_errors(pre / "metfiles", ta_add, ws_, ks_, dew))
            if args.perez == "published":
                from perez_formula_effect import published_clearness
                st.enter_context(published_clearness())
            if args.structure == "fitted":
                st.enter_context(patched_landcover(work, PAVED_CLASSES, {"ts_deg": TS_FIT}))
                st.enter_context(ground_heat_storage(4.0))
            run_utci_tiles(base_path=str(dst), preprocess_dir=str(pre), selected_date_str=args.date, save_tmrt=True)
        (dst / "done").touch()
        print(f"[suite] {args.tag} {args.errors}: {i + 1}/{len(jobs)} {rid} "
              f"({(time.perf_counter() - t_start) / 60:.1f} min)", flush=True)

    met = np.loadtxt(args.met, skiprows=1)
    hour = met[:, 2].astype(int) % 24
    ta, rh, ws = (np.zeros(24) for _ in range(3))
    ta[hour], rh[hour], ws[hour] = (met[:, MET_COLS[c]] for c in ("ta", "rh", "wind"))
    t0 = read_stack(runs / "base", "TMRT").astype(np.float64)
    s_ta = read_stack(runs / "plus1K", "TMRT") - t0
    s_up = (read_stack(runs / "kplus", "TMRT") - t0) / 0.1
    s_dn = (t0 - read_stack(runs / "kminus", "TMRT")) / 0.1
    table = np.stack([t0 if f == 1.0 else read_stack(runs / f"table_{f:.2f}", "TMRT") for f in GRID])
    lc = gdal.Open(str(args.rasters / "Landcover.tif")).ReadAsArray()[: t0.shape[1], : t0.shape[2]]
    outdoor = (lc != 2) & np.isfinite(t0).all(axis=0)
    shape = outdoor.shape

    def interp(h, k):
        k = min(max(k, GRID[0]), GRID[-1])
        i = min(int(np.searchsorted(GRID, k, side="right")) - 1, len(GRID) - 2)
        w = (k - GRID[i]) / (GRID[i + 1] - GRID[i])
        return (1 - w) * table[i, h] + w * table[i + 1, h]

    def indicators(u):
        return {"p_4h_above_32": ((u > 32.0).sum(axis=0) >= 4) * 1.0, "p_4h_above_38": ((u > 38.0).sum(axis=0) >= 4) * 1.0,
                "p_above_38_at_15": (u[15] > 38.0) * 1.0, "p_above_46_at_15": (u[15] > 46.0) * 1.0}

    keys = ("p_4h_above_32", "p_4h_above_38", "p_above_38_at_15", "p_above_46_at_15")
    names = "bcdef"
    p_full = {k: {q: np.zeros(shape) for q in keys} for k in "AB"}
    p_sc = {n: {q: np.zeros(shape) for q in keys} for n in names}
    err = {n: [] for n in names}
    for k in "AB":
        for r in ens[k]:
            full = read_stack(runs / r["id"], "UTCI")
            for q, v in indicators(full).items():
                p_full[k][q] += v / MEMBERS
            if k == "B":
                continue
            e, w, kf = (np.asarray(r[x]) for x in ("ta_add", "wind_scale", "kdown_scale"))
            rh_m = np.clip(rh * es(ta) / es(ta + e), 1.0, 100.0) if dew else rh
            dk = (kf - 1)[:, None, None]
            tms = {"b": t0, "c": t0 + e[:, None, None] * s_ta, "d": t0 + e[:, None, None] * s_ta + dk * s_up,
                   "e": t0 + e[:, None, None] * s_ta + np.where(dk >= 0, dk * s_up, dk * s_dn),
                   "f": np.stack([interp(h, kf[h]) for h in range(24)]) + e[:, None, None] * s_ta}
            for n, tm in tms.items():
                u = np.stack([utci_fn(np.full(shape, ta[h] + e[h]), np.full(shape, rh_m[h]), tm[h],
                                      np.full(shape, ws[h] * w[h])) for h in range(24)])
                for q, v in indicators(u).items():
                    p_sc[n][q] += v / MEMBERS
                d = np.abs(u - full)[:, outdoor]
                err[n].append((float(np.nanmean(d)), float(np.nanpercentile(d, 99)), float(np.nanmax(d))))

    def score(p, ref):
        return {q: {"mean_abs_dp": float(np.abs(p[q] - ref[q])[outdoor].mean()),
                    "pixels_off_one_member_pct": float((np.abs(p[q] - ref[q])[outdoor] > HALF_MEMBER).mean() * 100)}
                for q in keys}

    res = {"tag": args.tag, "errors": args.errors, "structure": args.structure, "perez": args.perez, "date": args.date, "met": str(args.met), "members": MEMBERS,
           "event_frequency": {q: float(p_full["A"][q][outdoor].mean()) for q in keys},
           "noise_A_vs_B": score(p_full["B"], p_full["A"]), "shortcuts": {}}
    runs_needed = {"b": 1, "c": 2, "d": 3, "e": 4, "f": len(GRID) + 1}
    for n in names:
        x = np.array(err[n])
        res["shortcuts"][n] = {"physics_runs": runs_needed[n],
                               "member_utci_error": {"mean_abs": float(x[:, 0].mean()), "p99_abs": float(x[:, 1].mean()),
                                                     "max_abs": float(x[:, 2].max())},
                               **score(p_sc[n], p_full["A"])}
    out_json.write_text(json.dumps(res, indent=2))
    nz = res["noise_A_vs_B"]["p_4h_above_32"]
    print(f"[suite] {args.tag} {args.errors}: event frequency P(4 h > 32) {res['event_frequency']['p_4h_above_32']:.2f}; "
          f"noise A vs B: {nz['pixels_off_one_member_pct']:.1f}% of pixels, mean |dp| {nz['mean_abs_dp']:.4f}", flush=True)
    for n in names:
        s = res["shortcuts"][n]
        print(f"[suite]   ({n}) {s['physics_runs']:2d} runs: member UTCI error mean {s['member_utci_error']['mean_abs']:.3f}, "
              f"max {s['member_utci_error']['max_abs']:.2f} C; P(4 h > 32) {s['p_4h_above_32']['pixels_off_one_member_pct']:.1f}%, "
              f"mean |dp| {s['p_4h_above_32']['mean_abs_dp']:.4f} ({s['p_4h_above_32']['mean_abs_dp'] / max(nz['mean_abs_dp'], 1e-9):.2f} x noise)",
              flush=True)
    print(f"[suite] wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
