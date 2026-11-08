"""A street-scale UTCI forecast driven by real HRRR forecasts: the venue tile in the Aug-Sep 2022 Los Angeles heatwave.

The forecast-error ensembles are perfect-model tests with assumed error sizes. This runs what a forecast system would
run: SOLWEIG on the Exposition Park venue tile (2 x 2 km at 2 m, our code C1-C9, SOLWEIG's own physics) forced by
HRRR forecasts at the tile centre, for each day of the heatwave, at three lead times:

    d0   HRRR run of 06 UTC on the day (23:00 PDT the evening before): the day's hours at leads 1-24 h
    d1   HRRR run of 18 UTC the day before (11:00 PDT): leads 13-36 h
    d2   HRRR run of 06 UTC the day before: leads 25-48 h

and compares the UTCI maps with two references:

    anl  forced by HRRR analyses (lead 0, each hour; radiation from the 1 h forecast of the hour before, since
         radiation is a forecast diagnostic): the forecast verified against the model's own best estimate
    obs  air temperature, humidity and wind from the Downtown USC station (KCQT, 1 km from the tile), radiation
         and pressure from the HRRR analysis (so radiation enters the same way in both and only the station
         variables differ)
    obsw the same with HRRR's analysis wind instead of the station's: the station is sheltered and often calm, the
         model's wind is for 10 m over its 3 km cell

Forcing at the HRRR grid point nearest the tile centre: 2 m temperature and humidity, 10 m wind, surface downward
shortwave (instantaneous at the hour), surface pressure. SOLWEIG splits global radiation itself.

    python scripts/forecast_demo.py hrrr --start 2022-08-30 --end 2022-09-10     (env with herbie-data, cfgrib)
    python scripts/forecast_demo.py run                                           (solweig env, GPU)
    python scripts/forecast_demo.py analyse                                       (solweig env)

Writes figures/forecast_demo/: hrrr_point.csv, met/*.txt, forecast_demo.json, forecast_demo.png; runs keep UTCI for
every hour as float16 in <work>/utci/.
"""

from __future__ import annotations

import sys as _sys

if __name__ == "__main__" and any(a in ("-h", "--help") for a in _sys.argv[1:]):
    print(__doc__)
    raise SystemExit(0)

import argparse
import csv
import json
import shutil
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "figures" / "forecast_demo"
LAT, LON = 34.0141, -118.2879  # venue tile centre
UTC_OFFSET = 7  # PDT
VARIANTS = {"d0": (0, 6), "d1": (-1, 18), "d2": (-1, 6)}  # init: (day offset, UTC hour)
SEARCH = ":(?:TMP|RH):2 m above|:(?:UGRD|VGRD):10 m above|:DSWRF:surface|:PRES:surface"
HOURS_KEPT = list(range(24))
HEADER = ("%iy  id  it imin   Q*      QH      QE      Qs      Qf    Wind    RH     Td     press   rain    Kdn    "
          "snow    ldown   fcld    wuh     xsmd    lai_hr  Kdiff   Kdir    Wd")


def days(a: str, b: str) -> list[date]:
    d0, d1 = date.fromisoformat(a), date.fromisoformat(b)
    return [d0 + timedelta(days=i) for i in range((d1 - d0).days + 1)]


def valid_utc(d: date, h: int) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc) + timedelta(hours=h + UTC_OFFSET)


def needed(ds: list[date]) -> set[tuple[datetime, int]]:
    """(init, lead) pairs: the analysis of every hour, and the three forecasts of every day."""
    out = set()
    for d in ds:
        for h in range(24):
            v = valid_utc(d, h)
            out.add((v, 0))
            out.add((v - timedelta(hours=1), 1))  # radiation of the analysis hour: the 1 h forecast
            for off, hh in VARIANTS.values():
                init = datetime(d.year, d.month, d.day, hh, tzinfo=timezone.utc) + timedelta(days=off)
                out.add((init, int((v - init).total_seconds() // 3600)))
    return out


GRID_IJ = (435, 264)  # HRRR grid point nearest the tile centre (checked with cfgrib, 8 Nov)
NX = 1799
SHORT = {"2t": "t2m", "2r": "r2", "10u": "u10", "10v": "v10", "dswrf": "sdswrf", "sdswrf": "sdswrf", "sp": "sp"}


def read_point(path: str) -> dict:
    """Values at the tile's grid point from a GRIB2 file, with eccodes directly (cfgrib is slow on mixed levels)."""
    import eccodes

    k = GRID_IJ[0] * NX + GRID_IJ[1]
    out = {}
    with open(path, "rb") as f:
        while True:
            gid = eccodes.codes_grib_new_from_file(f)
            if gid is None:
                break
            try:
                name = SHORT.get(eccodes.codes_get(gid, "shortName"))
                if name:
                    out[name] = float(eccodes.codes_get_values(gid)[k])
            finally:
                eccodes.codes_release(gid)
    return out


def hrrr(args) -> None:
    from concurrent.futures import ProcessPoolExecutor
    from herbie import Herbie

    OUT.mkdir(parents=True, exist_ok=True)
    pts = OUT / "hrrr_point.csv"
    have = set()
    if pts.exists():
        with open(pts) as f:
            have = {(r["init"], int(r["lead"])) for r in csv.DictReader(f) if r["note"] == ""}  # failures are retried
    jobs = sorted((i, L) for i, L in needed(days(args.start, args.end)) if (i.strftime("%Y-%m-%dT%H"), L) not in have)
    print(f"[demo] {len(jobs)} HRRR files to read", flush=True)
    tmp_root = Path(args.tmp)
    tmp_root.mkdir(parents=True, exist_ok=True)

    def fetch(init, L):
        d = Path(tempfile.mkdtemp(dir=tmp_root))
        try:
            H = Herbie(init.strftime("%Y-%m-%d %H:%M"), model="hrrr", product="sfc", fxx=L, save_dir=str(d), verbose=False)
            return init, L, str(H.download(SEARCH)), d
        except Exception as e:
            return init, L, repr(e)[:150], d

    new = not pts.exists()
    with open(pts, "a", newline="") as f, ThreadPoolExecutor(args.threads) as tex, ProcessPoolExecutor(6) as pex:
        w = csv.writer(f)
        if new:
            w.writerow(["init", "lead", "valid", "t2m_c", "rh", "u10", "v10", "sw_down", "psurf_kpa", "note"])
        reads = {}
        for fu in as_completed([tex.submit(fetch, i, L) for i, L in jobs]):
            init, L, path, d = fu.result()
            if Path(path).exists():
                reads[pex.submit(read_point, path)] = (init, L, d)
            else:
                w.writerow([init.strftime("%Y-%m-%dT%H"), L, (init + timedelta(hours=L)).strftime("%Y-%m-%dT%H")]
                           + [""] * 6 + [path])
                shutil.rmtree(d, ignore_errors=True)
            done = [r for r in reads if r.done()]
            for r in done:
                _write(w, r, reads.pop(r))
            if len(reads) == 0 or len(done):
                f.flush()
        for r in as_completed(list(reads)):
            _write(w, r, reads.pop(r))
    print("[demo] HRRR points done", flush=True)


def _write(w, fut, meta) -> None:
    init, L, d = meta
    row = [init.strftime("%Y-%m-%dT%H"), L, (init + timedelta(hours=L)).strftime("%Y-%m-%dT%H")]
    try:
        v = fut.result()
        row += [v.get("t2m", np.nan) - 273.15, v.get("r2", np.nan), v.get("u10", np.nan), v.get("v10", np.nan),
                v.get("sdswrf", np.nan), v.get("sp", np.nan) / 1000.0, ""]
    except Exception as e:
        row += [""] * 6 + [repr(e)[:150]]
    w.writerow(row)
    shutil.rmtree(d, ignore_errors=True)
    _write.n = getattr(_write, "n", 0) + 1
    if _write.n % 50 == 0:
        print(f"[demo] {_write.n} read", flush=True)


def load_points() -> dict:
    out = {}
    with open(OUT / "hrrr_point.csv") as f:
        for r in csv.DictReader(f):
            if r["note"] == "" and r["t2m_c"] != "":
                out[(r["init"], int(r["lead"]))] = {k: float(r[k]) for k in ("t2m_c", "rh", "u10", "v10", "sw_down", "psurf_kpa")}
    return out


def met_rows(d: date, series: list[dict]) -> list[list]:
    doy = d.timetuple().tm_yday
    rows = []
    for h, x in enumerate(series):
        wind = float(np.hypot(x["u10"], x["v10"]))
        wdir = float((270 - np.degrees(np.arctan2(x["v10"], x["u10"]))) % 360)
        rows.append([d.year, doy, h, 0] + [-999.0] * 5 + [max(wind, 0.1), min(max(x["rh"], 1.0), 100.0), x["t2m_c"],
                     x["psurf_kpa"], 0.0, max(x["sw_down"], 0.0)] + [-999.0] * 8 + [wdir])
    return rows


def write_met(path: Path, rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(HEADER + "\n")
        for r in rows:
            f.write(" ".join(f"{v:d}" if isinstance(v, int) else f"{v:.2f}" for v in r) + "\n")


def build_mets(ds: list[date]) -> dict:
    """Met files per day and variant; days with a missing hour in any variant are skipped (and reported)."""
    from make_la_met import station_hours

    P = load_points()
    made = {}
    for d in ds:
        series = {}
        try:
            series["anl"] = []
            for h in range(24):
                v = valid_utc(d, h)
                x = dict(P[(v.strftime("%Y-%m-%dT%H"), 0)])
                x["sw_down"] = P[((v - timedelta(hours=1)).strftime("%Y-%m-%dT%H"), 1)]["sw_down"]
                series["anl"].append(x)
            for name, (off, hh) in VARIANTS.items():
                init = datetime(d.year, d.month, d.day, hh, tzinfo=timezone.utc) + timedelta(days=off)
                series[name] = [P[(init.strftime("%Y-%m-%dT%H"), int((valid_utc(d, h) - init).total_seconds() // 3600))]
                                for h in range(24)]
        except KeyError as e:
            print(f"[demo] {d}: HRRR missing {e}; day skipped", flush=True)
            continue
        stn = station_hours("CQT", d)
        if len(stn) < 20:
            print(f"[demo] {d}: only {len(stn)} station hours; day skipped", flush=True)
            continue
        have = sorted(stn)
        obs = []
        for h in range(24):
            x = dict(series["anl"][h])  # radiation and pressure from the analysis
            for k, src in (("t2m_c", "ta"), ("rh", "rh")):
                x[k] = stn[h][src] if h in stn else float(np.interp(h, have, [stn[i][src] for i in have]))
            ws = stn[h]["wind"] if h in stn else float(np.interp(h, have, [stn[i]["wind"] for i in have]))
            x["u10"], x["v10"] = 0.0, -ws  # direction is not used by SOLWEIG's UTCI; speed is what matters
            obs.append(x)
        series["obs"] = obs
        # the station's air temperature and humidity with HRRR's wind: separates the forecast error from the gap between
        # a sheltered station wind and the model's 10 m wind
        series["obsw"] = [dict(o, u10=a["u10"], v10=a["v10"]) for o, a in zip(obs, series["anl"])]
        for name, s in series.items():
            p = OUT / "met" / f"met_{name}_{d:%Y%m%d}.txt"
            write_met(p, met_rows(d, s))
            made[(d.isoformat(), name)] = p
    return made


def run(args) -> None:
    import torch
    from case_study import day_patches
    from solweig_gpu import run_utci_tiles
    from solweig_sens.analysis import read_stack
    import contextlib

    ds = days(args.start, args.end)
    mets = build_mets(ds)
    work = Path(args.work)
    pre = work / "processed_inputs"
    if not pre.exists():
        shutil.copytree(Path(args.pre), pre)
    (work / "utci").mkdir(parents=True, exist_ok=True)
    timing = {}
    for (d, name), met in sorted(mets.items()):
        dst = work / "utci" / f"{name}_{d}.npy"
        if dst.exists():
            continue
        shutil.copy(met, pre / "metfiles" / "metfile_0_0.txt")
        out = work / "tmp_run"
        shutil.rmtree(out, ignore_errors=True)
        t = time.perf_counter()
        with contextlib.ExitStack() as st:
            for c in day_patches("ours"):
                st.enter_context(c)
            run_utci_tiles(base_path=str(out), preprocess_dir=str(pre), selected_date_str=d, save_tmrt=False)
        torch.cuda.synchronize()
        timing[f"{name}_{d}"] = round(time.perf_counter() - t, 1)
        u = read_stack(out, "UTCI")
        np.save(dst, u[HOURS_KEPT].astype(np.float16))
        shutil.rmtree(out, ignore_errors=True)
        print(f"[demo] ran {name} {d}: {timing[f'{name}_{d}']} s", flush=True)
    t_old = json.loads((OUT / "run_seconds.json").read_text()) if (OUT / "run_seconds.json").exists() else {}
    (OUT / "run_seconds.json").write_text(json.dumps(t_old | timing, indent=1))


def analyse(args) -> None:
    from osgeo import gdal
    from analyse_la_forecast_error import SCALES_PX, fractions, fss

    work = Path(args.work)
    lc = gdal.Open(str(ROOT / "data" / "la" / "expo_park" / "Landcover.tif")).ReadAsArray()
    P = load_points()
    h15 = HOURS_KEPT.index(15)
    rep = {"days": {}, "scales_m": [n * 2.0 for n in SCALES_PX]}
    for d in days(args.start, args.end):
        ds = d.isoformat()
        f = {n: work / "utci" / f"{n}_{ds}.npy" for n in ("obs", "obsw", "anl", "d0", "d1", "d2")}
        if not all(p.exists() for p in f.values()):
            continue
        U = {n: np.load(p).astype(np.float32) for n, p in f.items()}
        valid = (lc[: U["obs"].shape[1], : U["obs"].shape[2]] != 2) & np.isfinite(U["obs"]).all(axis=0)
        forcing = {}
        for n in ("anl", "obsw", "d0", "d1", "d2"):
            met = np.loadtxt(OUT / "met" / f"met_{n}_{d:%Y%m%d}.txt", skiprows=1)
            forcing[n] = {"ta_15": float(met[15, 11]), "rh_15": float(met[15, 10]), "wind_15": float(met[15, 9]),
                          "kdown_15": float(met[15, 14]), "ta_max": float(met[:, 11].max())}
        met = np.loadtxt(OUT / "met" / f"met_obs_{d:%Y%m%d}.txt", skiprows=1)
        forcing["obs"] = {"ta_15": float(met[15, 11]), "rh_15": float(met[15, 10]), "wind_15": float(met[15, 9]),
                          "kdown_15": float(met[15, 14]), "ta_max": float(met[:, 11].max())}
        day = {"forcing": forcing, "truth": {}}
        for truth in ("obs", "obsw", "anl"):
            T = U[truth]
            tr = {}
            for n in ("anl", "d0", "d1", "d2"):
                if n == truth:
                    continue
                F = U[n]
                dd = (F[h15] - T[h15])[valid]
                rec = {"utci15_mean_error": float(dd.mean()), "utci15_abs_error_mean": float(np.abs(dd).mean()),
                       "utci15_error_sd_across_pixels": float(dd.std()),
                       "area32_15_forecast_pct": float((F[h15][valid] > 32).mean() * 100),
                       "area32_15_truth_pct": float((T[h15][valid] > 32).mean() * 100),
                       "area38_15_forecast_pct": float((F[h15][valid] > 38).mean() * 100),
                       "area38_15_truth_pct": float((T[h15][valid] > 38).mean() * 100),
                       "area46_15_forecast_pct": float((F[h15][valid] > 46).mean() * 100),
                       "area46_15_truth_pct": float((T[h15][valid] > 46).mean() * 100)}
                for thr in (32, 38):
                    ef, et = ((F > thr).sum(axis=0) >= 4) & valid, ((T > thr).sum(axis=0) >= 4) & valid
                    rec[f"event{thr}_4h_truth_frac"] = float(et[valid].mean())
                    rec[f"event{thr}_4h_hit_rate"] = float((ef & et)[valid].sum() / max(et[valid].sum(), 1))
                    rec[f"event{thr}_4h_false_alarm_ratio"] = float((ef & ~et)[valid].sum() / max(ef[valid].sum(), 1))
                q = lambda U15: (U15 >= np.nanpercentile(U15[valid], 90)) & valid  # noqa: E731
                tt, ff = q(T[h15]).astype(float), q(F[h15]).astype(float)
                rec["top10_fss"] = [fss(fractions(ff, valid, n_), fractions(tt, valid, n_), valid) for n_ in SCALES_PX]
                for name_, pct, hi in (("cool10", 10, False), ("top50", 50, True)):
                    sel = lambda U15: ((U15 >= np.nanpercentile(U15[valid], pct)) if hi else  # noqa: E731
                                       (U15 <= np.nanpercentile(U15[valid], pct))) & valid
                    tt_, ff_ = sel(T[h15]).astype(float), sel(F[h15]).astype(float)
                    rec[f"{name_}_fss"] = [fss(fractions(ff_, valid, n_), fractions(tt_, valid, n_), valid) for n_ in SCALES_PX]
                at, af = T[h15][valid], F[h15][valid]
                rec["anomaly_correlation_15"] = float(np.corrcoef(at - at.mean(), af - af.mean())[0, 1])
                tr[n] = rec
            day["truth"][truth] = tr
        rep["days"][ds] = day
        o = day["truth"]["obs"]
        print(f"[demo] {ds}: Ta15 obs {forcing['obs']['ta_15']:.1f}, anl {forcing['anl']['ta_15']:.1f}, d0 "
              f"{forcing['d0']['ta_15']:.1f}, d1 {forcing['d1']['ta_15']:.1f}, d2 {forcing['d2']['ta_15']:.1f} C | UTCI15 "
              f"error vs obs: anl {o['anl']['utci15_mean_error']:+.2f}, d0 {o['d0']['utci15_mean_error']:+.2f}, d1 "
              f"{o['d1']['utci15_mean_error']:+.2f}, d2 {o['d2']['utci15_mean_error']:+.2f}; top10 FSS 2 m d1 "
              f"{o['d1']['top10_fss'][0]:.3f}", flush=True)
    secs = json.loads((OUT / "run_seconds.json").read_text()) if (OUT / "run_seconds.json").exists() else {}
    rep["run_seconds_median"] = float(np.median(list(secs.values()))) if secs else None
    (OUT / "forecast_demo.json").write_text(json.dumps(rep, indent=1))
    print(f"[demo] wrote {OUT / 'forecast_demo.json'}", flush=True)
    figure(rep)


def figure(rep) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from solweig_sens import plotstyle

    plotstyle.use()
    ds = sorted(rep["days"])
    x = np.arange(len(ds))
    lab = [d[5:] for d in ds]
    cols = {"obs": "k", "anl": "0.55", "d0": "#4c78a8", "d1": "#0e6a78", "d2": "#c4501b"}
    names = {"obs": "station (KCQT)", "anl": "HRRR analysis", "d0": "HRRR 1-24 h ahead", "d1": "HRRR 13-36 h ahead",
             "d2": "HRRR 25-48 h ahead"}
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(16, 4.2))
    for n in ("obs", "anl", "d0", "d1", "d2"):
        a.plot(x, [rep["days"][d]["forcing"][n]["ta_15"] for d in ds], "o-", color=cols[n], ms=4, lw=1.3,
               label=names[n])
    a.set_ylabel("air temperature (°C)")
    for n in ("anl", "d0", "d1", "d2"):
        b.plot(x, [rep["days"][d]["truth"]["obs"][n]["utci15_mean_error"] for d in ds], "o-", color=cols[n], ms=4, lw=1.3,
               label=names[n])
        c.plot(x, [rep["days"][d]["truth"]["obs"][n]["area38_15_forecast_pct"] for d in ds], "o-", color=cols[n], ms=4,
               lw=1.3, label=names[n])
    c.plot(x, [rep["days"][d]["truth"]["obs"]["d0"]["area38_15_truth_pct"] for d in ds], "o-", color="k", ms=4, lw=1.5,
           label=names["obs"])
    if all("obsw" in rep["days"][d]["truth"] for d in ds):  # the station's air temperature and humidity, HRRR's wind
        b.plot(x, [rep["days"][d]["truth"]["obsw"]["d1"]["utci15_mean_error"] for d in ds], "s--", color=cols["d1"], ms=4,
               lw=1.1, mfc="white", label="HRRR 13-36 h ahead, against the station with HRRR's wind")
        c.plot(x, [rep["days"][d]["truth"]["obsw"]["d0"]["area38_15_truth_pct"] for d in ds], "s--", color="k", ms=4, lw=1.1,
               mfc="white", label="station with HRRR's wind")

    b.axhline(0, color="0.6", lw=0.8)
    b.set_ylabel("UTCI error, tile mean (°C)")
    c.set_ylabel("outdoor area with UTCI > 38 °C (%)")
    c.set_ylim(0, 100)
    for ax, letter in zip((a, b, c), "abc"):
        ax.set_xticks(x, lab, rotation=45, fontsize=8)
        ax.set_xlabel("day (2022)")
        ax.grid(alpha=0.3)
        ax.text(-0.12, 1.02, letter, transform=ax.transAxes, fontsize=13, fontweight="bold", va="bottom")
    handles, labels = [], []
    for ax in (a, b, c):  # one legend under the figure
        for hnd, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels:
                handles.append(hnd)
                labels.append(lab)
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(OUT / "forecast_demo.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[demo] wrote {OUT / 'forecast_demo.png'}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["hrrr", "run", "analyse"])
    ap.add_argument("--start", default="2022-08-30")
    ap.add_argument("--end", default="2022-09-10")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--tmp", default=str(Path(tempfile.gettempdir()) / "hrrr_demo"))
    ap.add_argument("--work", default=str(Path.home() / "forecast_demo"))
    ap.add_argument("--pre", default=str(Path.home() / "case_study" / "expo" / "base" / "shared" / "processed_inputs"),
                    help="the venue tile's preprocessed inputs with its sky view cache")
    args = ap.parse_args()
    {"hrrr": hrrr, "run": run, "analyse": analyse}[args.cmd](args)


if __name__ == "__main__":
    main()
