"""HRRR 2 m temperature forecast errors over the Los Angeles basin: size by lead time, and how far they reach.

The forecast ensembles (run_la_forecast_error.py, shortcut_suite.py) assume an air-temperature error of 0.5-3 K,
the same over the whole 2 x 2 km tile. This measures what HRRR's errors actually are at the stations around our
two tiles, for the hour we analyse (15:00 PDT, 22 UTC), and how similar they are between stations a given
distance apart: if errors stay correlated over tens of km, one error per tile is a fair model of HRRR; if not,
the error varies inside a tile.

Forecasts: HRRR v4 (operational since Dec 2020; 48 h runs at 00/06/12/18 UTC), 2 m temperature, summers (Jun-Sep)
2021-2023, valid at 22 UTC, at leads 0, 1, 2 h (hourly runs) and 4, 10, ..., 46 h (the 6-hourly 48 h runs), from
the NOAA archive on AWS via Herbie. Value at the grid point nearest each station (3 km grid; no height correction).
Each GRIB message is read and deleted at once. Observations: ASOS reports from the Iowa Environmental Mesonet,
the one closest to the valid time within 20 min.

Outputs (figures/hrrr/): hrrr_points.csv (forecast values; the download step resumes from it), hrrr_errors.json
(bias, error standard deviation and RMSE by lead, overall and per station; error correlation between station
pairs against distance, with an exponential fit), hrrr_errors.png.

Usage (env with herbie-data, cfgrib, xarray, scipy, matplotlib):
    python scripts/hrrr_errors.py download --threads 16
    python scripts/hrrr_errors.py analyse
"""

from __future__ import annotations

import sys as _sys

if __name__ == "__main__" and any(a in ("-h", "--help") for a in _sys.argv[1:]):
    print(__doc__)
    raise SystemExit(0)

import argparse
import csv
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "hrrr"
STATIONS = ["VNY", "CQT", "BUR", "LAX", "SMO", "HHR", "LGB", "TOA", "EMT", "WHP", "POC", "FUL", "SNA", "CCB", "ONT",
            "CNO", "RAL", "CMA", "OXR"]  # Los Angeles basin and surroundings; VNY next to Pacoima, CQT next to the venue
YEARS = (2021, 2022, 2023)
MONTHS = (6, 7, 8, 9)
VALID_UTC = 22
LEADS = (0, 1, 2, 4, 10, 16, 22, 28, 34, 40, 46)


def station_meta() -> dict:
    url = "https://mesonet.agron.iastate.edu/geojson/network/CA_ASOS.geojson"
    feats = json.loads(urllib.request.urlopen(url, timeout=60).read())["features"]
    meta = {f["properties"]["sid"]: {"name": f["properties"]["sname"], "lon": f["geometry"]["coordinates"][0],
                                     "lat": f["geometry"]["coordinates"][1], "elev_m": f["properties"].get("elevation")}
            for f in feats if f["properties"]["sid"] in STATIONS}
    missing = set(STATIONS) - set(meta)
    if missing:
        print(f"[hrrr] no metadata for {sorted(missing)}; left out", flush=True)
    return meta


def valid_times() -> list[datetime]:
    out = []
    for y in YEARS:
        d = datetime(y, MONTHS[0], 1, VALID_UTC, tzinfo=timezone.utc)
        while d.month in MONTHS:
            out.append(d)
            d += timedelta(days=1)
    return out


def nearest_indices(lat2d, lon2d, meta) -> dict:
    """Grid index nearest each station (on the sphere)."""
    from scipy.spatial import cKDTree

    def xyz(lat, lon):
        la, lo = np.radians(lat), np.radians(lon)
        return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)

    lon2d = np.where(lon2d > 180, lon2d - 360, lon2d)
    tree = cKDTree(xyz(lat2d, lon2d).reshape(-1, 3))
    idx = {}
    for s, m in meta.items():
        dist, k = tree.query(xyz(m["lat"], m["lon"]))
        idx[s] = (int(k // lat2d.shape[1]), int(k % lat2d.shape[1]), float(dist * 6371.0))
    return idx


def download(args) -> None:
    from herbie import Herbie

    OUT.mkdir(parents=True, exist_ok=True)
    meta = station_meta()
    (OUT / "stations.json").write_text(json.dumps(meta, indent=2))
    pts = OUT / "hrrr_points.csv"
    done = set()
    if pts.exists():
        with open(pts) as f:
            done = {(r["valid"], int(r["lead"])) for r in csv.DictReader(f)}
    jobs = [(v, L) for v in valid_times() for L in LEADS if (v.strftime("%Y-%m-%dT%H:%M"), L) not in done]
    print(f"[hrrr] {len(jobs)} fields to fetch ({len(done)} already done)", flush=True)
    tmp_root = Path(args.tmp)
    tmp_root.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    read_lock = threading.Lock()  # eccodes is not thread-safe: downloads run in parallel, reads one at a time
    state = {"idx": None}
    new = not pts.exists()
    fout = open(pts, "a", newline="")
    w = csv.writer(fout)
    if new:
        w.writerow(["valid", "lead", "station", "t2m_c"])

    def one(v, L):
        d = Path(tempfile.mkdtemp(dir=tmp_root))
        try:
            H = Herbie((v - timedelta(hours=L)).strftime("%Y-%m-%d %H:%M"), model="hrrr", product="sfc", fxx=L,
                       save_dir=str(d), verbose=False)
            H.download(":TMP:2 m above")
            with read_lock:
                ds = H.xarray(":TMP:2 m above", remove_grib=True)
                t = ds.t2m.values
                if state["idx"] is None:
                    state["idx"] = nearest_indices(ds.latitude.values, ds.longitude.values, meta)
                    (OUT / "station_gridpoints.json").write_text(json.dumps(state["idx"], indent=2))
                ds.close()
            vals = {s: float(t[i, j]) - 273.15 for s, (i, j, _) in state["idx"].items()}
            return v, L, vals
        except Exception as e:  # missing in the archive, or a failed read: recorded as absent, not retried
            return v, L, repr(e)[:120]
        finally:
            shutil.rmtree(d, ignore_errors=True)

    n_ok = n_bad = 0
    with ThreadPoolExecutor(args.threads) as ex:
        futs = [ex.submit(one, v, L) for v, L in jobs]
        for k, fu in enumerate(as_completed(futs)):
            v, L, vals = fu.result()
            with lock:
                if isinstance(vals, dict):
                    for s, x in vals.items():
                        w.writerow([v.strftime("%Y-%m-%dT%H:%M"), L, s, f"{x:.3f}"])
                    n_ok += 1
                else:
                    w.writerow([v.strftime("%Y-%m-%dT%H:%M"), L, "MISSING", vals])
                    n_bad += 1
                if k % 100 == 0:
                    fout.flush()
                    print(f"[hrrr] {k + 1}/{len(jobs)} fields, {n_bad} missing", flush=True)
    fout.close()
    print(f"[hrrr] done: {n_ok} fields read, {n_bad} missing", flush=True)


def observations(meta) -> dict:
    """{(station, valid iso): temperature C}, the report nearest each valid time within 20 min."""
    qs = "&".join(f"station={s}" for s in meta)
    y0, y1 = YEARS[0], YEARS[-1]
    url = ("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?" + qs +
           f"&data=tmpf&year1={y0}&month1={MONTHS[0]}&day1=1&year2={y1}&month2={MONTHS[-1] + 1}&day2=2"
           "&tz=Etc/UTC&format=onlycomma&missing=M&trace=T&report_type=3&report_type=4")
    cache = OUT / "asos_tmpf.csv"
    if not cache.exists():
        cache.write_bytes(urllib.request.urlopen(url, timeout=600).read())
    want = {v.strftime("%Y-%m-%dT%H:%M") for v in valid_times()}
    best = {}
    for r in csv.DictReader(io.StringIO(cache.read_text())):
        if r["tmpf"] in ("M", ""):
            continue
        t = datetime.strptime(r["valid"], "%Y-%m-%d %H:%M")
        for cand in (t.replace(minute=0), t.replace(minute=0) + timedelta(hours=1)):
            key = cand.strftime("%Y-%m-%dT%H:%M")
            dt = abs((t - cand).total_seconds()) / 60
            if key in want and dt <= 20 and dt < best.get((r["station"], key), (99, 0))[0]:
                best[(r["station"], key)] = (dt, (float(r["tmpf"]) - 32) / 1.8)
    return {k: v[1] for k, v in best.items()}


def km(a, b):
    la1, lo1, la2, lo2 = map(np.radians, (a["lat"], a["lon"], b["lat"], b["lon"]))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return float(2 * 6371 * np.arcsin(np.sqrt(h)))


def analyse(args) -> None:
    from scipy.optimize import curve_fit

    meta = json.loads((OUT / "stations.json").read_text())
    obs = observations(meta)
    fc = {}
    with open(OUT / "hrrr_points.csv") as f:
        for r in csv.DictReader(f):
            if r["station"] != "MISSING":
                fc[(r["station"], r["valid"], int(r["lead"]))] = float(r["t2m_c"])
    stations = sorted(meta)
    valids = sorted({v.strftime("%Y-%m-%dT%H:%M") for v in valid_times()})
    E = np.full((len(LEADS), len(valids), len(stations)), np.nan)  # forecast minus observed
    for a, L in enumerate(LEADS):
        for b, v in enumerate(valids):
            for c, s in enumerate(stations):
                if (s, v, L) in fc and (s, v) in obs:
                    E[a, b, c] = fc[(s, v, L)] - obs[(s, v)]
    rep = {"valid_utc": VALID_UTC, "years": YEARS, "months": MONTHS, "leads": LEADS, "stations": meta,
           "n_valid_times": len(valids), "by_lead": {}, "by_station": {}, "correlation": {}}
    for a, L in enumerate(LEADS):
        e = E[a]
        bias_s = np.nanmean(e, axis=0)
        rep["by_lead"][L] = {"n": int(np.isfinite(e).sum()), "bias": float(np.nanmean(e)),
                             "rmse": float(np.sqrt(np.nanmean(e ** 2))),
                             "sd_after_station_bias": float(np.nanstd(e - bias_s)),
                             "rmse_VNY": float(np.sqrt(np.nanmean(e[:, stations.index("VNY")] ** 2))) if "VNY" in stations else None,
                             "rmse_CQT": float(np.sqrt(np.nanmean(e[:, stations.index("CQT")] ** 2))) if "CQT" in stations else None}
    for c, s in enumerate(stations):
        rep["by_station"][s] = {str(L): {"bias": float(np.nanmean(E[a, :, c])), "sd": float(np.nanstd(E[a, :, c])),
                                         "n": int(np.isfinite(E[a, :, c]).sum())} for a, L in enumerate(LEADS)}
    # how similar errors are at two stations, against their distance (each station's own mean bias removed first)
    def expo(d, c0, L):
        return c0 * np.exp(-d / L)

    for a, L in enumerate(LEADS):
        e = E[a] - np.nanmean(E[a], axis=0)
        pairs = []
        for i in range(len(stations)):
            for j in range(i + 1, len(stations)):
                ok = np.isfinite(e[:, i]) & np.isfinite(e[:, j])
                if ok.sum() >= 60:
                    pairs.append((km(meta[stations[i]], meta[stations[j]]), float(np.corrcoef(e[ok, i], e[ok, j])[0, 1]),
                                  stations[i], stations[j], int(ok.sum())))
        d, r = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
        try:
            (c0, Lk), _ = curve_fit(expo, d, r, p0=(0.8, 50.0), bounds=((0, 1), (1.5, 2000)))
        except Exception:
            c0, Lk = np.nan, np.nan
        rep["correlation"][L] = {"fit_c0": float(c0), "fit_length_km": float(Lk), "pairs": pairs,
                                 "corr_within_15km_mean": float(np.mean(r[d < 15])) if (d < 15).any() else None}
    (OUT / "hrrr_errors.json").write_text(json.dumps(rep, indent=1))
    for L in LEADS:
        b, c = rep["by_lead"][L], rep["correlation"][L]
        print(f"[hrrr] lead {L:2d} h: n {b['n']:5d}, bias {b['bias']:+.2f} K, RMSE {b['rmse']:.2f} K, sd after station bias "
              f"{b['sd_after_station_bias']:.2f} K, VNY {b['rmse_VNY']:.2f}, CQT {b['rmse_CQT']:.2f}; error correlation "
              f"c0 {c['fit_c0']:.2f}, length {c['fit_length_km']:.0f} km, pairs < 15 km {c['corr_within_15km_mean']}", flush=True)
    figure(rep)


def figure(rep) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys_path = str(ROOT / "src")
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    from solweig_sens import plotstyle

    plotstyle.use()
    leads = [int(L) for L in rep["by_lead"]]
    g = lambda k: [rep["by_lead"][L][k] for L in rep["by_lead"]]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    a.plot(leads, g("rmse"), "o-", color="#0e6a78", label="RMSE, all stations")
    a.plot(leads, g("sd_after_station_bias"), "s--", color="#0e6a78", mfc="white", label="standard deviation, each station's bias removed")
    a.plot(leads, g("bias"), "^:", color="#c4501b", label="bias")
    a.plot(leads, g("rmse_VNY"), "-", color="0.55", lw=1, label="RMSE, Van Nuys (KVNY)")
    a.plot(leads, g("rmse_CQT"), "--", color="0.55", lw=1, label="RMSE, Downtown USC (KCQT)")
    a.axhline(0, color="0.6", lw=0.6)
    a.set_xlabel("lead time (h)")
    a.set_ylabel("2 m temperature error (K)")
    a.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    for L, col in ((4, "#4c78a8"), (22, "#c4501b"), (46, "#0e6a78")):
        c = rep["correlation"][L] if L in rep["correlation"] else rep["correlation"][str(L)]
        d, r = np.array([p[0] for p in c["pairs"]]), np.array([p[1] for p in c["pairs"]])
        b.scatter(d, r, s=10, color=col, alpha=0.6, label=f"{L} h lead")
        x = np.linspace(0, d.max(), 100)
        if np.isfinite(c["fit_length_km"]):
            b.plot(x, c["fit_c0"] * np.exp(-x / c["fit_length_km"]), color=col, lw=1.2)
    b.set_xlabel("distance between stations (km)")
    b.set_ylabel("correlation of their errors")
    b.set_ylim(-0.2, 1)
    b.legend(frameon=False, fontsize=8)
    for ax, letter in ((a, "a"), (b, "b")):
        ax.text(-0.12, 1.02, letter, transform=ax.transAxes, fontsize=13, fontweight="bold", va="bottom")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "hrrr_errors.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[hrrr] wrote {OUT / 'hrrr_errors.png'}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--threads", type=int, default=16)
    d.add_argument("--tmp", default=str(Path(tempfile.gettempdir()) / "hrrr_dl"), help="where GRIB messages land briefly")
    sub.add_parser("analyse")
    args = ap.parse_args()
    {"download": download, "analyse": analyse}[args.cmd](args)


if __name__ == "__main__":
    main()
