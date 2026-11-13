"""Air temperature below HRRR's 3 km grid in Los Angeles: how large it is, how far it reaches, what explains it.

hrrr_errors.py found about 1 K of HRRR's error that is local to each airport station, at every lead time, and
spatial_error_fss.py had to guess how far such a local error stays correlated (250 m to 3 km tested). The airport
network is too sparse to answer either. This uses every station with air temperature in the public MADIS archive
(about 290 in the basin: MesoWest networks, citizen stations of the CWOP network, fire-weather RAWS) and compares
each with HRRR at the same place and hour.

Reference: the HRRR 1 h forecast valid at the hour (2 m temperature), bilinear at the station on the 3 km grid,
moved to the station's height with -6.5 K/km at 15:00; at 04:00 not moved, since night inversions make the
standard correction wrong there (with it, HRRR's error at the stations grows from 1.85 to 2.08 K; --lapse sets one
value for both hours, as a check). The 1 h forecast rather than the analysis, because the analysis
assimilates some of these stations. Observations: the report nearest the hour within its first 10 minutes,
MADIS quality flags X and B dropped.

The difference station minus HRRR is split three ways (per hour of day):
    r(s, t) = c(t) + m(s) + e(s, t)
c is the basin-wide error at time t (what a tile-uniform error model already covers), m the station's lasting
offset (siting, surroundings, and HRRR's fixed patterns), e what is left (day-to-day local error). The script
reports their sizes, how m and e vary with the distance between stations (semivariogram of m, correlation of e),
pairs closer than the 3 km grid, and how much of m the land cover around each station explains (ESA WorldCover
10 m fractions within 100 and 500 m, height above the HRRR terrain, distance to the ocean) under spatial
cross-validation.

Limits: stations sit where people put them (gardens, roofs, fire stations), not in streets; the closest pairs are
a few hundred metres apart, so nothing here resolves street scale (tens of metres); citizen stations can sit in sun
or on walls. The numbers are an upper bound on what a downscaler could explain at station scale.

Outputs (figures/street_ta/): street_ta.json, street_ta.png, stations.csv (per station: offsets, predictors; kept
out of git, since it gives the positions of citizen stations, mostly private homes).
Raw data in ~/street_ta (not in git): madis_obs.csv, hrrr/*.npy, worldcover window.

Usage:
    python scripts/street_ta_structure.py madis --threads 8      (env with scipy)
    python scripts/street_ta_structure.py hrrr --threads 8       (env with herbie-data, eccodes, pyproj)
    python scripts/street_ta_structure.py predictors             (env with GDAL)
    python scripts/street_ta_structure.py analyse                (env with scikit-learn, pyproj, matplotlib)
    python scripts/street_ta_structure.py baseline               (same env; writes street_ta_baseline.json)
    python scripts/street_ta_structure.py baseline --lapse 0 --tag _nolapse      (check: no height correction)
    python scripts/street_ta_structure.py plot_baseline
"""

from __future__ import annotations

import sys as _sys

if __name__ == "__main__" and any(a in ("-h", "--help") for a in _sys.argv[1:]):
    print(__doc__)
    raise SystemExit(0)

import argparse
import csv
import gzip
import io
import json
import shutil
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "street_ta"
RAW = Path.home() / "street_ta"
LAT = (33.60, 34.45)
LON = (-118.75, -117.85)
HOURS_UTC = (22, 11)  # 15:00 and 04:00 PDT
NAMES = {22: "15:00 PDT", 11: "04:00 PDT"}
MADIS = "https://madis-data.ncep.noaa.gov/madisPublic1/data/archive/{d:%Y/%m/%d}/LDAD/mesonet/netCDF/{d:%Y%m%d}_{h:02d}00.gz"
WORLDCOVER = ("/vsicurl/https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
              "ESA_WorldCover_10m_2021_v200_N33W120_Map.tif")
LAPSE = {h: (-0.0065 if 9 <= (h - 7) % 24 <= 20 else 0.0) for h in range(24)}  # K per m by UTC hour: standard
# from 09 to 20 PDT (mixed layer), none at night (inversions); 22 UTC = 15 PDT gets -6.5, 11 UTC = 04 PDT gets 0
TAG = ""
TILES = {"Pacoima": (34.262, -118.418), "venue": (34.014, -118.288)}


def valid_times(years, months=(6, 7, 8, 9), hours=HOURS_UTC) -> list[datetime]:
    out = []
    for y in years:
        d = date(y, months[0], 1)
        while d.month in months and d.year == y:
            for h in hours:
                out.append(datetime(d.year, d.month, d.day, h, tzinfo=timezone.utc))
            d += timedelta(days=1)
    return out


# ---------------------------------------------------------------------------------------------------- observations

def _chars(a) -> list[str]:
    return [b"".join(r).decode(errors="ignore").strip("\x00 ") for r in a]


def madis_hour(v: datetime) -> list[list] | str:
    """Reports in the box within 10 min after v, the one nearest v per station."""
    from scipy.io import netcdf_file

    url = MADIS.format(d=v, h=v.hour)
    for attempt in range(3):
        try:
            raw = urllib.request.urlopen(url, timeout=120).read()
            break
        except Exception as e:  # transient network errors: retry twice
            if attempt == 2:
                return repr(e)[:120]
            time.sleep(5)
    nc = netcdf_file(io.BytesIO(gzip.decompress(raw)), mmap=False)
    V = nc.variables
    lat = V["latitude"][:].astype(float)
    lon = V["longitude"][:].astype(float)
    T = V["temperature"][:].astype(float)
    tobs = V["observationTime"][:].astype(float)
    t0 = v.timestamp()
    sel = np.where((lat > LAT[0]) & (lat < LAT[1]) & (lon > LON[0]) & (lon < LON[1]) & (T > 250) & (T < 330)
                   & (tobs >= t0) & (tobs <= t0 + 600))[0]
    if len(sel) == 0:
        return "no reports"
    sid = _chars(V["stationId"][:][sel])
    prov = _chars(V["dataProvider"][:][sel])
    dd = [x.decode() if isinstance(x, bytes) else str(x) for x in V["temperatureDD"][:][sel]]
    elev = V["elevation"][:].astype(float)[sel]
    best = {}
    for k, i in enumerate(sel):
        if dd[k] in ("X", "B"):
            continue
        dt = tobs[i] - t0
        if sid[k] not in best or dt < best[sid[k]][0]:
            best[sid[k]] = (dt, [v.strftime("%Y-%m-%dT%H"), sid[k], prov[k], f"{lat[i]:.5f}", f"{lon[i]:.5f}",
                                 f"{elev[k]:.1f}", f"{T[i] - 273.15:.2f}", dd[k], int(dt)])
    nc.close()
    return [b[1] for b in best.values()]


def madis(args) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / "madis_obs.csv"
    done = set()
    if path.exists():
        with open(path) as f:
            done = {r["valid"] for r in csv.DictReader(f)}
    jobs = [v for v in valid_times(args.years, hours=args.hours) if v.strftime("%Y-%m-%dT%H") not in done]
    print(f"[madis] {len(jobs)} hours to fetch ({len(done)} done)", flush=True)
    new = not path.exists()
    with open(path, "a", newline="") as f, ThreadPoolExecutor(args.threads) as ex:
        w = csv.writer(f)
        if new:
            w.writerow(["valid", "station", "provider", "lat", "lon", "elev", "t_c", "dd", "dt_s"])
        futs = {ex.submit(madis_hour, v): v for v in jobs}
        for k, fu in enumerate(as_completed(futs)):
            rows = fu.result()
            if isinstance(rows, str):
                w.writerow([futs[fu].strftime("%Y-%m-%dT%H"), "MISSING", rows] + [""] * 6)
            else:
                w.writerows(rows)
            if k % 20 == 0:
                f.flush()
                print(f"[madis] {k + 1}/{len(jobs)}", flush=True)
    print("[madis] done", flush=True)


# ---------------------------------------------------------------------------------------------------- HRRR

def _crop_index(lat2d, lon2d):
    lon2d = np.where(lon2d > 180, lon2d - 360, lon2d)
    m = (lat2d > LAT[0] - 0.1) & (lat2d < LAT[1] + 0.1) & (lon2d > LON[0] - 0.1) & (lon2d < LON[1] + 0.1)
    ii, jj = np.where(m)
    return ii.min(), ii.max() + 1, jj.min(), jj.max() + 1


def read_crop(path: str, want) -> dict:
    """Crop of one GRIB field (shortName in want) over the box, with its lat/lon, via eccodes."""
    want = (want,) if isinstance(want, str) else tuple(want)
    import eccodes

    with open(path, "rb") as f:
        while True:
            gid = eccodes.codes_grib_new_from_file(f)
            if gid is None:
                break
            try:
                if eccodes.codes_get(gid, "shortName") not in want:
                    continue
                ny, nx = eccodes.codes_get(gid, "Ny"), eccodes.codes_get(gid, "Nx")
                val = eccodes.codes_get_values(gid).reshape(ny, nx)
                lat = eccodes.codes_get_array(gid, "latitudes").reshape(ny, nx)
                lon = eccodes.codes_get_array(gid, "longitudes").reshape(ny, nx)
                i0, i1, j0, j1 = _crop_index(lat, lon)
                return {"val": val[i0:i1, j0:j1].astype(np.float32), "lat": lat[i0:i1, j0:j1],
                        "lon": lon[i0:i1, j0:j1], "ij": (int(i0), int(j0))}
            finally:
                eccodes.codes_release(gid)
    raise RuntimeError(f"{want} not in {path}")


def hrrr(args) -> None:
    from concurrent.futures import ProcessPoolExecutor
    from herbie import Herbie

    out = RAW / "hrrr"
    out.mkdir(parents=True, exist_ok=True)
    tmp_root = RAW / "tmp"
    tmp_root.mkdir(exist_ok=True)
    if not (RAW / "hrrr_grid.npz").exists():  # terrain height and coordinates, once
        d = Path(tempfile.mkdtemp(dir=tmp_root))
        H = Herbie("2022-07-01 00:00", model="hrrr", product="sfc", fxx=0, save_dir=str(d), verbose=False)
        g = read_crop(str(H.download(":HGT:surface")), ("orog", "h", "gh"))
        np.savez(RAW / "hrrr_grid.npz", hgt=g["val"], lat=g["lat"], lon=g["lon"], ij=np.array(g["ij"]))
        shutil.rmtree(d, ignore_errors=True)
    jobs = [v for v in valid_times(args.years, hours=args.hours) if not (out / f"{v:%Y%m%d%H}.npy").exists()]
    print(f"[hrrr] {len(jobs)} fields to fetch", flush=True)

    def fetch(v):
        d = Path(tempfile.mkdtemp(dir=tmp_root))
        try:
            H = Herbie((v - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M"), model="hrrr", product="sfc", fxx=1,
                       save_dir=str(d), verbose=False)
            return v, str(H.download(":TMP:2 m above")), d
        except Exception as e:
            return v, repr(e)[:150], d

    n_bad = 0
    with ThreadPoolExecutor(args.threads) as tex, ProcessPoolExecutor(4) as pex:
        reads = {}
        for k, fu in enumerate(as_completed([tex.submit(fetch, v) for v in jobs])):
            v, path, d = fu.result()
            if Path(path).exists():
                reads[pex.submit(read_crop, path, "2t")] = (v, d)
            else:
                n_bad += 1
                shutil.rmtree(d, ignore_errors=True)
            for r in [r for r in reads if r.done()]:
                v2, d2 = reads.pop(r)
                try:
                    np.save(out / f"{v2:%Y%m%d%H}.npy", r.result()["val"] - 273.15)
                except Exception:
                    n_bad += 1
                shutil.rmtree(d2, ignore_errors=True)
            if k % 50 == 0:
                print(f"[hrrr] {k + 1}/{len(jobs)}, {n_bad} missing", flush=True)
        for r in as_completed(list(reads)):
            v2, d2 = reads.pop(r)
            try:
                np.save(out / f"{v2:%Y%m%d%H}.npy", r.result()["val"] - 273.15)
            except Exception:
                n_bad += 1
            shutil.rmtree(d2, ignore_errors=True)
    print(f"[hrrr] done, {n_bad} missing", flush=True)


# ---------------------------------------------------------------------------------------------------- predictors

def station_table() -> dict:
    """Median position and height per station over all its reports."""
    pos = {}
    with open(RAW / "madis_obs.csv") as f:
        for r in csv.DictReader(f):
            if r["station"] == "MISSING":
                continue
            pos.setdefault(r["station"], {"provider": r["provider"], "p": []})["p"].append(
                (float(r["lat"]), float(r["lon"]), float(r["elev"])))
    out = {}
    for s, d in pos.items():
        p = np.array(d["p"])
        out[s] = {"provider": d["provider"], "lat": float(np.median(p[:, 0])), "lon": float(np.median(p[:, 1])),
                  "elev": float(np.median(p[:, 2])), "moved_m": float(np.ptp(p[:, 0]) * 111e3 + np.ptp(p[:, 1]) * 92e3)}
    return out


def predictors(args) -> None:
    from osgeo import gdal
    from scipy import ndimage

    st = station_table()
    win = RAW / "worldcover.npy"
    ds = gdal.Open(WORLDCOVER)
    gt = ds.GetGeoTransform()  # 1/12000 degree pixels
    c0, r0 = int((LON[0] - gt[0]) / gt[1]), int((LAT[1] - gt[3]) / gt[5])
    c1, r1 = int((LON[1] - gt[0]) / gt[1]), int((LAT[0] - gt[3]) / gt[5])
    if win.exists():
        lc = np.load(win)
    else:
        print(f"[pred] reading WorldCover window {c1 - c0} x {r1 - r0}", flush=True)
        lc = ds.GetRasterBand(1).ReadAsArray(c0, r0, c1 - c0, r1 - r0)
        np.save(win, lc)
    lat_of = lambda r: gt[3] + (r0 + r + 0.5) * gt[5]
    lon_of = lambda c: gt[0] + (c0 + c + 0.5) * gt[1]
    # ocean: the largest water body in a 100 m version of the window; distance to it in km
    w100 = (lc[: lc.shape[0] // 10 * 10, : lc.shape[1] // 10 * 10].reshape(lc.shape[0] // 10, 10, -1, 10) == 80).mean(axis=(1, 3)) > 0.5
    lab, n = ndimage.label(w100)
    ocean = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1) if n else np.zeros_like(w100)
    cell = (10 * abs(gt[5]) * 111.2, 10 * gt[1] * 111.2 * np.cos(np.radians(34.0)))  # km per 10-pixel cell
    dist_km = ndimage.distance_transform_edt(~ocean, sampling=cell)
    rows = []
    mdeg_lat, mdeg_lon = 111.2e3, 111.2e3 * np.cos(np.radians(34.0))
    yy, xx = np.mgrid[-60:61, -60:61]
    for s, m in st.items():
        r = int((m["lat"] - lat_of(0)) / gt[5])
        c = int((m["lon"] - lon_of(0)) / gt[1])
        if not (60 <= r < lc.shape[0] - 60 and 60 <= c < lc.shape[1] - 60):
            continue
        patch = lc[r - 60:r + 61, c - 60:c + 61]
        dm = np.hypot(yy * abs(gt[5]) * mdeg_lat, xx * gt[1] * mdeg_lon)
        rec = {"station": s, **m}
        for rad in (100, 500):
            p = patch[dm <= rad]
            rec[f"tree_{rad}"] = float((p == 10).mean())
            rec[f"built_{rad}"] = float((p == 50).mean())
            rec[f"green_{rad}"] = float(np.isin(p, (20, 30, 40)).mean())
            rec[f"bare_{rad}"] = float((p == 60).mean())
            rec[f"water_{rad}"] = float((p == 80).mean())
        rec["coast_km"] = float(dist_km[min(r // 10, dist_km.shape[0] - 1), min(c // 10, dist_km.shape[1] - 1)])
        rows.append(rec)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(RAW / "predictors.json", "w") as f:
        json.dump(rows, f)
    print(f"[pred] {len(rows)} stations with predictors", flush=True)


# ---------------------------------------------------------------------------------------------------- analysis

def km(lat1, lon1, lat2, lon2):
    la1, la2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def hrrr_at(stations: dict):
    """Bilinear weights for each station on the cropped HRRR grid (in HRRR's own projection), plus terrain height."""
    from pyproj import Proj
    from scipy.interpolate import RegularGridInterpolator

    g = np.load(RAW / "hrrr_grid.npz")
    P = Proj(proj="lcc", lat_1=38.5, lat_2=38.5, lat_0=38.5, lon_0=-97.5, R=6371229)
    lon = np.where(g["lon"] > 180, g["lon"] - 360, g["lon"])
    X, Y = P(lon, g["lat"])
    x, y = X[0], Y[:, 0]
    assert np.allclose(X, x[None, :], atol=50) and np.allclose(Y, y[:, None], atol=50), "grid not regular"
    names = list(stations)
    sx, sy = P(np.array([stations[s]["lon"] for s in names]), np.array([stations[s]["lat"] for s in names]))
    pts = np.stack([sy, sx], axis=1)
    hgt = RegularGridInterpolator((y, x), g["hgt"])(pts)
    return names, (y, x), pts, hgt


def fit_exp(d, rho, w):
    """rho(d) = c0 exp(-d / L), weighted least squares on a grid of L."""
    best = None
    for L in np.geomspace(0.1, 300, 400):
        f = np.exp(-d / L)
        c0 = float(np.sum(w * f * rho) / np.sum(w * f * f))
        err = float(np.sum(w * (rho - c0 * f) ** 2))
        if best is None or err < best[0]:
            best = (err, c0, L)
    return {"c0": round(best[1], 3), "length_km": round(float(best[2]), 2)}


def load() -> dict:
    """Station minus HRRR (1 h forecast, bilinear, height-corrected), as a times x stations array."""
    from scipy.interpolate import RegularGridInterpolator

    st = station_table()
    names, (gy, gx), pts, z_h = hrrr_at(st)
    idx = {s: k for k, s in enumerate(names)}
    z_s = np.array([st[s]["elev"] for s in names])
    obs = {}
    with open(RAW / "madis_obs.csv") as f:
        for r in csv.DictReader(f):
            if r["station"] != "MISSING":
                obs.setdefault(r["valid"], []).append((idx[r["station"]], float(r["t_c"])))
    times = sorted(t for t in obs if (RAW / "hrrr" / f"{t.replace('-', '').replace('T', '')}.npy").exists())
    R = np.full((len(times), len(names)), np.nan)
    TH = np.full((len(times), len(names)), np.nan)
    for k, t in enumerate(times):
        field = np.load(RAW / "hrrr" / f"{t.replace('-', '').replace('T', '')}.npy")
        th = RegularGridInterpolator((gy, gx), field)(pts) + LAPSE[int(t[-2:])] * (z_s - z_h)
        TH[k] = th
        for i, tc in obs[t]:
            R[k, i] = tc - th[i]
    keep = (np.abs(z_s - z_h) < 400) & np.array([st[s]["moved_m"] < 200 for s in names])
    return {"names": names, "st": st, "z_s": z_s, "z_h": z_h, "R": R, "TH": TH, "times": times,
            "hour": np.array([int(t[-2:]) for t in times]), "keep": keep}


def decompose(Rh):
    """r(s, t) = c(t) + m(s) + e(s, t) by alternating means (the panel is unbalanced); m has mean zero."""
    m = np.zeros(Rh.shape[1])
    for _ in range(8):
        c = np.nanmean(Rh - m[None, :], axis=1)
        m = np.nanmean(Rh - c[:, None], axis=0)
        m = np.where(np.isfinite(m), m - np.nanmean(m), np.nan)
    c = np.nanmean(Rh - m[None, :], axis=1)
    return c, m, Rh - c[:, None] - m[None, :]


def clean(Rh, keep):
    """Quality control in the spirit of CrowdQC+: stations that moved or sit far off the HRRR terrain, gross
    errors (more than 5 robust SD from the other stations at that time), stations with under 40 reports, and
    stations whose day-to-day local error is over 3 times the typical one (sun on the sensor, faults)."""
    Rh = Rh.copy()
    Rh[:, ~keep] = np.nan
    qc = {"stations_bad_height_or_moved": int((~keep).sum())}
    med = np.nanmedian(Rh, axis=1, keepdims=True)
    mad = 1.4826 * np.nanmedian(np.abs(Rh - med), axis=1, keepdims=True)
    gross = np.abs(Rh - med) > 5 * mad
    qc["reports_gross"] = int(np.nansum(gross))
    Rh[gross] = np.nan
    n_obs = np.sum(np.isfinite(Rh), axis=0)
    Rh[:, n_obs < 40] = np.nan
    qc["stations_under_40_reports"] = int(((n_obs < 40) & keep).sum())
    _, _, e = decompose(Rh)
    sd_e = np.nanstd(e, axis=0)
    noisy = sd_e > 3 * np.nanmedian(sd_e)
    qc["stations_noisy"] = int(np.sum(noisy & np.isfinite(sd_e)))
    Rh[:, noisy] = np.nan
    qc["stations_used"] = int(np.any(np.isfinite(Rh), axis=0).sum())
    qc["reports_used"] = int(np.isfinite(Rh).sum())
    return Rh, qc


FEATS = ["tree_100", "built_100", "green_100", "bare_100", "water_100", "tree_500", "built_500", "green_500",
         "bare_500", "water_500", "coast_km"]


def baseline(args) -> None:
    """A first station-scale air-temperature layer on top of HRRR, scored on stations held out by 5 km blocks.

    Levels (all fitted on the training stations of the fold only):
      hrrr      HRRR 1 h forecast as is (height-corrected)
      static    + the mean bias and the station's lasting offset predicted from its surroundings, both learned
                  offline: what a forecast can use, since it has no live reports at the valid time
    and, for a nowcast with live reports, each adding to the one before:
      common    + the basin-wide error at that hour, from the training stations' reports (sparse live stations)
      offset    + the station's lasting offset predicted from its surroundings (ridge on FEATS + heights)
      kriged    + the local error interpolated from training stations at that hour (simple kriging with the
                  exponential correlation fitted on training pairs)
    Intervals: split conformal, calibrated on the training stations' own out-of-block errors (a nested block
    split), 90% target; coverage then counted on the held-out stations, overall, on the hottest tenth of days,
    and by network. Exchangeability between stations is only approximate, so the coverage is checked, not assumed.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    D = load()
    names, st, z_s, z_h = D["names"], D["st"], D["z_s"], D["z_h"]
    pred = {p["station"]: p for p in json.loads((RAW / "predictors.json").read_text())}
    lat = np.array([st[s]["lat"] for s in names])
    lon = np.array([st[s]["lon"] for s in names])
    prov = np.array([st[s]["provider"] for s in names])
    has = np.array([s in pred for s in names])
    F = np.array([[pred[s][f] for f in FEATS] if s in pred else [np.nan] * len(FEATS) for s in names])
    F = np.hstack([F, (z_s - z_h)[:, None], z_s[:, None]])
    blocks = (np.floor(lat / 0.045) * 1000 + np.floor(lon / 0.054)).astype(int)
    LEVELS = ("hrrr", "static", "common", "offset", "kriged")
    rep = {"levels": LEVELS, "lapse_K_per_km": {NAMES[h]: LAPSE[h] * 1000 for h in HOURS_UTC}, "hours": {}}

    def fit_levels(tr, te, Rh):
        """Predicted residual at stations te for every time, per level; uses only stations tr."""
        sub = np.full_like(Rh, np.nan)
        sub[:, tr] = Rh[:, tr]
        c, m, e = decompose(sub)
        mdl = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20)))
        trm = [k for k in tr if np.isfinite(m[k])]
        mdl.fit(F[trm], m[trm])
        mhat = mdl.predict(F[te])
        # exponential correlation of e on training pairs
        ii, jj = np.triu_indices(len(trm), 1)
        a, b = np.array(trm)[ii], np.array(trm)[jj]
        d = km(lat[a], lon[a], lat[b], lon[b])
        rho = np.full(len(a), np.nan)
        for k in range(len(a)):
            both = np.isfinite(e[:, a[k]]) & np.isfinite(e[:, b[k]])
            if both.sum() >= 30:
                rho[k] = np.corrcoef(e[both, a[k]], e[both, b[k]])[0, 1]
        f = fit_exp(d[np.isfinite(rho)], rho[np.isfinite(rho)], np.ones(np.isfinite(rho).sum()))
        c0, L = min(max(f["c0"], 0.0), 0.99), f["length_km"]
        out = {lv: np.zeros((Rh.shape[0], len(te))) for lv in LEVELS}
        out["static"] = np.repeat((np.nanmean(c) + mhat)[None, :], Rh.shape[0], axis=0)
        out["common"] = np.repeat(c[:, None], len(te), axis=1)
        out["offset"] = out["common"] + mhat[None, :]
        krig = np.zeros((Rh.shape[0], len(te)))
        trm_a = np.array(trm)
        for t in range(Rh.shape[0]):
            av = trm_a[np.isfinite(e[t, trm_a])]
            if len(av) == 0:
                continue
            for q, s in enumerate(te):
                dd = km(lat[s], lon[s], lat[av], lon[av])
                near = av[np.argsort(dd)[:15]]
                dn = np.sort(dd)[:15]
                dnn = km(lat[near][:, None], lon[near][:, None], lat[near][None, :], lon[near][None, :])
                Cnn = c0 * np.exp(-dnn / L) + (1 - c0) * np.eye(len(near))
                w = np.linalg.solve(Cnn, c0 * np.exp(-dn / L))
                krig[t, q] = w @ e[t, near]
        out["kriged"] = out["offset"] + krig
        return out, {"c0": c0, "length_km": L}

    for h in HOURS_UTC:
        sel_t = D["hour"] == h
        Rh, qc = clean(D["R"][sel_t], D["keep"])
        TH = D["TH"][sel_t]
        usable = np.where(np.any(np.isfinite(Rh), axis=0) & has)[0]
        g = blocks[usable]
        hot = np.nanmean(TH + Rh, axis=1)  # basin mean observed temperature at the hour
        hot = hot >= np.nanpercentile(hot, 90)
        err = {lv: np.full((Rh.shape[0], len(usable)), np.nan) for lv in LEVELS}
        q90 = {lv: np.full(len(usable), np.nan) for lv in LEVELS}
        q90m = {lv: np.full(len(usable), np.nan) for lv in LEVELS}  # Mondrian: calibrated per network
        fits = []
        for tr_i, te_i in GroupKFold(n_splits=5).split(usable, groups=g):
            tr, te = usable[tr_i], usable[te_i]
            pr, fit = fit_levels(tr, te, Rh)
            fits.append(fit)
            for lv in LEVELS:
                err[lv][:, te_i] = Rh[:, te] - pr[lv]
            # calibration: the same pipeline, nested, on the training stations only
            cal = {lv: [] for lv in LEVELS}
            calp = {lv: [] for lv in LEVELS}
            for a_i, b_i in GroupKFold(n_splits=4).split(tr, groups=blocks[tr]):
                pr2, _ = fit_levels(tr[a_i], tr[b_i], Rh)
                for lv in LEVELS:
                    r2 = Rh[:, tr[b_i]] - pr2[lv]
                    netw = np.broadcast_to(prov[tr[b_i]][None, :], r2.shape)
                    f2 = np.isfinite(r2)
                    cal[lv].append(r2[f2])
                    calp[lv].append(netw[f2])

            def conf_q(a):
                n = len(a)
                return np.quantile(np.abs(a), min(1.0, np.ceil(0.9 * (n + 1)) / n)) if n else np.nan

            for lv in LEVELS:
                a, pa = np.concatenate(cal[lv]), np.concatenate(calp[lv])
                q90[lv][te_i] = conf_q(a)
                for p in set(prov[te]):
                    q90m[lv][te_i[prov[te] == p]] = conf_q(a[pa == p]) if (pa == p).sum() >= 200 else conf_q(a)
            print(f"[baseline] {NAMES[h]}: fold done", flush=True)
        out = {"qc": qc, "stations": int(len(usable)), "fits": fits}
        for lv in LEVELS:
            E = err[lv]
            cover = np.abs(E) <= q90[lv][None, :]
            coverm = np.abs(E) <= q90m[lv][None, :]
            fin = np.isfinite(E)
            res = {"rmse": round(float(np.sqrt(np.nanmean(E ** 2))), 3),
                   "rmse_hot_days": round(float(np.sqrt(np.nanmean(E[hot] ** 2))), 3),
                   "half_width_90": round(float(np.median(q90[lv])), 3),
                   "coverage_90": round(float(cover[fin].mean()), 3),
                   "coverage_90_hot_days": round(float(cover[hot][fin[hot]].mean()), 3),
                   "mondrian_network": {"coverage_90": round(float(coverm[fin].mean()), 3),
                                        "coverage_90_hot_days": round(float(coverm[hot][fin[hot]].mean()), 3)},
                   "by_network": {}}
            for p in sorted(set(prov[usable])):
                sp = prov[usable] == p
                if sp.sum() >= 5:
                    res["by_network"][p] = {"n": int(sp.sum()),
                                            "rmse": round(float(np.sqrt(np.nanmean(E[:, sp] ** 2))), 3),
                                            "coverage_90": round(float(cover[:, sp][fin[:, sp]].mean()), 3),
                                            "coverage_90_mondrian": round(float(coverm[:, sp][fin[:, sp]].mean()), 3),
                                            "half_width_90_mondrian": round(float(np.median(q90m[lv][sp])), 3)}
            out[lv] = res
        rep["hours"][NAMES[h]] = out
    (OUT / f"street_ta_baseline{TAG}.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=1))


def analyse(args) -> None:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    D = load()
    names, st, z_s, z_h, R, times, hour = D["names"], D["st"], D["z_s"], D["z_h"], D["R"], D["times"], D["hour"]
    S, NT = len(names), len(times)
    rep = {"times": NT, "stations_total": S, "years": sorted({t[:4] for t in times}), "lapse_K_per_km": {NAMES[h]: LAPSE[h] * 1000 for h in HOURS_UTC},
           "hours": {}}
    pred = {p["station"]: p for p in json.loads((RAW / "predictors.json").read_text())}
    lat = np.array([st[s]["lat"] for s in names])
    lon = np.array([st[s]["lon"] for s in names])
    prov = np.array([st[s]["provider"] for s in names])
    station_rows = {s: {"provider": st[s]["provider"], "lat": st[s]["lat"], "lon": st[s]["lon"]} for s in names}
    for h in HOURS_UTC:
        Rh, qc = clean(R[hour == h], D["keep"])
        c, m, e = decompose(Rh)
        sd_e = np.nanstd(e, axis=0)
        ok = np.isfinite(m)
        out = {"qc": qc}
        out["raw_bias_mean"] = round(float(np.nanmean(Rh)), 3)
        out["sd"] = {"basin_common_c": round(float(np.nanstd(c)), 3),
                     "station_offset_m": round(float(np.nanstd(m[ok])), 3),
                     "local_daily_e": round(float(np.sqrt(np.nanmean(e[:, ok] ** 2))), 3),
                     "total_r": round(float(np.nanstd(Rh)), 3)}
        out["by_network"] = {}
        for p in sorted(set(prov[ok])):
            sel = ok & (prov == p)
            if sel.sum() >= 5:
                out["by_network"][p] = {"n": int(sel.sum()), "offset_mean": round(float(np.mean(m[sel])), 3),
                                        "offset_sd": round(float(np.std(m[sel])), 3),
                                        "local_sd": round(float(np.sqrt(np.nanmean(e[:, sel] ** 2))), 3)}
        # pairs: semivariogram of m, correlation of e, against distance
        ii, jj = np.triu_indices(S, 1)
        pair_ok = ok[ii] & ok[jj]
        ii, jj = ii[pair_ok], jj[pair_ok]
        d = km(lat[ii], lon[ii], lat[jj], lon[jj])
        edges = np.array([0, 0.5, 1, 2, 3, 5, 8, 12, 20, 30, 50, 80])
        semi, corr, dmid, npair = [], [], [], []
        rho_all = np.full(len(ii), np.nan)
        for k in range(len(ii)):
            a, b = e[:, ii[k]], e[:, jj[k]]
            both = np.isfinite(a) & np.isfinite(b)
            if both.sum() >= 30:
                rho_all[k] = np.corrcoef(a[both], b[both])[0, 1]
        for lo_, hi_ in zip(edges[:-1], edges[1:]):
            sel = (d >= lo_) & (d < hi_)
            if sel.sum() < 3:
                continue
            dmid.append(float(np.median(d[sel])))
            npair.append(int(sel.sum()))
            semi.append(float(0.5 * np.mean((m[ii[sel]] - m[jj[sel]]) ** 2)))
            corr.append(float(np.nanmean(rho_all[sel])) if np.isfinite(rho_all[sel]).any() else float("nan"))
        out["by_distance"] = {"km": [round(x, 2) for x in dmid], "pairs": npair,
                              "offset_semivariance": [round(x, 3) for x in semi],
                              "local_error_corr": [round(x, 3) for x in corr],
                              "offset_variance": round(float(np.nanvar(m[ok])), 3)}
        fin = np.isfinite(rho_all)
        out["local_error_corr_fit"] = fit_exp(d[fin], rho_all[fin], np.ones(fin.sum()))
        close = d < 3
        dm_close = m[ii[close]] - m[jj[close]]
        de_close = e[:, ii[close]] - e[:, jj[close]]
        out["pairs_within_3km"] = {"n": int(close.sum()), "median_km": round(float(np.median(d[close])), 2) if close.any() else None,
                                   "offset_diff_sd": round(float(np.std(dm_close)), 3) if close.any() else None,
                                   "local_diff_sd": round(float(np.sqrt(np.nanmean(de_close ** 2))), 3) if close.any() else None}
        # what the surroundings explain of m, with 5 km spatial blocks held out
        feats = ["tree_100", "built_100", "green_100", "bare_100", "water_100", "tree_500", "built_500", "green_500",
                 "bare_500", "water_500", "coast_km"]
        use = [k for k in range(S) if ok[k] and names[k] in pred]
        Xf = np.array([[pred[names[k]][f] for f in feats] + [z_s[k] - z_h[k], z_s[k]] for k in use])
        y = m[use]
        groups = (np.floor(lat[use] / 0.045) * 1000 + np.floor(lon[use] / 0.054)).astype(int)
        cv = GroupKFold(n_splits=5)
        res = {}
        for name, model, X in [("ridge", make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))), Xf),
                               ("boosting", HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05,
                                                                          max_iter=200, min_samples_leaf=10), Xf)]:
            yhat = np.zeros_like(y)
            for tr, te in cv.split(X, y, groups):
                yhat[te] = model.fit(X[tr], y[tr]).predict(X[te])
            res[name] = {"r2_cv": round(float(1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)), 3),
                         "resid_sd": round(float(np.std(y - yhat)), 3)}
            if name == "ridge":
                pred_ridge = yhat
        # with the network as a feature too (siting bias, not surroundings)
        nets = sorted(set(prov[use]))
        Xn = np.hstack([Xf, np.array([[prov[k] == p for p in nets] for k in use], float)])
        yhat = np.zeros_like(y)
        mdl = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20)))
        for tr, te in cv.split(Xn, y, groups):
            yhat[te] = mdl.fit(Xn[tr], y[tr]).predict(Xn[te])
        res["ridge_plus_network"] = {"r2_cv": round(float(1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)), 3),
                                     "resid_sd": round(float(np.std(y - yhat)), 3)}
        coef = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))).fit(Xf, y)
        res["ridge_coef_per_sd"] = dict(zip(feats + ["above_hrrr_terrain_m", "elev_m"],
                                            [round(float(x), 3) for x in coef[-1].coef_]))
        res["n_stations"] = len(use)
        out["explained_offset"] = res
        out["note"] = "sd values in K; local_daily_e is the RMS of e over stations and times"
        rep["hours"][NAMES[h]] = out
        for k in range(S):
            if np.isfinite(m[k]):
                station_rows[names[k]][f"offset_{h}z"] = round(float(m[k]), 3)
                station_rows[names[k]][f"local_sd_{h}z"] = round(float(sd_e[k]), 3)
        for k, u in enumerate(use):
            station_rows[names[u]][f"offset_pred_{h}z"] = round(float(pred_ridge[k]), 3)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"street_ta{TAG}.json").write_text(json.dumps(rep, indent=2))
    with open(OUT / f"stations{TAG}.csv", "w", newline="") as f:
        cols = sorted({c for r in station_rows.values() for c in r})
        w = csv.writer(f)
        w.writerow(["station"] + cols)
        for s, r in station_rows.items():
            w.writerow([s] + [r.get(c, "") for c in cols])
    figure(rep, station_rows)
    print(json.dumps(rep, indent=1))


def figure(rep, station_rows) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.2), constrained_layout=True)
    col = {"15:00 PDT": "#c8553d", "04:00 PDT": "#2f5d8a"}
    # (a) station offsets at 15:00
    rows = [r for r in station_rows.values() if "offset_22z" in r]
    sc = ax[0].scatter([r["lon"] for r in rows], [r["lat"] for r in rows], c=[r["offset_22z"] for r in rows],
                       cmap="RdBu_r", vmin=-3, vmax=3, s=10, edgecolors="none")
    for name, (la, lo) in TILES.items():
        ax[0].plot(lo, la, marker="s", mfc="none", mec="k", ms=6)
        ax[0].annotate(name, (lo, la), xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax[0].set_xlabel("longitude")
    ax[0].set_ylabel("latitude")
    ax[0].set_aspect(1 / np.cos(np.radians(34)))
    fig.colorbar(sc, ax=ax[0], shrink=0.8, label="station offset from HRRR at 15:00 (K)")
    ax[0].set_title("a", loc="left", fontweight="bold")
    # (b) semivariogram of offsets
    for name, o in rep["hours"].items():
        bd = o["by_distance"]
        ax[1].plot(bd["km"], np.sqrt(np.array(bd["offset_semivariance"])), "o-", color=col[name], ms=3, label=name)
        ax[1].axhline(np.sqrt(bd["offset_variance"]), color=col[name], lw=0.6, ls=":")
    ax[1].set_xscale("log")
    ax[1].axvline(3, color="0.6", lw=0.6)
    ax[1].text(3.1, ax[1].get_ylim()[0], " HRRR grid", fontsize=7, color="0.4", va="bottom")
    ax[1].set_xlabel("distance between stations (km)")
    ax[1].set_ylabel("RMS offset difference / sqrt 2 (K)")
    ax[1].legend(frameon=False)
    ax[1].set_title("b", loc="left", fontweight="bold")
    # (c) correlation of local daily errors
    for name, o in rep["hours"].items():
        bd = o["by_distance"]
        ax[2].plot(bd["km"], bd["local_error_corr"], "o", color=col[name], ms=3, label=name)
        f = o["local_error_corr_fit"]
        dd = np.geomspace(0.2, 80, 100)
        ax[2].plot(dd, f["c0"] * np.exp(-dd / f["length_km"]), color=col[name], lw=0.8)
    ax[2].set_xscale("log")
    ax[2].axhline(0, color="0.6", lw=0.6)
    ax[2].axvline(3, color="0.6", lw=0.6)
    ax[2].set_xlabel("distance between stations (km)")
    ax[2].set_ylabel("correlation of daily local errors")
    ax[2].set_title("c", loc="left", fontweight="bold")
    fig.savefig(OUT / f"street_ta{TAG}.png", dpi=200)


def plot_baseline(args) -> None:
    """Figure of the baseline layer (default height correction: -6.5 K/km at 15:00, none at 04:00)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rep = json.loads((OUT / "street_ta_baseline.json").read_text())["hours"]
    day, night = rep["15:00 PDT"], rep["04:00 PDT"]
    levels = ["hrrr", "static", "common", "offset", "kriged"]
    labels = ["HRRR", "forecast:\nHRRR + offsets", "nowcast:\n+ live bias", "+ offsets", "+ nearby\nstations"]
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.0), constrained_layout=True)
    x = np.arange(len(levels))
    for name, o, colr, dx in (("15:00 PDT", day, "#c8553d", -0.12), ("04:00 PDT", night, "#2f5d8a", 0.12)):
        ax[0].plot(x + dx, [o[lv]["rmse"] for lv in levels], "o", color=colr, ms=4, label=f"{name}, all days")
        ax[0].plot(x + dx, [o[lv]["rmse_hot_days"] for lv in levels], "^", mfc="none", color=colr, ms=4,
                   label=f"{name}, hottest tenth")
    ax[0].axvspan(1.5, 4.5, color="0.93", zorder=0)
    ax[0].text(3, 1.24, "with live reports at the hour", ha="center", fontsize=7, color="0.35")
    ax[0].set_xticks(x, labels)
    ax[0].set_ylabel("RMSE at held-out stations (K)")
    ax[0].set_ylim(1.2, 2.6)
    ax[0].legend(frameon=False, fontsize=7, loc="upper right")
    ax[0].set_title("a", loc="left", fontweight="bold")
    groups = [("all", lambda o: o["coverage_90"], lambda o: o["mondrian_network"]["coverage_90"]),
              ("citizen\n(CWOP)", lambda o: o["by_network"]["APRSWXNET"]["coverage_90"],
               lambda o: o["by_network"]["APRSWXNET"]["coverage_90_mondrian"]),
              ("MesoWest", lambda o: o["by_network"]["MesoWest"]["coverage_90"],
               lambda o: o["by_network"]["MesoWest"]["coverage_90_mondrian"]),
              ("hottest\ntenth", lambda o: o["coverage_90_hot_days"], lambda o: o["mondrian_network"]["coverage_90_hot_days"])]
    xg = np.arange(len(groups))
    for name, o, colr, dx in (("15:00 PDT", day, "#c8553d", -0.15), ("04:00 PDT", night, "#2f5d8a", 0.15)):
        lv = o["kriged"]
        ax[1].plot(xg + dx - 0.04, [g[1](lv) for g in groups], "o", mfc="none", color=colr, ms=4,
                   label=f"{name}, one interval for all")
        ax[1].plot(xg + dx + 0.04, [g[2](lv) for g in groups], "o", color=colr, ms=4,
                   label=f"{name}, per network (Mondrian)")
    ax[1].axhline(0.9, color="0.5", lw=0.7, ls="--")
    ax[1].set_xticks(xg, [g[0] for g in groups])
    ax[1].set_ylabel("coverage of 90% intervals")
    ax[1].set_ylim(0.78, 1.0)  # room above 0.95 for the legend
    ax[1].legend(frameon=False, fontsize=7, loc="upper center", ncol=2)
    ax[1].set_yticks([0.80, 0.85, 0.90, 0.95])
    ax[1].set_title("b", loc="left", fontweight="bold")
    fig.savefig(OUT / "street_ta_baseline.png", dpi=200)
    print("[plot] figures/street_ta/street_ta_baseline.png", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("madis", "hrrr"):
        a = sub.add_parser(name)
        a.add_argument("--threads", type=int, default=8)
        a.add_argument("--years", type=int, nargs="+", default=[2022, 2023])
        a.add_argument("--hours", default="22,11", help="UTC hours, comma-separated, or all")
    sub.add_parser("predictors")
    sub.add_parser("plot_baseline")
    for name in ("analyse", "baseline"):
        a = sub.add_parser(name)
        a.add_argument("--lapse", type=float, default=None,
                       help="K per km for the height correction of HRRR at both hours (default: -6.5 by day, 0 at night)")
        a.add_argument("--tag", default="", help="suffix for the output files")
    args = ap.parse_args()
    if args.cmd in ("madis", "hrrr"):
        args.hours = list(range(24)) if args.hours == "all" else [int(h) for h in args.hours.split(",")]
    if args.cmd in ("analyse", "baseline"):
        global LAPSE, TAG
        TAG = args.tag
        if args.lapse is not None:
            LAPSE = {h: args.lapse / 1000.0 for h in range(24)}
    {"madis": madis, "hrrr": hrrr, "predictors": predictors, "analyse": analyse, "baseline": baseline,
     "plot_baseline": plot_baseline}[args.cmd](args)


if __name__ == "__main__":
    main()
