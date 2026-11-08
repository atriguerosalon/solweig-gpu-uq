"""Street-scale skill when the air-temperature forecast error varies in space, not only in time.

SOLWEIG takes one air temperature per tile, so the forecast-error ensembles (run_la_forecast_error.py) put the same
error on every pixel, which leaves the map's pattern intact by construction. Here the error is a random field with
a correlation length L, applied pixel by pixel with the per-pixel shortcut of shortcut_suite.py: Tmrt = base Tmrt +
error x (dTmrt/dTa of that pixel and hour, from the run with air temperature + 1 K), and UTCI from the local air
temperature, humidity and wind. That shortcut reproduced full physics within 0.01 C on these tiles.

Error model: e(t, x) = sigma z(t, x); z has unit variance, an exponential correlation exp(-r / L) in space and
AR(1) in time (lag-1 correlation 0.9), z_t = 0.9 z_(t-1) + sqrt(1 - 0.81) eta_t with independent fields eta_t.
Fields come from circulant embedding on a periodic grid much larger than the tile (no wrap-around inside it), at a
spacing of L / 12 (20 to 100 m), bilinearly interpolated to 2 m. L = "uniform" is one value per hour for the
whole tile, as in the full-physics ensembles: it should give their result again. Optionally a mix: a share c0 of
the variance with length L and the rest with a short length Ls (station errors are only partly correlated even
when close, part of each is local).

Scored as in analyse_la_forecast_error.py: each of the 20 members in turn is the truth, the others the forecast;
fractions skill score against neighbourhood size for (i) UTCI above 32 C for 4 h or more, (ii) each map's
hottest 10% of outdoor pixels at 15:00, (iii) its coolest 10% (shade) and (iv) its hotter half. At 15:00 the top of
the UTCI distribution is flat (40% of the venue's outdoor pixels lie within 0.5 C of the 90th percentile), so (ii)
is decided by tenths of a degree; (iii) and (iv) test the sun-shade pattern. Also the correlation between members'
UTCI anomaly maps (tile mean removed) and the UTCI change at 15:00 split into the tile mean and the spread across
pixels.

Writes figures/spatial_error/spatial_error_fss_<tile>.json.

Usage (after shortcut_suite.py has run the tile; CPU only):
    python scripts/spatial_error_fss.py --tile pacoima --runs ~/shortcut_suite/pacoima_0730/runs_independent_fitted \
        --rasters data/la/pacoima --met data/la/met_pacoima_20190730.txt
"""

from __future__ import annotations

import sys as _sys

if __name__ == "__main__" and any(a in ("-h", "--help") for a in _sys.argv[1:]):
    print(__doc__)
    raise SystemExit(0)

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

PIXEL_M = 2.0
MEMBERS = 20
RHO = 0.9
TOP_SHARE = 0.10
G = {}  # per-process data, filled by init()


def grf_maker(L_m: float, shape: tuple, rng: np.random.Generator):
    """Returns a function giving independent unit-variance fields on the tile, exponential correlation length L_m."""
    from scipy.ndimage import map_coordinates

    dx = float(np.clip(L_m / 12, 20.0, 100.0))
    extent = max(shape) * PIXEL_M
    n = int(2 ** np.ceil(np.log2((extent + 8 * L_m) / dx)))
    k = np.minimum(np.arange(n), n - np.arange(n)) * dx
    r = np.hypot(k[:, None], k[None, :])
    lam = np.fft.fft2(np.exp(-r / L_m)).real
    sq = np.sqrt(np.clip(lam, 0, None) / lam.size)
    rows, cols = shape
    yy, xx = np.meshgrid(np.arange(rows) * PIXEL_M / dx, np.arange(cols) * PIXEL_M / dx, indexing="ij")
    coords = np.stack([yy.ravel(), xx.ravel()])
    m = int(np.ceil(extent / dx)) + 2

    def field():
        w = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        f = np.fft.fft2(sq * w).real[:m, :m]
        return map_coordinates(f, coords, order=1).reshape(shape)

    return field


def init(paths, members=MEMBERS):
    global MEMBERS
    MEMBERS = members
    import torch

    torch.set_num_threads(2)
    from osgeo import gdal
    from solweig_sens.analysis import read_stack
    from solweig_sens.perturb import MET_COLS

    runs, rasters, met = map(Path, paths)
    t0 = read_stack(runs / "base", "TMRT").astype(np.float32)
    s_ta = (read_stack(runs / "plus1K", "TMRT") - t0).astype(np.float32)
    m = np.loadtxt(met, skiprows=1)
    hour = m[:, 2].astype(int) % 24
    ta, rh, ws = (np.zeros(24) for _ in range(3))
    ta[hour], rh[hour], ws[hour] = (m[:, MET_COLS[c]] for c in ("ta", "rh", "wind"))
    lc = gdal.Open(str(rasters / "Landcover.tif")).ReadAsArray()[: t0.shape[1], : t0.shape[2]]
    G.update(t0=t0, s_ta=s_ta, ta=ta, rh=rh, ws=ws, valid=(lc != 2) & np.isfinite(t0).all(axis=0))


def utci_day(err):
    """UTCI for 24 h with error field err (24, rows, cols) or (24,) on air temperature."""
    from spatial_ta import utci as utci_fn

    shape = G["valid"].shape
    out = np.empty((24,) + shape, np.float32)
    for h in range(24):
        e = err[h]
        out[h] = utci_fn(np.full(shape, G["ta"][h], np.float32) + e, np.full(shape, G["rh"][h], np.float32),
                         G["t0"][h] + e * G["s_ta"][h], np.full(shape, G["ws"][h], np.float32))
    return out


def one_case(case):
    tile, sigma, L, c0, Ls, seed = case
    from analyse_la_forecast_error import SCALES_PX, fractions, fss

    t = time.perf_counter()
    rng = np.random.default_rng(seed)
    shape, valid = G["valid"].shape, G["valid"]
    makers = []  # (share of the variance, field maker); L = "uniform" gives one value per tile for its share
    makers.append((c0, (lambda: np.full(shape, rng.standard_normal(), np.float32)) if L == "uniform"
                   else grf_maker(float(L), shape, rng)))
    if c0 < 1:
        makers.append((1 - c0, grf_maker(float(Ls), shape, rng)))

    def draw():
        return sum(np.sqrt(w) * mk() for w, mk in makers).astype(np.float32)

    base = utci_day(np.zeros(24, np.float32))
    ev32, top, cool, half, shift, spread, anom = [], [], [], [], [], [], []
    for _ in range(MEMBERS):
        z = draw()
        err = np.empty((24,) + shape, np.float32)
        for h in range(24):
            if h:
                z = RHO * z + np.sqrt(1 - RHO ** 2) * draw()
            err[h] = sigma * z
        u = utci_day(err)
        ev32.append(((u > 32.0).sum(axis=0) >= 4) & valid)
        u15 = u[15]
        top.append((u15 >= np.nanpercentile(u15[valid], 100 * (1 - TOP_SHARE))) & valid)
        cool.append((u15 <= np.nanpercentile(u15[valid], 100 * TOP_SHARE)) & valid)
        half.append((u15 >= np.nanpercentile(u15[valid], 50)) & valid)
        anom.append((u15[valid] - u15[valid].mean()).astype(np.float32))
        d = (u15 - base[15])[valid]
        shift.append(abs(float(d.mean())))
        spread.append(float(d.std()))

    def skill(ev):
        ev = np.stack(ev).astype(float)
        det, ens, freq = {n: [] for n in SCALES_PX}, {n: [] for n in SCALES_PX}, []
        for k in range(len(ev)):
            truth = ev[k]
            freq.append(truth[valid].mean())
            other = [j for j in range(len(ev)) if j != k]
            prob, fc = ev[other].mean(axis=0), ev[other[k % len(other)]]
            for n in SCALES_PX:
                po = fractions(truth, valid, n)
                det[n].append(fss(fractions(fc, valid, n), po, valid))
                ens[n].append(fss(fractions(prob, valid, n), po, valid))
        f = float(np.mean(freq))
        target = 0.5 + f / 2
        rec = {"event_frequency": f, "useful_target": target,
               "fss_member": [float(np.nanmean(det[n])) for n in SCALES_PX],
               "fss_member_p10_p90": [[float(np.nanpercentile(det[n], q)) for q in (10, 90)] for n in SCALES_PX],
               "fss_ensemble": [float(np.nanmean(ens[n])) for n in SCALES_PX]}
        for kind in ("member", "ensemble"):
            ok = [n * PIXEL_M for n, v in zip(SCALES_PX, rec[f"fss_{kind}"]) if v >= target]
            rec[f"useful_scale_m_{kind}"] = ok[0] if ok else None
        return rec

    res = {"tile": tile, "sigma": sigma, "L_m": L, "c0": c0, "Ls_m": Ls, "seed": seed,
           "event_32_4h": skill(ev32), "event_top10": skill(top), "event_cool10": skill(cool), "event_top50": skill(half),
           "anomaly_correlation_mean": float(np.mean([np.corrcoef(anom[k], anom[j])[0, 1] for k in range(len(anom))
                                                      for j in range(len(anom)) if j != k][: 4 * len(anom)])),
           "utci_15h_level_shift_abs_mean": float(np.mean(shift)), "utci_15h_pattern_spread_mean": float(np.mean(spread)),
           "seconds": round(time.perf_counter() - t, 1)}
    print(f"[spatial] {tile} sigma {sigma} L {L} c0 {c0}: top10 FSS at 2 m member "
          f"{res['event_top10']['fss_member'][0]:.3f}, useful {res['event_top10']['useful_scale_m_member']} m; cool10 "
          f"{res['event_cool10']['fss_member'][0]:.3f}, top50 {res['event_top50']['fss_member'][0]:.3f}, anomaly r "
          f"{res['anomaly_correlation_mean']:.3f}; "
          f"level {res['utci_15h_level_shift_abs_mean']:.2f}, spread {res['utci_15h_pattern_spread_mean']:.3f} C "
          f"({res['seconds']} s)", flush=True)
    return res


def main() -> None:
    global MEMBERS
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", required=True)
    ap.add_argument("--runs", type=Path, required=True, help="shortcut_suite.py runs folder with base/ and plus1K/")
    ap.add_argument("--rasters", type=Path, required=True)
    ap.add_argument("--met", type=Path, required=True)
    ap.add_argument("--sigmas", default="1,2,3")
    ap.add_argument("--lengths", default="250,1000,3000,10000,uniform", help="correlation lengths in m, or uniform")
    ap.add_argument("--c0", type=float, default=1.0, help="share of the variance with length L (rest: length --short)")
    ap.add_argument("--shorts", default="250", help="the other length(s), m, when c0 < 1 (comma-separated)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--members", type=int, default=20, help="fewer only to check the script")
    args = ap.parse_args()
    MEMBERS = args.members
    sigmas = [float(s) for s in args.sigmas.split(",")]
    lengths = [x if x == "uniform" else float(x) for x in args.lengths.split(",")]
    shorts = [float(x) for x in args.shorts.split(",")]
    cases = [(args.tile, s, L, args.c0, Ls, 1000 * i + 17 * j + 101 * k + 1)
             for i, s in enumerate(sigmas) for j, L in enumerate(lengths) for k, Ls in enumerate(shorts)]
    if args.c0 >= 1:  # the second length plays no part: one case per (sigma, L), seeds as before
        cases = [(args.tile, s, L, args.c0, shorts[0], 1000 * i + 17 * j + 1) for i, s in enumerate(sigmas)
                 for j, L in enumerate(lengths)]
    with Pool(args.procs, initializer=init, initargs=([str(args.runs), str(args.rasters), str(args.met)], MEMBERS)) as pool:
        results = pool.map(one_case, cases, chunksize=1)
    from analyse_la_forecast_error import SCALES_PX

    out = ROOT / "figures" / "spatial_error"
    out.mkdir(parents=True, exist_ok=True)
    rep = {"tile": args.tile, "members": MEMBERS, "rho": RHO, "scales_m": [n * PIXEL_M for n in SCALES_PX],
           "c0": args.c0, "shorts_m": shorts, "results": results}
    dst = out / f"spatial_error_fss_{args.tile}{args.tag}.json"
    dst.write_text(json.dumps(rep, indent=1))
    print(f"[spatial] wrote {dst}", flush=True)


if __name__ == "__main__":
    main()
