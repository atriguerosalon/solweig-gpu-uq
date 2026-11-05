"""Does the two-run air-temperature shortcut survive ground heat storage?

case_study.py showed that an air-temperature forecast ensemble (20 members, AR(1) errors, sd 1.5 K)
is reproduced from two physics runs: the unperturbed run and one with Ta + 1 K, giving each
pixel's dTmrt/dTa per hour. That relies on SOLWEIG's own physics responding at each hour to that
hour's Ta only. The ground model we use for the measurements (heat_storage.py, tau 4 h) keeps the
ground's daytime excess longwave after sunset, set by the last daytime hour, so an error at that
hour reaches the night. Same venue tile, members and products as case_study.py; ground heat
storage on (tau 4 h; the fitted Pacoima ground heating is site-specific and left out). Shortcuts:

    (b)  the unperturbed run's Tmrt, member Ta in UTCI (1 physics run)
    (c)  + per-pixel dTmrt/dTa per hour from the Ta + 1 K run (2 physics runs)

against one full physics run per member.

Usage (after case_study.py):
    python scripts/case_study_storage.py --rasters data/la/expo_park --met data/la/met_expo_20190730.txt --tag expo
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

TAU = 4.0


def main() -> None:
    import torch
    from osgeo import gdal
    from solweig_gpu import run_utci_tiles
    from case_study import HALF_MEMBER, SIGMA, day_patches, timed
    from run_la_forecast_error import design, ta_error
    from solweig_sens.analysis import read_stack
    from solweig_sens.heat_storage import ground_heat_storage
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
    days = work / "days_storage"
    out_json = args.out / f"case_study_storage_{args.tag}.json"
    res = json.loads(out_json.read_text()) if out_json.exists() else {"tag": args.tag, "tau_hours": TAU, "runs": {}}
    res["members"] = args.members
    torch.zeros(1, device="cuda")
    ms = [r for r in design() if r["sigma"] == SIGMA and r["member"] < args.members]
    todo = [{"id": "base", "error": [0.0] * 24}, {"id": "plus1K", "error": [1.0] * 24}, *ms]
    for r in todo:
        dst = days / r["id"]
        if r["id"] in res["runs"] and (dst / "output_folder").exists():
            continue
        shutil.rmtree(dst, ignore_errors=True)

        def day(r=r, dst=dst):
            with contextlib.ExitStack() as st:
                for c in day_patches("ours"):
                    st.enter_context(c)
                st.enter_context(ta_error(pre / "metfiles", r["error"]))
                st.enter_context(ground_heat_storage(TAU))
                run_utci_tiles(base_path=str(dst), preprocess_dir=str(pre), selected_date_str=args.date, save_tmrt=True)

        res["runs"][r["id"]] = timed(day)
        out_json.write_text(json.dumps(res, indent=2))
        print(f"[store] {r['id']:9s} {res['runs'][r['id']]['seconds']:5.1f} s", flush=True)

    met = np.loadtxt(args.met, skiprows=1)
    hour = met[:, 2].astype(int) % 24
    ta, rh, ws = (np.zeros(24) for _ in range(3))
    ta[hour], rh[hour], ws[hour] = (met[:, MET_COLS[c]] for c in ("ta", "rh", "wind"))
    t0 = read_stack(days / "base", "TMRT").astype(np.float64)
    slope = read_stack(days / "plus1K", "TMRT") - t0
    lc = gdal.Open(str(args.rasters / "Landcover.tif")).ReadAsArray()[: t0.shape[1], : t0.shape[2]]
    outdoor = (lc != 2) & np.isfinite(t0).all(axis=0)
    shape = outdoor.shape

    def indicators(u):
        return {"p_4h_above_32": ((u > 32.0).sum(axis=0) >= 4) * 1.0, "p_above_38_at_15": (u[15] > 38.0) * 1.0}

    pf = {k: np.zeros(shape) for k in ("p_4h_above_32", "p_above_38_at_15")}
    psum = {n: {k: np.zeros(shape) for k in pf} for n in "bc"}
    err = {n: [] for n in "bc"}
    err_by_hour = {n: np.zeros(24) for n in "bc"}
    for r in ms:
        f = read_stack(days / r["id"], "UTCI")
        tm_full = read_stack(days / r["id"], "TMRT")
        for k, v in indicators(f).items():
            pf[k] += v / len(ms)
        e = np.asarray(r["error"])
        for n, tm in (("b", t0), ("c", t0 + e[:, None, None] * slope)):
            u = np.stack([utci_fn(np.full(shape, ta[h] + e[h]), np.full(shape, rh[h]), tm[h], np.full(shape, ws[h]))
                          for h in range(24)])
            for k, v in indicators(u).items():
                psum[n][k] += v / len(ms)
            d = np.abs(u - f)[:, outdoor]
            err[n].append((float(np.nanmean(d)), float(np.nanpercentile(d, 99)), float(np.nanmax(d))))
            err_by_hour[n] += np.abs(tm - tm_full)[:, outdoor].mean(axis=1) / len(ms)
    res["shortcuts"] = {}
    for n, lab in (("b", "1 physics run"), ("c", "2 physics runs: + per-pixel dTmrt/dTa per hour")):
        e = np.array(err[n])
        res["shortcuts"][n] = {
            "what": lab,
            "member_utci_error": {"mean_abs": float(e[:, 0].mean()), "p99_abs": float(e[:, 1].mean()),
                                  "max_abs": float(e[:, 2].max())},
            "tmrt_error_mean_abs_by_hour": err_by_hour[n].tolist(),
            "products_abs_diff": {k: {"mean": float(np.abs(psum[n][k] - pf[k])[outdoor].mean()),
                                      "pixels_differing_pct": float((np.abs(psum[n][k] - pf[k])[outdoor] > HALF_MEMBER).mean() * 100)}
                                  for k in pf}}
        s = res["shortcuts"][n]
        print(f"[store] ({n}) {lab}: member UTCI error mean {s['member_utci_error']['mean_abs']:.3f} C, p99 "
              f"{s['member_utci_error']['p99_abs']:.3f}, max {s['member_utci_error']['max_abs']:.2f} C; P(4 h > 32) differs at "
              f"{s['products_abs_diff']['p_4h_above_32']['pixels_differing_pct']:.1f}% of pixels", flush=True)
        print("        Tmrt error by hour: " + " ".join(f"{h}:{v:.3f}" for h, v in enumerate(err_by_hour[n])), flush=True)
    out_json.write_text(json.dumps(res, indent=2))
    print(f"[store] wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
