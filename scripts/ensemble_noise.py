"""How much do two equally valid 20-member ensembles disagree? A yardstick for the shortcuts.

case_study.py compares a two-run shortcut with the full-physics ensemble: their products differ by
one member or more at 1.6% of pixels. A 20-member ensemble has its own sampling noise, so the
fair comparison is with a second, independent ensemble of the same error statistics (AR(1) in
time, lag-1 correlation 0.9, sd 1.5 K; new seed), also run with full physics. Same venue tile,
our code, SOLWEIG's own physics, same products: P(UTCI > 32 C for 4 h or more), P(> 38 C at 15:00).

Usage (after case_study.py):
    python scripts/ensemble_noise.py --rasters data/la/expo_park --tag expo
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

SEED_B = 1


def main() -> None:
    import torch
    from osgeo import gdal
    from solweig_gpu import run_utci_tiles
    from case_study import HALF_MEMBER, MEMBERS, SIGMA, day_patches, timed
    from case_study_met import ar1
    from run_la_forecast_error import RHO, ta_error
    from solweig_sens.analysis import read_stack

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rasters", type=Path, required=True)
    ap.add_argument("--date", default="2019-07-30")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--work", type=Path, default=Path.home() / "case_study")
    ap.add_argument("--out", type=Path, default=ROOT / "figures")
    args = ap.parse_args()
    work = args.work / args.tag
    pre = work / "day" / "processed_inputs"
    days = work / "days_b"
    out_json = args.out / f"ensemble_noise_{args.tag}.json"
    res = json.loads(out_json.read_text()) if out_json.exists() else {"seed_b": SEED_B, "sigma": SIGMA, "rho": RHO,
                                                                      "members": MEMBERS, "runs": {}}
    torch.zeros(1, device="cuda")
    unit = ar1(SEED_B, MEMBERS, RHO)
    for m in range(MEMBERS):
        rid = f"b_m{m:02d}"
        dst = days / rid
        if rid in res["runs"] and (dst / "output_folder").exists():
            continue
        shutil.rmtree(dst, ignore_errors=True)
        err = (SIGMA * unit[m]).tolist()

        def day(dst=dst, err=err):
            with contextlib.ExitStack() as st:
                for c in day_patches("ours"):
                    st.enter_context(c)
                st.enter_context(ta_error(pre / "metfiles", err))
                run_utci_tiles(base_path=str(dst), preprocess_dir=str(pre), selected_date_str=args.date, save_tmrt=False)

        res["runs"][rid] = timed(day)
        out_json.write_text(json.dumps(res, indent=2))
        print(f"[noise] {rid} {res['runs'][rid]['seconds']:5.1f} s", flush=True)

    shape = None
    pb = None
    for m in range(MEMBERS):
        u = read_stack(days / f"b_m{m:02d}", "UTCI")
        if pb is None:
            shape = u.shape[1:]
            pb = {"p_4h_above_32": np.zeros(shape), "p_above_38_at_15": np.zeros(shape)}
        pb["p_4h_above_32"] += ((u > 32.0).sum(axis=0) >= 4) / MEMBERS
        pb["p_above_38_at_15"] += (u[15] > 38.0) / MEMBERS
    pa = np.load(work / "products_ours.npz")  # ensemble A: case_study.py's full-physics members
    lc = gdal.Open(str(args.rasters / "Landcover.tif")).ReadAsArray()[: shape[0], : shape[1]]
    outdoor = (lc != 2) & np.isfinite(pa["p_4h_above_32"])
    cs = json.loads((args.out / f"case_study_{args.tag}.json").read_text())
    res["a_vs_b"] = {}
    for k in pb:
        d = np.abs(pb[k] - pa[k])[outdoor]
        res["a_vs_b"][k] = {"mean": float(d.mean()), "p99": float(np.percentile(d, 99)), "max": float(d.max()),
                            "pixels_differing_pct": float((d > HALF_MEMBER).mean() * 100),
                            "pixels_differing_by_3_or_more_pct": float((d > 2.5 / MEMBERS).mean() * 100),
                            "mean_p_a": float(pa[k][outdoor].mean()), "mean_p_b": float(pb[k][outdoor].mean())}
        sc = cs["shortcut_corrected"]["products_abs_diff_vs_full_physics"][k]
        print(f"[noise] {k}: two full-physics ensembles differ (>= 1 member) at {res['a_vs_b'][k]['pixels_differing_pct']:.1f}% "
              f"of pixels, by >= 3 members at {res['a_vs_b'][k]['pixels_differing_by_3_or_more_pct']:.1f}%, mean |dp| "
              f"{res['a_vs_b'][k]['mean']:.4f}, max {res['a_vs_b'][k]['max']:.2f}; shortcut (c) vs full: "
              f"{sc['pixels_differing_pct']:.1f}%, mean {sc['mean']:.4f}, max {sc['max']:.2f}", flush=True)
    out_json.write_text(json.dumps(res, indent=2))
    print(f"[noise] wrote {out_json}")


if __name__ == "__main__":
    main()
