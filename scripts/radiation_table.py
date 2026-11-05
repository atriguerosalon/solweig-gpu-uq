"""Can radiation errors be shortcut once SOLWEIG's switches are known? A per-hour lookup table.

case_study_met.py: a per-hour radiation slope (one run at x 1.1) leaves products off at 5.4% of
pixels, because Tmrt responds to radiation in steps (radiation_sweep.py, radiation_switches.py).
But an error at one hour barely reaches later hours (radiation_memory.py), so Tmrt at hour h is
close to a function of that hour's radiation scale alone. Two shortcuts that use that:

    (e) two-sided slope: the x 1.1 run for positive errors, the x 0.9 run for negative ones
        (4 physics runs with the base and Ta + 1 K runs)
    (f) table: runs with the radiation scaled by f at every hour, f on a grid (0.70 to 1.30, step
        0.05; 13 runs); a member's Tmrt at hour h is interpolated linearly in f between the two
        grid runs around its scale at h, plus its Ta error times the Ta slope
        (13 + 1 physics runs, for any number of members)

Same members (Ta, wind, radiation errors) and full-physics runs as case_study_met.py.

Usage (after case_study_met.py and radiation_linearity.py):
    python scripts/radiation_table.py --rasters data/la/expo_park --met data/la/met_expo_20190730.txt --tag expo
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

GRID = np.round(np.arange(0.70, 1.3001, 0.05), 4)


def main() -> None:
    import torch
    from osgeo import gdal
    from solweig_gpu import run_utci_tiles
    from case_study import HALF_MEMBER, day_patches
    from case_study_met import K_STEP, members, met_error
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
    pre, days = work / "day" / "processed_inputs", work / "days" / "ours"
    for need in ("base", "plus1K", "kdown_plus", "kdown_minus"):
        if not (days / need / "output_folder").exists():
            raise SystemExit(f"missing run {need}: run case_study.py, case_study_met.py and radiation_linearity.py first")
    torch.zeros(1, device="cuda")
    table = []
    for f in GRID:
        dst = work / "table" / f"f{f:.2f}"
        if f == 1.0:
            dst = days / "base"
        elif not (dst / "output_folder").exists():
            shutil.rmtree(dst, ignore_errors=True)
            with contextlib.ExitStack() as st:
                for c in day_patches("ours"):
                    st.enter_context(c)
                st.enter_context(met_error(pre / "metfiles", [0.0] * 24, [1.0] * 24, [float(f)] * 24))
                run_utci_tiles(base_path=str(dst), preprocess_dir=str(pre), selected_date_str=args.date, save_tmrt=True)
            print(f"[table] f {f:.2f} done", flush=True)
        table.append(read_stack(dst, "TMRT"))
    table = np.stack(table)  # (grid, hours, rows, cols)

    met = np.loadtxt(args.met, skiprows=1)
    hour = met[:, 2].astype(int) % 24
    ta, rh, ws = (np.zeros(24) for _ in range(3))
    ta[hour], rh[hour], ws[hour] = (met[:, MET_COLS[c]] for c in ("ta", "rh", "wind"))
    t0 = read_stack(days / "base", "TMRT").astype(np.float64)
    s_ta = read_stack(days / "plus1K", "TMRT") - t0
    s_up = (read_stack(days / "kdown_plus", "TMRT") - t0) / K_STEP
    s_dn = (t0 - read_stack(days / "kdown_minus", "TMRT")) / K_STEP
    lc = gdal.Open(str(args.rasters / "Landcover.tif")).ReadAsArray()[: t0.shape[1], : t0.shape[2]]
    outdoor = (lc != 2) & np.isfinite(t0).all(axis=0)
    shape = outdoor.shape

    def interp(h, k):
        k = min(max(k, GRID[0]), GRID[-1])
        i = min(int(np.searchsorted(GRID, k, side="right")) - 1, len(GRID) - 2)
        w = (k - GRID[i]) / (GRID[i + 1] - GRID[i])
        return (1 - w) * table[i, h] + w * table[i + 1, h]

    def indicators(u):
        return {"p_4h_above_32": ((u > 32.0).sum(axis=0) >= 4) * 1.0, "p_above_38_at_15": (u[15] > 38.0) * 1.0}

    ms = members(args.members)
    names = ("e", "f")
    pf = {k: np.zeros(shape) for k in ("p_4h_above_32", "p_above_38_at_15")}
    psum = {n: {k: np.zeros(shape) for k in pf} for n in names}
    err = {n: [] for n in names}
    for r in ms:
        full = read_stack(days / r["id"], "UTCI")
        for k, v in indicators(full).items():
            pf[k] += v / len(ms)
        e, w, kf = (np.asarray(r[x]) for x in ("ta_add", "wind_scale", "kdown_scale"))
        two = t0 + e[:, None, None] * s_ta + np.where((kf - 1)[:, None, None] >= 0, (kf - 1)[:, None, None] * s_up,
                                                     (kf - 1)[:, None, None] * s_dn)
        tab = np.stack([interp(h, kf[h]) for h in range(24)]) + e[:, None, None] * s_ta
        for n, tm in (("e", two), ("f", tab)):
            u = np.stack([utci_fn(np.full(shape, ta[h] + e[h]), np.full(shape, rh[h]), tm[h], np.full(shape, ws[h] * w[h]))
                          for h in range(24)])
            for k, v in indicators(u).items():
                psum[n][k] += v / len(ms)
            d = np.abs(u - full)[:, outdoor]
            err[n].append((float(np.nanmean(d)), float(np.nanpercentile(d, 99)), float(np.nanmax(d))))
    res = {"grid": GRID.tolist(), "members": len(ms), "shortcuts": {}}
    labels = {"e": "4 physics runs: two-sided radiation slope (x 1.1 and x 0.9 runs) + Ta slope",
              "f": f"{len(GRID)} + 1 physics runs: per-hour table in the radiation scale + Ta slope"}
    for n in names:
        x = np.array(err[n])
        res["shortcuts"][n] = {"what": labels[n],
                               "member_utci_error": {"mean_abs": float(x[:, 0].mean()), "p99_abs": float(x[:, 1].mean()),
                                                     "max_abs": float(x[:, 2].max())},
                               "products_abs_diff": {k: {"mean": float(np.abs(psum[n][k] - pf[k])[outdoor].mean()),
                                                         "pixels_differing_pct": float((np.abs(psum[n][k] - pf[k])[outdoor] > HALF_MEMBER).mean() * 100)}
                                                     for k in pf}}
        s = res["shortcuts"][n]
        print(f"[table] ({n}) {labels[n]}: member UTCI error mean {s['member_utci_error']['mean_abs']:.3f}, p99 "
              f"{s['member_utci_error']['p99_abs']:.3f}, max {s['member_utci_error']['max_abs']:.2f} C; P(4 h > 32) differs at "
              f"{s['products_abs_diff']['p_4h_above_32']['pixels_differing_pct']:.1f}% (mean |dp| "
              f"{s['products_abs_diff']['p_4h_above_32']['mean']:.4f})", flush=True)
    out = args.out / f"radiation_table_{args.tag}.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"[table] wrote {out}")


if __name__ == "__main__":
    main()
