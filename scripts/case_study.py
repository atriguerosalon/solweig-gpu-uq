"""A street-scale heat forecast, run with SOLWEIG-GPU v2.0.0 and with ours: same answer, what cost?

How street-scale heat forecasting would use SOLWEIG for a venue: compute the tile's geometry once (sky view factor),
then run each forecast day as a deterministic run and an ensemble, and turn the ensemble into the
products decision makers use, such as the probability of strong heat stress (UTCI above 32 C) for
four hours or more, or of very strong heat stress (above 38 C) at 15:00. This script does that on
one 1000 x 1000 tile (2 x 2 km at 2 m) with both versions:

    v2.0.0   the package as installed, nothing switched on
    ours     C1-C9: fast functions, bit-packed masks, tight bound, sky-patch shadows without syncs,
             geometry once per run, packed mask cache

"Ours" is the committed, verified code (C1-C9); the torch.compile steps are experiments and stay
out. SOLWEIG's own physics in both (no fitted ground, no heat storage): this compares solvers, not
models. Ensemble: the air temperature plus an AR(1) error in time (lag-1 correlation 0.9,
standard deviation SIGMA), the members of run_la_forecast_error.py at that sigma; SIGMA is an
illustrative forecast error, its lead time to be taken from published HRRR verification.

A third way to make the same ensemble, the question for an emulator: (b) one physics run, then UTCI
recomputed per member from that run's Tmrt with the member's air temperature (SOLWEIG-GPU's own UTCI
function; nearly free; it misses only the member's Ta feeding back into Tmrt through longwave).
Its error against the full-physics ensemble, on the members and on the products, says whether a
learned emulator is needed for forecast ensembles at all. (c) the same, corrected to first order:
one more physics run with Ta + 1 K gives each pixel's dTmrt/dTa per hour, and a member's Tmrt is
the unperturbed one plus its error times that slope. Two physics runs for any number of members.

Recorded: wall time and peak GPU memory of every step, per version; the largest difference
between the two versions' outputs per member (bit-identical expected except C1's float32
rounding, <= 0.0001 C); and the ensemble products, which must agree between the versions.
Runs alternate between the versions, member by member.

Usage:
    python scripts/case_study.py --rasters data/la/expo_park --met data/la/met_expo_20190730.txt --tag expo
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

SIGMA = 1.5
MEMBERS = 20
# products are multiples of 1/MEMBERS; a real difference is at least one member, smaller ones are float32 rounding
HALF_MEMBER = 0.5 / MEMBERS


def svf_patches(version: str) -> list:
    from solweig_sens.compact_masks import compact_masks
    from solweig_sens.fast_sky_shadow import fast_sky_shadow
    from solweig_sens.tight_bound import tight_bound

    return [] if version == "v2.0.0" else [compact_masks("bits", svf=True), fast_sky_shadow(), tight_bound()]


def day_patches(version: str) -> list:
    from solweig_sens.compact_masks import compact_masks
    from solweig_sens.fast import fast_solweig
    from solweig_sens.hoist_geometry import hoist_geometry
    from solweig_sens.tight_bound import tight_bound

    if version == "v2.0.0":
        return []
    return [fast_solweig(), compact_masks("bits", svf=False, packed_cache=True), hoist_geometry(), tight_bound()]


def timed(fn) -> dict:
    import torch

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return {"seconds": time.perf_counter() - t0, "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2 ** 30}


def main() -> None:
    import torch
    from solweig_gpu import calculate_svf, run_utci_tiles
    from bench_steps import compare_zips
    from run_la_forecast_error import design, ta_error
    from solweig_sens.analysis import read_stack
    from solweig_sens.run import Setup, prepare
    from verify_fast_sky_shadow import compare as compare_files

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rasters", type=Path, required=True)
    ap.add_argument("--met", type=Path, required=True)
    ap.add_argument("--date", default="2019-07-30")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--work", type=Path, default=Path.home() / "case_study")
    ap.add_argument("--out", type=Path, default=ROOT / "figures")
    ap.add_argument("--members", type=int, default=MEMBERS, help="fewer only for a quick check of the script")
    args = ap.parse_args()
    work = args.work / args.tag
    out_json = args.out / f"case_study_{args.tag}.json"
    res = json.loads(out_json.read_text()) if out_json.exists() else {"tag": args.tag, "date": args.date,
                                                                      "sigma": SIGMA, "members": MEMBERS}
    torch.zeros(1, device="cuda")

    # walls, aspect and the met files once (ours); then the sky view factor timed per version
    setup = Setup(raster_dir=args.rasters.resolve(), met_file=args.met.resolve(), runs_dir=work / "base",
                  date=args.date, tile_size=1000)
    if not (setup.preprocess_dir / "SVF").exists():
        with contextlib.ExitStack() as st:
            for c in svf_patches("ours"):
                st.enter_context(c)
            prepare(setup)
    if "geometry" not in res:
        res["geometry"] = {}
        for version in ("v2.0.0", "ours"):
            pre = work / f"svf_{version}" / "processed_inputs"
            shutil.rmtree(pre.parent, ignore_errors=True)
            shutil.copytree(setup.preprocess_dir, pre, ignore=shutil.ignore_patterns("SVF"))

            def svf(version=version, pre=pre):
                with contextlib.ExitStack() as st:
                    for c in svf_patches(version):
                        st.enter_context(c)
                    calculate_svf(str(pre), patch_option=2, overwrite=True)

            res["geometry"][version] = timed(svf)
            print(f"[case] sky view, {version}: {res['geometry'][version]['seconds']:.0f} s, "
                  f"{res['geometry'][version]['peak_reserved_gib']:.2f} GiB", flush=True)
        a, b = (work / f"svf_{v}" / "processed_inputs" / "SVF" for v in ("v2.0.0", "ours"))
        n, bad = compare_files(a, b)
        zn = zb = 0
        for z in sorted(a.glob("*.zip")):
            m, d = compare_zips(z, b / z.name)
            zn, zb = zn + m, zb + d
        res["geometry"]["cache_files_differ"] = bad + zb
        res["geometry"]["cache_files_compared"] = n + zn
        out_json.write_text(json.dumps(res, indent=2))

    # the forecast day: deterministic run and ensemble, both versions, alternating
    pre = work / "day" / "processed_inputs"
    if not pre.exists():
        shutil.copytree(work / "svf_ours" / "processed_inputs", pre)
    members = [r for r in design() if r["sigma"] in (0.0, SIGMA) and r["member"] < args.members]
    res.setdefault("runs", {})
    fields = {}
    for r in members:
        for version in ("v2.0.0", "ours") if r["member"] % 2 else ("ours", "v2.0.0"):
            key = f"{r['id']}|{version}"
            dst = work / "days" / version / r["id"]
            if key in res["runs"] and (dst / "output_folder").exists():
                continue
            shutil.rmtree(dst, ignore_errors=True)

            def day(version=version, dst=dst, r=r):
                with contextlib.ExitStack() as st:
                    for c in day_patches(version):
                        st.enter_context(c)
                    st.enter_context(ta_error(pre / "metfiles", r["error"]))
                    run_utci_tiles(base_path=str(dst), preprocess_dir=str(pre), selected_date_str=args.date,
                                   save_tmrt=True)

            res["runs"][key] = timed(day)
            out_json.write_text(json.dumps(res, indent=2))
            print(f"[case] {r['id']:10s} {version:7s} {res['runs'][key]['seconds']:5.1f} s "
                  f"{res['runs'][key]['peak_reserved_gib']:.2f} GiB", flush=True)

    # same answer? per member, and the ensemble products
    diffs = {}
    for r in members:
        a = {v: read_stack(work / "days" / ver / r["id"], v) for v, ver in (("UTCI", "v2.0.0"), ("TMRT", "v2.0.0"))}
        b = {v: read_stack(work / "days" / "ours" / r["id"], v) for v in ("UTCI", "TMRT")}
        diffs[r["id"]] = {v: float(np.nanmax(np.abs(a[v].astype(np.float64) - b[v]))) for v in ("UTCI", "TMRT")}
        if r["member"] >= 0:
            for ver in ("v2.0.0", "ours"):
                fields.setdefault(ver, []).append(read_stack(work / "days" / ver / r["id"], "UTCI"))
    res["max_abs_diff_between_versions"] = diffs
    products = {}
    for ver, stack in fields.items():
        u = np.stack(stack)  # (members, hours, rows, cols)
        p4h32 = ((u > 32.0).sum(axis=1) >= 4).mean(axis=0)
        p38_15 = (u[:, 15] > 38.0).mean(axis=0)
        np.savez_compressed(work / f"products_{ver}.npz", p_4h_above_32=p4h32.astype(np.float32),
                            p_above_38_at_15=p38_15.astype(np.float32))
        products[ver] = {"mean_p_4h_above_32": float(np.nanmean(p4h32)), "mean_p_above_38_at_15": float(np.nanmean(p38_15))}
    a, b = (np.load(work / f"products_{v}.npz") for v in ("v2.0.0", "ours"))
    products["max_abs_diff"] = {k: float(np.nanmax(np.abs(a[k] - b[k]))) for k in a.files}
    products["pixels_differing"] = {k: int(np.sum(np.abs(a[k] - b[k]) > HALF_MEMBER)) for k in a.files}
    products["pixels"] = int(np.isfinite(a["p_4h_above_32"]).sum())
    res["products"] = products

    # (b) the shortcut: one physics run, UTCI recomputed per member from its Tmrt with the member's
    # air temperature (SOLWEIG-GPU's own UTCI function; v2.0.0 uses one Ta, RH and wind per hour, so
    # this is exact except that the member's Ta no longer feeds back into Tmrt through longwave)
    from osgeo import gdal
    from spatial_ta import utci as utci_fn
    from solweig_sens.perturb import MET_COLS

    t0 = time.perf_counter()
    met = np.loadtxt(args.met, skiprows=1)
    hour = met[:, 2].astype(int) % 24
    ta, rh, ws = (np.zeros(24) for _ in range(3))
    ta[hour], rh[hour], ws[hour] = (met[:, MET_COLS[c]] for c in ("ta", "rh", "wind"))
    tmrt0 = read_stack(work / "days" / "ours" / "base", "TMRT")
    lc = gdal.Open(str(args.rasters / "Landcover.tif")).ReadAsArray()[: tmrt0.shape[1], : tmrt0.shape[2]]
    outdoor = (lc != 2) & np.isfinite(tmrt0).all(axis=0)
    shortcut, errs = [], []
    phys = [m for m in members if m["member"] >= 0]
    for r, u_phys in zip(phys, fields["ours"]):
        e = np.asarray(r["error"])
        u = np.stack([utci_fn(np.full(outdoor.shape, ta[h] + e[h]), np.full(outdoor.shape, rh[h]), tmrt0[h],
                              np.full(outdoor.shape, ws[h])) for h in range(tmrt0.shape[0])])
        shortcut.append(u)
        d = np.abs(u - u_phys)[:, outdoor]
        errs.append({"mean_abs": float(np.nanmean(d)), "p99_abs": float(np.nanpercentile(d, 99)),
                     "max_abs": float(np.nanmax(d))})
    t_short = time.perf_counter() - t0
    u = np.stack(shortcut)
    p4h32 = ((u > 32.0).sum(axis=1) >= 4).mean(axis=0)
    p38_15 = (u[:, 15] > 38.0).mean(axis=0)
    phys_p = np.load(work / "products_ours.npz")
    dp = {"p_4h_above_32": np.abs(p4h32 - phys_p["p_4h_above_32"])[outdoor],
          "p_above_38_at_15": np.abs(p38_15 - phys_p["p_above_38_at_15"])[outdoor]}
    res["shortcut"] = {
        "what": "one physics run (ours, unperturbed), UTCI recomputed per member from its Tmrt with the member's Ta",
        "seconds": t_short,
        "member_utci_error_vs_full_physics": errs,
        "products_abs_diff_vs_full_physics": {k: {"mean": float(v.mean()), "p99": float(np.percentile(v, 99)),
                                                  "max": float(v.max()),
                                                  "pixels_differing_pct": float((v > HALF_MEMBER).mean() * 100)}
                                              for k, v in dp.items()}}
    print(f"[case] shortcut ensemble in {t_short:.0f} s: member UTCI error mean "
          f"{np.mean([x['mean_abs'] for x in errs]):.3f} C, max {max(x['max_abs'] for x in errs):.2f} C; "
          f"P(4 h > 32 C) differs at {res['shortcut']['products_abs_diff_vs_full_physics']['p_4h_above_32']['pixels_differing_pct']:.1f}% "
          f"of pixels", flush=True)

    # (c) the shortcut corrected to first order: one more physics run with the air temperature 1 K higher at
    # every hour gives each pixel's dTmrt/dTa per hour; a member's Tmrt is then the unperturbed Tmrt plus its
    # error times that slope (SOLWEIG's own physics keeps no heat from hour to hour, so each hour responds to
    # its own Ta), and its UTCI follows as in (b). Two physics runs for any number of members.
    plus = work / "days" / "ours" / "plus1K"
    if "plus1K|ours" not in res["runs"] or not (plus / "output_folder").exists():
        shutil.rmtree(plus, ignore_errors=True)

        def day_plus():
            with contextlib.ExitStack() as st:
                for c in day_patches("ours"):
                    st.enter_context(c)
                st.enter_context(ta_error(pre / "metfiles", [1.0] * 24))
                run_utci_tiles(base_path=str(plus), preprocess_dir=str(pre), selected_date_str=args.date,
                               save_tmrt=True)

        res["runs"]["plus1K|ours"] = timed(day_plus)
        out_json.write_text(json.dumps(res, indent=2))
    slope = read_stack(plus, "TMRT").astype(np.float64) - tmrt0  # K of Tmrt per K of Ta, (hours, rows, cols)
    t0 = time.perf_counter()
    corrected, errs_c = [], []
    for r, u_phys in zip(phys, fields["ours"]):
        e = np.asarray(r["error"])
        u = np.stack([utci_fn(np.full(outdoor.shape, ta[h] + e[h]), np.full(outdoor.shape, rh[h]),
                              tmrt0[h] + e[h] * slope[h], np.full(outdoor.shape, ws[h]))
                      for h in range(tmrt0.shape[0])])
        corrected.append(u)
        d = np.abs(u - u_phys)[:, outdoor]
        errs_c.append({"mean_abs": float(np.nanmean(d)), "p99_abs": float(np.nanpercentile(d, 99)),
                       "max_abs": float(np.nanmax(d))})
    t_corr = time.perf_counter() - t0
    u = np.stack(corrected)
    p4h32 = ((u > 32.0).sum(axis=1) >= 4).mean(axis=0)
    p38_15 = (u[:, 15] > 38.0).mean(axis=0)
    dp = {"p_4h_above_32": np.abs(p4h32 - phys_p["p_4h_above_32"])[outdoor],
          "p_above_38_at_15": np.abs(p38_15 - phys_p["p_above_38_at_15"])[outdoor]}
    sl = slope[:, outdoor]
    res["shortcut_corrected"] = {
        "what": "(b) plus each pixel's dTmrt/dTa per hour from one extra physics run with Ta + 1 K: two physics runs",
        "seconds_after_the_two_runs": t_corr,
        "slope_dtmrt_dta": {"median": float(np.nanmedian(sl)), "p1": float(np.nanpercentile(sl, 1)),
                            "p99": float(np.nanpercentile(sl, 99))},
        "member_utci_error_vs_full_physics": errs_c,
        "products_abs_diff_vs_full_physics": {k: {"mean": float(v.mean()), "p99": float(np.percentile(v, 99)),
                                                  "max": float(v.max()),
                                                  "pixels_differing_pct": float((v > HALF_MEMBER).mean() * 100)}
                                              for k, v in dp.items()}}
    print(f"[case] corrected shortcut (2 physics runs): dTmrt/dTa median {res['shortcut_corrected']['slope_dtmrt_dta']['median']:.2f}; "
          f"member UTCI error mean {np.mean([x['mean_abs'] for x in errs_c]):.3f} C, max "
          f"{max(x['max_abs'] for x in errs_c):.2f} C; P(4 h > 32 C) differs at "
          f"{res['shortcut_corrected']['products_abs_diff_vs_full_physics']['p_4h_above_32']['pixels_differing_pct']:.1f}% "
          f"of pixels", flush=True)
    t = {v: sum(x["seconds"] for k, x in res["runs"].items() if k.endswith("|" + v) and not k.startswith("plus1K"))
         for v in ("v2.0.0", "ours")}
    res["totals_seconds"] = {v: t[v] + res["geometry"][v]["seconds"] for v in t}
    out_json.write_text(json.dumps(res, indent=2))
    print(f"[case] total, geometry + {len(members)} days: v2.0.0 {res['totals_seconds']['v2.0.0'] / 60:.1f} min, "
          f"ours {res['totals_seconds']['ours'] / 60:.1f} min; largest member difference "
          f"{max(max(d.values()) for d in diffs.values()):.2g} C; products differ by "
          f"{max(products['max_abs_diff'].values()):.2g}", flush=True)


if __name__ == "__main__":
    main()
