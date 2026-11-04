"""Shortcut against full rerun for plantings of several trees (runs from run_la_treeplant.py).

For each planting S, the true effect is Tmrt(S) - Tmrt(no new trees), hour by hour. Two shortcuts
built from the single-tree runs:

    sum         the single-tree effects added up (each one exact, so this is the best case of an
                additive shortcut; TreePlanter also skips the sky view factor change)
    strongest   at each pixel the single-tree effect of largest magnitude, i.e. overlapping
                shade counted once

Error maps on outdoor pixels, reported where the planting changes Tmrt by more than 0.5 K
(either truth or shortcut): RMSE, worst pixel, share of pixels off by more than 1 K, and the
area-integrated effect (K x pixels) against the truth.

Usage:
    python scripts/analyse_la_treeplant.py --runs ~/treeplant --out figures
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solweig_sens.analysis import outdoor_mask, read_stack  # noqa: E402

HOURS = {"night (00-05)": range(0, 6), "morning (07-09)": range(7, 10), "midday (11-15)": range(11, 16),
         "evening (17-19)": range(17, 20)}


def metrics(pred: np.ndarray, truth: np.ndarray, overlap: np.ndarray) -> dict:
    """pred, truth: (hours, pixels) on outdoor pixels; overlap: where two or more trees act alone."""
    hit = (np.abs(truth) > 0.5) | (np.abs(pred) > 0.5)
    e = (pred - truth)[hit]
    err = np.abs(pred - truth)
    return {"affected_px_per_hour": float(hit.sum() / len(truth)),
            "rmse": float(np.sqrt((e ** 2).mean())) if e.size else 0.0,
            "max_abs": float(np.abs(e).max()) if e.size else 0.0,
            "pct_off_1K": float(100 * (np.abs(e) > 1).mean()) if e.size else 0.0,
            "area_truth": float(truth.sum() / len(truth)), "area_pred": float(pred.sum() / len(truth)),
            # mechanism: share of the absolute error inside overlapping single-tree footprints
            "error_share_in_overlap": float(err[overlap].sum() / err.sum()) if err.sum() else 0.0,
            "overlap_px_per_hour": float(overlap.sum() / len(truth))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path.home() / "treeplant")
    ap.add_argument("--out", type=Path, default=Path("figures"))
    args = ap.parse_args()
    design = json.loads((args.runs / "design.json").read_text())
    sites = np.array(design["sites"])
    tm = {name: read_stack(args.runs / name / "out", "TMRT") for name in design["cases"]
          if (args.runs / name / "out").exists()}
    base = tm.pop("base")
    shape = base.shape[1:]
    outdoor = outdoor_mask(args.runs / "base" / "rasters", shape)
    flat = lambda a: a[:, outdoor]  # noqa: E731
    single = {int(n.split("_")[1]): flat(tm[n] - base) for n in tm if n.startswith("single_")}
    # the mechanism: pixels a tree newly shades (Shadow 1 = sunlit) and pixels under its crown
    sh_base = read_stack(args.runs / "base" / "out", "Shadow")
    y, x = np.ogrid[: shape[0], : shape[1]]
    r = design.get("crown_radius_px", 3.0)
    r = r if isinstance(r, list) else [r] * len(sites)  # per tree in the random-pair design
    reach = {i: flat((sh_base - read_stack(args.runs / f"single_{i:02d}" / "out", "Shadow")) > 0.5)
             | ((y - sites[i][0]) ** 2 + (x - sites[i][1]) ** 2 <= r[i] ** 2)[outdoor][None]
             for i in single}

    out = {"design": design, "cases": {}}
    for name, idx in design["cases"].items():
        if len(idx) < 2 or name not in tm or not all(i in single for i in idx):
            continue
        truth = flat(tm[name] - base)
        stack = np.stack([single[i] for i in idx])
        pick = np.take_along_axis(stack, np.abs(stack).argmax(0)[None], 0)[0]
        overlap = (np.abs(stack) > 0.5).sum(0) >= 2  # where two trees each change Tmrt by > 0.5 K
        shade_overlap = np.stack([reach[i] for i in idx]).sum(0) >= 2  # shaded by, or under, two trees
        gaps = [float(np.hypot(*(sites[a] - sites[b])) * 2) for a, b in combinations(idx, 2)]  # metres
        rec = {"trees": idx, "closest_pair_m": min(gaps), "by_hours": {}}
        for label, hrs in HOURS.items():
            h = list(hrs)
            # shade counted once where the trees' shade or crowns overlap, effects added elsewhere
            hybrid = np.where(shade_overlap, pick, stack.sum(0))
            rec["by_hours"][label] = {"sum": metrics(stack.sum(0)[h], truth[h], shade_overlap[h]),
                                      "strongest": metrics(pick[h], truth[h], shade_overlap[h]),
                                      "hybrid": metrics(hybrid[h], truth[h], shade_overlap[h]),
                                      "hybrid_effect_threshold": metrics(np.where(overlap, pick, stack.sum(0))[h],
                                                                         truth[h], overlap[h])}
        out["cases"][name] = rec
    path = args.out / f"la_{args.runs.name}.json"  # la_treeplant.json, la_treepairs.json
    path.write_text(json.dumps(out, indent=1))

    print(f"{len(single)} single-tree runs; outdoor pixels {int(outdoor.sum())}")
    print("case        trees closest | hours            | sum: RMSE  max  >1K%  area/truth  in-overlap"
          " | strongest: RMSE  max  area/truth | hybrid: RMSE  max  area/truth")
    ratio = lambda x: x["area_pred"] / x["area_truth"] if x["area_truth"] else float("nan")  # noqa: E731
    for name, rec in out["cases"].items():
        for label, r in rec["by_hours"].items():
            s, m, y = r["sum"], r["strongest"], r["hybrid"]
            print(f"{name:11s} {len(rec['trees']):5d} {rec['closest_pair_m']:5.0f} m | {label:16s} | "
                  f"{s['rmse']:5.2f} {s['max_abs']:5.1f} {s['pct_off_1K']:5.1f} {ratio(s):6.2f} "
                  f"{100 * s['error_share_in_overlap']:5.0f}% | "
                  f"{m['rmse']:5.2f} {m['max_abs']:5.1f} {ratio(m):6.2f} | "
                  f"{y['rmse']:5.2f} {y['max_abs']:5.1f} {ratio(y):6.2f}")
    print(f"[plant] report: {path}")


if __name__ == "__main__":
    main()
