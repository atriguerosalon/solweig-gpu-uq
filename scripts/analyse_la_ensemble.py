"""Bias and spread of SOLWEIG against the Pacoima measurements (runs from run_la_ensemble.py).

For each measured time: the ensemble's median and 5-95% range of the median Tmrt on open sunlit
ground (over regular asphalt), and of the coated-minus-regular difference (paired samples). Then
bias = ensemble median - measurement, spread = ensemble standard deviation, and whether the
measurement falls inside the 90% range. If |bias| is much larger than the spread, varying the
inputs cannot explain the error: the model's physics (or the comparison) is the problem.

Usage:
    python scripts/analyse_la_ensemble.py --ens runs_la_ens --out figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyse_la_pavement import MEAS_DTMRT, MEAS_TMRT_ASPHALT, at  # noqa: E402


def summary(values: np.ndarray, measured: float) -> dict:
    lo, med, hi = np.percentile(values, [5, 50, 95])
    return {"measured": measured, "median": float(med), "p5": float(lo), "p95": float(hi),
            "std": float(values.std(ddof=1)), "bias": float(med - measured), "inside_90": bool(lo <= measured <= hi)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ens", type=Path, default=Path("runs_la_ens"))
    ap.add_argument("--out", type=Path, default=Path("figures"))
    args = ap.parse_args()
    n = json.loads((args.ens / "design.json").read_text())["n"]
    done = [i for i in range(n) if all((args.ens / f"{c}_{i:03d}.json").exists() for c in ("regular", "coated"))]
    med = {c: np.array([[h[1] for h in json.loads((args.ens / f"{c}_{i:03d}.json").read_text())["TMRT"]]
                        for i in done]) for c in ("regular", "coated")}  # (samples, hours): median Tmrt per hour
    out = {"samples": len(done), "absolute_regular": {}, "coated_minus_regular": {}}
    for t, m in MEAS_TMRT_ASPHALT.items():
        out["absolute_regular"][t] = summary(np.array([at(s, t) for s in med["regular"]]), m)
    diff = med["coated"] - med["regular"]
    for t, m in MEAS_DTMRT.items():
        out["coated_minus_regular"][t] = summary(np.array([at(s, t) for s in diff]), m)
    path = args.out / "la_ensemble.json"
    path.write_text(json.dumps(out, indent=2))

    print(f"{len(done)} paired samples (regular and coated asphalt)")
    for key, title in (("absolute_regular", "Tmrt over regular asphalt"),
                       ("coated_minus_regular", "coated minus regular")):
        print(f"== {title}: time PDT | measured | ensemble median [5-95%] | std | bias | inside")
        for t, s in out[key].items():
            print(f"   {t:5.1f} | {s['measured']:6.1f} | {s['median']:6.1f} [{s['p5']:6.1f} {s['p95']:6.1f}] | "
                  f"{s['std']:5.2f} | {s['bias']:+6.1f} | {'yes' if s['inside_90'] else 'NO'}")
    print(f"[ens] report: {path}")


if __name__ == "__main__":
    main()
