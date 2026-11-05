"""Does a radiation error at one hour change Tmrt at later hours? (memory in SOLWEIG's radiation path)

Follow-up to case_study_met.py, where the per-hour radiation slope left a larger error than the air
temperature slope. One run on the same venue tile with the global radiation x (1 + STEP) at a
single hour only; the change in Tmrt against the unperturbed run is then reported hour by hour.
If SOLWEIG had no memory, only that hour would change. Its ground temperature model carries heat
from earlier hours (the delayed surface temperature in Tgmaps / TgOut, Lindberg et al. 2016) and its
clearness index is kept from earlier hours at low sun, so later hours may move too.

Usage (after case_study.py):
    python scripts/radiation_memory.py --tag expo --hour 9
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

STEP = 0.1


def main() -> None:
    import torch
    from solweig_gpu import run_utci_tiles
    from case_study import day_patches, timed
    from case_study_met import met_error
    from solweig_sens.analysis import read_stack

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--hour", type=int, nargs="+", default=[9])
    ap.add_argument("--date", default="2019-07-30")
    ap.add_argument("--work", type=Path, default=Path.home() / "case_study")
    ap.add_argument("--out", type=Path, default=ROOT / "figures")
    args = ap.parse_args()
    work = args.work / args.tag
    pre = work / "day" / "processed_inputs"
    days = work / "days" / "ours"
    out_json = args.out / f"radiation_memory_{args.tag}.json"
    res = json.loads(out_json.read_text()) if out_json.exists() else {"step": STEP, "hours": {}}
    torch.zeros(1, device="cuda")
    t0 = read_stack(days / "base", "TMRT").astype(np.float64)
    ok = np.isfinite(t0).all(axis=0)
    for h0 in args.hour:
        dst = days / f"kdown_only_{h0:02d}"
        if not (dst / "output_folder").exists():
            k = [1.0] * 24
            k[h0] = 1 + STEP

            def day(k=k, dst=dst):
                with contextlib.ExitStack() as st:
                    for c in day_patches("ours"):
                        st.enter_context(c)
                    st.enter_context(met_error(pre / "metfiles", [0.0] * 24, [1.0] * 24, k))
                    run_utci_tiles(base_path=str(dst), preprocess_dir=str(pre), selected_date_str=args.date,
                                   save_tmrt=True)

            shutil.rmtree(dst, ignore_errors=True)
            timed(day)
        d = (read_stack(dst, "TMRT") - t0)[:, ok]
        res["hours"][str(h0)] = {"mean_abs_dtmrt_by_hour": np.mean(np.abs(d), axis=1).tolist(),
                                 "p99_abs_dtmrt_by_hour": np.percentile(np.abs(d), 99, axis=1).tolist()}
        print(f"[mem] radiation x {1 + STEP} at {h0} h only; mean |dTmrt| by hour (K):", flush=True)
        print("      " + "  ".join(f"{h}:{v:.3f}" for h, v in enumerate(res["hours"][str(h0)]["mean_abs_dtmrt_by_hour"])
                                   if h >= 5), flush=True)
    out_json.write_text(json.dumps(res, indent=2))
    print(f"[mem] wrote {out_json}")


if __name__ == "__main__":
    main()
