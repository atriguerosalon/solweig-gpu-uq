"""Tmrt against the radiation scale at single hours: smooth, or steps at SOLWEIG's clearness switches?

Follow-up to radiation_linearity.py. Global radiation is scaled by f (0.8 to 1.2) at 7, 12 and 17 h
in the same run (an error at one hour hardly reaches later hours: radiation_memory.py), on the venue
tile of case_study.py, our code, SOLWEIG's own physics. Recorded per hour and f: domain-mean Tmrt,
and Tmrt of the pixel at the median of open sunlit ground. If SOLWEIG's diffuse split (piecewise in
the clearness index Kt, breaks at 0.3 and 0.78) and its clear-sky switch (CI < 0.95) drive the
nonlinearity, the curves show kinks or jumps at some f; a smooth curve would mean plain curvature.

Usage (after case_study.py):
    python scripts/radiation_sweep.py --tag expo
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

HOURS = (7, 12, 17)
SCALES = np.round(np.linspace(0.8, 1.2, 13), 4)


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from solweig_gpu import run_utci_tiles
    from case_study import day_patches
    from case_study_met import met_error
    from solweig_sens import plotstyle
    from solweig_sens.analysis import read_stack

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--date", default="2019-07-30")
    ap.add_argument("--work", type=Path, default=Path.home() / "case_study")
    ap.add_argument("--out", type=Path, default=ROOT / "figures")
    args = ap.parse_args()
    work = args.work / args.tag
    pre, days = work / "day" / "processed_inputs", work / "days" / "ours"
    torch.zeros(1, device="cuda")
    t0 = read_stack(days / "base", "TMRT").astype(np.float64)
    ok = np.isfinite(t0).all(axis=0)
    res = {"hours": list(HOURS), "scales": SCALES.tolist(), "mean": {h: [] for h in HOURS},
           "sunlit_median": {h: [] for h in HOURS}}
    for f in SCALES:
        dst = work / "sweep" / f"f{f:.4f}"
        if not (dst / "output_folder").exists():
            k = [1.0] * 24
            for h in HOURS:
                k[h] = float(f)
            shutil.rmtree(dst, ignore_errors=True)
            with contextlib.ExitStack() as st:
                for c in day_patches("ours"):
                    st.enter_context(c)
                st.enter_context(met_error(pre / "metfiles", [0.0] * 24, [1.0] * 24, k))
                run_utci_tiles(base_path=str(dst), preprocess_dir=str(pre), selected_date_str=args.date, save_tmrt=True)
        tm = read_stack(dst, "TMRT")
        for h in HOURS:
            x = tm[h][ok]
            sun = t0[h][ok] > np.percentile(t0[h][ok], 75)  # the sunlit, open part at that hour
            res["mean"][h].append(float(x.mean()))
            res["sunlit_median"][h].append(float(np.median(x[sun])))
        print(f"[sweep] f {f:.3f}: " + "  ".join(f"{h} h {res['mean'][h][-1]:.2f}" for h in HOURS), flush=True)
    (args.out / f"radiation_sweep_{args.tag}.json").write_text(json.dumps(res, indent=2))

    plotstyle.use()
    fig, axes = plt.subplots(1, len(HOURS), figsize=(13, 3.8))
    for ax, h in zip(axes, HOURS):
        for key, ls, lab in (("mean", "-", "domain mean"), ("sunlit_median", "--", "sunlit quarter, median")):
            y = np.array(res[key][h])
            ax.plot(SCALES, y - y[len(SCALES) // 2], "o" + ls, ms=4, label=lab)
        ax.axvline(1.0, color="0.5", lw=0.8, ls=":")
        ax.set_title(f"{h}:00")
        ax.set_xlabel("global radiation x f at that hour")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Tmrt change from f = 1 (K)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.text(0.5, -0.04, "Exposition Park venue, 2 x 2 km at 2 m, 30 July 2019, SOLWEIG-GPU's own physics. One run per f; "
             "the three hours perturbed together (an error at one hour barely reaches later hours).", ha="center",
             fontsize=8, color="0.35")
    fig.savefig(args.out / f"radiation_sweep_{args.tag}.png")
    plt.close(fig)
    print(f"[sweep] wrote {args.out / f'radiation_sweep_{args.tag}.json'} and .png")


if __name__ == "__main__":
    main()
