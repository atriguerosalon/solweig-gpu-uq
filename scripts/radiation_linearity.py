"""Radiation x 0.9 at every hour against x 1.1 (case_study_met.py): is Tmrt linear in the radiation fraction?

Usage (after case_study_met.py): python scripts/radiation_linearity.py
"""

import sys as _sys

if __name__ == "__main__" and any(a in ("-h", "--help") for a in _sys.argv[1:]):
    print(__doc__)
    raise SystemExit(0)
import contextlib
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import torch  # noqa: E402
from solweig_gpu import run_utci_tiles  # noqa: E402
from case_study import day_patches  # noqa: E402
from case_study_met import met_error  # noqa: E402
from solweig_sens.analysis import read_stack  # noqa: E402

work = Path.home() / "case_study/expo"
pre, days = work / "day" / "processed_inputs", work / "days" / "ours"
dst = days / "kdown_minus"
if not (dst / "output_folder").exists():
    torch.zeros(1, device="cuda")
    shutil.rmtree(dst, ignore_errors=True)
    with contextlib.ExitStack() as st:
        for c in day_patches("ours"):
            st.enter_context(c)
        st.enter_context(met_error(pre / "metfiles", [0.0] * 24, [1.0] * 24, [0.9] * 24))
        run_utci_tiles(base_path=str(dst), preprocess_dir=str(pre), selected_date_str="2019-07-30", save_tmrt=True)
t0 = read_stack(days / "base", "TMRT").astype(np.float64)
ok = np.isfinite(t0).all(axis=0)
up = (read_stack(days / "kdown_plus", "TMRT") - t0)[:, ok]
dn = (read_stack(dst, "TMRT") - t0)[:, ok]
met = np.loadtxt(ROOT / "data/la/met_expo_20190730.txt", skiprows=1)
print("hour  Kdn   mean dT(+10%)  mean dT(-10%)  mean |dT+ + dT-| (curvature)  p99")
for h in range(6, 20):
    print(f"{h:4d} {met[h, 14]:6.0f}  {up[h].mean():+7.3f}  {dn[h].mean():+7.3f}   {np.abs(up[h] + dn[h]).mean():.3f}  {np.percentile(np.abs(up[h] + dn[h]), 99):.3f}")
