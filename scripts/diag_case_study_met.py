"""Where is the error of shortcut (d) in case_study_met.py? Per hour, against the member's radiation error.

Usage (after case_study_met.py): python scripts/diag_case_study_met.py
"""

import sys as _sys

if __name__ == "__main__" and any(a in ("-h", "--help") for a in _sys.argv[1:]):
    print(__doc__)
    raise SystemExit(0)
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from case_study_met import K_STEP, members  # noqa: E402
from solweig_sens.analysis import read_stack  # noqa: E402

days = Path.home() / "case_study/expo/days/ours"
t0 = read_stack(days / "base", "TMRT").astype(np.float64)
s_ta = read_stack(days / "plus1K", "TMRT") - t0
s_k = (read_stack(days / "kdown_plus", "TMRT") - t0) / K_STEP
ok = np.isfinite(t0).all(axis=0)
rows = []
for r in members(20)[:8]:
    e, k = np.asarray(r["ta_add"]), np.asarray(r["kdown_scale"])
    tm = read_stack(days / r["id"], "TMRT")
    pred = t0 + e[:, None, None] * s_ta + (k - 1)[:, None, None] * s_k
    d = (pred - tm)[:, ok]
    for h in range(24):
        rows.append((h, k[h] - 1, e[h], np.mean(np.abs(d[h])), np.percentile(np.abs(d[h]), 99), np.mean(d[h])))
a = np.array(rows)
print("hour | mean |dTmrt err| p99 (over 8 members) | corr with (k-1)^2")
for h in range(5, 21):
    x = a[a[:, 0] == h]
    print(f"{h:4d} | {x[:, 3].mean():.3f} {x[:, 4].mean():.3f} | k-1 range {x[:, 1].min():+.2f}..{x[:, 1].max():+.2f} | bias {x[:, 5].mean():+.3f}")
day = a[(a[:, 0] >= 8) & (a[:, 0] <= 18)]
print("daytime: corr(mean abs err, |k-1|) =", np.corrcoef(day[:, 3], np.abs(day[:, 1]))[0, 1],
      " corr with (k-1)^2 =", np.corrcoef(day[:, 3], day[:, 1] ** 2)[0, 1], " corr with |e_ta| =", np.corrcoef(day[:, 3], np.abs(day[:, 2]))[0, 1])
# is the error in the sunlit or shaded part? at 13 h, member 0
r = members(20)[0]
tm = read_stack(days / r["id"], "TMRT")
k = np.asarray(r["kdown_scale"]); e = np.asarray(r["ta_add"])
pred = t0 + e[:, None, None] * s_ta + (k - 1)[:, None, None] * s_k
for h in (9, 13, 17):
    d = np.abs(pred[h] - tm[h])[ok]
    sun = (s_k[h][ok] > np.median(s_k[h][ok]))
    print(f"m00 {h} h, k-1 {k[h]-1:+.3f}: err high-slope pixels {d[sun].mean():.3f} (p99 {np.percentile(d[sun], 99):.3f}), low-slope {d[~sun].mean():.3f}")
