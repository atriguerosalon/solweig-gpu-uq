"""Where are the 14-17 C member errors of the shortcuts on the heatwave day? (shortcut_suite.py, expo_0906)

For every member of ensemble A, the table shortcut (f) is rebuilt and its largest UTCI errors are
located: hour, air temperature, Tmrt (full physics and shortcut), Tmrt - Ta and wind. UTCI's
operational polynomial (Broede et al. 2012) is valid for Ta -50 to +50 C, Tmrt - Ta -30 to +70 K and
wind 0.5 to 17 m/s (at 10 m); outside that it is an extrapolation.

Usage (after shortcut_suite.py --tag expo_0906):
    python scripts/diag_heatwave_utci.py
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
from shortcut_suite import GRID, ensemble  # noqa: E402
from solweig_sens.analysis import read_stack  # noqa: E402
from solweig_sens.perturb import MET_COLS  # noqa: E402
from spatial_ta import utci as utci_fn  # noqa: E402

runs = Path.home() / "shortcut_suite/expo_0906/runs_independent"
met = np.loadtxt(ROOT / "data/la/met_expo_20200906.txt", skiprows=1)
ta, rh, ws = (met[:, MET_COLS[c]] for c in ("ta", "rh", "wind"))
t0 = read_stack(runs / "base", "TMRT").astype(np.float64)
s_ta = read_stack(runs / "plus1K", "TMRT") - t0
table = np.stack([t0 if f == 1.0 else read_stack(runs / f"table_{f:.2f}", "TMRT") for f in GRID])
ok = np.isfinite(t0).all(axis=0)
big_all = 0
for r in ensemble("A", "independent"):
    e, w, kf = (np.asarray(r[x]) for x in ("ta_add", "wind_scale", "kdown_scale"))
    full_u, full_t = read_stack(runs / r["id"], "UTCI"), read_stack(runs / r["id"], "TMRT")
    for h in range(24):
        k = min(max(kf[h], GRID[0]), GRID[-1])
        i = min(int(np.searchsorted(GRID, k, side="right")) - 1, len(GRID) - 2)
        wt = (k - GRID[i]) / (GRID[i + 1] - GRID[i])
        tm = (1 - wt) * table[i, h] + wt * table[i + 1, h] + e[h] * s_ta[h]
        u = utci_fn(np.full(ok.shape, ta[h] + e[h]), np.full(ok.shape, rh[h]), tm, np.full(ok.shape, ws[h] * w[h]))
        d = np.where(ok, np.abs(u - full_u[h]), 0)
        big = d > 5
        big_all += int(big.sum())
        if big.any():
            j = np.unravel_index(np.argmax(d), d.shape)
            print(f"{r['id']} {h:2d} h: {int(big.sum())} px > 5 C; worst: Ta {ta[h] + e[h]:.1f}, wind {ws[h] * w[h]:.2f} m/s, "
                  f"Tmrt full {full_t[h][j]:.1f} / shortcut {tm[j]:.1f} (Tmrt - Ta {full_t[h][j] - ta[h] - e[h]:.1f}), "
                  f"UTCI full {full_u[h][j]:.1f} / shortcut {u[j]:.1f}")
print(f"pixels-hours with error > 5 C in 20 members: {big_all} of {20 * 24 * int(ok.sum())}")
