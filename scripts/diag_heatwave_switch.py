"""Is the heatwave-morning shortcut failure the Kt = 0.78 switch? (shortcut_suite.py, expo_0906, member A_m05, 8 h)

At 8 h on 6 Sep 2020, member A_m05 (radiation scale 0.995, Ta error about -0.7 K) has Tmrt 17.7 K below
what the radiation table predicts at some pixels. Hypothesis: the member's Kt sits on the other side
of the diffuse split's break at 0.78 from the table runs around it, and its Ta error helps put it
there (the clearness index uses Ta and RH). Test: rerun the member, the two table runs around its
scale (f = 0.95 and 1.00) and the member's Ta error alone (radiation unperturbed), with SOLWEIG's
clearness values logged per hour (radiation_switches.py), and compare Kt at 6-10 h.

Result (4 Nov): the hypothesis is wrong. At 8 h (sun 11.5 deg) Kt is 1.23-1.29 in every case, CI is
capped at 1 and CI_TgG is above its cap, and the Ta error alone leaves Kt unchanged; yet mean Tmrt
jumps 9.5 K between radiation x 0.995 and x 1.00. The switch is elsewhere (not identified; next
suspect: the discrete sky-clearness categories of the Perez anisotropic sky).

Usage (after shortcut_suite.py --tag expo_0906):
    python scripts/diag_heatwave_switch.py
"""

from __future__ import annotations

import sys as _sys

if __name__ == "__main__" and any(a in ("-h", "--help") for a in _sys.argv[1:]):
    print(__doc__)
    raise SystemExit(0)

import contextlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

DATE = "2020-09-06"
HOURS = (6, 7, 8, 9, 10)


def main() -> None:
    import torch
    from solweig_gpu import run_utci_tiles
    from case_study import day_patches
    from radiation_switches import per_hour, recorded
    from shortcut_suite import ensemble, met_errors
    from solweig_sens.analysis import read_stack

    work = Path.home() / "shortcut_suite" / "expo_0906"
    pre = work / "prep" / "shared" / "processed_inputs"
    if not (pre / "metfiles_pristine").exists():
        if (pre / "metfiles_orig").exists():
            raise SystemExit("leftover metfiles_orig: check which copy is unperturbed first")
        shutil.copytree(pre / "metfiles", pre / "metfiles_pristine")
    m = next(r for r in ensemble("A", "independent") if r["id"] == "A_m05")
    one, zero = [1.0] * 24, [0.0] * 24
    cases = {"member A_m05": (m["ta_add"], m["wind_scale"], m["kdown_scale"]),
             "member's Ta error only": (m["ta_add"], one, one),
             "table f = 0.95": (zero, one, [0.95] * 24), "table f = 1.00 (base)": (zero, one, one)}
    torch.zeros(1, device="cuda")
    res = {"member_kdown_scale": {h: m["kdown_scale"][h] for h in HOURS},
           "member_ta_error": {h: m["ta_add"][h] for h in HOURS}, "cases": {}}
    tm = {}
    for name, (ta, ws, ks) in cases.items():
        dst = work / "diag_switch" / name.replace(" ", "_").replace("'", "").replace("=", "").replace(".", "")
        shutil.rmtree(dst, ignore_errors=True)
        log: list = []
        with contextlib.ExitStack() as st:
            for c in day_patches("ours"):
                st.enter_context(c)
            st.enter_context(met_errors(pre / "metfiles", ta, ws, ks, False))
            st.enter_context(recorded(log))
            run_utci_tiles(base_path=str(dst), preprocess_dir=str(pre), selected_date_str=DATE, save_tmrt=True)
        steps = per_hour(log)
        tm[name] = read_stack(dst, "TMRT")
        res["cases"][name] = {"steps": [{q: s[q] for q in ("radG", "alt", "Kt", "CI", "CI_TgG_uncapped")} for s in steps]}
    ok = np.isfinite(tm["table f = 1.00 (base)"]).all(axis=0)
    print("Kt per daytime step (sun up), first steps:")
    for name, r in res["cases"].items():
        print(f"  {name:24s} " + "  ".join(f"{s['alt']:4.1f} deg: Kt {s['Kt']:.3f}" for s in r["steps"][:5]))
    for h in (8, 9):
        a, b, c = (tm[n][h][ok] for n in ("table f = 0.95", "table f = 1.00 (base)", "member A_m05"))
        print(f"{h} h mean Tmrt: f 0.95 {a.mean():.2f}, f 1.00 {b.mean():.2f}, member {c.mean():.2f} "
              f"(member scale {m['kdown_scale'][h]:.3f}, Ta error {m['ta_add'][h]:+.2f} K)")
        res[f"tmrt_mean_{h}h"] = {"f095": float(a.mean()), "f100": float(b.mean()), "member": float(c.mean())}
    out = ROOT / "figures" / "shortcuts" / "diag_heatwave_switch.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
