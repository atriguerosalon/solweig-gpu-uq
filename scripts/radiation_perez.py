"""Are the two unexplained radiation jumps the Perez sky's clearness categories?

radiation_switches.py tied the 7 h jump (30 Jul) and the 12 h kink to the direct/diffuse split and
the clearness caps, but two jumps crossed none of those: 17 h on 30 Jul 2019 (radiation x 1.033 ->
x 1.067) and 8 h on 6 Sep 2020 (x 0.995 -> x 1.00, Tmrt +9.5 K). SOLWEIG-GPU's anisotropic sky
(Perez_v3, Perez et al. 1993) sorts the sky into 8 clearness categories by
PerezClearness = ((Idh + Ibn) / (Idh + 1.041 z^3)) / (1 + 1.041 z^3), z the zenith in radians, with
breaks at 1.065, 1.23, 1.5, 1.95, 2.8, 4.5 and 6.2, each category with its own luminance
coefficients. Here every Perez_v3 call is recorded (zenith, Idh, Ibn) and its clearness and
category recomputed with the same formula, for runs on both days with the radiation scaled at
every hour around the jumps. (The bracket placement in that formula differs from the published
definition, ((Idh + Ibn)/Idh + 1.041 z^3)/(1 + 1.041 z^3); both are reported.)

Usage (after shortcut_suite.py for expo_0730 and expo_0906):
    python scripts/radiation_perez.py
"""

from __future__ import annotations

import sys as _sys

if __name__ == "__main__" and any(a in ("-h", "--help") for a in _sys.argv[1:]):
    print(__doc__)
    raise SystemExit(0)

import contextlib
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

BREAKS = (1.065, 1.230, 1.500, 1.950, 2.800, 4.500, 6.200)
CASES = {"30 Jul 2019": ("expo_0730", "2019-07-30", 17, (1.033, 1.067)),
         "6 Sep 2020": ("expo_0906", "2020-09-06", 8, (0.95, 0.995, 1.0))}


def category(eps: float) -> int:
    return sum(eps >= b for b in BREAKS)


@contextlib.contextmanager
def recorded(log: list):
    import solweig_gpu.solweig as sw

    orig = sw.Perez_v3

    def wrapped(zen, azimuth, radD, radI, jday, patchchoice, patch_option):
        f = lambda x: float(x.item() if hasattr(x, "item") else x)  # noqa: E731
        z_deg, d, b = f(zen), f(radD), f(radI)
        z = math.radians(z_deg)
        k = 1.041 * z ** 3
        eps_code = ((d + b) / (d + k)) / (1 + k)
        eps_paper = ((d + b) / d + k) / (1 + k) if d > 0 else float("nan")
        log.append({"alt": 90 - z_deg, "Idh": d, "Ibn": b, "eps_code": eps_code, "cat_code": category(eps_code),
                    "eps_paper": eps_paper, "cat_paper": category(eps_paper) if d > 0 else None})
        return orig(zen, azimuth, radD, radI, jday, patchchoice, patch_option)

    sw.Perez_v3 = wrapped
    try:
        yield
    finally:
        sw.Perez_v3 = orig


def main() -> None:
    import torch
    from solweig_gpu import run_utci_tiles
    from case_study import day_patches
    from shortcut_suite import met_errors
    from solweig_sens.analysis import read_stack

    torch.zeros(1, device="cuda")
    res = {}
    for day, (tag, date, hour, scales) in CASES.items():
        pre = Path.home() / "shortcut_suite" / tag / "prep" / "shared" / "processed_inputs"
        if not (pre / "metfiles_pristine").exists():
            raise SystemExit(f"no pristine met files under {pre}: run shortcut_suite.py for {tag} first")
        res[day] = {"hour": hour, "runs": {}}
        for s in scales:
            dst = Path.home() / "radiation_perez" / tag / f"f{s:.3f}"
            shutil.rmtree(dst, ignore_errors=True)
            log: list = []
            k = [1.0] * 24
            k[hour] = s
            with contextlib.ExitStack() as st:
                for c in day_patches("ours"):
                    st.enter_context(c)
                st.enter_context(met_errors(pre / "metfiles", [0.0] * 24, [1.0] * 24, k, False))
                st.enter_context(recorded(log))
                run_utci_tiles(base_path=str(dst), preprocess_dir=str(pre), selected_date_str=date, save_tmrt=True)
            tm = read_stack(dst, "TMRT")
            ok = np.isfinite(tm).all(axis=0)
            # Perez is called once per daytime step, in order; match the step to the hour by its radiation
            res[day]["runs"][f"{s:.3f}"] = {"calls": log, "tmrt_mean_at_hour": float(tm[hour][ok].mean())}
        print(f"[perez] {day}, radiation scaled at {hour} h only; mean Tmrt at {hour} h: " + ", ".join(
            f"x {s} {r['tmrt_mean_at_hour']:.2f} C" for s, r in res[day]["runs"].items()), flush=True)
        runs = list(res[day]["runs"].items())
        for i in range(min(len(r["calls"]) for _, r in runs)):  # Perez calls in order, one per daytime step
            cs = [r["calls"][i] for _, r in runs]
            if len({round(c["Idh"], 3) for c in cs}) > 1:  # the step whose radiation was scaled
                print(f"   step at sun altitude {cs[0]['alt']:.1f} deg: " + " | ".join(
                    f"x {s}: Idh {c['Idh']:.1f}, Ibn {c['Ibn']:.1f}, eps {c['eps_code']:.3f} cat {c['cat_code']} "
                    f"(published formula {c['eps_paper']:.3f} cat {c['cat_paper']})" for (s, _), c in zip(runs, cs)), flush=True)
    out = ROOT / "figures" / "radiation_perez.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"[perez] wrote {out}")


if __name__ == "__main__":
    main()
