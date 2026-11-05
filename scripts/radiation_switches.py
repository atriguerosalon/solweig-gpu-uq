"""Which clearness switch makes Tmrt jump? Log SOLWEIG's clearness values per hour while scaling radiation.

Follow-up to radiation_sweep.py. SOLWEIG-GPU v2.0.0 (solweig.py, Solweig_2022a_calc, daytime) turns
the global radiation into three clearness values, each with a switch:

    Kt      clearness index: the diffuse/direct split (diffusefraction, Reindl et al. 1990) is
            piecewise in Kt, breaks at 0.3 and 0.78
    CI      clearness index after Crawford and Duchon, capped at 1; the sky longwave is blended
            towards cloudy when CI < 0.95
    CI_TgG  global radiation over its clear-sky value plus a low-sun correction, capped at 1; it
            scales the ground and wall temperature amplitude

The run is wrapped so every call of clearnessindex_2013b and diffusefraction is recorded; CI_TgG is
recomputed from the recorded values exactly as in the code. Radiation is scaled by f at 7, 12 and
17 h, as in radiation_sweep.py, for f around the jumps found there.

Usage (after case_study.py):
    python scripts/radiation_switches.py --tag expo
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

HOURS = (7, 12, 17)
SCALES = (0.9, 0.933, 0.967, 1.0, 1.033, 1.067, 1.1)


@contextlib.contextmanager
def recorded(log: list):
    """Wrap clearnessindex_2013b and diffusefraction in solweig_gpu.solweig; append what they see and return."""
    import solweig_gpu.solweig as sw

    ci0, df0 = sw.clearnessindex_2013b, sw.diffusefraction
    f = lambda x: float(x.item() if hasattr(x, "item") else x)  # noqa: E731

    def ci(zen, jday, Ta, RH, radG, location, P):
        out = ci0(zen, jday, Ta, RH, radG, location, P)
        log.append({"call": "ci", "zen_deg": math.degrees(f(zen)), "radG": f(radG), "I0": f(out[0]), "CI": f(out[1]),
                    "Kt": f(out[2])})
        return out

    def df(radG, altitude, Kt, Ta, RH):
        out = df0(radG, altitude, Kt, Ta, RH)
        log.append({"call": "df", "radG": f(radG), "alt": f(altitude), "Kt": f(Kt), "radI": f(out[0]), "radD": f(out[1])})
        return out

    sw.clearnessindex_2013b, sw.diffusefraction = ci, df
    try:
        yield
    finally:
        sw.clearnessindex_2013b, sw.diffusefraction = ci0, df0


def per_hour(log: list) -> list[dict]:
    """Group the calls of each daytime step: ci (once or twice), df(radG, Kt), df(I0, 1)."""
    out, cur = [], {}
    for e in log:
        if e["call"] == "ci":
            cur.update(CI=min(e["CI"], 1.0), Kt=e["Kt"], zen=e["zen_deg"], radG=e["radG"])
        elif e["Kt"] == 1.0:  # radI0, radD0 for the clear sky
            alt = e["alt"]
            radG0 = e["radI"] * math.sin(math.radians(alt)) + e["radD"]
            corr = 0.1473 * math.log(90 - cur["zen"]) + 0.3454
            cur.update(alt=alt, CI_TgG=min(cur["radG"] / radG0 + (1 - corr), 1.0),
                       CI_TgG_uncapped=cur["radG"] / radG0 + (1 - corr))
            out.append(cur)
            cur = {}
        else:
            cur["Kt_split"] = e["Kt"]
    return out


def main() -> None:
    import torch
    from solweig_gpu import run_utci_tiles
    from case_study import day_patches
    from case_study_met import met_error
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
    ok = np.isfinite(read_stack(days / "base", "TMRT")).all(axis=0)
    met = np.loadtxt(ROOT / "data" / "la" / f"met_{args.tag}_20190730.txt", skiprows=1)
    day_hours = [int(h) for h in met[:, 2] if met[int(h), 14] > 0]
    res = {"hours": list(HOURS), "scales": list(SCALES), "runs": {}}
    for s in SCALES:
        dst = work / "switches" / f"f{s:.3f}"
        shutil.rmtree(dst, ignore_errors=True)
        k = [1.0] * 24
        for h in HOURS:
            k[h] = s
        log: list = []
        with contextlib.ExitStack() as st:
            for c in day_patches("ours"):
                st.enter_context(c)
            st.enter_context(met_error(pre / "metfiles", [0.0] * 24, [1.0] * 24, k))
            st.enter_context(recorded(log))
            run_utci_tiles(base_path=str(dst), preprocess_dir=str(pre), selected_date_str=args.date, save_tmrt=True)
        steps = per_hour(log)
        tm = read_stack(dst, "TMRT")
        # SOLWEIG's daytime steps are the hours with the sun up; match them to the hours of the met file by radiation
        rec = {}
        for h in HOURS:
            radG = met[h, 14] * s
            st_h = min(steps, key=lambda x: abs(x["radG"] - radG) + 1e3 * (x["radG"] <= 0))
            rec[h] = {**{q: st_h[q] for q in ("radG", "alt", "Kt", "CI", "CI_TgG", "CI_TgG_uncapped")},
                      "tmrt_mean": float(tm[h][ok].mean())}
        res["runs"][f"{s:.3f}"] = rec
        print(f"[sw] f {s:.3f}: " + " | ".join(
            f"{h} h Kt {rec[h]['Kt']:.3f} CI {rec[h]['CI']:.3f} CI_TgG {rec[h]['CI_TgG_uncapped']:.3f} Tmrt {rec[h]['tmrt_mean']:.2f}"
            for h in HOURS), flush=True)
    res["daytime_steps_logged"] = len(steps)
    res["daytime_hours_in_met"] = len(day_hours)
    (args.out / f"radiation_switches_{args.tag}.json").write_text(json.dumps(res, indent=2))
    print(f"[sw] wrote {args.out / f'radiation_switches_{args.tag}.json'}")


if __name__ == "__main__":
    main()
