"""How much does SOLWEIG's Perez sky-clearness expression change Tmrt and UTCI?

SOLWEIG-GPU v2.0.0 (and UMEP's SOLWEIG, Perez_v3_moved.py, from which it is ported) compute the
Perez sky clearness as
    ((Idh + Ibn) / (Idh + 1.041 z^3)) / (1 + 1.041 z^3)
while the published definition (Perez et al. 1990, Solar Energy 44, 271-289; as implemented in
pvlib's irradiance.perez) is
    ((Idh + Ibn) / Idh + 1.041 z^3) / (1 + 1.041 z^3),
z the solar zenith in radians. At low sun the first drives the clearness towards 1, so clear skies
fall into Perez's most overcast categories; radiation_perez.py found the two remaining jumps of
the radiation sweeps at that boundary. Here one day is run with the expression as published (an
edited copy of Perez_v3, everything else unchanged) against the unchanged code, on the venue tile
for both days, and the difference in Tmrt and UTCI is reported by hour on outdoor pixels.
This is a check of the model, not a change to it: the published form is what Perez defined, but
SOLWEIG's coefficients and validation were obtained with its own code.

Usage (after shortcut_suite.py for expo_0730 and expo_0906):
    python scripts/perez_formula_effect.py
"""

from __future__ import annotations

import sys as _sys

if __name__ == "__main__" and any(a in ("-h", "--help") for a in _sys.argv[1:]):
    print(__doc__)
    raise SystemExit(0)

import contextlib
import inspect
import json
import shutil
import sys
import textwrap
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

CODE = "PerezClearness = ((Idh + Ibn) / (Idh + 1.041 * torch.pow(zen, 3))) / (1 + 1.041 * torch.pow(zen, 3))"
PAPER = "PerezClearness = ((Idh + Ibn) / Idh + 1.041 * torch.pow(zen, 3)) / (1 + 1.041 * torch.pow(zen, 3))"
CASES = {"30 Jul 2019": ("expo_0730", "2019-07-30"), "6 Sep 2020": ("expo_0906", "2020-09-06")}


@contextlib.contextmanager
def published_clearness():
    import solweig_gpu.solweig as sw

    src = textwrap.dedent(inspect.getsource(sw.Perez_v3))
    if src.count(CODE) != 1:
        raise RuntimeError("SOLWEIG source changed; cannot find the clearness line")
    ns = vars(sw)
    old = sw.Perez_v3
    exec(compile(src.replace(CODE, PAPER), sw.__file__, "exec"), ns)
    patched = ns["Perez_v3"]
    sw.Perez_v3 = patched
    try:
        yield
    finally:
        sw.Perez_v3 = old


def main() -> None:
    import torch
    from osgeo import gdal
    from solweig_gpu import run_utci_tiles
    from case_study import day_patches
    from solweig_sens.analysis import read_stack

    torch.zeros(1, device="cuda")
    lc = gdal.Open(str(ROOT / "data" / "la" / "expo_park" / "Landcover.tif")).ReadAsArray()
    res = {}
    for day, (tag, date) in CASES.items():
        pre = Path.home() / "shortcut_suite" / tag / "prep" / "shared" / "processed_inputs"
        out = {}
        for name, cm in (("as_coded", contextlib.nullcontext()), ("as_published", published_clearness())):
            dst = Path.home() / "perez_formula" / tag / name
            shutil.rmtree(dst, ignore_errors=True)
            with contextlib.ExitStack() as st:
                for c in day_patches("ours"):
                    st.enter_context(c)
                st.enter_context(cm)
                run_utci_tiles(base_path=str(dst), preprocess_dir=str(pre), selected_date_str=date, save_tmrt=True)
            out[name] = {v: read_stack(dst, v) for v in ("TMRT", "UTCI")}
        ok = (lc[: out["as_coded"]["TMRT"].shape[1], : out["as_coded"]["TMRT"].shape[2]] != 2) & \
            np.isfinite(out["as_coded"]["TMRT"]).all(axis=0)
        rec = {}
        for v in ("TMRT", "UTCI"):
            d = (out["as_published"][v] - out["as_coded"][v])[:, ok]
            rec[v] = {"mean_by_hour": np.nanmean(d, axis=1).tolist(), "p1_by_hour": np.nanpercentile(d, 1, axis=1).tolist(),
                      "p99_by_hour": np.nanpercentile(d, 99, axis=1).tolist(), "max_abs": float(np.nanmax(np.abs(d)))}
        res[day] = rec
        print(f"[perez-fix] {day}: published minus coded, outdoor mean (p1..p99), hours with any change:", flush=True)
        for v in ("TMRT", "UTCI"):
            r = rec[v]
            print(f"   {v}: " + "  ".join(f"{h}h {m:+.2f} ({lo:+.1f}..{hi:+.1f})" for h, (m, lo, hi) in
                                         enumerate(zip(r["mean_by_hour"], r["p1_by_hour"], r["p99_by_hour"]))
                                         if abs(lo) > 0.05 or abs(hi) > 0.05) + f"  | max |d| {r['max_abs']:.1f}", flush=True)
    path = ROOT / "figures" / "perez_formula_effect.json"
    path.write_text(json.dumps(res, indent=2))
    print(f"[perez-fix] wrote {path}")


if __name__ == "__main__":
    main()
