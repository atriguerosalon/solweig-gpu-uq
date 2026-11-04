"""Ground-temperature timing in Pacoima: sunrise in solar time against met data in clock time.

Solweig_2022a_calc shapes the daytime ground heating as a sine from sunrise to TmaxLST:
    Tg = Tgamp * sin(((dectime mod 1) - SNUP / 24) / (TmaxLST / 24 - SNUP / 24) * pi / 2)
SNUP = 12 - daylength / 2 is sunrise in local solar time, while dectime is the met file's clock
time (PDT in Los Angeles, where solar noon is near 13:00). So the curve starts about an hour
early. UMEP's SOLWEIG has the same formula; its ground heating was fitted in Gothenburg, where
summer clock time is also about an hour ahead of solar time.

One run with SNUP moved to clock time (+1.0 h for Pacoima on 30 July), against the baseline of
run_la_geometry.py (same geometry, fast functions).

Usage:
    python scripts/run_la_tg_timing.py --runs runs_la_geom
"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import sys
import textwrap
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solweig_sens.analysis import is_complete  # noqa: E402

# Solar noon in PDT at 118.42 W on 30 July: 12 h + (118.42 - 105) / 15 h - equation of time (-6.4 min)
SNUP_SHIFT_H = 1.0


@contextlib.contextmanager
def sunrise_shift(hours: float):
    import solweig_gpu.solweig as sw
    import solweig_gpu.utci_process as up

    src = textwrap.dedent(inspect.getsource(sw.Solweig_2022a_calc))
    old = "    _, _, _, SNUP = daylen(torch.tensor(jday.item()), torch.tensor(location['latitude']))\n"
    if src.count(old) != 1:
        raise RuntimeError("cannot patch SNUP in Solweig_2022a_calc")
    ns = vars(sw)  # compile into the live module, so it calls whatever functions are swapped in there
    original = sw.Solweig_2022a_calc
    try:
        exec(compile(src.replace(old, old + f"    SNUP = SNUP + {hours!r}\n"), sw.__file__, "exec"), ns)
        up.Solweig_2022a_calc = sw.Solweig_2022a_calc  # utci_process imported the name at load time
        yield
    finally:
        sw.Solweig_2022a_calc = original
        up.Solweig_2022a_calc = original


def main() -> None:
    from solweig_gpu import run_utci_tiles
    from solweig_sens.compact_masks import compact_masks
    from solweig_sens.fast import fast_solweig

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path("runs_la_geom"))
    ap.add_argument("--date", default="2019-07-30")
    args = ap.parse_args()
    runs = args.runs.resolve()
    out_dir = runs / "snup_clock" / "out"
    if is_complete(out_dir):
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with fast_solweig(), compact_masks("bits", svf=False), sunrise_shift(SNUP_SHIFT_H):
        run_utci_tiles(base_path=str(out_dir), preprocess_dir=str(runs / "baseline" / "shared" / "processed_inputs"),
                       selected_date_str=args.date, save_tmrt=True, save_shadow=True)
    cases = json.loads((runs / "cases.json").read_text())
    cases["snup_clock"] = {"snup_shift_h": SNUP_SHIFT_H}
    (runs / "cases.json").write_text(json.dumps(cases, indent=2))
    print(f"[tg] snup_clock done in {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
