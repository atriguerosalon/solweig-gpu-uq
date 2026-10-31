"""Run the baseline and every perturbed case, and log run times.

Usage (from the repo root, inside the solweig environment):
    python scripts/run_all.py --rasters data/austin/Input_rasters --met data/ownmet_Forcing_data.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solweig_sens.analysis import is_complete  # noqa: E402
from solweig_sens.perturb import default_cases, restore_metfiles  # noqa: E402
from solweig_sens.run import Setup, prepare, run_case  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rasters", type=Path, required=True)
    ap.add_argument("--met", type=Path, required=True)
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--tile-size", type=int, default=1000)
    ap.add_argument("--only", nargs="*", help="run only these case names")
    ap.add_argument("--date", default="2020-08-13", help="date of the met file; sets sun position and UTC offset")
    args = ap.parse_args()

    # SOLWEIG-GPU resolves relative paths against its own base_path, so pass absolute ones.
    setup = Setup(raster_dir=args.rasters.resolve(), met_file=args.met.resolve(),
                  runs_dir=args.runs.resolve(), tile_size=args.tile_size, date=args.date)
    if not setup.preprocess_dir.exists():
        prepare(setup)

    restore_metfiles(setup.preprocess_dir / "metfiles")  # in case a previous batch was killed mid-case

    def done(name: str) -> bool:  # lets an interrupted batch resume
        return is_complete(setup.runs_dir / name)

    timings = {}
    if not done("baseline"):
        timings["baseline"] = run_case(setup, None)
    cases = [c for c in default_cases() if not args.only or c.name in args.only]
    for case in cases:
        if done(case.name):
            continue
        print(f"--- {case.name}", flush=True)
        timings[case.name] = run_case(setup, case)

    (setup.runs_dir / "timings.json").write_text(json.dumps(timings, indent=2))
    print(json.dumps(timings, indent=2))


if __name__ == "__main__":
    main()
