"""Pacoima again, with ground heat kept after sunset (experimental patch), for three time constants.

Uses the fitted ground heating of run_la_pavement.py (regular and coated) and the shared
geometry in runs_la. Compare with the measurements of Middel et al. (2020) at 21:00 PDT:
Tmrt over asphalt 24.9 C, coated minus regular -1.0 C.

Usage:
    python scripts/run_la_storage.py --runs runs_la
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_la_pavement import CASES  # noqa: E402
from solweig_sens.analysis import is_complete  # noqa: E402
from solweig_sens.heat_storage import ground_heat_storage  # noqa: E402
from solweig_sens.perturb import PAVED_CLASSES, patched_landcover  # noqa: E402

TAUS = (1.0, 2.0, 4.0)


def main() -> None:
    from solweig_gpu import run_utci_tiles

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path("runs_la"))
    ap.add_argument("--date", default="2019-07-30")
    args = ap.parse_args()
    runs = args.runs.resolve()
    preprocess = runs / "shared" / "processed_inputs"

    for tau in TAUS:
        for base in ("regular_fit", "coated_fit"):
            name = f"{base}_storage_tau{tau:g}h"
            out_dir = runs / name
            if is_complete(out_dir):
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            t0 = time.perf_counter()
            with patched_landcover(out_dir, PAVED_CLASSES, CASES[base]), ground_heat_storage(tau):
                run_utci_tiles(base_path=str(out_dir), preprocess_dir=str(preprocess),
                               selected_date_str=args.date, save_tmrt=True)
            print(f"[storage] {name} done in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
