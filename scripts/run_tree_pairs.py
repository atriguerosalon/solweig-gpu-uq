"""When do two trees stop adding up? A spacing and orientation sweep on open ground.

Controlled companion to run_la_treeplant.py: flat paved ground (the open scene of
single_tree_whatif.py, 400 x 400 m at 2 m, Pacoima forcing of 30 July 2019), tree A in the
centre, tree B at 8-60 m east or north of it (10 m tall, 12 m crowns; below 12 m the crowns
touch). Runs: no trees, A alone, each B alone, and each pair, all full reruns (own sky view
factor). East-west pairs share long morning and evening shadows; north-south pairs share the
short midday ones. Same design format as run_la_treeplant.py, so analyse_la_treeplant.py reads it:

    python scripts/run_tree_pairs.py
    python scripts/analyse_la_treeplant.py --runs ~/treepairs --out figures

With --random N: N pairs with random spacing (6-40 m), direction, heights (6-16 m) and crown radii
(4-9 m), each pair also run tree by tree; training and test data for a learned correction of the
shortcut, which only fails where the two trees' shade or crowns overlap (~/treepairs_random):

    python scripts/run_tree_pairs.py --random 30 --seed 0
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from single_tree_whatif import DATE, MET, N, disk, write_rasters  # noqa: E402
from solweig_sens.analysis import is_complete  # noqa: E402
from solweig_sens.run import Setup, prepare  # noqa: E402

HEIGHT, CROWN = 10.0, 3.0
SPACINGS_M = [8, 12, 16, 20, 28, 40, 60]
A = (N // 2, N // 2)


def sweep() -> dict:
    """B to the east (same row) and to the north (row index decreases northwards), same trees."""
    sites = [A] + [(A[0], A[1] + d // 2) for d in SPACINGS_M] + [(A[0] - d // 2, A[1]) for d in SPACINGS_M]
    cases = {"base": [], **{f"single_{i:02d}": [i] for i in range(len(sites))},
             **{f"pair_{'ew' if i <= len(SPACINGS_M) else 'ns'}_{2 * max(abs(sites[i][0] - A[0]), abs(sites[i][1] - A[1])):02d}m": [0, i]
                for i in range(1, len(sites))}}
    return {"sites": [list(s) for s in sites], "cases": cases, "height_m": [HEIGHT] * len(sites),
            "crown_radius_px": [CROWN] * len(sites)}


def random_pairs(n: int, seed: int, overlap: bool = False) -> dict:
    """n pairs: A in the centre, B 6-40 m away in any direction; heights 6-16 m, crown radii 4-9 m.

    With overlap, B's distance is 30-95% of the two crown radii together, so the crowns always overlap
    (the case where shade counted once fails); otherwise the draws are in the original order.
    """
    rng = np.random.default_rng(seed)
    sites, heights, radii, cases = [], [], [], {"base": []}
    for k in range(n):
        if overlap:
            r = [round(float(v), 2) for v in rng.uniform(2, 4.5, 2)]
            d, theta = rng.uniform(0.3, 0.95) * (r[0] + r[1]), rng.uniform(0, np.pi)
            h = [round(float(v), 1) for v in rng.uniform(6, 16, 2)]
        else:
            d, theta = rng.uniform(6, 40) / 2, rng.uniform(0, np.pi)  # pixels; direction from east, anticlockwise
            h = [round(float(v), 1) for v in rng.uniform(6, 16, 2)]
            r = [round(float(v), 2) for v in rng.uniform(2, 4.5, 2)]
        b = (int(round(A[0] - d * np.sin(theta))), int(round(A[1] + d * np.cos(theta))))
        sites += [list(A), list(b)]
        heights += h
        radii += r
        cases |= {f"single_{2 * k:02d}": [2 * k], f"single_{2 * k + 1:02d}": [2 * k + 1], f"pair_{k:02d}": [2 * k, 2 * k + 1]}
    return {"sites": sites, "cases": cases, "height_m": heights, "crown_radius_px": radii}


def main() -> None:
    import argparse

    from solweig_gpu import run_utci_tiles
    from solweig_sens.compact_masks import compact_masks
    from solweig_sens.fast import fast_solweig
    from solweig_sens.tight_bound import tight_bound

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--random", type=int, default=0, help="n random pairs instead of the sweep (training data)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overlap", action="store_true", help="with --random: crowns always overlap")
    ap.add_argument("--work", type=Path, default=None)
    args = ap.parse_args()
    work = args.work or (Path.home() / ("treepairs_random" if args.random else "treepairs"))
    design = random_pairs(args.random, args.seed, args.overlap) if args.random else sweep()
    design |= {"met": str(MET), "date": DATE, "seed": args.seed, "overlap": args.overlap}
    sites, cases = design["sites"], design["cases"]
    work.mkdir(parents=True, exist_ok=True)
    (work / "design.json").write_text(json.dumps(design, indent=1))
    for name, idx in cases.items():
        out_dir = work / name / "out"
        if is_complete(out_dir):
            continue
        trees = np.zeros((N, N), np.float32)
        for i in idx:  # overlapping crowns keep the taller canopy
            crown = disk(*sites[i], design["crown_radius_px"][i])
            trees[crown] = np.maximum(trees[crown], design["height_m"][i])
        setup = Setup(raster_dir=work / name / "rasters", met_file=MET, runs_dir=work / name, date=DATE, tile_size=N)
        write_rasters(setup.raster_dir, np.zeros((N, N), np.float32), trees)
        t0 = time.perf_counter()
        with compact_masks("bits", svf=True), tight_bound():
            prepare(setup)
        out_dir.mkdir(parents=True, exist_ok=True)
        with fast_solweig(), compact_masks("bits", svf=False), tight_bound():
            run_utci_tiles(base_path=str(out_dir), preprocess_dir=str(setup.preprocess_dir), selected_date_str=DATE,
                           save_tmrt=True, save_shadow=True)
        print(f"[pairs] {name} in {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
