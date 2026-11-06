"""Does the heat-storage patch (B1) give the same results with all speed-ups (C1-C9) as with C1-C3 only?

Every speed-up was checked against the original with SOLWEIG's own physics, and a unit test checks that
B1 calls the fast functions of C1-C3. The fitted model of the Sobol, forecast and shortcut runs uses B1
with C1-C9, which no run had compared. This runs the same day twice with the fitted model (ground
heating fitted to the Pacoima measurements, B1 with tau 4 h) and one Sobol sample of the inputs, on the
centred 500 x 500 Pacoima crop of run_la_sobol.py:

    reference   C1-C3 (fast_solweig), float32 masks as SOLWEIG loads them
    combined    C1-C3 + C4/C9 (bit-packed masks, parallel loading, packed cache copy) + C8 + C6, entered in
                the order run_la_sobol.py uses; run twice, first writing the packed cache copy, then reading it

C5 and C7 only act in the sky view step, which is cached here; their cache files were checked byte-identical
before. UTCI, Tmrt and Shadow are compared on every pixel and hour; they should be bit-identical. A fresh
copy of the preprocessed crop is used, so the Sobol inputs are not touched. Stops before a run if the GPU
has less than --min-free-mib free (the GPU may be shared).

Writes figures/profile/verify_b1_combined.json.

Usage (after run_la_sobol.py prepare):
    python scripts/verify_b1_combined.py
"""

from __future__ import annotations

import sys as _sys

if __name__ == "__main__" and any(a in ("-h", "--help") for a in _sys.argv[1:]):
    print(__doc__)
    raise SystemExit(0)

import argparse
import contextlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

SOBOL = Path.home() / "sobol_la"
DATE = "2019-07-30"


def free_mib() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout.split(",")
    return int(out[1]) - int(out[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=Path.home() / "b1check")
    ap.add_argument("--sample", type=int, default=0, help="row of the Sobol design used for the inputs")
    ap.add_argument("--min-free-mib", type=int, default=2000)
    args = ap.parse_args()

    import torch
    from run_la_pavement import TS_FIT
    from solweig_gpu import run_utci_tiles
    from solweig_sens.analysis import read_stack
    from solweig_sens.compact_masks import compact_masks
    from solweig_sens.fast import fast_solweig
    from solweig_sens.heat_storage import ground_heat_storage
    from solweig_sens.hoist_geometry import hoist_geometry
    from solweig_sens.perturb import PAVED_CLASSES, applied_many, patched_landcover
    from solweig_sens.tight_bound import tight_bound

    p = json.loads((SOBOL / "problem.json").read_text())
    v = dict(zip(p["names"], map(float, np.load(SOBOL / "X.npy")[args.sample])))
    sample = {k: v[k] for k in ("ta_add", "rh_add", "wind_scale", "kdown_scale", "albedo_b", "ewall")}
    paved = {"albedo": v["regular_albedo"], "emissivity": v["paved_emissivity"], "ts_deg": TS_FIT}
    pre = args.work / "processed_inputs"
    shutil.rmtree(args.work, ignore_errors=True)
    shutil.copytree(Path((SOBOL / "preprocess_dir.txt").read_text()), pre)

    def run(name: str, speedups: list) -> dict:
        free = free_mib()
        if free < args.min_free_mib:
            raise SystemExit(f"[b1] stopped before {name}: {free} MiB free on the GPU, below {args.min_free_mib}")
        out = args.work / name
        tmp = args.work / f"tmp_{name}"
        out.mkdir()
        tmp.mkdir()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        with contextlib.ExitStack() as st:
            for c in speedups:
                st.enter_context(c)
            st.enter_context(applied_many(sample, tmp, pre / "metfiles"))
            st.enter_context(patched_landcover(tmp, PAVED_CLASSES, paved))
            st.enter_context(ground_heat_storage(4.0))
            run_utci_tiles(base_path=str(out), preprocess_dir=str(pre), selected_date_str=DATE, save_tmrt=True,
                           save_shadow=True)
        rec = {"seconds": round(time.perf_counter() - t0, 1), "gpu_free_mib_before": free,
               "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2 ** 30, 3)}
        print(f"[b1] {name}: {rec}", flush=True)
        return rec

    report = {"date": DATE, "sample": args.sample, "inputs": v, "tile": "Pacoima, centred 500 x 500 at 2 m",
              "model": "fitted ground heating (TS_FIT) + ground_heat_storage(4.0)", "runs": {}, "comparisons": {}}
    report["runs"]["reference"] = run("reference", [fast_solweig()])
    combined = lambda: [fast_solweig(), compact_masks("bits", svf=False, packed_cache=True), hoist_geometry(), tight_bound()]
    report["runs"]["combined_cold"] = run("combined_cold", combined())  # writes the packed cache copy
    report["runs"]["combined_warm"] = run("combined_warm", combined())  # reads it

    ok = True
    for name in ("combined_cold", "combined_warm"):
        for var in ("UTCI", "TMRT", "Shadow"):
            a, b = read_stack(args.work / "reference", var), read_stack(args.work / name, var)
            same_nan = bool(np.array_equal(np.isnan(a), np.isnan(b)))
            d = np.abs(a.astype(np.float64) - b.astype(np.float64))
            rec = {"bit_identical": bool(np.array_equal(a, b, equal_nan=True)), "same_missing": same_nan,
                   "max_abs": float(np.nanmax(d)) if np.isfinite(d).any() else 0.0,
                   "pixels_hours": int(a.size), "night_max_abs": float(np.nanmax(d[list(range(0, 6)) + list(range(21, d.shape[0]))]))}
            report["comparisons"][f"{name}|{var}"] = rec
            ok &= rec["bit_identical"]
            print(f"[b1] {name} vs reference, {var}: {rec}", flush=True)
    report["all_bit_identical"] = ok
    dst = ROOT / "figures" / "profile" / "verify_b1_combined.json"
    dst.write_text(json.dumps(report, indent=2))
    print(f"[b1] all bit-identical: {ok}; wrote {dst}", flush=True)


if __name__ == "__main__":
    main()
