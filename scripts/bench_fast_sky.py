"""Time and verify two fixes on one tile-day: outputs on a Linux file system, and the vectorised sky.

Variants, run one after another in one process (GPU to itself):
    A  original code, inputs and outputs on the Windows drive (/mnt/c), as in all runs so far
    B  original code, inputs and outputs on the WSL Linux file system
    C  vectorised sky (src/solweig_sens/fast_sky.py), Linux file system
    D  C plus ground view factors without per-step syncs (src/solweig_sens/fast_gvf.py)
Then compares outputs pixel by pixel and hour by hour: B against A (determinism, file location
should not matter), C against B (the sky rewrite) and D against C (should be bit-identical).

Usage:
    python scripts/bench_fast_sky.py --runs runs_la --date 2019-07-30 [--variants A B C D]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from solweig_sens.analysis import read_stack  # noqa: E402
from solweig_sens.fast import fast_solweig  # noqa: E402
from solweig_sens.fast_gvf import fast_gvf  # noqa: E402
from solweig_sens.fast_sky import fast_sky  # noqa: E402


@contextlib.contextmanager
def fast_sky_and_gvf():
    with fast_sky(), fast_gvf():
        yield


@contextlib.contextmanager
def fast_and_compact():
    from solweig_sens.compact_masks import compact_masks

    with fast_solweig(), compact_masks("bits"):
        yield


def compare(a: Path, b: Path) -> dict:
    out = {}
    for var in ("UTCI", "TMRT"):
        x, y = read_stack(a, var), read_stack(b, var)
        d = np.abs(x - y)
        d = d[np.isfinite(d)]
        out[var] = {"identical": bool(np.array_equal(x, y, equal_nan=True)), "max_abs": float(d.max()),
                    "p99_99": float(np.percentile(d, 99.99)), "n_above_0.01": int((d > 0.01).sum())}
    return out


def main() -> None:
    import torch
    from solweig_gpu import run_utci_tiles

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path("runs_la"))
    ap.add_argument("--date", default="2019-07-30")
    ap.add_argument("--linux-dir", type=Path, default=Path.home() / "solweig_bench")
    ap.add_argument("--variants", nargs="*", default=["A", "B", "C", "D"])
    args = ap.parse_args()
    win_pre = (args.runs / "shared" / "processed_inputs").resolve()
    lin = args.linux_dir
    lin_pre = lin / "processed_inputs"
    if not lin_pre.exists():
        shutil.copytree(win_pre, lin_pre)

    torch.zeros(1, device="cuda")  # create the CUDA context before timing
    variants = {
        "A": (args.runs.resolve() / "bench_A", win_pre, contextlib.nullcontext),
        "B": (lin / "runs" / "B", lin_pre, contextlib.nullcontext),
        "C": (lin / "runs" / "C", lin_pre, fast_sky),
        "D": (lin / "runs" / "D", lin_pre, fast_sky_and_gvf),
        "E": (lin / "runs" / "E", lin_pre, fast_solweig),  # all three, fingerprint-checked
        "F": (lin / "runs" / "F", lin_pre, fast_and_compact),  # + bit-packed sky-patch masks
    }
    times = {}
    for name, (out, pre, ctx) in variants.items():
        if name not in args.variants:
            continue
        shutil.rmtree(out, ignore_errors=True)
        t0 = time.perf_counter()
        with ctx():
            run_utci_tiles(base_path=str(out), preprocess_dir=str(pre), selected_date_str=args.date, save_tmrt=True)
        torch.cuda.synchronize()
        times[name] = time.perf_counter() - t0
        print(f"[bench] {name}: {times[name]:.1f} s", flush=True)

    result = {"seconds": times}
    for a, b in [("A", "B"), ("B", "C"), ("C", "D"), ("B", "D"), ("D", "E"), ("B", "E"), ("E", "F"), ("B", "F")]:
        if (variants[a][0] / "output_folder").exists() and (variants[b][0] / "output_folder").exists():
            result[f"{b}_vs_{a}"] = compare(variants[a][0], variants[b][0])
    Path("figures/profile").mkdir(parents=True, exist_ok=True)
    Path("figures/profile/bench_fast_sky.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
