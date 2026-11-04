"""Where does SOLWEIG-GPU still make the CPU wait for the GPU? Count CPU-GPU syncs per code location.

Uses PyTorch's own detector, torch.cuda.set_sync_debug_mode("warn"), which warns at every
synchronising operation (reading a GPU value on the CPU: .item(), int(), bool(), boolean-mask
indexing, nonzero, ...). Each warning is attributed to the innermost frame in solweig_gpu or
solweig_sens (patched functions are compiled with SOLWEIG's file name, so their line numbers refer
to the patched source). No event log is kept, so memory stays flat, and it works under WSL where
the PyTorch profiler cannot see GPU kernels. Counting slows the run: timings here mean nothing.

Configurations: v200 (untouched), ours (C1-C6), ours_c7 (C1-C7). Sky view factor and day (to 14 h)
counted separately, on a centred crop of Pacoima.

Usage:
    python scripts/count_syncs.py --rasters data/la/pacoima --met data/la/met_pacoima_20190730.txt --size 250
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import json
import shutil
import sys
import traceback
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from bound_test import centred_crop  # noqa: E402
from run_morris import trim_met  # noqa: E402
from solweig_sens.run import Setup, prepare  # noqa: E402


@contextlib.contextmanager
def sync_counter(counts: collections.Counter):
    """Count every synchronising CUDA operation by the innermost SOLWEIG frame that triggered it."""
    import torch

    def hook(message, category, filename, lineno, file=None, line=None):
        if "synchronizing" not in str(message):
            return
        where = "outside SOLWEIG"
        for fr in reversed(traceback.extract_stack()[:-1]):
            if "solweig_gpu" in fr.filename or "solweig_sens" in fr.filename:
                where = f"{Path(fr.filename).name}:{fr.name}:{fr.lineno}  {(fr.line or '').strip()[:70]}"
                break
        counts[where] += 1

    old_show, old_mode = warnings.showwarning, torch.cuda.get_sync_debug_mode()
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.showwarning = hook
        torch.cuda.set_sync_debug_mode("warn")
        try:
            yield
        finally:
            torch.cuda.set_sync_debug_mode(old_mode)
            warnings.showwarning = old_show


def contexts(cfg: str, stage: str) -> list:
    if cfg == "v200":
        return []
    from solweig_sens.compact_masks import compact_masks
    from solweig_sens.fast import fast_solweig
    from solweig_sens.fast_sky_shadow import fast_sky_shadow
    from solweig_sens.tight_bound import tight_bound

    if stage == "svf":
        return [compact_masks("bits", svf=True)] + ([fast_sky_shadow()] if cfg == "ours_c7" else []) + [tight_bound()]
    return [fast_solweig(), compact_masks("bits", svf=False), tight_bound()]


def main() -> None:
    from solweig_gpu import run_utci_tiles

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rasters", type=Path, required=True)
    ap.add_argument("--met", type=Path, required=True)
    ap.add_argument("--size", type=int, default=250)
    ap.add_argument("--work", type=Path, default=Path.home() / "count_syncs")
    ap.add_argument("--date", default="2019-07-30")
    ap.add_argument("--out", type=Path, default=ROOT / "figures")
    args = ap.parse_args()
    work = args.work
    work.mkdir(parents=True, exist_ok=True)
    met = work / "met_0to14.txt"
    trim_met(args.met.resolve(), met, 14)
    raster = work / f"rasters_{args.size}"
    centred_crop(args.rasters.resolve(), raster, args.size)
    res = {"size_px": args.size, "configs": {}}
    for cfg in ("v200", "ours", "ours_c7"):
        d = work / cfg
        shutil.rmtree(d, ignore_errors=True)
        setup = Setup(raster_dir=raster, met_file=met, runs_dir=d, date=args.date, tile_size=args.size)
        rec = {}
        for stage in ("svf", "day"):
            counts = collections.Counter()
            with contextlib.ExitStack() as st:
                for c in contexts(cfg, stage):
                    st.enter_context(c)
                with sync_counter(counts):
                    if stage == "svf":
                        prepare(setup)
                    else:
                        run_utci_tiles(base_path=str(d / "out"), preprocess_dir=str(setup.preprocess_dir),
                                       selected_date_str=args.date, save_tmrt=False)
            rec[stage] = {"total": sum(counts.values()), "top": counts.most_common(25)}
            print(f"[sync] {cfg:8s} {stage:3s}: {sum(counts.values()):8d} syncs", flush=True)
            for where, n in counts.most_common(8):
                print(f"           {n:8d}  {where}", flush=True)
        res["configs"][cfg] = rec
        (args.out / "sync_counts.json").write_text(json.dumps(res, indent=2))
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    main()
