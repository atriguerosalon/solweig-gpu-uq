"""Verify and measure compact sky-patch masks (src/solweig_sens/compact_masks.py) on real tiles.

For each configuration, runs a full day with float32 masks and with compact masks, both with the
original SOLWEIG functions and with the fast replacements, and compares every saved output (UTCI,
Tmrt, shadow, Kup, Kdown, Lup, Ldown). Compact masks should be bit-identical to float32 masks under
the same functions. Also records peak allocated and reserved GPU memory and wall time per run.

Writes figures/profile/verify_compact.json. GPU to itself recommended.

Usage:
    python scripts/verify_compact.py [--configs pacoima_summer austin tiles600]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from solweig_sens.compact_masks import compact_masks  # noqa: E402
from solweig_sens.fast import fast_solweig  # noqa: E402
from verify_fast import LIN, SAVE, compare_runs, prepare_configs  # noqa: E402

GIB = 2**30


@contextlib.contextmanager
def setup(fast: bool, masks: str):
    with contextlib.ExitStack() as st:
        if fast:
            st.enter_context(fast_solweig())
        if masks != "float32":
            st.enter_context(compact_masks(masks))
        yield


def run(pre: Path, date: str, out: Path, fast: bool, masks: str) -> dict:
    import torch
    from solweig_gpu import run_utci_tiles

    shutil.rmtree(out, ignore_errors=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    t0 = time.perf_counter()
    with setup(fast, masks):
        run_utci_tiles(base_path=str(out), preprocess_dir=str(pre), selected_date_str=date, **SAVE)
    torch.cuda.synchronize()
    return {"seconds": time.perf_counter() - t0,
            "peak_allocated_gib": (torch.cuda.max_memory_allocated() - base) / GIB,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / GIB}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="*", default=["pacoima_summer", "austin", "tiles600"])
    args = ap.parse_args()

    import torch

    torch.zeros(1, device="cuda")
    variants = [("original", False, "float32"), ("original + uint8 masks", False, "uint8"),
                ("original + bit masks", False, "bits"), ("fast", True, "float32"),
                ("fast + uint8 masks", True, "uint8"), ("fast + bit masks", True, "bits")]
    pairs = [("original", "original + uint8 masks"), ("original", "original + bit masks"),
             ("fast", "fast + uint8 masks"), ("fast", "fast + bit masks")]
    report = {}
    for name, (pre, date) in prepare_configs(args.configs).items():
        res = {"runs": {}, "comparisons": {}}
        outs = {}
        for label, fast, masks in variants:
            print(f"[compact] {name}: {label}", flush=True)
            outs[label] = LIN / "compact_out" / name / label.replace(" ", "_").replace("+", "")
            res["runs"][label] = run(pre, date, outs[label], fast, masks)
            print(f"[compact]   {res['runs'][label]}", flush=True)
        for a, b in pairs:
            res["comparisons"][f"{b} vs {a}"] = compare_runs(outs[a], outs[b])
            same = all(v["identical"] for v in res["comparisons"][f"{b} vs {a}"].values())
            print(f"[compact] {name}: {b} vs {a}: all outputs bit-identical = {same}", flush=True)
        report[name] = res
        shutil.rmtree(LIN / "compact_out" / name, ignore_errors=True)

    out = ROOT / "figures" / "profile" / "verify_compact.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"[compact] report: {out}")


if __name__ == "__main__":
    main()
