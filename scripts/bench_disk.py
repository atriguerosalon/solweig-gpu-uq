"""How much does writing outputs cost? One tile-day, outputs on the Windows drive or the Linux disk.

WSL reaches the Windows drive (/mnt/c) through the 9P file protocol, one round trip to a
Windows-side server per file operation; its own disk (~/) is a native Linux file system. Same
run (ours: fast functions, compact masks, tight bound; cached sky view factor of Pacoima
summer), three output setups, repeats interleaved:

    mntc_all    UTCI, Tmrt and Shadow written under the repo on /mnt/c
    linux_all   the same, under ~/
    linux_utci  only UTCI (SOLWEIG-GPU always writes it), under ~/

Usage:
    python scripts/bench_disk.py --src ~/solweig_verify/pacoima_summer --repeats 3
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CONFIGS = {"mntc_all": (ROOT / "tmp_bench_disk", True), "linux_all": (Path.home() / "bench_disk", True),
           "linux_utci": (Path.home() / "bench_disk", False)}


def main() -> None:
    import torch
    from solweig_gpu import run_utci_tiles
    from solweig_sens.compact_masks import compact_masks
    from solweig_sens.fast import fast_solweig
    from solweig_sens.tight_bound import tight_bound

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=Path.home() / "solweig_verify" / "pacoima_summer")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--date", default="2019-07-30")
    args = ap.parse_args()
    pre = Path.home() / "bench_disk_inputs"
    shutil.rmtree(pre, ignore_errors=True)
    shutil.copytree(args.src, pre)  # inputs on the Linux disk for every configuration
    times = {k: [] for k in CONFIGS}
    for rep in range(args.repeats):
        for name, (base, all_outputs) in CONFIGS.items():
            out = base / name
            shutil.rmtree(out, ignore_errors=True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with fast_solweig(), compact_masks("bits", svf=False), tight_bound():
                run_utci_tiles(base_path=str(out), preprocess_dir=str(pre), selected_date_str=args.date,
                               save_tmrt=all_outputs, save_shadow=all_outputs)
            torch.cuda.synchronize()
            times[name].append(time.perf_counter() - t0)
            print(f"[disk] {name:10s} rep {rep}: {times[name][-1]:.1f} s", flush=True)
            shutil.rmtree(out, ignore_errors=True)
    shutil.rmtree(CONFIGS["mntc_all"][0], ignore_errors=True)
    res = {k: {"median_s": statistics.median(v), "runs_s": v} for k, v in times.items()}
    (ROOT / "figures" / "bench_disk.json").write_text(json.dumps(res, indent=2))
    print("[disk] " + ", ".join(f"{k} {v['median_s']:.1f} s" for k, v in res.items()))


if __name__ == "__main__":
    main()
